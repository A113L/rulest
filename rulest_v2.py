#!/usr/bin/env python3
"""
rulest — GPU-Compatible Hashcat Rules Engine
====================================================================
Stripped version: only 5 basic token‑strip modes + insert mode.
All extra modes (swap, range, char‑duplicate, ascii‑transform, truncate,
purge, repeat, separator‑title, overwrite) have been removed.

# Main changes:
- No --token-strip-no-new-modes flag.
- No extra indexes (purge, substr, omit, prefix, ascii, overwrite) are built.
- Worker processes only call letter, digit, reverse, delete‑edge, duplicate,
  and insert extraction functions.
- Insert mode is kept (--max-depth >= 2).

# Bloom Filter — GPU Build (build_bloom_filter_gpu)

Previously the Bloom filter was always built on the CPU: every target word was
hashed twice in Python and the corresponding bits were set one at a time in a
NumPy byte array.  For large target wordlists (hundreds of thousands of words)
this was the single slowest part of startup.
The filter is now built entirely on the GPU via a dedicated OpenCL kernel
(build_bloom_filter_gpu).  Each work-item handles one word — computing the
same pair of FNV-1a hashes used by the membership-check kernel — and atomically
ORs the resulting bits into a flat int32 buffer using atomic_or.  The
uint8 view read back from the device is byte-for-byte identical to the CPU
output, so the check kernel required no changes.
A plain zeroed int32 array is used as the backing buffer (rather than uchar)
because OpenCL 1.x atomic_or requires a 32-bit pointer; the bit-index
arithmetic is adjusted accordingly, with an explicit 64-bit modulo guard for
512 MB filters where BLOOM_FILTER_SIZE == 2^32.
Falls back transparently to the CPU path on any driver or context error.

# Speed indicators in progress bars – All major stages now show average throughput:

- STAGE 0 → words/sec (CPU workers)
- STAGE 1 → combos/sec = words_processed × total_rules / elapsed
-  / STAGE 2 → combos/sec = accumulated (words × chains) / elapsed
- STAGE 3 GA → combos/sec = (pop_size × base_words × generations) / elapsed
- Minimization (both in‑memory and disk‑backed) → rules/sec
- Human‑readable speed formatting – Added _fmt_speed() helper that auto‑scales to K/M/G (e.g., 12.3M combos/s).

# Improved startup hint – The pause (p), resume (r), and early‑quit (q) key instructions are now displayed more prominently. display it well formatted in codeblock
"""

import os
import sys
import sqlite3
import hashlib
import tempfile
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
from collections import defaultdict, Counter
from tqdm import tqdm
import time
import math
import random
from typing import Dict, Set, Tuple, Optional, List, Iterator
import gc
import datetime
import multiprocessing as mp
import threading
import shutil

# Platform keyboard helpers (best-effort; silently disabled if unavailable)
try:
    import tty as _tty, termios as _termios, select as _select
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False
try:
    import msvcrt as _msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

# ================== GLOBAL FLAGS ===================
VERBOSE            = False   # set by --debug
ALLOW_REJECT_RULES = False   # set by --allow-reject-rules
# ==================================================

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

# ====================================================================
# --- KEYBOARD CONTROLLER (p=pause  r=resume  q=early-quit) ---
# ====================================================================
class KeyboardController:
    """Non-blocking keyboard listener for pause / resume / early-quit.

    Runs in a daemon thread.  Silently disabled when stdin is not a tty
    or when neither termios (Unix) nor msvcrt (Windows) is available.

    Controls
    --------
    p  — pause between batches (prints status, blocks until r or q)
    r  — resume after pause
    q  — request early exit; current results are saved & minimized
    """

    def __init__(self) -> None:
        self._paused  = False
        self._quit    = False
        self._lock    = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active  = False

    # ── public API ────────────────────────────────────────────────────

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        if not (_HAS_TERMIOS or _HAS_MSVCRT):
            return
        self._active = True
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name='kb-listener')
        self._thread.start()

    def stop(self) -> None:
        self._active = False

    def check_pause(self) -> None:
        """Block the calling thread while paused.  Returns immediately otherwise."""
        while True:
            with self._lock:
                if not self._paused or self._quit:
                    return
            time.sleep(0.15)

    @property
    def quit_requested(self) -> bool:
        with self._lock:
            return self._quit

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    # ── internals ─────────────────────────────────────────────────────

    def _handle(self, ch: str) -> None:
        ch = ch.lower()
        if ch == 'p':
            with self._lock:
                if self._quit:
                    return
                self._paused = True
            w = min(max(shutil.get_terminal_size((80, 24)).columns - 2, 44), 92)
            print(f"\n{yellow('─' * w)}")
            print(f"{yellow('│')} {bold('PAUSED')}  —  press {bold(green('r'))} to resume  |  {bold(yellow('q'))} to save current results & quit")
            print(f"{yellow('─' * w)}")
        elif ch == 'r':
            with self._lock:
                if not self._paused:
                    return
                self._paused = False
            w = min(max(shutil.get_terminal_size((80, 24)).columns - 2, 44), 92)
            print(f"\n{green('─' * w)}")
            print(f"{green('│')} {bold('RESUMED')}")
            print(f"{green('─' * w)}\n")
        elif ch == 'q':
            with self._lock:
                already = self._quit
                self._quit   = True
                self._paused = False   # unblock if currently paused
            if not already:
                w = min(max(shutil.get_terminal_size((80, 24)).columns - 2, 44), 92)
                print(f"\n{yellow('─' * w)}")
                print(f"{yellow('│')} {bold(yellow('EARLY EXIT REQUESTED'))}  —  finishing current batch then saving results …")
                print(f"{yellow('─' * w)}\n")

    def _reader_termios(self) -> None:
        fd  = sys.stdin.fileno()
        old = _termios.tcgetattr(fd)
        try:
            _tty.setcbreak(fd)          # keeps OPOST → \n still becomes \r\n
            while self._active:
                if _select.select([sys.stdin], [], [], 0.15)[0]:
                    ch = sys.stdin.read(1)
                    self._handle(ch)
        except Exception:
            pass
        finally:
            try:
                _termios.tcsetattr(fd, _termios.TCSADRAIN, old)
            except Exception:
                pass

    def _reader_msvcrt(self) -> None:
        while self._active:
            if _msvcrt.kbhit():
                ch = _msvcrt.getwch()
                self._handle(ch)
            else:
                time.sleep(0.05)

    def _reader(self) -> None:
        if _HAS_TERMIOS:
            self._reader_termios()
        elif _HAS_MSVCRT:
            self._reader_msvcrt()


_kb = KeyboardController()


# ====================================================================
# --- COLORS ---
# ====================================================================
class C:
    RED    = '\033[91m'; GREEN  = '\033[92m'; YELLOW = '\033[93m'
    BLUE   = '\033[94m'; CYAN   = '\033[96m'; MAGENTA= '\033[95m'
    BOLD   = '\033[1m';  DIM    = '\033[2m';  END    = '\033[0m'

def red(t):     return f"{C.RED}{t}{C.END}"
def green(t):   return f"{C.GREEN}{t}{C.END}"
def yellow(t):  return f"{C.YELLOW}{t}{C.END}"
def blue(t):    return f"{C.BLUE}{t}{C.END}"
def cyan(t):    return f"{C.CYAN}{t}{C.END}"
def magenta(t): return f"{C.MAGENTA}{t}{C.END}"
def bold(t):    return f"{C.BOLD}{t}{C.END}"
def dim(t):     return f"{C.DIM}{t}{C.END}"

# ====================================================================
# --- LOGGING SYSTEM ---
# ====================================================================
def log_info(msg: str) -> None:
    print(msg)

def log_debug(msg: str) -> None:
    if VERBOSE:
        print(f"{dim('[dbg]')} {msg}")

def log_warn(msg: str) -> None:
    print(yellow(f"[WARN] {msg}"))

def log_error(msg: str) -> None:
    print(red(f"[ERROR] {msg}"))

def _fmt_speed(n: float, unit: str = "combos") -> str:
    """Format a throughput number as e.g. '12.3M combos/s' or '4.5K rules/s'."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}G {unit}/s"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M {unit}/s"
    if n >= 1_000:
        return f"{n/1_000:.1f}K {unit}/s"
    return f"{n:.0f} {unit}/s"

def log_section(title: str) -> None:
    w   = min(max(shutil.get_terminal_size((80, 24)).columns - 2, 44), 92)
    bar = '─' * w
    print(f"\n{cyan(bar)}")
    print(f"{cyan('│')} {bold(title.upper())}")
    print(f"{cyan(bar)}")

# ====================================================================
# --- BANNER ---
# ====================================================================
BANNER = f"""{green(bold('''
 ██████╗ ██╗   ██╗██╗     ███████╗███████╗████████╗
 ██╔══██╗██║   ██║██║     ██╔════╝██╔════╝╚══██╔══╝
 ██████╔╝██║   ██║██║     █████╗  ███████╗   ██║
 ██╔══██╗██║   ██║██║     ██═══╝  ╚════██║   ██║
 ██║  ██║╚██████╔╝███████╗███████╗███████║   ██║
 ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝   ╚═╝'''))}{C.END}
  {dim('GPU-Compatible Hashcat Rules Engine')}
  {dim('github.com/A113L/rulest')}
"""

def print_banner() -> None:
    print(BANNER)

def _print_controls() -> None:
    """Print keyboard-control hint block — only when running in an interactive TTY."""
    if not (sys.stdin.isatty() and (_HAS_TERMIOS or _HAS_MSVCRT)):
        return
    w   = min(max(shutil.get_terminal_size((80, 24)).columns - 2, 44), 92)
    sep = dim('─' * (w - 2))
    print(f"  {sep}")
    print(f"  {bold('Controls')}  "
          f"{cyan(bold('[p]'))} pause processing   "
          f"{green(bold('[r]'))} resume after pause   "
          f"{yellow(bold('[q]'))} save partial results & quit")
    print(f"  {sep}")
    print()

# ====================================================================
# --- CONSTANTS ---
# ====================================================================
MAX_WORD_LEN         = 256
MAX_RULE_LEN         = 16
MAX_OUTPUT_LEN       = 512
MAX_CHAIN_STRING_LEN = 128
MAX_HASHCAT_CHAIN    = 31

VRAM_USAGE_FACTOR    = 0.55
BLOOM_HASH_FUNCTIONS = 4
# Default fallback if no device info available
DEFAULT_BLOOM_MB = 128

BASELINE_COMBOS_PER_SEC         = 120_000_000
LOW_END_COMPUTE_UNITS_THRESHOLD = 20
LOW_END_COMBOS_PER_SEC          = 40_000_000

POSSIBLE_WORK_GROUP_SIZES = [32, 64, 128, 256, 512, 1024]

BASE_WORDS_PER_BATCH         = 5000
BASE_CHAINS_PER_BATCH        = 2000
BASE_WORD_SUB_BATCH          = 20000
BASE_MAX_SAFE_RESULTS        = 25000

HOT_RULE_RATIO               = 0.6
EXTENSION_RATIO              = 0.3
MAX_ATTEMPTS_MULTIPLIER      = 5
TIME_SAFETY_FACTOR           = 0.9
OPTIMAL_GLOBAL_MULTIPLIER_BASE = 16

FNV1A_PRIME        = 16777619
FNV1A_OFFSET_BASIS = 2166136261
FNV1A_SEED1        = 0xDEADBEEF
FNV1A_SEED2        = 0xCAFEBABE

MAX_GPU_RULES = 255

_UNSUPPORTED_SENTINEL = object()

MINIMIZE_DISK_THRESHOLD  = 500_000
MINIMIZE_DISK_BATCH_SIZE =  10_000

SPECIAL_CHARS_TOP  = ['!', '@', '#', '$', '%', '^', '&', '*',
                      '?', '.', '-', '_', '+', '(', ')']
SPECIAL_CHARS_CORE = ['!', '@', '#', '$', '%', '*', '?']

LEET_SUBS: List[Tuple[str, str]] = [
    ('a', '@'), ('e', '3'), ('o', '0'), ('i', '1'), ('l', '1'),
    ('s', '5'), ('s', '$'), ('t', '7'), ('a', '4'), ('i', '!'),
]
LEET_OPS: List[str] = [f's{orig}{rep}' for orig, rep in LEET_SUBS]

# ── Token-Strip constants ──────────────────────────────────────────
TOKEN_STRIP_LEET_TABLE: List[Tuple[str, str, str]] = [
    ('@', 'a', 'sa@'), ('3', 'e', 'se3'), ('0', 'o', 'so0'),
    ('1', 'i', 'si1'), ('1', 'l', 'sl1'), ('5', 's', 'ss5'),
    ('$', 's', 'ss$'), ('7', 't', 'st7'), ('4', 'a', 'sa4'), ('!', 'i', 'si!'),
]
_TOKEN_STRIP_LEET_BY_CHAR: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
for _ts_enc, _ts_base, _ts_rule in TOKEN_STRIP_LEET_TABLE:
    _TOKEN_STRIP_LEET_BY_CHAR[_ts_enc].append((_ts_base, _ts_rule))

TOKEN_STRIP_LEET_CHARS: Set[str] = set(_TOKEN_STRIP_LEET_BY_CHAR.keys())
TOKEN_STRIP_BOUNDARY: Set[str] = set('0123456789!@#$%^&*?.-_+()')
TOKEN_STRIP_ALPHA_BOUNDARY: Set[str] = (
    set(string.ascii_letters) | set('!@#$%^&*?.-_+()')
)

# ── STAGE 0 multiprocessing worker globals ─────────────────────────
_p0_worker_base_set:       Set[str]                       = set()
_p0_worker_base_by_len:    Dict[int, Set[str]]            = {}

# ====================================================================
# --- BUILT-IN PROBE SET ---
# ====================================================================
BUILTIN_PROBES: List[str] = [
    "ab", "abc", "abcd", "pass", "root", "test", "admin", "login",
    "letmein", "welcome", "password", "sunshine", "football",
    "baseball", "princess", "dragon12", "qwertyuiop", "iloveyou12",
    "monkey12345", "superman123", "mustang2024",
    "Password", "AdminUser", "MySecret", "HelloWorld",
    "pass123", "admin2024", "test1234", "user9999",
    "p@ssw0rd", "s3cur1ty", "master", "leet", "elite", "access",
    "aaaa", "bbbb",
]
_seen = set(); _d = []
for w in BUILTIN_PROBES:
    if w not in _seen: _seen.add(w); _d.append(w)
BUILTIN_PROBES = _d; del _seen, _d, w

# ====================================================================
# --- RULE EXCLUSION FILTER ---
# ====================================================================
def should_exclude_rule(rule: str) -> bool:
    if ALLOW_REJECT_RULES: return False
    if not rule: return False
    if len(rule) == 1 and rule in ('_', 'M', '4', '6', 'Q'): return True
    if len(rule) == 2 and rule[0] in ('!', '/', '(', ')', '<', '>', '_'): return True
    if len(rule) == 3 and rule[0] in ('?', '=', 'v'): return True
    return False

# ====================================================================
# --- HASHCAT RULE VALIDATOR ---
# ====================================================================
class HashcatRuleValidator:
    MAX_GPU_RULES = MAX_GPU_RULES

    @staticmethod
    def is_digit(c: str) -> bool:
        return '0' <= c <= '9'

    @staticmethod
    def validate_rule_for_gpu(rule_str: str) -> bool:
        if should_exclude_rule(rule_str): return False
        pos = 0; cnt = 0; n = len(rule_str)
        isd = HashcatRuleValidator.is_digit
        while pos < n:
            c = rule_str[pos]
            if c == ' ': pos += 1; continue
            if c in ('p', 'z', 'Z'):
                cnt += 1; pos += 1
                if pos < n and isd(rule_str[pos]): pos += 1
                continue
            if c in (':', 'l', 'u', 'c', 'C', 't', 'r', 'd', 'f',
                     'a', 'q', 'k', 'K', 'E', '{', '}', '[', ']'):
                pos += 1; cnt += 1; continue
            if c in ('T', 'D', 'L', 'R', '+', '-', '.', ',', "'", 'y', 'Y'):
                pos += 1
                if pos >= n or not isd(rule_str[pos]): return False
                pos += 1; cnt += 1; continue
            if c in ('i', 'o', '3'):
                pos += 1
                if pos >= n or not isd(rule_str[pos]): return False
                pos += 1
                if pos >= n: return False
                pos += 1; cnt += 1; continue
            if c in ('x', '*', 'O'):
                pos += 1
                if pos >= n or not isd(rule_str[pos]): return False
                pos += 1
                if pos >= n or not isd(rule_str[pos]): return False
                pos += 1; cnt += 1; continue
            if c == 's':
                pos += 1
                if pos + 1 >= n: return False
                pos += 2; cnt += 1; continue
            if c in ('@', 'e', '$', '^'):
                pos += 1
                if pos >= n: return False
                pos += 1; cnt += 1; continue
            return False
        return cnt <= HashcatRuleValidator.MAX_GPU_RULES

    @staticmethod
    def validate_rules_for_gpu(rules):
        valid = []
        for r in rules:
            r = r.strip('\n\r')
            if r and HashcatRuleValidator.validate_rule_for_gpu(r):
                valid.append(r)
        return valid

# ====================================================================
# --- FNV-1a ---
# ====================================================================
def fnv1a_32(data, seed=FNV1A_SEED1):
    h = seed ^ FNV1A_OFFSET_BASIS
    for b in data:
        h ^= b; h = (h * FNV1A_PRIME) & 0xFFFFFFFF
    return h

# ====================================================================
# --- PYTHON-SIDE RULE APPLICATOR ---
# ====================================================================
def _py_apply_single_rule(rule: str, word: str) -> Optional[str]:
    if not rule: return word
    w   = list(word.encode('latin-1'))
    cmd = rule[0]
    def dg(c): return ord(c) - 48 if '0' <= c <= '9' else -1
    try:
        if   cmd == ':': pass
        elif cmd == 'l': w = [c | 0x20 if 65<=c<=90 else c for c in w]
        elif cmd == 'u': w = [c & ~0x20 if 97<=c<=122 else c for c in w]
        elif cmd == 'c':
            if w:
                w[0] = w[0] & ~0x20 if 97<=w[0]<=122 else w[0]
                w[1:] = [c | 0x20 if 65<=c<=90 else c for c in w[1:]]
        elif cmd == 'C':
            if w:
                w[0] = w[0] | 0x20 if 65<=w[0]<=90 else w[0]
                w[1:] = [c & ~0x20 if 97<=c<=122 else c for c in w[1:]]
        elif cmd == 't':
            w = [c|0x20 if 65<=c<=90 else (c&~0x20 if 97<=c<=122 else c) for c in w]
        elif cmd == 'r': w = w[::-1]
        elif cmd == 'd': w = w + w
        elif cmd == 'f': w = w + w[::-1]
        elif cmd == '{':
            if len(w) > 1: w = w[1:] + [w[0]]
        elif cmd == '}':
            if len(w) > 1: w = [w[-1]] + w[:-1]
        elif cmd == '[':
            if w: w = w[1:]
        elif cmd == ']':
            if w: w = w[:-1]
        elif cmd == 'k':
            if len(w) >= 2: w[0], w[1] = w[1], w[0]
        elif cmd == 'K':
            if len(w) >= 2: w[-1], w[-2] = w[-2], w[-1]
        elif cmd == 'q':
            out = []
            for c in w: out += [c, c]
            w = out
        elif cmd == 'E':
            out = []; cap = True
            for c in w:
                out.append(c & ~0x20 if cap and 97<=c<=122 else c)
                cap = c in (32, 45, 95)
            w = out
        elif cmd == '^' and len(rule)==2: w = [ord(rule[1])] + w
        elif cmd == '$' and len(rule)==2: w = w + [ord(rule[1])]
        elif cmd == '@' and len(rule)==2:
            ch = ord(rule[1]); w = [c for c in w if c != ch]
        elif cmd == 'p' and len(rule)==2:
            n = dg(rule[1])
            if n > 0: orig = w[:]; [w.__iadd__(orig) for _ in range(n)]
        elif cmd == 'T' and len(rule)==2:
            p = dg(rule[1])
            if 0 <= p < len(w):
                c = w[p]; w[p] = c|0x20 if 65<=c<=90 else (c&~0x20 if 97<=c<=122 else c)
        elif cmd == 'D' and len(rule)==2:
            p = dg(rule[1])
            if 0 <= p < len(w): w.pop(p)
        elif cmd == 'L' and len(rule)==2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] << 1) & 0xFF
        elif cmd == 'R' and len(rule)==2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] >> 1) & 0xFF
        elif cmd == '+' and len(rule)==2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] + 1) & 0xFF
        elif cmd == '-' and len(rule)==2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] - 1) & 0xFF
        elif cmd in ('.', ',') and len(rule)==2:
            p = dg(rule[1]); delta = 1 if cmd == '.' else -1
            if 0 <= p < len(w): w[p] = (w[p] + delta) & 0xFF
        elif cmd == "'" and len(rule)==2:
            p = dg(rule[1])
            if 0 <= p < len(w): w = w[:p+1]
        elif cmd == 'z' and len(rule)==2:
            n = dg(rule[1])
            if n > 0 and w: w = [w[0]] * n + w
        elif cmd == 'Z' and len(rule)==2:
            n = dg(rule[1])
            if n > 0 and w: w = w + [w[-1]] * n
        elif cmd == 'y' and len(rule)==2:
            n = dg(rule[1])
            if n > 0: w = w[:n] + w
        elif cmd == 'Y' and len(rule)==2:
            n = dg(rule[1])
            if n > 0 and len(w) >= n: w = w + w[-n:]
        elif cmd == 's' and len(rule)==3:
            a, b = ord(rule[1]), ord(rule[2])
            w = [b if c==a else c for c in w]
        elif cmd == 'i' and len(rule)==3:
            p, ch = dg(rule[1]), ord(rule[2])
            if 0 <= p <= len(w): w.insert(p, ch)
        elif cmd == 'o' and len(rule)==3:
            p, ch = dg(rule[1]), ord(rule[2])
            if 0 <= p < len(w): w[p] = ch
        elif cmd == 'e' and len(rule)>=2:
            sep = ord(rule[1]); out = []; cap = True
            for c in w:
                out.append(c & ~0x20 if cap and 97<=c<=122 else c)
                cap = (c == sep)
            w = out
        elif cmd == 'x' and len(rule)==3:
            a, b = dg(rule[1]), dg(rule[2])
            if a > b: a, b = b, a
            w = w[a:b+1]
        elif cmd == 'O' and len(rule)==3:
            p, m = dg(rule[1]), dg(rule[2])
            if 0 <= p < len(w) and m > 0: w = w[:p] + w[p+m:]
        elif cmd == '*' and len(rule)==3:
            a, b = dg(rule[1]), dg(rule[2])
            if 0<=a<len(w) and 0<=b<len(w) and a!=b: w[a], w[b] = w[b], w[a]
        elif cmd == '3' and len(rule)==3:
            n, sep = dg(rule[1]), ord(rule[2]); cnt = 0
            for i, c in enumerate(w):
                if c == sep:
                    cnt += 1
                    if cnt == n and i+1 < len(w):
                        ci = w[i+1]
                        w[i+1] = ci|0x20 if 65<=ci<=90 else (ci&~0x20 if 97<=ci<=122 else ci)
                        break
        else:
            return None
    except Exception:
        return None
    try:
        return bytes(w).decode('latin-1')
    except Exception:
        return None

def py_apply_chain(chain: str, word: str) -> Optional[str]:
    cur = word
    for r in chain.split():
        cur = _py_apply_single_rule(r, cur)
        if cur is None:
            return None
    return cur

# ====================================================================
# --- SIGNATURE-BASED FUNCTIONAL MINIMIZATION ---
# ====================================================================
def compute_rule_signature(rule: str, probe_words: List[str]) -> tuple:
    outputs = []
    for word in probe_words:
        out = py_apply_chain(rule, word)
        if out is None:
            return ('__UNSUPPORTED__',)
        outputs.append(out)
    return tuple(outputs)

def minimize_by_signature(rule_counter: Counter, probe_words: List[str]) -> Counter:
    if not rule_counter:
        return Counter()
    log_section("POST-PROCESSING — Signature-Based Functional Minimization")
    n = len(rule_counter)
    log_info(f"[MINIMIZE] Candidates  : {bold(str(n))}")
    log_info(f"[MINIMIZE] Probe words : {bold(str(len(probe_words)))}")
    if n > MINIMIZE_DISK_THRESHOLD:
        log_info(f"[MINIMIZE] {cyan(f'{n:,} rules exceeds threshold {MINIMIZE_DISK_THRESHOLD:,}')} — using {bold('disk-backed')} SQLite path")
        log_debug(f"[MINIMIZE] Disk path chosen: {n:,} rules > threshold {MINIMIZE_DISK_THRESHOLD:,}")
        return _minimize_disk(rule_counter, probe_words)
    else:
        log_debug(f"[MINIMIZE] In-memory path chosen: {n:,} rules ≤ threshold {MINIMIZE_DISK_THRESHOLD:,}")
        return _minimize_mem(rule_counter, probe_words)

def _minimize_mem(rule_counter: Counter, probe_words: List[str]) -> Counter:
    sig_map: Dict[tuple, List[Tuple[str, int]]] = defaultdict(list)
    rule_items = list(rule_counter.items())
    with tqdm(total=len(rule_items), desc=green("  Minimizing"), unit="rule", ncols=88,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
        n_groups = 0
        for rule, gpu_count in rule_items:
            sig = compute_rule_signature(rule, probe_words)
            sig_map[sig].append((rule, gpu_count))
            n_groups = len(sig_map)
            pbar.update(1)
            pbar.set_postfix({"unique_sigs": cyan(str(n_groups))}, refresh=False)
    def _group_key(item: Tuple[str, int]) -> tuple:
        rule, gpu_count = item
        return (-gpu_count, len(rule.split()), rule)
    survivors = Counter()
    n_unsupported = 0
    for sig, group in sig_map.items():
        if sig == ('__UNSUPPORTED__',):
            n_unsupported = len(group)
        best_rule, best_count = min(group, key=_group_key)
        survivors[best_rule] = best_count
    _log_minimize_stats(len(rule_counter), survivors, len(sig_map), n_unsupported)
    return survivors

def _minimize_disk(rule_counter: Counter, probe_words: List[str]) -> Counter:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.db', prefix='rulest_minimize_')
    os.close(tmp_fd)
    try:
        conn = sqlite3.connect(tmp_path)
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous  = OFF')
        conn.execute('PRAGMA temp_store   = MEMORY')
        conn.execute('PRAGMA cache_size   = -131072')
        conn.execute('CREATE TABLE sig_best (sig_hash TEXT PRIMARY KEY, rule TEXT NOT NULL, count INTEGER NOT NULL, depth INTEGER NOT NULL)')
        conn.commit()
        _UPSERT = '''
            INSERT INTO sig_best (sig_hash, rule, count, depth)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sig_hash) DO UPDATE SET
                rule  = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.rule
                    WHEN excluded.count  = sig_best.count AND excluded.depth  < sig_best.depth THEN excluded.rule
                    WHEN excluded.count  = sig_best.count AND excluded.depth  = sig_best.depth AND excluded.rule   < sig_best.rule THEN excluded.rule
                    ELSE sig_best.rule
                END,
                count = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.count
                    ELSE sig_best.count
                END,
                depth = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.depth
                    WHEN excluded.count  = sig_best.count AND excluded.depth  < sig_best.depth THEN excluded.depth
                    ELSE sig_best.depth
                END
        '''
        rule_items = list(rule_counter.items())
        n_total = len(rule_items)
        batch: List[Tuple[str, str, int, int]] = []
        log_info(f"[MINIMIZE] Temp DB     : {dim(tmp_path)}")
        with tqdm(total=n_total, desc=green("  Minimizing"), unit="rule", ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for rule, count in rule_items:
                sig = compute_rule_signature(rule, probe_words)
                sig_str = '\x00'.join(sig)
                sig_hash = hashlib.sha1(sig_str.encode('latin-1', errors='replace')).hexdigest()
                depth = len(rule.split())
                batch.append((sig_hash, rule, count, depth))
                if len(batch) >= MINIMIZE_DISK_BATCH_SIZE:
                    conn.executemany(_UPSERT, batch)
                    conn.commit()
                    batch.clear()
                    (n_sigs,) = conn.execute('SELECT COUNT(*) FROM sig_best').fetchone()
                    pbar.set_postfix({"unique_sigs": cyan(str(n_sigs))}, refresh=False)
                    pbar.update(MINIMIZE_DISK_BATCH_SIZE)
            if batch:
                conn.executemany(_UPSERT, batch)
                conn.commit()
                pbar.update(len(batch))
        survivors = Counter()
        (n_sigs,) = conn.execute('SELECT COUNT(*) FROM sig_best').fetchone()
        cursor = conn.execute('SELECT rule, count FROM sig_best')
        for rule_str, cnt in cursor:
            survivors[rule_str] = cnt
        conn.close()
        _log_minimize_stats(n_total, survivors, n_sigs, n_unsupported=0)
        return survivors
    finally:
        try:
            os.unlink(tmp_path)
            log_debug(f"[MINIMIZE] Temp DB deleted: {tmp_path}")
        except OSError:
            log_warn(f"[MINIMIZE] Could not delete temp DB: {tmp_path}")

def _log_minimize_stats(n_input: int, survivors: Counter, n_groups: int, n_unsupported: int) -> None:
    removed = n_input - len(survivors)
    log_info(f"[MINIMIZE] {green('Done')}")
    log_info(f"           Unique signatures : {bold(cyan(str(n_groups))):>12s}")
    log_info(f"           Rules kept        : {bold(green(str(len(survivors)))):>12s}")
    log_info(f"           Rules removed     : {bold(red(str(removed))):>12s}  ({removed / max(1, n_input):.1%})")
    if n_unsupported:
        log_info(f"           Unsupported (kept 1 each group) : {bold(str(n_unsupported))}")

# ====================================================================
# --- STAGE 0 — TOKEN-STRIP RULE EXTRACTION (core + insert) ---
# ====================================================================
def _hashcat_title_case(s: str) -> str:
    result = list(s)
    cap_next = True
    for i, c in enumerate(result):
        if cap_next and 'a' <= c <= 'z':
            result[i] = c.upper()
            cap_next = False
        if c in (' ', '-', '_'):
            cap_next = True
    return ''.join(result)

def _infer_case_rules(cased_stem: str) -> List[List[str]]:
    stem_lower = cased_stem.lower()
    if cased_stem == stem_lower:
        return [[]]
    candidates: List[List[str]] = []
    if cased_stem == stem_lower.upper():
        candidates.append(['u'])
    if (len(cased_stem) >= 1 and cased_stem[0] == cased_stem[0].upper() and cased_stem[1:] == cased_stem[1:].lower()):
        candidates.append(['c'])
    if (len(cased_stem) >= 1 and cased_stem[0] == cased_stem[0].lower() and cased_stem[1:] == cased_stem[1:].upper()):
        candidates.append(['C'])
    toggled = ''.join(c.lower() if c.isupper() else c.upper() for c in stem_lower)
    if cased_stem == toggled:
        candidates.append(['t'])
    if cased_stem == _hashcat_title_case(stem_lower):
        candidates.append(['E'])
    if not candidates:
        uppercase_positions = [i for i, c in enumerate(cased_stem) if c != stem_lower[i]]
        if uppercase_positions and all(p <= 9 for p in uppercase_positions):
            candidates.append([f'T{p}' for p in uppercase_positions])
    return candidates

def _leet_decode_variants(middle: str, max_ambiguous: int = 3) -> Iterator[Tuple[str, frozenset]]:
    for ch in middle:
        if not ch.isalpha() and ch not in TOKEN_STRIP_LEET_CHARS:
            return
    leet_positions: List[Tuple[int, str, List[Tuple[str, str]]]] = []
    for i, ch in enumerate(middle):
        if ch in TOKEN_STRIP_LEET_CHARS:
            options = _TOKEN_STRIP_LEET_BY_CHAR[ch]
            leet_positions.append((i, ch, options))
    if not leet_positions:
        yield (middle, frozenset())
        return
    n_ambiguous = sum(1 for _, _, opts in leet_positions if len(opts) > 1)
    if n_ambiguous > max_ambiguous:
        return
    choices_per_pos = [opts for _, _, opts in leet_positions]
    for combo in itertools.product(*choices_per_pos):
        decoded = list(middle)
        rules = set()
        for (pos, _orig, _opts), (base_ch, rule_str) in zip(leet_positions, combo):
            decoded[pos] = base_ch
            rules.add(rule_str)
        decoded_str = ''.join(decoded)
        if all(ch.isalpha() for ch in decoded_str):
            yield (decoded_str, frozenset(rules))

def _decode_middle(middle: str, max_ambiguous: int = 3) -> Iterator[Tuple[str, frozenset, List[List[str]]]]:
    for leet_decoded, leet_rules in _leet_decode_variants(middle, max_ambiguous):
        cased_stem = leet_decoded
        stem = cased_stem.lower()
        case_candidates = _infer_case_rules(cased_stem)
        if not case_candidates:
            case_candidates = [[]]
        yield (stem, leet_rules, case_candidates)

def _rule_chain_orderings(case_ops: List[str], leet_ops: List[str], prepend_ops: List[str],
                          append_ops: List[str], leading_ops: Optional[List[str]] = None) -> List[List[str]]:
    lead = leading_ops or []
    seen: Set[tuple] = set()
    result: List[List[str]] = []
    def _add(ops: List[str]) -> None:
        full = lead + ops
        key = tuple(full)
        if key not in seen:
            seen.add(key)
            result.append(full)
    _add(case_ops + leet_ops + prepend_ops + append_ops)
    if case_ops and leet_ops:
        _add(leet_ops + case_ops + prepend_ops + append_ops)
    if prepend_ops and (case_ops or leet_ops):
        _add(prepend_ops + case_ops + leet_ops + append_ops)
    return result

def _boundary_scan(word: str, boundary_chars: Set[str], max_prefix_len: int,
                   max_suffix_len: int, min_stem_len: int) -> List[Tuple[str, str, str]]:
    wlen = len(word)
    triples: List[Tuple[str, str, str]] = []
    for p in range(0, min(max_prefix_len + 1, wlen + 1)):
        if p > 0 and word[p - 1] not in boundary_chars:
            break
        for s in range(0, min(max_suffix_len + 1, wlen - p + 1)):
            if s == 0:
                mid = word[p:]
                suf = ''
            else:
                mid = word[p: wlen - s]
                suf = word[wlen - s:]
            if s > 0 and word[wlen - s] not in boundary_chars:
                break
            if len(mid) >= min_stem_len:
                triples.append((word[:p], mid, suf))
    return triples

def _chains_from_middle(middle: str, prefix: str, suffix: str, base_set: Set[str],
                        max_depth: int, min_stem_len: int, max_leet_amb: int,
                        leading_ops: Optional[List[str]] = None) -> Set[str]:
    target_word = prefix + middle + suffix
    lead = leading_ops or []
    lead_depth = len(lead)
    prepend_ops: List[str] = [f'^{c}' for c in reversed(prefix)]
    append_ops: List[str] = [f'${c}' for c in suffix]
    boundary_depth = len(prepend_ops) + len(append_ops)
    found: Set[str] = set()
    for stem, leet_rules, case_candidates in _decode_middle(middle, max_leet_amb):
        if len(stem) < min_stem_len or stem not in base_set:
            continue
        leet_ops: List[str] = sorted(leet_rules)
        for case_ops in case_candidates:
            transform_depth = len(case_ops) + len(leet_ops)
            total = lead_depth + transform_depth + boundary_depth
            if total > max_depth:
                continue
            for ops in _rule_chain_orderings(case_ops, leet_ops, prepend_ops, append_ops, leading_ops=lead):
                if not ops or len(ops) > max_depth:
                    continue
                chain = ' '.join(ops)
                if not HashcatRuleValidator.validate_rule_for_gpu(chain):
                    continue
                if py_apply_chain(chain, stem) == target_word:
                    found.add(chain)
    return found

def _extract_letter_mode(word: str, base_set: Set[str], max_depth: int, min_stem_len: int,
                         max_prefix_len: int, max_suffix_len: int, max_leet_amb: int) -> Set[str]:
    found: Set[str] = set()
    for prefix, middle, suffix in _boundary_scan(word, TOKEN_STRIP_BOUNDARY, max_prefix_len,
                                                  max_suffix_len, min_stem_len):
        found |= _chains_from_middle(middle, prefix, suffix, base_set, max_depth,
                                     min_stem_len, max_leet_amb)
    return found

def _extract_digit_mode(word: str, base_set: Set[str], max_depth: int, min_stem_len: int,
                        max_prefix_len: int, max_suffix_len: int) -> Set[str]:
    found: Set[str] = set()
    for prefix, middle, suffix in _boundary_scan(word, TOKEN_STRIP_ALPHA_BOUNDARY,
                                                  max_prefix_len, max_suffix_len, min_stem_len):
        if not middle.isdigit():
            continue
        if middle not in base_set:
            continue
        prepend_ops = [f'^{c}' for c in reversed(prefix)]
        append_ops = [f'${c}' for c in suffix]
        total = len(prepend_ops) + len(append_ops)
        if total == 0 or total > max_depth:
            continue
        chain = ' '.join(prepend_ops + append_ops)
        if HashcatRuleValidator.validate_rule_for_gpu(chain):
            if py_apply_chain(chain, middle) == word:
                found.add(chain)
    return found

def _extract_reverse_mode(word: str, base_set: Set[str], max_depth: int, min_stem_len: int,
                          max_prefix_len: int, max_suffix_len: int, max_leet_amb: int) -> Set[str]:
    found: Set[str] = set()
    if max_depth < 1:
        return found
    for prefix, middle, suffix in _boundary_scan(word, TOKEN_STRIP_BOUNDARY, max_prefix_len,
                                                  max_suffix_len, min_stem_len):
        rev_middle = middle[::-1]
        found |= _chains_from_middle(rev_middle, prefix, suffix, base_set, max_depth,
                                     min_stem_len, max_leet_amb, leading_ops=['r'])
    return found

def _extract_duplicate_mode(word: str, base_set: Set[str], max_depth: int,
                            min_stem_len: int) -> Set[str]:
    found: Set[str] = set()
    wlen = len(word)
    for op, builder in (('d', lambda s: s + s), ('f', lambda s: s + s[::-1])):
        for half in range(min_stem_len, wlen // 2 + 1):
            stem_cand = word[:half]
            expected = builder(stem_cand)
            if expected == word and stem_cand in base_set:
                chain = op
                if HashcatRuleValidator.validate_rule_for_gpu(chain):
                    if py_apply_chain(chain, stem_cand) == word:
                        found.add(chain)
    return found

def _extract_delete_edge_mode(word: str, base_set: Set[str], max_depth: int, min_stem_len: int,
                              max_prefix_len: int, max_suffix_len: int, max_leet_amb: int) -> Set[str]:
    found: Set[str] = set()
    if max_depth < 1 or len(word) < min_stem_len + 1:
        return found
    for op, trimmed in (('[', word[1:]), (']', word[:-1])):
        for prefix, middle, suffix in _boundary_scan(trimmed, TOKEN_STRIP_BOUNDARY,
                                                      max_prefix_len, max_suffix_len, min_stem_len):
            found |= _chains_from_middle(middle, prefix, suffix, base_set, max_depth,
                                         min_stem_len, max_leet_amb, leading_ops=[op])
    return found

def _extract_insert_mode(word: str, base_set: Set[str], max_depth: int, min_stem_len: int,
                         max_leet_amb: int, base_by_len: Dict[int, Set[str]]) -> Set[str]:
    found: Set[str] = set()
    if max_depth < 2:
        return found
    wlen = len(word)
    if wlen - 1 >= min_stem_len:
        for pos in range(min(wlen, 10)):
            candidate = word[:pos] + word[pos + 1:]
            if candidate in base_set:
                rule = f"i{pos}{word[pos]}"
                if HashcatRuleValidator.validate_rule_for_gpu(rule):
                    if py_apply_chain(rule, candidate) == word:
                        found.add(rule)
    if max_depth >= 3 and wlen - 2 >= min_stem_len:
        for i in range(min(wlen, 10)):
            for j in range(i + 1, min(wlen, 10)):
                candidate = word[:i] + word[i + 1:j] + word[j + 1:]
                if len(candidate) < min_stem_len:
                    continue
                if candidate in base_set:
                    chain = f"i{i}{word[i]} i{j}{word[j]}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain):
                        if py_apply_chain(chain, candidate) == word:
                            found.add(chain)
    return found

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 multiprocessing infrastructure (core only)
# ──────────────────────────────────────────────────────────────────────────────
def _worker_init_p0(base_set=None, base_by_len=None) -> None:
    global _p0_worker_base_set, _p0_worker_base_by_len
    if base_set is not None: _p0_worker_base_set = base_set
    if base_by_len is not None: _p0_worker_base_by_len = base_by_len

def _process_chunk_p0(args: Tuple) -> Set[str]:
    (words_chunk, max_depth, min_stem_len, max_leet_amb,
     max_prefix_len, max_suffix_len) = args
    base_set = _p0_worker_base_set
    base_by_len = _p0_worker_base_by_len
    found: Set[str] = set()
    for word in words_chunk:
        if not word or len(word) > MAX_WORD_LEN:
            continue
        n_digits = sum(1 for c in word if c.isdigit())
        n_alpha = sum(1 for c in word if c.isalpha())
        digit_primary = n_digits > n_alpha
        if digit_primary:
            found |= _extract_digit_mode(word, base_set, max_depth, min_stem_len,
                                         max_prefix_len, max_suffix_len)
            if n_alpha >= min_stem_len:
                found |= _extract_letter_mode(word, base_set, max_depth, min_stem_len,
                                              max_prefix_len, max_suffix_len, max_leet_amb)
        else:
            found |= _extract_letter_mode(word, base_set, max_depth, min_stem_len,
                                          max_prefix_len, max_suffix_len, max_leet_amb)
            if n_digits > 0:
                found |= _extract_digit_mode(word, base_set, max_depth, min_stem_len,
                                             max_prefix_len, max_suffix_len)
        if max_depth >= 2:
            found |= _extract_reverse_mode(word, base_set, max_depth, min_stem_len,
                                           max_prefix_len, max_suffix_len, max_leet_amb)
            found |= _extract_delete_edge_mode(word, base_set, max_depth, min_stem_len,
                                               max_prefix_len, max_suffix_len, max_leet_amb)
        if len(word) >= 2 * min_stem_len:
            found |= _extract_duplicate_mode(word, base_set, max_depth, min_stem_len)
        if max_depth >= 2:
            found |= _extract_insert_mode(word, base_set, max_depth, min_stem_len,
                                          max_leet_amb, base_by_len)
    return found

def _generate_toggle_chain_seeds(max_depth: int) -> List[str]:
    LEET_OPS = ['sa@', 'se3', 'so0', 'si1', 'sl1', 'ss5', 'ss$', 'st7', 'sa4', 'si!']
    DOUBLE_LEET = [('se3', 'si1'), ('se3', 'sl1'), ('sa@', 'so0'), ('ss5', 'so0'),
                   ('si1', 'so0'), ('se3', 'so0'), ('ss$', 'se3'), ('sa4', 'sl1')]
    seeds: Set[str] = set()
    def _add(ops: List[str]) -> None:
        if not ops or len(ops) > max_depth:
            return
        chain = ' '.join(ops)
        if HashcatRuleValidator.validate_rule_for_gpu(chain):
            seeds.add(chain)
    for n in range(0, min(10, max_depth)):
        t_ops = [f'T{i}' for i in range(n + 1)]
        _add(t_ops)
        for leet in LEET_OPS:
            _add(t_ops + [leet])
            if n >= 1:
                _add([leet] + t_ops)
        for l1, l2 in DOUBLE_LEET:
            _add(t_ops + [l1, l2])
            _add([l1] + t_ops + [l2])
    for n in range(1, min(5, max_depth)):
        t_ops = [f'T{i * 2}' for i in range(n + 1)]
        _add(t_ops)
        for leet in LEET_OPS:
            _add(t_ops + [leet])
            _add([leet] + t_ops)
        for l1, l2 in DOUBLE_LEET:
            _add(t_ops + [l1, l2])
    for n in range(1, min(5, max_depth)):
        t_ops = [f'T{i * 2 + 1}' for i in range(n + 1)]
        _add(t_ops)
        for leet in LEET_OPS:
            _add(t_ops + [leet])
            _add([leet] + t_ops)
        for l1, l2 in DOUBLE_LEET:
            _add(t_ops + [l1, l2])
    for leet in LEET_OPS:
        _add(['T0', leet])
        _add([leet, 'T0'])
    return sorted(seeds)

def extract_token_strip_rules(target_words: List[str], base_set: Set[str],
                              max_depth: int = 0, min_stem_len: int = 4,
                              max_prefix_len: int = 4, max_suffix_len: int = 4,
                              max_leet_ambiguity: int = 3, workers: int = 0,
                              chunk_size: int = 0) -> List[str]:
    if max_depth <= 0:
        max_depth = MAX_HASHCAT_CHAIN
    n_workers = workers or mp.cpu_count()
    n_words = len(target_words)
    if chunk_size <= 0:
        chunk_size = max(500, n_words // (n_workers * 4) + 1)
    base_by_len: Dict[int, Set[str]] = defaultdict(set)
    for w in base_set:
        base_by_len[len(w)].add(w)
    base_by_len = dict(base_by_len)

    global _p0_worker_base_set, _p0_worker_base_by_len
    _p0_worker_base_set = base_set
    _p0_worker_base_by_len = base_by_len

    chunks = [target_words[i:i + chunk_size] for i in range(0, n_words, chunk_size)]
    n_chunks = len(chunks)
    task_args = [(chunk, max_depth, min_stem_len, max_leet_ambiguity,
                  max_prefix_len, max_suffix_len) for chunk in chunks]

    log_info(f"[S0]    Workers: {bold(str(n_workers))}  |  chunks: {bold(str(n_chunks))} × ~{chunk_size}")
    found: Set[str] = set()
    use_fork = hasattr(os, 'fork')
    ctx = mp.get_context('fork' if use_fork else 'spawn')
    if use_fork:
        pool_kw = dict(processes=n_workers, initializer=_worker_init_p0)
    else:
        pool_kw = dict(processes=n_workers, initializer=_worker_init_p0,
                       initargs=(base_set, base_by_len))
    _s0_t0 = time.time(); _s0_words_done = 0
    with ctx.Pool(**pool_kw) as pool:
        with tqdm(total=n_words, desc=green("  STAGE 0 "), unit="word", ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for task_arg, chunk_result in zip(task_args, pool.imap_unordered(_process_chunk_p0, task_args)):
                found |= chunk_result
                _s0_words_done += len(task_arg[0])
                _s0_el = time.time() - _s0_t0
                _s0_spd = _fmt_speed(_s0_words_done / _s0_el if _s0_el > 0 else 0, "words")
                pbar.set_postfix({"rules": cyan(str(len(found))), "spd": green(_s0_spd)}, refresh=False)
                pbar.update(len(task_arg[0]))
    _p0_worker_base_set = set()
    _p0_worker_base_by_len = {}
    log_debug("[S0]    Worker globals cleared")
    return sorted(found)

def _log_token_strip_stats(n_words: int, rules: List[str], inject_sbd: bool) -> None:
    if not rules:
        log_info(f"[S0]    {yellow('0')} rules extracted by token-strip ({n_words:,} target words scanned)")
        return
    depth_dist: Dict[int, int] = defaultdict(int)
    mode_counts: Dict[str, int] = defaultdict(int)
    for r in rules:
        depth_dist[len(r.split())] += 1
        toks = r.split()
        first = toks[0] if toks else ''
        if first == 'r':
            mode_counts['reverse'] += 1
        elif first == 'd':
            mode_counts['dup'] += 1
        elif first == 'f':
            mode_counts['fold'] += 1
        elif first in ('[', ']'):
            mode_counts['del-edge'] += 1
        elif first.startswith('T') and len(first) == 2:
            mode_counts['toggle'] += 1
        elif first.startswith('i') and len(first) == 3:
            mode_counts['insert'] += 1
        elif all(c.isdigit() or c in ('^', '$', ' ') for c in r) and any(c.isdigit() for c in r):
            mode_counts['digit-bnd'] += 1
        else:
            mode_counts['letter'] += 1
    depth_summary = '  '.join(f"d{d}:{depth_dist[d]:,}" for d in sorted(depth_dist))
    inj = green('injected into STAGE S sbd') if inject_sbd else dim('STAGE S inactive')
    mode_str = '  '.join(f"{k}:{v}" for k, v in sorted(mode_counts.items()) if v)
    log_info(f"[S0]    {bold(green(str(len(rules))))} rules extracted by token-strip  ({depth_summary})  → {inj}")
    if mode_str:
        log_info(f"[S0]    Mode breakdown  : {dim(mode_str)}")

# --------------------------------------------------------------------
# GPU device helpers (unchanged)
# --------------------------------------------------------------------
def get_all_devices():
    devices = []
    for plat in cl.get_platforms():
        for dtype in (cl.device_type.GPU, cl.device_type.CPU):
            try:
                for dev in plat.get_devices(dtype): devices.append((plat, dev))
            except Exception: pass
    return devices

def list_devices():
    devs = get_all_devices()
    if not devs:
        log_error("No OpenCL devices found."); sys.exit(1)
    log_info(f"\n{blue('Available OpenCL devices:')}")
    for i, (p, d) in enumerate(devs):
        t = cl.device_type.to_string(d.get_info(cl.device_info.TYPE))
        log_info(f"  {cyan(str(i)+':')} {d.get_info(cl.device_info.NAME)} ({t}) — {p.name}")
    print()

def get_device_by_spec(spec: Optional[str]):
    if spec is None: return get_best_gpu_device()
    devs = get_all_devices()
    if not devs: raise RuntimeError("No OpenCL devices found")
    if spec.isdigit():
        i = int(spec)
        if 0 <= i < len(devs): return devs[i][1]
        raise RuntimeError(f"Device index {i} out of range")
    lo = spec.lower()
    m  = [d for _, d in devs if lo in d.get_info(cl.device_info.NAME).lower()]
    if len(m) == 1: return m[0]
    if len(m) >  1:
        log_warn(f"Multiple devices match '{spec}'; using first."); return m[0]
    raise RuntimeError(f"No device matching '{spec}'")

def get_best_gpu_device():
    best = None; best_score = -1
    for p in cl.get_platforms():
        try: devs = p.get_devices(cl.device_type.GPU)
        except: continue
        for d in devs:
            name  = d.get_info(cl.device_info.NAME).upper()
            score = (10 if 'NVIDIA' in name or 'AMD' in name else 0)
            score += (5 if 'RTX' in name or 'GTX' in name else 0)
            score += d.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            if score > best_score: best_score = score; best = d
    if best is None:
        for p in cl.get_platforms():
            try: return p.get_devices(cl.device_type.GPU)[0]
            except: pass
    if best is None: raise RuntimeError("No GPU found")
    return best

def estimate_free_vram(device):
    try: return int(device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * VRAM_USAGE_FACTOR)
    except: return 1 * 1024**3

def get_max_allocation(device):
    try: return device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    except: return 1024**3

# --------------------------------------------------------------------
# Automatic bloom size based on VRAM
# --------------------------------------------------------------------
def get_auto_bloom_mb(free_vram_gb: float) -> int:
    """Return recommended bloom filter size in MB based on free VRAM."""
    if free_vram_gb >= 8.0:
        return 512
    elif free_vram_gb >= 4.0:
        return 128
    else:
        return 64

# --------------------------------------------------------------------
# Dynamic parameters (unchanged except bloom sizing and removed warnings)
# --------------------------------------------------------------------
def calculate_dynamic_parameters(base_count, target_count, device=None,
                                  target_hours=0.5, bloom_mb_override=None):
    if device:
        try:
            mwgs    = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            mcu     = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            mhz     = device.get_info(cl.device_info.MAX_CLOCK_FREQUENCY)   # MHz
            fv      = estimate_free_vram(device)
            vgb     = fv / 1024**3
            name_up = device.get_info(cl.device_info.NAME).upper()
            isn     = 'NVIDIA' in name_up
            lws     = max(s for s in POSSIBLE_WORK_GROUP_SIZES if s <= mwgs)
            if isn and mcu >= 38: lws = min(512, lws)

            # Dynamic throughput estimate: CU × clock × base_rate × arch_factor
            # Calibrated so that 40 CU @ 1500 MHz → 120 M/s (= BASELINE_COMBOS_PER_SEC)
            # and 20 CU @ 1000 MHz → 40 M/s (= LOW_END_COMBOS_PER_SEC).
            _BASE_RATE = 2_000   # combos/s per (CU · MHz)
            if isn:
                _arch = 1.5   # NVIDIA: wider warps, stronger int pipelines
            elif 'AMD' in name_up or 'RADEON' in name_up or 'GFX' in name_up:
                _arch = 1.2
            elif 'INTEL' in name_up:
                _arch = 0.7
            else:
                _arch = 1.0
            est = max(5_000_000, min(2_000_000_000, int(mcu * mhz * _BASE_RATE * _arch)))

            log_debug(f"GPU: CU={mcu}, MHz={mhz}, VRAM~{vgb:.1f}GB, WGS={lws}, "
                      f"arch_factor={_arch}, est={est//1_000_000}M combos/s")
        except Exception:
            lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2*1024**3; vgb = 2.0
    else:
        lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2*1024**3; vgb = 2.0

    vram_scale = max(0.25, min(1.0, vgb / 8.0))
    ts         = target_hours * 3600

    # Automatic bloom size if not overridden
    if bloom_mb_override is not None and bloom_mb_override > 0:
        eff_bloom = bloom_mb_override
    else:
        eff_bloom = get_auto_bloom_mb(vgb)
        log_debug(f"[BLOOM] Auto bloom size: {eff_bloom} MB (VRAM: {vgb:.1f} GB)")

    bsize_b    = eff_bloom * 1024 * 1024
    bloom_bits = bsize_b * 8

    if target_count > 0:
        fill = 1.0 - math.exp(-BLOOM_HASH_FUNCTIONS * target_count / bloom_bits)
        fpr  = fill ** BLOOM_HASH_FUNCTIONS
        msg  = f"Bloom: {bsize_b//1024//1024}MB, fill={fill:.3%}, FPR~{fpr:.6%}"
        if fpr > 0.01:
            log_warn(f"Bloom filter FPR {fpr:.3%} is high — consider using --bloom-mb with a larger value (e.g. {eff_bloom*2} MB)")
        log_debug(msg)

    params = {
        'BLOOM_FILTER_SIZE'         : bloom_bits,
        'WORDS_PER_BATCH'           : max(1000, int(BASE_WORDS_PER_BATCH  * vram_scale)),
        'CHAINS_PER_BATCH'          : max(500,  int(BASE_CHAINS_PER_BATCH * vram_scale)),
        'WORD_SUB_BATCH'            : max(5000, int(BASE_WORD_SUB_BATCH   * vram_scale)),
        'MAX_SAFE_RESULTS_PER_BATCH': max(5000, int(BASE_MAX_SAFE_RESULTS * vram_scale)),
        'MAX_CHAINS_TO_FIND'        : 2**31 - 1,
        'LOCAL_WORK_SIZE'           : lws,
        'OPTIMAL_GLOBAL_MULTIPLIER' : mcu * OPTIMAL_GLOBAL_MULTIPLIER_BASE,
        'EST_COMBOS_PER_SEC'        : est,
        'TARGET_SECONDS'            : ts,
        'vram_scale'                : vram_scale,
        'free_vram'                 : fv,
    }
    log_debug(
        f"[PARAMS] bloom={bloom_bits//8//1024//1024}MB  "
        f"words_batch={params['WORDS_PER_BATCH']:,}  "
        f"chains_batch={params['CHAINS_PER_BATCH']:,}  "
        f"word_sub={params['WORD_SUB_BATCH']:,}  "
        f"max_results={params['MAX_SAFE_RESULTS_PER_BATCH']:,}  "
        f"lws={lws}  est={est//1_000_000}M/s  "
        f"target={ts:.0f}s  vram_scale={vram_scale:.2f}"
    )
    return params

# --------------------------------------------------------------------
# GPU-compatible rules generator (unchanged)
# --------------------------------------------------------------------
class GPUCompatibleRulesGenerator:
    def __init__(self):
        self.validator = HashcatRuleValidator()

    def generate_gpu_compatible_rules(self):
        rules  = set()
        digits = '0123456789'
        rules.update(['l','u','c','C','t','r','d','f','q','E','{','}','[',']','k','K',':'])
        for cmd in ('T','D','L','R','+','-','.',',', "'", 'z','Z','y','Y'):
            for pos in digits: rules.add(f'{cmd}{pos}')
        for pos in digits:
            rules.add(f'p{pos}')
        for cmd in ('x','*','O'):
            for p1 in digits:
                for p2 in digits: rules.add(f'{cmd}{p1}{p2}')
        for i in range(33, 127):
            ch = chr(i)
            rules.add(f'^{ch}'); rules.add(f'${ch}'); rules.add(f'@{ch}')
        for orig in string.ascii_lowercase + string.ascii_uppercase:
            for sub in string.digits + string.punctuation:
                if orig != sub: rules.add(f's{orig}{sub}')
        chars = string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?/~'
        for pos in digits:
            for ch in chars:
                rules.add(f'i{pos}{ch}'); rules.add(f'o{pos}{ch}')
        for n in range(1, 10):
            for ch in ('p','y','Y','z','Z'): rules.add(f'{ch}{n}')
        for sep in '-_.,;:|/\\+*&^%$#@!~`':
            rules.add(f'e{sep}')
        chars2 = string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?/~'
        for n in digits:
            for sep in chars2: rules.add(f'3{n}{sep}')
        valid = [r for r in rules
                 if self.validator.validate_rule_for_gpu(r) and 1 <= len(r) <= MAX_RULE_LEN]
        log_debug(f"Generated {len(valid):,} GPU-compatible atomic rules")
        return valid

# --------------------------------------------------------------------
# OpenCL kernel (unchanged)
# --------------------------------------------------------------------
GPU_KERNEL_TEMPLATE = r"""
/* Enable global int32 atomics on OpenCL 1.x (harmless on 2.0+).
   Without this pragma on OpenCL 1.x (e.g. NVIDIA) atomic_or on
   __global memory is undefined and silently produces wrong results. */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define MAX_WORD_LEN         {MAX_WORD_LEN}
#define MAX_RULE_LEN         {MAX_RULE_LEN}
#define MAX_OUTPUT_LEN       {MAX_OUTPUT_LEN}
#define MAX_CHAIN_STRING_LEN {MAX_CHAIN_STRING_LEN}
#define MAX_CHAINS_TO_FIND   {MAX_SAFE_RESULTS_PER_BATCH}
#define MAX_CHAIN_DEPTH      {MAX_CHAIN_DEPTH}
#define BLOOM_FILTER_SIZE    {BLOOM_FILTER_SIZE}
#define BLOOM_HASH_FUNCTIONS {BLOOM_HASH_FUNCTIONS}

uint fnv1a(const unsigned char *d,int len,uint seed){{
    uint h=seed^2166136261U;for(int i=0;i<len;i++){{h^=d[i];h*=16777619U;}}return h;
}}
inline int isl(unsigned char c){{return c>='a'&&c<='z';}}
inline int isu(unsigned char c){{return c>='A'&&c<='Z';}}
inline int isd(unsigned char c){{return c>='0'&&c<='9';}}
inline unsigned char tol(unsigned char c){{return isu(c)?c+32:c;}}
inline unsigned char tou(unsigned char c){{return isl(c)?c-32:c;}}
inline unsigned char tog(unsigned char c){{return isl(c)?c-32:(isu(c)?c+32:c);}}

int bloom(__global const uchar *bf,const unsigned char *w,int len){{
    uint h1=fnv1a(w,len,0xDEADBEEF),h2=fnv1a(w,len,0xCAFEBABE);
    for(int i=0;i<BLOOM_HASH_FUNCTIONS;i++){{
        uint idx=(uint)(((ulong)h1+(ulong)i*(ulong)h2)%(ulong)BLOOM_FILTER_SIZE);
        if(!(bf[idx/8]&(1<<(idx%8))))return 0;
    }}return 1;
}}

int apply(const unsigned char *rs,int rl,
          const unsigned char *in,int il,
          unsigned char *out,int *ol){{
    *ol=il;for(int i=0;i<il;i++)out[i]=in[i];out[il]='\0';
    if(!rl||!il)return 1;
    unsigned char cmd=rs[0];int changed=0;
    if(rl==1){{
        switch(cmd){{
        case 'l':for(int i=0;i<*ol;i++)out[i]=tol(out[i]);changed=1;break;
        case 'u':for(int i=0;i<*ol;i++)out[i]=tou(out[i]);changed=1;break;
        case 'c':if(*ol>0){{out[0]=tou(out[0]);for(int i=1;i<*ol;i++)out[i]=tol(out[i]);}}changed=1;break;
        case 'C':if(*ol>0){{out[0]=tol(out[0]);for(int i=1;i<*ol;i++)out[i]=tou(out[i]);}}changed=1;break;
        case 't':for(int i=0;i<*ol;i++)out[i]=tog(out[i]);changed=1;break;
        case 'r':for(int i=0;i<*ol/2;i++){{unsigned char t=out[i];out[i]=out[*ol-1-i];out[*ol-1-i]=t;}}changed=1;break;
        case 'd':if(*ol*2<=MAX_OUTPUT_LEN){{for(int i=0;i<*ol;i++)out[*ol+i]=out[i];*ol*=2;}}changed=1;break;
        case 'f':if(*ol*2<=MAX_OUTPUT_LEN){{for(int i=0;i<*ol;i++)out[*ol+i]=out[*ol-1-i];*ol*=2;}}changed=1;break;
        case '{{':if(*ol>1){{unsigned char f=out[0];for(int i=0;i<*ol-1;i++)out[i]=out[i+1];out[*ol-1]=f;}}changed=1;break;
        case '}}':if(*ol>1){{unsigned char l=out[*ol-1];for(int i=*ol-1;i>0;i--)out[i]=out[i-1];out[0]=l;}}changed=1;break;
        case '[':if(*ol>0){{for(int i=0;i<*ol-1;i++)out[i]=out[i+1];(*ol)--;}}changed=1;break;
        case ']':if(*ol>0)(*ol)--;changed=1;break;
        case 'k':if(*ol>=2){{unsigned char t=out[0];out[0]=out[1];out[1]=t;}}changed=1;break;
        case 'K':if(*ol>=2){{unsigned char t=out[*ol-2];out[*ol-2]=out[*ol-1];out[*ol-1]=t;}}changed=1;break;
        case ':':break;
        case 'q':if(*ol*2<=MAX_OUTPUT_LEN){{
            unsigned char tmp[MAX_OUTPUT_LEN];for(int i=0;i<*ol;i++)tmp[i]=out[i];
            int idx=0;for(int i=0;i<*ol;i++){{out[idx++]=tmp[i];out[idx++]=tmp[i];}}*ol*=2;
        }}changed=1;break;
        case 'E':{{int cap=1;for(int i=0;i<*ol;i++){{
            if(cap&&isl(out[i])){{out[i]=tou(out[i]);cap=0;}}
            if(out[i]==' '||out[i]=='-'||out[i]=='_')cap=1;
        }}}}changed=1;break;
        }}
    }}else if(rl==2){{
        unsigned char p=rs[1];
        if(cmd=='^'){{if(*ol+1<=MAX_OUTPUT_LEN){{for(int i=*ol;i>0;i--)out[i]=out[i-1];out[0]=p;(*ol)++;changed=1;}}}}
        else if(cmd=='$'){{if(*ol+1<=MAX_OUTPUT_LEN){{out[*ol]=p;(*ol)++;changed=1;}}}}
        else if(cmd=='@'){{int nl=0;for(int i=0;i<*ol;i++){{if(out[i]!=p)out[nl++]=out[i];else changed=1;}}*ol=nl;}}
        else if(cmd=='p'){{int n=p-'0';if(n>0&&*ol*(n+1)<=MAX_OUTPUT_LEN){{int o=*ol;for(int r=0;r<n;r++){{for(int i=0;i<o;i++)out[*ol+i]=out[i];*ol+=o;}}changed=1;}}}}
        else if(cmd=='T'&&isd(p)){{int pos=p-'0';if(pos<*ol){{out[pos]=tog(out[pos]);changed=1;}}}}
        else if(cmd=='D'&&isd(p)){{int pos=p-'0';if(pos<*ol){{for(int i=pos;i<*ol-1;i++)out[i]=out[i+1];(*ol)--;changed=1;}}}}
        else if(cmd=='L'&&isd(p)){{int pos=p-'0';if(pos<*ol){{out[pos]<<=1;changed=1;}}}}
        else if(cmd=='R'&&isd(p)){{int pos=p-'0';if(pos<*ol){{out[pos]>>=1;changed=1;}}}}
        else if(cmd=='+'&&isd(p)){{int pos=p-'0';if(pos<*ol&&out[pos]<255){{out[pos]++;changed=1;}}}}
        else if(cmd=='-'&&isd(p)){{int pos=p-'0';if(pos<*ol&&out[pos]>0){{out[pos]--;changed=1;}}}}
        else if((cmd=='.'||cmd==',')&&isd(p)){{int pos=p-'0';if(pos<*ol){{out[pos]+=(cmd=='.'?1:-1);changed=1;}}}}
        else if(cmd=='\''&&isd(p)){{int pos=p-'0';if(pos<*ol){{*ol=pos+1;changed=1;}}}}
        else if(cmd=='z'&&isd(p)){{int n=p-'0';if(*ol+n<=MAX_OUTPUT_LEN){{unsigned char f=out[0];for(int i=*ol+n-1;i>=n;i--)out[i]=out[i-n];for(int i=0;i<n;i++)out[i]=f;*ol+=n;changed=1;}}}}
        else if(cmd=='Z'&&isd(p)){{int n=p-'0';if(*ol+n<=MAX_OUTPUT_LEN){{unsigned char l=out[*ol-1];for(int i=0;i<n;i++)out[*ol+i]=l;*ol+=n;changed=1;}}}}
        else if(cmd=='y'&&isd(p)){{int n=p-'0';if(*ol+n<=MAX_OUTPUT_LEN){{for(int i=0;i<n;i++)out[*ol+i]=out[i];*ol+=n;changed=1;}}}}
        else if(cmd=='Y'&&isd(p)){{int n=p-'0';if(*ol+n<=MAX_OUTPUT_LEN){{for(int i=0;i<n;i++)out[*ol+i]=out[*ol-n+i];*ol+=n;changed=1;}}}}
    }}else if(rl==3){{
        unsigned char p1=rs[1],p2=rs[2];
        if(cmd=='s'){{for(int i=0;i<*ol;i++)if(out[i]==p1){{out[i]=p2;changed=1;}}}}
        else if(cmd=='i'&&isd(p1)){{int pos=p1-'0';if(pos<=*ol&&*ol+1<=MAX_OUTPUT_LEN){{for(int i=*ol;i>pos;i--)out[i]=out[i-1];out[pos]=p2;(*ol)++;changed=1;}}}}
        else if(cmd=='o'&&isd(p1)){{int pos=p1-'0';if(pos<*ol){{out[pos]=p2;changed=1;}}}}
        else if(cmd=='e'){{int cap=1;for(int i=0;i<*ol;i++){{if(cap&&isl(out[i])){{out[i]=tou(out[i]);cap=0;}}if(out[i]==p1)cap=1;}}changed=1;}}
        else if(cmd=='x'&&isd(p1)&&isd(p2)){{int a=p1-'0',b=p2-'0';if(a>b){{int t=a;a=b;b=t;}}if(a<*ol){{int nl=0;for(int i=a;i<=b&&i<*ol;i++)out[nl++]=out[i];*ol=nl;changed=1;}}}}
        else if(cmd=='O'&&isd(p1)&&isd(p2)){{int n=p1-'0',m=p2-'0';if(n<*ol&&m>0){{int e=n+m;if(e>*ol)e=*ol;int sh=e-n;for(int i=e;i<*ol;i++)out[i-sh]=out[i];*ol-=sh;changed=1;}}}}
        else if(cmd=='*'&&isd(p1)&&isd(p2)){{int a=p1-'0',b=p2-'0';if(a<*ol&&b<*ol&&a!=b){{unsigned char t=out[a];out[a]=out[b];out[b]=t;changed=1;}}}}
        else if(cmd=='3'&&isd(p1)){{int n=p1-'0',cnt=0,found=-1;for(int i=0;i<*ol;i++)if(out[i]==p2&&++cnt==n){{found=i;break;}}if(found!=-1&&found+1<*ol){{out[found+1]=tog(out[found+1]);changed=1;}}}}
    }}
    out[*ol]='\0';return changed?1:0;
}}

__kernel void find_single_rules_gpu(
    __global const unsigned char *bw,__global const int *bo,__global const int *bl,
    __global const unsigned char *rs,__global const int *ro,__global const int *rl,
    __global const uchar *bf,const int nw,const int nr,
    __global char *found,__global volatile int *cnt)
{{
    int gid=get_global_id(0);if(gid>=nw*nr)return;
    int wi=gid/nr,ri=gid%nr;
    unsigned char iw[MAX_WORD_LEN],ow[MAX_OUTPUT_LEN],rr[MAX_RULE_LEN];
    int wl=bl[wi];for(int i=0;i<wl;i++)iw[i]=bw[bo[wi]+i];iw[wl]='\0';
    int rlen=rl[ri];for(int i=0;i<rlen;i++)rr[i]=rs[ro[ri]+i];rr[rlen]='\0';
    int ol;apply(rr,rlen,iw,wl,ow,&ol);
    int same=(ol==wl);for(int i=0;i<wl&&same;i++)if(ow[i]!=iw[i])same=0;
    if(!same&&ol>0&&bloom(bf,ow,ol)){{
        int idx=atomic_inc(cnt);
        if(idx<MAX_CHAINS_TO_FIND){{
            __global char *p=found+idx*MAX_CHAIN_STRING_LEN;
            for(int i=0;i<rlen&&i<MAX_CHAIN_STRING_LEN-1;i++)p[i]=rr[i];p[rlen]='\0';
        }}
    }}
}}

__kernel void find_rule_chains_gpu(
    __global const unsigned char *bw,__global const int *bo,__global const int *bl,
    __global const unsigned char *rs,__global const int *ro,__global const int *rl,
    __global const int *cseq,__global const int *cdep,
    __global const uchar *bf,const int nw,const int nc,const int mcd,
    __global char *found,__global volatile int *cnt)
{{
    int gid=get_global_id(0);if(gid>=nw*nc)return;
    int wi=gid/nc,ci=gid%nc;
    unsigned char cur[MAX_OUTPUT_LEN],tmp[MAX_OUTPUT_LEN],rr[MAX_RULE_LEN];
    char cb[MAX_CHAIN_STRING_LEN];int cp=0;
    int wl=bl[wi];for(int i=0;i<wl;i++)cur[i]=bw[bo[wi]+i];cur[wl]='\0';
    int cl_=wl,dep=cdep[ci];if(dep<1||dep>mcd)return;
    __global const unsigned char *wp=bw+bo[wi];
    for(int d=0;d<dep;d++){{
        int ri=cseq[ci*mcd+d];if(ri<0)break;
        int rlen=rl[ri];for(int i=0;i<rlen;i++)rr[i]=rs[ro[ri]+i];rr[rlen]='\0';
        for(int i=0;i<rlen&&cp<MAX_CHAIN_STRING_LEN-2;i++)cb[cp++]=rr[i];
        if(d<dep-1&&cp<MAX_CHAIN_STRING_LEN-1)cb[cp++]=' ';
        int nl;apply(rr,rlen,cur,cl_,tmp,&nl);
        if(nl==0)return;
        for(int i=0;i<nl;i++)cur[i]=tmp[i];cur[nl]='\0';cl_=nl;
    }}
    cb[cp]='\0';
    int same=(cl_==wl);for(int i=0;i<wl&&same;i++)if(cur[i]!=wp[i])same=0;
    if(!same&&bloom(bf,cur,cl_)){{
        int idx=atomic_inc(cnt);
        if(idx<MAX_CHAINS_TO_FIND){{
            __global char *p=found+idx*MAX_CHAIN_STRING_LEN;
            for(int i=0;i<cp&&i<MAX_CHAIN_STRING_LEN-1;i++)p[i]=cb[i];p[cp]='\0';
        }}
    }}
}}

/* ---------------------------------------------------------------
 * build_bloom_filter_gpu
 *
 * Each work-item processes one target word and sets its
 * BLOOM_HASH_FUNCTIONS bits using the same FNV-1a pair that
 * bloom() uses for membership checks.  atomic_or guarantees
 * race-free writes when multiple work-items hash to the same bit.
 *
 * The filter is a flat int32 array (BLOOM_FILTER_SIZE / 32 words).
 * Bit `idx` lives at bf[idx/32], bit position idx%32.
 * After readback, bf.view(uint8) is byte-identical to the CPU path,
 * so the byte-level bloom() check kernel needs no changes.
 *
 * Index arithmetic uses ulong throughout to avoid the (uint) cast
 * overflow when BLOOM_FILTER_SIZE == 2^32 (512 MB filter).
 * --------------------------------------------------------------- */
__kernel void build_bloom_filter_gpu(
    __global const unsigned char *bw,
    __global const int            *bo,
    __global const int            *bl,
    const int                      nw,
    volatile __global int         *bf)
{{
    int gid = get_global_id(0);
    if (gid >= nw) return;

    int wlen = bl[gid];
    __global const unsigned char *word = bw + bo[gid];

    uint h1 = 0xDEADBEEFu ^ 2166136261U;
    for (int i = 0; i < wlen; i++) {{ h1 ^= word[i]; h1 *= 16777619U; }}

    uint h2 = 0xCAFEBABEu ^ 2166136261U;
    for (int i = 0; i < wlen; i++) {{ h2 ^= word[i]; h2 *= 16777619U; }}

    for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {{
        /* 64-bit modulo: when BLOOM_FILTER_SIZE == 2^32 the literal is
           typed ulong by the OpenCL compiler; casting to (uint) wraps
           it to 0, making every % 0 undefined — use ulong throughout. */
        uint idx  = (uint)(((ulong)h1 + (ulong)(uint)i * (ulong)h2)
                           % (ulong)BLOOM_FILTER_SIZE);
        uint widx = idx >> 5u;
        int  mask = (int)(1u << (idx & 31u));
        atomic_or(bf + (int)widx, mask);
    }}
}}
"""

# --------------------------------------------------------------------
# GPU Engine (unchanged except minor logging)
# --------------------------------------------------------------------
class GPUEngine:
    def __init__(self, params):
        self.params              = params
        self.context = self.queue = self.device = self.program = None
        self.max_work_group_size = 512
        self.local_work_size     = params.get('LOCAL_WORK_SIZE', 512)
        self.bloom_buf           = None
        self.bloom_np            = None
        self.rule_index          = {}
        self.gpu_rules           = []
        self.kernel_single       = None
        self.kernel_chain        = None
        self.kernel_bloom        = None
        self._consecutive_errors = 0
        self._MAX_CONSECUTIVE_ERRORS = 5
        self._cached_free_vram   = None
        self._cached_max_alloc   = None
        self._rules_buf          = None
        self._rules_offsets_buf  = None
        self._rules_lengths_buf  = None
        self._rules_buf_key      = None

    def get_free_vram(self):      return estimate_free_vram(self.device)
    def get_max_allocation(self): return get_max_allocation(self.device)

    def _refresh_cached_limits(self):
        self._cached_free_vram = estimate_free_vram(self.device)
        self._cached_max_alloc = get_max_allocation(self.device)

    def safe_output_buffer_size(self, words_count, chains_count):
        fv    = self._cached_free_vram if self._cached_free_vram is not None else self.get_free_vram()
        ma    = self._cached_max_alloc if self._cached_max_alloc is not None else self.get_max_allocation()
        avail = min(fv, ma) - 5*1024**2
        return min(max(avail, 0) // MAX_CHAIN_STRING_LEN,
                   self.params['MAX_SAFE_RESULTS_PER_BATCH'],
                   words_count * chains_count,
                   5000) or 1

    def initialize_gpu(self, device_spec):
        try:
            self.device          = get_device_by_spec(device_spec)
            self.context         = cl.Context([self.device])
            self.queue           = cl.CommandQueue(self.context)
            self.max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            self.local_work_size = min(self.local_work_size, self.max_work_group_size)
            while self.max_work_group_size % self.local_work_size != 0 and self.local_work_size > 32:
                self.local_work_size //= 2
            self._refresh_cached_limits()
            vram_gb = self._cached_free_vram / 1024**3
            log_info(f"[GPU]  {bold(self.device.name.strip())}")
            _cl_ver  = self.device.get_info(cl.device_info.VERSION)
            _drv_ver = self.device.get_info(cl.device_info.DRIVER_VERSION)
            log_debug(f"       WGS={self.local_work_size}, VRAM~{vram_gb:.1f}GB, "
                      f"CU={self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)}, "
                      f"OpenCL={_cl_ver.strip()}, driver={_drv_ver.strip()}")
            return True
        except Exception as e:
            log_error(f"GPU init failed: {e}"); return False

    def compile_kernel(self):
        if self.program is not None:
            log_debug("compile_kernel: already compiled — skipping")
            return self.program
        try:
            src = GPU_KERNEL_TEMPLATE.format(
                BLOOM_FILTER_SIZE          = self.params['BLOOM_FILTER_SIZE'],
                MAX_SAFE_RESULTS_PER_BATCH = self.params['MAX_SAFE_RESULTS_PER_BATCH'],
                MAX_CHAIN_DEPTH            = self.params['MAX_CHAIN_DEPTH'],
                MAX_CHAIN_STRING_LEN       = MAX_CHAIN_STRING_LEN,
                MAX_WORD_LEN               = MAX_WORD_LEN,
                MAX_RULE_LEN               = MAX_RULE_LEN,
                MAX_OUTPUT_LEN             = MAX_OUTPUT_LEN,
                BLOOM_HASH_FUNCTIONS       = BLOOM_HASH_FUNCTIONS,
            )
            context = self.context
            result: list = [None]
            def _build():
                try:
                    prog = cl.Program(context, src)
                    prog.build()
                    result[0] = prog
                except Exception as exc:
                    result[0] = exc
            t = threading.Thread(target=_build, daemon=True)
            t.start()
            print("[GPU]  Compiling OpenCL kernel ", end='', flush=True)
            while t.is_alive():
                t.join(timeout=10)
                if t.is_alive():
                    print('.', end='', flush=True)
            print(" done", flush=True)
            if isinstance(result[0], Exception):
                raise result[0]
            prog = result[0]
            self.program       = prog
            self.kernel_single = prog.find_single_rules_gpu
            self.kernel_chain  = prog.find_rule_chains_gpu
            self.kernel_bloom  = prog.build_bloom_filter_gpu
            log_debug("OpenCL kernel compiled successfully")
            return self.program
        except Exception as e:
            log_error(f"Kernel compile failed: {e}"); return None

    def _reset_gpu(self, error: Exception) -> bool:
        log_warn(f"[GPU] Fatal kernel error: {error}  — attempting full context reset")
        _RELEASE_TIMEOUT = 5.0
        def _timed_release(obj):
            def _do():
                try: obj.release()
                except Exception: pass
            t = threading.Thread(target=_do, daemon=True)
            t.start()
            t.join(timeout=_RELEASE_TIMEOUT)
        for attr in ('bloom_buf', 'program', 'kernel_single', 'kernel_chain',
                     'kernel_bloom', 'queue', 'context'):
            obj = getattr(self, attr, None)
            if obj is not None:
                _timed_release(obj)
            setattr(self, attr, None)
        for attr in ('_rules_buf', '_rules_offsets_buf', '_rules_lengths_buf'):
            b = getattr(self, attr, None)
            if b is not None:
                _timed_release(b)
            setattr(self, attr, None)
        self._rules_buf_key = None
        if self.device is None:
            log_error("[GPU] No device reference — cannot recover context")
            return False
        try:
            self.context         = cl.Context([self.device])
            self.queue           = cl.CommandQueue(self.context)
            self.local_work_size = min(self.params.get('LOCAL_WORK_SIZE', 512), self.max_work_group_size)
            if not self.compile_kernel():
                log_error("[GPU] Context reset: kernel recompile failed")
                return False
            self._refresh_cached_limits()
            if self.bloom_np is not None:
                self.upload_bloom_filter(self.bloom_np)
            log_info("[GPU] Context reset successful — resuming")
            return True
        except Exception as exc:
            log_error(f"[GPU] Context reset failed: {exc}")
            return False

    _QUEUE_FINISH_TIMEOUT = 90

    def _safe_queue_finish(self) -> bool:
        if self.queue is None:
            return False
        result: list = [None]
        def _finish():
            try:
                self.queue.finish()
                result[0] = True
            except Exception as exc:
                result[0] = exc
        t = threading.Thread(target=_finish, daemon=True)
        t.start()
        t.join(timeout=self._QUEUE_FINISH_TIMEOUT)
        if t.is_alive():
            log_warn(f"[GPU] queue.finish() hung for {self._QUEUE_FINISH_TIMEOUT}s — forcing reset")
            self._reset_gpu(RuntimeError("queue.finish timeout"))
            return False
        if isinstance(result[0], Exception):
            log_warn(f"[GPU] queue.finish() raised: {result[0]}")
            self._reset_gpu(result[0])
            return False
        return True

    def generate_bloom_filter(self, target_words):
        bsz = self.params['BLOOM_FILTER_SIZE'] // 8
        bf  = np.zeros(bsz, dtype=np.uint8)
        log_debug(f"[BLOOM] CPU bloom build: {bsz//1024//1024}MB for {len(target_words):,} words")
        _t_bloom = time.time()
        for w in tqdm(target_words, desc=green("  Bloom filter"), unit="word", ncols=88,
                      leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
            wb = w.encode('latin-1')
            h1 = fnv1a_32(wb, FNV1A_SEED1); h2 = fnv1a_32(wb, FNV1A_SEED2)
            for i in range(BLOOM_HASH_FUNCTIONS):
                idx = (h1 + i*h2) % self.params['BLOOM_FILTER_SIZE']
                bf[idx//8] |= 1 << (idx%8)
        log_debug(f"[BLOOM] CPU bloom build done in {time.time()-_t_bloom:.1f}s")
        return bf

    def generate_bloom_filter_gpu(self, target_words: list) -> np.ndarray:
        """Build the Bloom filter on the GPU using OpenCL atomic_or operations.

        Each work-item handles one word, computing the same two FNV-1a hashes
        used by the existing bloom() check kernel and atomically ORing the
        corresponding bits into a flat int32 buffer.  The resulting uint8 view
        is byte-for-byte identical to what the CPU path produces, so the check
        kernel needs no changes.

        Falls back transparently to the CPU path on any error.
        """
        bsz_bits   = self.params['BLOOM_FILTER_SIZE']
        bsz_uint32 = bsz_bits // 32      # int32 words in the filter
        bsz_bytes  = bsz_bits // 8       # bytes in the filter
        nw         = len(target_words)

        if not self.context or not self.program or not self.kernel_bloom:
            log_debug("[BLOOM] GPU kernel not ready — falling back to CPU")
            return self.generate_bloom_filter(target_words)

        log_debug(f"[BLOOM] GPU bloom build: {bsz_bytes//1024//1024}MB, {nw:,} words, "
                  f"lws={self.local_work_size}")
        mf   = cl.mem_flags
        bufs = []
        try:
            wf, wo, wl = self._flatten(target_words)

            words_buf   = cl.Buffer(self.context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=wf)
            offsets_buf = cl.Buffer(self.context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=wo)
            lengths_buf = cl.Buffer(self.context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=wl)
            bufs.extend([words_buf, offsets_buf, lengths_buf])

            # Bloom filter as zeroed int32 array (atomic_or needs int* in OpenCL 1.x)
            bf_init = np.zeros(bsz_uint32, dtype=np.int32)
            bf_buf  = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bf_init)
            bufs.append(bf_buf)

            gs = ((nw + self.local_work_size - 1) // self.local_work_size) * self.local_work_size

            self.kernel_bloom.set_args(
                words_buf, offsets_buf, lengths_buf,
                np.int32(nw), bf_buf,
            )
            cl.enqueue_nd_range_kernel(
                self.queue, self.kernel_bloom, (gs,), (self.local_work_size,)
            )

            if not self._safe_queue_finish():
                raise RuntimeError("queue.finish() timed out during GPU bloom build")

            # Read back the int32 result and reinterpret as bytes.
            # On a little-endian host (x86 / ARM) the byte ordering is:
            #   bf_int32[k] byte[0..3]  ↔  bf_bytes[4k..4k+3]
            # which matches the bit layout expected by bloom() in the check kernel.
            bf_result = np.empty(bsz_uint32, dtype=np.int32)
            cl.enqueue_copy(self.queue, bf_result, bf_buf)
            self._safe_queue_finish()

            bf_bytes = bf_result.view(np.uint8).copy()   # .copy() ensures C-contiguous
            assert len(bf_bytes) == bsz_bytes, (
                f"Bloom byte length mismatch: {len(bf_bytes)} != {bsz_bytes}"
            )

            # Sanity-check: a GPU that silently ignores atomic_or (e.g. an
            # OpenCL 1.x driver that doesn't support global atomics even with
            # the extension pragma) produces an all-zeros filter.  Detect it
            # here and fall back rather than letting a broken filter propagate.
            n_set = int(np.count_nonzero(bf_result))
            if n_set == 0 and nw > 0:
                log_warn("[BLOOM] GPU bloom build returned all-zeros — "
                         "atomic_or may be unsupported on this device; "
                         "falling back to CPU")
                return self.generate_bloom_filter(target_words)

            log_debug(f"[BLOOM] GPU bloom build done — "
                      f"{n_set:,} / {bsz_uint32:,} int32 words non-zero "
                      f"({n_set * 3200 / bsz_bits:.2f}% bit fill)")
            return bf_bytes

        except Exception as exc:
            log_warn(f"[BLOOM] GPU bloom build failed ({exc}) — falling back to CPU")
            return self.generate_bloom_filter(target_words)

        finally:
            for b in bufs:
                try: b.release()
                except Exception: pass

    def upload_bloom_filter(self, bf):
        mf = cl.mem_flags
        if self.bloom_buf:
            try: self.bloom_buf.release()
            except Exception: pass
        self.bloom_np  = bf
        self.bloom_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bf)

    def _flatten(self, items):
        flat = b''.join(x.encode('latin-1') for x in items)
        offs = []; lens = []; off = 0
        for x in items:
            b = x.encode('latin-1')
            offs.append(off); lens.append(len(b)); off += len(b)
        return (np.frombuffer(flat, dtype=np.uint8),
                np.array(offs, dtype=np.int32),
                np.array(lens, dtype=np.int32))

    def prepare_batch_data(self, words, rules):
        wf, wo, wl = self._flatten(words)
        rf, ro, rl = self._flatten(rules)
        return dict(words_flat=wf, word_offsets=wo, word_lengths=wl,
                    rules_flat=rf, rule_offsets=ro, rule_lengths=rl,
                    num_words=len(words), num_rules=len(rules))

    def prepare_words_data(self, words):
        wf, wo, wl = self._flatten(words)
        return dict(words_flat=wf, word_offsets=wo, word_lengths=wl,
                    num_words=len(words), num_rules=len(self.gpu_rules))

    def _get_rules_buffers(self, mf):
        key = id(self.gpu_rules)
        if self._rules_buf_key != key:
            for attr in ('_rules_buf', '_rules_offsets_buf', '_rules_lengths_buf'):
                b = getattr(self, attr, None)
                if b is not None:
                    try: b.release()
                    except Exception: pass
            rf, ro, rl = self._flatten(self.gpu_rules)
            self._rules_buf         = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rf)
            self._rules_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ro)
            self._rules_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rl)
            self._rules_buf_key     = key
        return self._rules_buf, self._rules_offsets_buf, self._rules_lengths_buf

    def process_all_words_single_rule(self, base_words, rules, bloom_filter):
        self.upload_bloom_filter(bloom_filter)
        if not self.compile_kernel(): return Counter()
        self.gpu_rules  = HashcatRuleValidator.validate_rules_for_gpu(rules)
        self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}
        log_debug(f"STAGE 1: {len(base_words):,} words × {len(self.gpu_rules):,} rules")
        counter = Counter()
        bs      = self.params['WORDS_PER_BATCH']
        _s1_t0 = time.time(); _s1_words_done = 0; _s1_n_rules = len(self.gpu_rules)
        with tqdm(total=len(base_words), desc=green("  STAGE 1 "), unit="word", ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for i in range(0, len(base_words), bs):
                _kb.check_pause()
                if _kb.quit_requested:
                    log_warn("[S1]    Early exit — stopping STAGE 1")
                    break
                batch = base_words[i:i+bs]
                if batch:
                    found = self._run_single_kernel(self.prepare_words_data(batch))
                    if found:
                        self._consecutive_errors = 0
                        counter.update(found)
                    elif self.queue is None:
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                            log_warn(f"[GPU] {self._consecutive_errors} consecutive failures — aborting STAGE 1")
                            break
                    _s1_words_done += len(batch)
                    _s1_el = time.time() - _s1_t0
                    _s1_combos = _s1_words_done * _s1_n_rules
                    _s1_spd = _fmt_speed(_s1_combos / _s1_el if _s1_el > 0 else 0)
                    pbar.set_postfix({"rules": cyan(str(len(counter))), "spd": green(_s1_spd)}, refresh=False)
                pbar.update(len(batch))
        gc.collect()
        log_info(f"[S1]    {bold(green(str(len(counter))))} unique rules passed bloom filter")
        return counter

    def _run_single_kernel(self, bd):
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f | mf.COPY_HOST_PTR, hostbuf=arr)
                bufs.append(b); return b
            bb=B(bd['words_flat']); bbo=B(bd['word_offsets']); bbl=B(bd['word_lengths'])
            rb, rbo, rbl = self._get_rules_buffers(mf)
            outs = self.safe_output_buffer_size(bd['num_words'], bd['num_rules'])
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4);                         bufs.append(fc)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))
            tot = bd['num_words'] * bd['num_rules']
            gs  = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_single.set_args(bb, bbo, bbl, rb, rbo, rbl, self.bloom_buf,
                                        np.int32(bd['num_words']), np.int32(bd['num_rules']),
                                        fo, fc)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_single, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish():
                return []
            cnt = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs); out = []
            if cnt[0] > 0:
                log_debug(f"[S1-K] {int(cnt[0])} hit(s)  (cap={outs}  → {n} decoded)")
            if n > 0:
                data = np.zeros(n*MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, data, fo)
                for i in range(n):
                    r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]
                              ).split(b'\0')[0].decode('latin-1', errors='ignore')
                    if r: out.append(r)
            return out
        except Exception as e:
            log_warn(f"Single-rule kernel error: {e}")
            self._reset_gpu(e)
            return []
        finally:
            for b in bufs:
                try: b.release()
                except: pass

    def _gen_random_chains(self, depth, count, valid, hot, existing, new_set):
        gen = set(); max_att = count * MAX_ATTEMPTS_MULTIPLIER
        hot_budget = int(count * HOT_RULE_RATIO) if hot else 0
        for budget, use_hot in [(hot_budget, True), (count-hot_budget, False)]:
            att = 0
            while len(gen) < budget and att < max_att:
                att += 1
                if use_hot and hot:
                    hp    = random.randint(0, depth-1)
                    parts = [random.choice(hot) if i==hp else random.choice(valid)
                             for i in range(depth)]
                else:
                    parts = [random.choice(valid) for _ in range(depth)]
                k = ' '.join(parts)
                if k not in existing and k not in gen and k not in new_set:
                    gen.add(k)
        return gen

    def build_numeric_seed_families(self, max_depth: int = 4) -> dict:
        digits = '0123456789'
        sbd: Dict[int, set] = defaultdict(set)
        # A
        for depth in range(1, min(4, max_depth) + 1):
            for combo in itertools.product(digits, repeat=depth):
                sbd[depth].add(' '.join(f'^{ch}' for ch in reversed(combo)))
        # B
        for depth in range(1, min(4, max_depth) + 1):
            for combo in itertools.product(digits, repeat=depth):
                sbd[depth].add(' '.join(f'${ch}' for ch in combo))
        # C
        for depth in range(1, min(4, max_depth) + 1):
            for ops in itertools.product(['^', '$'], repeat=depth):
                for digs in itertools.product(digits, repeat=depth):
                    sbd[depth].add(' '.join(f'{o}{d}' for o, d in zip(ops, digs)))
        # D
        transform_ops = ['l','u','c','C','t','r','d','f','E','k','K','{','}','[',']']
        t_digit_ops = [f'^{d}' for d in digits] + [f'${d}' for d in digits] + ['[', ']']
        for depth in range(2, min(4, max_depth) + 1):
            for t_op in transform_ops:
                for ops in itertools.product(t_digit_ops, repeat=depth - 1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed):
                        sbd[depth].add(seed)
        # E
        _days   = [f"{d:02d}" for d in range(1, 32)]
        _months = [f"{m:02d}" for m in range(1, 13)]
        _years2 = ([f"{y:02d}" for y in range(60, 100)] + [f"{y:02d}" for y in range(0, 31)])
        _years4 = [str(y) for y in range(1960, 2031)]
        _date4 = set(); _date6 = set(); _date8 = set()
        for _d in _days:
            for _m in _months:
                _date4.add(_d + _m)
                _date4.add(_m + _d)
        for _y in _years4:
            _date4.add(_y)
        for _d in _days:
            for _m in _months:
                for _y in _years2:
                    _date6.add(_d + _m + _y)
                    _date6.add(_m + _d + _y)
        for _d in _days:
            for _m in _months:
                for _y in _years4:
                    _date8.add(_d + _m + _y)
                    _date8.add(_m + _d + _y)
        for _ds_set, _base_depth in ((_date4, 4), (_date6, 6), (_date8, 8)):
            if _base_depth > max_depth: continue
            for _ds in _ds_set:
                sbd[_base_depth].add(' '.join(f'${c}' for c in _ds))
                sbd[_base_depth].add(' '.join(f'^{c}' for c in reversed(_ds)))
        if max_depth >= 5:
            for _ds in _date4:
                _app = ' '.join(f'${c}' for c in _ds)
                _pre = ' '.join(f'^{c}' for c in reversed(_ds))
                for t_op in transform_ops:
                    for _chain in (f"{t_op} {_app}", f"{t_op} {_pre}"):
                        if HashcatRuleValidator.validate_rule_for_gpu(_chain):
                            sbd[5].add(_chain)
        _bracket_ops = ['[', ']']
        for _bds, _bdepth, _brange in ((_date4, 4, range(2,5)), (_date6, 6, range(1,3)), (_date8, 8, range(1,2))):
            for _num_b in _brange:
                _new_depth = _bdepth + _num_b
                if _new_depth > max_depth: continue
                for _brackets in itertools.product(_bracket_ops, repeat=_num_b):
                    _bpfx = ' '.join(_brackets)
                    for _ds in _bds:
                        _app = ' '.join(f'${c}' for c in _ds)
                        _pre = ' '.join(f'^{c}' for c in reversed(_ds))
                        for _chain in (f"{_bpfx} {_app}", f"{_bpfx} {_pre}"):
                            if HashcatRuleValidator.validate_rule_for_gpu(_chain):
                                sbd[_new_depth].add(_chain)
        # F
        for depth in range(1, min(3, max_depth) + 1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                chain = ' '.join(f'${ch}' for ch in combo)
                if HashcatRuleValidator.validate_rule_for_gpu(chain):
                    sbd[depth].add(chain)
        # G
        for depth in range(1, min(3, max_depth) + 1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                chain = ' '.join(f'^{ch}' for ch in reversed(combo))
                if HashcatRuleValidator.validate_rule_for_gpu(chain):
                    sbd[depth].add(chain)
        # H
        sp_ops_top = [f'${ch}' for ch in SPECIAL_CHARS_TOP] + [f'^{ch}' for ch in SPECIAL_CHARS_TOP]
        for depth in range(2, min(3, max_depth) + 1):
            for t_op in transform_ops:
                for ops in itertools.product(sp_ops_top, repeat=depth - 1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed):
                        sbd[depth].add(seed)
        # I
        for depth in range(2, min(4, max_depth) + 1):
            n_digits = depth - 1
            for digit_combo in itertools.product(digits, repeat=n_digits):
                for sp in SPECIAL_CHARS_CORE:
                    app = ' '.join(f'${d}' for d in digit_combo) + f' ${sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(app):
                        sbd[depth].add(app)
                    pre = f'^{sp} ' + ' '.join(f'^{d}' for d in reversed(digit_combo))
                    if HashcatRuleValidator.validate_rule_for_gpu(pre):
                        sbd[depth].add(pre)
        # J
        for op in LEET_OPS:
            if HashcatRuleValidator.validate_rule_for_gpu(op):
                sbd[1].add(op)
        if max_depth >= 2:
            leet_followup = [f'${d}' for d in digits] + [f'^{d}' for d in digits] + \
                            [f'${ch}' for ch in SPECIAL_CHARS_CORE] + [f'^{ch}' for ch in SPECIAL_CHARS_CORE]
            for leet_op in LEET_OPS:
                for follow in leet_followup:
                    chain = f"{leet_op} {follow}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain):
                        sbd[2].add(chain)
            for i_l, l1 in enumerate(LEET_OPS):
                for l2 in LEET_OPS:
                    if l1 != l2:
                        chain = f"{l1} {l2}"
                        if HashcatRuleValidator.validate_rule_for_gpu(chain):
                            sbd[2].add(chain)
        # K
        if max_depth >= 2:
            for t1 in transform_ops:
                for t2 in transform_ops:
                    chain = f"{t1} {t2}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain):
                        sbd[2].add(chain)
        # L
        for depth in range(2, min(3, max_depth) + 1):
            n_digits = depth - 1
            for sp in SPECIAL_CHARS_CORE:
                for digit_combo in itertools.product(digits, repeat=n_digits):
                    app = f'${sp} ' + ' '.join(f'${d}' for d in digit_combo)
                    if HashcatRuleValidator.validate_rule_for_gpu(app):
                        sbd[depth].add(app)
                    pre = ' '.join(f'^{d}' for d in digit_combo) + f' ^{sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(pre):
                        sbd[depth].add(pre)
        # M
        if max_depth >= 2:
            for leet_op in LEET_OPS:
                for t_op in transform_ops:
                    chain_lt = f"{leet_op} {t_op}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain_lt):
                        sbd[2].add(chain_lt)
                    chain_tl = f"{t_op} {leet_op}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain_tl):
                        sbd[2].add(chain_tl)
        log_debug(f"Seed families built max_depth={max_depth} total={sum(len(v) for v in sbd.values()):,}")
        return dict(sbd)

    def run_seed_extraction_pass(self, base_words: list, sbd: dict,
                                  bloom_filter, phase1_rules: list) -> Counter:
        if self.bloom_buf is None:
            self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.gpu_rules  = HashcatRuleValidator.validate_rules_for_gpu(phase1_rules)
            self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}
        multi_seeds: List[str] = []
        for depth, chains in sorted(sbd.items()):
            if depth >= 2:
                multi_seeds.extend(chains)
        if not multi_seeds:
            log_info("[SEED]    No multi-depth seeds to test")
            return Counter()
        total = sum(len(v) for d, v in sbd.items() if d >= 2)
        log_info(f"[SEED]    Numeric seed pass: {total:,} chains across {sum(1 for d in sbd if d >= 2)} depth(s)")
        counter = Counter()
        cbs = self.params['CHAINS_PER_BATCH']
        wsb = self.params['WORD_SUB_BATCH']
        n_batches = (len(multi_seeds) + cbs - 1) // cbs
        _sp_t0 = time.time(); _sp_accum_combos = 0
        with tqdm(total=n_batches, desc=green("  "), unit="batch", ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for ci in range(0, len(multi_seeds), cbs):
                _kb.check_pause()
                if _kb.quit_requested:
                    log_warn("[SEED]  Early exit — stopping seed pass")
                    break
                cb = multi_seeds[ci:ci + cbs]
                for wi in range(0, len(base_words), wsb):
                    wb = base_words[wi:wi + wsb]
                    if wb:
                        found = self._run_chain_kernel(wb, cb)
                        if found:
                            self._consecutive_errors = 0
                            counter.update(found)
                        elif self.queue is None:
                            self._consecutive_errors += 1
                    if not self._safe_queue_finish():
                        self._consecutive_errors += 1
                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    log_warn(f"[GPU] {self._consecutive_errors} consecutive failures — aborting seed pass")
                    break
                _sp_accum_combos += len(cb) * len(base_words)
                _sp_el = time.time() - _sp_t0
                _sp_spd = _fmt_speed(_sp_accum_combos / _sp_el if _sp_el > 0 else 0)
                pbar.update(1)
                pbar.set_postfix({"hits": cyan(str(len(counter))), "spd": green(_sp_spd)}, refresh=False)
        log_info(f"[SEED]    {bold(green(str(len(counter))))} unique seed chains passed bloom filter")
        return counter

    def generate_informed_chains(self, rules, single_found, max_depth,
                                   seed_chains=None, prebuilt_sbd=None):
        max_depth = min(max_depth, MAX_HASHCAT_CHAIN)
        valid   = [r for r in rules if HashcatRuleValidator.validate_rule_for_gpu(r)]
        if not valid: return []
        found_s = set(single_found.keys()) if single_found else set()
        hot     = [r for r in valid if r in found_s]
        log_debug(f"[S2]    generate_informed_chains: {len(valid)} valid rules, {len(hot)} hot rules from STAGE 1")
        chains  = set(valid)
        n_user_direct = 0
        if seed_chains:
            for sc in seed_chains:
                if sc.count(' ') >= 1:
                    chains.add(sc)
                    n_user_direct += 1
            if n_user_direct:
                log_debug(f"User seed chains injected as STAGE 2 candidates: {n_user_direct}")
        for depth in range(2, max_depth + 1):
            budget = self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 0)
            if budget <= 0: continue
            budget = min(budget, len(valid) ** depth)
            new    = self._gen_random_chains(depth, budget, valid, hot, chains, set())
            chains.update(new)
        log_debug(f"Total STAGE 2 candidates: {len(chains):,}")
        return list(chains)

    def process_all_words_chain_rules(self, base_words, rules, max_depth,
                                      bloom_filter, single_counter, seed_chains=None,
                                      prebuilt_sbd=None):
        if self.bloom_buf is None: self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}
        chains = self.generate_informed_chains(rules, single_counter, max_depth, seed_chains,
                                                    prebuilt_sbd=prebuilt_sbd)
        if not chains: return Counter()
        log_debug(f"STAGE 2: {len(chains):,} chains × {len(base_words):,} words")
        counter = Counter()
        cbs     = self.params['CHAINS_PER_BATCH']
        wsb     = self.params['WORD_SUB_BATCH']
        n_batches = (len(chains)+cbs-1)//cbs
        _s2_t0 = time.time(); _s2_accum_combos = 0
        with tqdm(total=n_batches, desc=green("  STAGE 2 "), unit="batch", ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for ci in range(0, len(chains), cbs):
                _kb.check_pause()
                if _kb.quit_requested:
                    log_warn("[S2]    Early exit — stopping STAGE 2")
                    break
                cb = chains[ci:ci+cbs]
                for wi in range(0, len(base_words), wsb):
                    wb = base_words[wi:wi+wsb]
                    if wb:
                        found = self._run_chain_kernel(wb, cb)
                        if found:
                            self._consecutive_errors = 0
                            counter.update(found)
                        elif self.queue is None:
                            self._consecutive_errors += 1
                    if not self._safe_queue_finish():
                        self._consecutive_errors += 1
                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    log_warn(f"[GPU] {self._consecutive_errors} consecutive failures — aborting STAGE 2")
                    break
                _s2_accum_combos += len(cb) * len(base_words)
                _s2_el = time.time() - _s2_t0
                _s2_spd = _fmt_speed(_s2_accum_combos / _s2_el if _s2_el > 0 else 0)
                pbar.update(1)
                pbar.set_postfix({"rules": cyan(str(len(counter))), "spd": green(_s2_spd)}, refresh=False)
        gc.collect()
        log_info(f"[S2]    {bold(green(str(len(counter))))} unique chain rules passed bloom filter")
        return counter

    def _run_chain_kernel(self, words, chains):
        seqs = []; depths = []
        for chain in chains:
            parts = chain.split()
            depths.append(len(parts))
            idxs = [self.rule_index.get(r, -1) for r in parts]
            while len(idxs) < self.params['MAX_CHAIN_DEPTH']:
                idxs.append(-1)
            seqs.extend(idxs)
        bd = self.prepare_words_data(words)
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f | mf.COPY_HOST_PTR, hostbuf=arr)
                bufs.append(b); return b
            bb=B(bd['words_flat']); bbo=B(bd['word_offsets']); bbl=B(bd['word_lengths'])
            rb, rbo, rbl = self._get_rules_buffers(mf)
            csb=B(np.array(seqs,   dtype=np.int32))
            cdb=B(np.array(depths, dtype=np.int32))
            outs = self.safe_output_buffer_size(len(words), len(chains))
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4);                         bufs.append(fc)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))
            tot = len(words)*len(chains)
            gs  = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_chain.set_args(bb, bbo, bbl, rb, rbo, rbl, csb, cdb, self.bloom_buf,
                                       np.int32(len(words)), np.int32(len(chains)),
                                       np.int32(self.params['MAX_CHAIN_DEPTH']), fo, fc)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_chain, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish():
                return []
            cnt = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs); out = []
            if cnt[0] > 0:
                log_debug(f"[S2-K] {int(cnt[0])} hit(s)  (cap={outs}  → {n} decoded)")
            if n > 0:
                data = np.zeros(n*MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, data, fo)
                for i in range(n):
                    r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]
                              ).split(b'\0')[0].decode('latin-1', errors='ignore')
                    if r: out.append(r)
            return out
        except Exception as e:
            log_warn(f"Chain kernel error: {e}")
            self._reset_gpu(e)
            return []
        finally:
            for b in bufs:
                try: b.release()
                except: pass

# ====================================================================
# --- STAGE 3 — GENETIC ALGORITHM RULE EVOLVER (unchanged) ---
# ====================================================================
class GeneticRuleEvolver:
    def __init__(
        self,
        gpu_engine,
        base_words: list,
        rule_pool:  list,
        max_depth:  int,
        pop_size:       int   = 200,
        elite_frac:     float = 0.15,
        tournament_k:   int   = 4,
        crossover_p:    float = 0.80,
        mut_replace_p:  float = 0.60,
        mut_insert_p:   float = 0.20,
        mut_delete_p:   float = 0.20,
        seed_hits:   Optional[Counter] = None,
        known_rules: Optional[set]    = None,
    ):
        self.gpu_engine   = gpu_engine
        self.base_words   = base_words
        self.rule_pool    = rule_pool
        self.max_depth    = max(2, max_depth)
        self.pop_size     = pop_size
        self.elite_frac   = elite_frac
        self.tournament_k = tournament_k
        self.crossover_p  = crossover_p
        _total = mut_replace_p + mut_insert_p + mut_delete_p
        if _total <= 0:
            _total = 1.0
        self._mut_weights = [mut_replace_p / _total, mut_insert_p / _total, mut_delete_p / _total]
        self.seed_hits = seed_hits or Counter()
        self.seed_chains_sorted: List[str] = [r for r, _ in sorted(self.seed_hits.items(), key=lambda kv: -kv[1])] if self.seed_hits else []
        self.known_rules: set = known_rules if known_rules is not None else set()
        self._sig_cache:   Dict[str, tuple] = {}
        self._sig_to_best: Dict[tuple, Tuple[str, int]] = {}

    def _random_chain(self, depth: int = 0) -> list:
        if depth <= 0:
            depth = random.randint(2, self.max_depth)
        return [random.choice(self.rule_pool) for _ in range(depth)]

    def _clamp(self, tokens: list) -> list:
        lo, hi = 2, self.max_depth
        if len(tokens) < lo:
            tokens = tokens + self._random_chain(lo - len(tokens))
        return tokens[:hi]

    def _get_sig(self, chain_str: str) -> tuple:
        if chain_str not in self._sig_cache:
            self._sig_cache[chain_str] = compute_rule_signature(chain_str, BUILTIN_PROBES)
        return self._sig_cache[chain_str]

    def _update_sig_registry(self, raw_hit_map: dict) -> int:
        new_sigs = 0
        for chain_str, raw_hits in raw_hit_map.items():
            if raw_hits <= 0:
                continue
            sig = self._get_sig(chain_str)
            if sig == ('__UNSUPPORTED__',):
                continue
            existing = self._sig_to_best.get(sig)
            if existing is None:
                self._sig_to_best[sig] = (chain_str, raw_hits)
                new_sigs += 1
            elif raw_hits > existing[1]:
                self._sig_to_best[sig] = (chain_str, raw_hits)
        for _sig, (best_chain, _cnt) in self._sig_to_best.items():
            self.known_rules.add(best_chain)
        return new_sigs

    def _sig_is_covered(self, chain_str: str) -> bool:
        if not self._sig_to_best:
            return False
        sig = self._get_sig(chain_str)
        if sig == ('__UNSUPPORTED__',):
            return False
        return sig in self._sig_to_best

    def _mutate_adaptive(self, tokens: list) -> list:
        tokens = self._mutate(tokens)
        if self._sig_to_best:
            for _attempt in range(2):
                if not self._sig_is_covered(' '.join(tokens)):
                    break
                tokens = self._mutate(tokens)
        return tokens

    def initial_population(self, hot_rules: list) -> list:
        hot = hot_rules[:min(len(hot_rules), 50)]
        pop_set: set = set()
        n_hot    = int(self.pop_size * 0.30)
        n_seeded = int(self.pop_size * 0.30)
        max_tries = n_hot * 20
        tries = 0
        while len(pop_set) < n_hot and tries < max_tries:
            tries += 1
            if len(hot) >= 2:
                a, b = random.sample(hot, 2)
            elif len(hot) == 1:
                a = hot[0]; b = random.choice(self.rule_pool)
            else:
                break
            pop_set.add((a, b))
        max_tries = n_seeded * 20
        tries = 0
        while len(pop_set) < n_hot + n_seeded and tries < max_tries:
            tries += 1
            if self.max_depth >= 3:
                depth = random.randint(3, self.max_depth) if random.random() < 0.70 else 2
            else:
                depth = random.randint(2, self.max_depth)
            if hot:
                tokens = [random.choice(hot)] + [random.choice(self.rule_pool) for _ in range(depth - 1)]
                random.shuffle(tokens)
            else:
                tokens = self._random_chain(depth)
            pop_set.add(tuple(tokens))
        n_fill = int(self.pop_size * 0.40)
        fill_set: set = set()
        if self.seed_chains_sorted:
            novel_seeds = [s for s in self.seed_chains_sorted if s not in self.known_rules]
            known_seeds = [s for s in self.seed_chains_sorted if s in self.known_rules]
            candidate_pool = novel_seeds if len(novel_seeds) >= n_fill // 2 else (novel_seeds + known_seeds)
            if candidate_pool:
                selected = random.sample(candidate_pool[:max(n_fill * 3, 100)], k=min(n_fill, len(candidate_pool)))
                for sc_str in selected:
                    tokens = sc_str.split()
                    if 2 <= len(tokens) <= self.max_depth:
                        fill_set.add(tuple(tokens))
            max_fill_tries = n_fill * 20
            fill_tries = 0
            while len(fill_set) < n_fill and fill_tries < max_fill_tries:
                fill_tries += 1
                d = random.randint(3, self.max_depth) if self.max_depth >= 3 and random.random() < 0.60 else random.randint(2, self.max_depth)
                fill_set.add(tuple(self._random_chain(d)))
        else:
            max_tries = n_fill * 20
            tries = 0
            while len(fill_set) < n_fill and tries < max_tries:
                tries += 1
                d = random.randint(3, self.max_depth) if self.max_depth >= 3 and random.random() < 0.60 else random.randint(2, self.max_depth)
                fill_set.add(tuple(self._random_chain(d)))
        for ind in fill_set:
            pop_set.add(ind)
        result = [list(ind) for ind in pop_set]
        while len(result) < self.pop_size:
            d = random.randint(3, self.max_depth) if self.max_depth >= 3 and random.random() < 0.50 else random.randint(2, self.max_depth)
            result.append(self._random_chain(d))
        return result[:self.pop_size]

    def evaluate_population(self, population: list) -> dict:
        chain_strs = [' '.join(tokens) for tokens in population]
        valid_chains = [c for c in chain_strs if HashcatRuleValidator.validate_rule_for_gpu(c)]
        raw_map: dict = {c: 0 for c in chain_strs}
        if not valid_chains:
            return raw_map
        wsb = self.gpu_engine.params.get('WORD_SUB_BATCH', 20_000)
        cbs = self.gpu_engine.params.get('CHAINS_PER_BATCH', 2_000)
        batch_hits: Counter = Counter()
        for ci in range(0, len(valid_chains), cbs):
            cb = valid_chains[ci:ci + cbs]
            for wi in range(0, len(self.base_words), wsb):
                wb = self.base_words[wi:wi + wsb]
                if wb:
                    found = self.gpu_engine._run_chain_kernel(wb, cb)
                    if found:
                        batch_hits.update(found)
            self.gpu_engine._safe_queue_finish()
        raw_map.update(batch_hits)
        return raw_map

    def _tournament_select(self, fitness_list: list) -> list:
        k = min(self.tournament_k, len(fitness_list))
        contenders = random.sample(fitness_list, k)
        winner, _ = max(contenders, key=lambda x: x[1])
        return list(winner)

    def _crossover(self, p1: list, p2: list) -> tuple:
        if len(p1) < 2 or len(p2) < 2 or random.random() > self.crossover_p:
            return list(p1), list(p2)
        cut1 = random.randint(1, len(p1) - 1)
        cut2 = random.randint(1, len(p2) - 1)
        child1 = self._clamp(p1[:cut1] + p2[cut2:])
        child2 = self._clamp(p2[:cut2] + p1[cut1:])
        return child1, child2

    def _mutate(self, tokens: list) -> list:
        tokens = list(tokens)
        op = random.choices(['replace', 'insert', 'delete'], weights=self._mut_weights)[0]
        if op == 'replace':
            idx = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)
        elif op == 'insert' and len(tokens) < self.max_depth:
            idx = random.randint(0, len(tokens))
            tokens.insert(idx, random.choice(self.rule_pool))
        elif op == 'delete' and len(tokens) > 2:
            idx = random.randrange(len(tokens))
            tokens.pop(idx)
        else:
            idx = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)
        return tokens

    def evolve(self, hot_rules: list, generations: int, time_budget: float) -> Counter:
        if not self.rule_pool:
            log_warn("[S3]    Rule pool is empty — skipping STAGE 3.")
            return Counter()
        if time_budget <= 0:
            log_warn("[S3]    No time budget remaining — skipping STAGE 3.")
            return Counter()
        t_start = time.time()
        all_new: Counter = Counter()
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        STAGNATION_THRESHOLD = 5
        stagnation_counter = 0
        best_ever_score = 0
        log_info(f"[S3]    pop={self.pop_size}  max_gen={generations}  elite={self.elite_frac:.0%}  budget={time_budget:.0f}s  pool={len(self.rule_pool):,} rules  known={len(self.known_rules):,}")
        pop = self.initial_population(hot_rules)
        last_gen = 0
        with tqdm(total=generations, desc=green("  STAGE 3 "), unit="gen", ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for gen in range(generations):
                last_gen = gen
                _kb.check_pause()
                if _kb.quit_requested:
                    log_debug(f"[S3]    Early exit at generation {gen} — stopping GA")
                    break
                if time.time() - t_start >= time_budget:
                    log_debug(f"[S3]    Time budget exhausted at generation {gen}.")
                    break
                raw_map = self.evaluate_population(pop)
                new_sigs = self._update_sig_registry(raw_map)
                n_novel_this_gen = 0
                for chain_str, raw_hits in raw_map.items():
                    if raw_hits > 0 and HashcatRuleValidator.validate_rule_for_gpu(chain_str):
                        if raw_hits > all_new[chain_str]:
                            all_new[chain_str] = raw_hits
                        if all_new[chain_str] == raw_hits:
                            n_novel_this_gen += 1
                fitness_list = sorted(
                    [(tuple(ind), raw_map.get(' '.join(ind), 0) * (2 if ' '.join(ind) not in self.known_rules else 1)) for ind in pop],
                    key=lambda x: -x[1],
                )
                best_score = fitness_list[0][1] if fitness_list else 0
                if best_score > best_ever_score:
                    best_ever_score = best_score
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                if stagnation_counter >= STAGNATION_THRESHOLD:
                    stagnation_counter = 0
                    n_refresh = max(1, int(self.pop_size * 0.30))
                    depth_bias = self.max_depth >= 3
                    refresh_chains = []
                    rt = 0
                    refresh_set = {fitness_list[0][0]} if fitness_list else set()
                    while len(refresh_chains) < n_refresh and rt < n_refresh * 20:
                        rt += 1
                        d = random.randint(3, self.max_depth) if depth_bias and random.random() < 0.60 else random.randint(2, self.max_depth)
                        ind = tuple(self._random_chain(d))
                        if ind not in refresh_set:
                            refresh_chains.append(list(ind))
                            refresh_set.add(ind)
                    keep_top = [list(ind) for ind, _ in fitness_list[:self.pop_size - n_refresh]]
                    pop = keep_top + refresh_chains
                    pop = pop[:self.pop_size]
                    pbar.update(1)
                    pbar.set_postfix({"best": cyan(str(best_score)), "new": cyan(str(len(all_new))), "sigs": cyan(str(len(self._sig_to_best))), "stag": yellow("REFRESH")}, refresh=False)
                    continue
                elites = [list(ind) for ind, _ in fitness_list[:n_elite]]
                next_pop = list(elites)
                next_set = {tuple(e) for e in elites}
                depth_bias = self.max_depth >= 3
                max_breed_attempts = (self.pop_size - len(next_pop)) * 8
                breed_attempts = 0
                n_sig_replaced = 0
                while len(next_pop) < self.pop_size and breed_attempts < max_breed_attempts:
                    breed_attempts += 1
                    p1 = self._tournament_select(fitness_list)
                    p2 = self._tournament_select(fitness_list)
                    child1, child2 = self._crossover(p1, p2)
                    child1 = self._mutate_adaptive(child1)
                    child2 = self._mutate_adaptive(child2)
                    for child in (child1, child2):
                        if len(next_pop) >= self.pop_size:
                            break
                        key = tuple(child)
                        if key in next_set:
                            continue
                        if self._sig_is_covered(' '.join(child)):
                            n_sig_replaced += 1
                            d = random.randint(3, self.max_depth) if depth_bias and random.random() < 0.70 else random.randint(2, self.max_depth)
                            child = self._random_chain(d)
                            key = tuple(child)
                            if key in next_set:
                                continue
                        next_pop.append(child)
                        next_set.add(key)
                fill_attempts = 0
                while len(next_pop) < self.pop_size and fill_attempts < self.pop_size * 4:
                    fill_attempts += 1
                    d = random.randint(3, self.max_depth) if depth_bias and random.random() < 0.50 else random.randint(2, self.max_depth)
                    ind = tuple(self._random_chain(d))
                    if ind not in next_set:
                        next_pop.append(list(ind))
                        next_set.add(ind)
                pop = next_pop[:self.pop_size]
                pbar.update(1)
                pbar.set_postfix({"best": cyan(str(best_score)), "new": cyan(str(len(all_new))), "sigs": cyan(str(len(self._sig_to_best)))}, refresh=False)
        elapsed = time.time() - t_start
        n_chains = len(all_new)
        n_sig_classes = len(self._sig_to_best)
        log_info(f"[S3]    Evolution complete — {bold(green(str(n_chains)))} unique chains passed bloom filter  ({bold(cyan(str(n_sig_classes)))} distinct functional signatures)  ({elapsed:.1f}s, {last_gen + 1} generation(s))")
        return all_new

# --------------------------------------------------------------------
# GPU Extractor
# --------------------------------------------------------------------
class GPUExtractor:
    def __init__(self, base_count, target_count, max_depth, device_spec=None,
                 target_hours=0.5, max_chains=None, seed_rules_file=None, bloom_mb=None,
                 builtin_seeds=True,
                 genetic=False, genetic_generations=50,
                 genetic_pop=200, genetic_elite=0.15,
                 token_strip=False, token_strip_min_stem=4,
                 token_strip_max_prefix=4, token_strip_max_suffix=4,
                 token_strip_min_leet_amb=3,
                 token_strip_workers=0, token_strip_chunk_size=0):
        self.base_count               = base_count
        self.target_count             = target_count
        self.max_depth                = max_depth
        self.device_spec              = device_spec
        self.max_chains               = max_chains
        self.seed_rules_file          = seed_rules_file
        self.bloom_mb                 = bloom_mb
        self.builtin_seeds            = builtin_seeds
        self.genetic                  = genetic
        self.genetic_generations      = genetic_generations
        self.genetic_pop              = genetic_pop
        self.genetic_elite            = genetic_elite
        self.token_strip              = token_strip
        self.token_strip_min_stem     = token_strip_min_stem
        self.token_strip_max_prefix   = token_strip_max_prefix
        self.token_strip_max_suffix   = token_strip_max_suffix
        self.token_strip_min_leet_amb = token_strip_min_leet_amb
        self.token_strip_workers      = token_strip_workers
        self.token_strip_chunk_size   = token_strip_chunk_size
        self.params                   = calculate_dynamic_parameters(base_count, target_count, None, target_hours, bloom_mb_override=bloom_mb)
        self.params['MAX_CHAIN_DEPTH'] = max_depth
        self.rules_gen                = GPUCompatibleRulesGenerator()
        self.gpu_engine               = GPUEngine(self.params)
        self.validator                = HashcatRuleValidator()

    def load_seed_rules(self):
        if not self.seed_rules_file: return []
        seeds = []
        rejected = 0
        try:
            log_debug(f"[SEED] Loading seed rules from: {self.seed_rules_file}")
            with open(self.seed_rules_file, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip('\n\r')
                    if line and not line.startswith('#'):
                        if self.validator.validate_rule_for_gpu(line):
                            seeds.append(line)
                        else:
                            rejected += 1
            log_info(f"[SEED] Loaded {bold(str(len(seeds)))} seed rules")
            if rejected:
                log_debug(f"[SEED] Rejected {rejected} invalid/GPU-incompatible seed rule(s)")
        except Exception as e:
            log_warn(f"Seed rules load failed: {e}")
        return seeds

    def extract_rules(self, base_words, target_words, **depth_overrides):
        all_counts = Counter()
        rules      = self.rules_gen.generate_gpu_compatible_rules()
        log_debug(f"[RULES] {len(rules):,} GPU-compatible atomic rules generated")
        ts_rules_singles: List[str] = []
        ts_rules_chains:  List[str] = []
        ts_sbd: Dict[int, set]      = defaultdict(set)
        ts_extra_singles: List[str] = []
        builtin_set = set(rules)
        all_seeds   = self.load_seed_rules()
        log_debug(f"[RULES] Seed rules after load: {len(all_seeds)} (builtin_seeds={self.builtin_seeds}, token_strip={self.token_strip})")

        if self.token_strip:
            log_section("STAGE 0 — Token-Strip Rule Extraction (Core + Insert)")
            base_set_for_ts = set(base_words)
            log_info(f"[S0]    {len(target_words):,} target words  base {len(base_set_for_ts):,}  min-stem={self.token_strip_min_stem}  prefix={self.token_strip_max_prefix}  suffix={self.token_strip_max_suffix}  leet-amb={self.token_strip_min_leet_amb}")
            ts_all = extract_token_strip_rules(
                target_words, base_set_for_ts,
                max_depth=self.max_depth,
                min_stem_len=self.token_strip_min_stem,
                max_prefix_len=self.token_strip_max_prefix,
                max_suffix_len=self.token_strip_max_suffix,
                max_leet_ambiguity=self.token_strip_min_leet_amb,
                workers=self.token_strip_workers,
                chunk_size=self.token_strip_chunk_size,
            )
            for r in ts_all:
                depth = len(r.split())
                if depth == 1:
                    ts_rules_singles.append(r)
                else:
                    ts_rules_chains.append(r)
                    if depth <= self.max_depth:
                        ts_sbd[depth].add(r)
            _log_token_strip_stats(len(target_words), ts_all, inject_sbd=self.builtin_seeds)
            if self.max_depth >= 2:
                toggle_seeds = _generate_toggle_chain_seeds(self.max_depth)
                n_toggle_new = 0
                for ts_chain in toggle_seeds:
                    depth = len(ts_chain.split())
                    if depth >= 2:
                        ts_rules_chains.append(ts_chain)
                        ts_sbd.setdefault(depth, set()).add(ts_chain)
                        n_toggle_new += 1
                if n_toggle_new:
                    log_info(f"[S0]    Toggle-chain seeds: {bold(cyan(str(n_toggle_new)))}")
            ts_extra_singles = [r for r in ts_rules_singles if r not in builtin_set]
            if ts_rules_chains:
                all_seeds = list(all_seeds) + ts_rules_chains

        if not self.gpu_engine.initialize_gpu(self.device_spec):
            return all_counts
        self.params = calculate_dynamic_parameters(
            self.base_count, self.target_count, self.gpu_engine.device,
            self.params['TARGET_SECONDS']/3600, bloom_mb_override=self.bloom_mb)
        self.params['MAX_CHAIN_DEPTH'] = self.max_depth
        self.gpu_engine.params = self.params

        extra_seeds = [s for s in all_seeds if ' ' not in s.strip()]
        extra_seeds_valid = [s for s in extra_seeds if s not in builtin_set]
        rules_phase1 = rules + ts_extra_singles + extra_seeds_valid
        seed_chains = [s for s in all_seeds if ' ' in s.strip()]
        if extra_seeds_valid:
            log_info(f"[SEED] {len(extra_seeds_valid)} seed single-rule(s) added to STAGE 1")
        if seed_chains and self.max_depth < 2:
            log_warn(f"[SEED] {len(seed_chains)} chain seed(s) ignored — requires --max-depth >= 2")

        # Clear STAGE 0 globals
        global _p0_worker_base_set, _p0_worker_base_by_len
        _p0_worker_base_set = set()
        _p0_worker_base_by_len = {}
        log_debug("[MEM] STAGE 0 index globals cleared before STAGE 1")

        log_info("[GPU]  Building bloom filter on GPU …")
        if not self.gpu_engine.compile_kernel(): return all_counts
        bloom_filter = self.gpu_engine.generate_bloom_filter_gpu(target_words)
        self.gpu_engine.upload_bloom_filter(bloom_filter)

        log_section("STAGE 1 — Single Rule Search")
        seed_note = f"  ({len(extra_seeds_valid)} from seeds)" if extra_seeds_valid else ""
        log_info(f"[S1]    {len(base_words):,} base words × {len(rules_phase1):,} atomic rules{seed_note}")
        t0 = time.time()
        single = self.gpu_engine.process_all_words_single_rule(base_words, rules_phase1, bloom_filter)
        t1 = time.time()
        log_debug(f"[S1]    Elapsed: {t1-t0:.1f}s  ({len(single)} unique rules so far)")
        all_counts.update(single)

        seed_hits = Counter()
        if self.builtin_seeds and not _kb.quit_requested:
            log_section("STAGE S — Seed Extraction (families A-M)")
            sbd = self.gpu_engine.build_numeric_seed_families(max_depth=self.max_depth)
            if ts_sbd:
                n_injected = 0
                for depth, chains in ts_sbd.items():
                    before = len(sbd.setdefault(depth, set()))
                    sbd[depth].update(chains)
                    n_injected += len(sbd[depth]) - before
                if n_injected:
                    log_info(f"[SEED]    STAGE 0 injected {bold(cyan(str(n_injected)))} chain(s) into STAGE S sbd")
            seed_hits = self.gpu_engine.run_seed_extraction_pass(base_words, sbd, bloom_filter, rules_phase1)
            all_counts.update(seed_hits)
            ts = time.time()
        else:
            if _kb.quit_requested:
                log_warn("[SEED]    Skipped — early exit requested")
            else:
                log_info(f"[SEED]    {yellow('Skipped')} (--no-builtin-seeds)")
            sbd = {}
            ts = t1

        if self.max_depth > 1 and not _kb.quit_requested:
            log_section("STAGE 2 — Rule Chain Search")
            if self.genetic and self.max_depth >= 2:
                _min_ga_secs    = 120.0
                _ga_frac        = 0.20
                _reserved_for_ga = max(_min_ga_secs, self.params['TARGET_SECONDS'] * _ga_frac)
            else:
                _reserved_for_ga = 0.0
            remaining = max(0, self.params['TARGET_SECONDS'] - (ts-t0) - _reserved_for_ga)
            budget    = remaining * self.params['EST_COMBOS_PER_SEC'] * TIME_SAFETY_FACTOR
            depths    = list(range(2, self.max_depth+1))
            depth_budgets = ({d: int(budget/len(depths)/(len(base_words)*d)) for d in depths} if budget > 0 and base_words and depths else {d: 0 for d in depths})
            for d in depths:
                key = f'depth{d}_override'
                if key in depth_overrides and depth_overrides[key] is not None:
                    depth_budgets[d] = depth_overrides[key]
                depth_budgets[d] = max(0, depth_budgets[d])
            if self.max_chains:
                total = sum(depth_budgets.values())
                if total > self.max_chains:
                    scale = self.max_chains / total
                    depth_budgets = {d: int(v*scale) for d,v in depth_budgets.items()}
            for d, bgt in depth_budgets.items():
                self.params[f'CHAIN_GEN_LIMIT_{d}'] = bgt
            log_info(f"[S2]    depth 2-{self.max_depth} | " + " | ".join(f"d{d}:{v:,}" for d,v in depth_budgets.items()))
            chains = self.gpu_engine.process_all_words_chain_rules(base_words, rules_phase1, self.max_depth, bloom_filter, single, seed_chains=seed_chains, prebuilt_sbd=sbd)
            all_counts.update(chains)

        if self.genetic and self.max_depth >= 2 and not _kb.quit_requested:
            log_section("STAGE 3 — Genetic Algorithm Rule Evolution")
            rule_pool = HashcatRuleValidator.validate_rules_for_gpu(rules_phase1)
            hot_rules = [r for r, _ in sorted(single.items(), key=lambda kv: -kv[1])]
            t_now = time.time()
            remaining = max(0.0, self.params['TARGET_SECONDS'] - (t_now - t0))
            if self.genetic and self.max_depth >= 2:
                _min_ga_secs    = 120.0
                _ga_frac        = 0.20
                _reserved_for_ga = max(_min_ga_secs, self.params['TARGET_SECONDS'] * _ga_frac)
            else:
                _reserved_for_ga = 0.0
            ga_budget = max(remaining, _reserved_for_ga)
            if ga_budget < 5.0:
                log_warn(f"[S3]    Only {ga_budget:.1f}s available — consider raising --target-hours")
            else:
                log_info(f"[S3]    Budget: {bold(f'{ga_budget:.0f}s')}")
            known_rules_set = set(all_counts.keys())
            evolver = GeneticRuleEvolver(
                gpu_engine=self.gpu_engine, base_words=base_words,
                rule_pool=rule_pool, max_depth=self.max_depth,
                pop_size=self.genetic_pop, elite_frac=self.genetic_elite,
                seed_hits=seed_hits, known_rules=known_rules_set,
            )
            ga_hits = evolver.evolve(hot_rules, self.genetic_generations, ga_budget)
            before = len(all_counts)
            all_counts.update(ga_hits)
            new_from_ga = len(all_counts) - before
            n_truly_novel = sum(1 for r in ga_hits if r not in known_rules_set)
            log_info(f"[S3]    {bold(cyan(str(new_from_ga)))} net new rules from STAGE 3  ({bold(green(str(len(ga_hits))))} total GA hits, {bold(cyan(str(n_truly_novel)))} genuinely novel)")

        validated = Counter({r: c for r,c in all_counts.items() if HashcatRuleValidator.validate_rule_for_gpu(r)})
        return validated

# --------------------------------------------------------------------
# Wordlist loader
# --------------------------------------------------------------------
def load_wordlist(filename: str) -> list:
    words = set()
    try:
        with open(filename, 'r', encoding='latin-1', errors='ignore') as f:
            for line in tqdm(f, desc=green("  Loading  "), unit="line", ncols=88,
                             leave=False, bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}] {postfix}"):
                w = line.strip()
                if w and len(w) <= MAX_WORD_LEN: words.add(w)
    except FileNotFoundError:
        log_error(f"File not found: {filename}"); sys.exit(1)
    result = list(words)
    log_info(f"[LOAD] {bold(os.path.basename(filename))}: {bold(cyan(f'{len(result):,}'))} unique words")
    return result

# ====================================================================
# --- ENTRY POINT ---
# ====================================================================
def main() -> None:
    global VERBOSE, ALLOW_REJECT_RULES
    ap = argparse.ArgumentParser(prog='rulest', description='GPU-Compatible Hashcat Rules Engine (core token‑strip)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('base_wordlist', nargs='?', default=None)
    ap.add_argument('target_wordlist', nargs='?', default=None)
    ap.add_argument('-o', '--output', default='rulest_output.txt')
    ap.add_argument('--device', default=None)
    ap.add_argument('--list-devices', action='store_true')
    ap.add_argument('--max-depth', type=int, default=2)
    ap.add_argument('--target-hours', type=float, default=0.5)
    ap.add_argument('--max-chains', type=int, default=0)
    ap.add_argument('--bloom-mb', type=int, default=0)
    ap.add_argument('--seed-rules', default=None)
    for i in range(2, 11):
        ap.add_argument(f'--depth{i}-chains', type=int, default=None, dest=f'depth{i}_chains')
    ap.add_argument('--allow-reject-rules', action='store_true')
    ap.add_argument('--no-builtin-seeds', action='store_true')
    pt = ap.add_argument_group('STAGE 0 — Token-Strip Rule Extraction')
    pt.add_argument('--token-strip', action='store_true')
    pt.add_argument('--token-strip-min-stem', type=int, default=4, metavar='N')
    pt.add_argument('--token-strip-max-prefix', type=int, default=4, metavar='N')
    pt.add_argument('--token-strip-max-suffix', type=int, default=4, metavar='N')
    pt.add_argument('--token-strip-min-leet-amb', type=int, default=3, metavar='N')
    pt.add_argument('--token-strip-workers', type=int, default=0, metavar='N')
    pt.add_argument('--token-strip-chunk-size', type=int, default=0, metavar='N')
    ga = ap.add_argument_group('STAGE 3 — Genetic Algorithm')
    ga.add_argument('--genetic', action='store_true')
    ga.add_argument('--genetic-generations', type=int, default=50, metavar='N')
    ga.add_argument('--genetic-pop', type=int, default=200, metavar='N')
    ga.add_argument('--genetic-elite', type=float, default=0.15, metavar='F')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    ALLOW_REJECT_RULES = args.allow_reject_rules
    VERBOSE            = args.debug
    print_banner()
    _print_controls()
    if args.list_devices: list_devices(); sys.exit(0)
    if not args.base_wordlist or not args.target_wordlist:
        ap.print_help(); print(); log_error("Both BASE and TARGET wordlists are required."); sys.exit(1)
    if args.max_depth > MAX_HASHCAT_CHAIN:
        log_warn(f"Depth capped to {MAX_HASHCAT_CHAIN}"); args.max_depth = MAX_HASHCAT_CHAIN
    elif args.max_depth < 1:
        log_error("--max-depth must be >= 1"); sys.exit(1)

    log_info(f"  base      : {bold(args.base_wordlist)}")
    log_info(f"  target    : {bold(args.target_wordlist)}")
    log_info(f"  depth     : {bold(str(args.max_depth))}  |  hours: {bold(str(args.target_hours))}  |  bloom: {bold(str(args.bloom_mb or 'auto'))}MB  {bold(cyan('(GPU-accelerated)'))}")
    _bs = red('DISABLED (--no-builtin-seeds)') if args.no_builtin_seeds else green('enabled (families A-M)')
    log_info(f"  builtin seeds (STAGE S) : {_bs}")
    if args.token_strip:
        _inj = green('→ STAGE S sbd') if not args.no_builtin_seeds else yellow('→ STAGE 1 only')
        log_info(f"  {bold(cyan('STAGE 0'))} : {green('CPU exact-match (core + insert)')}  min-stem={args.token_strip_min_stem}  prefix={args.token_strip_max_prefix}  suffix={args.token_strip_max_suffix}  leet-amb={args.token_strip_min_leet_amb}  workers={args.token_strip_workers or mp.cpu_count()}  {_inj}")
    if args.seed_rules: log_info(f"  seeds     : {bold(args.seed_rules)}")
    if args.genetic:
        if not 0.0 < args.genetic_elite < 1.0:
            log_error("--genetic-elite must be between 0.0 and 1.0"); sys.exit(1)
        log_info(f"  {bold(green('STAGE 3 GA'))} : pop={args.genetic_pop}  gen={args.genetic_generations}  elite={args.genetic_elite:.0%}")
    print()

    base_words   = load_wordlist(args.base_wordlist)
    target_words = load_wordlist(args.target_wordlist)
    print()

    _kb.start()

    t_start   = time.time()
    extractor = GPUExtractor(
        len(base_words), len(target_words), args.max_depth,
        args.device, args.target_hours, args.max_chains,
        args.seed_rules, args.bloom_mb,
        builtin_seeds             = not args.no_builtin_seeds,
        genetic                   = args.genetic,
        genetic_generations       = args.genetic_generations,
        genetic_pop               = args.genetic_pop,
        genetic_elite             = args.genetic_elite,
        token_strip               = args.token_strip,
        token_strip_min_stem      = args.token_strip_min_stem,
        token_strip_max_prefix    = args.token_strip_max_prefix,
        token_strip_max_suffix    = args.token_strip_max_suffix,
        token_strip_min_leet_amb  = args.token_strip_min_leet_amb,
        token_strip_workers       = args.token_strip_workers,
        token_strip_chunk_size    = args.token_strip_chunk_size,
    )
    extractor._output_path = args.output
    depth_overrides = {f'depth{i}_override': getattr(args, f'depth{i}_chains') for i in range(2, 11)}
    raw_counts = extractor.extract_rules(base_words, target_words, **depth_overrides)
    _kb.stop()
    del target_words; gc.collect()
    if _kb.quit_requested:
        log_warn("[QUIT]  Early exit — saving partial results")
    log_info(f"\n[GPU]  Raw bloom-filter candidates: {bold(cyan(str(len(raw_counts))))}"); print()

    final_counts = minimize_by_signature(raw_counts, BUILTIN_PROBES)
    if ':' not in final_counts: final_counts[':'] = 0

    dd: Dict[int, int] = defaultdict(int)
    for r in final_counts:
        if r != ':': dd[len(r.split())] += 1
    ds = '  '.join(f"d{d}:{dd[d]:,}" for d in sorted(dd))
    final_rules = len(final_counts) - (1 if ':' in final_counts else 0)
    removed     = len(raw_counts) - len(final_counts)

    si = sorted(final_counts.items(), key=lambda kv: (-kv[1], len(kv[0].split()), kv[0]))
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("# rulest — GPU-Compatible Hashcat Rules Engine (core token‑strip)\n")
        f.write(f"# Generated      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Base           : {os.path.basename(args.base_wordlist)}\n")
        f.write(f"# Target         : {os.path.basename(args.target_wordlist)}\n")
        f.write(f"# Depth          : 1-{args.max_depth}\n")
        bloom_mb = args.bloom_mb if args.bloom_mb > 0 else "auto"
        f.write(f"# Bloom          : {bloom_mb} MB  (GPU-accelerated, CPU fallback)\n")
        f.write(f"# STAGE 0        : core + insert mode  GPU-verified candidates\n")
        if args.genetic:
            f.write(f"# STAGE 3 GA     : pop={args.genetic_pop}  gen={args.genetic_generations}  elite={args.genetic_elite:.0%}\n")
        f.write("#\n")
        f.write(f"# GPU raw candidates      : {len(raw_counts):,}\n")
        mp2 = 'disk-backed SQLite' if len(raw_counts) > MINIMIZE_DISK_THRESHOLD else 'in-memory'
        f.write(f"# Minimization            : {mp2}  (threshold {MINIMIZE_DISK_THRESHOLD:,})\n")
        f.write(f"#   Probe words           : {len(BUILTIN_PROBES)}  (built-in)\n")
        f.write(f"#   Equiv. rules removed  : {removed:,}\n")
        f.write("#\n")
        f.write(f"# Rules kept     : {final_rules:,}  ({ds})\n")
        f.write(f"# Sorted by      : GPU frequency (descending, UTF-8)\n")
        f.write(":\n")
        for r, _ in si:
            if r != ':': f.write(f"{r}\n")
    log_info(f"[OUT]  Minimized rules written to: {bold(args.output)}")

    elapsed = time.time() - t_start
    sep = '─' * min(max(shutil.get_terminal_size((80, 24)).columns - 2, 44), 92)
    print()
    log_info(cyan(sep))
    log_info(f"  {bold('DONE')}  rulest finished in {bold(f'{elapsed:.1f}s')}")
    log_info(cyan(sep))
    log_info(f"  GPU raw candidates : {bold(str(len(raw_counts)))}")
    log_info(f"  Rules kept         : {bold(green(str(final_rules)))}  {dim('('+ds+')')}")
    log_info(f"  Rules removed      : {bold(red(str(removed)))}")
    log_info(f"  Output file        : {bold(args.output)}")
    log_info(cyan(sep))
    top = sorted([(r,s) for r,s in final_counts.items() if r!=':'], key=lambda kv: (-kv[1], len(kv[0].split()), kv[0]))[:20]
    if top:
        print(); log_info(f"  Top {len(top)} rules by GPU frequency:")
        for i,(r,s) in enumerate(top, 1):
            log_info(f"  {dim(str(i).rjust(3)+'.')}  {dim(f'd={len(r.split())}')}  {r:<42s}  {cyan(str(s))}")
    print()

if __name__ == '__main__':
    main()
