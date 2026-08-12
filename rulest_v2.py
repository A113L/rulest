#!/usr/bin/env python3
"""
rulest — GPU-Compatible Hashcat Rules Engine
====================================================================
Optimized version: single‑shard fast path by default, automatic
sharding for filters >512 MB, bit‑mask bloom indexing,
disk fallback for huge target wordlists in Stage 0.
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

# Windows' WaitForMultipleObjects (used internally by multiprocessing.Pool on
# win32) can only wait on at most 63 handles at once. With many CPU cores
# (e.g. 128 logical processors) a Pool sized to mp.cpu_count() raises:
#   ValueError: need at most 63 handles, got a sequence of length N
# This does not happen on POSIX (fork-based) systems, only on Windows/spawn.
# Cap the worker count there so Pool never exceeds the OS limit.
_WIN_MAX_POOL_WORKERS = 60  # stay safely under the 63-handle ceiling

def safe_worker_count(n_workers):
    """Clamp a requested worker/process count to a value that Windows'
    multiprocessing.Pool can actually handle."""
    n_workers = max(1, int(n_workers))
    if os.name == 'nt':
        n_workers = min(n_workers, _WIN_MAX_POOL_WORKERS)
    return n_workers
import functools
import concurrent.futures
import shutil

# ================== GLOBAL FLAGS ===================
VERBOSE            = False
ALLOW_REJECT_RULES = False
# ==================================================

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

# ----------------------------------------------------------------------
# Keyboard controller (pause/resume/quit)
# ----------------------------------------------------------------------
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

class KeyboardController:
    def __init__(self):
        self._paused = False
        self._quit = False
        self._lock = threading.Lock()
        self._thread = None
        self._active = False

    def start(self):
        if not sys.stdin.isatty(): return
        if not (_HAS_TERMIOS or _HAS_MSVCRT): return
        self._active = True
        self._thread = threading.Thread(target=self._reader, daemon=True, name='kb-listener')
        self._thread.start()

    def stop(self): self._active = False

    def check_pause(self):
        while True:
            with self._lock:
                if not self._paused or self._quit: return
            time.sleep(0.15)

    @property
    def quit_requested(self):
        with self._lock:
            return self._quit

    def _handle(self, ch):
        ch = ch.lower()
        if ch == 'p':
            with self._lock:
                if self._quit: return
                self._paused = True
            w = shutil.get_terminal_size((80,24)).columns
            print(f"\n{yellow('─'*w)}")
            print(f"{yellow('│')} {bold('PAUSED')}  —  press {bold(green('r'))} resume  |  {bold(yellow('q'))} save & quit")
            print(f"{yellow('─'*w)}")
        elif ch == 'r':
            with self._lock:
                if not self._paused: return
                self._paused = False
            w = shutil.get_terminal_size((80,24)).columns
            print(f"\n{green('─'*w)}")
            print(f"{green('│')} {bold('RESUMED')}")
            print(f"{green('─'*w)}\n")
        elif ch == 'q':
            with self._lock:
                already = self._quit
                self._quit = True
                self._paused = False
            if not already:
                w = shutil.get_terminal_size((80,24)).columns
                print(f"\n{yellow('─'*w)}")
                print(f"{yellow('│')} {bold(yellow('EARLY EXIT REQUESTED'))}  —  finishing current batch then saving …")
                print(f"{yellow('─'*w)}\n")

    def _reader_termios(self):
        fd = sys.stdin.fileno()
        old = _termios.tcgetattr(fd)
        try:
            _tty.setcbreak(fd)
            while self._active:
                if _select.select([sys.stdin],[],[],0.15)[0]:
                    self._handle(sys.stdin.read(1))
        except Exception: pass
        finally:
            try: _termios.tcsetattr(fd, _termios.TCSADRAIN, old)
            except Exception: pass

    def _reader_msvcrt(self):
        while self._active:
            if _msvcrt.kbhit(): self._handle(_msvcrt.getwch())
            else: time.sleep(0.05)

    def _reader(self):
        if _HAS_TERMIOS: self._reader_termios()
        elif _HAS_MSVCRT: self._reader_msvcrt()

_kb = KeyboardController()

# ----------------------------------------------------------------------
# Colors & helpers
# ----------------------------------------------------------------------
class C:
    RED = '\033[91m'; GREEN = '\033[92m'; YELLOW = '\033[93m'
    BLUE = '\033[94m'; CYAN = '\033[96m'; MAGENTA= '\033[95m'
    BOLD = '\033[1m';  DIM = '\033[2m';   END = '\033[0m'

def red(t): return f"{C.RED}{t}{C.END}"
def green(t): return f"{C.GREEN}{t}{C.END}"
def yellow(t): return f"{C.YELLOW}{t}{C.END}"
def blue(t): return f"{C.BLUE}{t}{C.END}"
def cyan(t): return f"{C.CYAN}{t}{C.END}"
def bold(t): return f"{C.BOLD}{t}{C.END}"
def dim(t): return f"{C.DIM}{t}{C.END}"

def _fmt_speed(n, unit="combos"):
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.1f}G {unit}/s"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M {unit}/s"
    if n >= 1_000: return f"{n/1_000:.1f}K {unit}/s"
    return f"{n:.0f} {unit}/s"

def log_info(msg): print(msg)
def log_debug(msg):
    if VERBOSE: print(f"{dim('[dbg]')} {msg}")
def log_warn(msg): print(yellow(f"[WARN] {msg}"))
def log_error(msg): print(red(f"[ERROR] {msg}"))

def log_section(title):
    w = shutil.get_terminal_size((80,24)).columns
    bar = '─' * w
    print(f"\n{cyan(bar)}")
    print(f"{cyan('│')} {bold(title.upper())}")
    print(f"{cyan(bar)}")

# ----------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------
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

def print_banner():
    print(BANNER)

def _print_controls():
    if not (sys.stdin.isatty() and (_HAS_TERMIOS or _HAS_MSVCRT)):
        return
    w = shutil.get_terminal_size((80,24)).columns - 2
    sep = dim('─' * w)
    print(f"  {sep}")
    print(f"  {bold('Controls')}  "
          f"{cyan(bold('[p]'))} pause processing   "
          f"{green(bold('[r]'))} resume after pause   "
          f"{yellow(bold('[q]'))} save partial results & quit")
    print(f"  {sep}")
    print()

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MAX_WORD_LEN = 256
MAX_RULE_LEN = 16
MAX_OUTPUT_LEN = 512
MAX_CHAIN_STRING_LEN = 128  # 512 kills GPU occupancy (cb[] is a per-thread private array in the chain kernel);
                            # 128 is already truncation-safe (see bounds checks in find_rule_chains_gpu)
MAX_HASHCAT_CHAIN = 31
VRAM_USAGE_FACTOR = 0.55
BLOOM_HASH_FUNCTIONS = 4
BASELINE_COMBOS_PER_SEC = 120_000_000
LOW_END_COMPUTE_UNITS_THRESHOLD = 20
LOW_END_COMBOS_PER_SEC = 40_000_000
POSSIBLE_WORK_GROUP_SIZES = [32,64,128,256,512,1024]
BASE_WORDS_PER_BATCH = 5000
BASE_CHAINS_PER_BATCH = 2000
BASE_WORD_SUB_BATCH = 20000
BASE_MAX_SAFE_RESULTS = 25000
# Hard ceiling on MAX_SAFE_RESULTS_PER_BATCH. Lowering this shrinks the GPU
# output buffer allocated every batch (faster alloc/free churn), and — unlike
# capping the buffer size alone — it lowers the kernel's compiled
# MAX_CHAINS_TO_FIND bound by the same amount, so the two can never diverge.
# Tradeoff: if a single batch finds more matches than this cap, the excess
# are silently skipped by the kernel's bounds check (lost, not corrupted).
MAX_SAFE_RESULTS_CAP = 5000
HOT_RULE_RATIO = 0.6
MAX_ATTEMPTS_MULTIPLIER = 5
TIME_SAFETY_FACTOR = 0.9
OPTIMAL_GLOBAL_MULTIPLIER_BASE = 16
FNV1A_PRIME = 16777619
FNV1A_OFFSET_BASIS = 2166136261
FNV1A_SEED1 = 0xDEADBEEF
FNV1A_SEED2 = 0xCAFEBABE
MAX_GPU_RULES = 255
# Hashcat position encoding: '0'-'9' -> 0-9, 'A'-'Z' -> 10-35 (base36). Used by all
# single-position rule functions (T,D,L,R,+,-,.,,,',z,Z,y,Y,p...) so words longer
# than 9 chars can still be targeted at any position.
POSITION_CHARS = string.digits + string.ascii_uppercase
_UNSUPPORTED_SENTINEL = object()
MINIMIZE_DISK_THRESHOLD = 500_000
MINIMIZE_DISK_BATCH_SIZE = 10_000
SPECIAL_CHARS_TOP = ['!','@','#','$','%','^','&','*','?','.','-','_','+','(',')']
SPECIAL_CHARS_CORE = ['!','@','#','$','%','*','?']
LEET_SUBS = [('a','@'),('e','3'),('o','0'),('i','1'),('l','1'),('s','5'),('s','$'),('t','7'),('a','4'),('i','!')]
LEET_OPS = [f's{orig}{rep}' for orig,rep in LEET_SUBS]
TOKEN_STRIP_LEET_TABLE = [
    ('@','a','sa@'),('3','e','se3'),('0','o','so0'),('1','i','si1'),
    ('1','l','sl1'),('5','s','ss5'),('$','s','ss$'),('7','t','st7'),
    ('4','a','sa4'),('!','i','si!')
]
_TOKEN_STRIP_LEET_BY_CHAR = defaultdict(list)
for enc,base,rule in TOKEN_STRIP_LEET_TABLE:
    _TOKEN_STRIP_LEET_BY_CHAR[enc].append((base,rule))
TOKEN_STRIP_LEET_CHARS = set(_TOKEN_STRIP_LEET_BY_CHAR.keys())
TOKEN_STRIP_BOUNDARY = set('0123456789!@#$%^&*?.-_+()')
TOKEN_STRIP_ALPHA_BOUNDARY = set(string.ascii_letters) | set('!@#$%^&*?.-_+()')
_p0_worker_base_set = set()
_p0_worker_base_by_len = {}

# ----------------------------------------------------------------------
# Built-in probe set — 50 words, identical to minimizer.py v1.4 BUILTIN_PROBES.
# Covers: all positions 0-35 (A-Z), all 95 printable ASCII chars, short/long
# words, mixed-case, digit, special-char, and repeated-char patterns.
# ----------------------------------------------------------------------
BUILTIN_PROBES = [
    "ab", "abc", "abcd", "pass", "root", "test", "admin", "login",
    "letmein", "welcome", "password", "sunshine", "football", "baseball",
    "princess", "dragon12", "qwertyuiop", "iloveyou12", "monkey12345",
    "superman123", "mustang2024", "administrator1", "iloveyouforever",
    "qwertyuiopasdfgh", "correcthorsebattery", "averylongpassword1234",
    "averylongpassword12345678", "averylongpassword1234567890ab",
    "averylongpassword1234567890abcdef", "averylongpassword1234567890abcdefghi",
    "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "!@#$%^&*()-_=+[]{}|;:,.<>?/~", "a`b", 'a"b', "a'b", "a\\b", "a b",
    "Password", "AdminUser", "MySecret", "HelloWorld", "pass123", "admin2024",
    "test1234", "user9999", "p@ssw0rd", "s3cur1ty", "aaaa", "bbbb",
]
BUILTIN_PROBES = list(dict.fromkeys(BUILTIN_PROBES))

# ----------------------------------------------------------------------
# Rule validator
# ----------------------------------------------------------------------
def should_exclude_rule(rule):
    if ALLOW_REJECT_RULES: return False
    if not rule: return False
    if len(rule)==1 and rule in ('_','M','4','6','Q'): return True
    if len(rule)==2 and rule[0] in ('!','/','(',')','<','>','_'): return True
    if len(rule)==3 and rule[0] in ('?','=','v'): return True
    return False

def _is_digit_char(c): return '0' <= c <= '9'
def _is_pos_char(c): return ('0' <= c <= '9') or ('A' <= c <= 'Z')

@functools.lru_cache(maxsize=200000)
def _validate_rule_for_gpu_impl(rule_str):
    if should_exclude_rule(rule_str): return False
    pos = cnt = 0
    n = len(rule_str)
    isd = _is_digit_char
    isp = _is_pos_char
    while pos < n:
        c = rule_str[pos]
        if c == ' ': pos+=1; continue
        if c in ('p','z','Z'):
            cnt+=1; pos+=1
            if pos<n and isp(rule_str[pos]): pos+=1
            continue
        if c in (':','l','u','c','C','t','r','d','f','q','k','K','E','{','}','[',']'):
            pos+=1; cnt+=1; continue
        if c in ('T','D','L','R','+','-','.',',',"'",'y','Y'):
            pos+=1
            if pos>=n or not isp(rule_str[pos]): return False
            pos+=1; cnt+=1; continue
        if c in ('i','o','3'):
            pos+=1
            if pos>=n or not isd(rule_str[pos]): return False
            pos+=1
            if pos>=n: return False
            pos+=1; cnt+=1; continue
        if c in ('x','*','O'):
            pos+=1
            if pos>=n or not isd(rule_str[pos]): return False
            pos+=1
            if pos>=n or not isd(rule_str[pos]): return False
            pos+=1; cnt+=1; continue
        if c == 's':
            pos+=1
            if pos+1>=n: return False
            pos+=2; cnt+=1; continue
        if c in ('@','e','$','^'):
            pos+=1
            if pos>=n: return False
            pos+=1; cnt+=1; continue
        return False
    return cnt <= MAX_GPU_RULES

class HashcatRuleValidator:
    MAX_GPU_RULES = MAX_GPU_RULES
    @staticmethod
    def is_digit(c): return '0' <= c <= '9'
    @staticmethod
    def is_pos_char(c): return ('0' <= c <= '9') or ('A' <= c <= 'Z')
    @staticmethod
    def validate_rule_for_gpu(rule_str):
        return _validate_rule_for_gpu_impl(rule_str)

    @staticmethod
    def validate_rules_for_gpu(rules):
        valid = []
        for r in rules:
            r = r.strip('\n\r')
            if r and _validate_rule_for_gpu_impl(r):
                valid.append(r)
        return valid

# ----------------------------------------------------------------------
# FNV-1a
# ----------------------------------------------------------------------
def fnv1a_32(data, seed=FNV1A_SEED1):
    h = seed ^ FNV1A_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * FNV1A_PRIME) & 0xFFFFFFFF
    return h

# ----------------------------------------------------------------------
# Python rule applicator
# ----------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def _dg(c):
    if '0' <= c <= '9': return ord(c) - 48
    if 'A' <= c <= 'Z': return ord(c) - 55
    return -1

def _py_apply_single_rule(rule, word):
    if not rule: return word
    w = list(word.encode('latin-1'))
    cmd = rule[0]
    # use module-level cached _dg
    try:
        if cmd == ':': pass
        elif cmd == 'l': w = [c|0x20 if 65<=c<=90 else c for c in w]
        elif cmd == 'u': w = [c&~0x20 if 97<=c<=122 else c for c in w]
        elif cmd == 'c':
            if w:
                w[0] = w[0]&~0x20 if 97<=w[0]<=122 else w[0]
                w[1:] = [c|0x20 if 65<=c<=90 else c for c in w[1:]]
        elif cmd == 'C':
            if w:
                w[0] = w[0]|0x20 if 65<=w[0]<=90 else w[0]
                w[1:] = [c&~0x20 if 97<=c<=122 else c for c in w[1:]]
        elif cmd == 't': w = [c|0x20 if 65<=c<=90 else (c&~0x20 if 97<=c<=122 else c) for c in w]
        elif cmd == 'r': w = w[::-1]
        elif cmd == 'd': w = w + w
        elif cmd == 'f': w = w + w[::-1]
        elif cmd == '{':
            if len(w)>1: w = w[1:] + [w[0]]
        elif cmd == '}':
            if len(w)>1: w = [w[-1]] + w[:-1]
        elif cmd == '[':
            if w: w = w[1:]
        elif cmd == ']':
            if w: w = w[:-1]
        elif cmd == 'k':
            if len(w)>=2: w[0],w[1] = w[1],w[0]
        elif cmd == 'K':
            if len(w)>=2: w[-1],w[-2] = w[-2],w[-1]
        elif cmd == 'q':
            out = []
            for c in w: out += [c,c]
            w = out
        elif cmd == 'E':
            out = []; cap = True
            for c in w:
                if cap and 97 <= c <= 122:
                    out.append(c & ~0x20)
                elif not cap and 65 <= c <= 90:
                    out.append(c | 0x20)
                else:
                    out.append(c)
                cap = (c == 32)
            w = out
        elif cmd == '^' and len(rule)>=2: w = [ord(rule[1])] + w
        elif cmd == '$' and len(rule)>=2: w = w + [ord(rule[1])]
        elif cmd == '@' and len(rule)>=2:
            ch = ord(rule[1]); w = [c for c in w if c != ch]
        elif cmd == 'p' and len(rule)>=2:
            n = _dg(rule[1])
            if n>0:
                orig = w[:]
                for _ in range(n): w += orig
        elif cmd == 'T' and len(rule)>=2:
            p = _dg(rule[1])
            if 0<=p<len(w):
                c = w[p]; w[p] = c|0x20 if 65<=c<=90 else (c&~0x20 if 97<=c<=122 else c)
        elif cmd == 'D' and len(rule)>=2:
            p = _dg(rule[1])
            if 0<=p<len(w): w.pop(p)
        elif cmd == 'L' and len(rule)>=2:
            p = _dg(rule[1])
            if 0<=p<len(w): w[p] = (w[p]<<1)&0xFF
        elif cmd == 'R' and len(rule)>=2:
            p = _dg(rule[1])
            if 0<=p<len(w): w[p] = (w[p]>>1)&0xFF
        elif cmd == '+' and len(rule)>=2:
            p = _dg(rule[1])
            if 0<=p<len(w) and w[p] < 255: w[p] = w[p]+1
        elif cmd == '-' and len(rule)>=2:
            p = _dg(rule[1])
            if 0<=p<len(w) and w[p] > 0: w[p] = w[p]-1
        elif cmd in ('.',',') and len(rule)>=2:
            p = _dg(rule[1]); delta = 1 if cmd=='.' else -1
            if 0<=p<len(w): w[p] = (w[p]+delta)&0xFF
        elif cmd == "'" and len(rule)>=2:
            p = _dg(rule[1])
            if 0<=p: w = w[:p]
        elif cmd == 'z' and len(rule)>=2:
            n = _dg(rule[1])
            if n>0 and w: w = [w[0]]*n + w
        elif cmd == 'Z' and len(rule)>=2:
            n = _dg(rule[1])
            if n>0 and w: w = w + [w[-1]]*n
        elif cmd == 'y' and len(rule)>=2:
            n = _dg(rule[1])
            if n>0: w = w[:n] + w
        elif cmd == 'Y' and len(rule)>=2:
            n = _dg(rule[1])
            if n>0 and len(w)>=n: w = w + w[-n:]
        elif cmd == 's' and len(rule)>=3:
            a,b = ord(rule[1]), ord(rule[2])
            w = [b if c==a else c for c in w]
        elif cmd == 'i' and len(rule)>=3:
            p,ch = _dg(rule[1]), ord(rule[2])
            if 0<=p<=len(w): w.insert(p,ch)
        elif cmd == 'o' and len(rule)>=3:
            p,ch = _dg(rule[1]), ord(rule[2])
            if 0<=p<len(w): w[p] = ch
        elif cmd == 'e' and len(rule)>=2:
            sep = ord(rule[1]); out = []; cap = True
            for c in w:
                if cap and 97<=c<=122:
                    out.append(c & ~0x20)
                elif not cap and 65<=c<=90:
                    out.append(c | 0x20)
                else:
                    out.append(c)
                cap = (c == sep)
            w = out
        elif cmd == 'x' and len(rule)>=3:
            n, m = _dg(rule[1]), _dg(rule[2])
            if n>=0 and m>=0 and n<len(w): w = w[n:n+m]
        elif cmd == 'O' and len(rule)>=3:
            p,m = _dg(rule[1]), _dg(rule[2])
            if 0<=p<len(w) and m>0: w = w[:p] + w[p+m:]
        elif cmd == '*' and len(rule)>=3:
            a,b = _dg(rule[1]), _dg(rule[2])
            if 0<=a<len(w) and 0<=b<len(w) and a!=b: w[a],w[b]=w[b],w[a]
        elif cmd == '3' and len(rule)>=3:
            n,sep = _dg(rule[1]), ord(rule[2]); cnt=0
            for i,c in enumerate(w):
                if c==sep:
                    cnt+=1
                    if cnt==n+1 and i+1<len(w):
                        ci = w[i+1]
                        w[i+1] = ci|0x20 if 65<=ci<=90 else (ci&~0x20 if 97<=ci<=122 else ci)
                        break
        else: return None
    except: return None
    try: return bytes(w).decode('latin-1')
    except: return None

def py_apply_chain(chain, word):
    cur = word
    for r in chain.split():
        cur = _py_apply_single_rule(r, cur)
        if cur is None: return None
    return cur

# ----------------------------------------------------------------------
# Signature minimisation — using SHA1 hash of outputs
# ----------------------------------------------------------------------

# Module-level globals used by the multiprocessing worker so it is
# picklable (local/closure functions cannot be pickled on spawn/forkserver).
_minimize_worker_probe_words: list = []

def _minimize_worker_init(probe_words):
    """Pool initializer — stores probe words in each worker process."""
    global _minimize_worker_probe_words
    _minimize_worker_probe_words = probe_words

def _minimize_worker_sig(rule_count):
    """Top-level (picklable) worker: compute signature for one (rule, count) pair."""
    rule, count = rule_count
    sig = compute_rule_signature_hash(rule, _minimize_worker_probe_words)
    return sig, rule, count


def compute_rule_signature_hash(rule, probe_words):
    outputs = []
    for w in probe_words:
        out = py_apply_chain(rule, w)
        if out is None:
            # Return the sentinel string directly so callers can detect it with
            # sig.startswith('__UNSUPPORTED__').  Previously this returned a SHA1
            # digest of the sentinel, making the check unreachable.
            return f"__UNSUPPORTED__::{rule}"
        outputs.append(out)
    combined = '\x00'.join(outputs).encode('utf-8', errors='replace')
    return hashlib.sha1(combined).hexdigest()

def minimize_by_signature(rule_counter, probe_words):
    if not rule_counter: return Counter()
    log_section("POST-PROCESSING — Signature-Based Functional Minimization")
    n = len(rule_counter)
    log_info(f"[MINIMIZE] Candidates  : {bold(str(n))}")
    log_info(f"[MINIMIZE] Probe words : {bold(str(len(probe_words)))}")
    if n > MINIMIZE_DISK_THRESHOLD:
        return _minimize_disk(rule_counter, probe_words)
    else:
        return _minimize_mem(rule_counter, probe_words)

def _minimize_mem(rule_counter, probe_words):
    sig_map = defaultdict(list)
    rule_items = list(rule_counter.items())
    n_total = len(rule_items)
    ncols = shutil.get_terminal_size((80,24)).columns
    n_workers = safe_worker_count(min(mp.cpu_count(), n_total // 200 + 1))
    use_mp = n_total >= 500 and n_workers > 1
    log_info(f"[MINIMIZE] Workers    : {bold(str(n_workers if use_mp else 1))}")
    t0 = time.time()
    if use_mp:
        if hasattr(os, 'fork'):
            ctx = mp.get_context('fork')
        else:
            ctx = mp.get_context('spawn')
        with tqdm(total=n_total, desc=green("  Minimizing"), unit="rule", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            with ctx.Pool(processes=n_workers,
                          initializer=_minimize_worker_init,
                          initargs=(probe_words,)) as pool:
                for sig, rule, count in pool.imap_unordered(_minimize_worker_sig, rule_items, chunksize=64):
                    sig_map[sig].append((rule, count))
                    pbar.update(1)
                    elapsed = time.time() - t0
                    spd = _fmt_speed(pbar.n / elapsed if elapsed>0 else 0, "rules")
                    pbar.set_postfix({"unique_sigs": cyan(str(len(sig_map))), "spd": green(spd)}, refresh=False)
    else:
        with tqdm(total=n_total, desc=green("  Minimizing"), unit="rule", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for rule, count in rule_items:
                sig = compute_rule_signature_hash(rule, probe_words)
                sig_map[sig].append((rule, count))
                pbar.update(1)
                elapsed = time.time() - t0
                spd = _fmt_speed(pbar.n / elapsed if elapsed>0 else 0, "rules")
                pbar.set_postfix({"unique_sigs": cyan(str(len(sig_map))), "spd": green(spd)}, refresh=False)
    survivors = Counter()
    n_unsupported = 0
    for sig, group in sig_map.items():
        if sig.startswith('__UNSUPPORTED__'):
            n_unsupported += len(group) - 1  # count duplicates removed, not total kept
        best_rule, best_count = min(group, key=lambda x: (-x[1], len(x[0].split()), x[0]))
        survivors[best_rule] = best_count
    removed = n_total - len(survivors)
    log_info(f"[MINIMIZE] {green('Done')}")
    log_info(f"           Unique signatures : {bold(cyan(str(len(sig_map)))):>12s}")
    log_info(f"           Rules kept        : {bold(green(str(len(survivors)))):>12s}")
    log_info(f"           Rules removed     : {bold(red(str(removed))):>12s}  ({removed/max(1,n_total):.1%})")
    if n_unsupported:
        log_info(f"           Unsupported (kept 1 each group) : {bold(str(n_unsupported))}")
    return survivors

def _minimize_disk(rule_counter, probe_words):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.db', prefix='rulest_minimize_')
    os.close(tmp_fd)
    try:
        conn = sqlite3.connect(tmp_path)
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = OFF')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA cache_size = -131072')
        conn.execute('CREATE TABLE sig_best (sig_hash TEXT PRIMARY KEY, rule TEXT NOT NULL, count INTEGER NOT NULL, depth INTEGER NOT NULL)')
        conn.commit()
        _UPSERT = '''
            INSERT INTO sig_best (sig_hash, rule, count, depth)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sig_hash) DO UPDATE SET
                rule  = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.rule
                    WHEN excluded.count = sig_best.count AND excluded.depth < sig_best.depth THEN excluded.rule
                    WHEN excluded.count = sig_best.count AND excluded.depth = sig_best.depth AND excluded.rule < sig_best.rule THEN excluded.rule
                    ELSE sig_best.rule
                END,
                count = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.count
                    ELSE sig_best.count
                END,
                depth = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.depth
                    WHEN excluded.count = sig_best.count AND excluded.depth < sig_best.depth THEN excluded.depth
                    WHEN excluded.count = sig_best.count AND excluded.depth = sig_best.depth AND excluded.rule < sig_best.rule THEN excluded.depth
                    ELSE sig_best.depth
                END
        '''
        rule_items = list(rule_counter.items())
        n_total = len(rule_items)
        batch = []
        log_info(f"[MINIMIZE] Temp DB : {dim(tmp_path)}")
        ncols = shutil.get_terminal_size((80,24)).columns
        with tqdm(total=n_total, desc=green("  Minimizing"), unit="rule", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            t0 = time.time()
            done = 0
            for rule, count in rule_items:
                sig = compute_rule_signature_hash(rule, probe_words)
                depth = len(rule.split())
                batch.append((sig, rule, count, depth))
                if len(batch) >= MINIMIZE_DISK_BATCH_SIZE:
                    conn.executemany(_UPSERT, batch); conn.commit()
                    done += len(batch)
                    elapsed = time.time() - t0
                    spd = _fmt_speed(done / elapsed if elapsed>0 else 0, "rules")
                    n_sigs = conn.execute('SELECT COUNT(*) FROM sig_best').fetchone()[0]
                    pbar.set_postfix({"unique_sigs": cyan(str(n_sigs)), "spd": green(spd)}, refresh=False)
                    pbar.update(len(batch)); batch.clear()
            if batch:
                conn.executemany(_UPSERT, batch); conn.commit()
                pbar.update(len(batch))
        survivors = Counter()
        for rule_str, cnt in conn.execute('SELECT rule, count FROM sig_best'):
            survivors[rule_str] = cnt
        n_unique_sigs = len(survivors)
        conn.close()
        removed = n_total - n_unique_sigs
        log_info(f"[MINIMIZE] {green('Done')}")
        log_info(f"           Unique signatures : {bold(cyan(str(n_unique_sigs))):>12s}")
        log_info(f"           Rules kept        : {bold(green(str(n_unique_sigs))):>12s}")
        log_info(f"           Rules removed     : {bold(red(str(removed))):>12s}  ({removed/max(1,n_total):.1%})")
        return survivors
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

# ----------------------------------------------------------------------
# Stage 0 – Token‑strip rule extraction (core + insert)
# ----------------------------------------------------------------------
def _hashcat_title_case(s):
    """Mirror hashcat's 'E' rule: capitalize the first letter of each space-separated word."""
    res = list(s); cap = True
    for i,c in enumerate(res):
        if cap and 'a'<=c<='z': res[i]=c.upper(); cap=False
        elif not cap and 'A'<=c<='Z': res[i]=c.lower()
        if c == ' ': cap=True
    return ''.join(res)

def _infer_case_rules(cased_stem):
    lower = cased_stem.lower()
    if cased_stem == lower: return [[]]
    cand = []
    if cased_stem == lower.upper(): cand.append(['u'])
    if len(cased_stem)>=1 and cased_stem[0].isupper() and cased_stem[1:].islower(): cand.append(['c'])
    if len(cased_stem)>=1 and cased_stem[0].islower() and cased_stem[1:].isupper(): cand.append(['C'])
    toggled = ''.join(c.lower() if c.isupper() else c.upper() for c in cased_stem)
    if cased_stem == toggled: cand.append(['t'])
    if cased_stem == _hashcat_title_case(lower): cand.append(['E'])
    if not cand:
        up = [i for i,c in enumerate(cased_stem) if c != lower[i]]
        if up and all(p<=9 for p in up): cand.append([f'T{p}' for p in up])
    return cand

def _leet_decode_variants(mid, max_amb=3):
    for ch in mid:
        if not ch.isalpha() and ch not in TOKEN_STRIP_LEET_CHARS: return
    leet = [(i,ch,_TOKEN_STRIP_LEET_BY_CHAR[ch]) for i,ch in enumerate(mid) if ch in TOKEN_STRIP_LEET_CHARS]
    if not leet:
        yield (mid, frozenset()); return
    if sum(1 for _,_,opts in leet if len(opts)>1) > max_amb: return
    for combo in itertools.product(*[opts for _,_,opts in leet]):
        dec = list(mid); rules = set()
        for (pos,_,_), (b,r) in zip(leet,combo):
            dec[pos]=b; rules.add(r)
        dstr = ''.join(dec)
        if all(ch.isalpha() for ch in dstr): yield (dstr, frozenset(rules))

def _decode_middle(mid, max_amb=3):
    for leet_dec, leet_rules in _leet_decode_variants(mid, max_amb):
        stem = leet_dec.lower()
        case_cand = _infer_case_rules(leet_dec)
        yield (stem, leet_rules, case_cand if case_cand else [[]])

def _rule_chain_orderings(case_ops, leet_ops, prepend_ops, append_ops, lead=None):
    lead = lead or []
    seen = set()
    def _add(ops):
        full = lead + ops
        key = tuple(full)
        if key not in seen:
            seen.add(key); yield full
    yield from _add(case_ops+leet_ops+prepend_ops+append_ops)
    if case_ops and leet_ops: yield from _add(leet_ops+case_ops+prepend_ops+append_ops)
    if prepend_ops and (case_ops or leet_ops): yield from _add(prepend_ops+case_ops+leet_ops+append_ops)

def _boundary_scan(word, bound, max_pre, max_suf, min_len):
    L = len(word)
    for p in range(min(max_pre+1, L+1)):
        if p>0 and word[p-1] not in bound: break
        for s in range(min(max_suf+1, L-p+1)):
            if s==0: mid = word[p:]; suf=''
            else: mid = word[p:L-s]; suf = word[L-s:]
            if s>0 and word[L-s] not in bound: break
            if len(mid) >= min_len:
                yield (word[:p], mid, suf)

def _chains_from_middle(mid, pre, suf, base_set, max_depth, min_len, max_leet_amb, lead=None):
    target = pre+mid+suf
    lead = lead or []
    lead_depth = len(lead)
    prepend = [f'^{c}' for c in reversed(pre)]
    append = [f'${c}' for c in suf]
    found = set()
    for stem, leet_rules, case_cand in _decode_middle(mid, max_leet_amb):
        if len(stem) < min_len or stem not in base_set: continue
        leet_ops = sorted(leet_rules)
        for case_ops in case_cand:
            if lead_depth + len(case_ops)+len(leet_ops)+len(prepend)+len(append) > max_depth:
                continue
            for ops in _rule_chain_orderings(case_ops, leet_ops, prepend, append, lead=lead):
                if not ops or len(ops)>max_depth: continue
                chain = ' '.join(ops)
                if not HashcatRuleValidator.validate_rule_for_gpu(chain): continue
                if py_apply_chain(chain, stem) == target:
                    found.add(chain)
    return found

def _extract_letter_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb):
    found = set()
    for pre,mid,suf in _boundary_scan(word, TOKEN_STRIP_BOUNDARY, max_pre, max_suf, min_len):
        found |= _chains_from_middle(mid, pre, suf, base_set, max_depth, min_len, max_amb)
    return found

def _extract_digit_mode(word, base_set, max_depth, min_len, max_pre, max_suf):
    found = set()
    for pre,mid,suf in _boundary_scan(word, TOKEN_STRIP_ALPHA_BOUNDARY, max_pre, max_suf, min_len):
        if not mid.isdigit(): continue
        if mid not in base_set: continue
        prepend = [f'^{c}' for c in reversed(pre)]
        append = [f'${c}' for c in suf]
        if len(prepend)+len(append)==0 or len(prepend)+len(append)>max_depth: continue
        chain = ' '.join(prepend+append)
        if HashcatRuleValidator.validate_rule_for_gpu(chain) and py_apply_chain(chain, mid)==word:
            found.add(chain)
    return found

def _extract_reverse_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb):
    if max_depth<1: return set()
    found = set()
    for pre,mid,suf in _boundary_scan(word, TOKEN_STRIP_BOUNDARY, max_pre, max_suf, min_len):
        rev = mid[::-1]
        found |= _chains_from_middle(rev, pre, suf, base_set, max_depth, min_len, max_amb, lead=['r'])
    return found

def _extract_duplicate_mode(word, base_set, max_depth, min_len):
    found = set()
    L = len(word)
    for op,builder in (('d',lambda s:s+s), ('f',lambda s:s+s[::-1])):
        for h in range(min_len, L//2+1):
            stem = word[:h]
            if builder(stem)==word and stem in base_set:
                if HashcatRuleValidator.validate_rule_for_gpu(op) and py_apply_chain(op, stem)==word:
                    found.add(op)
    return found

def _extract_delete_edge_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb):
    if max_depth<1 or len(word)<min_len+1: return set()
    found = set()
    for op,trim in (('[',word[1:]), (']',word[:-1])):
        for pre,mid,suf in _boundary_scan(trim, TOKEN_STRIP_BOUNDARY, max_pre, max_suf, min_len):
            found |= _chains_from_middle(mid, pre, suf, base_set, max_depth, min_len, max_amb, lead=[op])
    return found

def _extract_insert_mode(word, base_set, max_depth, min_len, max_amb, base_by_len):
    if max_depth<2: return set()
    found = set()
    L = len(word)
    if L-1 >= min_len:
        for pos in range(min(L,10)):
            cand = word[:pos]+word[pos+1:]
            if cand in base_set:
                rule = f"i{pos}{word[pos]}"
                if HashcatRuleValidator.validate_rule_for_gpu(rule) and py_apply_chain(rule, cand)==word:
                    found.add(rule)
    if max_depth>=3 and L-2 >= min_len:
        for i in range(min(L,10)):
            for j in range(i+1, min(L,10)):
                cand = word[:i]+word[i+1:j]+word[j+1:]
                if len(cand) < min_len: continue
                if cand in base_set:
                    # After inserting word[i] at position i, the string grows by 1,
                    # so the second insert position must be j+1 (not j) to land correctly.
                    chain = f"i{i}{word[i]} i{j+1}{word[j]}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain) and py_apply_chain(chain, cand)==word:
                        found.add(chain)
    return found

def _worker_init_p0(base_set=None, base_by_len=None):
    global _p0_worker_base_set, _p0_worker_base_by_len
    if base_set is not None: _p0_worker_base_set = base_set
    if base_by_len is not None: _p0_worker_base_by_len = base_by_len

def _process_chunk_p0(args):
    words, max_depth, min_len, max_amb, max_pre, max_suf = args
    base_set = _p0_worker_base_set  # read-only, no copy needed
    base_by_len = _p0_worker_base_by_len
    found = set()
    for w in words:
        if not w or len(w)>MAX_WORD_LEN: continue
        nd = sum(1 for c in w if c.isdigit())
        na = sum(1 for c in w if c.isalpha())
        if nd > na:
            found |= _extract_digit_mode(w, base_set, max_depth, min_len, max_pre, max_suf)
            if na >= min_len:
                found |= _extract_letter_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
        else:
            found |= _extract_letter_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
            if nd>0:
                found |= _extract_digit_mode(w, base_set, max_depth, min_len, max_pre, max_suf)
        if max_depth >= 2:
            found |= _extract_reverse_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
            found |= _extract_delete_edge_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
        if len(w) >= 2*min_len:
            found |= _extract_duplicate_mode(w, base_set, max_depth, min_len)
        if max_depth >= 2:
            found |= _extract_insert_mode(w, base_set, max_depth, min_len, max_amb, base_by_len)
    return found

def _generate_toggle_chain_seeds(max_depth):
    leet_ops = ['sa@','se3','so0','si1','sl1','ss5','ss$','st7','sa4','si!']
    double_leet = [('se3','si1'),('se3','sl1'),('sa@','so0'),('ss5','so0'),
                   ('si1','so0'),('se3','so0'),('ss$','se3'),('sa4','sl1')]
    seeds = set()
    def _add(ops):
        if not ops or len(ops)>max_depth: return
        chain = ' '.join(ops)
        if HashcatRuleValidator.validate_rule_for_gpu(chain): seeds.add(chain)
    for n in range(min(10,max_depth)):
        togg = [f'T{i}' for i in range(n+1)]
        _add(togg)
        for l in leet_ops:
            _add(togg+[l])
            if n>=1: _add([l]+togg)
        for l1,l2 in double_leet:
            _add(togg+[l1,l2]); _add([l1]+togg+[l2])
    for n in range(1,min(5,max_depth)):
        togg = [f'T{i*2}' for i in range(n+1)]
        for l in leet_ops: _add(togg+[l]); _add([l]+togg)
        for l1,l2 in double_leet: _add(togg+[l1,l2])
    for n in range(1,min(5,max_depth)):
        togg = [f'T{i*2+1}' for i in range(n+1)]
        for l in leet_ops: _add(togg+[l]); _add([l]+togg)
        for l1,l2 in double_leet: _add(togg+[l1,l2])
    for l in leet_ops: _add(['T0',l]); _add([l,'T0'])
    return sorted(seeds)

def extract_token_strip_rules(target_words, base_set, max_depth=0, min_stem_len=4,
                              max_prefix_len=4, max_suffix_len=4, max_leet_ambiguity=3,
                              workers=0, chunk_size=0):
    if max_depth<=0: max_depth=MAX_HASHCAT_CHAIN
    n_workers = safe_worker_count(workers or mp.cpu_count())
    n_words = len(target_words)
    if chunk_size<=0: chunk_size = max(500, n_words//(n_workers*4)+1)
    base_by_len = defaultdict(set)
    for w in base_set: base_by_len[len(w)].add(w)
    base_by_len = dict(base_by_len)
    total_bytes = sum(len(w) for w in target_words)
    DISK_THRESHOLD_BYTES = 150 * 1024 * 1024
    if total_bytes > DISK_THRESHOLD_BYTES:
        log_warn(f"[S0] Target wordlist size {total_bytes/1e6:.0f}MB > {DISK_THRESHOLD_BYTES/1e6:.0f}MB — using single worker & disk buffer")
        n_workers = 1
    global _p0_worker_base_set, _p0_worker_base_by_len
    _p0_worker_base_set = base_set
    _p0_worker_base_by_len = base_by_len
    chunks = [target_words[i:i+chunk_size] for i in range(0,n_words,chunk_size)]
    task_args = [(ch, max_depth, min_stem_len, max_leet_ambiguity, max_prefix_len, max_suffix_len) for ch in chunks]
    log_info(f"[S0]    Workers: {bold(str(n_workers))}  |  chunks: {bold(str(len(chunks)))} × ~{chunk_size}")
    found = set()
    use_fork = hasattr(os,'fork')
    ctx = mp.get_context('fork' if use_fork else 'spawn')
    if use_fork: pool_kw = dict(processes=n_workers, initializer=_worker_init_p0)
    else: pool_kw = dict(processes=n_workers, initializer=_worker_init_p0, initargs=(base_set, base_by_len))
    ncols = shutil.get_terminal_size((80,24)).columns
    with ctx.Pool(**pool_kw) as pool:
        with tqdm(total=n_words, desc=green("  STAGE 0 "), unit="word", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            t0 = time.time()
            words_done = 0
            for task, res in zip(task_args, pool.imap_unordered(_process_chunk_p0, task_args)):
                found |= res
                words_done += len(task[0])
                elapsed = time.time() - t0
                spd = _fmt_speed(words_done / elapsed if elapsed>0 else 0, "words")
                pbar.set_postfix({"rules": cyan(str(len(found))), "spd": green(spd)}, refresh=False)
                pbar.update(len(task[0]))
    _p0_worker_base_set.clear()
    _p0_worker_base_by_len.clear()
    return sorted(found)

def _log_token_strip_stats(n_words, rules, inject_sbd):
    if not rules:
        log_info(f"[S0]    {yellow('0')} rules extracted ({n_words:,} words)")
        return
    depth = defaultdict(int); mode = defaultdict(int)
    for r in rules:
        toks = r.split()
        depth[len(toks)] += 1
        first = toks[0] if toks else ''
        if first == 'r': mode['reverse']+=1
        elif first == 'd': mode['dup']+=1
        elif first == 'f': mode['fold']+=1
        elif first in ('[',']'): mode['del-edge']+=1
        elif first.startswith('T') and len(first)==2: mode['toggle']+=1
        elif first.startswith('i') and len(first)==3: mode['insert']+=1
        elif all(c.isdigit() or c in ('^','$',' ') for c in r) and any(c.isdigit() for c in r): mode['digit-bnd']+=1
        else: mode['letter']+=1
    depth_sum = '  '.join(f"d{d}:{depth[d]:,}" for d in sorted(depth))
    inj = green('→ STAGE S sbd → STAGE 2') if inject_sbd else dim('STAGE S inactive')
    mod_str = '  '.join(f"{k}:{v}" for k,v in sorted(mode.items()) if v)
    log_info(f"[S0]    {bold(green(str(len(rules))))} rules extracted  ({depth_sum})  {inj}")
    if mod_str: log_info(f"[S0]    Mode breakdown : {dim(mod_str)}")

# ----------------------------------------------------------------------
# GPU device helpers
# ----------------------------------------------------------------------
def get_all_devices():
    devs = []
    for plat in cl.get_platforms():
        for dtype in (cl.device_type.GPU, cl.device_type.CPU):
            try:
                for d in plat.get_devices(dtype): devs.append((plat,d))
            except Exception: pass
    return devs

def list_devices():
    devs = get_all_devices()
    if not devs: log_error("No OpenCL devices."); sys.exit(1)
    log_info(f"\n{blue('Available OpenCL devices:')}")
    for i,(p,d) in enumerate(devs):
        try:
            t = cl.device_type.to_string(d.get_info(cl.device_info.TYPE))
        except Exception:
            dtype = d.get_info(cl.device_info.TYPE)
            if dtype == cl.device_type.GPU: t = "GPU"
            elif dtype == cl.device_type.CPU: t = "CPU"
            else: t = "OTHER"
        log_info(f"  {cyan(str(i)+':')} {d.get_info(cl.device_info.NAME)} ({t}) — {p.name}")
    gpu_count = len(get_all_gpu_only_devices())
    if gpu_count > 1:
        log_info(f"\n  {dim('Tip:')} use {cyan('--device all')} to run on all {gpu_count} GPU(s) in parallel, "
                 f"or {cyan('--device 0,1')} to pick specific GPUs")
    print()

def get_all_gpu_only_devices():
    devs = []
    for p in cl.get_platforms():
        try:
            for d in p.get_devices(cl.device_type.GPU):
                devs.append(d)
        except Exception: pass
    return devs

def get_device_by_spec(spec):
    if spec is None: return get_best_gpu_device()
    spec = spec.strip()
    devs = get_all_devices()
    if not devs: raise RuntimeError("No OpenCL devices")
    if spec.isdigit():
        i = int(spec)
        if 0<=i<len(devs): return devs[i][1]
        raise RuntimeError(f"Index {i} out of range")
    lo = spec.lower()
    match = [d for _,d in devs if lo in d.get_info(cl.device_info.NAME).lower()]
    if len(match)==1: return match[0]
    if match: return match[0]
    raise RuntimeError(f"No device matching '{spec}'")

def get_best_gpu_device():
    best = None; best_score = -1
    for p in cl.get_platforms():
        try: devs = p.get_devices(cl.device_type.GPU)
        except Exception: continue
        for d in devs:
            name = d.get_info(cl.device_info.NAME).upper()
            score = (10 if 'NVIDIA' in name or 'AMD' in name else 0)
            score += (5 if 'RTX' in name or 'GTX' in name else 0)
            score += d.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            if score > best_score: best_score = score; best = d
    if best is None:
        for p in cl.get_platforms():
            try: return p.get_devices(cl.device_type.GPU)[0]
            except Exception: pass
    if best is None: raise RuntimeError("No GPU found")
    return best

def estimate_free_vram(device):
    try: return int(device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * VRAM_USAGE_FACTOR)
    except: return 1<<30

def get_max_allocation(device):
    try: return device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    except: return 1<<30

# ----------------------------------------------------------------------
# Bloom size & dynamic parameters (power-of-two & sharding logic)
# ----------------------------------------------------------------------
def get_auto_bloom_mb(free_vram_gb):
    if free_vram_gb >= 8.0: return 512
    elif free_vram_gb >= 4.0: return 256
    else: return 128

def calculate_dynamic_parameters(base_count, target_count, device=None, target_hours=0.5,
                                  bloom_mb_override=None, bloom_no_shard=False):
    if device:
        try:
            mwgs = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            mcu = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            fv = estimate_free_vram(device)
            vgb = fv / 1024**3
            name_up = device.get_info(cl.device_info.NAME).upper()
            lws = max(s for s in POSSIBLE_WORK_GROUP_SIZES if s <= mwgs)
            if 'NVIDIA' in name_up and mcu >= 38: lws = min(512, lws)
            est = LOW_END_COMBOS_PER_SEC if mcu < LOW_END_COMPUTE_UNITS_THRESHOLD else BASELINE_COMBOS_PER_SEC
            log_info(f"[GPU]  CU={mcu}, VRAM~{vgb:.1f}GB, WGS={lws}, est={est//1_000_000}M/s")
        except:
            lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2<<30; vgb = 2.0
    else:
        lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2<<30; vgb = 2.0

    vram_scale = max(0.25, min(1.0, vgb/8.0))
    target_sec = target_hours * 3600
    if bloom_mb_override and bloom_mb_override > 0:
        eff_bloom = bloom_mb_override
    else:
        eff_bloom = get_auto_bloom_mb(vgb)

    total_bits = eff_bloom * 1024 * 1024 * 8
    if total_bits & (total_bits - 1):
        total_bits = 1 << total_bits.bit_length()
        eff_bloom = total_bits // (1024*1024*8)
    SHARD_BITS = 1 << 32
    if not bloom_no_shard and total_bits > SHARD_BITS:
        num_shards = (total_bits + SHARD_BITS - 1) // SHARD_BITS
        shard_bits = SHARD_BITS
        bloom_bits = num_shards * SHARD_BITS
        log_info(f"[BLOOM] Sharding enabled: {eff_bloom}MB -> {num_shards} shards of 512MB, total {bloom_bits//(1024*1024*8)}MB")
    else:
        num_shards = 1
        if total_bits & (total_bits - 1):
            total_bits = 1 << total_bits.bit_length()
            eff_bloom  = total_bits // (1024 * 1024 * 8)
        shard_bits = total_bits
        bloom_bits = total_bits
        log_info(f"[BLOOM] Single shard (fast path), size {bloom_bits//(1024*1024*8)}MB")

    if target_count > 0:
        fill = 1.0 - math.exp(-BLOOM_HASH_FUNCTIONS * target_count / bloom_bits)
        fpr = fill ** BLOOM_HASH_FUNCTIONS
        log_info(f"[BLOOM] {bloom_bits//8//1024//1024}MB, shards={num_shards}, fill={fill:.3%}, FPR={fpr:.6%}")
        if fpr > 0.01: log_warn(f"High FPR {fpr:.3%} — increase --bloom-mb")

    return {
        'BLOOM_FILTER_SIZE': bloom_bits,
        'BLOOM_NUM_SHARDS': num_shards,
        'BLOOM_SHARD_BITS': shard_bits,
        'BLOOM_SHARD_BYTES': shard_bits // 8,
        'WORDS_PER_BATCH': max(1000, int(BASE_WORDS_PER_BATCH * vram_scale)),
        'CHAINS_PER_BATCH': max(500, int(BASE_CHAINS_PER_BATCH * vram_scale)),
        'WORD_SUB_BATCH': max(5000, int(BASE_WORD_SUB_BATCH * vram_scale)),
        'MAX_SAFE_RESULTS_PER_BATCH': min(MAX_SAFE_RESULTS_CAP, max(5000, int(BASE_MAX_SAFE_RESULTS * vram_scale))),
        'MAX_CHAINS_TO_FIND': 2**31-1,
        'LOCAL_WORK_SIZE': lws,
        'OPTIMAL_GLOBAL_MULTIPLIER': mcu * OPTIMAL_GLOBAL_MULTIPLIER_BASE,
        'EST_COMBOS_PER_SEC': est,
        'TARGET_SECONDS': target_sec,
        'vram_scale': vram_scale,
        'free_vram': fv,
    }

# ----------------------------------------------------------------------
# GPU-compatible atomic rules generator
# ----------------------------------------------------------------------
class GPUCompatibleRulesGenerator:
    def __init__(self): self.validator = HashcatRuleValidator()
    def generate_gpu_compatible_rules(self):
        rules = set()
        digits = '0123456789'
        rules.update(['l','u','c','C','t','r','d','f','q','E','{','}','[',']','k','K',':'])
        for cmd in ('T','D','L','R','+','-','.',',',"'",'z','Z','y','Y'):
            for p in POSITION_CHARS: rules.add(f'{cmd}{p}')
        for p in POSITION_CHARS: rules.add(f'p{p}')
        for cmd in ('x','*','O'):
            for p1 in digits:
                for p2 in digits: rules.add(f'{cmd}{p1}{p2}')
        for i in range(33,127):
            ch = chr(i); rules.add(f'^{ch}'); rules.add(f'${ch}'); rules.add(f'@{ch}')
        for orig in string.ascii_lowercase+string.ascii_uppercase:
            for sub in string.digits+string.punctuation:
                if orig != sub: rules.add(f's{orig}{sub}')
        chars = string.ascii_letters+string.digits+'!@#$%^&*()_+-=[]{}|;:,.<>?/~'
        for p in digits:
            for ch in chars: rules.add(f'i{p}{ch}'); rules.add(f'o{p}{ch}')
        for n in range(1,10):
            for ch in ('p','y','Y','z','Z'): rules.add(f'{ch}{n}')
        for sep in '-_.,;:|/\\+*&^%$#@!~`': rules.add(f'e{sep}')
        chars2 = string.ascii_letters+string.digits+'!@#$%^&*()_+-=[]{}|;:,.<>?/~'
        for n in digits:
            for sep in chars2: rules.add(f'3{n}{sep}')
        valid = [r for r in rules if self.validator.validate_rule_for_gpu(r) and 1<=len(r)<=MAX_RULE_LEN]
        log_info(f"Generated {len(valid):,} atomic rules")
        return valid

# ----------------------------------------------------------------------
# OpenCL kernel template (unchanged – full version)
# ----------------------------------------------------------------------
GPU_KERNEL_TEMPLATE = r"""
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define MAX_WORD_LEN         {MAX_WORD_LEN}
#define MAX_RULE_LEN         {MAX_RULE_LEN}
#define MAX_OUTPUT_LEN       {MAX_OUTPUT_LEN}
#define MAX_CHAIN_STRING_LEN {MAX_CHAIN_STRING_LEN}
#define MAX_CHAINS_TO_FIND   {MAX_SAFE_RESULTS_PER_BATCH}
#define MAX_CHAIN_DEPTH      {MAX_CHAIN_DEPTH}
#define BLOOM_HASH_FUNCTIONS {BLOOM_HASH_FUNCTIONS}
#define BLOOM_NUM_SHARDS     {BLOOM_NUM_SHARDS}
#define BLOOM_SHARD_BITS     {BLOOM_SHARD_BITS}
#define BLOOM_SHARD_BYTES    {BLOOM_SHARD_BYTES}

inline uint bloom_shard_idx(uint h1, uint h2, uint i, uint shard_bits) {{
    return (h1 + i * h2) & (shard_bits - 1);
}}

int bloom(__global const uchar *bf, const unsigned char *w, int len) {{
    for (int s = 0; s < BLOOM_NUM_SHARDS; s++) {{
        uint seed1 = 0xDEADBEEFu + (uint)s;
        uint seed2 = 0xCAFEBABEu + (uint)s;
        uint h1 = seed1 ^ 2166136261U;
        uint h2 = seed2 ^ 2166136261U;
        for (int i = 0; i < len; i++) {{
            h1 ^= w[i]; h1 *= 16777619U;
            h2 ^= w[i]; h2 *= 16777619U;
        }}
        for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {{
            uint idx = bloom_shard_idx(h1, h2, i, BLOOM_SHARD_BITS);
            uint byte_off = s * BLOOM_SHARD_BYTES + (idx >> 3);
            if (!(bf[byte_off] & (1 << (idx & 7)))) return 0;
        }}
    }}
    return 1;
}}

// Decodes hashcat's position/count encoding: '0'-'9' -> 0-9, 'A'-'Z' -> 10-35, else -1.
inline int dpos(unsigned char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return -1;
}}

int apply(const unsigned char *rs, int rl,
          const unsigned char *in, int il,
          unsigned char *out, int *ol) {{
    *ol = il; for (int i = 0; i < il; i++) out[i] = in[i]; out[il] = '\0';
    if (!rl || !il) return 1;
    unsigned char cmd = rs[0]; int changed = 0;
    if (rl == 1) {{
        switch (cmd) {{
        case 'l': for (int i = 0; i < *ol; i++) out[i] = (out[i] >= 'A' && out[i] <= 'Z') ? out[i] + 32 : out[i]; changed = 1; break;
        case 'u': for (int i = 0; i < *ol; i++) out[i] = (out[i] >= 'a' && out[i] <= 'z') ? out[i] - 32 : out[i]; changed = 1; break;
        case 'c': if (*ol > 0) {{
            out[0] = (out[0] >= 'a' && out[0] <= 'z') ? out[0] - 32 : out[0];
            for (int i = 1; i < *ol; i++) out[i] = (out[i] >= 'A' && out[i] <= 'Z') ? out[i] + 32 : out[i];
        }} changed = 1; break;
        case 'C': if (*ol > 0) {{
            out[0] = (out[0] >= 'A' && out[0] <= 'Z') ? out[0] + 32 : out[0];
            for (int i = 1; i < *ol; i++) out[i] = (out[i] >= 'a' && out[i] <= 'z') ? out[i] - 32 : out[i];
        }} changed = 1; break;
        case 't': for (int i = 0; i < *ol; i++) {{
            if (out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
            else if (out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
        }} changed = 1; break;
        case 'r': for (int i = 0; i < *ol/2; i++) {{
            unsigned char t = out[i]; out[i] = out[*ol-1-i]; out[*ol-1-i] = t;
        }} changed = 1; break;
        case 'd': if (*ol*2 <= MAX_OUTPUT_LEN) {{
            for (int i = 0; i < *ol; i++) out[*ol+i] = out[i]; *ol *= 2;
        }} changed = 1; break;
        case 'f': if (*ol*2 <= MAX_OUTPUT_LEN) {{
            for (int i = 0; i < *ol; i++) out[*ol+i] = out[*ol-1-i]; *ol *= 2;
        }} changed = 1; break;
        case '{{': if (*ol > 1) {{
            unsigned char f = out[0];
            for (int i = 0; i < *ol-1; i++) out[i] = out[i+1];
            out[*ol-1] = f;
        }} changed = 1; break;
        case '}}': if (*ol > 1) {{
            unsigned char l = out[*ol-1];
            for (int i = *ol-1; i > 0; i--) out[i] = out[i-1];
            out[0] = l;
        }} changed = 1; break;
        case '[': if (*ol > 0) {{
            for (int i = 0; i < *ol-1; i++) out[i] = out[i+1]; (*ol)--;
        }} changed = 1; break;
        case ']': if (*ol > 0) (*ol)--; changed = 1; break;
        case 'k': if (*ol >= 2) {{
            unsigned char t = out[0]; out[0] = out[1]; out[1] = t;
        }} changed = 1; break;
        case 'K': if (*ol >= 2) {{
            unsigned char t = out[*ol-2]; out[*ol-2] = out[*ol-1]; out[*ol-1] = t;
        }} changed = 1; break;
        case ':': break;
        case 'q': if (*ol*2 <= MAX_OUTPUT_LEN) {{
            unsigned char tmp[MAX_OUTPUT_LEN];
            for (int i = 0; i < *ol; i++) tmp[i] = out[i];
            int idx = 0;
            for (int i = 0; i < *ol; i++) {{
                out[idx++] = tmp[i]; out[idx++] = tmp[i];
            }} *ol *= 2;
        }} changed = 1; break;
        case 'E': {{
            int cap = 1;
            for (int i = 0; i < *ol; i++) {{
                if (cap && out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                else if (!cap && out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                cap = (out[i] == ' ');
            }} changed = 1;
        }} break;
        }}
    }} else if (rl == 2) {{
        unsigned char p = rs[1];
        int pv = dpos(p);
        if (cmd == '^') {{
            if (*ol+1 <= MAX_OUTPUT_LEN) {{
                for (int i = *ol; i > 0; i--) out[i] = out[i-1];
                out[0] = p; (*ol)++;
            }} changed = 1;
        }} else if (cmd == '$') {{
            if (*ol+1 <= MAX_OUTPUT_LEN) {{
                out[*ol] = p; (*ol)++;
            }} changed = 1;
        }} else if (cmd == '@') {{
            int nl = 0;
            for (int i = 0; i < *ol; i++) if (out[i] != p) out[nl++] = out[i];
            *ol = nl; changed = 1;
        }} else if (cmd == 'p' && pv >= 0) {{
            int n = pv;
            if (n > 0 && *ol*(n+1) <= MAX_OUTPUT_LEN) {{
                int o = *ol;
                for (int r = 0; r < n; r++) {{
                    for (int i = 0; i < o; i++) out[*ol+i] = out[i];
                    *ol += o;
                }} changed = 1;
            }}
        }} else if (cmd == 'T' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{
                if (out[pos] >= 'a' && out[pos] <= 'z') out[pos] -= 32;
                else if (out[pos] >= 'A' && out[pos] <= 'Z') out[pos] += 32;
                changed = 1;
            }}
        }} else if (cmd == 'D' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{
                for (int i = pos; i < *ol-1; i++) out[i] = out[i+1];
                (*ol)--; changed = 1;
            }}
        }} else if (cmd == 'L' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{ out[pos] <<= 1; changed = 1; }}
        }} else if (cmd == 'R' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{ out[pos] >>= 1; changed = 1; }}
        }} else if (cmd == '+' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol && out[pos] < 255) {{ out[pos]++; changed = 1; }}
        }} else if (cmd == '-' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol && out[pos] > 0) {{ out[pos]--; changed = 1; }}
        }} else if ((cmd == '.' || cmd == ',') && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{
                out[pos] += (cmd == '.') ? 1 : -1; changed = 1;
            }}
        }} else if (cmd == '\'' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{ *ol = pos; changed = 1; }}
        }} else if (cmd == 'z' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                unsigned char f = out[0];
                for (int i = *ol+n-1; i >= n; i--) out[i] = out[i-n];
                for (int i = 0; i < n; i++) out[i] = f;
                *ol += n; changed = 1;
            }}
        }} else if (cmd == 'Z' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                unsigned char l = out[*ol-1];
                for (int i = 0; i < n; i++) out[*ol+i] = l;
                *ol += n; changed = 1;
            }}
        }} else if (cmd == 'y' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                for (int i = 0; i < n; i++) out[*ol+i] = out[i];
                *ol += n; changed = 1;
            }}
        }} else if (cmd == 'Y' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                for (int i = 0; i < n; i++) out[*ol+i] = out[*ol-n+i];
                *ol += n; changed = 1;
            }}
        }}
    }} else if (rl == 3) {{
        unsigned char p1 = rs[1], p2 = rs[2];
        if (cmd == 's') {{
            for (int i = 0; i < *ol; i++) if (out[i] == p1) {{ out[i] = p2; changed = 1; }}
        }} else if (cmd == 'i' && p1>='0' && p1<='9') {{
            int pos = p1-'0';
            if (pos <= *ol && *ol+1 <= MAX_OUTPUT_LEN) {{
                for (int i = *ol; i > pos; i--) out[i] = out[i-1];
                out[pos] = p2; (*ol)++; changed = 1;
            }}
        }} else if (cmd == 'o' && p1>='0' && p1<='9') {{
            int pos = p1-'0';
            if (pos < *ol) {{ out[pos] = p2; changed = 1; }}
        }} else if (cmd == 'e') {{
            int cap = 1;
            for (int i = 0; i < *ol; i++) {{
                if (cap && out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                else if (!cap && out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                cap = (out[i] == p1);
            }} changed = 1;
        }} else if (cmd == 'x' && p1>='0' && p1<='9' && p2>='0' && p2<='9') {{
            int a = p1-'0', b = p2-'0';
            if (a >= 0 && b >= 0 && a < *ol) {{
                int nl = 0;
                int end = a + b;
                for (int i = a; i < end && i < *ol; i++) out[nl++] = out[i];
                *ol = nl; changed = 1;
            }}
        }} else if (cmd == 'O' && p1>='0' && p1<='9' && p2>='0' && p2<='9') {{
            int n = p1-'0', m = p2-'0';
            if (n < *ol && m > 0) {{
                int e = n+m; if (e > *ol) e = *ol;
                int sh = e - n;
                for (int i = e; i < *ol; i++) out[i-sh] = out[i];
                *ol -= sh; changed = 1;
            }}
        }} else if (cmd == '*' && p1>='0' && p1<='9' && p2>='0' && p2<='9') {{
            int a = p1-'0', b = p2-'0';
            if (a < *ol && b < *ol && a != b) {{
                unsigned char t = out[a]; out[a] = out[b]; out[b] = t; changed = 1;
            }}
        }} else if (cmd == '3' && p1>='0' && p1<='9') {{
            int n = p1-'0', cnt = 0, found = -1;
            for (int i = 0; i < *ol; i++) if (out[i] == p2 && ++cnt == n + 1) {{ found = i; break; }}
            if (found != -1 && found+1 < *ol) {{
                if (out[found+1] >= 'a' && out[found+1] <= 'z') out[found+1] -= 32;
                else if (out[found+1] >= 'A' && out[found+1] <= 'Z') out[found+1] += 32;
                changed = 1;
            }}
        }}
    }}
    out[*ol] = '\0';
    return changed ? 1 : 0;
}}

__kernel void find_single_rules_gpu(
    __global const unsigned char *bw, __global const int *bo, __global const int *bl,
    __global const unsigned char *rs, __global const int *ro, __global const int *rl,
    __global const uchar *bf, const int nw, const int nr,
    __global char *found, __global volatile int *cnt)
{{
    int gid = get_global_id(0);
    if (gid >= nw * nr) return;
    int wi = gid / nr, ri = gid % nr;
    unsigned char iw[MAX_WORD_LEN], ow[MAX_OUTPUT_LEN], rr[MAX_RULE_LEN];
    int wl = bl[wi]; for (int i = 0; i < wl; i++) iw[i] = bw[bo[wi] + i]; iw[wl] = '\0';
    int rlen = rl[ri]; for (int i = 0; i < rlen; i++) rr[i] = rs[ro[ri] + i]; rr[rlen] = '\0';
    int ol; apply(rr, rlen, iw, wl, ow, &ol);
    int same = (ol == wl); for (int i = 0; i < wl && same; i++) if (ow[i] != iw[i]) same = 0;
    if (!same && ol > 0 && bloom(bf, ow, ol)) {{
        int idx = atomic_inc(cnt);
        if (idx < MAX_CHAINS_TO_FIND) {{
            __global char *p = found + idx * MAX_CHAIN_STRING_LEN;
            for (int i = 0; i < rlen && i < MAX_CHAIN_STRING_LEN-1; i++) p[i] = rr[i];
            p[rlen] = '\0';
        }}
    }}
}}

__kernel void find_rule_chains_gpu(
    __global const unsigned char *bw, __global const int *bo, __global const int *bl,
    __global const unsigned char *rs, __global const int *ro, __global const int *rl,
    __global const int *cseq, __global const int *cdep,
    __global const uchar *bf, const int nw, const int nc, const int mcd,
    __global char *found, __global volatile int *cnt, __global int *foundw)
{{
    int gid = get_global_id(0);
    if (gid >= nw * nc) return;
    int wi = gid / nc, ci = gid % nc;
    unsigned char cur[MAX_OUTPUT_LEN], tmp[MAX_OUTPUT_LEN], rr[MAX_RULE_LEN];
    char cb[MAX_CHAIN_STRING_LEN]; int cp = 0;
    int wl = bl[wi]; for (int i = 0; i < wl; i++) cur[i] = bw[bo[wi] + i]; cur[wl] = '\0';
    int cl_ = wl, dep = cdep[ci]; if (dep < 1 || dep > mcd) return;
    __global const unsigned char *wp = bw + bo[wi];
    for (int d = 0; d < dep; d++) {{
        int ri = cseq[ci * mcd + d]; if (ri < 0) break;
        int rlen = rl[ri]; for (int i = 0; i < rlen; i++) rr[i] = rs[ro[ri] + i]; rr[rlen] = '\0';
        for (int i = 0; i < rlen && cp < MAX_CHAIN_STRING_LEN-2; i++) cb[cp++] = rr[i];
        if (d < dep-1 && cp < MAX_CHAIN_STRING_LEN-1) cb[cp++] = ' ';
        int nl; apply(rr, rlen, cur, cl_, tmp, &nl);
        if (nl == 0) return;
        for (int i = 0; i < nl; i++) cur[i] = tmp[i]; cur[nl] = '\0'; cl_ = nl;
    }}
    cb[cp] = '\0';
    int same = (cl_ == wl); for (int i = 0; i < wl && same; i++) if (cur[i] != wp[i]) same = 0;
    if (!same && bloom(bf, cur, cl_)) {{
        int idx = atomic_inc(cnt);
        if (idx < MAX_CHAINS_TO_FIND) {{
            __global char *p = found + idx * MAX_CHAIN_STRING_LEN;
            for (int i = 0; i < cp && i < MAX_CHAIN_STRING_LEN-1; i++) p[i] = cb[i];
            p[cp] = '\0';
            foundw[idx] = wi;
        }}
    }}
}}

__kernel void build_bloom_filter_gpu(
    __global const unsigned char *bw, __global const int *bo, __global const int *bl,
    const int nw, volatile __global int *bf)
{{
    int gid = get_global_id(0);
    if (gid >= nw) return;
    int wlen = bl[gid];
    __global const unsigned char *word = bw + bo[gid];
    for (int s = 0; s < BLOOM_NUM_SHARDS; s++) {{
        uint seed1 = 0xDEADBEEFu + (uint)s;
        uint seed2 = 0xCAFEBABEu + (uint)s;
        uint h1 = seed1 ^ 2166136261U;
        uint h2 = seed2 ^ 2166136261U;
        for (int i = 0; i < wlen; i++) {{
            h1 ^= word[i]; h1 *= 16777619U;
            h2 ^= word[i]; h2 *= 16777619U;
        }}
        for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {{
            uint idx = (h1 + i * h2) & (BLOOM_SHARD_BITS - 1);
            uint widx = (s * (BLOOM_SHARD_BYTES / 4)) + (idx >> 5);
            int mask = 1 << (idx & 31);
            atomic_or(bf + widx, mask);
        }}
    }}
}}
"""

_BLOOM_ALREADY_ON_GPU = object()

class _KernelBuildTimeout(Exception):
    """Raised by GPUEngine._safe_build_program when the OpenCL driver's
    build() call doesn't return within the watchdog timeout."""
    pass

# ----------------------------------------------------------------------
# GPU Engine (with sharding, fast bloom, and fixed depth handling)
# ----------------------------------------------------------------------
class GPUEngine:
    def __init__(self, params):
        self.params = params
        self.context = self.queue = self.device = self.program = None
        self.max_work_group_size = 512
        self.local_work_size = params.get('LOCAL_WORK_SIZE', 512)
        self.bloom_buf = None
        self.bloom_np = None
        self.rule_index = {}
        self.gpu_rules = []
        self.kernel_single = None
        self.kernel_chain = None
        self.kernel_bloom = None
        self._consecutive_errors = 0
        self._MAX_CONSECUTIVE_ERRORS = 5
        self._cached_free_vram = None
        self._cached_max_alloc = None
        self.disabled = False  # set True if a kernel build hangs/times out — this
                                # device's driver may be stuck holding an internal
                                # lock, so we stop issuing further work to it rather
                                # than risk every subsequent call blocking forever.
        self._program_cache = {}  # MAX_CHAIN_DEPTH -> (program, kernel_single, kernel_chain, kernel_bloom)
        self._rules_buf = None
        self._rules_offsets_buf = None
        self._rules_lengths_buf = None
        self._rules_buf_key = None
        self._bloom_recovery_fn = None
        self._compiled_depth = None   # track compiled MAX_CHAIN_DEPTH
        self._base_words_precomputed = None  # (wf, wo, wl)

    def get_free_vram(self): return estimate_free_vram(self.device)
    def get_max_allocation(self): return get_max_allocation(self.device)

    def _refresh_cached_limits(self):
        self._cached_free_vram = estimate_free_vram(self.device)
        self._cached_max_alloc = get_max_allocation(self.device)

    def safe_output_buffer_size(self, words_count, chains_count):
        fv = self._cached_free_vram or self.get_free_vram()
        ma = self._cached_max_alloc or self.get_max_allocation()
        avail = min(fv, ma) - 5*1024**2
        # Must match the kernel constant MAX_CHAINS_TO_FIND = MAX_SAFE_RESULTS_PER_BATCH;
        # do NOT cap below that value or the kernel will write past the end of fo.
        return max(1, min(avail//MAX_CHAIN_STRING_LEN,
                          self.params['MAX_SAFE_RESULTS_PER_BATCH'],
                          words_count*chains_count))

    def initialize_gpu(self, device_spec):
        try:
            self.device = get_device_by_spec(device_spec)
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            self.max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            self.local_work_size = min(self.local_work_size, self.max_work_group_size)
            while self.max_work_group_size % self.local_work_size != 0 and self.local_work_size > 32:
                self.local_work_size //= 2
            self._refresh_cached_limits()
            log_info(f"[GPU]  {bold(self.device.name.strip())}")
            vgb = self._cached_free_vram/1024**3
            log_debug(f"       WGS={self.local_work_size}, VRAM~{vgb:.1f}GB, CU={self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)}")
            return True
        except Exception as e: log_error(f"GPU init failed: {e}"); return False

    def compile_kernel(self, force=False):
        current_depth = self.params.get('MAX_CHAIN_DEPTH', 2)
        if not force and self.program is not None and self._compiled_depth == current_depth:
            log_debug("kernel already compiled with same depth")
            return self.program
        # Reuse a previously-built program for this exact depth if we have
        # one cached. The pipeline flips MAX_CHAIN_DEPTH back and forth
        # (e.g. Stage2 depth -> Seed-pass depth -> back to Stage2 depth), so
        # this avoids re-triggering a slow/flaky driver compile — a real fix
        # for the "hang after seed pass" case, since the Stage 2 recompile is
        # very often just reusing what was already built at pipeline start.
        cached = self._program_cache.get(current_depth)
        if cached is not None:
            self.program, self.kernel_single, self.kernel_chain, self.kernel_bloom = cached
            self._compiled_depth = current_depth
            log_info(f"[GPU]  Reusing cached kernel for depth {current_depth} (skipping rebuild)")
            return self.program
        if self.program:
            log_info(f"[GPU]  Recompiling kernel for depth {current_depth} (was {self._compiled_depth})")
        else:
            log_info("[GPU]  Compiling kernel ...")
        try:
            src = GPU_KERNEL_TEMPLATE.format(
                BLOOM_FILTER_SIZE=self.params['BLOOM_FILTER_SIZE'],
                BLOOM_NUM_SHARDS=self.params['BLOOM_NUM_SHARDS'],
                BLOOM_SHARD_BITS=self.params['BLOOM_SHARD_BITS'],
                BLOOM_SHARD_BYTES=self.params['BLOOM_SHARD_BYTES'],
                MAX_SAFE_RESULTS_PER_BATCH=self.params['MAX_SAFE_RESULTS_PER_BATCH'],
                MAX_CHAIN_DEPTH=current_depth,
                MAX_CHAIN_STRING_LEN=MAX_CHAIN_STRING_LEN,
                MAX_WORD_LEN=MAX_WORD_LEN,
                MAX_RULE_LEN=MAX_RULE_LEN,
                MAX_OUTPUT_LEN=MAX_OUTPUT_LEN,
                BLOOM_HASH_FUNCTIONS=BLOOM_HASH_FUNCTIONS,
            )
        except Exception as e:
            log_error(f"Kernel compile failed: {e}")
            return None

        dev_name = self.device.name.strip() if self.device else "device"
        prog = None
        try:
            prog = self._safe_build_program(src, timeout=180)
        except _KernelBuildTimeout:
            # Many Intel NEO / iGPU driver hangs are caused by the optimizer
            # choking on a particular kernel rather than the device being
            # truly broken. Give it one real second chance on a brand-new
            # context (the old one may still have a build stuck inside it)
            # with the optimizer disabled — this is a well-known practical
            # workaround for exactly this class of hang.
            log_warn(f"[GPU]  Build stalled on {dev_name} — retrying once on a fresh context "
                     f"with the optimizer disabled (-cl-opt-disable)")
            if self._recreate_context():
                try:
                    prog = self._safe_build_program(src, timeout=120, build_options=['-cl-opt-disable'])
                except _KernelBuildTimeout:
                    prog = None
                except Exception as e:
                    log_error(f"[GPU]  Retry build failed on {dev_name}: {e}")
                    prog = None
            else:
                prog = None
        except Exception as e:
            log_error(f"Kernel compile failed: {e}")
            return None

        if prog is None:
            log_error(f"[GPU]  Disabling {dev_name} after repeated build failure/hang")
            self.disabled = True
            self.context = None
            self.queue = None
            self.program = self.kernel_single = self.kernel_chain = self.kernel_bloom = None
            return None

        self.program = prog
        self.kernel_single = prog.find_single_rules_gpu
        self.kernel_chain = prog.find_rule_chains_gpu
        self.kernel_bloom = prog.build_bloom_filter_gpu
        self._compiled_depth = current_depth
        self._program_cache[current_depth] = (prog, self.kernel_single, self.kernel_chain, self.kernel_bloom)
        log_info("[GPU]  Kernel compiled successfully")
        return prog

    def _recreate_context(self):
        """Create a fresh OpenCL context/queue on the same device, abandoning
        a previous one that may still have a build stuck inside it. Clears
        buffer/kernel state tied to the old context since it's no longer
        valid to use."""
        try:
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            for buf in [self.bloom_buf, self._rules_buf, self._rules_offsets_buf, self._rules_lengths_buf]:
                if buf is not None:
                    try: buf.release()
                    except Exception: pass
            self.bloom_buf = self._rules_buf = self._rules_offsets_buf = self._rules_lengths_buf = None
            self._rules_buf_key = None
            self.gpu_rules = []
            self.rule_index = {}
            self.program = self.kernel_single = self.kernel_chain = self.kernel_bloom = None
            self._program_cache = {}  # programs from the old context are invalid
            return True
        except Exception as e:
            log_error(f"[GPU]  Failed to create fresh context for retry: {e}")
            return False

    def _reset_gpu(self, error):
        log_warn(f"[GPU] Fatal kernel error: {error} — resetting context")
        # Release all buffers
        for buf in [self.bloom_buf, self._rules_buf, self._rules_offsets_buf, self._rules_lengths_buf]:
            if buf is not None:
                try: buf.release()
                except Exception: pass
        if self.queue is not None:
            try: self.queue.finish()
            except Exception: pass
        # Clear state
        self.bloom_buf = self._rules_buf = self._rules_offsets_buf = self._rules_lengths_buf = None
        self._rules_buf_key = None  # force re-upload of rules after reset
        self.program = self.kernel_single = self.kernel_chain = self.kernel_bloom = None
        self._program_cache = {}  # programs from the old context are invalid
        self.context = None
        self.queue = None
        self.gpu_rules = []
        self.rule_index = {}
        self._rules_buf_key = None
        # Recreate — NVIDIA OpenCL on Windows needs ~1-2 s to exit TDR/error state
        for attempt in range(3):
            try:
                time.sleep(1.5)
                self.context = cl.Context([self.device])
                self.queue = cl.CommandQueue(self.context)
                if not self.compile_kernel(force=True):
                    return False
                if self._bloom_recovery_fn:
                    bf = self._bloom_recovery_fn()
                    self.upload_bloom_filter(bf)
                elif self.bloom_np is not None:
                    self.upload_bloom_filter(self.bloom_np)
                return True
            except Exception as exc:
                log_error(f"Reset failed (attempt {attempt+1}/3): {exc}")
                self.context = None
                self.queue = None
        return False

    def _safe_queue_finish(self):
        if self.queue is None: return False
        res = []
        def fin():
            try:
                self.queue.finish()
                res.append(True)
            except Exception:
                res.append(False)
        t = threading.Thread(target=fin, daemon=True); t.start()
        t.join(timeout=90)
        if t.is_alive(): return False
        return bool(res) and res[0]

    def _safe_build_program(self, src, timeout=180, build_options=None):
        """Build an OpenCL program on a watchdog thread so a driver-level
        compile hang (seen on some multi-GPU / laptop-hybrid setups) can't
        freeze the whole run forever. Mirrors _safe_queue_finish's pattern:
        if the build thread is still alive after `timeout` seconds, we give
        up on it (the thread is left as a daemon and abandoned — OpenCL
        offers no safe way to cancel an in-flight build) and return None so
        the caller can fail gracefully instead of hanging at "100%"."""
        prog = cl.Program(self.context, src)
        result = {}
        def do_build():
            try:
                prog.build(options=build_options or [])
                result['ok'] = True
            except Exception as e:
                result['ok'] = False
                result['err'] = e
        t = threading.Thread(target=do_build, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise _KernelBuildTimeout(f"build did not return within {timeout}s")
        if not result.get('ok'):
            raise result.get('err', RuntimeError("kernel build failed"))
        return prog

    def generate_bloom_filter(self, target_words):
        total_bytes = self.params['BLOOM_NUM_SHARDS'] * self.params['BLOOM_SHARD_BYTES']
        bf = np.zeros(total_bytes, dtype=np.uint8)
        log_info(f"[BLOOM] CPU build: {total_bytes//1024//1024}MB, {len(target_words):,} words, {self.params['BLOOM_NUM_SHARDS']} shard(s)")
        shard_bits = self.params['BLOOM_SHARD_BITS']
        shard_bytes = self.params['BLOOM_SHARD_BYTES']
        ns = self.params['BLOOM_NUM_SHARDS']
        ncols = shutil.get_terminal_size((80,24)).columns
        for w in tqdm(target_words, desc=green("  Bloom filter"), unit="word", ncols=ncols, leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
            wb = w.encode('latin-1')
            for s in range(ns):
                seed1 = (FNV1A_SEED1 + s) & 0xFFFFFFFF
                seed2 = (FNV1A_SEED2 + s) & 0xFFFFFFFF
                h1 = fnv1a_32(wb, seed1); h2 = fnv1a_32(wb, seed2)
                for i in range(BLOOM_HASH_FUNCTIONS):
                    idx = (h1 + i*h2) % shard_bits
                    byte_off = s*shard_bytes + (idx//8)
                    bf[byte_off] |= 1 << (idx%8)
        return bf

    def generate_bloom_filter_gpu(self, target_words):
        ns = self.params['BLOOM_NUM_SHARDS']
        shard_bytes = self.params['BLOOM_SHARD_BYTES']
        total_bytes = ns * shard_bytes
        nw = len(target_words)
        if not self.context or not self.program or not self.kernel_bloom:
            reason = ("no GPU context" if not self.context else
                      "kernel not compiled" if not self.program else
                      "bloom kernel missing after compile")
            log_warn(f"[BLOOM] GPU not ready ({reason}) — falling back to CPU build")
            return self.generate_bloom_filter(target_words)
        log_info(f"[BLOOM] GPU build: {total_bytes//1024//1024}MB, {nw} words, {ns} shards")
        mf = cl.mem_flags
        bufs = []
        try:
            wf, wo, wl = self._flatten(target_words)
            words_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=wf)
            offsets_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=wo)
            lengths_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=wl)
            bufs.extend([words_buf, offsets_buf, lengths_buf])
            bf_int32_size = (total_bytes + 3)//4
            bf_init = np.zeros(bf_int32_size, dtype=np.int32)
            bf_buf = cl.Buffer(self.context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=bf_init)
            bufs.append(bf_buf)  # register for cleanup on exception; removed below on success
            gs = ((nw + self.local_work_size - 1) // self.local_work_size) * self.local_work_size
            self.kernel_bloom.set_args(words_buf, offsets_buf, lengths_buf, np.int32(nw), bf_buf)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_bloom, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish(): raise RuntimeError("queue finish timeout")
            # Verify the GPU actually wrote something non-trivial before trusting
            # this buffer. A bloom filter can legitimately have a very low fill
            # ratio (e.g. fill=0.001% on a large --bloom-mb with a small
            # wordlist), in which case the set bits are sparse and can easily
            # fall entirely outside a small sampled region — sampling only the
            # first few KB of a 256MB+ buffer produced false "all-zero"
            # positives and silently discarded valid GPU results every time.
            # Downloading the full buffer is a one-off cost (once per bloom
            # build, not per batch) and gives an exact answer instead of a
            # statistical guess.
            verify = np.empty(bf_int32_size, dtype=np.int32)
            cl.enqueue_copy(self.queue, verify, bf_buf)
            all_zero = np.count_nonzero(verify) == 0
            del verify
            if all_zero and nw > 0:
                log_warn("[BLOOM] GPU build all-zero -> fallback to CPU")
                raise RuntimeError("all-zero")
            if self.bloom_buf: self.bloom_buf.release()
            bufs.remove(bf_buf)  # hand ownership to self.bloom_buf; don't release in finally
            self.bloom_buf = bf_buf
            self.bloom_np = None
            self._bloom_recovery_fn = lambda: self.generate_bloom_filter(target_words)
            return _BLOOM_ALREADY_ON_GPU
        except Exception as exc:
            log_warn(f"[BLOOM] GPU build failed ({exc}) -> CPU")
            return self.generate_bloom_filter(target_words)
        finally:
            for b in bufs: b.release()

    def upload_bloom_filter(self, bf):
        if bf is _BLOOM_ALREADY_ON_GPU: return
        if self.bloom_buf: self.bloom_buf.release()
        self.bloom_np = bf
        self.bloom_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=bf)

    def _flatten(self, items):
        if not items:
            return (np.array([], dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        enc = [x.encode('latin-1') for x in items]
        flat = b''.join(enc)
        offs = [0]; lens = [len(enc[0])]
        for b in enc[1:]: offs.append(offs[-1]+lens[-1]); lens.append(len(b))
        return (np.frombuffer(flat, dtype=np.uint8), np.array(offs, dtype=np.int32), np.array(lens, dtype=np.int32))

    def prepare_batch_data(self, words, rules):
        wf,wo,wl = self._flatten(words)
        rf,ro,rl = self._flatten(rules)
        return dict(words_flat=wf, word_offsets=wo, word_lengths=wl,
                    rules_flat=rf, rule_offsets=ro, rule_lengths=rl,
                    num_words=len(words), num_rules=len(rules))

    def prepare_words_data(self, words):
        wf,wo,wl = self._flatten(words)
        return dict(words_flat=wf, word_offsets=wo, word_lengths=wl, num_words=len(words), num_rules=len(self.gpu_rules))

    def _get_rules_buffers(self, mf):
        # Use a tuple of rules as key — immune to Python id() reuse after list reassignment
        key = tuple(self.gpu_rules)
        if self._rules_buf_key != key:
            for attr in ('_rules_buf','_rules_offsets_buf','_rules_lengths_buf'):
                if getattr(self, attr): getattr(self, attr).release()
            rf,ro,rl = self._flatten(self.gpu_rules)
            self._rules_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=rf)
            self._rules_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=ro)
            self._rules_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=rl)
            self._rules_buf_key = key
        return self._rules_buf, self._rules_offsets_buf, self._rules_lengths_buf

    def set_base_words(self, words):
        self._base_words_precomputed = self._flatten(words)

    def get_word_batch_data(self, start, end):
        if self._base_words_precomputed is None:
            raise RuntimeError("Base words not precomputed — call set_base_words() first")
        full_wf, full_wo, full_wl = self._base_words_precomputed
        if start >= end or start >= len(full_wo):
            return dict(words_flat=np.array([], dtype=np.uint8),
                        word_offsets=np.array([], dtype=np.int32),
                        word_lengths=np.array([], dtype=np.int32),
                        num_words=0, num_rules=len(self.gpu_rules))
        end = min(end, len(full_wo))
        base_off = int(full_wo[start])
        end_off = int(full_wo[end-1]) + int(full_wl[end-1])
        wf = full_wf[base_off:end_off]
        wo = full_wo[start:end] - base_off
        wl = full_wl[start:end]
        return dict(words_flat=wf, word_offsets=wo, word_lengths=wl,
                    num_words=end-start, num_rules=len(self.gpu_rules))

    def process_all_words_single_rule(self, base_words, rules, bloom_filter, _silent=False,
                                       _shared_combos=None, _shared_lock=None):
        self.upload_bloom_filter(bloom_filter)
        if not self.compile_kernel(): return Counter()
        # Refresh GPU rules list
        self.gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(rules)
        self.rule_index = {r:i for i,r in enumerate(self.gpu_rules)}
        counter = Counter()
        bs = self.params['WORDS_PER_BATCH']
        ncols = shutil.get_terminal_size((80,24)).columns
        t0 = time.time(); words_done = 0; nr = len(self.gpu_rules)
        with tqdm(total=len(base_words), desc=green("  STAGE 1 "), unit="word", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  disable=_silent) as pbar:
            for i in range(0, len(base_words), bs):
                if _kb.quit_requested: break
                batch = base_words[i:i+bs]
                if batch:
                    found = self._run_single_kernel(self.prepare_words_data(batch))
                    if found:
                        self._consecutive_errors = 0
                        counter.update(found)
                    elif self.queue is None:
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS: break
                    words_done += len(batch)
                    elapsed = time.time()-t0
                    delta = len(batch) * nr
                    if _shared_combos is not None and _shared_lock is not None:
                        with _shared_lock:
                            _shared_combos[0] += delta
                        combos_for_speed = _shared_combos[0]
                    else:
                        combos_for_speed = words_done * nr
                    spd = _fmt_speed(combos_for_speed / elapsed if elapsed>0 else 0)
                    pbar.set_postfix({"rules": cyan(str(len(counter))), "spd": green(spd)}, refresh=False)
                pbar.update(len(batch))
        log_info(f"[S1]    {bold(green(str(len(counter))))} unique rules passed bloom filter")
        return counter

    def _run_single_kernel(self, bd):
        if not self.context or not self.queue or not self.kernel_single:
            return []
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f|mf.COPY_HOST_PTR, hostbuf=arr); bufs.append(b); return b
            bb = B(bd['words_flat']); bbo = B(bd['word_offsets']); bbl = B(bd['word_lengths'])
            rb, rbo, rbl = self._get_rules_buffers(mf)
            outs = self.safe_output_buffer_size(bd['num_words'], bd['num_rules'])
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4); bufs.append(fc)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))
            tot = bd['num_words'] * bd['num_rules']
            gs = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_single.set_args(bb, bbo, bbl, rb, rbo, rbl, self.bloom_buf,
                                        np.int32(bd['num_words']), np.int32(bd['num_rules']), fo, fc)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_single, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish():
                self._reset_gpu(RuntimeError("queue.finish() timed out or failed"))
                return []
            cnt = np.zeros(1, dtype=np.int32); cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs); out = []
            if n>0:
                data = np.zeros(n*MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, data, fo)
                for i in range(n):
                    r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]).split(b'\0')[0].decode('latin-1', errors='ignore')
                    if r: out.append(r)
            return out
        except Exception as e:
            log_warn(f"Single kernel error: {e}")
            if not self._reset_gpu(e):
                log_error("GPU reset failed - aborting batch")
            return []
        finally:
            for b in bufs:
                try: b.release()
                except Exception: pass

    def _gen_random_chains(self, depth, count, valid, hot, existing, new_set):
        gen = set(); max_att = count * MAX_ATTEMPTS_MULTIPLIER
        hot_budget = int(count * HOT_RULE_RATIO) if hot else 0
        # Each pass targets an *absolute* size for gen, not a relative delta.
        # Original code used (count-hot_budget) as the cold target, meaning cold
        # pass exited immediately once len(gen) exceeded that threshold — which it
        # always did after the hot pass when HOT_RULE_RATIO >= 0.5.
        for target, use_hot in [(hot_budget, True), (count, False)]:
            att = 0
            while len(gen) < target and att < max_att:
                att += 1
                if use_hot and hot:
                    hp = random.randint(0, depth-1)
                    parts = [random.choice(hot) if i==hp else random.choice(valid) for i in range(depth)]
                else:
                    parts = [random.choice(valid) for _ in range(depth)]
                k = ' '.join(parts)
                if k not in existing and k not in gen and k not in new_set:
                    gen.add(k)
        return gen

    def build_numeric_seed_families(self, max_depth):
        digits = '0123456789'
        sbd = defaultdict(set)
        # A
        for depth in range(1, min(4, max_depth)+1):
            for combo in itertools.product(digits, repeat=depth):
                sbd[depth].add(' '.join(f'^{ch}' for ch in reversed(combo)))
        # B
        for depth in range(1, min(4, max_depth)+1):
            for combo in itertools.product(digits, repeat=depth):
                sbd[depth].add(' '.join(f'${ch}' for ch in combo))
        # C
        for depth in range(1, min(4, max_depth)+1):
            for ops in itertools.product(['^','$'], repeat=depth):
                for digs in itertools.product(digits, repeat=depth):
                    sbd[depth].add(' '.join(f'{o}{d}' for o,d in zip(ops,digs)))
        # D
        transform_ops = ['l','u','c','C','t','r','d','f','E','k','K','{','}','[',']']
        t_digit_ops = [f'^{d}' for d in digits] + [f'${d}' for d in digits] + ['[',']']
        for depth in range(2, min(4, max_depth)+1):
            for t_op in transform_ops:
                for ops in itertools.product(t_digit_ops, repeat=depth-1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed):
                        sbd[depth].add(seed)
        # E (dates)
        days = [f"{d:02d}" for d in range(1,32)]
        months = [f"{m:02d}" for m in range(1,13)]
        years2 = [f"{y:02d}" for y in range(60,100)] + [f"{y:02d}" for y in range(0,31)]
        years4 = [str(y) for y in range(1960,2031)]
        date4 = set(); date6 = set(); date8 = set()
        for d in days:
            for m in months:
                date4.add(d+m); date4.add(m+d)
        for y in years4: date4.add(y)
        for d in days:
            for m in months:
                for y in years2:
                    date6.add(d+m+y); date6.add(m+d+y)
        for d in days:
            for m in months:
                for y in years4:
                    date8.add(d+m+y); date8.add(m+d+y)
        for dset,bd in ((date4,4),(date6,6),(date8,8)):
            if bd>max_depth: continue
            for ds in dset:
                sbd[bd].add(' '.join(f'${c}' for c in ds))
                sbd[bd].add(' '.join(f'^{c}' for c in reversed(ds)))
        if max_depth>=5:
            for ds in date4:
                app = ' '.join(f'${c}' for c in ds)
                pre = ' '.join(f'^{c}' for c in reversed(ds))
                for t_op in transform_ops:
                    for chain in (f"{t_op} {app}", f"{t_op} {pre}"):
                        if HashcatRuleValidator.validate_rule_for_gpu(chain):
                            sbd[5].add(chain)
        bracket_ops = ['[',']']
        for dset,bd,br in ((date4,4,range(2,5)),(date6,6,range(1,3)),(date8,8,range(1,2))):
            for num_b in br:
                nd = bd+num_b
                if nd>max_depth: continue
                for brackets in itertools.product(bracket_ops, repeat=num_b):
                    bpfx = ' '.join(brackets)
                    for ds in dset:
                        app = ' '.join(f'${c}' for c in ds)
                        pre = ' '.join(f'^{c}' for c in reversed(ds))
                        for chain in (f"{bpfx} {app}", f"{bpfx} {pre}"):
                            if HashcatRuleValidator.validate_rule_for_gpu(chain):
                                sbd[nd].add(chain)
        # F
        for depth in range(1, min(3,max_depth)+1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                chain = ' '.join(f'${ch}' for ch in combo)
                if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[depth].add(chain)
        # G
        for depth in range(1, min(3,max_depth)+1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                chain = ' '.join(f'^{ch}' for ch in reversed(combo))
                if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[depth].add(chain)
        # H
        sp_ops_top = [f'${ch}' for ch in SPECIAL_CHARS_TOP] + [f'^{ch}' for ch in SPECIAL_CHARS_TOP]
        for depth in range(2, min(3,max_depth)+1):
            for t_op in transform_ops:
                for ops in itertools.product(sp_ops_top, repeat=depth-1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed): sbd[depth].add(seed)
        # I
        for depth in range(2, min(4,max_depth)+1):
            n_digits = depth-1
            for digit_combo in itertools.product(digits, repeat=n_digits):
                for sp in SPECIAL_CHARS_CORE:
                    app = ' '.join(f'${d}' for d in digit_combo) + f' ${sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(app): sbd[depth].add(app)
                    pre = f'^{sp} ' + ' '.join(f'^{d}' for d in reversed(digit_combo))
                    if HashcatRuleValidator.validate_rule_for_gpu(pre): sbd[depth].add(pre)
        # J
        for op in LEET_OPS:
            if HashcatRuleValidator.validate_rule_for_gpu(op): sbd[1].add(op)
        if max_depth>=2:
            leet_followup = [f'${d}' for d in digits] + [f'^{d}' for d in digits] + \
                            [f'${ch}' for ch in SPECIAL_CHARS_CORE] + [f'^{ch}' for ch in SPECIAL_CHARS_CORE]
            for leet_op in LEET_OPS:
                for follow in leet_followup:
                    chain = f"{leet_op} {follow}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[2].add(chain)
            for l1 in LEET_OPS:
                for l2 in LEET_OPS:
                    if l1!=l2 and HashcatRuleValidator.validate_rule_for_gpu(f"{l1} {l2}"):
                        sbd[2].add(f"{l1} {l2}")
        # K
        if max_depth>=2:
            for t1 in transform_ops:
                for t2 in transform_ops:
                    chain = f"{t1} {t2}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[2].add(chain)
        # L
        for depth in range(2, min(3,max_depth)+1):
            n_digits = depth-1
            for sp in SPECIAL_CHARS_CORE:
                for digit_combo in itertools.product(digits, repeat=n_digits):
                    app = f'${sp} ' + ' '.join(f'${d}' for d in digit_combo)
                    if HashcatRuleValidator.validate_rule_for_gpu(app): sbd[depth].add(app)
                    pre = ' '.join(f'^{d}' for d in digit_combo) + f' ^{sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(pre): sbd[depth].add(pre)
        # M
        if max_depth>=2:
            for leet_op in LEET_OPS:
                for t_op in transform_ops:
                    if HashcatRuleValidator.validate_rule_for_gpu(f"{leet_op} {t_op}"):
                        sbd[2].add(f"{leet_op} {t_op}")
                    if HashcatRuleValidator.validate_rule_for_gpu(f"{t_op} {leet_op}"):
                        sbd[2].add(f"{t_op} {leet_op}")
        return dict(sbd)

    def run_seed_extraction_pass(self, base_words, sbd, bloom_filter, phase1_rules, _silent=False):
        if self.bloom_buf is None: self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(phase1_rules)
            self.rule_index = {r:i for i,r in enumerate(self.gpu_rules)}
        multi = []
        for depth, chains in sorted(sbd.items()):
            if depth >= 2:
                multi.extend(chains)
        if not multi:
            log_info("[SEED]    No multi-depth seeds")
            return Counter()
        total = sum(len(v) for d,v in sbd.items() if d>=2)
        d_levels = sorted(d for d in sbd if d>=2)
        depth_range_str = (f"d{d_levels[0]}–d{d_levels[-1]}" if len(d_levels) > 1
                           else (f"d{d_levels[0]}" if d_levels else "none"))
        log_info(f"[SEED]    Numeric seed pass: {total:,} chains across {len(d_levels)} depth level(s) ({depth_range_str})")
        counter = self._run_chains_against_words(base_words, multi, "  SEED PASS ", _silent)
        log_info(f"[SEED]    {bold(green(str(len(counter))))} unique seed chains passed bloom filter")
        return counter

    def generate_informed_chains(self, rules, single_found, max_depth, seed_chains=None, prebuilt_sbd=None):
        max_depth = min(max_depth, MAX_HASHCAT_CHAIN)
        valid = [r for r in rules if HashcatRuleValidator.validate_rule_for_gpu(r)]
        if not valid: return []
        found_s = set(single_found.keys()) if single_found else set()
        hot = [r for r in valid if r in found_s]
        chains = set(valid)
        if seed_chains:
            for sc in seed_chains:
                if sc.count(' ') >= 1: chains.add(sc)
        for depth in range(2, max_depth+1):
            budget = self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 0)
            if budget <= 0: continue
            budget = min(budget, len(valid)**depth)
            new = self._gen_random_chains(depth, budget, valid, hot, chains, set())
            chains.update(new)
        return list(chains)

    def process_all_words_chain_rules(self, base_words, rules, max_depth, bloom_filter,
                                      single_counter, seed_chains=None, prebuilt_sbd=None, _silent=False):
        if self.bloom_buf is None: self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.rule_index = {r:i for i,r in enumerate(self.gpu_rules)}
        chains = self.generate_informed_chains(rules, single_counter, max_depth, seed_chains, prebuilt_sbd)
        if not chains: return Counter()
        counter = self._run_chains_against_words(base_words, chains, "  STAGE 2 ", _silent)
        log_info(f"[S2]    {bold(green(str(len(counter))))} unique chain rules passed bloom filter")
        return counter

    def _run_chains_against_words(self, base_words, chains, desc="  CHAINS  ", _silent=False,
                                   _shared_combos=None, _shared_lock=None):
        """Shared inner loop: iterate chain batches x word sub-batches, collect matches.
        Uses the same pre-flattened word-batch fast path the original seed-extraction
        pass used (set_base_words/get_word_batch_data), avoiding repeated
        latin-1 re-encoding of base_words on every chain batch."""
        counter = Counter()
        cbs = self.params['CHAINS_PER_BATCH']
        wsb = self.params['WORD_SUB_BATCH']
        n_batches = (len(chains) + cbs - 1) // cbs
        ncols = shutil.get_terminal_size((80, 24)).columns
        t0 = time.time()
        total_combos = 0
        self.set_base_words(base_words)
        with tqdm(total=n_batches, desc=green(desc), unit="batch", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  disable=_silent) as pbar:
            for ci in range(0, len(chains), cbs):
                if _kb.quit_requested: break
                cb = chains[ci:ci + cbs]
                seqs, depths = self._compute_chain_seqs(cb)
                for wi in range(0, len(base_words), wsb):
                    end = min(wi + wsb, len(base_words))
                    bd = self.get_word_batch_data(wi, end)
                    if bd['num_words'] == 0: continue
                    found = self._run_chain_kernel(bd, cb, (seqs, depths))
                    if found is None:
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                            log_error(f"Too many consecutive GPU errors, aborting {desc.strip()}")
                            break
                    elif found:
                        self._consecutive_errors = 0
                        counter.update(found)
                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS: break
                delta = len(cb) * len(base_words)
                total_combos += delta
                if _shared_combos is not None and _shared_lock is not None:
                    with _shared_lock:
                        _shared_combos[0] += delta
                    combos_for_speed = _shared_combos[0]
                else:
                    combos_for_speed = total_combos
                elapsed = time.time() - t0
                spd = _fmt_speed(combos_for_speed / elapsed if elapsed > 0 else 0)
                pbar.set_postfix({"rules": cyan(str(len(counter))), "spd": green(spd)}, refresh=False)
                pbar.update(1)
        return counter

    def _compute_chain_seqs(self, chains):
        maxd = self.params['MAX_CHAIN_DEPTH']
        seqs = []
        depths = []
        truncated = 0
        for c in chains:
            parts = c.split()
            if len(parts) > maxd:
                truncated += 1
                parts = parts[:maxd]
            depths.append(len(parts))
            idxs = [self.rule_index.get(r, -1) for r in parts]
            while len(idxs) < maxd:
                idxs.append(-1)
            seqs.extend(idxs)
        if truncated:
            log_warn(f"[GPU] {truncated} chain(s) truncated to max depth {maxd}")
        return seqs, depths

    def _run_chain_kernel(self, words, chains, _precomputed=None, return_word_idx=False):
        # `words` may be either a raw list of word strings, or an already-flattened
        # batch dict as returned by get_word_batch_data()/prepare_words_data() —
        # accept both so callers can pre-flatten once (SEED PASS) and avoid re-flattening
        # every sub-batch, without silently corrupting the word count (that was the bug:
        # passing a dict straight to prepare_words_data() iterated over its *keys* as
        # if they were words, and len(dict) was used as the word count).
        if not self.context or not self.queue or not self.kernel_chain:
            return []
        if _precomputed is not None:
            seqs_np = np.array(_precomputed[0], dtype=np.int32)
            depths_np = np.array(_precomputed[1], dtype=np.int32)
        else:
            seqs, depths = self._compute_chain_seqs(chains)
            seqs_np = np.array(seqs, dtype=np.int32)
            depths_np = np.array(depths, dtype=np.int32)
        if isinstance(words, dict):
            bd = words
        else:
            bd = self.prepare_words_data(words)
        n_words = bd['num_words']
        if n_words == 0 or not chains:
            return []
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f|mf.COPY_HOST_PTR, hostbuf=arr); bufs.append(b); return b
            bb = B(bd['words_flat']); bbo = B(bd['word_offsets']); bbl = B(bd['word_lengths'])
            rb, rbo, rbl = self._get_rules_buffers(mf)
            csb = B(seqs_np); cdb = B(depths_np)
            outs = self.safe_output_buffer_size(n_words, len(chains))
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4); bufs.append(fc)
            # foundw: parallel per-hit word-index buffer. Always allocated/passed since it's
            # now part of the compiled kernel's fixed signature; only decoded when the caller
            # (Stage 3 GA marginal-coverage fitness) actually needs per-word hit identity.
            fw = cl.Buffer(self.context, mf.WRITE_ONLY, outs*4); bufs.append(fw)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))
            tot = n_words*len(chains)
            gs = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_chain.set_args(bb, bbo, bbl, rb, rbo, rbl, csb, cdb, self.bloom_buf,
                                       np.int32(n_words), np.int32(len(chains)),
                                       np.int32(self.params['MAX_CHAIN_DEPTH']), fo, fc, fw)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_chain, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish():
                self._reset_gpu(RuntimeError("queue.finish() timed out or failed"))
                return None
            cnt = np.zeros(1, dtype=np.int32); cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs)
            if n<=0:
                return []
            data = np.zeros(n*MAX_CHAIN_STRING_LEN, dtype=np.uint8)
            cl.enqueue_copy(self.queue, data, fo)
            if return_word_idx:
                widx = np.zeros(n, dtype=np.int32)
                cl.enqueue_copy(self.queue, widx, fw)
                out = []
                for i in range(n):
                    r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]).split(b'\0')[0].decode('latin-1', errors='ignore')
                    if r: out.append((int(widx[i]), r))
                return out
            out = []
            for i in range(n):
                r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]).split(b'\0')[0].decode('latin-1', errors='ignore')
                if r: out.append(r)
            return out
        except Exception as e:
            log_warn(f"Chain kernel error: {e}"); self._reset_gpu(e); return None
        finally:
            for b in bufs:
                try: b.release()
                except Exception: pass

    def ensure_rules_ready(self, phase1_rules):
        """Populate gpu_rules/rule_index on demand. Split out of the various
        call sites so it can also be invoked remotely as a single RPC by
        ProcessEngineProxy (a remote caller can't just poke .gpu_rules /
        .rule_index attributes and expect them to affect the real engine
        living in another process)."""
        if not self.rule_index:
            self.gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(phase1_rules)
            self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}
        return True

# ----------------------------------------------------------------------
# Process-isolated GPU worker
# ----------------------------------------------------------------------
# A Python *thread* stuck inside a blocking C driver call (a hung
# clBuildProgram/clFinish — the Intel NEO / iGPU driver is the most
# frequent offender, and it gets much worse once a second vendor's
# driver (e.g. NVIDIA) is active in the same process, per the mixed-lock
# behaviour noted in MultiGPUEngine.compile_kernel below) cannot be
# forcibly killed by Python. A *process* can be (SIGTERM/SIGKILL), which
# is why every GPU in multi-GPU mode now gets its own OS process: if one
# device's driver wedges, the parent detects the silence and hard-kills
# just that process, and the run continues on whatever GPU(s) remain
# instead of hanging forever. Single-GPU runs are untouched — they keep
# using GPUEngine directly in-process, exactly as before.
# ----------------------------------------------------------------------

_WORKER_HEARTBEAT_SECS  = 3     # how often the child pings "still alive"
_WORKER_HANG_TIMEOUT    = 480   # no heartbeat for this long -> declare the
                                 # device hung and kill its process (covers
                                 # the existing 180s+120s compile-retry path
                                 # with headroom)
_WORKER_STARTUP_TIMEOUT = 120   # time allowed for initialize_gpu() to reply

# Methods whose return value is either not picklable (a live pyopencl
# Program) or needs server-side resolution before it can cross the
# process boundary (the "already uploaded to this GPU's VRAM" sentinel).
# Handled specially in the worker loop instead of via plain getattr-forward.
_WORKER_SPECIAL_METHODS = {'compile_kernel', 'generate_bloom_filter_gpu'}


def _gpu_worker_entry(device_spec, params, cmd_q, res_q, hb_q):
    """Runs inside the child process. Owns exactly one real GPUEngine and
    executes RPC-style commands sent by the parent over cmd_q/res_q, while
    a background thread pings hb_q so the parent can tell 'still working'
    apart from 'driver wedged'."""
    try:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
        eng = GPUEngine(params)
        ok = False
        try:
            ok = eng.initialize_gpu(device_spec)
        except Exception as e:
            res_q.put(('__init__', 'err', f"init crashed: {e}"))
            return
        dev_name = eng.device.name.strip() if (ok and eng.device) else None
        res_q.put(('__init__', 'ok' if ok else 'err', dev_name))
        if not ok:
            return

        stop_evt = threading.Event()
        def _ticker():
            while not stop_evt.is_set():
                try: hb_q.put(time.time())
                except Exception: pass
                stop_evt.wait(_WORKER_HEARTBEAT_SECS)
        threading.Thread(target=_ticker, daemon=True, name='hb-ticker').start()

        while True:
            try:
                call_id, method, args, kwargs = cmd_q.get()
            except Exception:
                break
            if method == '__shutdown__':
                break
            if method == '__set_params__':
                eng.params = args[0]
                try: res_q.put((call_id, 'ok', None))
                except Exception: pass
                continue
            try:
                if method == 'compile_kernel':
                    prog = eng.compile_kernel(*args, **kwargs)
                    result = prog is not None
                elif method == 'generate_bloom_filter_gpu':
                    bf = eng.generate_bloom_filter_gpu(*args, **kwargs)
                    if bf is _BLOOM_ALREADY_ON_GPU:
                        buf = eng.bloom_buf
                        bf_np = np.empty(buf.size // 4, dtype=np.int32)
                        cl.enqueue_copy(eng.queue, bf_np, buf)
                        eng.queue.finish()
                        bf = bf_np
                    result = bf
                else:
                    fn = getattr(eng, method)
                    result = fn(*args, **kwargs)
                res_q.put((call_id, 'ok', result))
            except Exception as e:
                try:
                    res_q.put((call_id, 'err', f"{type(e).__name__}: {e}"))
                except Exception:
                    pass
        stop_evt.set()
    except Exception as e:
        try:
            res_q.put(('__init__', 'err', f"worker crashed before init: {e}"))
        except Exception:
            pass


class _RemoteDeviceHandle:
    """Stand-in for a pyopencl Device across the process boundary — callers
    only ever read .name off the primary engine's .device for logging."""
    def __init__(self, name):
        self.name = name or "remote device"


class ProcessEngineProxy:
    """Drop-in stand-in for a GPUEngine that actually lives in its own OS
    process, used by MultiGPUEngine/GeneticRuleEvolver in place of a real
    GPUEngine whenever more than one GPU is active. Mirrors the subset of
    the GPUEngine surface those callers use (.device, .disabled,
    .bloom_buf/.program/.rule_index as truthy readiness flags, .queue, and
    method calls). Any call that goes silent (no heartbeat) for longer
    than _WORKER_HANG_TIMEOUT gets its process hard-killed and the device
    marked disabled — a legitimately long GPU sweep keeps pinging the
    whole time, so this only trips on an actual wedged driver."""

    def __init__(self, params, device_spec):
        self._params = params
        self.device_spec = device_spec
        self.disabled = False
        self.device = None
        self._proc = None
        self._cmd_q = None
        self._res_q = None
        self._hb_q = None
        self._call_id = 0
        self._lock = threading.Lock()
        self._bloom_ready = False
        self._compiled = False
        self._rules_ready = False
        self._last_heartbeat = None
        self._hb_drain_stop = None
        self._hb_drain_thread = None

    # -- lifecycle --------------------------------------------------------
    def initialize_gpu(self, device_spec=None):
        if device_spec is not None:
            self.device_spec = device_spec
        ctx = mp.get_context('spawn')
        self._cmd_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._hb_q = ctx.Queue()
        self._proc = ctx.Process(target=_gpu_worker_entry,
                                  args=(self.device_spec, self._params,
                                        self._cmd_q, self._res_q, self._hb_q),
                                  daemon=True, name=f'gpu-worker-{self.device_spec}')
        self._proc.start()
        try:
            tag, status, payload = self._res_q.get(timeout=_WORKER_STARTUP_TIMEOUT)
        except Exception:
            log_warn(f"[GPU]  Worker for device {self.device_spec} did not respond to init in "
                     f"{_WORKER_STARTUP_TIMEOUT}s — killing")
            self._kill()
            return False
        if status != 'ok':
            log_warn(f"[GPU]  Worker for device {self.device_spec} failed to init: {payload}")
            self._kill()
            return False
        self.device = _RemoteDeviceHandle(payload)
        self._last_heartbeat = time.time()
        self._start_heartbeat_drain()
        return True

    def _start_heartbeat_drain(self):
        """Continuously drain hb_q in the background, independent of whether
        an RPC call is currently in flight. Without this, an engine that
        sits idle for a long stretch (e.g. every non-primary GPU during
        STAGE 3's genetic algorithm, which can legitimately run for hours
        and only ever talks to engines[0]) still gets a heartbeat ping every
        _WORKER_HEARTBEAT_SECS from its worker. Nobody was reading those
        pings between calls, so they piled up in the pipe's OS buffer; once
        that buffer filled, the child's ticker thread blocked on hb_q.put()
        forever, and that half-stuck queue could then hang the parent at
        shutdown (multiprocessing.Queue.join_thread() blocks until a
        queue's buffered data is flushed). Keeping the queue drained at all
        times avoids both problems."""
        stop_evt = threading.Event()
        def _drain():
            while not stop_evt.is_set():
                try:
                    self._hb_q.get(timeout=1.0)
                    self._last_heartbeat = time.time()
                except Exception:
                    pass
        t = threading.Thread(target=_drain, daemon=True, name=f'hb-drain-{self.device_spec}')
        self._hb_drain_stop = stop_evt
        self._hb_drain_thread = t
        t.start()

    def _kill(self):
        self.disabled = True
        if self._hb_drain_stop is not None:
            self._hb_drain_stop.set()
            self._hb_drain_stop = None
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
            except Exception:
                pass
        # Never let a leftover queue block interpreter/process shutdown —
        # cancel_join_thread() tells the Queue's internal feeder thread not
        # to wait for a full flush, which is the standard fix for
        # multiprocessing hanging on exit when a queue still has buffered
        # (or partially written) data nobody is going to read anymore.
        for q in (self._cmd_q, self._res_q, self._hb_q):
            if q is None: continue
            try: q.close()
            except Exception: pass
            try: q.cancel_join_thread()
            except Exception: pass

    def terminate(self):
        if self.disabled or self._proc is None:
            self._kill()
            return
        try:
            self._cmd_q.put((None, '__shutdown__', (), {}))
        except Exception:
            pass
        self._kill()

    # -- generic RPC --------------------------------------------------------
    def _call(self, method, *args, **kwargs):
        if self.disabled or self._proc is None:
            return None
        with self._lock:
            self._call_id += 1
            call_id = self._call_id
            try:
                self._cmd_q.put((call_id, method, args, kwargs))
            except Exception as e:
                log_warn(f"[GPU]  Failed to send '{method}' to worker: {e}")
                self._kill()
                return None
            while True:
                try:
                    tag, status, payload = self._res_q.get(timeout=1.0)
                    if tag != call_id:
                        continue  # stale reply from an earlier, since-abandoned call
                    if status == 'ok':
                        return payload
                    log_warn(f"[GPU]  Worker error in '{method}': {payload}")
                    return None
                except Exception:
                    pass
                if self._proc is None or not self._proc.is_alive():
                    log_warn(f"[GPU]  Worker process died during '{method}' — dropping device")
                    self._kill()
                    return None
                last_beat = self._last_heartbeat or 0
                if time.time() - last_beat > _WORKER_HANG_TIMEOUT:
                    log_warn(f"[GPU]  Worker unresponsive for {_WORKER_HANG_TIMEOUT}s during "
                             f"'{method}' — killing device (driver hang)")
                    self._kill()
                    return None

    # -- GPUEngine-compatible surface --------------------------------------
    @property
    def params(self): return self._params

    @params.setter
    def params(self, value):
        self._params = value
        if not self.disabled and self._proc is not None:
            self._call('__set_params__', value)

    @property
    def bloom_buf(self):
        return True if self._bloom_ready else None

    @property
    def program(self):
        return True if self._compiled else None

    @property
    def rule_index(self):
        return {'ready': True} if self._rules_ready else {}

    @property
    def queue(self):
        return True if (not self.disabled and self._proc is not None) else None

    def compile_kernel(self, force=False):
        ok = bool(self._call('compile_kernel', force=force))
        self._compiled = ok
        return ok

    def upload_bloom_filter(self, bf):
        r = self._call('upload_bloom_filter', bf)
        self._bloom_ready = not self.disabled
        return r

    def generate_bloom_filter_gpu(self, target_words):
        bf = self._call('generate_bloom_filter_gpu', target_words)
        if bf is not None:
            self._bloom_ready = True
        return bf

    def ensure_rules_ready(self, phase1_rules):
        r = self._call('ensure_rules_ready', phase1_rules)
        self._rules_ready = bool(r)
        return r

    def build_numeric_seed_families(self, max_depth):
        return self._call('build_numeric_seed_families', max_depth=max_depth) or {}

    def process_all_words_single_rule(self, base_words, rules, bloom_filter, _silent=False,
                                       _shared_combos=None, _shared_lock=None):
        # _shared_combos/_shared_lock are in-process threading primitives from
        # MultiGPUEngine._parallel_run and can't cross a process boundary —
        # each worker just reports its own progress bar instead.
        return self._call('process_all_words_single_rule', base_words, rules, bloom_filter,
                           _silent=_silent) or Counter()

    def run_seed_extraction_pass(self, base_words, sbd, bloom_filter, phase1_rules, _silent=False):
        return self._call('run_seed_extraction_pass', base_words, sbd, bloom_filter,
                           phase1_rules, _silent=_silent) or Counter()

    def process_all_words_chain_rules(self, base_words, rules, max_depth, bloom_filter,
                                       single_counter, seed_chains=None, prebuilt_sbd=None, _silent=False):
        r = self._call('process_all_words_chain_rules', base_words, rules, max_depth, bloom_filter,
                        single_counter, seed_chains=seed_chains, prebuilt_sbd=prebuilt_sbd, _silent=_silent)
        if r is not None: self._compiled = True; self._bloom_ready = True; self._rules_ready = True
        return r or Counter()

    def generate_informed_chains(self, rules, single_found, max_depth, seed_chains=None, prebuilt_sbd=None):
        return self._call('generate_informed_chains', rules, single_found, max_depth,
                           seed_chains, prebuilt_sbd) or []

    def _run_chains_against_words(self, base_words, chains, desc="  CHAINS  ", _silent=False,
                                   _shared_combos=None, _shared_lock=None):
        return self._call('_run_chains_against_words', base_words, chains, desc, _silent) or Counter()

    def _compute_chain_seqs(self, chains):
        return self._call('_compute_chain_seqs', chains) or ([], [])

    def _run_chain_kernel(self, words, chains, _precomputed=None, return_word_idx=False):
        return self._call('_run_chain_kernel', words, chains, _precomputed, return_word_idx) or []

    def _safe_queue_finish(self):
        return bool(self._call('_safe_queue_finish'))


# ----------------------------------------------------------------------
# Multi-GPU engine — distributes word/chain batches across N process-
# isolated GPU workers. Only used when the user explicitly requests
# --device all (or an explicit "0,1" list) and more than one GPU is
# present; a single detected GPU always falls back to a plain in-process
# GPUEngine so single-GPU performance/behaviour is untouched.
# ----------------------------------------------------------------------
class MultiGPUEngine:
    """Wraps N GPUEngine instances and splits work across them.

    Stage 1 splits base_words across GPUs (word-dominated).
    Stage S and Stage 2 split chains across GPUs (chain-dominated) — each GPU
    tests its chain slice against all base_words.  Stage 3 (genetic) always
    runs on the primary engine only — pass .engines[0] to GeneticRuleEvolver.
    """

    def __init__(self, params):
        self._params = params
        self.engines: List['GPUEngine'] = []

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value
        for eng in self.engines:
            eng.params = value

    @property
    def device(self):
        return self.engines[0].device if self.engines else None

    def initialize_gpu(self, device_specs):
        # Each GPU gets its own OS process (see ProcessEngineProxy above) so a
        # single wedged driver can be hard-killed instead of hanging the run.
        for spec in device_specs:
            eng = ProcessEngineProxy(self._params, spec)
            if eng.initialize_gpu(spec):
                self.engines.append(eng)
            else:
                log_warn(f"[GPU]  Skipping device {spec} — init failed")
        if not self.engines:
            return False
        if len(self.engines) == 1:
            log_info("[GPU]  Only one GPU initialised — falling back to single-GPU mode")
        else:
            log_info(f"[GPU]  {bold(str(len(self.engines)))} GPUs active for parallel processing "
                      f"(process-isolated — a hung driver on one GPU is killed without affecting the others)")
        return True

    def _prune_disabled(self):
        """Drop any engine whose kernel build timed out (see GPUEngine.disabled).
        That device's driver may still be stuck holding an internal lock, so we
        stop issuing it further work and continue with whatever GPU(s) remain
        instead of retrying (and potentially hanging on) it again next stage."""
        alive = []
        for eng in self.engines:
            if eng.disabled:
                name = eng.device.name.strip() if eng.device else "unknown device"
                log_warn(f"[GPU]  Dropping unresponsive device ({name}) — continuing with remaining GPU(s)")
            else:
                alive.append(eng)
        removed = len(self.engines) - len(alive)
        self.engines = alive
        return removed

    def compile_kernel(self, force=False):
        # Recompiling right on the heels of concurrent multi-GPU execution
        # (Stage 1 / Seed Pass just ran both devices at once) has been seen to
        # trigger a build-time hang on mixed-vendor setups (e.g. NVIDIA +
        # Intel) even though each device compiles fine completely on its own
        # — some driver stacks share internal locking across vendors within
        # the same process that a single-GPU run never exercises. Draining
        # every engine's queue first, and giving the driver stack a brief
        # moment to settle, avoids issuing a build while another vendor's
        # driver may still be mid-transition.
        if len(self.engines) > 1:
            for eng in self.engines:
                if eng.queue is not None:
                    eng._safe_queue_finish()
            time.sleep(0.5)
        ok = True
        for eng in self.engines:
            ok = eng.compile_kernel(force=force) and ok
        self._prune_disabled()
        if not self.engines:
            log_error("[GPU]  All GPUs became unresponsive during kernel compile")
            return False
        return ok

    def generate_bloom_filter_gpu(self, target_words):
        primary = self.engines[0]
        # ProcessEngineProxy.generate_bloom_filter_gpu already resolves the
        # "already on this GPU's VRAM" sentinel server-side (inside the
        # worker process, where the real cl.Buffer/queue actually live) and
        # always hands back a plain numpy array here — never a live GPU
        # handle, since that can't cross a process boundary.
        bf = primary.generate_bloom_filter_gpu(target_words)
        if len(self.engines) > 1 and bf is not None:
            for eng in self.engines[1:]:
                eng.upload_bloom_filter(bf)
        return bf

    def upload_bloom_filter(self, bf):
        for eng in self.engines:
            eng.upload_bloom_filter(bf)

    def build_numeric_seed_families(self, max_depth):
        return self.engines[0].build_numeric_seed_families(max_depth)

    def shutdown(self):
        """Cleanly terminate every worker process. Safe to call multiple
        times / on partially-initialised state."""
        for eng in self.engines:
            try: eng.terminate()
            except Exception: pass

    def _split_words(self, words):
        n = len(self.engines)
        if n <= 1:
            return [words]
        chunk = max(1, (len(words) + n - 1) // n)
        return [words[i * chunk:(i + 1) * chunk] for i in range(n)]

    def _parallel_run(self, fn_per_engine, slices):
        n = len(self.engines)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(fn_per_engine, eng, sl)
                    for eng, sl in zip(self.engines, slices) if sl]
            merged = Counter()
            for f in futs:
                try:
                    merged.update(f.result())
                except Exception as exc:
                    log_warn(f"[GPU]  Worker thread error: {exc}")
        return merged

    def process_all_words_single_rule(self, base_words, rules, bloom_filter):
        if len(self.engines) == 1:
            return self.engines[0].process_all_words_single_rule(base_words, rules, bloom_filter)
        slices = self._split_words(base_words)
        log_info(f"[GPU]  STAGE 1 split across {len(self.engines)} GPUs (~{len(slices[0]):,} words each; progress shows GPU 0 only)")
        shared_combos = [0]; shared_lock = threading.Lock()
        return self._parallel_run(
            lambda eng, sl: eng.process_all_words_single_rule(
                sl, rules, bloom_filter, _silent=(eng is not self.engines[0]),
                _shared_combos=shared_combos, _shared_lock=shared_lock),
            slices)

    def _split_chains(self, chains):
        n = len(self.engines)
        if n <= 1:
            return [chains]
        chunk = max(1, (len(chains) + n - 1) // n)
        return [chains[i * chunk:(i + 1) * chunk] for i in range(n)]

    def _ensure_engine_ready(self, eng, bloom_filter, phase1_rules):
        if eng.bloom_buf is None: eng.upload_bloom_filter(bloom_filter)
        if not eng.program: eng.compile_kernel()
        if not eng.rule_index: eng.ensure_rules_ready(phase1_rules)

    def run_seed_extraction_pass(self, base_words, sbd, bloom_filter, phase1_rules):
        if len(self.engines) == 1:
            return self.engines[0].run_seed_extraction_pass(base_words, sbd, bloom_filter, phase1_rules)
        # Pre-extract chains from sbd so we can split them across GPUs
        multi = []
        for depth, chains in sorted(sbd.items()):
            if depth >= 2:
                multi.extend(chains)
        if not multi:
            log_info("[SEED]    No multi-depth seeds")
            return Counter()
        total = len(multi)
        d_levels = sorted(d for d in sbd if d >= 2)
        depth_range_str = (f"d{d_levels[0]}–d{d_levels[-1]}" if len(d_levels) > 1
                           else (f"d{d_levels[0]}" if d_levels else "none"))
        log_info(f"[SEED]    Numeric seed pass: {total:,} chains across {len(d_levels)} depth level(s) ({depth_range_str})")
        if len(self.engines) > 1:
            for eng in self.engines:
                if eng.queue is not None:
                    eng._safe_queue_finish()
            time.sleep(0.5)
        for eng in self.engines:
            self._ensure_engine_ready(eng, bloom_filter, phase1_rules)
        self._prune_disabled()
        if not self.engines:
            log_error("[SEED]    All GPUs became unresponsive — aborting seed pass")
            return Counter()
        if len(self.engines) == 1:
            result = self.engines[0]._run_chains_against_words(base_words, multi, "  SEED PASS ")
            log_info(f"[SEED]    {bold(green(str(len(result))))} unique seed chains passed bloom filter")
            return result
        chain_slices = self._split_chains(multi)
        log_info(f"[GPU]  SEED split across {len(self.engines)} GPUs (~{len(chain_slices[0]):,} chains each)")
        shared_combos = [0]; shared_lock = threading.Lock()
        merged = self._parallel_run(
            lambda eng, sl: eng._run_chains_against_words(
                base_words, sl, "  SEED PASS ", eng is not self.engines[0],
                shared_combos, shared_lock),
            chain_slices)
        log_info(f"[SEED]    {bold(green(str(len(merged))))} unique seed chains passed bloom filter")
        return merged

    def process_all_words_chain_rules(self, base_words, rules, max_depth, bloom_filter,
                                       single_counter, seed_chains=None, prebuilt_sbd=None):
        if len(self.engines) == 1:
            return self.engines[0].process_all_words_chain_rules(
                base_words, rules, max_depth, bloom_filter, single_counter, seed_chains, prebuilt_sbd)
        # Generate chains once on the primary engine, then split across GPUs
        primary = self.engines[0]
        if primary.bloom_buf is None: primary.upload_bloom_filter(bloom_filter)
        if not primary.rule_index: primary.ensure_rules_ready(rules)
        chains = primary.generate_informed_chains(rules, single_counter, max_depth, seed_chains, prebuilt_sbd)
        if not chains: return Counter()
        if len(self.engines) > 1:
            for eng in self.engines:
                if eng.queue is not None:
                    eng._safe_queue_finish()
            time.sleep(0.5)
        for eng in self.engines[1:]:
            self._ensure_engine_ready(eng, bloom_filter, rules)
        self._prune_disabled()
        if len(self.engines) == 1:
            result = primary._run_chains_against_words(base_words, chains, "  STAGE 2 ")
            log_info(f"[S2]    {bold(green(str(len(result))))} unique chain rules passed bloom filter")
            return result
        chain_slices = self._split_chains(chains)
        log_info(f"[GPU]  STAGE 2 split across {len(self.engines)} GPUs (~{len(chain_slices[0]):,} chains each)")
        shared_combos = [0]; shared_lock = threading.Lock()
        merged = self._parallel_run(
            lambda eng, sl: eng._run_chains_against_words(
                base_words, sl, "  STAGE 2 ", eng is not self.engines[0],
                shared_combos, shared_lock),
            chain_slices)
        log_info(f"[S2]    {bold(green(str(len(merged))))} unique chain rules passed bloom filter")
        return merged


# ----------------------------------------------------------------------
# Genetic Algorithm Rule Evolver (with signature cache limit)
# ----------------------------------------------------------------------
class GeneticRuleEvolver:
    def __init__(self, gpu_engine, base_words, rule_pool, max_depth,
                 pop_size=200, elite_frac=0.15, tournament_k=4, crossover_p=0.80,
                 mut_replace_p=0.60, mut_insert_p=0.20, mut_delete_p=0.20,
                 seed_hits=None, known_rules=None):
        self.gpu_engine = gpu_engine
        self.base_words = base_words
        self.rule_pool = rule_pool
        self.max_depth = max(2, max_depth)
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.tournament_k = tournament_k
        self.crossover_p = crossover_p
        total = mut_replace_p+mut_insert_p+mut_delete_p
        if total<=0: total=1.0
        self._mut_weights = [mut_replace_p/total, mut_insert_p/total, mut_delete_p/total]
        self.seed_hits = seed_hits or Counter()
        self.seed_chains_sorted = [r for r,_ in sorted(self.seed_hits.items(), key=lambda kv:-kv[1])] if self.seed_hits else []
        self.known_rules = known_rules or set()
        self._sig_cache = {}
        self._sig_cache_max = 50000
        self._sig_to_best = {}

    def _get_sig(self, chain_str):
        if chain_str not in self._sig_cache:
            if len(self._sig_cache) > self._sig_cache_max:
                for k in list(self._sig_cache.keys())[:self._sig_cache_max//10]:
                    del self._sig_cache[k]
            self._sig_cache[chain_str] = compute_rule_signature_hash(chain_str, BUILTIN_PROBES)
        return self._sig_cache[chain_str]

    def _update_sig_registry(self, raw_map):
        new=0
        for cs, hits in raw_map.items():
            if hits<=0: continue
            sig = self._get_sig(cs)
            if sig.startswith('__UNSUPPORTED__'): continue
            if sig not in self._sig_to_best or hits>self._sig_to_best[sig][1]:
                self._sig_to_best[sig] = (cs, hits)
                new+=1
        for sig,(best,_) in self._sig_to_best.items():
            self.known_rules.add(best)
        return new

    def _sig_is_covered(self, cs):
        if not self._sig_to_best: return False
        sig = self._get_sig(cs)
        return sig in self._sig_to_best

    def _mutate_adaptive(self, tokens):
        tokens = self._mutate(tokens)
        for _ in range(2):
            if not self._sig_is_covered(' '.join(tokens)):
                break
            tokens = self._mutate(tokens)
        return tokens

    def initial_population(self, hot_rules):
        hot = hot_rules[:50]
        pop_set = set()
        n_hot = int(self.pop_size*0.3)
        n_seeded = int(self.pop_size*0.3)
        tries=0
        while len(pop_set)<n_hot and tries<n_hot*20:
            tries+=1
            if len(hot)>=2:
                a,b = random.sample(hot,2)
            elif len(hot)==1:
                a,b = hot[0], random.choice(self.rule_pool)
            else: break
            pop_set.add((a,b))
        tries=0
        while len(pop_set)<n_hot+n_seeded and tries<(n_hot+n_seeded)*20:
            tries+=1
            depth = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.7 else random.randint(2,self.max_depth)
            if hot:
                tok = [random.choice(hot)] + [random.choice(self.rule_pool) for _ in range(depth-1)]
                random.shuffle(tok)
            else:
                tok = self._random_chain(depth)
            pop_set.add(tuple(tok))
        n_fill = self.pop_size - len(pop_set)
        fill_set = set()
        if self.seed_chains_sorted:
            novel = [s for s in self.seed_chains_sorted if s not in self.known_rules]
            known = [s for s in self.seed_chains_sorted if s in self.known_rules]
            pool = novel if len(novel)>=n_fill//2 else novel+known
            if pool:
                selected = random.sample(pool[:max(n_fill*3,100)], min(n_fill, len(pool)))
                for sc in selected:
                    tok = sc.split()
                    if 2<=len(tok)<=self.max_depth:
                        fill_set.add(tuple(tok))
        fill_tries=0
        while len(fill_set)<n_fill and fill_tries<n_fill*20:
            fill_tries+=1
            d = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.6 else random.randint(2,self.max_depth)
            fill_set.add(tuple(self._random_chain(d)))
        pop_set.update(fill_set)
        pop = [list(ind) for ind in pop_set]
        while len(pop)<self.pop_size:
            d = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.5 else random.randint(2,self.max_depth)
            pop.append(self._random_chain(d))
        return pop[:self.pop_size]

    def _random_chain(self, depth=0):
        if depth<=0: depth = random.randint(2, self.max_depth)
        return [random.choice(self.rule_pool) for _ in range(depth)]

    def _clamp(self, tokens):
        if len(tokens)<2: tokens += self._random_chain(2-len(tokens))
        return tokens[:self.max_depth]

    def _tournament_select(self, fitness):
        k = min(self.tournament_k, len(fitness))
        contenders = random.sample(fitness, k)
        return list(max(contenders, key=lambda x:x[1])[0])

    def _crossover(self, p1, p2):
        if len(p1)<2 or len(p2)<2 or random.random()>self.crossover_p:
            return list(p1), list(p2)
        cut1 = random.randint(1,len(p1)-1)
        cut2 = random.randint(1,len(p2)-1)
        c1 = self._clamp(p1[:cut1] + p2[cut2:])
        c2 = self._clamp(p2[:cut2] + p1[cut1:])
        return c1, c2

    def _mutate(self, tokens):
        tokens = list(tokens)
        op = random.choices(['replace','insert','delete'], weights=self._mut_weights)[0]
        if op=='replace':
            idx = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)
        elif op=='insert' and len(tokens)<self.max_depth:
            idx = random.randint(0,len(tokens))
            tokens.insert(idx, random.choice(self.rule_pool))
        elif op=='delete' and len(tokens)>2:
            idx = random.randrange(len(tokens))
            tokens.pop(idx)
        else:
            idx = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)
        return tokens

    def evaluate_population(self, population):
        """Returns {chain_str: set(global_word_index)} — the set of base_words indices
        each chain successfully hit. Tracking word identity (not just a hit count) is
        what makes marginal-coverage-gain fitness possible in evolve()."""
        cs = [' '.join(tok) for tok in population]
        valid = [c for c in cs if HashcatRuleValidator.validate_rule_for_gpu(c)]
        hit_words = {c: set() for c in cs}
        if not valid: return hit_words
        wsb = self.gpu_engine.params.get('WORD_SUB_BATCH',20000)
        cbs = self.gpu_engine.params.get('CHAINS_PER_BATCH',2000)
        for ci in range(0, len(valid), cbs):
            cb = valid[ci:ci+cbs]
            seqs, depths = self.gpu_engine._compute_chain_seqs(cb)
            for wi in range(0, len(self.base_words), wsb):
                wb = self.base_words[wi:wi+wsb]
                if wb:
                    found = self.gpu_engine._run_chain_kernel(wb, cb, (seqs, depths), return_word_idx=True)
                    if found:
                        for local_idx, rule_str in found:
                            hit_words[rule_str].add(wi + local_idx)
            self.gpu_engine._safe_queue_finish()
        return hit_words

    def evolve(self, hot_rules, generations, time_budget):
        if not self.rule_pool: log_warn("[S3] empty rule pool"); return Counter()
        if time_budget<=0: log_warn("[S3] no time budget"); return Counter()
        start = time.time()
        all_new = Counter()
        n_elite = max(1, int(self.pop_size*self.elite_frac))
        stagnation = 0
        best_ever = 0
        log_info(f"[S3]    pop={self.pop_size}  max_gen={generations}  elite={self.elite_frac:.0%}  budget={time_budget:.0f}s  pool={len(self.rule_pool):,}  known={len(self.known_rules):,}")
        pop = self.initial_population(hot_rules)
        last_gen=0
        ncols = shutil.get_terminal_size((80,24)).columns
        t0 = time.time()
        total_combos = 0
        with tqdm(total=generations, desc=green("  STAGE 3 "), unit="gen", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for gen in range(generations):
                last_gen=gen
                if _kb.quit_requested: break
                if time.time()-start >= time_budget: break
                hit_words = self.evaluate_population(pop)
                raw = {c: len(s) for c, s in hit_words.items()}
                self._update_sig_registry(raw)
                for cs, hits in raw.items():
                    if hits>0 and HashcatRuleValidator.validate_rule_for_gpu(cs):
                        if hits > all_new[cs]: all_new[cs]=hits

                # --- Marginal coverage gain (submodular greedy) ---
                # Process individuals in descending order of raw hit count and credit each
                # one only for the base_words it covers that no higher-priority individual
                # in this generation already covered. This is what "efficiency" means in
                # the concentrator/Pareto sense: a chain that duplicates coverage already
                # provided by a better individual contributes ~0 fitness, regardless of its
                # own raw hit count, so the population stops rewarding near-duplicates.
                order = sorted(pop, key=lambda ind: -len(hit_words.get(' '.join(ind), ())))
                covered = set()
                marginal = {}
                for ind in order:
                    cs_i = ' '.join(ind)
                    s = hit_words.get(cs_i, set())
                    gain = len(s - covered)
                    marginal[cs_i] = gain
                    covered |= s

                # Novelty bonus now checks functional-signature membership (_sig_is_covered)
                # instead of an exact string match against known_rules. A syntactically
                # different chain that is functionally identical (on BUILTIN_PROBES) to an
                # already-known rule no longer gets rewarded as if it were novel.
                fitness = sorted(
                    [(tuple(ind), marginal.get(' '.join(ind), 0) * (1 if self._sig_is_covered(' '.join(ind)) else 2))
                     for ind in pop],
                    key=lambda x: -x[1])
                best = fitness[0][1] if fitness else 0
                total_combos += len(pop) * len(self.base_words)
                elapsed = time.time()-t0
                spd = _fmt_speed(total_combos / elapsed if elapsed>0 else 0)
                if best>best_ever: best_ever=best; stagnation=0
                else: stagnation+=1
                if stagnation>=5:
                    stagnation=0
                    n_ref = max(1, int(self.pop_size*0.3))
                    refresh = []
                    refresh_set = {fitness[0][0]} if fitness else set()
                    rt=0
                    while len(refresh)<n_ref and rt<n_ref*20:
                        rt+=1
                        d = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.6 else random.randint(2,self.max_depth)
                        ind = tuple(self._random_chain(d))
                        if ind not in refresh_set:
                            refresh.append(list(ind)); refresh_set.add(ind)
                    keep = [list(ind) for ind,_ in fitness[:self.pop_size-n_ref]]
                    pop = keep+refresh
                    pbar.update(1)
                    pbar.set_postfix({"best":cyan(str(best)), "new":cyan(str(len(all_new))), "sigs":cyan(str(len(self._sig_to_best))), "spd":green(spd), "stag":yellow("REFRESH")}, refresh=False)
                    continue
                elites = [list(ind) for ind,_ in fitness[:n_elite]]
                next_pop = elites[:]
                next_set = {tuple(e) for e in elites}
                breed_attempts=0
                while len(next_pop)<self.pop_size and breed_attempts<(self.pop_size-len(next_pop))*8:
                    breed_attempts+=1
                    p1 = self._tournament_select(fitness)
                    p2 = self._tournament_select(fitness)
                    c1,c2 = self._crossover(p1,p2)
                    c1 = self._mutate_adaptive(c1)
                    c2 = self._mutate_adaptive(c2)
                    for c in (c1,c2):
                        if len(next_pop)>=self.pop_size: break
                        key = tuple(c)
                        if key in next_set: continue
                        if self._sig_is_covered(' '.join(c)):
                            d = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.7 else random.randint(2,self.max_depth)
                            c = self._random_chain(d)
                            key = tuple(c)
                            if key in next_set: continue
                        next_pop.append(c); next_set.add(key)
                fill_attempts=0
                while len(next_pop)<self.pop_size and fill_attempts<self.pop_size*4:
                    fill_attempts+=1
                    d = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.5 else random.randint(2,self.max_depth)
                    ind = tuple(self._random_chain(d))
                    if ind not in next_set:
                        next_pop.append(list(ind)); next_set.add(ind)
                pop = next_pop[:self.pop_size]
                pbar.update(1)
                pbar.set_postfix({"best":cyan(str(best)), "new":cyan(str(len(all_new))), "sigs":cyan(str(len(self._sig_to_best))), "spd":green(spd)}, refresh=False)
        elapsed = time.time()-start
        log_info(f"[S3]    Evolution complete — {bold(green(str(len(all_new))))} unique chains passed bloom filter  ({bold(cyan(str(len(self._sig_to_best))))} distinct functional signatures)  ({elapsed:.1f}s, {last_gen+1} generation(s))")
        return all_new

# ----------------------------------------------------------------------
# Main GPU Extractor (with depth fixes)
# ----------------------------------------------------------------------
class GPUExtractor:
    def __init__(self, base_count, target_count, max_depth, device_spec=None,
                 target_hours=0.5, max_chains=None, seed_rules_file=None, bloom_mb=None,
                 builtin_seeds=True, bloom_no_shard=False,
                 genetic=False, genetic_generations=50, genetic_pop=200, genetic_elite=0.15,
                 token_strip=False, token_strip_min_stem=4, token_strip_max_prefix=4,
                 token_strip_max_suffix=4, token_strip_min_leet_amb=3,
                 token_strip_workers=0, token_strip_chunk_size=0):
        self.base_count = base_count
        self.target_count = target_count
        self.user_max_depth = max_depth
        self.device_spec = device_spec
        self.max_chains = max_chains
        self.seed_rules_file = seed_rules_file
        self.bloom_mb = bloom_mb
        self.builtin_seeds = builtin_seeds
        self.bloom_no_shard = bloom_no_shard
        self.genetic = genetic
        self.genetic_generations = genetic_generations
        self.genetic_pop = genetic_pop
        self.genetic_elite = genetic_elite
        self.token_strip = token_strip
        self.token_strip_min_stem = token_strip_min_stem
        self.token_strip_max_prefix = token_strip_max_prefix
        self.token_strip_max_suffix = token_strip_max_suffix
        self.token_strip_min_leet_amb = token_strip_min_leet_amb
        self.token_strip_workers = token_strip_workers
        self.token_strip_chunk_size = token_strip_chunk_size
        self.params = calculate_dynamic_parameters(base_count, target_count, None, target_hours,
                                                   bloom_mb_override=bloom_mb, bloom_no_shard=bloom_no_shard)
        self.params['MAX_CHAIN_DEPTH'] = self.user_max_depth
        self.rules_gen = GPUCompatibleRulesGenerator()
        self.gpu_engine = None
        self.validator = HashcatRuleValidator()

    def load_seed_rules(self):
        if not self.seed_rules_file: return []
        seeds = []
        try:
            with open(self.seed_rules_file,'r',encoding='latin-1') as f:
                for line in f:
                    line=line.strip('\n\r')
                    if line and not line.startswith('#') and self.validator.validate_rule_for_gpu(line):
                        seeds.append(line)
            log_info(f"[SEED] Loaded {bold(str(len(seeds)))} seed rules")
        except Exception as e: log_warn(f"Seed load failed: {e}")
        return seeds

    def extract_rules(self, base_words, target_words, **depth_overrides):
        all_counts = Counter()
        rules = self.rules_gen.generate_gpu_compatible_rules()
        ts_singles = []
        ts_chains = []
        ts_sbd = defaultdict(set)
        builtin_set = set(rules)
        all_seeds = self.load_seed_rules()
        n_s0_chains_to_stage2 = 0
        ts_extra_singles = []  # defined early to avoid UnboundLocalError

        if self.token_strip:
            log_section("STAGE 0 — Token-Strip Rule Extraction (Core + Insert)")
            base_set_ts = set(base_words)
            log_info(f"[S0]    {len(target_words):,} target words  base {len(base_set_ts):,}  min-stem={self.token_strip_min_stem}  prefix={self.token_strip_max_prefix}  suffix={self.token_strip_max_suffix}  leet-amb={self.token_strip_min_leet_amb}")
            s0_max_depth = min(self.user_max_depth, MAX_HASHCAT_CHAIN)
            ts_all = extract_token_strip_rules(target_words, base_set_ts, max_depth=s0_max_depth,
                                               min_stem_len=self.token_strip_min_stem,
                                               max_prefix_len=self.token_strip_max_prefix,
                                               max_suffix_len=self.token_strip_max_suffix,
                                               max_leet_ambiguity=self.token_strip_min_leet_amb,
                                               workers=self.token_strip_workers,
                                               chunk_size=self.token_strip_chunk_size)
            for r in ts_all:
                d = len(r.split())
                if d==1: ts_singles.append(r)
                else:
                    if d <= self.user_max_depth:
                        ts_chains.append(r)
                        ts_sbd[d].add(r)
                    else:
                        log_debug(f"[S0] Dropping chain depth {d} > user max depth {self.user_max_depth}: {r}")
            _log_token_strip_stats(len(target_words), ts_all, inject_sbd=self.builtin_seeds)
            if self.user_max_depth >= 2:
                toggle_max_depth = min(12, self.user_max_depth)
                toggle = _generate_toggle_chain_seeds(toggle_max_depth)
                for tc in toggle:
                    d = len(tc.split())
                    if d>=2 and d <= self.user_max_depth:
                        ts_chains.append(tc)
                        ts_sbd.setdefault(d,set()).add(tc)
                        n_s0_chains_to_stage2 += 1
            ts_extra_singles = [r for r in ts_singles if r not in builtin_set]
            if ts_chains:
                all_seeds = list(all_seeds) + ts_chains
            n_s0_chains_to_stage2 += len(ts_chains)

        is_explicit_multi = (
            self.device_spec is not None
            and self.device_spec != 'all'
            and ',' in self.device_spec
        )
        if self.device_spec == 'all' or is_explicit_multi:
            if is_explicit_multi:
                # User (or the RCR GUI) passed an explicit comma-separated
                # device list, e.g. "--device 0,1". Use exactly those specs
                # instead of auto-discovering every GPU on the system.
                specs = [s.strip() for s in self.device_spec.split(',') if s.strip()]
                if not specs:
                    log_error(f"[GPU]  No valid device IDs parsed from '{self.device_spec}'"); return all_counts
            else:
                gpu_devs = get_all_gpu_only_devices()
                if not gpu_devs:
                    log_error("[GPU]  No GPU devices found for --device all"); return all_counts
                all_devs = get_all_devices()
                specs = [str(i) for i, (_p, d) in enumerate(all_devs) if d in gpu_devs]
            if len(specs) > 1:
                self.gpu_engine = MultiGPUEngine(self.params)
                if not self.gpu_engine.initialize_gpu(specs):
                    return all_counts
            else:
                self.gpu_engine = GPUEngine(self.params)
                if not self.gpu_engine.initialize_gpu(specs[0] if specs else None):
                    return all_counts
        else:
            self.gpu_engine = GPUEngine(self.params)
            if not self.gpu_engine.initialize_gpu(self.device_spec):
                return all_counts
        self.params = calculate_dynamic_parameters(self.base_count, self.target_count, self.gpu_engine.device,
                                                   self.params['TARGET_SECONDS']/3600,
                                                   bloom_mb_override=self.bloom_mb, bloom_no_shard=self.bloom_no_shard)
        seed_pass_depth = max(9, self.user_max_depth)  # seed pass explores numeric/year patterns that need
                                                        # depth >=9 regardless of the user's general --max-depth
        self.params['MAX_CHAIN_DEPTH'] = seed_pass_depth
        self.gpu_engine.params = self.params

        extra_seeds = [s for s in all_seeds if ' ' not in s.strip()]
        extra_valid = [s for s in extra_seeds if s not in builtin_set]
        rules_phase1 = list(dict.fromkeys(rules + ts_extra_singles + extra_valid))
        seed_chains = [s for s in all_seeds if ' ' in s.strip() and len(s.split()) <= self.user_max_depth]

        if extra_valid: log_info(f"[SEED] {len(extra_valid)} seed single-rule(s) added to STAGE 1")
        if seed_chains and self.user_max_depth<2: log_warn(f"[SEED] {len(seed_chains)} chain seed(s) ignored — requires --max-depth >= 2")

        global _p0_worker_base_set, _p0_worker_base_by_len
        _p0_worker_base_set = set()
        _p0_worker_base_by_len = {}

        if not base_words or not target_words:
            log_warn("Empty wordlist(s), aborting extraction")
            return Counter()
        log_info("[GPU]  Building bloom filter on GPU …")
        if not self.gpu_engine.compile_kernel(): return all_counts
        bloom_filter = self.gpu_engine.generate_bloom_filter_gpu(target_words)
        self.gpu_engine.upload_bloom_filter(bloom_filter)

        log_section("STAGE 1 — Single Rule Search")
        seed_note = f"  ({len(extra_valid)} from seeds)" if extra_valid else ""
        log_info(f"[S1]    {len(base_words):,} base words × {len(rules_phase1):,} atomic rules{seed_note}")
        t0 = time.time()
        single = self.gpu_engine.process_all_words_single_rule(base_words, rules_phase1, bloom_filter)
        t1 = time.time()
        all_counts.update(single)

        seed_hits = Counter()
        if self.builtin_seeds and not _kb.quit_requested:
            log_section("STAGE S — Seed Extraction (families A-M)")
            sbd = self.gpu_engine.build_numeric_seed_families(max_depth=seed_pass_depth)
            if ts_sbd:
                n_injected = 0
                n_ts_total = 0
                for depth, chains in ts_sbd.items():
                    n_ts_total += len(chains)
                    before = len(sbd.setdefault(depth, set()))
                    sbd[depth].update(chains)
                    n_injected += len(sbd[depth]) - before
                n_dupes = n_ts_total - n_injected
                dupe_str = f"  ({n_dupes:,} already in builtin families)" if n_dupes else ""
                if n_injected:
                    log_info(f"[SEED]    STAGE 0 injected {bold(cyan(str(n_injected)))} chain(s) into STAGE S sbd{dupe_str}")
            seed_hits = self.gpu_engine.run_seed_extraction_pass(base_words, sbd, bloom_filter, rules_phase1)
            all_counts.update(seed_hits)
            ts = time.time()
        else:
            if _kb.quit_requested: log_warn("[SEED]    Skipped — early exit")
            else: log_info(f"[SEED]    {yellow('Skipped')} (--no-builtin-seeds)")
            sbd = {}
            ts = t1

        self.params['MAX_CHAIN_DEPTH'] = self.user_max_depth
        self.gpu_engine.params = self.params
        stage2_gpu_ready = self.gpu_engine.compile_kernel(force=True)
        if not stage2_gpu_ready:
            log_warn("Failed to recompile kernel for Stage 2 depth")
        # If this is a MultiGPUEngine and every device ended up disabled (e.g. a
        # driver hung during compile and was dropped), or a single GPUEngine got
        # poisoned the same way, there's nothing left to run Stage 2 on — skip it
        # cleanly instead of calling into a dead engine.
        no_gpu_left = (hasattr(self.gpu_engine, 'engines') and not self.gpu_engine.engines) or \
                      (not hasattr(self.gpu_engine, 'engines') and self.gpu_engine.context is None)
        if no_gpu_left:
            log_error("[S2]    No usable GPU remaining — skipping STAGE 2 and STAGE 3")

        if self.user_max_depth > 1 and not _kb.quit_requested and not no_gpu_left:
            log_section("STAGE 2 — Rule Chain Search")
            log_info(f"[S2]    {bold(cyan(str(n_s0_chains_to_stage2)))} rules from STAGE 0 injected into chain search")

            if self.genetic and self.user_max_depth >= 2:
                _min_ga = 120.0
                _ga_frac = 0.20
                reserved_ga = max(_min_ga, self.params['TARGET_SECONDS'] * _ga_frac)
            else:
                reserved_ga = 0.0
            remaining = max(0, self.params['TARGET_SECONDS'] - (t1-t0) - reserved_ga)
            budget = remaining * self.params['EST_COMBOS_PER_SEC'] * TIME_SAFETY_FACTOR
            depths = list(range(2, self.user_max_depth+1))
            if budget>0 and base_words and depths:
                depth_budgets = {d: max(0, int(budget/len(depths)/(len(base_words)*d))) for d in depths}
            else:
                depth_budgets = {d:0 for d in depths}
            MIN_CHAINS = 5_000
            depth_budgets = {d: max(v, MIN_CHAINS) for d,v in depth_budgets.items()}
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
            log_info(f"[S2]    depth 2-{self.user_max_depth} | " + " | ".join(f"d{d}:{v:,}" for d,v in depth_budgets.items()))
            # IMPORTANT: with a single in-process GPUEngine, self.params and
            # gpu_engine.params were always the *same* dict object, so mutating
            # self.params in place (just above) was automatically visible to the
            # engine — no re-push needed. With process-isolated workers
            # (MultiGPUEngine/ProcessEngineProxy) that's no longer true: each
            # "params =" assignment ships a *copy* of the dict into the worker
            # process over IPC, so any mutation made afterwards (like the
            # CHAIN_GEN_LIMIT_* budgets we just computed) never reaches the
            # workers unless we push it again here. Without this, every worker
            # reads CHAIN_GEN_LIMIT_* as 0, generates ~no chains, and STAGE 2
            # finishes suspiciously fast with an almost-empty result.
            self.gpu_engine.params = self.params
            chains = self.gpu_engine.process_all_words_chain_rules(base_words, rules_phase1, self.user_max_depth,
                                                                   bloom_filter, single, seed_chains=seed_chains,
                                                                   prebuilt_sbd=sbd)
            all_counts.update(chains)

        if self.genetic and self.user_max_depth >= 2 and not _kb.quit_requested and not no_gpu_left:
            log_section("STAGE 3 — Genetic Algorithm Rule Evolution")
            rule_pool = HashcatRuleValidator.validate_rules_for_gpu(rules_phase1)
            hot_rules = [r for r,_ in sorted(single.items(), key=lambda kv:-kv[1])]
            t_now = time.time()
            remaining = max(0.0, self.params['TARGET_SECONDS'] - (t_now - t0))
            if self.genetic and self.user_max_depth >= 2:
                _min_ga = 120.0
                _ga_frac = 0.20
                reserved_ga = max(_min_ga, self.params['TARGET_SECONDS'] * _ga_frac)
            else:
                reserved_ga = 0.0
            ga_budget = remaining + reserved_ga
            if ga_budget < 5.0:
                log_warn(f"[S3]    Only {ga_budget:.1f}s available — consider raising --target-hours")
            else:
                log_info(f"[S3]    Budget: {bold(f'{ga_budget:.0f}s')}")
            known_set = set(all_counts.keys())
            _ga_engine = self.gpu_engine.engines[0] if isinstance(self.gpu_engine, MultiGPUEngine) else self.gpu_engine
            evolver = GeneticRuleEvolver(_ga_engine, base_words, rule_pool, self.user_max_depth,
                                         pop_size=self.genetic_pop, elite_frac=self.genetic_elite,
                                         seed_hits=seed_hits, known_rules=known_set)
            ga_hits = evolver.evolve(hot_rules, self.genetic_generations, ga_budget)
            before = len(all_counts)
            all_counts.update(ga_hits)
            new_from_ga = len(all_counts)-before
            truly_novel = sum(1 for r in ga_hits if r not in known_set)
            log_info(f"[S3]    {bold(cyan(str(new_from_ga)))} net new rules from STAGE 3  ({bold(green(str(len(ga_hits))))} total GA hits, {bold(cyan(str(truly_novel)))} genuinely novel)")

        validated = Counter({r:c for r,c in all_counts.items() if HashcatRuleValidator.validate_rule_for_gpu(r) and all(ord(ch)<=127 for ch in r)})
        return validated

# ----------------------------------------------------------------------
# Wordlist loader
# ----------------------------------------------------------------------
def load_wordlist(filename):
    words = set()
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        for line in data.split(b'\n'):
            w = line.decode('latin-1', errors='ignore').strip()
            if w and len(w) <= MAX_WORD_LEN:
                words.add(w)
    except FileNotFoundError:
        log_error(f"File not found: {filename}"); sys.exit(1)
    res = list(words)
    log_info(f"[LOAD] {bold(os.path.basename(filename))}: {bold(cyan(f'{len(res):,}'))} unique words")
    return res

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    global VERBOSE, ALLOW_REJECT_RULES
    ap = argparse.ArgumentParser(prog='rulest', description='GPU-Compatible Hashcat Rules Engine (optimized)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('base_wordlist', nargs='?')
    ap.add_argument('target_wordlist', nargs='?')
    ap.add_argument('-o','--output', default='rulest_output.txt')
    ap.add_argument('--device', default=None, metavar='SPEC',
                    help='Device index, name substring, "all" to run every detected GPU in parallel, '
                         'or a comma-separated list of indices (e.g. "0,1") to run only those GPUs in parallel '
                         '(default: auto-select best single GPU)')
    ap.add_argument('--list-devices', action='store_true')
    ap.add_argument('--max-depth', type=int, default=2)
    ap.add_argument('--target-hours', type=float, default=0.5)
    ap.add_argument('--max-chains', type=int, default=0)
    ap.add_argument('--bloom-mb', type=int, default=0)
    ap.add_argument('--bloom-no-shard', action='store_true')
    ap.add_argument('--seed-rules', default=None)
    for i in range(2,11):
        ap.add_argument(f'--depth{i}-chains', type=int, default=None, dest=f'depth{i}_chains')
    ap.add_argument('--allow-reject-rules', action='store_true')
    ap.add_argument('--no-builtin-seeds', action='store_true')
    pt = ap.add_argument_group('STAGE 0 — Token-Strip')
    pt.add_argument('--token-strip', action='store_true')
    pt.add_argument('--token-strip-min-stem', type=int, default=4)
    pt.add_argument('--token-strip-max-prefix', type=int, default=4)
    pt.add_argument('--token-strip-max-suffix', type=int, default=4)
    pt.add_argument('--token-strip-min-leet-amb', type=int, default=3)
    pt.add_argument('--token-strip-workers', type=int, default=0)
    pt.add_argument('--token-strip-chunk-size', type=int, default=0)
    ga = ap.add_argument_group('Genetic Algorithm')
    ga.add_argument('--genetic', action='store_true')
    ga.add_argument('--genetic-generations', type=int, default=50)
    ga.add_argument('--genetic-pop', type=int, default=200)
    ga.add_argument('--genetic-elite', type=float, default=0.15)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    ALLOW_REJECT_RULES = args.allow_reject_rules
    VERBOSE = args.debug
    print_banner()
    _print_controls()
    if args.list_devices: list_devices(); sys.exit(0)
    if not args.base_wordlist or not args.target_wordlist:
        ap.print_help(); print(); log_error("Both BASE and TARGET wordlists required"); sys.exit(1)
    if args.max_depth > MAX_HASHCAT_CHAIN:
        log_warn(f"Depth capped to {MAX_HASHCAT_CHAIN}"); args.max_depth=MAX_HASHCAT_CHAIN
    elif args.max_depth<1: log_error("--max-depth >=1"); sys.exit(1)

    log_info(f"  base      : {bold(args.base_wordlist)}")
    log_info(f"  target    : {bold(args.target_wordlist)}")
    shard_note = yellow('[sharding forced off]') if args.bloom_no_shard else dim('(auto: shard if >512MB)')
    log_info(f"  depth     : {bold(str(args.max_depth))}  |  hours: {bold(str(args.target_hours))}  |  bloom: {bold(str(args.bloom_mb or 'auto'))}MB  {dim('(GPU-only)')}  {shard_note}")
    if args.seed_rules: log_info(f"  seeds     : {bold(args.seed_rules)}")
    print()

    _EXCL = dim('[time budget: excluded]')
    _INCL = dim('[time budget: included]')

    if args.token_strip:
        _inj = green('→ STAGE S sbd → STAGE 2') if not args.no_builtin_seeds else yellow('→ STAGE 1 only → STAGE 2')
        log_info(f"  {bold(cyan('STAGE 0'))}  : {green('enabled')} — CPU exact-match (core+insert)  min-stem={args.token_strip_min_stem}  prefix={args.token_strip_max_prefix}  suffix={args.token_strip_max_suffix}  leet-amb={args.token_strip_min_leet_amb}  workers={safe_worker_count(args.token_strip_workers or mp.cpu_count())}  {_inj}  {_EXCL}")
    else:
        log_info(f"  {bold(dim('STAGE 0'))}  : {red('disabled')}")

    log_info(f"  {bold(cyan('STAGE 1'))}  : {green('enabled')} — single-rule GPU sweep  {_INCL}")

    if not args.no_builtin_seeds:
        log_info(f"  {bold(cyan('STAGE S'))}  : {green('enabled')} — numeric/pattern seed families A-M  {_EXCL}")
    else:
        log_info(f"  {bold(dim('STAGE S'))}  : {red('disabled')}  (--no-builtin-seeds)")

    log_info(f"  {bold(cyan('STAGE 2'))}  : {green('enabled')} — rule-chain GPU search  depth 2-{args.max_depth}  {_INCL}")

    if args.genetic:
        if not 0.0<args.genetic_elite<1.0: log_error("--genetic-elite must be between 0 and 1"); sys.exit(1)
        log_info(f"  {bold(cyan('STAGE 3'))}  : {green('enabled')} — genetic algorithm  pop={args.genetic_pop}  gen={args.genetic_generations}  elite={args.genetic_elite:.0%}  {_INCL}")
    else:
        log_info(f"  {bold(dim('STAGE 3'))}  : {red('disabled')}  (--genetic to enable)")
    print()

    base_words = load_wordlist(args.base_wordlist)
    target_words = load_wordlist(args.target_wordlist)
    print()

    _kb.start()
    t_start = time.time()
    extractor = GPUExtractor(
        len(base_words), len(target_words), args.max_depth, args.device,
        args.target_hours, args.max_chains, args.seed_rules, args.bloom_mb,
        builtin_seeds=not args.no_builtin_seeds,
        bloom_no_shard=args.bloom_no_shard, genetic=args.genetic,
        genetic_generations=args.genetic_generations, genetic_pop=args.genetic_pop,
        genetic_elite=args.genetic_elite, token_strip=args.token_strip,
        token_strip_min_stem=args.token_strip_min_stem,
        token_strip_max_prefix=args.token_strip_max_prefix,
        token_strip_max_suffix=args.token_strip_max_suffix,
        token_strip_min_leet_amb=args.token_strip_min_leet_amb,
        token_strip_workers=args.token_strip_workers,
        token_strip_chunk_size=args.token_strip_chunk_size,
    )
    depth_overrides = {f'depth{i}_override': getattr(args, f'depth{i}_chains') for i in range(2,11)}
    try:
        raw_counts = extractor.extract_rules(base_words, target_words, **depth_overrides)
    finally:
        if isinstance(extractor.gpu_engine, MultiGPUEngine):
            extractor.gpu_engine.shutdown()
    _kb.stop()
    del target_words
    gc.collect()
    if _kb.quit_requested: log_warn("[QUIT]  Early exit — saving partial results")
    log_info(f"\n[GPU]  Raw bloom-filter candidates: {bold(cyan(str(len(raw_counts))))}"); print()

    final_counts = minimize_by_signature(raw_counts, BUILTIN_PROBES)
    if ':' not in final_counts: final_counts[':'] = 0

    depth_dist = defaultdict(int)
    for r in final_counts:
        if r != ':': depth_dist[len(r.split())] += 1
    ds = '  '.join(f"d{d}:{depth_dist[d]:,}" for d in sorted(depth_dist))
    final_rules = len(final_counts) - (1 if ':' in final_counts else 0)
    removed = len(raw_counts) - len(final_counts)

    si = sorted(final_counts.items(), key=lambda kv: (-kv[1], len(kv[0].split()), kv[0]))
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("# rulest — GPU-Compatible Hashcat Rules Engine (optimized)\n")
        f.write(f"# Generated      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Base           : {os.path.basename(args.base_wordlist)}\n")
        f.write(f"# Target         : {os.path.basename(args.target_wordlist)}\n")
        f.write(f"# Depth          : 1-{args.max_depth}\n")
        bloom_mb = args.bloom_mb if args.bloom_mb>0 else "auto"
        f.write(f"# Bloom          : {bloom_mb} MB  (sharding: {'disabled' if args.bloom_no_shard else 'auto'})\n")
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
        for r,_ in si:
            if r!=':': f.write(f"{r}\n")
    log_info(f"[OUT]  Minimized rules written to: {bold(args.output)}")

    elapsed = time.time()-t_start
    sep = '─' * shutil.get_terminal_size((80,24)).columns
    print()
    log_info(cyan(sep))
    log_info(f"  {bold('DONE')}  rulest finished in {bold(f'{elapsed:.1f}s')}")
    log_info(cyan(sep))
    log_info(f"  GPU raw candidates : {bold(str(len(raw_counts)))}")
    log_info(f"  Rules kept         : {bold(green(str(final_rules)))}  {dim('('+ds+')')}")
    log_info(f"  Rules removed      : {bold(red(str(removed)))}")
    log_info(f"  Output file        : {bold(args.output)}")
    log_info(cyan(sep))
    top = sorted([(r,c) for r,c in final_counts.items() if r!=':'], key=lambda kv:(-kv[1],len(kv[0].split()),kv[0]))[:20]
    if top:
        print(); log_info(f"  Top {len(top)} rules by GPU frequency:")
        for i,(r,c) in enumerate(top,1):
            log_info(f"  {dim(str(i).rjust(3)+'.')}  {dim(f'd={len(r.split())}')}  {r:<42s}  {cyan(str(c))}")
    print()

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except Exception:
        pass
    main()
