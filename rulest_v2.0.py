#!/usr/bin/env python3
"""
rulest — GPU-Compatible Hashcat Rules Engine
=============================================
Extracts hashcat rules/chains by comparing a base wordlist against a target
wordlist.  GPU bloom filter screening (Phase 1: single rules, Phase 2: chains)
is followed by signature-based functional minimization.

Key feature: --sig-words / --min-word-len
  After GPU extraction, rules are deduplicated by their functional signature —
  the tuple of outputs produced when applied to a fixed probe set of words with
  length >= min_word_len.  Rules that produce identical outputs on every probe
  word are considered functionally equivalent; only the one with the highest
  GPU hit-count is retained.  The final ruleset is written to a single file
  sorted by GPU frequency (descending, UTF-8).

"""

import os
import sys
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
from typing import Dict, Set, Tuple, Optional, List
import gc
import multiprocessing as mp
from functools import partial
import datetime

# ================== GLOBAL FLAGS ===================
VERBOSE            = False   # set by --debug
ALLOW_REJECT_RULES = False   # set by --allow-reject-rules
# ==================================================

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

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
    """Always printed — key milestones and final stats."""
    print(msg)

def log_debug(msg: str) -> None:
    """Only printed in --debug mode."""
    if VERBOSE:
        print(f"{dim('[dbg]')} {msg}")

def log_warn(msg: str) -> None:
    """Always printed — non-fatal warnings."""
    print(yellow(f"[WARN] {msg}"))

def log_error(msg: str) -> None:
    """Always printed — errors."""
    print(red(f"[ERROR] {msg}"))

def log_section(title: str) -> None:
    """Section header — only in --debug mode."""
    if VERBOSE:
        bar = '─' * 56
        print(f"\n{cyan(bar)}")
        print(f"{cyan('│')} {bold(title)}")
        print(f"{cyan(bar)}")

# ====================================================================
# --- BANNER ---
# ====================================================================
BANNER = f"""{green(bold('''
 ██████╗ ██╗   ██╗██╗     ███████╗███████╗████████╗
 ██╔══██╗██║   ██║██║     ██╔════╝██╔════╝╚══██╔══╝
 ██████╔╝██║   ██║██║     █████╗  ███████╗   ██║
 ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║   ██║
 ██║  ██║╚██████╔╝███████╗███████╗███████║   ██║
 ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝   ╚═╝'''))}{C.END}
  {dim('GPU-Compatible Hashcat Rules Engine')}
  {dim('github.com/A113L/rulest')}
"""

def print_banner() -> None:
    print(BANNER)

# ====================================================================
# --- CONSTANTS ---
# ====================================================================
MAX_WORD_LEN         = 256
MAX_RULE_LEN         = 16
MAX_OUTPUT_LEN       = 512
MAX_CHAIN_STRING_LEN = 128
MAX_HASHCAT_CHAIN    = 31          # hashcat's maximum rule chain length

VRAM_USAGE_FACTOR    = 0.55
BLOOM_HASH_FUNCTIONS = 4
BLOOM_FILTER_MAX_MB  = 256

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

# Default probe-set parameters for signature-based minimization
DEFAULT_SIG_WORDS    = 21   # number of words to use as the probe set
DEFAULT_MIN_WORD_LEN = 10   # minimum word length for probe words

# Sentinel used in signatures when a rule contains unsupported opcodes
_UNSUPPORTED_SENTINEL = object()

# ====================================================================
# --- RULE EXCLUSION FILTER ---
# ====================================================================
def should_exclude_rule(rule: str) -> bool:
    if ALLOW_REJECT_RULES:
        return False
    if not rule:
        return False
    if len(rule) == 1 and rule in ('_', 'M', '4', '6', 'Q'):
        return True
    if len(rule) == 2 and rule[0] in ('!', '/', '(', ')', '<', '>', '_'):
        return True
    if len(rule) == 3 and rule[0] in ('?', '=', 'v'):
        return True
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
        if should_exclude_rule(rule_str):
            return False
        pos = 0; cnt = 0; n = len(rule_str)
        isd = HashcatRuleValidator.is_digit
        while pos < n:
            c = rule_str[pos]
            if c == ' ':
                pos += 1; continue
            # p / z / Z  — digit suffix is OPTIONAL
            if c in ('p', 'z', 'Z'):
                cnt += 1
                pos += 1
                if pos < n and isd(rule_str[pos]):
                    pos += 1
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
    """Apply one hashcat rule. Returns None for unsupported opcodes."""
    if not rule:
        return word
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
    """Apply space-separated rule chain. Returns None if any opcode unsupported."""
    cur = word
    for r in chain.split():
        cur = _py_apply_single_rule(r, cur)
        if cur is None:
            return None
    return cur

# ====================================================================
# --- SIGNATURE-BASED FUNCTIONAL MINIMIZATION ---
# ====================================================================

def _build_probe_set(base_words: list, sig_words: int, min_word_len: int,
                     seed: int = 42) -> List[str]:
    """
    Select *sig_words* probe words of length >= min_word_len from base_words.
    Falls back to shorter words if not enough long words are available.
    The selection is deterministic (fixed seed).
    """
    long_words = [w for w in base_words if len(w) >= min_word_len]
    rng = random.Random(seed)

    if len(long_words) >= sig_words:
        return rng.sample(long_words, sig_words)

    # Not enough long words — use what we have + fill from the rest
    short_words = [w for w in base_words if len(w) < min_word_len]
    needed = sig_words - len(long_words)
    fill   = rng.sample(short_words, min(needed, len(short_words)))
    probe  = long_words + fill

    if len(probe) < sig_words:
        log_warn(
            f"Only {len(probe)} words available for probe set "
            f"(requested {sig_words}). Minimization may be less precise."
        )
    return probe


def compute_rule_signature(rule: str, probe_words: List[str]) -> tuple:
    """
    Compute the functional signature of *rule* over *probe_words*.

    The signature is a tuple of outputs (one per probe word).  When the rule
    contains an unsupported opcode, the entire tuple is replaced by the
    single-element tuple ``('__UNSUPPORTED__',)`` so that all unsupported rules
    are bucketed together and the highest-GPU-count one survives.
    """
    outputs = []
    for word in probe_words:
        out = py_apply_chain(rule, word)
        if out is None:
            return ('__UNSUPPORTED__',)
        outputs.append(out)
    return tuple(outputs)


def minimize_by_signature(
    rule_counter: Counter,
    base_words:   list,
    sig_words:    int = DEFAULT_SIG_WORDS,
    min_word_len: int = DEFAULT_MIN_WORD_LEN,
) -> Counter:
    """
    Deduplicate *rule_counter* by functional signature.

    Algorithm
    ---------
    1. Build a probe set of *sig_words* words with length >= *min_word_len*
       drawn deterministically from *base_words*.
    2. For every candidate rule (with a tqdm progress bar), compute its
       signature = tuple of outputs on the probe set.
    3. Group rules that share the same signature — they are functionally
       equivalent on the probe set.
    4. Within each group, keep the rule with the **highest GPU hit-count**
       (ties broken by shortest depth, then lexicographic order).
    5. Return a Counter of surviving rules with their original GPU counts.

    Rules containing unsupported opcodes are bucketed into a single
    '__UNSUPPORTED__' group; one representative (highest count) is kept.

    Parameters
    ----------
    rule_counter : Counter
        Raw GPU bloom-filter hit counts keyed by rule string.
    base_words : list
        Full list of base words loaded from the base wordlist.
    sig_words : int
        Number of probe words used to build each signature.
    min_word_len : int
        Minimum character length for probe words.

    Returns
    -------
    Counter
        Surviving rules mapped to their original GPU hit-counts.
    """
    if not rule_counter:
        return Counter()

    probe = _build_probe_set(base_words, sig_words, min_word_len)

    log_section("POST-PROCESSING — Signature-Based Functional Minimization")
    log_info(f"[MINIMIZE] Candidates  : {bold(str(len(rule_counter)))}")
    log_info(f"[MINIMIZE] Probe words : {bold(str(len(probe)))}  "
             f"(min length {min_word_len}+)")
    log_debug(f"Probe set: {probe}")

    # sig_map[signature] = list of (rule, gpu_count)
    sig_map: Dict[tuple, List[Tuple[str, int]]] = defaultdict(list)

    rule_items = list(rule_counter.items())

    with tqdm(
        total=len(rule_items),
        desc=green("  Minimizing"),
        unit="rule",
        ncols=88,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as pbar:
        n_groups = 0
        for rule, gpu_count in rule_items:
            sig = compute_rule_signature(rule, probe)
            sig_map[sig].append((rule, gpu_count))
            n_groups = len(sig_map)
            pbar.update(1)
            pbar.set_postfix({"unique_sigs": cyan(str(n_groups))}, refresh=False)

    # Within each signature group, keep the rule with the highest GPU count.
    # Tie-break: prefer shorter chain depth, then lexicographic order.
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

    removed    = len(rule_counter) - len(survivors)
    n_groups   = len(sig_map)
    log_info(f"[MINIMIZE] {green('Done')}")
    log_info(f"           Unique signatures : {bold(cyan(str(n_groups))):>12s}")
    log_info(f"           Rules kept        : {bold(green(str(len(survivors)))):>12s}")
    log_info(f"           Rules removed     : {bold(red(str(removed))):>12s}  "
             f"({removed / max(1, len(rule_counter)):.1%})")
    if n_unsupported:
        log_info(f"           Unsupported (kept 1 each group) : "
                 f"{bold(str(n_unsupported))}")
    log_debug(f"minimize_by_signature complete: "
              f"kept={len(survivors)}, removed={removed}, sig_groups={n_groups}")

    return survivors


# ====================================================================
# ====================================================================
#  PRESERVE THIS BLOCK FROM THE ORIGINAL rulest_v2.py
#  (lines ~622 – ~1498 in the original file)
#
#  Copy-paste the following functions/classes verbatim:
#
#    def load_wordlist(...)
#    class GPUExtractor:
#        ...   (all methods)
#
#  Nothing in that block needs to be changed.
# ====================================================================
# ====================================================================

# --------------------------------------------------------------------
# GPU device helpers (from original)
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
# Dynamic parameters (from original)
# --------------------------------------------------------------------
def calculate_dynamic_parameters(base_count, target_count, device=None,
                                  target_hours=0.5, bloom_mb_override=None):
    if device:
        try:
            mwgs = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            mcu  = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            fv   = estimate_free_vram(device)
            vgb  = fv / 1024**3
            isn  = 'NVIDIA' in device.get_info(cl.device_info.NAME).upper()
            lws  = max(s for s in POSSIBLE_WORK_GROUP_SIZES if s <= mwgs)
            if isn and mcu >= 38: lws = min(512, lws)
            est  = (LOW_END_COMBOS_PER_SEC if mcu < LOW_END_COMPUTE_UNITS_THRESHOLD
                    else BASELINE_COMBOS_PER_SEC)
            log_debug(f"GPU: CU={mcu}, VRAM~{vgb:.1f}GB, WGS={lws}, "
                      f"est={est//1_000_000}M combos/s")
        except Exception:
            lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2*1024**3; vgb = 2.0
    else:
        lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2*1024**3; vgb = 2.0

    vram_scale = max(0.25, min(1.0, vgb / 8.0))
    ts         = target_hours * 3600
    eff_bloom  = bloom_mb_override or BLOOM_FILTER_MAX_MB
    bsize_b    = min(1024*1024*eff_bloom*2, eff_bloom*1024*1024)
    if vgb < 4: bsize_b = min(bsize_b, 32*1024*1024)
    bloom_bits = bsize_b * 8

    if target_count > 0:
        fill = 1.0 - math.exp(-BLOOM_HASH_FUNCTIONS * target_count / bloom_bits)
        fpr  = fill ** BLOOM_HASH_FUNCTIONS
        msg  = (f"Bloom: {bsize_b//1024//1024}MB, fill={fill:.3%}, FPR~{fpr:.6%}"
                + (" ← HIGH, raise --bloom-mb" if fpr > 0.01 else ""))
        log_debug(msg)
        if fpr > 0.01:
            log_warn(f"Bloom filter FPR {fpr:.3%} is high — consider --bloom-mb {eff_bloom*2}")

    return {
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

# --------------------------------------------------------------------
# GPU-compatible rules generator (from original)
# --------------------------------------------------------------------
class GPUCompatibleRulesGenerator:
    def __init__(self):
        self.validator = HashcatRuleValidator()

    def generate_gpu_compatible_rules(self):
        rules  = set()
        digits = '0123456789'

        # Single‑letter rules (excluding p, z, Z because they need a digit)
        rules.update(['l','u','c','C','t','r','d','f',
                      'q','E','{','}','[',']','k','K',':'])

        # Two‑letter rules that need a digit
        for cmd in ('T','D','L','R','+','-','.',',', "'", 'z','Z','y','Y'):
            for pos in digits: rules.add(f'{cmd}{pos}')

        # Two‑letter rules that need a digit (p)
        for pos in digits:
            rules.add(f'p{pos}')

        # Three‑letter rules that need two digits
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
# OpenCL kernel (from original)
# --------------------------------------------------------------------
GPU_KERNEL_TEMPLATE = r"""
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
        uint idx=(h1+i*h2)%BLOOM_FILTER_SIZE;
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

/* Chain kernel: do NOT abort on result==0 (rule had no effect on this particular
   word — still a valid chain step).  Only abort if the word becomes empty.      */
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
"""

# --------------------------------------------------------------------
# GPU Engine (from original)
# --------------------------------------------------------------------
class GPUEngine:
    def __init__(self, params):
        self.params              = params
        self.context = self.queue = self.device = self.program = None
        self.max_work_group_size = 512
        self.local_work_size     = params.get('LOCAL_WORK_SIZE', 512)
        self.bloom_buf           = None
        self.rule_index          = {}
        self.gpu_rules           = []
        self.kernel_single       = None
        self.kernel_chain        = None

    def get_free_vram(self):      return estimate_free_vram(self.device)
    def get_max_allocation(self): return get_max_allocation(self.device)

    def safe_output_buffer_size(self, words_count, chains_count):
        avail = min(self.get_free_vram(), self.get_max_allocation()) - 5*1024**2
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
            vram_gb = self.get_free_vram() / 1024**3
            log_info(f"[GPU]  {bold(self.device.name.strip())}")
            log_debug(f"       WGS={self.local_work_size}, VRAM~{vram_gb:.1f}GB, "
                      f"CU={self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)}")
            return True
        except Exception as e:
            log_error(f"GPU init failed: {e}"); return False

    def compile_kernel(self):
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
            self.program       = cl.Program(self.context, src).build()
            self.kernel_single = self.program.find_single_rules_gpu
            self.kernel_chain  = self.program.find_rule_chains_gpu
            log_debug("OpenCL kernel compiled successfully")
            return self.program
        except Exception as e:
            log_error(f"Kernel compile failed: {e}"); return None

    def generate_bloom_filter(self, target_words):
        bsz = self.params['BLOOM_FILTER_SIZE'] // 8
        bf  = np.zeros(bsz, dtype=np.uint8)
        log_debug(f"Building bloom filter: {bsz//1024//1024}MB for {len(target_words):,} words")
        for w in tqdm(target_words,
                      desc=green("  Bloom filter"),
                      unit="word",
                      ncols=88,
                      leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
            wb = w.encode('latin-1')
            h1 = fnv1a_32(wb, FNV1A_SEED1); h2 = fnv1a_32(wb, FNV1A_SEED2)
            for i in range(BLOOM_HASH_FUNCTIONS):
                idx = (h1 + i*h2) % self.params['BLOOM_FILTER_SIZE']
                bf[idx//8] |= 1 << (idx%8)
        return bf

    def upload_bloom_filter(self, bf):
        mf = cl.mem_flags
        if self.bloom_buf: self.bloom_buf.release()
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

    # ---------------------------------------------------------------- Phase 1
    def process_all_words_single_rule(self, base_words, rules, bloom_filter):
        self.upload_bloom_filter(bloom_filter)
        if not self.compile_kernel(): return Counter()
        self.gpu_rules  = HashcatRuleValidator.validate_rules_for_gpu(rules)
        self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}
        log_debug(f"Phase 1: {len(base_words):,} words × {len(self.gpu_rules):,} rules")

        counter = Counter()
        bs      = self.params['WORDS_PER_BATCH']

        with tqdm(total=len(base_words),
                  desc=green("  Phase 1 "),
                  unit="word",
                  ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  ) as pbar:

            for i in range(0, len(base_words), bs):
                batch = base_words[i:i+bs]
                if batch:
                    found = self._run_single_kernel(self.prepare_batch_data(batch, self.gpu_rules))
                    if found:
                        counter.update(found)
                        pbar.set_postfix({"rules": cyan(str(len(counter)))}, refresh=False)
                pbar.update(len(batch))
                self.queue.finish(); gc.collect()

        log_info(f"[P1]   {bold(green(str(len(counter))))} unique rules passed bloom filter")
        log_debug(f"Phase 1 complete: {len(counter)} rules in counter")
        return counter

    def _run_single_kernel(self, bd):
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f | mf.COPY_HOST_PTR, hostbuf=arr)
                bufs.append(b); return b

            bb=B(bd['words_flat']); bbo=B(bd['word_offsets']); bbl=B(bd['word_lengths'])
            rb=B(bd['rules_flat']); rbo=B(bd['rule_offsets']); rbl=B(bd['rule_lengths'])
            outs = self.safe_output_buffer_size(bd['num_words'], bd['num_rules'])
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4);                         bufs.append(fc)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))

            tot = bd['num_words'] * bd['num_rules']
            gs  = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_single.set_args(bb,bbo,bbl,rb,rbo,rbl,self.bloom_buf,
                                        np.int32(bd['num_words']),np.int32(bd['num_rules']),fo,fc)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_single, (gs,), (self.local_work_size,))
            self.queue.finish()

            cnt = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs); out = []
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
            self.queue = cl.CommandQueue(self.context); return []
        finally:
            for b in bufs:
                try: b.release()
                except: pass

    # ---------------------------------------------------------------- Phase 2
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

    def generate_informed_chains(self, rules, single_found, max_depth, seed_chains=None):
        # Cap max_depth to hashcat limit
        max_depth = min(max_depth, MAX_HASHCAT_CHAIN)
        valid   = [r for r in rules if HashcatRuleValidator.validate_rule_for_gpu(r)]
        if not valid: return []
        found_s = set(single_found.keys()) if single_found else set()
        hot     = [r for r in valid if r in found_s]
        chains  = set(valid)
        sbd     = defaultdict(set)
        digits  = '0123456789'

        # Numeric seed families up to depth 4 (but not exceeding max_depth)
        max_seed_depth = min(max_depth, 4)

        # Family A: pure prepend
        a_cnt: Dict[int, int] = defaultdict(int)
        for n in range(10 ** max_seed_depth):
            s = str(n)
            d = len(s)
            sbd[d].add(' '.join(f'^{ch}' for ch in reversed(s)))
            a_cnt[d] += 1

        # Family B: pure append
        b_cnt: Dict[int, int] = defaultdict(int)
        for n in range(10 ** max_seed_depth):
            s = str(n)
            d = len(s)
            sbd[d].add(' '.join(f'${ch}' for ch in s))
            b_cnt[d] += 1

        # Family C: mixed
        for depth in range(1, max_seed_depth + 1):
            for ops in itertools.product(['^', '$'], repeat=depth):
                for digs in itertools.product(digits, repeat=depth):
                    sbd[depth].add(' '.join(f'{o}{d}' for o, d in zip(ops, digs)))

        # User-supplied seed chains
        if seed_chains:
            for sc in seed_chains:
                sbd[sc.count(' ') + 1].add(sc)
            log_debug(f"Loaded {len(seed_chains):,} user seed chains")

        # Debug report
        c_total = sum(2**d * 10**d for d in range(1, max_seed_depth + 1))
        log_debug(f"Numeric seeds  max_seed_depth={max_seed_depth}")
        log_debug("  A (pure ^): " +
                  ", ".join(f"d{d}={a_cnt[d]:,}" for d in sorted(a_cnt)) +
                  f"  [{sum(a_cnt.values()):,} total]")
        log_debug("  B (pure $): " +
                  ", ".join(f"d{d}={b_cnt[d]:,}" for d in sorted(b_cnt)) +
                  f"  [{sum(b_cnt.values()):,} total]")
        log_debug(f"  C (mixed) : [{c_total:,} total]")
        log_debug("  A∪B∪C     : " +
                  ", ".join(f"d{d}={len(sbd[d]):,}" for d in sorted(sbd)) +
                  f"  [{sum(len(v) for v in sbd.values()):,} total]")

        for ds in sbd.values(): chains.update(ds)

        for depth in range(2, max_depth+1):
            budget = self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 0)
            if budget <= 0: continue
            budget   = min(budget, len(valid)**depth)
            new      = set(sbd.get(depth, set())) - chains
            prev     = list(sbd.get(depth-1, set()))
            ext_tgt  = int(budget * EXTENSION_RATIO)
            att = 0
            while len(new) < ext_tgt and att < ext_tgt*MAX_ATTEMPTS_MULTIPLIER:
                att += 1
                if prev:
                    nc = random.choice(prev) + ' ' + random.choice(valid)
                    if nc not in chains and nc not in new: new.add(nc)
            rem = budget - len(new)
            if rem > 0:
                new.update(self._gen_random_chains(depth, rem, valid, hot, chains, new))
            chains.update(new); sbd[depth].update(new)
            log_debug(f"Depth {depth}: budget={budget:,}, generated={len(new):,}")

        log_debug(f"Total chains generated: {len(chains):,}")
        return list(chains)

    def process_all_words_chain_rules(self, base_words, rules, max_depth,
                                      bloom_filter, single_counter, seed_chains=None):
        if self.bloom_buf is None: self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}

        chains = self.generate_informed_chains(rules, single_counter, max_depth, seed_chains)
        if not chains: return Counter()

        log_debug(f"Phase 2: {len(chains):,} chains × {len(base_words):,} words")

        counter = Counter()
        cbs     = self.params['CHAINS_PER_BATCH']
        wsb     = self.params['WORD_SUB_BATCH']
        n_batches = (len(chains)+cbs-1)//cbs

        with tqdm(total=n_batches,
                  desc=green("  Phase 2 "),
                  unit="batch",
                  ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  ) as pbar:

            for ci in range(0, len(chains), cbs):
                cb = chains[ci:ci+cbs]
                for wi in range(0, len(base_words), wsb):
                    wb = base_words[wi:wi+wsb]
                    if wb:
                        found = self._run_chain_kernel(wb, cb)
                        if found:
                            counter.update(found)
                    self.queue.finish()

                pbar.update(1)
                pbar.set_postfix({"rules": cyan(str(len(counter)))}, refresh=False)
                gc.collect()

        log_info(f"[P2]   {bold(green(str(len(counter))))} unique chain rules passed bloom filter")
        log_debug(f"Phase 2 complete: {len(counter)} chain rules in counter")
        return counter

    def _run_chain_kernel(self, words, chains):
        seqs = []; depths = []
        for chain in chains:
            parts = chain.split(); depths.append(len(parts))
            idxs  = [self.rule_index.get(r,-1) for r in parts]
            while len(idxs) < self.params['MAX_CHAIN_DEPTH']: idxs.append(-1)
            seqs.extend(idxs)

        bd = self.prepare_batch_data(words, self.gpu_rules)
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f | mf.COPY_HOST_PTR, hostbuf=arr)
                bufs.append(b); return b

            bb=B(bd['words_flat']); bbo=B(bd['word_offsets']); bbl=B(bd['word_lengths'])
            rb=B(bd['rules_flat']); rbo=B(bd['rule_offsets']); rbl=B(bd['rule_lengths'])
            csb=B(np.array(seqs,   dtype=np.int32))
            cdb=B(np.array(depths, dtype=np.int32))
            outs = self.safe_output_buffer_size(len(words), len(chains))
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4);                         bufs.append(fc)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))

            tot = len(words)*len(chains)
            gs  = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_chain.set_args(bb,bbo,bbl,rb,rbo,rbl,csb,cdb,self.bloom_buf,
                                       np.int32(len(words)),np.int32(len(chains)),
                                       np.int32(self.params['MAX_CHAIN_DEPTH']),fo,fc)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_chain, (gs,), (self.local_work_size,))
            self.queue.finish()

            cnt = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs); out = []
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
            self.queue = cl.CommandQueue(self.context); return []
        finally:
            for b in bufs:
                try: b.release()
                except: pass

# --------------------------------------------------------------------
# GPU Extractor (from original)
# --------------------------------------------------------------------
class GPUExtractor:
    def __init__(self, base_count, target_count, max_depth, device_spec=None,
                 target_hours=0.5, max_chains=None, seed_rules_file=None, bloom_mb=None):
        self.base_count      = base_count
        self.target_count    = target_count
        self.max_depth       = max_depth
        self.device_spec     = device_spec
        self.max_chains      = max_chains
        self.seed_rules_file = seed_rules_file
        self.bloom_mb        = bloom_mb
        self.params          = calculate_dynamic_parameters(
            base_count, target_count, None, target_hours, bloom_mb_override=bloom_mb)
        self.params['MAX_CHAIN_DEPTH'] = max_depth
        self.rules_gen       = GPUCompatibleRulesGenerator()
        self.gpu_engine      = GPUEngine(self.params)
        self.validator       = HashcatRuleValidator()

    def load_seed_rules(self):
        if not self.seed_rules_file: return []
        seeds = []
        try:
            with open(self.seed_rules_file, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip('\n\r')
                    if line and not line.startswith('#'):
                        if self.validator.validate_rule_for_gpu(line):
                            seeds.append(line)
            log_info(f"[SEED] Loaded {bold(str(len(seeds)))} seed rules from {self.seed_rules_file}")
            log_debug(f"Seed rules file: {self.seed_rules_file}, valid={len(seeds)}")
        except Exception as e:
            log_warn(f"Seed rules load failed: {e}")
        return seeds

    def extract_rules(self, base_words, target_words, **depth_overrides):
        all_counts = Counter()
        rules      = self.rules_gen.generate_gpu_compatible_rules()

        if not self.gpu_engine.initialize_gpu(self.device_spec):
            return all_counts

        self.params = calculate_dynamic_parameters(
            self.base_count, self.target_count,
            self.gpu_engine.device,
            self.params['TARGET_SECONDS'] / 3600,
            bloom_mb_override=self.bloom_mb)
        self.params['MAX_CHAIN_DEPTH'] = self.max_depth
        self.gpu_engine.params = self.params

        seed_chains  = self.load_seed_rules()
        bloom_filter = self.gpu_engine.generate_bloom_filter(target_words)

        # --- Phase 1: single rules ---
        log_section("PHASE 1 — Single Rule Search")
        log_info(f"[P1]   {len(base_words):,} base words × {len(rules):,} atomic rules")
        t0     = time.time()
        single = self.gpu_engine.process_all_words_single_rule(base_words, rules, bloom_filter)
        t1     = time.time()
        all_counts.update(single)
        log_debug(f"Phase 1 elapsed: {t1-t0:.1f}s")

        # --- Phase 2: rule chains ---
        if self.max_depth > 1:
            log_section("PHASE 2 — Rule Chain Search")
            remaining = max(0, self.params['TARGET_SECONDS'] - (t1-t0))
            budget    = remaining * self.params['EST_COMBOS_PER_SEC'] * TIME_SAFETY_FACTOR
            depths    = list(range(2, self.max_depth+1))
            depth_budgets = ({d: int(budget/len(depths)/(len(base_words)*d)) for d in depths}
                             if budget > 0 and base_words and depths
                             else {d: 0 for d in depths})

            for d in depths:
                key = f'depth{d}_override'
                if key in depth_overrides and depth_overrides[key] is not None:
                    depth_budgets[d] = depth_overrides[key]
                    log_debug(f"Depth {d} chain budget overridden to {depth_overrides[key]:,}")
                depth_budgets[d] = max(0, depth_budgets[d])

            if self.max_chains:
                total = sum(depth_budgets.values())
                if total > self.max_chains:
                    scale = self.max_chains / total
                    depth_budgets = {d: int(v*scale) for d,v in depth_budgets.items()}
                    log_debug(f"Budgets scaled by {scale:.3f} to fit --max-chains={self.max_chains}")

            for d, bgt in depth_budgets.items():
                self.params[f'CHAIN_GEN_LIMIT_{d}'] = bgt

            log_info(f"[P2]   depth 2–{self.max_depth} | "
                     + " | ".join(f"d{d}:{v:,}" for d,v in depth_budgets.items()))
            log_debug(f"Remaining time budget: {remaining:.1f}s")

            chains = self.gpu_engine.process_all_words_chain_rules(
                base_words, rules, self.max_depth, bloom_filter, single, seed_chains)
            all_counts.update(chains)
            log_debug(f"Phase 2 elapsed: {time.time()-t1:.1f}s")

        validated = Counter({r: c for r,c in all_counts.items()
                             if HashcatRuleValidator.validate_rule_for_gpu(r)})
        log_debug(f"Post-validation: {len(validated)} rules from {len(all_counts)} raw")
        return validated

# --------------------------------------------------------------------
# Wordlist loader (from original)
# --------------------------------------------------------------------
def load_wordlist(filename: str) -> list:
    words = set()
    try:
        with open(filename, 'r', encoding='latin-1', errors='ignore') as f:
            for line in tqdm(f,
                             desc=green(f"  Loading  "),
                             unit="line",
                             ncols=88,
                             leave=False,
                             bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}] {postfix}"):
                w = line.strip()
                if w and len(w) <= MAX_WORD_LEN: words.add(w)
    except FileNotFoundError:
        log_error(f"File not found: {filename}"); sys.exit(1)
    result = list(words)
    log_info(f"[LOAD] {bold(os.path.basename(filename))}: "
             f"{bold(cyan(f'{len(result):,}'))} unique words")
    return result

# ====================================================================
# --- ENTRY POINT ---
# ====================================================================
def main() -> None:
    global VERBOSE, ALLOW_REJECT_RULES

    ap = argparse.ArgumentParser(
        prog='rulest',
        description='GPU-Compatible Hashcat Rules Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Positional / core ----------------------------------------
    ap.add_argument('base_wordlist',   nargs='?', default=None,
                    help='Base wordlist (input words to transform)')
    ap.add_argument('target_wordlist', nargs='?', default=None,
                    help='Target wordlist (desired transformation outputs)')

    # ---- Output ---------------------------------------------------
    ap.add_argument('-o', '--output', default='rulest_output.txt',
                    help='Output file for the minimized rule set (default: rulest_output.txt)')

    # ---- GPU / extraction ----------------------------------------
    ap.add_argument('--device',       default=None,
                    help='OpenCL device index or name substring')
    ap.add_argument('--list-devices', action='store_true',
                    help='List available OpenCL devices and exit')
    ap.add_argument('--max-depth',    type=int, default=2,
                    help='Maximum rule chain depth (default: 2, max: 31)')
    ap.add_argument('--target-hours', type=float, default=0.5,
                    help='Target GPU runtime in hours (default: 0.5)')
    ap.add_argument('--max-chains',   type=int, default=0,
                    help='Hard cap on total chain candidates (0 = auto)')
    ap.add_argument('--bloom-mb',     type=int, default=0,
                    help=f'Bloom filter size in MB (default: {BLOOM_FILTER_MAX_MB})')
    ap.add_argument('--seed-rules',   default=None,
                    help='Path to a file of seed rules to prioritise')

    # ---- Depth overrides -----------------------------------------
    for i in range(2, 11):
        ap.add_argument(f'--depth{i}-chains', type=int, default=None,   # ← changed from 0 to None
                        dest=f'depth{i}_chains',
                        help=f'Override chain count for depth {i} (default: auto)')

    # ---- Signature-based minimization ----------------------------
    ap.add_argument('--sig-words',    type=int, default=DEFAULT_SIG_WORDS,
                    help=f'Number of probe words for signature computation '
                         f'(default: {DEFAULT_SIG_WORDS})')
    ap.add_argument('--min-word-len', type=int, default=DEFAULT_MIN_WORD_LEN,
                    help=f'Minimum word length for probe words '
                         f'(default: {DEFAULT_MIN_WORD_LEN})')

    # ---- Misc ----------------------------------------------------
    ap.add_argument('--allow-reject-rules', action='store_true',
                    help='Allow rules that hashcat would reject (reject-class opcodes)')
    ap.add_argument('--debug',        action='store_true',
                    help='Enable verbose/debug output')

    args = ap.parse_args()

    # ---- Apply globals -------------------------------------------
    ALLOW_REJECT_RULES = args.allow_reject_rules
    VERBOSE            = args.debug

    # ---- Banner --------------------------------------------------
    print_banner()

    if args.list_devices:
        list_devices(); sys.exit(0)

    if not args.base_wordlist or not args.target_wordlist:
        ap.print_help(); print()
        log_error("Both BASE and TARGET wordlists are required.")
        sys.exit(1)

    # Cap depth to hashcat limit
    if args.max_depth > MAX_HASHCAT_CHAIN:
        log_warn(f"Depth {args.max_depth} exceeds hashcat's maximum chain length "
                 f"({MAX_HASHCAT_CHAIN}). Limiting to {MAX_HASHCAT_CHAIN}.")
        args.max_depth = MAX_HASHCAT_CHAIN
    elif args.max_depth < 1:
        log_error("--max-depth must be >= 1"); sys.exit(1)

    # ---- Summary line --------------------------------------------
    log_info(f"  base      : {bold(args.base_wordlist)}")
    log_info(f"  target    : {bold(args.target_wordlist)}")
    log_info(f"  depth     : {bold(str(args.max_depth))}  |  "
             f"hours: {bold(str(args.target_hours))}  |  "
             f"bloom: {bold(str(args.bloom_mb or BLOOM_FILTER_MAX_MB))}MB  |  "
             f"sig_words: {bold(str(args.sig_words))}  "
             f"min_len: {bold(str(args.min_word_len))}")
    log_debug(f"Full args: {vars(args)}")
    print()

    # ---- Load ----------------------------------------------------
    base_words   = load_wordlist(args.base_wordlist)
    target_words = load_wordlist(args.target_wordlist)
    print()

    # ---- GPU extraction ------------------------------------------
    t_start   = time.time()
    extractor = GPUExtractor(
        len(base_words), len(target_words), args.max_depth,
        args.device, args.target_hours, args.max_chains,
        args.seed_rules, args.bloom_mb)

    depth_overrides = {f'depth{i}_override': getattr(args, f'depth{i}_chains')
                       for i in range(2, 11)}

    raw_counts = extractor.extract_rules(base_words, target_words, **depth_overrides)
    log_info(f"\n[GPU]  Raw bloom-filter candidates: {bold(cyan(str(len(raw_counts))))}")
    log_debug(f"Raw counts: {len(raw_counts)} rules total")
    print()

    # ---- Signature-based functional minimization -----------------
    final_counts = minimize_by_signature(
        raw_counts,
        base_words,
        sig_words    = args.sig_words,
        min_word_len = args.min_word_len,
    )

    # Always include the identity rule
    if ':' not in final_counts:
        final_counts[':'] = 0

    # ---- Depth distribution for header ---------------------------
    depth_dist: Dict[int, int] = defaultdict(int)
    for rule in final_counts:
        if rule != ':':
            depth_dist[len(rule.split())] += 1
    depth_summary = '  '.join(f"d{d}:{depth_dist[d]:,}" for d in sorted(depth_dist))

    final_rules = len(final_counts) - (1 if ':' in final_counts else 0)
    removed     = len(raw_counts)   - len(final_counts)

    # ---- Write single output file — sorted by frequency (GPU hits desc) --
    sorted_items = sorted(
        final_counts.items(),
        key=lambda kv: (-kv[1], len(kv[0].split()), kv[0])
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("# rulest — GPU-Compatible Hashcat Rules Engine\n")
        f.write(f"# Generated      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Base           : {os.path.basename(args.base_wordlist)}\n")
        f.write(f"# Target         : {os.path.basename(args.target_wordlist)}\n")
        f.write(f"# Depth          : 1–{args.max_depth}\n")
        f.write(f"# Bloom          : {args.bloom_mb or BLOOM_FILTER_MAX_MB} MB\n")
        f.write("#\n")
        f.write(f"# GPU raw candidates      : {len(raw_counts):,}  "
                f"(bloom hits, includes false positives)\n")
        f.write(f"# Post-processing         : signature-based minimization\n")
        f.write(f"#   Probe words           : {args.sig_words}  "
                f"(min length {args.min_word_len})\n")
        f.write(f"#   Equiv. rules removed  : {removed:,}\n")
        f.write("#\n")
        f.write(f"# Rules kept     : {final_rules:,}  ({depth_summary})\n")
        f.write(f"# Sorted by      : GPU frequency (descending, UTF-8)\n")
        f.write(":\n")
        for rule, _ in sorted_items:
            if rule != ':':
                f.write(f"{rule}\n")

    log_info(f"[OUT]  Minimized rules written to: {bold(args.output)}")

    # ---- Final report --------------------------------------------
    elapsed = time.time() - t_start
    sep     = '─' * 56

    print()
    log_info(cyan(sep))
    log_info(f"  {bold('DONE')}  rulest finished in {bold(f'{elapsed:.1f}s')} "
             f"({elapsed/3600:.3f}h)")
    log_info(cyan(sep))
    log_info(f"  GPU raw candidates : {bold(str(len(raw_counts)))}")
    log_info(f"  Rules kept         : {bold(green(str(final_rules)))}  "
             f"{dim('('+depth_summary+')')}")
    log_info(f"  Rules removed      : {bold(red(str(removed)))}")
    log_info(f"  Output file        : {bold(args.output)}")
    log_info(cyan(sep))

    # Top-20 by GPU frequency
    top_sorted = sorted(
        [(r, s) for r, s in final_counts.items() if r != ':'],
        key=lambda kv: (-kv[1], len(kv[0].split()), kv[0])
    )[:20]
    if top_sorted:
        print()
        log_info(f"  Top {len(top_sorted)} rules by GPU frequency:")
        for i, (rule, score) in enumerate(top_sorted, 1):
            depth = len(rule.split())
            log_info(f"  {dim(str(i).rjust(3)+'.')} "
                     f"{dim(f'd={depth}')}  "
                     f"{rule:<42s}  "
                     f"{cyan(str(score))}")
    print()


if __name__ == '__main__':
    main()

