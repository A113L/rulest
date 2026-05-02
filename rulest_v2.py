#!/usr/bin/env python3
"""
rulest — GPU-Compatible Hashcat Rules Engine
=============================================
Extracts hashcat rules/chains by comparing a base wordlist against a target
wordlist.  GPU bloom filter screening (Phase 1: single rules, Phase 2: chains)
is followed by signature-based functional minimization.

Key feature: built-in probe set for functional minimisation.
  Rules are deduplicated by their functional signature — the tuple of outputs
  produced when applied to a fixed hand-curated set of probe words.  Rules that
  produce identical outputs on every probe word are considered functionally
  equivalent; only the one with the highest GPU hit-count is retained.
  The built-in probe set covers short words (including "password"), mixed case,
  words with digits/specials, and repeated-char words – no external wordlist
  is required for accurate minimisation.

  Large-scale minimisation (v2 improvement)
  ─────────────────────────────────────────
  The original in-memory path stores every signature as a Python tuple of
  len(probe_words) strings inside a dict.  At ~2.5 KB per key, 8 M+ rules
  require ~20 GB of Python heap — typically causing an OOM kill or a C-
  extension crash reported as "core dumped".

  ``minimize_by_signature`` now dispatches automatically:

    ≤ MINIMIZE_DISK_THRESHOLD rules (default 500 k)
      → ``_minimize_mem``  — original in-memory algorithm (fast, zero I/O).

    > MINIMIZE_DISK_THRESHOLD rules
      → ``_minimize_disk`` — SQLite-backed algorithm:
          • Computes one signature at a time and immediately SHA-1-hashes it
            (160-bit digest replaces the ~2.5 KB tuple in Python heap).
          • Writes (sig_hash, rule, count, depth) in batches of
            MINIMIZE_DISK_BATCH_SIZE rows to a temporary on-disk SQLite DB.
          • Uses a single-pass INSERT … ON CONFLICT DO UPDATE so the DB
            table holds at most one row per equivalence class at any moment.
          • Reads survivors with one SELECT scan; deletes the temp file.
          Peak Python heap is O(MINIMIZE_DISK_BATCH_SIZE × avg_rule_len)
          — a few hundred KB regardless of total rule count.
  Both paths apply the same tie-breaking: highest GPU count → shortest
  chain depth → lexicographic rule order.

Phase S — Built-in Seed Families (A–M)
  Thirteen seed families are always tested via a dedicated GPU chain-kernel
  pass, independent of the random-chain time budget (unless --no-builtin-seeds).

  Numeric families
    A  Pure Prepend digits           (depths 1–4)
    B  Pure Append digits            (depths 1–4)
    C  Mixed Prepend/Append digits   (depths 1–4)
    D  Transform + digit/bracket     (depths 2–4)
    E  Date patterns DDMM/YYYY/…     (depths 4–9)

  Special-character families
    F  Pure Append special chars     (depths 1–3, top-15 chars)
    G  Pure Prepend special chars    (depths 1–3, top-15 chars)
    H  Transform + special char      (depths 2–3, top-15 chars)
    I  Digit(s) + special char       (depths 2–4, core-7 chars)
       — covers the ubiquitous "word123!" / "!word123" patterns

  New families
    J  Leet substitutions            (depths 1–2, 10 core pairs)
       — sa@ se3 so0 si1 sl1 ss5 ss$ st7 sa4 si!
       — depth 2: leet + digit/special suffix/prefix
       — depth 2: double-leet chains (e.g. "sa@ so0" → "p@ssw0rd")
    K  Double-transform chains       (depth 2, all 15×15 pairs)
       — covers "c r", "u d", "t f", "E l", "c {", "l ]", etc.
    L  Special-before-digit patterns (depths 2–3, core-7 chars)
       — reverse orientation of Family I: "!1word" / "word!12"
       — append: $sp $d…  prepend: ^d… ^sp
    M  Leet + transform chains       (depth 2)
       — leet substitution followed by a transform op (all 15)
       — and transform op followed by leet substitution
       — covers "P@ssword", "@DMIN", "p@SSW0RD" patterns

  Special chars — top-15 (F/G/H):  ! @ # $ % ^ & * ? . - _ + ( )
  Special chars — core-7  (I/L):   ! @ # $ % * ?

Phase 0 — Token-Strip Rule Extraction  (--token-strip)
  An optional CPU-only pre-pass that extracts rules empirically by
  decomposing target passwords into their constituent token categories
  and building hashcat rule chains that reconstruct each password from
  a base-wordlist stem.

  Five extraction modes
  ─────────────────────
  LETTER MODE  (primary when letters ≥ digits)
    Boundary = digits + specials → prepend/append ops.
    Middle   = letters + leet   → case-transform + leet-substitution ops.
    Covers:  "Password123!", "P@ssw0rd", "ADMIN2024"

  DIGIT MODE   (primary when digits > letters — dynamic boundary)
    Boundary = letters + specials → prepend/append ops.
    Middle   = pure digit string → looked up verbatim in base wordlist.
    Covers:  "abc2024" (stem "2024"), "123456xyz" (stem "123456")
    Selects which characters become ^/$ rules based on word composition
    so common numeric-base passwords are found regardless of orientation.

  REVERSE MODE (chain prefix 'r')
    The middle segment is reversed before stem lookup and case/leet decode.
    Covers:  "drowssap!" → stem "password", rule "r $!"

  DELETE-EDGE MODE (chain prefix '[' or ']')
    One non-boundary character is stripped from start ('[') or end (']')
    before running the normal letter-mode extraction.

  DUPLICATE / FOLD MODE (chain 'd' or 'f')
    Detects passwords built by duplicating ('d': stem+stem) or folding
    ('f': stem+reverse(stem)) a base-wordlist word.

  Toggle-chain seeds (separate, direct injection into Phase 2)
  ─────────────────────────────────────────────────────────────
  Deterministic T0..TN chains (sequential, even-position, odd-position)
  combined with every core leet op are generated independently of stem
  lookups and injected directly into Phase S sbd + Phase 2 seed pool.
  Captures high-frequency patterns observed in practice:
    "T0 T1 T2 T3 T4 T5 T6 T7 se3"  →  mixed-case + e→3 on 8-char words
    "T0 T2 T4 T6 sa@"               →  alternating toggle + a→@

  Phase S injection
  ─────────────────
  All Phase 0 chains (from all modes) are injected into the Phase S
  seed-by-depth (sbd) pool before the GPU sweep.  Depth slots not present
  in sbd (e.g. depth 10 with --max-depth 10) are created automatically.

  Single-rule discoveries are merged into the Phase 1 atomic-rule pool;
  multi-rule chains are forwarded to Phase 2 as seed chains.

  CLI flags
  ─────────
    --token-strip                 Enable Phase 0 (default: disabled)
    --token-strip-min-stem N      Minimum stem length (default: 4)
    --token-strip-max-prefix N    Max boundary prefix/suffix length (default: 4)
    --token-strip-min-leet-amb N  Max ambiguous leet positions per word
                                  (default: 3; limits branching for '1')

Phase 3 — Genetic Algorithm Rule Evolution  (--genetic)
  An optional evolutionary search that runs after Phase 2 and complements
  random chain sampling with guided, coverage-driven optimisation.

  Why it fits this project
  ────────────────────────
  • The fitness function (bloom-filter hits) is already computed by the
    existing GPU chain kernel — no new GPU code is required.
  • Phase 2 samples chains *uniformly at random* from the atomic-rule pool.
    For depth ≥ 3 the search space is |pool|^depth (millions of candidates);
    the GA focuses probability mass on high-hit-rate regions of that space.
  • Hot atomic rules from Phase 1 seed the initial population, giving the GA
    a strong head start rather than searching from scratch.
  • **Improvement (v2)**: the original 40 % purely random portion of the
    initial population is now replaced by high-hit chains from Phase S
    (families A–M) when builtin seeds are enabled. This dramatically
    improves starting coverage while gracefully falling back to random
    when --no-builtin-seeds is used.
  • **Improvement (v2)**: three key fixes make Phase 3 produce genuinely
    new rules rather than rediscovering Phase 2 results:
      1. Novelty-weighted fitness — chains NOT already found by Phase 1/S/2
         receive a 2× fitness bonus so the GA is driven toward new territory
         rather than cycling through already-known high-scorers.
      2. Unexplored-seed initial population — the 40 % Phase-S fill slot
         now prefers seeds NOT yet in all_counts; known seeds are used only
         as fallback.  Depth-3+ chains are also biased (70 % probability)
         in the seeded and fill portions when max_depth ≥ 3.
      3. Dedicated time budget — 20 % of --target-hours (min 120 s) is
         reserved for Phase 3 instead of relying on leftover scraps after
         Phase 1+S+2.  This guarantees the GA actually runs.
  • Stagnation detection — if the best fitness score does not improve for
    5 consecutive generations, the bottom 30 % of the population is replaced
    with fresh random chains, preventing premature convergence.
  • All Phase-3 discoveries are merged into the global hit counter before
    signature-based minimisation, so they benefit from the same deduplication
    and sorting as Phase 1 and Phase 2 results.

  Algorithm summary
  ─────────────────
  1. Initial population — 30 % depth-2 hot-rule combos (Phase 1),
     30 % seeded deeper chains (Phase 1 hot + random, depth-3+ biased),
     40 % unexplored Phase-S chains (NOT in known_rules, novel-first)
     — ensures both exploitation and exploration with far better
     starting coverage when seeds are active.
  2. GPU-batch fitness evaluation — returns raw bloom hits; novelty
     weighting (× 2 for chains not in known_rules) is applied in the
     evolve loop so raw counts are always stored honestly.
  3. Incremental signature registry (_sig_to_best) — after every GPU
     eval, every hit chain is indexed by its functional signature
     (compute_rule_signature / BUILTIN_PROBES).  The representative of
     each equivalence class is added to known_rules so that functionally
     equivalent variants receive no bonus in subsequent generations.
  4. Tournament selection (k = 4) — low-pressure, maintains diversity.
  5. One-point crossover (p = 0.80) — exchanges rule-token sub-sequences.
  6. Adaptive mutation (_mutate_adaptive) — apply standard mutation;
     if result is functionally covered by _sig_to_best, apply up to
     2 extra escape mutations to break out of the equivalence class.
  7. Signature-based offspring filter — offspring still covered after
     adaptive mutation are replaced by fresh random chains, keeping the
     population structurally diverse generation-over-generation.
  8. Elitism — top <elite_frac> individuals survive unchanged each gen.
  9. Stagnation guard — if best effective score does not improve for
     5 generations, the bottom 30 % of the population is replaced with
     fresh random chains (depth 3+ biased).
  10. Terminates when <--genetic-generations> is reached or wall-clock
      time budget (--target-hours reservation) is exhausted.

  CLI flags
  ─────────
    --genetic                   Enable Phase 3 (default: disabled)
    --genetic-generations N     Max generations (default: 50)
    --genetic-pop N             Population size (default: 200)
    --genetic-elite F           Elite fraction, e.g. 0.15 (default: 0.15)
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
 ██╔══██╗██║   ██║██║     ██═══╝  ╚════██║   ██║
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

# Sentinel used in signatures when a rule contains unsupported opcodes
_UNSUPPORTED_SENTINEL = object()

# ── Minimization constants ───────────────────────────────────────────────────
# Above this rule count, minimize_by_signature switches to a disk-backed
# SQLite path to avoid storing millions of signature tuples in Python heap.
# Each in-memory signature tuple costs ~2.5 KB (40 probe-word outputs).
# At 8 M rules that is ~20 GB RAM → OOM / core dump.
# The disk path caps peak Python heap to O(MINIMIZE_DISK_BATCH_SIZE) rows.
MINIMIZE_DISK_THRESHOLD  = 500_000   # rules; switch to disk above this value
MINIMIZE_DISK_BATCH_SIZE =  10_000   # rows per SQLite executemany call

# ── Special-character seed constants ────────────────────────────────────────
# Ordered by real-world frequency of appearance as password suffix/prefix.
SPECIAL_CHARS_TOP  = ['!', '@', '#', '$', '%', '^', '&', '*',
                      '?', '.', '-', '_', '+', '(', ')']
# Reduced "core" set used where combinatorial explosion must be limited.
SPECIAL_CHARS_CORE = ['!', '@', '#', '$', '%', '*', '?']

# ── Leet-substitution seed constants (Family J / M) ─────────────────────────
# Ten most common character→character leet mappings found in real passwords,
# ordered by real-world frequency.  Each entry is (original, replacement).
LEET_SUBS: List[Tuple[str, str]] = [
    ('a', '@'),   # p@ssword
    ('e', '3'),   # s3cur1ty
    ('o', '0'),   # passw0rd
    ('i', '1'),   # pass1word
    ('l', '1'),   # 1eet
    ('s', '5'),   # pa55word
    ('s', '$'),   # pa$$word
    ('t', '7'),   # 7error
    ('a', '4'),   # p4ssword
    ('i', '!'),   # pass!word
]
# Pre-built hashcat rule strings derived from LEET_SUBS (e.g. "sa@").
LEET_OPS: List[str] = [f's{orig}{rep}' for orig, rep in LEET_SUBS]

# ── Token-Strip constants (Phase 0) ──────────────────────────────────────────
# Leet decode table used in Phase 0 to reverse substitutions found in target
# passwords.  Each entry is (encoded_char, base_char, hashcat_sub_rule).
# Multiple entries may share the same encoded_char (e.g. '1' → 'i' or 'l').
TOKEN_STRIP_LEET_TABLE: List[Tuple[str, str, str]] = [
    ('@', 'a', 'sa@'),   # p@ssword  ← password
    ('3', 'e', 'se3'),   # s3cur1ty  ← security
    ('0', 'o', 'so0'),   # passw0rd  ← password
    ('1', 'i', 'si1'),   # pass1word ← passiword  (ambiguous: also 'l')
    ('1', 'l', 'sl1'),   # 1eet      ← leet
    ('5', 's', 'ss5'),   # pa55word  ← password
    ('$', 's', 'ss$'),   # pa$$word  ← password
    ('7', 't', 'st7'),   # 7error    ← terror
    ('4', 'a', 'sa4'),   # p4ssword  ← password
    ('!', 'i', 'si!'),   # pass!word ← passiword
]

# Per-char lookup: encoded_char → [(base_char, rule_str), ...]
_TOKEN_STRIP_LEET_BY_CHAR: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
for _ts_enc, _ts_base, _ts_rule in TOKEN_STRIP_LEET_TABLE:
    _TOKEN_STRIP_LEET_BY_CHAR[_ts_enc].append((_ts_base, _ts_rule))

# Set of characters that can be leet-decoded inside a middle segment
TOKEN_STRIP_LEET_CHARS: Set[str] = set(_TOKEN_STRIP_LEET_BY_CHAR.keys())

# Characters allowed in boundary prefix/suffix positions (digits + common specials)
# Used in LETTER mode: letters form the stem, digits/specials are boundary (^/$).
TOKEN_STRIP_BOUNDARY: Set[str] = set('0123456789!@#$%^&*?.-_+()')

# Boundary set for DIGIT mode: letters + specials form the boundary (^/$),
# pure digit sequences form the stem.  Selected when target word has more
# digits than letters — common for passwords like "123abc456" or "abc2024".
TOKEN_STRIP_ALPHA_BOUNDARY: Set[str] = (
    set(string.ascii_letters) | set('!@#$%^&*?.-_+()')
)

# ====================================================================
# --- BUILT-IN PROBE SET (hand‑curated, covers all important cases) ---
# ====================================================================
# This set replaces the old sampling‑based probe set.  It ensures that
# functional equivalence is detected even when the base wordlist lacks
# long words or has an unusual distribution.
BUILTIN_PROBES: List[str] = [
    # ── very short — edge cases for k, K, {, }, [, ] ────────────────
    "ab",
    "abc",
    "abcd",
    # ── short alphanumeric (len 4–6) ─────────────────────────────────
    "pass",
    "root",
    "test",
    "admin",
    "login",
    # ── typical password base words (len 7–9) ────────────────────────
    "letmein",          # len 7
    "welcome",          # len 7
    "password",         # len 8
    "sunshine",         # len 8
    "football",         # len 8
    "baseball",         # len 8
    "princess",         # len 8
    "dragon12",         # len 8, ends with digits
    # ── longer words (len 10+) — truncation / repeat ops ─────────────
    "qwertyuiop",       # len 10
    "iloveyou12",       # len 10, trailing digits
    "monkey12345",      # len 11
    "superman123",      # len 11
    "mustang2024",      # len 11
    # ── mixed-case — l/u/c/C/t/E/T/k/K ─────────────────────────────
    "Password",
    "AdminUser",
    "MySecret",
    "HelloWorld",
    # ── words with embedded digits — s, o, @, T ──────────────────────
    "pass123",
    "admin2024",
    "test1234",
    "user9999",
    # ── words with special chars — @ removal, s substitution ─────────
    "p@ssw0rd",
    "s3cur1ty",
    # ── leet-substitution targets — exercises Family J/M ─────────────
    "master",           # sa@ → m@ster, se3 → mast3r
    "leet",             # sl1 → 1eet
    "elite",            # si1 → e11te
    "access",           # sa@ → @ccess, ss$ → acce$$
    # ── repeated chars — q (double each), z/Z (extend) ───────────────
    "aaaa",
    "bbbb",
]
# Deduplicate while preserving order (already unique, but safe)
_seen = set()
_deduped = []
for w in BUILTIN_PROBES:
    if w not in _seen:
        _seen.add(w)
        _deduped.append(w)
BUILTIN_PROBES = _deduped
del _seen, _deduped, w

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
    def dg(c):
        if '0' <= c <= '9': return ord(c) - 48
        if 'A' <= c <= 'Z': return ord(c) - 55   # A=10, B=11, …, Z=35
        return -1
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
            # Title-case: lowercase everything first, then uppercase after space/hyphen/underscore.
            out = []; cap = True
            for c in w:
                if cap and 97 <= c <= 122:
                    out.append(c & ~0x20)      # lowercase → uppercase (word start)
                elif not cap and 65 <= c <= 90:
                    out.append(c | 0x20)       # uppercase → lowercase (mid-word)
                else:
                    out.append(c)
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
            # 'N — keep only the first N characters (w[:N])
            p = dg(rule[1])
            if 0 <= p: w = w[:p]
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
            # Title-case with custom separator: lowercase everything, then uppercase after sep.
            sep = ord(rule[1]); out = []; cap = True
            for c in w:
                if cap and 97 <= c <= 122:
                    out.append(c & ~0x20)
                elif not cap and 65 <= c <= 90:
                    out.append(c | 0x20)
                else:
                    out.append(c)
                cap = (c == sep)
            w = out
        elif cmd == 'x' and len(rule)==3:
            # xNM — extract M characters starting at position N  (M is a count, not end-index)
            n, m = dg(rule[1]), dg(rule[2])
            if n >= 0 and m >= 0: w = w[n:n+m]
        elif cmd == 'O' and len(rule)==3:
            p, m = dg(rule[1]), dg(rule[2])
            if 0 <= p < len(w) and m > 0: w = w[:p] + w[p+m:]
        elif cmd == '*' and len(rule)==3:
            a, b = dg(rule[1]), dg(rule[2])
            if 0<=a<len(w) and 0<=b<len(w) and a!=b: w[a], w[b] = w[b], w[a]
        elif cmd == '3' and len(rule)==3:
            # 3NX — toggle after the Nth separator X (N is 0-based: 30X = first sep)
            n, sep = dg(rule[1]), ord(rule[2]); cnt = 0
            for i, c in enumerate(w):
                if c == sep:
                    cnt += 1
                    if cnt == n + 1 and i+1 < len(w):
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

def compute_rule_signature(rule: str, probe_words: List[str]) -> tuple:
    """
    Compute the functional signature of *rule* over *probe_words*.

    The signature is a tuple of outputs (one per probe word).  When the rule
    contains an unsupported opcode, a UNIQUE sentinel tuple is returned:
    ``('__UNSUPPORTED__', rule)`` — embedding the rule text ensures that two
    different unsupported rules never share a bucket and collapse to one.

    The old behaviour — returning the shared constant ``('__UNSUPPORTED__',)``
    for every unsupported rule — caused false deduplication: e.g. 200 rules
    using reject ops (<, >, !, /, …) all mapped to the same bucket, keeping
    only the highest-count survivor and silently discarding the other 199.
    """
    outputs = []
    for word in probe_words:
        out = py_apply_chain(rule, word)
        if out is None:
            return ('__UNSUPPORTED__', rule)   # unique per rule
        outputs.append(out)
    return tuple(outputs)


def _is_unsupported_sig(sig: tuple) -> bool:
    """Return True if *sig* is an unsupported-opcode sentinel (any variant)."""
    return len(sig) >= 1 and sig[0] == '__UNSUPPORTED__'


def minimize_by_signature(
    rule_counter: Counter,
    probe_words:  List[str],
) -> Counter:
    """
    Deduplicate *rule_counter* by functional signature.

    Dispatcher — selects in-memory or disk-backed path based on
    ``MINIMIZE_DISK_THRESHOLD``:

    * ``len(rule_counter) <= MINIMIZE_DISK_THRESHOLD`` → ``_minimize_mem``
      Classic in-memory dict approach.  Fast for small/medium sets.

    * ``len(rule_counter) > MINIMIZE_DISK_THRESHOLD``  → ``_minimize_disk``
      SQLite-backed path.  Keeps Python heap to O(MINIMIZE_DISK_BATCH_SIZE)
      rows regardless of total rule count — safe for 8 M+ candidates.

    Both paths produce identical output (same tie-breaking, same Counter).

    Parameters
    ----------
    rule_counter : Counter
        Raw GPU bloom-filter hit counts keyed by rule string.
    probe_words : list
        Fixed list of words used to compute signatures.

    Returns
    -------
    Counter
        Surviving rules mapped to their original GPU hit-counts.
    """
    if not rule_counter:
        return Counter()

    log_section("POST-PROCESSING — Signature-Based Functional Minimization")
    n = len(rule_counter)
    log_info(f"[MINIMIZE] Candidates  : {bold(str(n))}")
    log_info(f"[MINIMIZE] Probe words : {bold(str(len(probe_words)))}")

    if n > MINIMIZE_DISK_THRESHOLD:
        log_info(
            f"[MINIMIZE] {cyan(f'{n:,} rules exceeds threshold {MINIMIZE_DISK_THRESHOLD:,}')} "
            f"— using {bold('disk-backed')} SQLite path to avoid OOM"
        )
        return _minimize_disk(rule_counter, probe_words)
    else:
        return _minimize_mem(rule_counter, probe_words)


def _minimize_mem(
    rule_counter: Counter,
    probe_words:  List[str],
) -> Counter:
    """
    In-memory signature minimization (original v1 algorithm).

    Stores all signature tuples as Python dict keys.  Each signature is a
    tuple of ``len(probe_words)`` strings.  Peak heap usage is proportional
    to ``len(rule_counter) * len(probe_words) * avg_output_length``.

    Safe up to roughly ``MINIMIZE_DISK_THRESHOLD`` rules (default 500 k).
    Above that the heap cost (~2.5 KB × N rules) causes OOM / core dumps on
    typical machines; use ``_minimize_disk`` instead.

    Algorithm
    ---------
    1. Compute signature = tuple of transformed probe outputs for every rule.
    2. Group rules sharing the same signature into equivalence classes.
    3. Keep the best representative per class (highest GPU hit-count;
       ties broken by shortest chain depth, then lexicographic rule order).
    4. Return a Counter of survivors with their original GPU counts.
    """
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
            sig = compute_rule_signature(rule, probe_words)
            sig_map[sig].append((rule, gpu_count))
            n_groups = len(sig_map)
            pbar.update(1)
            pbar.set_postfix({"unique_sigs": cyan(str(n_groups))}, refresh=False)

    def _group_key(item: Tuple[str, int]) -> tuple:
        rule, gpu_count = item
        return (-gpu_count, len(rule.split()), rule)

    survivors    = Counter()
    n_unsupported = 0
    for sig, group in sig_map.items():
        if _is_unsupported_sig(sig):
            n_unsupported += len(group)
        best_rule, best_count = min(group, key=_group_key)
        survivors[best_rule] = best_count

    _log_minimize_stats(len(rule_counter), survivors, len(sig_map), n_unsupported)
    return survivors


def _minimize_disk(
    rule_counter: Counter,
    probe_words:  List[str],
) -> Counter:
    """
    Disk-backed signature minimization for large rule sets (8 M+ candidates).

    Why in-memory fails at scale
    ----------------------------
    The in-memory path (``_minimize_mem``) stores every signature as a Python
    tuple of ``len(probe_words)`` strings in a dict.  Each tuple costs
    ~2.5 KB of heap (40 probe words × ~57 B/string + tuple overhead).
    At 8 M rules that is **~20 GB** of Python heap — well beyond what most
    machines can provide, causing an OOM kill or C-extension crash (core dump).

    How this path avoids the problem
    ---------------------------------
    Instead of keeping all signatures alive simultaneously, this function:

    1. Computes the signature for one rule at a time.
    2. Converts the signature to a compact SHA-1 hex digest (40 bytes) and
       immediately discards the full tuple.
    3. Writes ``(sig_hash, rule, count, depth)`` to a **temporary SQLite
       database on disk** in batches of ``MINIMIZE_DISK_BATCH_SIZE`` rows.
       SQLite's B-tree storage keeps only a small page cache in RAM
       (configured to 128 MB here).
    4. Uses ``INSERT … ON CONFLICT DO UPDATE`` so the table holds *at most
       one row per equivalence class* at all times — the best representative
       is kept in-place as better candidates arrive.
    5. After all rules are processed, reads the final survivors in a single
       ``SELECT rule, count FROM sig_best`` scan.
    6. Deletes the temporary file in a ``finally`` block.

    Peak Python heap is O(MINIMIZE_DISK_BATCH_SIZE × avg_rule_length) —
    roughly a few hundred KB regardless of total rule count.

    Tie-breaking
    ------------
    The SQL ``ON CONFLICT DO UPDATE`` clause implements the same three-level
    tie-break as the in-memory path:
        1. Highest GPU hit-count wins.
        2. Equal count → shorter chain depth wins.
        3. Equal count and depth → lexicographically smaller rule string wins.

    Hash collisions
    ---------------
    SHA-1 produces a 160-bit digest.  For 8 M rules the birthday-collision
    probability is ~(8e6)² / 2^161 ≈ 2.5 × 10⁻³⁵ — negligible in practice.
    Rules with unsupported opcodes receive a unique sentinel
    ``('__UNSUPPORTED__', rule_text)`` which joins to a unique string before
    hashing, so every unsupported rule gets its own row — no false deduplication.

    SQLite version requirement
    --------------------------
    ``ON CONFLICT DO UPDATE`` (upsert) requires SQLite ≥ 3.24 (June 2018).
    Python 3.8 ships with SQLite ≥ 3.31, so this is universally available
    on supported Python versions.

    Parameters
    ----------
    rule_counter : Counter
        Raw GPU bloom-filter hit counts keyed by rule string.
    probe_words : list
        Fixed list of words used to compute signatures.

    Returns
    -------
    Counter
        Surviving rules mapped to their original GPU hit-counts.
        Identical to what ``_minimize_mem`` would return.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix='.db', prefix='rulest_minimize_'
    )
    os.close(tmp_fd)

    try:
        conn = sqlite3.connect(tmp_path)

        # Performance pragmas — we accept losing the last batch on a hard
        # crash (acceptable for a disposable temp file).
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous  = OFF')
        conn.execute('PRAGMA temp_store   = MEMORY')
        conn.execute('PRAGMA cache_size   = -131072')   # 128 MB page cache

        conn.execute('''
            CREATE TABLE sig_best (
                sig_hash  TEXT    PRIMARY KEY,
                rule      TEXT    NOT NULL,
                count     INTEGER NOT NULL,
                depth     INTEGER NOT NULL
            )
        ''')
        conn.commit()

        # ON CONFLICT DO UPDATE — keep the best (rule, count, depth) per
        # equivalence class in a single pass, no post-processing needed.
        _UPSERT = '''
            INSERT INTO sig_best (sig_hash, rule, count, depth)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sig_hash) DO UPDATE SET
                rule  = CASE
                    WHEN excluded.count > sig_best.count
                        THEN excluded.rule
                    WHEN excluded.count  = sig_best.count
                     AND excluded.depth  < sig_best.depth
                        THEN excluded.rule
                    WHEN excluded.count  = sig_best.count
                     AND excluded.depth  = sig_best.depth
                     AND excluded.rule   < sig_best.rule
                        THEN excluded.rule
                    ELSE sig_best.rule
                END,
                count = CASE
                    WHEN excluded.count > sig_best.count
                        THEN excluded.count
                    ELSE sig_best.count
                END,
                depth = CASE
                    WHEN excluded.count > sig_best.count
                        THEN excluded.depth
                    WHEN excluded.count  = sig_best.count
                     AND excluded.depth  < sig_best.depth
                        THEN excluded.depth
                    ELSE sig_best.depth
                END
        '''

        rule_items  = list(rule_counter.items())
        n_total     = len(rule_items)
        batch: List[Tuple[str, str, int, int]] = []
        n_committed = 0    # rows flushed to SQLite so far (for unique-sig estimate)

        log_info(
            f"[MINIMIZE] Temp DB     : {dim(tmp_path)}"
        )

        with tqdm(
            total=n_total,
            desc=green("  Minimizing"),
            unit="rule",
            ncols=88,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}] {postfix}",
        ) as pbar:
            for rule, count in rule_items:
                sig        = compute_rule_signature(rule, probe_words)
                # Encode the full signature as a single Latin-1 string
                # using the NULL byte (0x00) as a field separator — it
                # cannot appear in normal hashcat rule outputs.
                sig_str    = '\x00'.join(sig)
                sig_hash   = hashlib.sha1(
                    sig_str.encode('latin-1', errors='replace')
                ).hexdigest()
                depth      = len(rule.split())
                batch.append((sig_hash, rule, count, depth))

                if len(batch) >= MINIMIZE_DISK_BATCH_SIZE:
                    conn.executemany(_UPSERT, batch)
                    conn.commit()
                    n_committed += len(batch)
                    batch.clear()

                    # Cheap estimate of unique signatures seen so far
                    (n_sigs,) = conn.execute(
                        'SELECT COUNT(*) FROM sig_best'
                    ).fetchone()
                    pbar.set_postfix(
                        {"unique_sigs": cyan(str(n_sigs))}, refresh=False
                    )
                    pbar.update(MINIMIZE_DISK_BATCH_SIZE)

            # Flush the final (partial) batch
            if batch:
                conn.executemany(_UPSERT, batch)
                conn.commit()
                pbar.update(len(batch))

        # Read survivors — one row per equivalence class
        survivors   = Counter()
        (n_sigs,)   = conn.execute('SELECT COUNT(*) FROM sig_best').fetchone()
        cursor      = conn.execute('SELECT rule, count FROM sig_best')
        for rule_str, cnt in cursor:
            survivors[rule_str] = cnt

        conn.close()

        # n_unsupported: each unsupported rule gets a unique sig_hash
        # ('__UNSUPPORTED__\x00' + rule_text), so ALL unsupported rules survive
        # the upsert — no false deduplication.  We report 0 here because we'd
        # need a second scan to count them accurately; the mem-path log covers it.
        _log_minimize_stats(n_total, survivors, n_sigs, n_unsupported=0)
        return survivors

    finally:
        try:
            os.unlink(tmp_path)
            log_debug(f"[MINIMIZE] Temp DB deleted: {tmp_path}")
        except OSError:
            log_warn(f"[MINIMIZE] Could not delete temp DB: {tmp_path}")


def _log_minimize_stats(
    n_input:       int,
    survivors:     Counter,
    n_groups:      int,
    n_unsupported: int,
) -> None:
    """Print post-minimization statistics (shared by both paths)."""
    removed = n_input - len(survivors)
    log_info(f"[MINIMIZE] {green('Done')}")
    log_info(f"           Unique signatures : {bold(cyan(str(n_groups))):>12s}")
    log_info(f"           Rules kept        : {bold(green(str(len(survivors)))):>12s}")
    log_info(f"           Rules removed     : {bold(red(str(removed))):>12s}  "
             f"({removed / max(1, n_input):.1%})")
    if n_unsupported:
        log_info(
            f"           Unsupported (kept 1 each group) : "
            f"{bold(str(n_unsupported))}"
        )
    log_debug(
        f"minimize_by_signature complete: "
        f"kept={len(survivors)}, removed={removed}, sig_groups={n_groups}"
    )

# ====================================================================
# --- PHASE 0 — TOKEN-STRIP RULE EXTRACTION ---
# ====================================================================

def _hashcat_title_case(s: str) -> str:
    """
    Simulate hashcat's 'E' (title-case) rule on an all-lowercase string.
    Capitalises the first letter and any letter immediately following a
    space, hyphen, or underscore.
    """
    result    = list(s)
    cap_next  = True
    for i, c in enumerate(result):
        if cap_next and 'a' <= c <= 'z':
            result[i] = c.upper()
            cap_next   = False
        if c in (' ', '-', '_'):
            cap_next = True
    return ''.join(result)


def _infer_case_rules(cased_stem: str) -> List[List[str]]:
    """
    Given a string of letters (possibly mixed case), return all candidate
    lists of hashcat case-transform ops that convert ``cased_stem.lower()``
    into ``cased_stem``.

    Candidates are ordered from shortest to longest so callers can try
    the cheapest option first.  Each inner list is one complete set of ops
    (e.g. ``['c']`` or ``['T0', 'T3']``).

    Returns an empty list if no pattern can be expressed within positions
    0–9 (hashcat's positional-op limit).
    """
    stem_lower = cased_stem.lower()

    if cased_stem == stem_lower:
        return [[]]                           # already lowercase — no op needed

    candidates: List[List[str]] = []

    # ── Single-op patterns (depth 1) ──────────────────────────────────────────
    # u — uppercase all
    if cased_stem == stem_lower.upper():
        candidates.append(['u'])

    # c — capitalize (first char upper, rest lower)
    if (len(cased_stem) >= 1
            and cased_stem[0] == cased_stem[0].upper()
            and cased_stem[1:] == cased_stem[1:].lower()):
        candidates.append(['c'])

    # C — lowercase first char, uppercase rest
    if (len(cased_stem) >= 1
            and cased_stem[0] == cased_stem[0].lower()
            and cased_stem[1:] == cased_stem[1:].upper()):
        candidates.append(['C'])

    # t — toggle every character's case
    toggled = ''.join(c.lower() if c.isupper() else c.upper() for c in stem_lower)
    if cased_stem == toggled:
        candidates.append(['t'])

    # E — title case (first letter of each word)
    if cased_stem == _hashcat_title_case(stem_lower):
        candidates.append(['E'])

    # ── Per-position TN ops (depth N) ─────────────────────────────────────────
    # TN toggles position N.  Only positions 0–9 are addressable.
    # Only emit TN ops when no single-op candidate (u/c/C/t/E) already
    # covers the pattern — TN ops are a fallback for irregular mixed-case
    # patterns that can't be expressed more compactly.
    if not candidates:
        uppercase_positions = [i for i, c in enumerate(cased_stem)
                               if c != stem_lower[i]]    # differs from lowercase
        if uppercase_positions and all(p <= 9 for p in uppercase_positions):
            tn_ops = [f'T{p}' for p in uppercase_positions]
            candidates.append(tn_ops)

    return candidates


def _leet_decode_variants(
    middle: str,
    max_ambiguous: int = 3,
) -> Iterator[Tuple[str, frozenset]]:
    """
    Yield ``(decoded, leet_rules)`` pairs for every way to replace leet
    characters in *middle* with their base-letter equivalents.

    decoded    : *middle* with leet chars substituted by letters (may still
                 contain uppercase letters from the original).
    leet_rules : ``frozenset`` of hashcat ``s`` rule strings applied.

    Only yields variants whose decoded form consists entirely of ASCII
    letters (a-z, A-Z) — non-decodable chars abort the generator.

    The branching factor is capped at *max_ambiguous* positions that have
    more than one possible mapping (e.g. ``'1'`` → ``'i'`` or ``'l'``).
    """
    # Abort early if any char is not a letter and not a known leet char
    for ch in middle:
        if not ch.isalpha() and ch not in TOKEN_STRIP_LEET_CHARS:
            return

    # Find leet positions and their candidate decodings
    leet_positions: List[Tuple[int, str, List[Tuple[str, str]]]] = []
    for i, ch in enumerate(middle):
        if ch in TOKEN_STRIP_LEET_CHARS:
            options = _TOKEN_STRIP_LEET_BY_CHAR[ch]  # [(base_char, rule_str), ...]
            leet_positions.append((i, ch, options))

    if not leet_positions:
        # No leet chars — yield as-is (already all letters due to guard above)
        yield (middle, frozenset())
        return

    # Limit combinatorial explosion from ambiguous positions
    n_ambiguous = sum(1 for _, _, opts in leet_positions if len(opts) > 1)
    if n_ambiguous > max_ambiguous:
        return

    # Enumerate all combinations of decodings
    choices_per_pos = [opts for _, _, opts in leet_positions]
    for combo in itertools.product(*choices_per_pos):
        decoded  = list(middle)
        rules    = set()
        for (pos, _orig, _opts), (base_ch, rule_str) in zip(leet_positions, combo):
            decoded[pos] = base_ch
            rules.add(rule_str)
        decoded_str = ''.join(decoded)
        if all(ch.isalpha() for ch in decoded_str):
            yield (decoded_str, frozenset(rules))


def _decode_middle(
    middle: str,
    max_ambiguous: int = 3,
) -> Iterator[Tuple[str, frozenset, List[List[str]]]]:
    """
    Yield ``(stem, leet_rules, case_candidates)`` triples for every way
    to decode the *middle* segment of a target password.

    stem            : all-lowercase base string to look up in the wordlist
    leet_rules      : ``frozenset`` of ``s`` rule strings for leet subs
    case_candidates : list of lists — each inner list is one set of case-
                      transform ops that converts *stem* into the cased
                      form of *middle* after leet decoding
    """
    for leet_decoded, leet_rules in _leet_decode_variants(middle, max_ambiguous):
        # leet_decoded is all ASCII letters (a-z / A-Z)
        cased_stem      = leet_decoded
        stem            = cased_stem.lower()
        case_candidates = _infer_case_rules(cased_stem)
        if not case_candidates:
            case_candidates = [[]]   # fall back: no case op (stem stays lowercase)
        yield (stem, leet_rules, case_candidates)


def _rule_chain_orderings(
    case_ops:    List[str],
    leet_ops:    List[str],
    prepend_ops: List[str],
    append_ops:  List[str],
    leading_ops: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Return candidate rule-chain orderings for the four op groups.

    leading_ops   : optional fixed prefix (e.g. ['r'], ['d'], ['[']) that is
                    prepended to every ordering unchanged.

    Canonical ordering : [leading] case → leet → prepend → append
    Alternative 1      : [leading] leet → case → prepend → append
    Alternative 2      : [leading] prepend → case → leet → append

    All orderings are verified by the caller via py_apply_chain.
    """
    lead = leading_ops or []
    seen: Set[tuple] = set()
    result: List[List[str]] = []

    def _add(ops: List[str]) -> None:
        full = lead + ops
        key  = tuple(full)
        if key not in seen:
            seen.add(key)
            result.append(full)

    _add(case_ops + leet_ops + prepend_ops + append_ops)          # canonical
    if case_ops and leet_ops:
        _add(leet_ops + case_ops + prepend_ops + append_ops)       # leet-first
    if prepend_ops and (case_ops or leet_ops):
        _add(prepend_ops + case_ops + leet_ops + append_ops)       # prefix-first

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers called by extract_token_strip_rules
# ──────────────────────────────────────────────────────────────────────────────

def _boundary_scan(
    word:           str,
    boundary_chars: Set[str],
    max_prefix_len: int,
    max_suffix_len: int,
    min_stem_len:   int,
) -> List[Tuple[str, str, str]]:
    """
    Return (prefix, middle, suffix) triples from *word* where:
      • prefix  consists entirely of *boundary_chars* (length 0..max_prefix_len)
      • suffix  consists entirely of *boundary_chars* (length 0..max_suffix_len)
      • middle  contains at least *min_stem_len* characters

    The boundary scan terminates early (break) as soon as a non-boundary
    character is encountered, matching the left-to-right scanning order used
    in the original extraction.
    """
    wlen    = len(word)
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


def _chains_from_middle(
    middle:          str,
    prefix:          str,
    suffix:          str,
    base_set:        Set[str],
    max_depth:       int,
    min_stem_len:    int,
    max_leet_amb:    int,
    leading_ops:     Optional[List[str]] = None,
) -> Set[str]:
    """
    Try to build rule chains that transform a base-wordlist stem into *middle*
    (after prepend/append boundary ops from *prefix*/*suffix* are applied).

    Handles leet decoding and case inference.  Verifies every candidate chain
    against the original target via py_apply_chain.
    """
    # The full original target word is always prefix + middle + suffix.
    # py_apply_chain(chain, stem) must equal this exactly.
    target_word = prefix + middle + suffix

    lead      = leading_ops or []
    lead_depth = len(lead)

    prepend_ops: List[str] = [f'^{c}' for c in reversed(prefix)]
    append_ops:  List[str] = [f'${c}' for c in suffix]
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
            for ops in _rule_chain_orderings(
                case_ops, leet_ops, prepend_ops, append_ops,
                leading_ops=lead,
            ):
                if not ops or len(ops) > max_depth:
                    continue
                chain = ' '.join(ops)
                if not HashcatRuleValidator.validate_rule_for_gpu(chain):
                    continue
                if py_apply_chain(chain, stem) == target_word:
                    found.add(chain)
    return found


def _extract_letter_mode(
    word:           str,
    base_set:       Set[str],
    max_depth:      int,
    min_stem_len:   int,
    max_prefix_len: int,
    max_suffix_len: int,
    max_leet_amb:   int,
) -> Set[str]:
    """
    LETTER MODE (original behaviour).
    Boundary chars = digits + specials; middle = letters (+ leet).
    Reconstructs target passwords of the form:
        [digits/specials] STEM [digits/specials]
    where STEM is a (possibly case-transformed / leet-substituted) word
    from the base wordlist.
    """
    found: Set[str] = set()
    for prefix, middle, suffix in _boundary_scan(
        word, TOKEN_STRIP_BOUNDARY, max_prefix_len, max_suffix_len, min_stem_len
    ):
        found |= _chains_from_middle(
            middle, prefix, suffix, base_set,
            max_depth, min_stem_len, max_leet_amb,
        )
    return found


def _extract_digit_mode(
    word:           str,
    base_set:       Set[str],
    max_depth:      int,
    min_stem_len:   int,
    max_prefix_len: int,
    max_suffix_len: int,
) -> Set[str]:
    """
    DIGIT MODE (new, dynamic boundary).
    Boundary chars = letters + specials; middle = pure digit sequence.
    Used when the target word contains more digits than letters, e.g.:
        "abc2024"  → stem "2024", boundary "abc" → rule '^c ^b ^a'
        "12345abc" → stem "12345", boundary "abc" → rule '$a $b $c'
    The digit sequence is looked up verbatim in the base wordlist (common
    numeric passwords such as "123456" appear in rockyou-style lists).
    No leet or case transforms apply to a pure-digit middle segment.
    """
    found: Set[str] = set()
    for prefix, middle, suffix in _boundary_scan(
        word, TOKEN_STRIP_ALPHA_BOUNDARY, max_prefix_len, max_suffix_len, min_stem_len
    ):
        if not middle.isdigit():
            continue
        if middle not in base_set:
            continue
        prepend_ops: List[str] = [f'^{c}' for c in reversed(prefix)]
        append_ops:  List[str] = [f'${c}' for c in suffix]
        total = len(prepend_ops) + len(append_ops)
        if total == 0 or total > max_depth:
            continue
        chain = ' '.join(prepend_ops + append_ops)
        if HashcatRuleValidator.validate_rule_for_gpu(chain):
            if py_apply_chain(chain, middle) == word:
                found.add(chain)
    return found


def _extract_reverse_mode(
    word:           str,
    base_set:       Set[str],
    max_depth:      int,
    min_stem_len:   int,
    max_prefix_len: int,
    max_suffix_len: int,
    max_leet_amb:   int,
) -> Set[str]:
    """
    REVERSE MODE — chain starts with 'r'.
    Checks whether the middle segment of the target, when reversed, decodes
    to a base-wordlist stem.  Handles leet and case transforms on the
    reversed middle, combined with boundary prepend/append ops.

    Example:
        target  "drowssap!"   stem "password" → rule  'r $!'
        target  "1drowssap"   stem "password" → rule  'r ^1'
        target  "3DROWSSAP"   stem "password" → rule  'r u se3'
    """
    found: Set[str] = set()
    if max_depth < 1:
        return found
    for prefix, middle, suffix in _boundary_scan(
        word, TOKEN_STRIP_BOUNDARY, max_prefix_len, max_suffix_len, min_stem_len
    ):
        rev_middle = middle[::-1]
        found |= _chains_from_middle(
            rev_middle, prefix, suffix, base_set,
            max_depth, min_stem_len, max_leet_amb,
            leading_ops=['r'],
        )
    return found


def _extract_duplicate_mode(
    word:         str,
    base_set:     Set[str],
    max_depth:    int,
    min_stem_len: int,
) -> Set[str]:
    """
    DUPLICATE / FOLD MODE — detect passwords formed by duplicating or folding
    a base-wordlist word, optionally with leading/trailing boundary ops.

    'd' (duplicate)  : stem → stem+stem
        e.g. "passwordpassword" from "password"  → rule 'd'
    'f' (fold)        : stem → stem+reverse(stem)
        e.g. "passworddrowssap" from "password"  → rule 'f'
    """
    found:  Set[str] = set()
    wlen   = len(word)

    for op, builder in (
        ('d', lambda s: s + s),
        ('f', lambda s: s + s[::-1]),
    ):
        for half in range(min_stem_len, wlen // 2 + 1):
            stem_cand = word[:half]
            expected  = builder(stem_cand)
            if expected == word and stem_cand in base_set:
                chain = op
                if HashcatRuleValidator.validate_rule_for_gpu(chain):
                    if py_apply_chain(chain, stem_cand) == word:
                        found.add(chain)
    return found


def _extract_delete_edge_mode(
    word:           str,
    base_set:       Set[str],
    max_depth:      int,
    min_stem_len:   int,
    max_prefix_len: int,
    max_suffix_len: int,
    max_leet_amb:   int,
) -> Set[str]:
    """
    DELETE-EDGE MODE — chain starts with '[' (delete first) or ']' (delete last).
    Useful when the target word has one extra character at the start or end
    that does not fit the normal boundary model.

    Example:
        target "Xpassword!"  stem "password" — the 'X' prefix char is not a
        typical boundary digit/special.  Trying '[' on 'Xpassword' after
        stripping the '!' suffix finds the stem.
    """
    found: Set[str] = set()
    if max_depth < 1 or len(word) < min_stem_len + 1:
        return found
    for op, trimmed in (('[', word[1:]), (']', word[:-1])):
        for prefix, middle, suffix in _boundary_scan(
            trimmed, TOKEN_STRIP_BOUNDARY,
            max_prefix_len, max_suffix_len, min_stem_len
        ):
            # The full original word must be reconstructed:
            # op transforms "prefix+middle+suffix_in_trimmed" which came from
            # original word.  py_apply_chain(chain, stem) == word handles this.
            found |= _chains_from_middle(
                middle, prefix, suffix, base_set,
                max_depth, min_stem_len, max_leet_amb,
                leading_ops=[op],
            )
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Toggle-chain seed generator (T0..TN patterns for Phase 2 injection)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_toggle_chain_seeds(max_depth: int) -> List[str]:
    """
    Generate T-position toggle chains for direct injection into Phase 2 as
    seed chains.  These are NOT derived from stem lookups — they are
    deterministic patterns that the empirical evidence shows produce large
    numbers of Phase-2 hits when used as seed scaffolding.

    Patterns generated
    ──────────────────
    1. Sequential  T0 T1 … TN   (N = 0..min(9, max_depth-1))
       Toggles the first N+1 characters.  Combined with every leet op.

    2. Even-position  T0 T2 T4 T6 T8
       Toggles even positions.  Combined with every leet op.

    3. Odd-position   T1 T3 T5 T7 T9
       Toggles odd positions.  Combined with every leet op.

    4. Single T0 prepended to every leet op (depth 2, highest utility per
       observation: "T0 se3", "T0 si1", "T0 sa@", …).

    Leet ops combined
    ─────────────────
    Each toggle base pattern is paired with:
      • every single leet op (depth base+1)
      • the two most common double-leet combos: se3+si1, sa@+so0 (depth base+2)

    All chains are validated via HashcatRuleValidator before being included.
    """
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

    # ── 1. Sequential T0 T1 … TN ──────────────────────────────────────────────
    for n in range(0, min(10, max_depth)):           # T0 alone … T0..T9
        t_ops = [f'T{i}' for i in range(n + 1)]
        _add(t_ops)                                   # pure toggle chain
        for leet in LEET_OPS:
            _add(t_ops + [leet])                      # toggle then leet
            if n >= 1:
                _add([leet] + t_ops)                  # leet then toggle
        for l1, l2 in DOUBLE_LEET:
            _add(t_ops + [l1, l2])                    # toggle + double leet
            _add([l1] + t_ops + [l2])                 # leet before and after

    # ── 2. Even positions T0 T2 T4 T6 T8 ─────────────────────────────────────
    for n in range(1, min(5, max_depth)):
        t_ops = [f'T{i * 2}' for i in range(n + 1)]
        _add(t_ops)
        for leet in LEET_OPS:
            _add(t_ops + [leet])
            _add([leet] + t_ops)
        for l1, l2 in DOUBLE_LEET:
            _add(t_ops + [l1, l2])

    # ── 3. Odd positions T1 T3 T5 T7 T9 ──────────────────────────────────────
    for n in range(1, min(5, max_depth)):
        t_ops = [f'T{i * 2 + 1}' for i in range(n + 1)]
        _add(t_ops)
        for leet in LEET_OPS:
            _add(t_ops + [leet])
            _add([leet] + t_ops)
        for l1, l2 in DOUBLE_LEET:
            _add(t_ops + [l1, l2])

    # ── 4. T0 + single leet (highest-utility depth-2 seeds) ───────────────────
    for leet in LEET_OPS:
        _add(['T0', leet])
        _add([leet, 'T0'])

    return sorted(seeds)


def extract_token_strip_rules(
    target_words:      List[str],
    base_set:          Set[str],
    max_depth:         int   = 0,
    min_stem_len:      int   = 4,
    max_prefix_len:    int   = 4,
    max_suffix_len:    int   = 4,
    max_leet_ambiguity: int  = 3,
) -> List[str]:
    """
    Phase 0 — Token-Strip rule extraction.

    Runs four complementary extraction modes for each target word:

    LETTER MODE (original)
        Boundary = digits + specials.  Middle segment = letters (+ leet chars).
        Produces case-transform / leet-substitution / prepend / append chains.
        Selected as primary when the word contains more letters than digits.

    DIGIT MODE (dynamic boundary)
        Boundary = letters + specials.  Middle segment = pure digit sequence.
        Digit middle is looked up verbatim in the base wordlist (common
        numeric passwords such as "123456" appear in rockyou-style lists).
        Selected as primary when the word contains more digits than letters.
        Produces prepend/append chains where letters/specials are boundary ops.

    REVERSE MODE
        Chain starts with 'r'.  The middle segment is reversed before stem
        lookup, generating rules like 'r $!' for "drowssap!" from "password".

    DELETE-EDGE MODE
        Chain starts with '[' or ']'.  Tries stripping one non-boundary
        character from the start or end of the word before normal extraction.

    DUPLICATE / FOLD MODE
        Detects passwords formed by duplicating ('d') or folding ('f') a
        base-wordlist word: "passwordpassword" → 'd', "passworddrowssap" → 'f'.

    Every candidate chain is verified by py_apply_chain before being accepted.

    Parameters
    ----------
    target_words       : words from the target wordlist
    base_set           : set of words from the base wordlist
    max_depth          : maximum rule chain length; 0 = MAX_HASHCAT_CHAIN
    min_stem_len       : reject stems shorter than this (default 4)
    max_prefix_len     : maximum boundary prefix length to try (default 4)
    max_suffix_len     : maximum boundary suffix length to try (default 4)
    max_leet_ambiguity : max ambiguous leet positions per word (default 3)

    Returns
    -------
    Sorted, deduplicated list of valid hashcat rule strings.
    """
    if max_depth <= 0:
        max_depth = MAX_HASHCAT_CHAIN

    found: Set[str] = set()

    with tqdm(total=len(target_words),
              desc=green("  Phase 0 "),
              unit="word",
              ncols=88,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
              ) as pbar:
        for word in target_words:
            pbar.set_postfix({"rules": cyan(str(len(found)))}, refresh=False)
            pbar.update(1)
            if not word or len(word) > MAX_WORD_LEN:
                continue

            n_digits  = sum(1 for c in word if c.isdigit())
            n_alpha   = sum(1 for c in word if c.isalpha())
            n_special = len(word) - n_digits - n_alpha  # noqa: F841

            # ── Dynamic mode selection ────────────────────────────────────────
            # Primary extraction mode is chosen based on which character class
            # dominates the word:
            #
            #   n_alpha >= n_digits (letters dominate or tie)
            #     → LETTER primary: boundary = digits+specials, stem = letters
            #       Example: "Password123!" → stem "password", boundary "123!"
            #
            #   n_digits > n_alpha (digits dominate)
            #     → DIGIT primary: boundary = letters+specials, stem = digits
            #       Example: "abc2024"   → stem "2024",   boundary "abc"
            #       Example: "123456!!"  → stem "123456", boundary "!!"
            #
            # Both modes still run — they catch different decompositions of the
            # same word and their results are unioned.

            digit_primary = n_digits > n_alpha

            if digit_primary:
                # Digit-heavy: digit-stem first, then letter-stem
                found |= _extract_digit_mode(
                    word, base_set, max_depth, min_stem_len,
                    max_prefix_len, max_suffix_len,
                )
                if n_alpha >= min_stem_len:
                    found |= _extract_letter_mode(
                        word, base_set, max_depth, min_stem_len,
                        max_prefix_len, max_suffix_len, max_leet_ambiguity,
                    )
            else:
                # Letter-heavy or mixed: letter-stem first, then digit-stem
                found |= _extract_letter_mode(
                    word, base_set, max_depth, min_stem_len,
                    max_prefix_len, max_suffix_len, max_leet_ambiguity,
                )
                if n_digits > 0:
                    found |= _extract_digit_mode(
                        word, base_set, max_depth, min_stem_len,
                        max_prefix_len, max_suffix_len,
                    )

            # ── REVERSE MODE ─────────────────────────────────────────────────
            if max_depth >= 2:
                found |= _extract_reverse_mode(
                    word, base_set, max_depth, min_stem_len,
                    max_prefix_len, max_suffix_len, max_leet_ambiguity,
                )

            # ── DELETE-EDGE MODE ─────────────────────────────────────────────
            if max_depth >= 2:
                found |= _extract_delete_edge_mode(
                    word, base_set, max_depth, min_stem_len,
                    max_prefix_len, max_suffix_len, max_leet_ambiguity,
                )

            # ── DUPLICATE / FOLD MODE ─────────────────────────────────────────
            if len(word) >= 2 * min_stem_len:
                found |= _extract_duplicate_mode(
                    word, base_set, max_depth, min_stem_len,
                )

    return sorted(found)


def _log_token_strip_stats(
    n_words:     int,
    rules:       List[str],
    inject_sbd:  bool,
) -> None:
    """Print Phase 0 summary statistics with per-mode rule prefix breakdown."""
    if not rules:
        log_info(f"[P0]   {yellow('0')} rules extracted by token-strip "
                 f"({n_words:,} target words scanned)")
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
        elif all(c.isdigit() or c in ('^', '$', ' ')
                 for c in r) and any(c.isdigit() for c in r):
            mode_counts['digit-bnd'] += 1
        else:
            mode_counts['letter'] += 1

    depth_summary = '  '.join(f"d{d}:{depth_dist[d]:,}" for d in sorted(depth_dist))
    inj = green('injected into Phase S sbd') if inject_sbd else dim('Phase S inactive')
    mode_str = '  '.join(f"{k}:{v}" for k, v in sorted(mode_counts.items()) if v)
    log_info(
        f"[P0]   {bold(green(str(len(rules))))} rules extracted by token-strip"
        f"  ({depth_summary})  → {inj}"
    )
    if mode_str:
        log_info(f"[P0]   Mode breakdown  : {dim(mode_str)}")


# --------------------------------------------------------------------
# GPU device helpers
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
# Dynamic parameters
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
# GPU-compatible rules generator
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
# OpenCL kernel
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
# GPU Engine
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

    def build_numeric_seed_families(self, max_depth: int = 4) -> dict:
        """
        Build nine seed families for Phase S direct extraction.

        Every family produces chains that are tested against the bloom filter
        individually via the GPU chain kernel — they are extraction candidates
        first, not only scaffolding for random chain generation.

        ── Numeric families (A–E) ──────────────────────────────────────────

        Family A — Pure Prepend (depths 1–4)
            One ^digit op per digit, right-to-left so the number reads
            correctly.  e.g. prepend 12 → "^2 ^1".
            Counts per depth: 10, 100, 1 000, 10 000.

        Family B — Pure Append (depths 1–4)
            One $digit op per digit, left-to-right.
            e.g. append 1990 → "$1 $9 $9 $0".
            Counts per depth: same as Family A.

        Family C — Mixed Prepend/Append (depths 1–4)
            All {^d, $d}^depth × digits^depth combinations, covering numeric
            bookends and interleaved prefix/suffix patterns.
            Counts per depth: 20^depth (20, 400, 8 000, 160 000).

        Family D — Transform + Digit/Bracket (depths 2–4)
            A single case-/position-transformation op at position 1
            (l u c C t r d f E k K { } [ ]), followed by 1–3 digit ops
            (^d, $d) or bracket ops ([ ]).  Depth 4 is the maximum.
            depth 2: transform + 1 op   e.g. "u $1", "l ^7", "c ["
            depth 3: transform + 2 ops  e.g. "u ^1 $9", "c [ ]", "t [ ["
            depth 4: transform + 3 ops  e.g. "u ^1 $2 ^9", "c [ ] ["

        Family E — Date Patterns (depths 4–9)
            Append and prepend orientations for the most common numeric date
            formats found in real passwords.  Date ranges:
              days 01–31, months 01–12,
              2-digit years 60–99 ∪ 00–30,
              4-digit years 1960–2030.
            depth 4:  DDMM, MMDD, YYYY          → append / prepend
            depth 5:  transform + 4-digit date   → all transform variants
            depth 6:  DDMMYY, MMDDYY            → append / prepend
            depth 6–8: 2–4 brackets + 4-digit date → bracket-prefix a/p
            depth 7–8: 1–2 brackets + 6-digit date → bracket-prefix a/p
            depth 8:  DDMMYYYY, MMDDYYYY        → append / prepend
            depth 9:  1 bracket  + 8-digit date  → bracket-prefix a/p
            (at most 4 bracket ops total)

        ── Special-character families (F–I) ────────────────────────────────

        Special chars used (by real-world frequency):
            TOP  (15): ! @ # $ % ^ & * ? . - _ + ( )
            CORE  (7): ! @ # $ % * ?   ← used where combos must stay bounded

        Family F — Pure Append Special Chars (depths 1–2)
            One or two $X ops for the top 15 special characters.
            depth 1 : 15 chains  ($!, $@, ...)
            depth 2 : up to 15² = 225 chains  ($! $!, $! $@, $1 $!, ...)

        Family G — Pure Prepend Special Chars (depths 1–2)
            One or two ^X ops (right-to-left order so the string reads
            correctly).
            depth 1 : 15 chains  (^!, ^@, ...)
            depth 2 : up to 15² chains

        Family H — Transform + Special Char (depths 2–3)
            A single transform op followed by 1–2 special-char append/
            prepend ops.  Uses the full top-15 set.
            depth 2: 15 transforms × 30 ops   = 450 chains
            depth 3: 15 transforms × 30² ops  = 13 500 chains

        Family I — Number + Special Char Combos (depths 2–4)
            The most common real-world suffix pattern: digits followed by
            a special character (e.g. "password1!", "password123!").
            Uses the CORE 7 special chars to keep depth-4 counts reasonable.
            depth 2: 1 digit  + 1 special char → 10 × 7 = 70 chains (append)
                                                  10 × 7 = 70 chains (prepend)
            depth 3: 2 digits + 1 special char → 100 × 7 = 700 + 700 chains
            depth 4: 3 digits + 1 special char → 1 000 × 7 = 7 000 + 7 000

            Append orientation : $d₁ … $dₙ $sp  → word<digits><sp>
            Prepend orientation: ^sp ^dₙ … ^d₁  → <sp><digits>word
              (prepend ops are applied right-to-left so the final string
               reads <digits><sp> as a prefix — no, see note below)

            Note on prepend order in hashcat:
              Each ^X prepends X on top of the current word.
              To obtain "<sp><digits>word" the ops must be applied in reverse
              reading order: last character prepended first, first character
              prepended last.
              e.g. to get "1!word" → apply ^! first → "!word", then ^1 → "1!word"
              So the rule chain is: ^! ^1  (special char first, then digits)

        Parameters
        ----------
        max_depth : int
            Upper bound on chain depth.  Seeds deeper than max_depth are
            skipped — the chain kernel cannot handle them anyway.

        Returns
        -------
        dict  depth -> set[str chain]
        """
        digits = '0123456789'
        sbd: Dict[int, set] = defaultdict(set)

        # ── Family A: Pure Prepend ────────────────────────────────────────
        # depths 1–4  (capped by max_depth so we never exceed the kernel limit)
        # Each depth d produces 10^d chains.
        a_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(1, min(4, max_depth) + 1):
            for combo in itertools.product(digits, repeat=depth):
                # right-to-left: prepending "12" → "^2 ^1"
                sbd[depth].add(' '.join(f'^{ch}' for ch in reversed(combo)))
                a_cnt[depth] += 1

        # ── Family B: Pure Append ─────────────────────────────────────────
        # depths 1–4  (capped by max_depth)
        # Each depth d produces 10^d chains.
        b_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(1, min(4, max_depth) + 1):
            for combo in itertools.product(digits, repeat=depth):
                # left-to-right: appending "1990" → "$1 $9 $9 $0"
                sbd[depth].add(' '.join(f'${ch}' for ch in combo))
                b_cnt[depth] += 1

        # ── Family C: Mixed Prepend/Append ────────────────────────────────
        # depths 1–4  (capped by max_depth)
        # All {^d, $d}^depth × digits^depth combinations → 20^depth per depth.
        for depth in range(1, min(4, max_depth) + 1):
            for ops in itertools.product(['^', '$'], repeat=depth):
                for digs in itertools.product(digits, repeat=depth):
                    sbd[depth].add(' '.join(f'{o}{d}' for o, d in zip(ops, digs)))

        # ── Family D: Transform + Digit/Bracket (depths 2–4) ──────────────
        # Only depths 2 to min(4, max_depth) are generated.
        transform_ops = [
            'l', 'u', 'c', 'C', 't', 'r', 'd', 'f',
            'E', 'k', 'K', '{', '}', '[', ']',
        ]
        t_digit_ops = (
            [f'^{d}' for d in digits] +
            [f'${d}' for d in digits] +
            ['[', ']']
        )
        d_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(2, min(4, max_depth) + 1):
            for t_op in transform_ops:
                for ops in itertools.product(t_digit_ops, repeat=depth - 1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed):
                        sbd[depth].add(seed)
                        d_cnt[depth] += 1

        # ── Family E: Date Patterns ───────────────────────────────────────
        _days   = [f"{d:02d}" for d in range(1, 32)]
        _months = [f"{m:02d}" for m in range(1, 13)]
        _years2 = ([f"{y:02d}" for y in range(60, 100)] +
                   [f"{y:02d}" for y in range(0,  31)])
        _years4 = [str(y) for y in range(1960, 2031)]

        _date4: set = set()   # DDMM, MMDD, YYYY   → depth 4
        _date6: set = set()   # DDMMYY, MMDDYY     → depth 6
        _date8: set = set()   # DDMMYYYY, MMDDYYYY  → depth 8

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

        e_cnt: Dict[int, int] = defaultdict(int)

        # Base depth 4 / 6 / 8 → append + prepend
        for _ds_set, _base_depth in ((_date4, 4), (_date6, 6), (_date8, 8)):
            if _base_depth > max_depth:
                continue
            for _ds in _ds_set:
                _app = ' '.join(f'${c}' for c in _ds)
                _pre = ' '.join(f'^{c}' for c in reversed(_ds))
                sbd[_base_depth].add(_app); e_cnt[_base_depth] += 1
                sbd[_base_depth].add(_pre); e_cnt[_base_depth] += 1

        # depth 5: transform + 4-digit date (all transform variants)
        if max_depth >= 5:
            for _ds in _date4:
                _app = ' '.join(f'${c}' for c in _ds)
                _pre = ' '.join(f'^{c}' for c in reversed(_ds))
                for t_op in transform_ops:
                    for _chain in (f"{t_op} {_app}", f"{t_op} {_pre}"):
                        if HashcatRuleValidator.validate_rule_for_gpu(_chain):
                            sbd[5].add(_chain); e_cnt[5] += 1

        # Bracket-prefix date variants (at most 4 bracket ops total):
        #   2–4 brackets + 4-digit date  → new depths 6, 7, 8
        #   1–2 brackets + 6-digit date  → new depths 7, 8
        #   1   bracket  + 8-digit date  → new depth  9
        _bracket_ops = ['[', ']']
        _bracket_date_schedule = [
            (_date4, 4, range(2, 5)),  # +2, +3, +4 → depth 6, 7, 8
            (_date6, 6, range(1, 3)),  # +1, +2     → depth 7, 8
            (_date8, 8, range(1, 2)),  # +1          → depth 9
        ]
        for _bds, _bdepth, _brange in _bracket_date_schedule:
            for _num_b in _brange:
                _new_depth = _bdepth + _num_b
                if _new_depth > max_depth:
                    continue
                for _brackets in itertools.product(_bracket_ops, repeat=_num_b):
                    _bpfx = ' '.join(_brackets)
                    for _ds in _bds:
                        _app = ' '.join(f'${c}' for c in _ds)
                        _pre = ' '.join(f'^{c}' for c in reversed(_ds))
                        for _chain in (f"{_bpfx} {_app}", f"{_bpfx} {_pre}"):
                            if HashcatRuleValidator.validate_rule_for_gpu(_chain):
                                sbd[_new_depth].add(_chain)
                                e_cnt[_new_depth] += 1

        # ── Family F: Pure Append Special Chars ──────────────────────────
        # depths 1–3  (capped by max_depth)
        # Covers the 15 most commonly appended special characters in real
        # passwords.  Depth 2 covers two-character suffixes (!!, !@, @!, …).
        # Depth 3 covers three-character suffixes such as "!!!" or "!@#".
        f_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(1, min(3, max_depth) + 1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                chain = ' '.join(f'${ch}' for ch in combo)
                if HashcatRuleValidator.validate_rule_for_gpu(chain):
                    sbd[depth].add(chain)
                    f_cnt[depth] += 1

        # ── Family G: Pure Prepend Special Chars ─────────────────────────
        # depths 1–3  (capped by max_depth)
        # Mirror of Family F but using the ^ (prepend) operator.
        # right-to-left order so the final string reads left-to-right.
        # Depth 3 adds three-char prefixes such as "!!!" or "!@#".
        g_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(1, min(3, max_depth) + 1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                # e.g. prepend "!@" → "^@ ^!"  (@ applied first, ! on top)
                chain = ' '.join(f'^{ch}' for ch in reversed(combo))
                if HashcatRuleValidator.validate_rule_for_gpu(chain):
                    sbd[depth].add(chain)
                    g_cnt[depth] += 1

        # ── Family H: Transform + Special Char (depths 2–3) ──────────────
        # A single case-/position-transform op followed by 1–2 special-char
        # append/prepend ops (both ^ and $, top-15 special chars each).
        # depth 2: 15 transforms × 30 ops        =   450 seeds
        # depth 3: 15 transforms × 30² ops        = 13 500 seeds
        sp_ops_top = (
            [f'${ch}' for ch in SPECIAL_CHARS_TOP] +
            [f'^{ch}' for ch in SPECIAL_CHARS_TOP]
        )
        h_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(2, min(3, max_depth) + 1):
            for t_op in transform_ops:
                for ops in itertools.product(sp_ops_top, repeat=depth - 1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed):
                        sbd[depth].add(seed)
                        h_cnt[depth] += 1

        # ── Family I: Number + Special Char Combos (depths 2–4) ──────────
        # Real-world pattern: word followed by digits then a special char,
        # e.g. "password1!", "password123!".
        # Uses SPECIAL_CHARS_CORE (7 chars) to limit depth-4 cardinality.
        #
        # Append orientation:  $d₁ … $dₙ $sp  → word<digits><sp>
        # Prepend orientation: ^sp ^dₙ … ^d₁  → <sp><digits>word
        #   (each ^ prepends on top, so the last char prepended ends up first)
        i_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(2, min(4, max_depth) + 1):
            n_digits = depth - 1   # number of digit ops; 1 slot for the special char
            for digit_combo in itertools.product(digits, repeat=n_digits):
                for sp in SPECIAL_CHARS_CORE:
                    # ---- Append: $d1 $d2 ... $sp  (e.g. "$1 $2 $3 $!")
                    app = ' '.join(f'${d}' for d in digit_combo) + f' ${sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(app):
                        sbd[depth].add(app)
                        i_cnt[depth] += 1
                    # ---- Prepend: ^sp ^dn ... ^d1
                    # Result: <sp><digits>word
                    # e.g. to prepend "1!" → apply ^! first → "!word",
                    #      then ^1 → "1!word", chain: "^! ^1"
                    pre = f'^{sp} ' + ' '.join(f'^{d}' for d in reversed(digit_combo))
                    if HashcatRuleValidator.validate_rule_for_gpu(pre):
                        sbd[depth].add(pre)
                        i_cnt[depth] += 1

        # ── Family J: Leet Substitutions (depths 1–2) ────────────────────
        # Ten most common character→character leet substitutions, drawn from
        # LEET_OPS (defined at module level).
        #
        # Depth 1: single leet op (e.g. "sa@" → "p@ssword" from "password").
        #   ~10 seeds — negligible cost, extremely high practical value.
        #
        # Depth 2a — leet + digit/special append or prepend:
        #   Each leet op combined with every digit/special append or prepend
        #   from SPECIAL_CHARS_CORE.
        #   10 leet × (10 $d + 10 ^d + 7 $sp + 7 ^sp) = 10 × 34 = 340 seeds
        #   Catches "p@ssword1", "p@ssword!", "1p@ssword", etc.
        #
        # Depth 2b — double-leet chains:
        #   Two distinct leet ops in sequence; covers passwords with multiple
        #   simultaneous substitutions (e.g. "sa@ so0" → "p@ssw0rd").
        #   10 × 9 = 90 seeds.
        j_cnt: Dict[int, int] = defaultdict(int)

        # Depth 1: pure leet substitutions
        for op in LEET_OPS:
            if HashcatRuleValidator.validate_rule_for_gpu(op):
                sbd[1].add(op)
                j_cnt[1] += 1

        if max_depth >= 2:
            leet_followup = (
                [f'${d}' for d in digits] +
                [f'^{d}' for d in digits] +
                [f'${ch}' for ch in SPECIAL_CHARS_CORE] +
                [f'^{ch}' for ch in SPECIAL_CHARS_CORE]
            )
            # 2a: leet + one append/prepend op
            for leet_op in LEET_OPS:
                for follow in leet_followup:
                    chain = f"{leet_op} {follow}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain):
                        sbd[2].add(chain)
                        j_cnt[2] += 1
            # 2b: two distinct leet substitutions (double-leet)
            for i_l, l1 in enumerate(LEET_OPS):
                for l2 in LEET_OPS:
                    if l1 != l2:
                        chain = f"{l1} {l2}"
                        if HashcatRuleValidator.validate_rule_for_gpu(chain):
                            sbd[2].add(chain)
                            j_cnt[2] += 1

        # ── Family K: Double-Transform Chains (depth 2) ──────────────────
        # All ordered pairs of pure transformation ops from transform_ops
        # (l u c C t r d f E k K { } [ ]).  No digit or special-char
        # appends; pure structural transforms only.
        #
        # depth 2: 15 × 15 = 225 pairs
        # Covers patterns like "c r" (capitalize+reverse), "u d"
        # (uppercase+duplicate), "t f" (toggle+fold), "E l" (title+lower),
        # "c {" (capitalize+rotate-left), "l ]" (lowercase+drop-last), etc.
        # None of these are generated by any other family, so there is
        # zero overlap and full coverage gain.
        k_cnt: Dict[int, int] = defaultdict(int)
        if max_depth >= 2:
            for t1 in transform_ops:
                for t2 in transform_ops:
                    chain = f"{t1} {t2}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain):
                        sbd[2].add(chain)
                        k_cnt[2] += 1

        # ── Family L: Special-before-Digit patterns (depths 2–3) ─────────
        # Reverse orientation of Family I: the special character is placed
        # BEFORE (not after) the digit sequence.
        # Captures the "word!1", "word@12", "!1word", "@12word" class that
        # Family I does not cover.
        # Uses SPECIAL_CHARS_CORE (7 chars) to keep cardinality bounded.
        #
        # Append orientation:  $sp $d₁ … $dₙ  → word<sp><digits>
        # Prepend orientation: ^dₙ … ^d₁ ^sp  → <digits><sp>word
        #   (prepend last char first; applying ^sp last puts sp leftmost)
        #
        # depth 2: 7 sp × 10 d = 70 append + 70 prepend = 140 seeds
        # depth 3: 7 sp × 100 dd = 700 append + 700 prepend = 1 400 seeds
        l_cnt: Dict[int, int] = defaultdict(int)
        for depth in range(2, min(3, max_depth) + 1):
            n_digits = depth - 1
            for sp in SPECIAL_CHARS_CORE:
                for digit_combo in itertools.product(digits, repeat=n_digits):
                    # ---- Append: $sp $d1 ... $dn  (e.g. "$! $1 $2" → "word!12")
                    app = f'${sp} ' + ' '.join(f'${d}' for d in digit_combo)
                    if HashcatRuleValidator.validate_rule_for_gpu(app):
                        sbd[depth].add(app)
                        l_cnt[depth] += 1
                    # ---- Prepend: ^dn ... ^d1 ^sp  → "<digits><sp>word"
                    # To get "12!word": apply ^! first, then ^2, then ^1.
                    # Chain: "^! ^2 ^1"  (digits in reverse reading order,
                    #                     special char last in the rule chain)
                    pre = ' '.join(f'^{d}' for d in digit_combo) + f' ^{sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(pre):
                        sbd[depth].add(pre)
                        l_cnt[depth] += 1

        # ── Family M: Leet + Transform chains (depth 2) ──────────────────
        # A leet substitution op paired with a structural transform op in
        # both orderings (leet-then-transform and transform-then-leet).
        #
        # leet-then-transform: apply the substitution first, then transform
        #   e.g. "sa@ c" → "p@ssword" → "P@ssword"  (leet then capitalise)
        #   e.g. "so0 u" → "passw0rd" → "PASSW0RD"  (leet then upper)
        # transform-then-leet: transform first, then substitute
        #   e.g. "c sa@" → "Password" → "P@ssword"  (capitalise then leet)
        #   e.g. "u so0" → "PASSWORD" → "PASSW0RD"  (upper then leet)
        #
        # depth 2: 10 leet × 15 transforms × 2 orderings = 300 seeds
        # After dedup (a few orderings produce identical output on some
        # words) the real count is typically ~280–295.
        m_cnt: Dict[int, int] = defaultdict(int)
        if max_depth >= 2:
            for leet_op in LEET_OPS:
                for t_op in transform_ops:
                    # leet → transform
                    chain_lt = f"{leet_op} {t_op}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain_lt):
                        sbd[2].add(chain_lt)
                        m_cnt[2] += 1
                    # transform → leet
                    chain_tl = f"{t_op} {leet_op}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain_tl):
                        sbd[2].add(chain_tl)
                        m_cnt[2] += 1

        # ── Debug summary ─────────────────────────────────────────────────
        c_total = sum(20 ** d for d in range(1, min(4, max_depth) + 1))
        log_debug(f"Seed families  max_depth={max_depth}")
        log_debug("  A (pure ^)           : " +
                  ", ".join(f"d{d}={a_cnt[d]:,}" for d in sorted(a_cnt)) +
                  f"  [{sum(a_cnt.values()):,} total]")
        log_debug("  B (pure $)           : " +
                  ", ".join(f"d{d}={b_cnt[d]:,}" for d in sorted(b_cnt)) +
                  f"  [{sum(b_cnt.values()):,} total]")
        log_debug(f"  C (mixed ^/$)        : [{c_total:,} total]")
        log_debug("  D (transform+digit)  : " +
                  ", ".join(f"d{d}={d_cnt[d]:,}" for d in sorted(d_cnt)) +
                  f"  [{sum(d_cnt.values()):,} total]")
        log_debug("  E (dates)            : " +
                  ", ".join(f"d{d}={e_cnt[d]:,}" for d in sorted(e_cnt)) +
                  f"  [{sum(e_cnt.values()):,} total]")
        log_debug("  F (append special)   : " +
                  ", ".join(f"d{d}={f_cnt[d]:,}" for d in sorted(f_cnt)) +
                  f"  [{sum(f_cnt.values()):,} total]")
        log_debug("  G (prepend special)  : " +
                  ", ".join(f"d{d}={g_cnt[d]:,}" for d in sorted(g_cnt)) +
                  f"  [{sum(g_cnt.values()):,} total]")
        log_debug("  H (transform+spec.)  : " +
                  ", ".join(f"d{d}={h_cnt[d]:,}" for d in sorted(h_cnt)) +
                  f"  [{sum(h_cnt.values()):,} total]")
        log_debug("  I (num+special)      : " +
                  ", ".join(f"d{d}={i_cnt[d]:,}" for d in sorted(i_cnt)) +
                  f"  [{sum(i_cnt.values()):,} total]")
        log_debug("  J (leet subs)        : " +
                  ", ".join(f"d{d}={j_cnt[d]:,}" for d in sorted(j_cnt)) +
                  f"  [{sum(j_cnt.values()):,} total]")
        log_debug("  K (double transform) : " +
                  ", ".join(f"d{d}={k_cnt[d]:,}" for d in sorted(k_cnt)) +
                  f"  [{sum(k_cnt.values()):,} total]")
        log_debug("  L (special-b-digit)  : " +
                  ", ".join(f"d{d}={l_cnt[d]:,}" for d in sorted(l_cnt)) +
                  f"  [{sum(l_cnt.values()):,} total]")
        log_debug("  M (leet+transform)   : " +
                  ", ".join(f"d{d}={m_cnt[d]:,}" for d in sorted(m_cnt)) +
                  f"  [{sum(m_cnt.values()):,} total]")
        log_debug("  A∪…∪M (all seeds)    : " +
                  ", ".join(f"d{d}={len(sbd[d]):,}" for d in sorted(sbd)) +
                  f"  [{sum(len(v) for v in sbd.values()):,} total]")
        return dict(sbd)

    def run_seed_extraction_pass(self, base_words: list, sbd: dict,
                                  bloom_filter, phase1_rules: list) -> Counter:
        """
        Run the numeric seed families as a *dedicated direct extraction pass*.

        Every seed chain is tested against the bloom filter individually via
        the GPU chain kernel — they are extraction candidates first, not only
        chain-building atoms.

        Depth-1 seeds are single rules (e.g. "^5") that are already covered
        by Phase 1; they are skipped here to avoid double-counting.
        Depth >= 2 seeds (e.g. "^1 ^2", "$3 $1 $4") are always run here,
        independent of --max-depth and the random-chain time budget.

        The method ensures self.rule_index is populated (reuses Phase 1's
        index) and runs the chain kernel in the same batches used by Phase 2.
        """
        # Ensure kernel + bloom filter are ready (Phase 1 sets these up)
        if self.bloom_buf is None:
            self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.gpu_rules  = HashcatRuleValidator.validate_rules_for_gpu(phase1_rules)
            self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}

        # Collect only depth >= 2 seeds (depth-1 already done in Phase 1)
        multi_seeds: List[str] = []
        for depth, chains in sorted(sbd.items()):
            if depth >= 2:
                multi_seeds.extend(chains)

        if not multi_seeds:
            log_info("[PS]   No multi-depth seeds to test (max_seed_depth=1 or none generated)")
            return Counter()

        total = sum(len(v) for d, v in sbd.items() if d >= 2)
        log_info(f"[PS]   Numeric seed pass: {total:,} chains across "
                 f"{sum(1 for d in sbd if d >= 2)} depth(s)")

        counter = Counter()
        cbs = self.params['CHAINS_PER_BATCH']
        wsb = self.params['WORD_SUB_BATCH']
        n_batches = (len(multi_seeds) + cbs - 1) // cbs

        with tqdm(total=n_batches,
                  desc=green("  SeedPass"),
                  unit="batch",
                  ncols=88,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  ) as pbar:
            for ci in range(0, len(multi_seeds), cbs):
                cb = multi_seeds[ci:ci + cbs]
                for wi in range(0, len(base_words), wsb):
                    wb = base_words[wi:wi + wsb]
                    if wb:
                        found = self._run_chain_kernel(wb, cb)
                        if found:
                            counter.update(found)
                    self.queue.finish()
                pbar.update(1)
                pbar.set_postfix({"hits": cyan(str(len(counter)))}, refresh=False)

        log_info(f"[PS]   {bold(green(str(len(counter))))} unique seed chains passed bloom filter")
        log_debug(f"Seed extraction pass complete: {len(counter)} hits")
        return counter

    def generate_informed_chains(self, rules, single_found, max_depth,
                                   seed_chains=None, prebuilt_sbd=None):
        """
        Generate random chain candidates for Phase 2.

        Chains are built entirely from atomic rules discovered in Phase 1
        (valid / hot).  Seeds play no role here:
          - Built-in seed families (A–E) are tested in Phase S only and are
            never used as scaffolding or candidates in this phase.
          - User-supplied single-rule seeds are already present in *valid*.
          - User-supplied multi-rule seed chains are injected as direct
            Phase 2 candidates but are NOT used as building blocks for
            further chain extension.

        *prebuilt_sbd* is accepted for signature compatibility but ignored.
        """
        # Cap max_depth to hashcat limit
        max_depth = min(max_depth, MAX_HASHCAT_CHAIN)
        valid   = [r for r in rules if HashcatRuleValidator.validate_rule_for_gpu(r)]
        if not valid: return []
        found_s = set(single_found.keys()) if single_found else set()
        hot     = [r for r in valid if r in found_s]
        # Phase 2 candidate set — starts with atomic rules only.
        chains  = set(valid)

        # ── User-supplied multi-rule seed chains (direct candidates only) ─────
        # Single-rule user seeds are already captured in *valid* above.
        # Multi-rule chains are added directly as candidates; they are not
        # used as building blocks for further extension.
        n_user_direct = 0
        if seed_chains:
            for sc in seed_chains:
                if sc.count(' ') >= 1:      # depth >= 2
                    chains.add(sc)
                    n_user_direct += 1
            if n_user_direct:
                log_debug(f"User seed chains injected as Phase 2 candidates: "
                          f"{n_user_direct}")

        # ── Random chain extension (atomic rules only) ────────────────────────
        # _gen_random_chains builds from *valid* / *hot* atoms exclusively.
        for depth in range(2, max_depth + 1):
            budget = self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 0)
            if budget <= 0: continue
            budget = min(budget, len(valid) ** depth)
            new    = self._gen_random_chains(depth, budget, valid, hot, chains, set())
            chains.update(new)
            log_debug(f"Depth {depth}: budget={budget:,}, generated={len(new):,}")

        log_debug(f"Total Phase 2 candidates: {len(chains):,}  "
                  f"(atomic rules + user chains + random extensions)")
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

# ====================================================================
# --- PHASE 3 — GENETIC ALGORITHM RULE EVOLVER ---
# ====================================================================
#
# Motivation
# ----------
# Phase 2 samples rule chains *uniformly at random* from the atomic-rule
# pool.  For long chains (depth >= 3) the search space grows as
# |pool|^depth, which makes random sampling extremely inefficient — most
# candidates score zero hits.
#
# A genetic algorithm (GA) addresses this by directing the search: chains
# that produce many bloom-filter hits ("high-fitness individuals") are
# preferentially recombined and mutated, so successive generations
# concentrate probability mass on promising regions of the rule space.
#
# Fitness function
# ----------------
# The fitness of a candidate chain is the number of unique base-word
# transformations that pass the bloom filter — identical to the hit-count
# already computed in Phases 1 and 2.  No new GPU infrastructure is
# required; the existing `_run_chain_kernel` is reused directly.
#
# Algorithm
# ---------
# 1. Initial population  — hot Phase-1 rules seeded into depth-2 combos
#                          plus random chains from the full rule pool.
# 2. Fitness evaluation  — GPU batch via _run_chain_kernel.
# 3. Tournament selection (k = 4)
# 4. One-point crossover on rule-token lists (crossover_p = 0.80)
# 5. Mutation            — replace / insert / delete one token
# 6. Elitism             — top <elite_frac> carried unchanged
# 7. Diversity guard     — duplicate individuals are re-seeded randomly
# 8. Repeat until <generations> reached or wall-clock budget exhausted.
#
# Integration
# -----------
# Activated by --genetic flag.  Runs after Phase 2, consuming whatever
# time remains from --target-hours.  Newly discovered chains are merged
# into `all_counts` before signature minimisation.
#
# v2 improvement: the original 40 % purely random portion of the initial
# population is now replaced by high-hit chains from Phase S (families A–M)
# when builtin seeds are enabled. This dramatically improves starting
# coverage while gracefully falling back to random when --no-builtin-seeds
# is used.

class GeneticRuleEvolver:
    """
    Phase 3 — Genetic Algorithm to evolve high-coverage hashcat rule chains.

    Parameters
    ----------
    gpu_engine      : GPUEngine instance (provides _run_chain_kernel).
    base_words      : List of base words to transform.
    rule_pool       : Validated atomic rules (Phase-1 rule set).
    max_depth       : Maximum rule chain length (tokens per chain).
    pop_size        : Number of individuals in each generation.
    elite_frac      : Fraction of the population kept unchanged (elites).
    tournament_k    : Tournament size for parent selection.
    crossover_p     : Probability that two parents exchange genetic material.
    mut_replace_p   : Relative probability of a replace mutation.
    mut_insert_p    : Relative probability of an insert mutation.
    mut_delete_p    : Relative probability of a delete mutation.
    seed_hits       : Optional Counter of Phase-S seed chains (high-hit
                      chains from families A–M). If provided, the original
                      40 % random portion of the initial population is
                      replaced by top-scoring Phase-S chains. Falls back
                      gracefully when Phase S is disabled.
    known_rules     : Optional set of chain strings already discovered by
                      Phase 1 / Phase S / Phase 2.  Used to:
                        (a) compute the novelty bonus in fitness evaluation
                        (b) prefer unexplored Phase-S chains in seeding
                        (c) seed the incremental signature registry so that
                            representatives are reported as known immediately.

    Improvements (v2)
    -----------------
    1. Incremental signature registry (_sig_to_best)
       After every GPU evaluation, chains with raw_hits > 0 are indexed by
       their functional signature (via compute_rule_signature / BUILTIN_PROBES).
       The representative of each equivalence class is added to known_rules so
       that the novelty bonus in subsequent generations is functionally aware,
       not just string-aware.  A chain that is functionally identical to an
       already-discovered rule receives no bonus even if it has a different
       string representation.

    2. Signature-based offspring filter
       Before accepting a child into next_pop the GA checks whether its
       functional signature is already present in _sig_to_best.  Covered
       offspring are escape-mutated; if still covered they are replaced by a
       fresh random chain.  This keeps the population structurally diverse and
       prevents it from filling up with equivalent rules.

    3. Adaptive mutation (_mutate_adaptive)
       The standard mutation is applied first.  If the result is functionally
       equivalent to a known sig, up to two additional escape mutations are
       attempted.  This costs at most 3× the single-mutation overhead per
       offspring and dramatically reduces the fraction of offspring that fall
       into already-covered equivalence classes.
    """

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

        # Normalise mutation probabilities so they always sum to 1.0
        _total = mut_replace_p + mut_insert_p + mut_delete_p
        if _total <= 0:
            _total = 1.0
        self._mut_weights = [
            mut_replace_p / _total,
            mut_insert_p  / _total,
            mut_delete_p  / _total,
        ]

        # v2: Phase S seed chains replace the original 40 % random portion
        self.seed_hits = seed_hits or Counter()
        self.seed_chains_sorted: List[str] = [
            r for r, _ in sorted(
                self.seed_hits.items(), key=lambda kv: -kv[1]
            )
        ] if self.seed_hits else []

        # v2: known_rules — chains already discovered by Phase 1 / Phase S / Phase 2.
        # Used to compute the novelty bonus in evaluate_population and to seed
        # the initial population with *unexplored* Phase-S chains.
        # v2: also updated dynamically as GA discovers new sig representatives.
        self.known_rules: set = known_rules if known_rules is not None else set()

        # v2: Incremental signature registry.
        #
        # _sig_cache  : lazy memoized signatures  (chain_str → tuple)
        # _sig_to_best: best-per-equivalence-class discovered during the GA
        #               (signature tuple → (chain_str, raw_hit_count))
        #
        # Updated by _update_sig_registry() after every GPU evaluation.
        # Used by _sig_is_covered() and _mutate_adaptive().
        self._sig_cache:   Dict[str, tuple]             = {}
        self._sig_to_best: Dict[tuple, Tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_chain(self, depth: int = 0) -> list:
        """Return a random chain of atomic rules as a token list."""
        if depth <= 0:
            depth = random.randint(2, self.max_depth)
        return [random.choice(self.rule_pool) for _ in range(depth)]

    def _clamp(self, tokens: list) -> list:
        """Ensure a token list has between 2 and max_depth elements."""
        lo, hi = 2, self.max_depth
        if len(tokens) < lo:
            tokens = tokens + self._random_chain(lo - len(tokens))
        return tokens[:hi]

    # ------------------------------------------------------------------
    # v2 — Signature registry helpers (Improvements 1, 2, 3)
    # ------------------------------------------------------------------

    def _get_sig(self, chain_str: str) -> tuple:
        """
        Return the functional signature of *chain_str*, computing and
        caching it on first access.

        The signature is the tuple of outputs produced by applying the chain
        to every word in BUILTIN_PROBES.  Rules with unsupported opcodes
        receive the sentinel ('__UNSUPPORTED__',).
        """
        if chain_str not in self._sig_cache:
            self._sig_cache[chain_str] = compute_rule_signature(
                chain_str, BUILTIN_PROBES
            )
        return self._sig_cache[chain_str]

    def _update_sig_registry(self, raw_hit_map: dict) -> int:
        """
        Update _sig_to_best from *raw_hit_map* (chain_str → raw GPU hits).

        Only chains with raw_hits > 0 are indexed — zero-hit chains carry
        no functionally useful information.

        Algorithm
        ---------
        For each chain with hits > 0:
          1. Compute (or retrieve cached) functional signature.
          2. If the signature is new, add the chain as the representative.
          3. If the signature already exists and this chain has a higher
             hit count, promote it to representative.
        After updating _sig_to_best, every representative is added to
        self.known_rules so that the novelty bonus in subsequent generations
        is functionally aware.

        Returns the number of *new* distinct signatures added in this call.
        """
        new_sigs = 0
        for chain_str, raw_hits in raw_hit_map.items():
            if raw_hits <= 0:
                continue
            sig = self._get_sig(chain_str)
            if _is_unsupported_sig(sig):
                continue
            existing = self._sig_to_best.get(sig)
            if existing is None:
                self._sig_to_best[sig] = (chain_str, raw_hits)
                new_sigs += 1
            elif raw_hits > existing[1]:
                # Better representative found — update but do NOT
                # remove the old one from known_rules (it's harmless there).
                self._sig_to_best[sig] = (chain_str, raw_hits)

        # Expose all current representatives via known_rules.
        # This ensures the novelty bonus fires correctly in the next
        # generation: any chain whose sig is already covered scores × 1.
        for _sig, (best_chain, _cnt) in self._sig_to_best.items():
            self.known_rules.add(best_chain)

        return new_sigs

    def _sig_is_covered(self, chain_str: str) -> bool:
        """
        Return True when *chain_str* is functionally equivalent to a chain
        already indexed in _sig_to_best.

        Chains with unsupported opcodes are never considered covered —
        they bypass the filter unconditionally.

        This check is O(probe_words) on a cache miss and O(1) on a hit.
        With 40 probe words and pop_size = 200, the amortised cost per
        generation is negligible (< 0.1 s).
        """
        if not self._sig_to_best:
            return False
        sig = self._get_sig(chain_str)
        if _is_unsupported_sig(sig):
            return False
        return sig in self._sig_to_best

    def _mutate_adaptive(self, tokens: list) -> list:
        """
        Adaptive mutation — Improvement 3.

        Apply the standard mutation operator.  If the resulting chain is
        functionally equivalent to one already in _sig_to_best, apply up
        to two additional escape mutations to break out of the covered
        equivalence class.

        At most three mutations are applied total (1 base + 2 escape
        attempts), keeping the overhead bounded and the individual still
        closely related to its parent.

        If all escape attempts fail (the covered equivalence class is very
        large), the final result is returned as-is; the signature filter in
        the breeding loop will handle it.
        """
        tokens = self._mutate(tokens)
        # Skip the extra check when the registry is empty — avoids paying
        # signature computation cost during the very first generation.
        if self._sig_to_best:
            for _attempt in range(2):
                if not self._sig_is_covered(' '.join(tokens)):
                    break
                tokens = self._mutate(tokens)
        return tokens

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def initial_population(self, hot_rules: list) -> list:
        """
        Build the initial population using three seeding strategies:

        1. Depth-2 combos of the top-50 hot Phase-1 rules  (30 %)
           These pairs are already "known good" atoms — crossing them
           often finds useful depth-2 chains immediately.

        2. Seeded deeper chains: one hot rule + random pool atoms  (30 %)
           Biased toward depth 3+ when max_depth >= 3 to explore regions
           that Phase 2's random sampling may have undersampled.

        3. Phase-S builtin seed families (A–M) — chains NOT already known
           are preferred (40 %). If all Phase-S seeds are already in
           known_rules, this portion falls back to random chains so the
           GA still starts with a diverse, unexplored population.
        """
        hot = hot_rules[:min(len(hot_rules), 50)]
        pop_set: set = set()

        n_hot    = int(self.pop_size * 0.30)
        n_seeded = int(self.pop_size * 0.30)
        # remainder (40 %) filled by unexplored Phase S or random

        # 1 — depth-2 hot pairs
        max_tries = n_hot * 20
        tries = 0
        while len(pop_set) < n_hot and tries < max_tries:
            tries += 1
            if len(hot) >= 2:
                a, b = random.sample(hot, 2)
            elif len(hot) == 1:
                a = hot[0]
                b = random.choice(self.rule_pool)
            else:
                break
            pop_set.add((a, b))

        # 2 — seeded deeper chains
        # Bias toward depth 3+ when max_depth >= 3: exploring deeper chains
        # is more valuable since Phase 2 already covers depth-2 exhaustively.
        max_tries = n_seeded * 20
        tries = 0
        while len(pop_set) < n_hot + n_seeded and tries < max_tries:
            tries += 1
            if self.max_depth >= 3:
                # Bias: 70 % chance of depth 3+, 30 % chance of depth 2
                depth = (random.randint(3, self.max_depth)
                         if random.random() < 0.70
                         else 2)
            else:
                depth = random.randint(2, self.max_depth)
            if hot:
                tokens = [random.choice(hot)] + [
                    random.choice(self.rule_pool) for _ in range(depth - 1)
                ]
                random.shuffle(tokens)
            else:
                tokens = self._random_chain(depth)
            pop_set.add(tuple(tokens))

        # 3 — Phase S seeds, preferring chains NOT yet in known_rules (40 %)
        n_fill = int(self.pop_size * 0.40)
        fill_set: set = set()

        if self.seed_chains_sorted:
            # Separate novel seeds (not yet discovered) from known seeds
            novel_seeds = [s for s in self.seed_chains_sorted
                           if s not in self.known_rules]
            known_seeds = [s for s in self.seed_chains_sorted
                           if s in self.known_rules]

            log_debug(
                f"[GA]   Phase-S seed pool: {len(novel_seeds)} novel, "
                f"{len(known_seeds)} already known"
            )

            # Prefer novel seeds; fall back to known ones only if needed
            candidate_pool = (
                novel_seeds if len(novel_seeds) >= n_fill // 2
                else (novel_seeds + known_seeds)
            )

            if candidate_pool:
                top_candidates = candidate_pool[:max(n_fill * 3, 100)]
                selected = random.sample(
                    top_candidates, k=min(n_fill, len(top_candidates))
                )
                for sc_str in selected:
                    tokens = sc_str.split()
                    if 2 <= len(tokens) <= self.max_depth:
                        fill_set.add(tuple(tokens))

            # Pad with random chains (preferring depth 3+) if insufficient
            max_fill_tries = n_fill * 20
            fill_tries = 0
            while len(fill_set) < n_fill and fill_tries < max_fill_tries:
                fill_tries += 1
                d = (random.randint(3, self.max_depth)
                     if self.max_depth >= 3 and random.random() < 0.60
                     else random.randint(2, self.max_depth))
                fill_set.add(tuple(self._random_chain(d)))
        else:
            # Phase S disabled or produced zero hits — pure random fallback
            max_tries = n_fill * 20
            tries = 0
            while len(fill_set) < n_fill and tries < max_tries:
                tries += 1
                d = (random.randint(3, self.max_depth)
                     if self.max_depth >= 3 and random.random() < 0.60
                     else random.randint(2, self.max_depth))
                fill_set.add(tuple(self._random_chain(d)))

        for ind in fill_set:
            pop_set.add(ind)

        result = [list(ind) for ind in pop_set]

        # Safety: pad to exactly pop_size if set was too sparse
        while len(result) < self.pop_size:
            d = (random.randint(3, self.max_depth)
                 if self.max_depth >= 3 and random.random() < 0.50
                 else random.randint(2, self.max_depth))
            result.append(self._random_chain(d))

        return result[:self.pop_size]

    # ------------------------------------------------------------------
    # Fitness evaluation (GPU-batch)
    # ------------------------------------------------------------------

    def evaluate_population(self, population: list) -> dict:
        """
        Evaluate the entire population in batched GPU kernel calls.

        Returns a dict {chain_string: raw_bloom_hit_count}.

        Raw counts (without any bonus multiplier) are returned so that the
        caller can apply novelty weighting, update the signature registry,
        and record honest GPU-frequency values — all in one consistent pass.

        Invalid chains (fail the hashcat validator) score 0 but are kept
        in the pool to maintain population diversity.
        """
        chain_strs   = [' '.join(tokens) for tokens in population]
        valid_chains = [
            c for c in chain_strs
            if HashcatRuleValidator.validate_rule_for_gpu(c)
        ]

        # Initialise all scores to 0
        raw_map: dict = {c: 0 for c in chain_strs}
        if not valid_chains:
            return raw_map

        wsb = self.gpu_engine.params.get('WORD_SUB_BATCH',   20_000)
        cbs = self.gpu_engine.params.get('CHAINS_PER_BATCH',  2_000)

        batch_hits: Counter = Counter()
        for ci in range(0, len(valid_chains), cbs):
            cb = valid_chains[ci:ci + cbs]
            for wi in range(0, len(self.base_words), wsb):
                wb = self.base_words[wi:wi + wsb]
                if wb:
                    found = self.gpu_engine._run_chain_kernel(wb, cb)
                    if found:
                        batch_hits.update(found)
            self.gpu_engine.queue.finish()

        raw_map.update(batch_hits)
        return raw_map

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(self, fitness_list: list) -> list:
        """
        Tournament selection.

        Randomly draw <tournament_k> individuals; return the token list of
        the one with the highest fitness score.
        """
        k          = min(self.tournament_k, len(fitness_list))
        contenders = random.sample(fitness_list, k)
        winner, _  = max(contenders, key=lambda x: x[1])
        return list(winner)

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(self, p1: list, p2: list) -> tuple:
        """
        One-point crossover on rule-token lists.

        A random cut point is chosen independently for each parent.
        p1[:cut1] + p2[cut2:] forms child 1; p2[:cut2] + p1[cut1:] forms
        child 2.  Both offspring are clamped to [2, max_depth] tokens.

        If crossover is skipped (random draw > crossover_p) the parents
        are returned unchanged, ensuring the operation is conservative.
        """
        if len(p1) < 2 or len(p2) < 2 or random.random() > self.crossover_p:
            return list(p1), list(p2)

        cut1 = random.randint(1, len(p1) - 1)
        cut2 = random.randint(1, len(p2) - 1)

        child1 = self._clamp(p1[:cut1] + p2[cut2:])
        child2 = self._clamp(p2[:cut2] + p1[cut1:])
        return child1, child2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, tokens: list) -> list:
        """
        Apply a single random mutation to a token list.

        Three operators (weights normalised at init):
          replace  — swap one token with a rule chosen from rule_pool
          insert   — insert one rule at a random position
                     (only if current length < max_depth)
          delete   — remove one token
                     (only if current length > 2)

        If the selected operator cannot be applied (length constraint),
        a replace mutation is performed as fallback.
        """
        tokens = list(tokens)
        op     = random.choices(['replace', 'insert', 'delete'],
                                weights=self._mut_weights)[0]

        if op == 'replace':
            idx         = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)

        elif op == 'insert' and len(tokens) < self.max_depth:
            idx = random.randint(0, len(tokens))
            tokens.insert(idx, random.choice(self.rule_pool))

        elif op == 'delete' and len(tokens) > 2:
            idx = random.randrange(len(tokens))
            tokens.pop(idx)

        else:
            # Fallback: replace
            idx         = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)

        return tokens

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(
        self,
        hot_rules:   list,
        generations: int,
        time_budget: float,
    ) -> Counter:
        """
        Run the genetic algorithm for up to *generations* generations or
        until *time_budget* seconds have elapsed (wall clock).

        Parameters
        ----------
        hot_rules     : Phase-1 hit rules, sorted descending by hit count.
        generations   : Hard cap on generation count.
        time_budget   : Maximum wall-clock seconds for Phase 3.

        Returns
        -------
        Counter
            All discovered chains that passed the bloom filter, mapped to
            their raw (non-bonus) hit count, highest value across all gens.
        """
        if not self.rule_pool:
            log_warn("[GA]   Rule pool is empty — skipping Phase 3.")
            return Counter()

        if time_budget <= 0:
            log_warn("[GA]   No time budget remaining — skipping Phase 3.")
            return Counter()

        if self.max_depth < 3:
            log_warn(
                "[GA]   max_depth is only 2 — Phase 2 already covers depth-2 "
                "exhaustively.  Consider --max-depth 3 or higher to get "
                "meaningful GA discoveries."
            )

        t_start = time.time()
        all_new: Counter = Counter()
        n_elite = max(1, int(self.pop_size * self.elite_frac))

        # Stagnation tracking
        STAGNATION_THRESHOLD = 5   # generations without improvement before refresh
        stagnation_counter   = 0
        best_ever_score      = 0

        log_info(
            f"[GA]   pop={self.pop_size}  max_gen={generations}  "
            f"elite={self.elite_frac:.0%}  budget={time_budget:.0f}s  "
            f"pool={len(self.rule_pool):,} rules  "
            f"known={len(self.known_rules):,}"
        )

        # --- Initialise population ---
        pop = self.initial_population(hot_rules)
        last_gen = 0

        with tqdm(
            total=generations,
            desc=green("  Phase 3 "),
            unit="gen",
            ncols=88,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        ) as pbar:

            for gen in range(generations):
                last_gen = gen

                # Time guard — checked at the start of every generation
                if time.time() - t_start >= time_budget:
                    log_debug(f"[GA]   Time budget exhausted at generation {gen}.")
                    break

                # ── Step 1: GPU fitness evaluation (raw hits, no bonus) ────────
                raw_map = self.evaluate_population(pop)

                # ── Step 2: Update incremental signature registry (Improvement 1)
                # Indexes every hit chain by functional signature.  Representatives
                # are added to known_rules so the novelty bonus is functionally
                # aware from the next generation onwards.
                new_sigs = self._update_sig_registry(raw_map)

                # ── Step 3: Update all_new with highest raw hit counts ─────────
                # pre_ga_known was captured before the GA started (passed in as
                # the original known_rules set).  We compare against all_new to
                # count chains that are genuinely new discoveries this generation.
                n_novel_this_gen = 0
                for chain_str, raw_hits in raw_map.items():
                    if raw_hits > 0 and HashcatRuleValidator.validate_rule_for_gpu(chain_str):
                        if raw_hits > all_new[chain_str]:
                            all_new[chain_str] = raw_hits
                        # "Novel this generation" = hit chain not yet in all_new
                        # before this update (i.e. first time it was discovered).
                        if all_new[chain_str] == raw_hits:
                            n_novel_this_gen += 1

                # ── Step 4: Build novelty-weighted fitness list for selection ──
                # Novelty bonus: chains not in known_rules score × 2.
                # known_rules now includes sig representatives discovered so far,
                # so functionally equivalent variants are penalised correctly.
                fitness_list = sorted(
                    [
                        (
                            tuple(ind),
                            raw_map.get(' '.join(ind), 0) * (
                                2 if ' '.join(ind) not in self.known_rules else 1
                            ),
                        )
                        for ind in pop
                    ],
                    key=lambda x: -x[1],
                )

                best_score = fitness_list[0][1] if fitness_list else 0

                # ── Step 5: Stagnation detection ──────────────────────────────
                if best_score > best_ever_score:
                    best_ever_score    = best_score
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter >= STAGNATION_THRESHOLD:
                    # Refresh the bottom 30 % of non-elite individuals with
                    # novel random chains to escape local optima.
                    stagnation_counter = 0
                    n_refresh = max(1, int(self.pop_size * 0.30))
                    log_debug(
                        f"[GA]   Stagnation ({STAGNATION_THRESHOLD} gens) — "
                        f"refreshing {n_refresh} individuals with random chains"
                    )
                    depth_bias = self.max_depth >= 3
                    refresh_chains = []
                    rt = 0
                    refresh_set = {fitness_list[0][0]} if fitness_list else set()
                    while len(refresh_chains) < n_refresh and rt < n_refresh * 20:
                        rt += 1
                        d = (random.randint(3, self.max_depth)
                             if depth_bias and random.random() < 0.60
                             else random.randint(2, self.max_depth))
                        ind = tuple(self._random_chain(d))
                        if ind not in refresh_set:
                            refresh_chains.append(list(ind))
                            refresh_set.add(ind)
                    keep_top = [list(ind) for ind, _ in fitness_list[:self.pop_size - n_refresh]]
                    pop = keep_top + refresh_chains
                    pop = pop[:self.pop_size]
                    pbar.update(1)
                    pbar.set_postfix(
                        {"best": cyan(str(best_score)),
                         "new":  cyan(str(len(all_new))),
                         "sigs": cyan(str(len(self._sig_to_best))),
                         "stag": yellow("REFRESH")},
                        refresh=False,
                    )
                    continue   # skip normal breeding this generation

                # ── Step 6: Elitism — carry top individuals unchanged ─────────
                elites   = [list(ind) for ind, _ in fitness_list[:n_elite]]
                next_pop = list(elites)
                next_set = {tuple(e) for e in elites}

                # ── Step 7: Breed remainder via selection + crossover + mutation
                # Improvements 2 & 3 are applied here:
                #   - _mutate_adaptive escapes covered sig classes (Improvement 3)
                #   - Offspring still landing in a covered sig after escape are
                #     replaced by a random chain (Improvement 2)
                depth_bias = self.max_depth >= 3
                max_breed_attempts = (self.pop_size - len(next_pop)) * 8
                breed_attempts     = 0
                n_sig_replaced     = 0   # diagnostic counter

                while len(next_pop) < self.pop_size and breed_attempts < max_breed_attempts:
                    breed_attempts += 1

                    p1 = self._tournament_select(fitness_list)
                    p2 = self._tournament_select(fitness_list)

                    child1, child2 = self._crossover(p1, p2)

                    # Adaptive mutation (Improvement 3): if child lands in a
                    # covered sig class, up to 2 extra escape mutations are applied.
                    child1 = self._mutate_adaptive(child1)
                    child2 = self._mutate_adaptive(child2)

                    for child in (child1, child2):
                        if len(next_pop) >= self.pop_size:
                            break
                        key = tuple(child)
                        if key in next_set:
                            continue

                        # Improvement 2: signature filter.
                        # If the child is still functionally covered after
                        # adaptive mutation, replace it with a random chain.
                        # This keeps the population structurally diverse.
                        if self._sig_is_covered(' '.join(child)):
                            n_sig_replaced += 1
                            d = (random.randint(3, self.max_depth)
                                 if depth_bias and random.random() < 0.70
                                 else random.randint(2, self.max_depth))
                            child = self._random_chain(d)
                            key   = tuple(child)
                            if key in next_set:
                                continue

                        next_pop.append(child)
                        next_set.add(key)

                # ── Step 8: Diversity fill — pad any remaining slots randomly ─
                fill_attempts = 0
                while len(next_pop) < self.pop_size and fill_attempts < self.pop_size * 4:
                    fill_attempts += 1
                    d = (random.randint(3, self.max_depth)
                         if depth_bias and random.random() < 0.50
                         else random.randint(2, self.max_depth))
                    ind = tuple(self._random_chain(d))
                    if ind not in next_set:
                        next_pop.append(list(ind))
                        next_set.add(ind)

                pop = next_pop[:self.pop_size]

                log_debug(
                    f"[GA]   gen={gen}  new_sigs={new_sigs}  "
                    f"sig_replaced={n_sig_replaced}  "
                    f"total_sigs={len(self._sig_to_best)}"
                )
                pbar.update(1)
                pbar.set_postfix(
                    {"best": cyan(str(best_score)),
                     "new":  cyan(str(len(all_new))),
                     "sigs": cyan(str(len(self._sig_to_best)))},
                    refresh=False,
                )

        elapsed = time.time() - t_start

        # Summary metrics:
        #   all_new           — every chain with ≥1 bloom hit (includes pre-GA known ones
        #                       whose score was updated by the GA)
        #   _sig_to_best      — distinct functional equivalence classes discovered by the GA
        #   sig_cache         — total signatures computed (chains × probe evaluations cached)
        n_chains      = len(all_new)
        n_sig_classes = len(self._sig_to_best)
        log_info(
            f"[GA]   Evolution complete — "
            f"{bold(green(str(n_chains)))} unique chains passed bloom filter  "
            f"({bold(cyan(str(n_sig_classes)))} distinct functional signatures)  "
            f"({elapsed:.1f}s, {last_gen + 1} generation(s))"
        )
        log_debug(
            f"Phase 3 GA complete: chains={n_chains}, "
            f"sig_classes={n_sig_classes}, "
            f"sig_cache={len(self._sig_cache)}, "
            f"elapsed={elapsed:.1f}s"
        )
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
                 token_strip_min_leet_amb=3):
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
        # Phase 0 — Token-Strip
        self.token_strip              = token_strip
        self.token_strip_min_stem     = token_strip_min_stem
        self.token_strip_max_prefix   = token_strip_max_prefix
        self.token_strip_max_suffix   = token_strip_max_suffix
        self.token_strip_min_leet_amb = token_strip_min_leet_amb
        self.params              = calculate_dynamic_parameters(
            base_count, target_count, None, target_hours, bloom_mb_override=bloom_mb)
        self.params['MAX_CHAIN_DEPTH'] = max_depth
        self.rules_gen           = GPUCompatibleRulesGenerator()
        self.gpu_engine          = GPUEngine(self.params)
        self.validator           = HashcatRuleValidator()

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

        # ── Seed rules: split by depth ──────────────────────────────────────
        # Single-rule seeds (no spaces) are injected into Phase 1 so they are
        # tested as standalone extraction candidates — not only as chain
        # building blocks.  Multi-rule chain seeds are forwarded to Phase 2
        # unchanged (they are already run directly there via generate_informed_chains).
        all_seeds    = self.load_seed_rules()
        seed_singles = [s for s in all_seeds if ' ' not in s.strip()]
        seed_chains  = [s for s in all_seeds if ' ' in  s.strip()]

        # Merge seed singles into Phase 1 rule set (dedup while preserving order)
        builtin_set  = set(rules)
        extra_seeds  = [s for s in seed_singles if s not in builtin_set]
        rules_phase1 = rules + extra_seeds          # Phase 1 rule list
        if extra_seeds:
            log_info(f"[SEED] {len(extra_seeds)} seed single-rule(s) added to Phase 1 "
                     f"({len(seed_singles) - len(extra_seeds)} already in built-in set)")
        if seed_chains and self.max_depth < 2:
            log_warn(f"[SEED] {len(seed_chains)} chain seed(s) ignored — "
                     f"requires --max-depth >= 2 to run Phase 2")
        log_debug(f"Seed split: {len(seed_singles)} singles → Phase 1, "
                  f"{len(seed_chains)} chains → Phase 2")
        # ────────────────────────────────────────────────────────────────────

        bloom_filter = self.gpu_engine.generate_bloom_filter(target_words)

        # --- Phase 0: Token-Strip Rule Extraction (CPU-only, optional) -------
        # Runs before Phase 1 so discovered rules can be injected into the
        # Phase 1 atomic pool (single-rule discoveries) and Phase S seed
        # families (multi-rule chains).
        ts_rules_singles: List[str] = []
        ts_rules_chains:  List[str] = []
        ts_sbd: Dict[int, set]      = defaultdict(set)    # for Phase S injection

        if self.token_strip:
            log_section("PHASE 0 — Token-Strip Rule Extraction")
            base_set_for_ts = set(base_words)
            log_info(
                f"[P0]   Scanning {len(target_words):,} target words  |  "
                f"base set {len(base_set_for_ts):,} words  |  "
                f"min-stem={self.token_strip_min_stem}  "
                f"max-prefix={self.token_strip_max_prefix}  "
                f"max-suffix={self.token_strip_max_suffix}  "
                f"min-leet-amb={self.token_strip_min_leet_amb}"
            )
            log_info(
                f"[P0]   Modes: {bold('letter')} (alpha-stem)  "
                f"{bold('digit')} (digit-stem, dynamic boundary)  "
                f"{bold('reverse')} (r-prefix)  "
                f"{bold('delete-edge')} ([/]-prefix)  "
                f"{bold('dup/fold')} (d/f)"
            )
            ts_all = extract_token_strip_rules(
                target_words,
                base_set_for_ts,
                max_depth          = self.max_depth,
                min_stem_len       = self.token_strip_min_stem,
                max_prefix_len     = self.token_strip_max_prefix,
                max_suffix_len     = self.token_strip_max_suffix,
                max_leet_ambiguity = self.token_strip_min_leet_amb,
            )

            # Split by depth
            for r in ts_all:
                depth = len(r.split())
                if depth == 1:
                    ts_rules_singles.append(r)
                else:
                    ts_rules_chains.append(r)
                    if depth <= self.max_depth:
                        ts_sbd[depth].add(r)       # schedule for Phase S injection

            _log_token_strip_stats(
                len(target_words), ts_all, inject_sbd=self.builtin_seeds
            )
            if ts_rules_singles:
                log_info(
                    f"[P0]   Single-rule discoveries  : "
                    f"{bold(cyan(str(len(ts_rules_singles))))}  "
                    f"→ merged into Phase 1 atomic pool"
                )
            if ts_rules_chains:
                log_info(
                    f"[P0]   Multi-rule chain discoveries : "
                    f"{bold(cyan(str(len(ts_rules_chains))))}  "
                    f"→ Phase S sbd injection + Phase 2 seed chains"
                )

            # ── Toggle-chain seeds ─────────────────────────────────────────────
            # Generate T0..TN + leet combo chains for Phase 2 injection.
            # These are NOT derived from stem lookups — they are deterministic
            # structural seeds based on empirical evidence of T-chain effectiveness.
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
                    log_info(
                        f"[P0]   Toggle-chain seeds       : "
                        f"{bold(cyan(str(n_toggle_new)))}  "
                        f"(T0..TN patterns + leet combos)  "
                        f"→ Phase S sbd + Phase 2 seeds"
                    )
        # ── Debug dump — write all Phase 0 rules to a sidecar file ───────────
        # Written only when --debug is active (VERBOSE=True).  The file is
        # placed next to the main output file with a ".phase0.txt" suffix and
        # contains every rule discovered by all extraction modes plus every
        # toggle-chain seed, sorted by depth then alphabetically.
        if VERBOSE and self.token_strip:
            _all_p0 = sorted(
                set(ts_rules_singles) | set(ts_rules_chains),
                key=lambda r: (len(r.split()), r),
            )
            _p0_path = getattr(self, '_output_path', 'rulest_output.txt')
            _p0_path = (_p0_path.rsplit('.', 1)[0] + '.phase0.txt'
                        if '.' in _p0_path else _p0_path + '.phase0.txt')
            try:
                with open(_p0_path, 'w', encoding='utf-8') as _f:
                    _f.write("# Phase 0 — Token-Strip Rule Extraction (debug dump)\n")
                    _f.write(f"# Singles  : {len(ts_rules_singles)}\n")
                    _f.write(f"# Chains   : {len(ts_rules_chains)}\n")
                    _f.write(f"# Total    : {len(_all_p0)}\n")
                    _f.write("#\n")
                    _f.write("# Columns: depth  rule\n")
                    _f.write("#\n")
                    for _r in _all_p0:
                        _f.write(f"{len(_r.split())}\t{_r}\n")
                log_debug(
                    f"[P0]   Debug dump written : {bold(_p0_path)}  "
                    f"({len(_all_p0)} rules)"
                )
            except OSError as _e:
                log_debug(f"[P0]   Debug dump failed : {_e}")

        # ── Merge Phase 0 single-rules into Phase 1 atomic pool ───────────────
        # Single-rule seeds from token-strip are merged with any user seed
        # singles; duplicates vs built-in set are dropped.
        ts_extra_singles = [r for r in ts_rules_singles if r not in builtin_set]
        rules_phase1     = rules_phase1 + ts_extra_singles
        if ts_extra_singles:
            log_debug(
                f"Phase 0: {len(ts_extra_singles)} new single-rule(s) added to Phase 1"
            )
        # ── Phase 0 chains also forwarded to Phase 2 as seed chains ──────────
        # They are appended to all_seeds so generate_informed_chains picks them up.
        if ts_rules_chains:
            all_seeds = list(all_seeds) + ts_rules_chains
            log_debug(
                f"Phase 0: {len(ts_rules_chains)} chain(s) forwarded to Phase 2"
            )
        # ─────────────────────────────────────────────────────────────────────

        # --- Phase 1: single rules + seed singles ---
        log_section("PHASE 1 — Single Rule Search")
        seed_note = f"  ({len(extra_seeds)} from seeds)" if extra_seeds else ""
        log_info(f"[P1]   {len(base_words):,} base words × "
                 f"{len(rules_phase1):,} atomic rules{seed_note}")
        t0     = time.time()
        single = self.gpu_engine.process_all_words_single_rule(
            base_words, rules_phase1, bloom_filter)
        t1     = time.time()
        all_counts.update(single)
        log_debug(f"Phase 1 elapsed: {t1-t0:.1f}s")

        # --- Phase S: Numeric Seed Extraction (controlled by --no-builtin-seeds) ---
        # Build the numeric seed families and run them as a *dedicated direct
        # extraction pass* — independent of --max-depth and the random-chain
        # time budget.  Seeds up to depth 4 are always tested (depth-1 seeds
        # are skipped here since Phase 1 already covered them).
        # The prebuilt sbd is kept locally for reference but is not forwarded
        # to Phase 2 — seeds are never used as scaffolding in chain generation.
        seed_hits = Counter()   # will be passed to Phase 3 GA
        if self.builtin_seeds:
            log_section("PHASE S — Seed Extraction (numeric + special-char families A–M)")
            sbd = self.gpu_engine.build_numeric_seed_families(max_depth=self.max_depth)

            # ── Phase 0 → Phase S injection ───────────────────────────────────
            # Add token-strip chains into the sbd pool so they receive the same
            # full GPU coverage sweep as families A–M at no extra GPU cost.
            if ts_sbd:
                n_injected = 0
                for depth, chains in ts_sbd.items():
                    before = len(sbd.setdefault(depth, set()))
                    sbd[depth].update(chains)
                    n_injected += len(sbd[depth]) - before
                if n_injected:
                    log_info(
                        f"[PS]   Phase 0 injected {bold(cyan(str(n_injected)))} "
                        f"chain(s) into Phase S sbd"
                    )
            # ─────────────────────────────────────────────────────────────────

            seed_hits = self.gpu_engine.run_seed_extraction_pass(
                base_words, sbd, bloom_filter, rules_phase1)
            all_counts.update(seed_hits)
            ts     = time.time()
            log_debug(f"Seed pass elapsed: {ts-t1:.1f}s")
        else:
            log_info(f"[PS]   {yellow('Skipped')} — built-in numeric seed families disabled "
                     f"(--no-builtin-seeds)")
            sbd = {}
            ts  = t1

        # --- Phase 2: random rule chains ---
        # Phase 2 chains are built purely from atomic rules (Phase 1 output).
        # Seeds are not used as scaffolding or building blocks here.
        #
        # If Phase 3 GA is enabled, we reserve a dedicated time budget for it
        # (20 % of TARGET_SECONDS, minimum 120 s) so that GA always gets a
        # meaningful run rather than consuming only leftover scraps.
        if self.genetic and self.max_depth >= 2:
            _min_ga_secs    = 120.0
            _ga_frac        = 0.20
            _reserved_for_ga = max(_min_ga_secs,
                                   self.params['TARGET_SECONDS'] * _ga_frac)
        else:
            _reserved_for_ga = 0.0

        if self.max_depth > 1:
            log_section("PHASE 2 — Rule Chain Search")
            remaining = max(0, self.params['TARGET_SECONDS'] - (ts-t0) - _reserved_for_ga)
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
                base_words, rules_phase1, self.max_depth, bloom_filter, single,
                seed_chains=all_seeds, prebuilt_sbd=sbd)
            all_counts.update(chains)
            log_debug(f"Phase 2 elapsed: {time.time()-ts:.1f}s")

        # --- Phase 3: Genetic Algorithm Rule Evolution (optional) ---
        if self.genetic and self.max_depth >= 2:
            log_section("PHASE 3 — Genetic Algorithm Rule Evolution")

            # Build the rule pool from Phase-1 validated rules.
            # We use only the rules that the GPU validator accepts so that
            # every individual in the population is always a legal chain.
            rule_pool = HashcatRuleValidator.validate_rules_for_gpu(rules_phase1)

            # Seed the GA with the hottest Phase-1 hits (most bloom hits first).
            hot_rules = [
                r for r, _ in sorted(
                    single.items(), key=lambda kv: -kv[1]
                )
            ]

            # Time remaining: use the dedicated GA reservation first, then
            # any unconsumed Phase-2 budget.
            t_now     = time.time()
            # Effective GA budget = reserved portion + any unused Phase-2 time
            elapsed_p12s = t_now - t0
            remaining = max(0.0, self.params['TARGET_SECONDS'] - elapsed_p12s)
            ga_budget = max(remaining, _reserved_for_ga)

            if ga_budget < 5.0:
                log_warn(
                    f"[GA]   Only {ga_budget:.1f}s available — "
                    "consider raising --target-hours for Phase 3."
                )
            else:
                log_info(
                    f"[GA]   Reserved budget for Phase 3: "
                    f"{bold(f'{ga_budget:.0f}s')}  "
                    f"(target={self.params['TARGET_SECONDS']:.0f}s, "
                    f"used={elapsed_p12s:.0f}s)"
                )

            # Pass all rules already discovered (Phase 1 + Phase S + Phase 2)
            # so the GA can compute novelty-weighted fitness.
            known_rules_set = set(all_counts.keys())

            evolver = GeneticRuleEvolver(
                gpu_engine        = self.gpu_engine,
                base_words        = base_words,
                rule_pool         = rule_pool,
                max_depth         = self.max_depth,
                pop_size          = self.genetic_pop,
                elite_frac        = self.genetic_elite,
                seed_hits         = seed_hits,
                known_rules       = known_rules_set,  # v2: novelty-aware fitness
            )

            ga_hits = evolver.evolve(
                hot_rules   = hot_rules,
                generations = self.genetic_generations,
                time_budget = ga_budget,
            )

            before = len(all_counts)
            all_counts.update(ga_hits)
            new_from_ga   = len(all_counts) - before
            n_truly_novel = sum(1 for r in ga_hits if r not in known_rules_set)
            log_info(
                f"[GA]   {bold(cyan(str(new_from_ga)))} net new rules added "
                f"by Phase 3  ({bold(green(str(len(ga_hits))))} total GA hits, "
                f"{bold(cyan(str(n_truly_novel)))} genuinely novel)"
            )

        validated = Counter({r: c for r,c in all_counts.items()
                             if HashcatRuleValidator.validate_rule_for_gpu(r)})
        log_debug(f"Post-validation: {len(validated)} rules from {len(all_counts)} raw")
        return validated

# --------------------------------------------------------------------
# Wordlist loader
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
                    help='Path to a file of seed rules. Single-rule seeds are '
                         'injected into Phase 1 as standalone extraction candidates '
                         'AND used as prioritised chain atoms in Phase 2. '
                         'Multi-rule chain seeds are tested directly against the '
                         'bloom filter in Phase 2 (not only as chain building blocks).')

    # ---- Depth overrides -----------------------------------------
    for i in range(2, 11):
        ap.add_argument(f'--depth{i}-chains', type=int, default=None,
                        dest=f'depth{i}_chains',
                        help=f'Override chain count for depth {i} (default: auto)')

    # ---- Signature-based minimization (built-in probe set only) ---
    # (Note: --sig-words and --min-word-len have been removed; built-in probe set is always used)

    # ---- Misc ----------------------------------------------------
    ap.add_argument('--allow-reject-rules', action='store_true',
                    help='Allow rules that hashcat would reject (reject-class opcodes)')
    ap.add_argument('--no-builtin-seeds', action='store_true',
                    help='Disable the built-in seed families (Phase S: '
                         'pure prepend/append, mixed, transform+digit, date patterns, '
                         'special-char families F–I: append/prepend special chars, '
                         'transform+special, number+special combos, and new families '
                         'J–M: leet substitutions, double-transform, special-before-digit, '
                         'leet+transform). '
                         'By default Phase S always runs; pass this flag to skip it '
                         'and rely solely on Phase 1 atomic rules and Phase 2 random chains.')

    # ---- Phase 0: Token-Strip ----------------------------------------
    pt = ap.add_argument_group(
        'Phase 0 — Token-Strip Rule Extraction',
        'Optional CPU-only pre-pass that decomposes target passwords into\n'
        'token categories (lowercase=stem, uppercase=case rules,\n'
        'leet chars=substitution rules, boundary digits/specials=prepend/append)\n'
        'and generates hashcat rule chains that reconstruct each password\n'
        'from a base-wordlist stem.  Activated with --token-strip.'
    )
    pt.add_argument(
        '--token-strip', action='store_true',
        help='Enable Phase 0: empirical CPU-only rule extraction by decomposing '
             'target passwords into stem + transform rules.  Discovered rules are '
             'injected into the Phase 1 atomic pool (single-rule) and Phase S sbd '
             '(multi-rule chains) before any GPU work begins.',
    )
    pt.add_argument(
        '--token-strip-min-stem', type=int, default=4, metavar='N',
        help='Minimum stem length after token decoding (default: 4).  '
             'Shorter stems produce noisy rules and are discarded.',
    )
    pt.add_argument(
        '--token-strip-max-prefix', type=int, default=4, metavar='N',
        help='Maximum number of boundary characters to strip from the start '
             'of a target word (default: 4).  These become prepend (^) rules.',
    )
    pt.add_argument(
        '--token-strip-max-suffix', type=int, default=4, metavar='N',
        help='Maximum number of boundary characters to strip from the end '
             'of a target word (default: 4).  These become append ($) rules.',
    )
    pt.add_argument(
        '--token-strip-min-leet-amb', type=int, default=3, metavar='N',
        help="Maximum number of ambiguous leet positions per word (default: 3). "
             "A position is ambiguous when its leet char maps to more than one "
             "base letter (e.g. '1' → 'i' or 'l').  Higher values allow more "
             "combinations but increase CPU time.",
    )

    # ---- Phase 3: Genetic Algorithm ----------------------------------
    ga = ap.add_argument_group(
        'Phase 3 — Genetic Algorithm',
        'Optional evolutionary search that runs after Phase 2 and guides '
        'chain generation toward high-coverage rules. '
        'Activated with --genetic.'
    )
    ga.add_argument(
        '--genetic', action='store_true',
        help='Enable Phase 3 genetic algorithm rule evolution '
             '(runs after Phase 2, uses remaining time budget).',
    )
    ga.add_argument(
        '--genetic-generations', type=int, default=50,
        metavar='N',
        help='Maximum number of GA generations to run (default: 50). '
             'Each generation performs a full GPU fitness evaluation of '
             'the entire population, so larger values extend runtime.',
    )
    ga.add_argument(
        '--genetic-pop', type=int, default=200,
        metavar='N',
        help='GA population size — number of rule chains per generation '
             '(default: 200).  Larger populations improve coverage at the '
             'cost of more GPU evaluations per generation.',
    )
    ga.add_argument(
        '--genetic-elite', type=float, default=0.15,
        metavar='F',
        help='Fraction of the top-scoring individuals carried unchanged '
             'into the next generation (default: 0.15 = 15 %%).  '
             'Higher values stabilise convergence; lower values increase '
             'diversity.',
    )

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
             f"bloom: {bold(str(args.bloom_mb or BLOOM_FILTER_MAX_MB))}MB")
    _bs_status = red('DISABLED (--no-builtin-seeds)') if args.no_builtin_seeds else green('enabled (families A–M)')
    log_info(f"  builtin seeds (Phase S) : {_bs_status}")
    # Phase 0 status
    if args.token_strip:
        _ts_inj = green('→ Phase S sbd') if not args.no_builtin_seeds else yellow('→ Phase 1 only (Phase S disabled)')
        log_info(
            f"  {bold(cyan('Phase 0 token-strip'))} : "
            f"{green('enabled')}  |  "
            f"min-stem={bold(str(args.token_strip_min_stem))}  "
            f"prefix={bold(str(args.token_strip_max_prefix))}  "
            f"suffix={bold(str(args.token_strip_max_suffix))}  "
            f"leet-amb={bold(str(args.token_strip_min_leet_amb))}  "
            f"{_ts_inj}"
        )
    if args.seed_rules:
        log_info(f"  seeds     : {bold(args.seed_rules)}  "
                 f"{dim('(singles -> Phase 1 + Phase 2 atoms | chains -> Phase 2 direct)')}")
    if args.genetic:
        # Validate --genetic-elite range
        if not 0.0 < args.genetic_elite < 1.0:
            log_error("--genetic-elite must be between 0.0 and 1.0 (exclusive).")
            sys.exit(1)
        log_info(
            f"  {bold(green('Phase 3 GA'))} : "
            f"enabled  |  "
            f"pop={bold(str(args.genetic_pop))}  "
            f"gen={bold(str(args.genetic_generations))}  "
            f"elite={bold(f'{args.genetic_elite:.0%}')}"
        )
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
    )

    extractor._output_path = args.output   # used by Phase 0 debug dump

    depth_overrides = {f'depth{i}_override': getattr(args, f'depth{i}_chains')
                       for i in range(2, 11)}

    raw_counts = extractor.extract_rules(base_words, target_words, **depth_overrides)
    log_info(f"\n[GPU]  Raw bloom-filter candidates: {bold(cyan(str(len(raw_counts))))}")
    log_debug(f"Raw counts: {len(raw_counts)} rules total")
    print()

    # ---- Signature-based functional minimization using built-in probe set ---
    final_counts = minimize_by_signature(
        raw_counts,
        BUILTIN_PROBES,
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
        if args.genetic:
            f.write(f"# Phase 3 GA     : enabled  "
                    f"pop={args.genetic_pop}  "
                    f"gen={args.genetic_generations}  "
                    f"elite={args.genetic_elite:.0%}\n")
        f.write("#\n")
        f.write(f"# GPU raw candidates      : {len(raw_counts):,}  "
                f"(bloom hits, includes false positives)\n")
        f.write(f"# Post-processing         : signature-based minimization\n")
        _min_path = 'disk-backed SQLite' if len(raw_counts) > MINIMIZE_DISK_THRESHOLD else 'in-memory'
        f.write(f"#   Path                  : {_min_path}  "
                f"(threshold {MINIMIZE_DISK_THRESHOLD:,})\n")
        f.write(f"#   Probe words           : {len(BUILTIN_PROBES)}  (built-in)\n")
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
    if args.genetic:
        log_info(f"  Phase 3 GA         : {bold(green('enabled'))}  "
                 f"pop={args.genetic_pop}  gen={args.genetic_generations}")
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
