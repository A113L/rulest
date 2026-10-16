"""
Shared low-level utilities: colors/logging, the pause/resume keyboard
controller, global constants, the hashcat rule validator, the pure-Python
(CPU) rule applicator used for verification, and FNV-1a hashing.

Everything in this module is safe to `from .common import *` — nothing here
depends on any other rulest module.
"""
import os
import sys
import time
import string
import functools
import shutil
import threading
from collections import defaultdict

from . import state

os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '0')

_WIN_MAX_POOL_WORKERS = 60  # stay safely under the 63-handle ceiling

def safe_worker_count(n_workers):
    """Clamp a requested worker/process count to a value that Windows'
    multiprocessing.Pool can actually handle."""
    n_workers = max(1, int(n_workers))
    if os.name == 'nt':
        n_workers = min(n_workers, _WIN_MAX_POOL_WORKERS)
    return n_workers

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
        """Block while paused (unless quit was requested). Call from every
        long-running stage loop so [p]/[r] work outside STAGE 1/2 as well."""
        while True:
            with self._lock:
                if not self._paused or self._quit:
                    return
            time.sleep(0.15)

    def poll(self):
        """Honour pause, then report whether quit was requested.
        Preferred one-liner for stage loops: ``if _kb.poll(): break``."""
        self.check_pause()
        return self.quit_requested

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
    if state.VERBOSE: print(f"{dim('[dbg]')} {msg}")
def log_warn(msg): print(yellow(f"[WARN] {msg}"))
def log_error(msg): print(red(f"[ERROR] {msg}"))

def log_section(title):
    w = shutil.get_terminal_size((80,24)).columns
    bar = '─' * w
    print(f"\n{cyan(bar)}")
    print(f"{cyan('│')} {bold(title.upper())}")
    print(f"{cyan(bar)}")

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
MINIMIZE_DISK_THRESHOLD = 1_000_000
MINIMIZE_DISK_BATCH_SIZE = 10_000
# Mirrors MINIMIZE_DISK_THRESHOLD's role for selection.py's coverage matrix:
# below this many candidate rules, an in-memory dict is small enough that
# disk-backed streaming would only add point-lookup overhead for no real
# memory benefit, so evaluate_full_coverage() stays purely in-memory. Above
# it, the candidate pool is large enough that holding the whole coverage
# matrix in RAM stops being safe, so it streams to a disk-backed
# CoverageStore instead (see coverage_store.py).
SELECTION_DISK_THRESHOLD = 1_000_000
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

def should_exclude_rule(rule):
    if state.ALLOW_REJECT_RULES: return False
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

def fnv1a_32(data, seed=FNV1A_SEED1):
    h = seed ^ FNV1A_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * FNV1A_PRIME) & 0xFFFFFFFF
    return h

@functools.lru_cache(maxsize=256)
def _dg(c):
    if '0' <= c <= '9': return ord(c) - 48
    if 'A' <= c <= 'Z': return ord(c) - 55
    return -1

@functools.lru_cache(maxsize=500_000)
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

# Sentinel used by GPUEngine / the multi-GPU worker to mean "the bloom
# filter is already resident on this device, skip re-upload". Must be the
# *same* object across gpu_engine.py and gpu_worker.py, so it lives here.
_BLOOM_ALREADY_ON_GPU = object()
