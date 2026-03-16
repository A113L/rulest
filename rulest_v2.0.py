import os
import sys
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
from collections import defaultdict, deque, Counter
from tqdm import tqdm
import mmap
import time
import hashlib
import signal
import math
import zlib
import random
import json
from typing import List, Dict, Set, Tuple, Optional
import gc

# Suppress compiler warnings
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

# -------------------------------------------------------------------
# Color codes (moved to top)
# -------------------------------------------------------------------
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def red(text): return f"{Colors.RED}{text}{Colors.END}"
def green(text): return f"{Colors.GREEN}{text}{Colors.END}"
def yellow(text): return f"{Colors.YELLOW}{text}{Colors.END}"
def blue(text): return f"{Colors.BLUE}{text}{Colors.END}"
def magenta(text): return f"{Colors.MAGENTA}{text}{Colors.END}"
def cyan(text): return f"{Colors.CYAN}{text}{Colors.END}"
def bold(text): return f"{Colors.BOLD}{text}{Colors.END}"

# -------------------------------------------------------------------
# Constants (will be overridden by dynamic params)
# -------------------------------------------------------------------
MAX_WORD_LEN = 256
MAX_RULE_LEN = 16
MAX_OUTPUT_LEN = 512
MAX_CHAIN_STRING_LEN = 128

# ====================================================================
# --- HASHCAT RULE VALIDATION (GPU COMPATIBILITY) ---
# ====================================================================

class HashcatRuleValidator:
    """Validates rules according to Hashcat's official GPU compatibility"""

    @staticmethod
    def is_digit(c):
        return '0' <= c <= '9'

    @staticmethod
    def is_hex_digit(c):
        return ('0' <= c <= '9') or ('A' <= c <= 'F') or ('a' <= c <= 'f')

    @staticmethod
    def conv_ctoi(c):
        if '0' <= c <= '9':
            return ord(c) - ord('0')
        elif 'A' <= c <= 'Z':
            return ord(c) - ord('A') + 10
        elif 'a' <= c <= 'z':
            return ord(c) - ord('a') + 10
        return -1

    @staticmethod
    def validate_rule_for_gpu(rule_str):
        """Validate rule for GPU compatibility (max 255 ops, correct arguments)"""
        line_len = len(rule_str)
        pos = 0
        cnt = 0

        # Hard limit from Hashcat's GPU implementation
        MAX_GPU_RULES = 255

        while pos < line_len:
            c = rule_str[pos]

            if c == ' ':
                pos += 1
                continue

            # --- Commands with no arguments ---
            if c in (':', 'l', 'u', 'c', 'C', 't', 'r', 'd', 'f', 'a', 'q', 'k', 'K', 'E'):
                pos += 1

            # --- Commands with one decimal digit ---
            elif c in ('T', 'D', 'L', 'R', '+', '-', '.', ',', "'", 'z', 'Z', 'y', 'Y'):
                pos += 1
                if pos >= line_len or not HashcatRuleValidator.is_digit(rule_str[pos]):
                    return False
                pos += 1

            # --- Commands with one digit and then a character (any) ---
            elif c in ('i', 'o'):
                pos += 1
                if pos >= line_len or not HashcatRuleValidator.is_digit(rule_str[pos]):
                    return False
                pos += 1
                if pos >= line_len:   # need the character
                    return False
                pos += 1

            # --- Commands with two decimal digits ---
            elif c in ('x', '*'):
                pos += 1
                if pos >= line_len or not HashcatRuleValidator.is_digit(rule_str[pos]):
                    return False
                pos += 1
                if pos >= line_len or not HashcatRuleValidator.is_digit(rule_str[pos]):
                    return False
                pos += 1

            # --- p : duplicate word times (one digit) ---
            elif c == 'p':
                pos += 1
                if pos >= line_len or not HashcatRuleValidator.is_digit(rule_str[pos]):
                    return False
                pos += 1

            # --- s : substitution (two chars) ---
            elif c == 's':
                pos += 1
                if pos >= line_len: return False
                pos += 1
                if pos >= line_len: return False
                pos += 1

            # --- @ : purge character (one char) ---
            elif c == '@':
                pos += 1
                if pos >= line_len: return False
                pos += 1

            # --- e : title case with separator (one char) ---
            elif c == 'e':
                pos += 1
                if pos >= line_len: return False
                pos += 1

            # --- {, }, [, ] : single character only, no arguments ---
            elif c in ('{', '}', '[', ']'):
                pos += 1
                # In Hashcat these never take a digit; do not consume next char

            # --- Memory / reject rules (not supported on GPU) ---
            elif c in ('X', '4', '6', 'M', 'v', '3',           # memory rules
                       '<', '>', '!', '/', '(', ')', '=', '%', 'Q', '?'):  # reject rules
                return False

            else:
                # Unknown command
                return False

            cnt += 1
            if cnt > MAX_GPU_RULES:
                return False

        return True

    @staticmethod
    def validate_rules_for_gpu(rules):
        """Validate and filter rules for GPU compatibility"""
        valid_rules = []
        for rule in rules:
            rule = rule.strip()
            if not rule:
                continue
            if HashcatRuleValidator.validate_rule_for_gpu(rule):
                valid_rules.append(rule)
        return valid_rules

# ====================================================================
# --- FNV-1a HASH FOR BLOOM FILTER (GPU/CPU COMPATIBLE) ---
# ====================================================================

def fnv1a_32(data, seed=0xDEADBEEF):
    """FNV-1a 32-bit hash – identical to GPU version"""
    h = seed ^ 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h

# ====================================================================
# --- GPU DEVICE SELECTION ---
# ====================================================================

def get_all_devices():
    """Return list of (platform, device) for all OpenCL devices (GPU and CPU)"""
    devices = []
    platforms = cl.get_platforms()
    for plat in platforms:
        try:
            gpu_devices = plat.get_devices(cl.device_type.GPU)
            for dev in gpu_devices:
                devices.append((plat, dev))
        except:
            pass
        try:
            cpu_devices = plat.get_devices(cl.device_type.CPU)
            for dev in cpu_devices:
                devices.append((plat, dev))
        except:
            pass
    return devices

def list_devices():
    """Print all available OpenCL devices with indices"""
    devices = get_all_devices()
    if not devices:
        print(f"{red('[ERROR]')} No OpenCL devices found.")
        sys.exit(1)
    print(f"\n{blue('[DEVICES]')} Available OpenCL devices:")
    for idx, (plat, dev) in enumerate(devices):
        name = dev.get_info(cl.device_info.NAME)
        typ = cl.device_type.to_string(dev.get_info(cl.device_info.TYPE))
        print(f"  {cyan(f'{idx}:')} {name} ({typ}) – Platform: {plat.name}")
    print()

def get_device_by_spec(spec: Optional[str]):
    """Return device matching spec (index or substring), or best GPU if spec is None"""
    if spec is None:
        return get_best_gpu_device()

    devices = get_all_devices()
    if not devices:
        raise RuntimeError("No OpenCL devices found")

    # Try as index
    if spec.isdigit():
        idx = int(spec)
        if 0 <= idx < len(devices):
            return devices[idx][1]
        else:
            raise RuntimeError(f"Device index {idx} out of range (0-{len(devices)-1})")

    # Try as substring (case-insensitive)
    spec_lower = spec.lower()
    matches = []
    for plat, dev in devices:
        name = dev.get_info(cl.device_info.NAME).lower()
        if spec_lower in name:
            matches.append(dev)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"{yellow('[WARN]')} Multiple devices match '{spec}'; using the first one.")
        return matches[0]
    else:
        raise RuntimeError(f"No device found matching '{spec}'")

def get_best_gpu_device():
    """Return the most suitable GPU device (discrete NVIDIA/AMD preferred)"""
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found")

    best_device = None
    best_score = -1

    for plat in platforms:
        try:
            devices = plat.get_devices(cl.device_type.GPU)
        except:
            continue
        for dev in devices:
            name = dev.get_info(cl.device_info.NAME).upper()
            score = 0
            if 'NVIDIA' in name or 'AMD' in name:
                score += 10
            if 'RTX' in name or 'GTX' in name:
                score += 5
            # Prefer more compute units
            cu = dev.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            score += cu
            if score > best_score:
                best_score = score
                best_device = dev

    if best_device is None:
        # fallback to first GPU device
        for plat in platforms:
            try:
                best_device = plat.get_devices(cl.device_type.GPU)[0]
                break
            except:
                continue

    if best_device is None:
        raise RuntimeError("No GPU device found")
    return best_device

def estimate_free_vram(device):
    """Conservative estimate of free VRAM in bytes"""
    try:
        total = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        # Assume driver and overhead use about 45% on average
        return int(total * 0.55)
    except:
        return 1 * 1024**3  # fallback to 1GB

# ====================================================================
# --- DYNAMIC CONSTANTS CALCULATION ---
# ====================================================================

def calculate_dynamic_parameters(base_count, target_count, device=None, target_hours=0.5):
    """Calculate dynamic parameters based on input data size, GPU capabilities and available VRAM"""

    # Get GPU work group info and VRAM if available
    if device:
        try:
            max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            global_mem = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            free_vram = estimate_free_vram(device)
            vram_gb = free_vram / (1024**3)
            is_nvidia = 'NVIDIA' in device.get_info(cl.device_info.NAME).upper()

            # Dynamic work group size
            possible_sizes = [32, 64, 128, 256, 512, 1024]
            LOCAL_WORK_SIZE = max([s for s in possible_sizes if s <= max_work_group_size])

            if is_nvidia and max_compute_units >= 38:
                LOCAL_WORK_SIZE = min(512, LOCAL_WORK_SIZE)

            OPTIMAL_GLOBAL_MULTIPLIER = max_compute_units * 16
            EST_COMBOS_PER_SEC = 120000000  # baseline (RTX 3060 Ti)
            # Scale speed estimate based on compute units (rough)
            if max_compute_units < 20:  # low-end GPU
                EST_COMBOS_PER_SEC = 40000000

            print(f"{blue('[GPU]')} {bold('Work Group Limits:')}")
            print(f"  {cyan('[*]')} Max work group size: {max_work_group_size}")
            print(f"  {cyan('[*]')} Compute units: {max_compute_units}")
            print(f"  {cyan('[*]')} Global memory: {global_mem // (1024**3)}GB")
            print(f"  {cyan('[*]')} Estimated free VRAM: {vram_gb:.1f}GB")
            print(f"  {cyan('[*]')} Using work group size: {LOCAL_WORK_SIZE}")

        except:
            # fallback if device info fails
            LOCAL_WORK_SIZE = 256
            OPTIMAL_GLOBAL_MULTIPLIER = 38 * 16
            EST_COMBOS_PER_SEC = 80000000
            max_compute_units = 38
            free_vram = 2 * 1024**3  # assume 2GB
            vram_gb = 2.0
    else:
        LOCAL_WORK_SIZE = 256
        OPTIMAL_GLOBAL_MULTIPLIER = 608
        EST_COMBOS_PER_SEC = 80000000
        max_compute_units = 38
        free_vram = 2 * 1024**3
        vram_gb = 2.0

    # VRAM scaling factor (baseline 8GB)
    vram_scale = min(1.0, vram_gb / 8.0)
    # Ensure minimum scale for very low VRAM
    vram_scale = max(0.25, vram_scale)

    target_seconds = target_hours * 3600
    max_combinations_time_limit = int(EST_COMBOS_PER_SEC * target_seconds * 0.8)

    # Bloom filter size - 64MB max, scale down slightly for low VRAM but keep at least 16MB
    BASE_BLOOM_SIZE = 1024 * 1024 * 64
    bloom_scale = max(1.0, math.log10(base_count + target_count) / 2.0)
    BLOOM_FILTER_SIZE_BYTES = int(BASE_BLOOM_SIZE * min(bloom_scale, 2.0))
    MAX_BLOOM_BYTES = 64 * 1024 * 1024
    # For low VRAM, cap bloom filter to 32MB if needed
    if vram_gb < 4:
        MAX_BLOOM_BYTES = 32 * 1024 * 1024
    BLOOM_FILTER_SIZE_BYTES = min(BLOOM_FILTER_SIZE_BYTES, MAX_BLOOM_BYTES)
    BLOOM_FILTER_SIZE = BLOOM_FILTER_SIZE_BYTES * 8

    # Batch sizes scaled by VRAM
    BASE_WORDS_PER_BATCH = 5000
    BASE_CHAINS_PER_BATCH = 2000
    BASE_WORD_SUB_BATCH = 20000
    BASE_MAX_SAFE_RESULTS = 25000

    WORDS_PER_BATCH = max(1000, int(BASE_WORDS_PER_BATCH * vram_scale))
    CHAINS_PER_BATCH = max(500, int(BASE_CHAINS_PER_BATCH * vram_scale))
    WORD_SUB_BATCH = max(5000, int(BASE_WORD_SUB_BATCH * vram_scale))
    MAX_SAFE_RESULTS_PER_BATCH = max(5000, int(BASE_MAX_SAFE_RESULTS * vram_scale))

    # Global cap removed - effectively unlimited
    MAX_CHAINS_TO_FIND = 2**31 - 1  # huge number, essentially no limit

    print(f"\n{blue('[TIME]')} {bold(f'Target completion: {target_hours} hours')}")
    print(f"{blue('[PERF]')} {bold('Estimated processing speed:')} {cyan(f'{EST_COMBOS_PER_SEC:,}')} combos/sec")
    print(f"{blue('[PERF]')} {bold('Max combinations in time:')} {cyan(f'{max_combinations_time_limit:,}')}")
    print(f"{blue('[VRAM]')} {bold('Bloom filter size:')} {cyan(f'{BLOOM_FILTER_SIZE_BYTES / 1024 / 1024:.1f}MB')}")
    print(f"{blue('[VRAM]')} {bold('Batch sizes (words/chain/word_sub):')} {cyan(f'{WORDS_PER_BATCH}/{CHAINS_PER_BATCH}/{WORD_SUB_BATCH}')}")
    print(f"{blue('[VRAM]')} {bold('Max output per batch:')} {cyan(f'{MAX_SAFE_RESULTS_PER_BATCH:,}')}")
    print(f"{blue('[LIMIT]')} {bold('Global cap:')} {cyan('unlimited')}")

    return {
        'BLOOM_FILTER_SIZE': BLOOM_FILTER_SIZE,
        'WORDS_PER_BATCH': WORDS_PER_BATCH,
        'CHAINS_PER_BATCH': CHAINS_PER_BATCH,
        'WORD_SUB_BATCH': WORD_SUB_BATCH,
        'MAX_SAFE_RESULTS_PER_BATCH': MAX_SAFE_RESULTS_PER_BATCH,
        'MAX_CHAINS_TO_FIND': MAX_CHAINS_TO_FIND,
        'LOCAL_WORK_SIZE': LOCAL_WORK_SIZE,
        'OPTIMAL_GLOBAL_MULTIPLIER': OPTIMAL_GLOBAL_MULTIPLIER,
        'EST_COMBOS_PER_SEC': EST_COMBOS_PER_SEC,
        'TARGET_SECONDS': target_seconds,
        'bloom_scale': bloom_scale,
        'vram_scale': vram_scale,
        'free_vram': free_vram
    }

# ====================================================================
# --- GPU-COMPATIBLE HASHCAT RULES GENERATION ---
# ====================================================================

class GPUCompatibleRulesGenerator:
    """Generate ONLY GPU-compatible Hashcat rules"""

    def __init__(self):
        self.validator = HashcatRuleValidator()

    def generate_gpu_compatible_rules(self):
        """Generate GPU-compatible Hashcat rules only"""
        rules = set()

        print(f"{blue('[SETUP]')} {bold('Generating GPU-compatible Hashcat rules...')}")

        # ===== CATEGORY 1: SIMPLE RULES =====
        print(f"  {cyan('[*]')} Simple rules...")
        # Removed 'a' because it causes errors in Hashcat
        simple_rules = [
            'l', 'u', 'c', 'C', 't', 'r', 'd', 'f', 'p', 'z', 'Z', 'q', 'E',
            '{', '}', '[', ']', 'k', 'K', ':'
        ]
        rules.update(simple_rules)

        # ===== CATEGORY 2: POSITION-BASED RULES (decimal digits) =====
        print(f"  {cyan('[*]')} Position-based rules (0-9 only)...")
        digits = '0123456789'

        position_cmds = ['T', 'D', 'L', 'R', '+', '-', '.', ',', "'", 'z', 'Z', 'y', 'Y']
        for cmd in position_cmds:
            for pos in digits:
                rules.add(f'{cmd}{pos}')

        # Two position rules (both digits)
        for cmd in ['x', '*']:
            for pos1 in digits:
                for pos2 in digits:
                    rules.add(f'{cmd}{pos1}{pos2}')

        # ===== CATEGORY 3: PREFIX/SUFFIX =====
        print(f"  {cyan('[*]')} Prefix/suffix rules...")
        for i in range(32, 127):
            char = chr(i)
            rules.add(f'^{char}')
            rules.add(f'${char}')
            rules.add(f'@{char}')

        # ===== CATEGORY 4: SUBSTITUTIONS (sXY) =====
        print(f"  {cyan('[*]')} Substitution rules...")
        leet_subs = [
            ('a', '@'), ('a', '4'), ('e', '3'), ('i', '1'), ('o', '0'),
            ('s', '$'), ('s', '5'), ('t', '7'), ('l', '1'), ('g', '9'),
            ('b', '8'), ('z', '2')
        ]
        for orig, sub in leet_subs:
            rules.add(f's{orig}{sub}')

        for orig in string.ascii_lowercase + string.ascii_uppercase:
            for sub in string.digits + string.punctuation:
                if orig != sub:
                    rules.add(f's{orig}{sub}')

        # ===== CATEGORY 5: INSERTION/OVERWRITE =====
        print(f"  {cyan('[*]')} Insertion/overwrite rules...")
        for pos in digits:
            for char in string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?/~':
                rules.add(f'i{pos}{char}')
                rules.add(f'o{pos}{char}')

        # ===== CATEGORY 6: EXTRACTION/SWAP =====
        print(f"  {cyan('[*]')} Extraction/swap rules...")
        for n in digits:
            for m in digits:
                if n != m:
                    rules.add(f'x{n}{m}')
                    rules.add(f'*{n}{m}')

        # ===== CATEGORY 7: DUPLICATION =====
        print(f"  {cyan('[*]')} Duplication rules...")
        for n in range(1, 10):
            rules.add(f'p{n}')
            rules.add(f'y{n}')
            rules.add(f'Y{n}')
            rules.add(f'z{n}')
            rules.add(f'Z{n}')

        # ===== CATEGORY 8: TITLE CASE WITH SEPARATOR =====
        print(f"  {cyan('[*]')} Title case rules...")
        for separator in [' ', '-', '_', '.', ',', ';', ':', '|', '/', '\\', '+', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`']:
            rules.add(f'e{separator}')

        # Convert to list and validate
        rules_list = list(rules)
        valid_rules = []

        print(f"  {cyan('[*]')} Validating rules for GPU compatibility...")
        for rule in tqdm(rules_list, desc="Validating", leave=False):
            if self.validator.validate_rule_for_gpu(rule):
                if 1 <= len(rule) <= MAX_RULE_LEN:
                    valid_rules.append(rule)

        print(f"{green('[OK]')} {bold('Generated:')} {cyan(f'{len(valid_rules):,}')} {bold('GPU-compatible Hashcat rules')}")
        return valid_rules

# ====================================================================
# --- GPU ENGINE WITH DYNAMIC WORKLOAD PROCESSING ---
# ====================================================================

class GPUEngine:
    """GPU-accelerated engine with dynamic workload processing"""

    def __init__(self, params):
        self.params = params
        self.context = None
        self.queue = None
        self.device = None
        self.program = None
        self.max_work_group_size = 512
        self.local_work_size = params.get('LOCAL_WORK_SIZE', 512)

        # Persistent buffers
        self.bloom_buf = None
        self.rule_index = {}  # Rule name -> index lookup
        self.gpu_rules = []   # List of rules in order

        # Kernel objects (to avoid repeated retrieval)
        self.kernel_single = None
        self.kernel_chain = None

    def get_free_vram(self):
        """Estimate free VRAM using OpenCL"""
        try:
            total = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            # Conservative: assume 40% is used by driver + bloom + rules
            return int(total * 0.55)
        except Exception:
            return 1 * 1024**3

    def safe_output_buffer_size(self, words_count, chains_count):
        """Calculate safe output buffer within VRAM budget (minimum 1)"""
        free_vram = self.get_free_vram()
        output_budget = max(0, free_vram - 500 * 1024**2)
        max_slots = output_budget // 128
        if max_slots < 1:
            max_slots = 1
        return min(max_slots, self.params['MAX_SAFE_RESULTS_PER_BATCH'], words_count * chains_count)

    def initialize_gpu(self, device_spec):
        """Initialize OpenCL with dynamic parameters, using specified device if given"""
        try:
            self.device = get_device_by_spec(device_spec)
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)

            global_mem = self.device.global_mem_size
            self.max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            max_compute_units = self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            free_vram = self.get_free_vram()

            print(f"{green('[GPU]')} {bold('GPU:')} {cyan(self.device.name.strip())}")
            print(f"{blue('[INFO]')} {bold('Global Memory:')} {cyan(f'{global_mem // (1024**3)}GB')}")
            print(f"{blue('[INFO]')} {bold('Max Work Group Size:')} {cyan(self.max_work_group_size)}")
            print(f"{blue('[INFO]')} {bold('Compute Units:')} {cyan(max_compute_units)}")
            print(f"{blue('[INFO]')} {bold('Free VRAM estimate:')} {cyan(f'{free_vram / 1024**3:.1f}GB')}")

            self.local_work_size = min(self.local_work_size, self.max_work_group_size)
            while self.max_work_group_size % self.local_work_size != 0 and self.local_work_size > 32:
                self.local_work_size //= 2

            print(f"{blue('[INFO]')} {bold('Using Work Group Size:')} {cyan(self.local_work_size)}")
            return True
        except Exception as e:
            print(f"{red('[ERROR]')} {bold('GPU initialization failed:')} {e}")
            return False

    def compile_kernel(self):
        """Compile the GPU kernel with injected constants"""
        try:
            print(f"{blue('[SETUP]')} {bold('Compiling GPU-compatible kernel...')}")

            # Use a template and format with dynamic values
            kernel_template = GPU_COMPATIBLE_KERNEL_TEMPLATE
            kernel_source = kernel_template.format(
                BLOOM_FILTER_SIZE=self.params['BLOOM_FILTER_SIZE'],
                MAX_SAFE_RESULTS_PER_BATCH=self.params['MAX_SAFE_RESULTS_PER_BATCH'],
                MAX_CHAIN_DEPTH=self.params['MAX_CHAIN_DEPTH'],
                MAX_CHAIN_STRING_LEN=MAX_CHAIN_STRING_LEN,
                MAX_WORD_LEN=MAX_WORD_LEN,
                MAX_RULE_LEN=MAX_RULE_LEN,
                MAX_OUTPUT_LEN=MAX_OUTPUT_LEN,
                BLOOM_HASH_FUNCTIONS=4
            )

            self.program = cl.Program(self.context, kernel_source).build()
            # Retrieve kernel objects once and store them
            self.kernel_single = self.program.find_single_rules_gpu
            self.kernel_chain = self.program.find_rule_chains_gpu
            print(f"{green('[OK]')} {bold('Kernel compiled successfully')}")
            return self.program
        except Exception as e:
            print(f"{red('[ERROR]')} {bold('Kernel compilation failed:')}")
            print(f"  {str(e)}")
            return None

    def generate_bloom_filter(self, target_words):
        """Generate Bloom filter with FNV-1a hashing"""
        print(f"{blue('[SETUP]')} {bold('Generating comprehensive Bloom filter...')}")

        bloom_size_bytes = self.params['BLOOM_FILTER_SIZE'] // 8
        bloom_filter = np.zeros(bloom_size_bytes, dtype=np.uint8)

        print(f"  {cyan('[*]')} Bloom filter size: {bloom_size_bytes / 1024 / 1024:.1f} MB")
        print(f"  {cyan('[*]')} Hashing ALL target words: {len(target_words):,}")

        for word in tqdm(target_words, desc="Building bloom filter", leave=False):
            word_bytes = word.encode('latin-1')
            h1 = fnv1a_32(word_bytes, 0xDEADBEEF)
            h2 = fnv1a_32(word_bytes, 0xCAFEBABE)

            for i in range(4):
                idx = (h1 + i * h2) % self.params['BLOOM_FILTER_SIZE']
                byte_idx = idx // 8
                bit_idx = idx % 8
                bloom_filter[byte_idx] |= (1 << bit_idx)

        bits_set = np.sum(np.unpackbits(bloom_filter))
        fill_ratio = bits_set / self.params['BLOOM_FILTER_SIZE']
        print(f"  {cyan('[*]')} Bloom filter fill ratio: {fill_ratio:.3%}")
        # approximate false positive rate (4 independent hashes)
        fpr = (fill_ratio ** 4) if fill_ratio < 0.5 else 1.0
        print(f"  {cyan('[*]')} Approx false positive rate: {fpr:.6%}")

        return bloom_filter

    def upload_bloom_filter(self, bloom_filter):
        """Upload bloom filter ONCE and reuse across all batches"""
        mf = cl.mem_flags
        if self.bloom_buf is not None:
            self.bloom_buf.release()
        self.bloom_buf = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=bloom_filter
        )
        return self.bloom_buf

    def prepare_batch_data(self, words, rules):
        """Prepare data for a single batch – efficient flat arrays"""
        # Words
        word_bytes_list = [w.encode('latin-1') for w in words]
        words_flat = np.frombuffer(b''.join(word_bytes_list), dtype=np.uint8)
        word_offsets = []
        word_lengths = []
        offset = 0
        for wb in word_bytes_list:
            word_offsets.append(offset)
            word_lengths.append(len(wb))
            offset += len(wb)

        # Rules
        rule_bytes_list = [r.encode('latin-1') for r in rules]
        rules_flat = np.frombuffer(b''.join(rule_bytes_list), dtype=np.uint8)
        rule_offsets = []
        rule_lengths = []
        offset = 0
        for rb in rule_bytes_list:
            rule_offsets.append(offset)
            rule_lengths.append(len(rb))
            offset += len(rb)

        return {
            'words_flat': words_flat,
            'word_offsets': np.array(word_offsets, dtype=np.int32),
            'word_lengths': np.array(word_lengths, dtype=np.int32),
            'rules_flat': rules_flat,
            'rule_offsets': np.array(rule_offsets, dtype=np.int32),
            'rule_lengths': np.array(rule_lengths, dtype=np.int32),
            'num_words': len(words),
            'num_rules': len(rules)
        }

    def process_all_words_single_rule(self, base_words, rules, bloom_filter):
        """Process ALL base words with single rules"""
        print(f"{blue('[GPU]')} {bold('Processing ALL words with single rules...')}")

        self.upload_bloom_filter(bloom_filter)
        if not self.compile_kernel():
            return []

        self.gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(rules)
        self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}

        print(f"{blue('[INFO]')} {bold('GPU-compatible rules:')} {len(self.gpu_rules):,}")

        found_rules_set = set()
        # Global cap removed – no early stop

        batch_size = self.params['WORDS_PER_BATCH']
        num_batches = (len(base_words) + batch_size - 1) // batch_size

        print(f"{blue('[INFO]')} {bold('Processing ALL')} {len(base_words):,} {bold('words in')} {num_batches} {bold('batches')}")
        print(f"{blue('[INFO]')} {bold('Batch size:')} {batch_size:,} words")
        print(f"{blue('[INFO]')} {bold('Global cap:')} {cyan('unlimited')}")

        with tqdm(total=num_batches, desc="Processing all words", unit="batch") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(base_words))
                batch_words = base_words[start_idx:end_idx]

                if not batch_words:
                    pbar.update(1)
                    continue

                batch_data = self.prepare_batch_data(batch_words, self.gpu_rules)
                batch_found = self.process_batch_single(batch_data)

                if batch_found:
                    found_rules_set.update(batch_found)

                pbar.set_postfix({'found': len(found_rules_set), 'progress': f"{end_idx:,}/{len(base_words):,}"})
                pbar.update(1)

                self.queue.finish()
                gc.collect()

        found_rules = list(found_rules_set)
        print(f"\n{green('[OK]')} {bold('Total unique single rules found:')} {cyan(len(found_rules))}")
        return found_rules

    def process_batch_single(self, batch_data):
        """Process a single batch on GPU"""
        mf = cl.mem_flags
        try:
            base_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['words_flat'])
            base_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_offsets'])
            base_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_lengths'])

            rules_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['rules_flat'])
            rule_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_offsets'])
            rule_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_lengths'])

            bloom_buf = self.bloom_buf

            max_output_size = self.safe_output_buffer_size(
                batch_data['num_words'],
                batch_data['num_rules']
            )

            found_rules_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                      max_output_size * MAX_CHAIN_STRING_LEN)
            found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)

            zero_count = np.array([0], dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count_buf, zero_count)

            kernel = self.kernel_single  # Use stored kernel
            total_combinations = batch_data['num_words'] * batch_data['num_rules']
            global_size = ((total_combinations + self.local_work_size - 1) //
                          self.local_work_size) * self.local_work_size

            kernel.set_args(
                base_buf, base_offsets_buf, base_lengths_buf,
                rules_buf, rule_offsets_buf, rule_lengths_buf,
                bloom_buf,
                np.int32(batch_data['num_words']),
                np.int32(batch_data['num_rules']),
                found_rules_buf,
                found_count_buf
            )

            cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (self.local_work_size,))
            self.queue.finish()

            found_count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count, found_count_buf)
            num_found = min(found_count[0], max_output_size)

            batch_found = []
            if num_found > 0:
                found_data = np.zeros(num_found * MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, found_data, found_rules_buf)
                for i in range(num_found):
                    start = i * MAX_CHAIN_STRING_LEN
                    rule_bytes = bytes(found_data[start:start + MAX_CHAIN_STRING_LEN])
                    rule_str = rule_bytes.split(b'\0')[0].decode('latin-1', errors='ignore')
                    if rule_str:
                        batch_found.append(rule_str)

            return batch_found

        except Exception as e:
            print(f"{yellow('[WARN]')} GPU processing failed: {e}")
            return []
        finally:
            self.queue.finish()
            # Release buffers (list comprehension to avoid NameError)
            for buf in (base_buf, base_offsets_buf, base_lengths_buf,
                        rules_buf, rule_offsets_buf, rule_lengths_buf,
                        found_rules_buf, found_count_buf):
                try:
                    buf.release()
                except:
                    pass

    def generate_informed_chains(self, rules, single_rules_found, max_depth):
        """
        Generate chains biased toward Phase 1 successes.
        For depths 2 and 3, first generate a seed set of common patterns
        (e.g., all combinations of prepend/append with digits) to ensure
        that simple chains like '$1 $2' are covered.
        Then fill remaining budget with random chains.
        """
        print(f"  {cyan('->')} Generating informed chains up to depth {max_depth}...")

        valid_rules = [r for r in rules if HashcatRuleValidator.validate_rule_for_gpu(r)]
        if not valid_rules:
            print(f"  {yellow('[WARN]')} No valid rules found")
            return []

        found_set = set(single_rules_found) if single_rules_found else set()
        hot_rules = [r for r in valid_rules if r in found_set]
        cold_rules = [r for r in valid_rules if r not in found_set]

        print(f"  {cyan('[*]')} Using all {len(valid_rules)} rules for chain generation")
        print(f"  {cyan('[*]')} Hot rules (from Phase 1): {len(hot_rules)}")
        print(f"  {cyan('[*]')} Cold rules: {len(cold_rules)}")

        chains = set()
        # Include single rules as chains of depth 1
        for rule in valid_rules:
            chains.add(rule)

        # Define common rule patterns for seeding (prepend/append with digits)
        common_ops = []
        for d in '0123456789':
            common_ops.append(f'^{d}')
            common_ops.append(f'${d}')
        # Also add a few other frequent rules like 'l', 'u', etc. if desired
        # But to keep seed manageable, we'll focus on ^ and $ with digits.
        # For depth 2, seed all combos of common_ops with common_ops
        # For depth 3, seed all combos of common_ops repeated 3 times.

        for depth in range(2, max_depth + 1):
            print(f"  {cyan('->')} Depth {depth} chains...")

            # Retrieve budget for this depth from params
            budget_key = f'CHAIN_GEN_LIMIT_{depth}'
            target_combinations = self.params.get(budget_key, 0)
            if target_combinations <= 0:
                print(f"  {yellow('[*]')} Budget for depth {depth} is 0, skipping.")
                continue

            max_combinations = len(valid_rules) ** depth
            target_combinations = min(target_combinations, max_combinations)

            print(f"  {cyan('[*]')} Generating up to {target_combinations:,} chains...")

            chains_added = 0
            attempts = 0
            max_attempts = target_combinations * 3  # increased for seeding overhead
            generated_patterns = set()

            # --- Seed generation for depth 2 and 3 (common patterns) ---
            seed_chains = []
            if depth == 2:
                # All combinations of two common operations (^d and $d)
                seed_chains = [f"{a} {b}" for a in common_ops for b in common_ops]
            elif depth == 3:
                # All combinations of three common operations
                seed_chains = [f"{a} {b} {c}" for a in common_ops for b in common_ops for c in common_ops]
            # For higher depths, we could also generate seeds but it might be too many; we'll rely on random.

            # Add seeds to chains (up to a limit to avoid blowing budget)
            max_seeds = min(len(seed_chains), target_combinations // 2)  # use at most half the budget for seeds
            if max_seeds > 0:
                random.shuffle(seed_chains)
                for sc in seed_chains[:max_seeds]:
                    if sc not in generated_patterns:
                        chains.add(sc)
                        generated_patterns.add(sc)
                        chains_added += 1
                print(f"  {cyan('[*]')} Added {max_seeds} seed chains for depth {depth}")

            # --- Random generation for remaining budget ---
            remaining = target_combinations - chains_added
            if remaining > 0:
                # Decide split between hot-biased and fully random
                hot_budget = int(remaining * 0.6) if hot_rules else 0
                cold_budget = remaining - hot_budget

                # Hot chains (at least one hot rule)
                if hot_rules and hot_budget > 0:
                    for _ in range(hot_budget):
                        attempts += 1
                        if attempts > max_attempts:
                            break
                        hot_pos = random.randint(0, depth - 1)
                        parts = []
                        for i in range(depth):
                            if i == hot_pos and hot_rules:
                                parts.append(random.choice(hot_rules))
                            else:
                                parts.append(random.choice(valid_rules))
                        pattern_key = ' '.join(parts)
                        if pattern_key not in generated_patterns:
                            chains.add(pattern_key)
                            generated_patterns.add(pattern_key)
                            chains_added += 1

                # Cold chains (fully random)
                for _ in range(cold_budget):
                    attempts += 1
                    if attempts > max_attempts:
                        break
                    parts = [random.choice(valid_rules) for _ in range(depth)]
                    pattern_key = ' '.join(parts)
                    if pattern_key not in generated_patterns:
                        chains.add(pattern_key)
                        generated_patterns.add(pattern_key)
                        chains_added += 1

        chains_list = list(chains)
        print(f"  {cyan('[*]')} Generated {len(chains_list):,} chains total")
        return chains_list

    def process_all_words_chain_rules(self, base_words, rules, max_depth, bloom_filter, single_rules_found):
        """Process ALL base words with rule chains"""
        print(f"{blue('[GPU]')} {bold('Processing ALL words with rule chains...')}")

        if self.bloom_buf is None:
            self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel():
                return []
        if not self.rule_index:
            self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}

        print(f"{blue('[SETUP]')} {bold('Generating rule chains...')}")
        chains = self.generate_informed_chains(rules, single_rules_found, max_depth)

        if not chains:
            return []

        print(f"{blue('[INFO]')} {bold('Total chains generated:')} {len(chains):,}")

        found_chains_set = set()
        # Global cap removed – no early stop

        chain_batch_size = self.params['CHAINS_PER_BATCH']
        word_sub_batch = self.params['WORD_SUB_BATCH']

        num_chain_batches = (len(chains) + chain_batch_size - 1) // chain_batch_size

        print(f"{blue('[INFO]')} {bold('Processing')} {len(chains):,} {bold('chains in')} {num_chain_batches} {bold('batches')}")
        print(f"{blue('[INFO]')} {bold('Chain batch size:')} {chain_batch_size:,}")
        print(f"{blue('[INFO]')} {bold('Word sub-batch size:')} {word_sub_batch:,}")
        print(f"{blue('[INFO]')} {bold('Global cap:')} {cyan('unlimited')}")

        with tqdm(total=num_chain_batches, desc="Chain batches", unit="batch") as chain_pbar:
            for chain_batch_idx in range(0, len(chains), chain_batch_size):
                chain_end = min(chain_batch_idx + chain_batch_size, len(chains))
                chain_batch = chains[chain_batch_idx:chain_end]

                for word_start in range(0, len(base_words), word_sub_batch):
                    word_end = min(word_start + word_sub_batch, len(base_words))
                    word_batch = base_words[word_start:word_end]

                    if not word_batch:
                        continue

                    batch_chains = self._process_chain_batch(word_batch, chain_batch)
                    if batch_chains:
                        found_chains_set.update(batch_chains)

                    self.queue.finish()

                chain_pbar.update(1)
                chain_pbar.set_postfix({'found': len(found_chains_set), 'progress': f"{chain_end}/{len(chains)}"})
                gc.collect()

        found_chains = list(found_chains_set)
        print(f"\n{green('[OK]')} {bold('Total unique chains found:')} {cyan(len(found_chains))}")
        return found_chains

    def _process_chain_batch(self, words, chains):
        """Process a single chain batch"""
        chain_sequences = []
        chain_depths = []

        for chain in chains:
            chain_rules = chain.split()
            depth = len(chain_rules)
            chain_depths.append(depth)
            rule_indices = []
            for rule in chain_rules:
                idx = self.rule_index.get(rule, -1)
                rule_indices.append(idx)
            while len(rule_indices) < self.params['MAX_CHAIN_DEPTH']:
                rule_indices.append(-1)
            chain_sequences.extend(rule_indices)

        batch_data = self.prepare_batch_data(words, self.gpu_rules)

        mf = cl.mem_flags
        try:
            base_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['words_flat'])
            base_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_offsets'])
            base_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_lengths'])

            rules_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['rules_flat'])
            rule_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_offsets'])
            rule_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_lengths'])

            chain_seq_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=np.array(chain_sequences, dtype=np.int32))
            chain_depth_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=np.array(chain_depths, dtype=np.int32))

            bloom_buf = self.bloom_buf

            max_output_size = self.safe_output_buffer_size(len(words), len(chains))

            found_chains_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                       max_output_size * MAX_CHAIN_STRING_LEN)
            found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)

            zero_count = np.array([0], dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count_buf, zero_count)

            kernel = self.kernel_chain  # Use stored kernel
            total_items = len(words) * len(chains)
            global_size = ((total_items + self.local_work_size - 1) //
                          self.local_work_size) * self.local_work_size

            kernel.set_args(
                base_buf, base_offsets_buf, base_lengths_buf,
                rules_buf, rule_offsets_buf, rule_lengths_buf,
                chain_seq_buf, chain_depth_buf,
                bloom_buf,
                np.int32(len(words)),
                np.int32(len(chains)),
                np.int32(self.params['MAX_CHAIN_DEPTH']),
                found_chains_buf,
                found_count_buf
            )

            cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (self.local_work_size,))
            self.queue.finish()

            found_count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count, found_count_buf)
            num_found = min(found_count[0], max_output_size)

            batch_chains = []
            if num_found > 0:
                found_data = np.zeros(num_found * MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, found_data, found_chains_buf)
                for i in range(num_found):
                    start = i * MAX_CHAIN_STRING_LEN
                    chain_bytes = bytes(found_data[start:start + MAX_CHAIN_STRING_LEN])
                    chain_str = chain_bytes.split(b'\0')[0].decode('latin-1', errors='ignore')
                    if chain_str:
                        batch_chains.append(chain_str)

            return batch_chains

        except Exception as e:
            print(f"{yellow('[WARN]')} GPU chain processing failed: {e}")
            return []
        finally:
            self.queue.finish()
            for buf in (base_buf, base_offsets_buf, base_lengths_buf,
                        rules_buf, rule_offsets_buf, rule_lengths_buf,
                        chain_seq_buf, chain_depth_buf,
                        found_chains_buf, found_count_buf):
                try:
                    buf.release()
                except:
                    pass

# ====================================================================
# --- GPU EXTRACTOR ---
# ====================================================================

class GPUExtractor:
    """GPU-optimized extractor with complete processing"""

    def __init__(self, base_count, target_count, max_depth, device_spec=None, target_hours=0.5):
        self.base_count = base_count
        self.target_count = target_count
        self.max_depth = max_depth
        self.device_spec = device_spec
        self.params = calculate_dynamic_parameters(base_count, target_count, None, target_hours)  # device passed later
        self.params['MAX_CHAIN_DEPTH'] = max_depth  # Add user-specified max depth

        print(f"{blue('[CONFIG]')} {bold('GPU-Optimized Configuration:')}")
        for k, v in self.params.items():
            if isinstance(v, (int, float)) and k not in ('bloom_scale', 'vram_scale', 'free_vram'):
                print(f"  {cyan('[*]')} {k}: {v:,}" if isinstance(v, int) else f"  {cyan('[*]')} {k}: {v}")

        self.rules_generator = GPUCompatibleRulesGenerator()
        self.gpu_engine = GPUEngine(self.params)
        self.validator = HashcatRuleValidator()

    def extract_rules(self, base_words, target_words,
                      depth2_override=None, depth3_override=None,
                      depth4_override=None, depth5_override=None, depth6_override=None):
        """Extract GPU-compatible rules using complete processing"""
        print(f"{blue('[MAIN]')} {bold('Starting GPU-optimized rule extraction...')}")

        all_chains = []

        rules = self.rules_generator.generate_gpu_compatible_rules()

        if not self.gpu_engine.initialize_gpu(self.device_spec):
            print(f"{yellow('[WARN]')} {bold('GPU not available')}")
            return []

        # Recalculate params with actual device (now that we have device)
        self.params = calculate_dynamic_parameters(self.base_count, self.target_count, self.gpu_engine.device, self.params['TARGET_SECONDS']/3600)
        self.params['MAX_CHAIN_DEPTH'] = self.max_depth
        self.gpu_engine.params = self.params

        bloom_filter = self.gpu_engine.generate_bloom_filter(target_words)

        print(f"\n{blue('=' * 60)}")
        print(f"{bold('PHASE 1: SINGLE RULE SEARCH (ALL WORDS)')}")
        print(f"{blue('=' * 60)}")

        phase1_start = time.time()
        single_chains = self.gpu_engine.process_all_words_single_rule(
            base_words, rules, bloom_filter
        )
        phase1_time = time.time() - phase1_start
        all_chains.extend(single_chains)
        print(f"{green('[OK]')} {bold('Single rules found:')} {cyan(len(single_chains))}")

        # Dynamic chain budget calculation based on actual Phase 1 time
        if self.max_depth > 1:
            print(f"\n{blue('=' * 60)}")
            print(f"{bold('PHASE 2: RULE CHAIN SEARCH (ALL WORDS)')}")
            print(f"{blue('=' * 60)}")

            remaining_time = max(0, self.params['TARGET_SECONDS'] - phase1_time)
            total_work_budget = remaining_time * self.params['EST_COMBOS_PER_SEC'] * 0.9  # 90% safety margin
            base_words_len = len(base_words)

            # Distribute work across depths 2..max_depth
            # Each depth d chain consumes d times the work of a depth-1 chain (approx)
            # We'll allocate work proportionally so that each depth gets equal total work.
            depths = list(range(2, self.max_depth + 1))
            if total_work_budget > 0 and base_words_len > 0 and depths:
                # Total work = sum_{d in depths} (num_words * budget_d * d)
                # We want equal work per depth: for each d, num_words * budget_d * d = W (constant)
                # Then sum over d of W = total_work  =>  W = total_work / len(depths)
                # So budget_d = W / (num_words * d)
                W = total_work_budget / len(depths)
                depth_budgets = {}
                for d in depths:
                    raw_budget = int(W / (base_words_len * d))
                    # Global cap removed – no upper limit
                    depth_budgets[d] = raw_budget
            else:
                depth_budgets = {d: 0 for d in depths}

            # Apply overrides if provided (overrides dictionary)
            overrides = {
                2: depth2_override,
                3: depth3_override,
                4: depth4_override,
                5: depth5_override,
                6: depth6_override
            }
            for d, val in overrides.items():
                if val is not None and d in depth_budgets:
                    depth_budgets[d] = val
                    print(f"{blue('[OVERRIDE]')} {bold(f'Depth {d} chains set to:')} {cyan(val)}")

            # Ensure budgets are non-negative
            for d in depth_budgets:
                depth_budgets[d] = max(0, depth_budgets[d])

            # Store budgets in params for chain generation
            for d, budget in depth_budgets.items():
                self.params[f'CHAIN_GEN_LIMIT_{d}'] = budget

            print(f"{blue('[DYNAMIC]')} {bold('Phase 1 time:')} {phase1_time:.2f}s, {bold('Remaining:')} {remaining_time:.2f}s")
            for d in depths:
                print(f"{blue('[DYNAMIC]')} {bold(f'Depth {d} chain limit:')} {cyan(f'{depth_budgets[d]:,}')}")

            chain_chains = self.gpu_engine.process_all_words_chain_rules(
                base_words, rules, self.max_depth, bloom_filter, single_chains
            )
            all_chains.extend(chain_chains)
            print(f"{green('[OK]')} {bold('Rule chains found:')} {cyan(len(chain_chains))}")

        print(f"\n{blue('=' * 60)}")
        print(f"{bold('FINAL CLEANUP')}")
        print(f"{blue('=' * 60)}")

        final_chains = self.validator.validate_rules_for_gpu(all_chains)
        return final_chains

# ====================================================================
# --- GPU KERNEL TEMPLATE (with placeholders) ---
# ====================================================================

GPU_COMPATIBLE_KERNEL_TEMPLATE = """
#define MAX_WORD_LEN {MAX_WORD_LEN}
#define MAX_RULE_LEN {MAX_RULE_LEN}
#define MAX_OUTPUT_LEN {MAX_OUTPUT_LEN}
#define MAX_CHAIN_STRING_LEN {MAX_CHAIN_STRING_LEN}
#define MAX_CHAINS_TO_FIND {MAX_SAFE_RESULTS_PER_BATCH}
#define MAX_CHAIN_DEPTH {MAX_CHAIN_DEPTH}
#define BLOOM_FILTER_SIZE {BLOOM_FILTER_SIZE}
#define BLOOM_HASH_FUNCTIONS {BLOOM_HASH_FUNCTIONS}

// ============================================================================
// FNV-1a HASH FUNCTION (GPU VERSION)
// ============================================================================

uint fnv1a_32_local(const unsigned char *data, int len, uint seed) {{
    uint hash = seed ^ 2166136261U;
    for (int i = 0; i < len; i++) {{
        hash ^= data[i];
        hash *= 16777619U;
    }}
    return hash;
}}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

inline int is_lower(unsigned char c) {{
    return (c >= 'a' && c <= 'z');
}}

inline int is_upper(unsigned char c) {{
    return (c >= 'A' && c <= 'Z');
}}

inline int is_digit(unsigned char c) {{
    return (c >= '0' && c <= '9');
}}

inline unsigned char to_lower(unsigned char c) {{
    if (is_upper(c)) return c + 32;
    return c;
}}

inline unsigned char to_upper(unsigned char c) {{
    if (is_lower(c)) return c - 32;
    return c;
}}

inline unsigned char toggle_case(unsigned char c) {{
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}}

// ============================================================================
// BLOOM FILTER CHECK
// ============================================================================

int bloom_check(__global const uchar *bloom_filter, const unsigned char *word, int len) {{
    uint h1 = fnv1a_32_local(word, len, 0xDEADBEEF);
    uint h2 = fnv1a_32_local(word, len, 0xCAFEBABE);

    for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {{
        uint hash_val = (h1 + i * h2) % BLOOM_FILTER_SIZE;
        uint byte_idx = hash_val / 8;
        uint bit_idx = hash_val % 8;

        if (!(bloom_filter[byte_idx] & (1 << bit_idx))) {{
            return 0;
        }}
    }}
    return 1;
}}

// ============================================================================
// GPU-COMPATIBLE RULE APPLICATION (SAFE FOR OPENCL)
// ============================================================================

int apply_gpu_rule(
    const unsigned char *rule_str, int rule_len,
    const unsigned char *input_word, int input_len,
    unsigned char *output_word, int *output_len
) {{
    // Initialize with input
    *output_len = input_len;
    for (int i = 0; i < input_len; i++) {{
        output_word[i] = input_word[i];
    }}
    output_word[input_len] = '\\0';

    if (rule_len == 0 || input_len == 0) return 1;

    unsigned char cmd = rule_str[0];
    int changed = 0;

    // ------------------------------------------------------------------------
    // SINGLE CHARACTER RULES
    // ------------------------------------------------------------------------
    if (rule_len == 1) {{
        switch (cmd) {{
            case 'l': // Lowercase all
                for (int i = 0; i < *output_len; i++) {{
                    output_word[i] = to_lower(output_word[i]);
                }}
                changed = 1;
                break;
            case 'u': // Uppercase all
                for (int i = 0; i < *output_len; i++) {{
                    output_word[i] = to_upper(output_word[i]);
                }}
                changed = 1;
                break;
            case 'c': // Capitalize first, lowercase rest
                if (*output_len > 0) {{
                    output_word[0] = to_upper(output_word[0]);
                    for (int i = 1; i < *output_len; i++) {{
                        output_word[i] = to_lower(output_word[i]);
                    }}
                }}
                changed = 1;
                break;
            case 'C': // Lowercase first, uppercase rest
                if (*output_len > 0) {{
                    output_word[0] = to_lower(output_word[0]);
                    for (int i = 1; i < *output_len; i++) {{
                        output_word[i] = to_upper(output_word[i]);
                    }}
                }}
                changed = 1;
                break;
            case 't': // Toggle case
                for (int i = 0; i < *output_len; i++) {{
                    output_word[i] = toggle_case(output_word[i]);
                }}
                changed = 1;
                break;
            case 'r': // Reverse
                for (int i = 0; i < *output_len / 2; i++) {{
                    unsigned char temp = output_word[i];
                    output_word[i] = output_word[*output_len - 1 - i];
                    output_word[*output_len - 1 - i] = temp;
                }}
                changed = 1;
                break;
            case 'd': // Duplicate
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < *output_len; i++) {{
                        output_word[*output_len + i] = output_word[i];
                    }}
                    *output_len *= 2;
                }}
                changed = 1;
                break;
            case 'f': // Reflect
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {{
                    for (int i = 0; i < *output_len; i++) {{
                        output_word[*output_len + i] = output_word[*output_len - 1 - i];
                    }}
                    *output_len *= 2;
                }}
                changed = 1;
                break;
            case '{{': // Rotate left
                if (*output_len > 1) {{
                    unsigned char first = output_word[0];
                    for (int i = 0; i < *output_len - 1; i++) {{
                        output_word[i] = output_word[i + 1];
                    }}
                    output_word[*output_len - 1] = first;
                }}
                changed = 1;
                break;
            case '}}': // Rotate right
                if (*output_len > 1) {{
                    unsigned char last = output_word[*output_len - 1];
                    for (int i = *output_len - 1; i > 0; i--) {{
                        output_word[i] = output_word[i - 1];
                    }}
                    output_word[0] = last;
                }}
                changed = 1;
                break;
            case '[': // Delete first char
                if (*output_len > 0) {{
                    for (int i = 0; i < *output_len - 1; i++) {{
                        output_word[i] = output_word[i + 1];
                    }}
                    (*output_len)--;
                }}
                changed = 1;
                break;
            case ']': // Delete last char
                if (*output_len > 0) {{
                    (*output_len)--;
                }}
                changed = 1;
                break;
            case 'k': // Swap first two
                if (*output_len >= 2) {{
                    unsigned char temp = output_word[0];
                    output_word[0] = output_word[1];
                    output_word[1] = temp;
                }}
                changed = 1;
                break;
            case 'K': // Swap last two
                if (*output_len >= 2) {{
                    unsigned char temp = output_word[*output_len - 2];
                    output_word[*output_len - 2] = output_word[*output_len - 1];
                    output_word[*output_len - 1] = temp;
                }}
                changed = 1;
                break;
            case ':': // No operation
                changed = 0;
                break;
            case 'q': // Duplicate all chars
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {{
                    unsigned char temp[MAX_OUTPUT_LEN];
                    for (int i = 0; i < *output_len; i++) temp[i] = output_word[i];
                    int idx = 0;
                    for (int i = 0; i < *output_len; i++) {{
                        output_word[idx++] = temp[i];
                        output_word[idx++] = temp[i];
                    }}
                    *output_len *= 2;
                }}
                changed = 1;
                break;
            case 'E': // Title case
                if (*output_len > 0) {{
                    int capitalize = 1;
                    for (int i = 0; i < *output_len; i++) {{
                        if (capitalize && is_lower(output_word[i])) {{
                            output_word[i] = to_upper(output_word[i]);
                            capitalize = 0;
                        }}
                        if (output_word[i] == ' ' || output_word[i] == '-' || output_word[i] == '_') {{
                            capitalize = 1;
                        }}
                    }}
                }}
                changed = 1;
                break;
            // 'a' removed because it causes errors in Hashcat
        }}
    }}
    // ------------------------------------------------------------------------
    // TWO CHARACTER RULES (cmd + one parameter)
    // ------------------------------------------------------------------------
    else if (rule_len == 2) {{
        unsigned char param = rule_str[1];

        if (cmd == '^') {{ // Prepend
            if (*output_len + 1 <= MAX_OUTPUT_LEN) {{
                for (int i = *output_len; i > 0; i--) {{
                    output_word[i] = output_word[i - 1];
                }}
                output_word[0] = param;
                (*output_len)++;
                changed = 1;
            }}
        }}
        else if (cmd == '$') {{ // Append
            if (*output_len + 1 <= MAX_OUTPUT_LEN) {{
                output_word[*output_len] = param;
                (*output_len)++;
                changed = 1;
            }}
        }}
        else if (cmd == '@') {{ // Delete all instances of char
            int new_len = 0;
            for (int i = 0; i < *output_len; i++) {{
                if (output_word[i] != param) {{
                    output_word[new_len++] = output_word[i];
                }} else {{
                    changed = 1;
                }}
            }}
            *output_len = new_len;
        }}
        else if (cmd == 'p') {{ // Duplicate word times (already handled in single? No, pN is two-char)
            // Actually pN is two characters (p and a digit). But we already handle pN in three-char? No, it's two-char.
            // We'll handle it here as a two-char rule.
            int n = param - '0';
            if (n > 0 && *output_len * (n + 1) <= MAX_OUTPUT_LEN) {{
                int original_len = *output_len;
                for (int rep = 0; rep < n; rep++) {{
                    for (int i = 0; i < original_len; i++) {{
                        output_word[*output_len + i] = output_word[i];
                    }}
                    *output_len += original_len;
                }}
                changed = 1;
            }}
        }}
        else if (cmd == 'T' && is_digit(param)) {{ // Toggle at position
            int pos = param - '0';
            if (pos < *output_len) {{
                output_word[pos] = toggle_case(output_word[pos]);
                changed = 1;
            }}
        }}
        else if (cmd == 'D' && is_digit(param)) {{ // Delete at position
            int pos = param - '0';
            if (pos < *output_len) {{
                for (int i = pos; i < *output_len - 1; i++) {{
                    output_word[i] = output_word[i + 1];
                }}
                (*output_len)--;
                changed = 1;
            }}
        }}
        else if (cmd == 'L' && is_digit(param)) {{ // Delete left (keep from pos to end)
            int pos = param - '0';
            if (pos < *output_len) {{
                int new_len = 0;
                for (int i = pos; i < *output_len; i++) {{
                    output_word[new_len++] = output_word[i];
                }}
                *output_len = new_len;
                changed = 1;
            }}
        }}
        else if (cmd == 'R' && is_digit(param)) {{ // Delete right (keep up to pos)
            int pos = param - '0';
            if (pos < *output_len) {{
                *output_len = pos + 1;
                changed = 1;
            }}
        }}
        else if (cmd == '+' && is_digit(param)) {{ // Increment at position
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] < 255) {{
                output_word[pos]++;
                changed = 1;
            }}
        }}
        else if (cmd == '-' && is_digit(param)) {{ // Decrement at position
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] > 0) {{
                output_word[pos]--;
                changed = 1;
            }}
        }}
        else if (cmd == '.' && is_digit(param)) {{ // Replace with '.' at position
            int pos = param - '0';
            if (pos < *output_len) {{
                output_word[pos] = '.';
                changed = 1;
            }}
        }}
        else if (cmd == ',' && is_digit(param)) {{ // Replace with ',' at position
            int pos = param - '0';
            if (pos < *output_len) {{
                output_word[pos] = ',';
                changed = 1;
            }}
        }}
        else if (cmd == '\\'' && is_digit(param)) {{ // Truncate at position (keep up to pos)
            int pos = param - '0';
            if (pos < *output_len) {{
                *output_len = pos + 1;
                changed = 1;
            }}
        }}
        else if (cmd == 'z' && is_digit(param)) {{ // Duplicate first character N times
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {{
                unsigned char first = output_word[0];
                for (int i = *output_len + n - 1; i >= n; i--) {{
                    output_word[i] = output_word[i - n];
                }}
                for (int i = 0; i < n; i++) {{
                    output_word[i] = first;
                }}
                *output_len += n;
                changed = 1;
            }}
        }}
        else if (cmd == 'Z' && is_digit(param)) {{ // Duplicate last character N times
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {{
                unsigned char last = output_word[*output_len - 1];
                for (int i = 0; i < n; i++) {{
                    output_word[*output_len + i] = last;
                }}
                *output_len += n;
                changed = 1;
            }}
        }}
        else if (cmd == 'y' && is_digit(param)) {{ // Duplicate first block of length N
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {{
                for (int i = 0; i < n; i++) {{
                    output_word[*output_len + i] = output_word[i];
                }}
                *output_len += n;
                changed = 1;
            }}
        }}
        else if (cmd == 'Y' && is_digit(param)) {{ // Duplicate last block of length N
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {{
                for (int i = 0; i < n; i++) {{
                    output_word[*output_len + i] = output_word[*output_len - n + i];
                }}
                *output_len += n;
                changed = 1;
            }}
        }}
    }}
    // ------------------------------------------------------------------------
    // THREE CHARACTER RULES
    // ------------------------------------------------------------------------
    else if (rule_len == 3) {{
        unsigned char param1 = rule_str[1];
        unsigned char param2 = rule_str[2];

        if (cmd == 's') {{ // Substitute
            for (int i = 0; i < *output_len; i++) {{
                if (output_word[i] == param1) {{
                    output_word[i] = param2;
                    changed = 1;
                }}
            }}
        }}
        else if (cmd == 'i' && is_digit(param1)) {{ // Insert at position
            int pos = param1 - '0';
            if (pos <= *output_len && *output_len + 1 <= MAX_OUTPUT_LEN) {{
                for (int i = *output_len; i > pos; i--) {{
                    output_word[i] = output_word[i - 1];
                }}
                output_word[pos] = param2;
                (*output_len)++;
                changed = 1;
            }}
        }}
        else if (cmd == 'o' && is_digit(param1)) {{ // Overwrite at position
            int pos = param1 - '0';
            if (pos < *output_len) {{
                output_word[pos] = param2;
                changed = 1;
            }}
        }}
        else if (cmd == 'e') {{ // Title case with separator
            unsigned char separator = param1;
            if (*output_len > 0) {{
                int capitalize = 1;
                for (int i = 0; i < *output_len; i++) {{
                    if (capitalize && is_lower(output_word[i])) {{
                        output_word[i] = to_upper(output_word[i]);
                        capitalize = 0;
                    }}
                    if (output_word[i] == separator) {{
                        capitalize = 1;
                    }}
                }}
                changed = 1;
            }}
        }}
        else if (cmd == 'x' && is_digit(param1) && is_digit(param2)) {{ // Extract
            int n = param1 - '0';
            int m = param2 - '0';
            if (n > m) {{ int temp = n; n = m; m = temp; }}
            if (n < *output_len) {{
                int new_len = 0;
                for (int i = n; i <= m && i < *output_len; i++) {{
                    output_word[new_len++] = output_word[i];
                }}
                *output_len = new_len;
                changed = 1;
            }}
        }}
        else if (cmd == '*' && is_digit(param1) && is_digit(param2)) {{ // Swap positions
            int n = param1 - '0';
            int m = param2 - '0';
            if (n < *output_len && m < *output_len && n != m) {{
                unsigned char temp = output_word[n];
                output_word[n] = output_word[m];
                output_word[m] = temp;
                changed = 1;
            }}
        }}
        // K with two digits is NOT a valid Hashcat rule; we omit it.
    }}

    output_word[*output_len] = '\\0';
    return changed ? 1 : 0;
}}

// ============================================================================
// KERNEL: FIND SINGLE RULES
// ============================================================================

__kernel void find_single_rules_gpu(
    __global const unsigned char *base_words,
    __global const int *base_offsets,
    __global const int *base_lengths,
    __global const unsigned char *rules,
    __global const int *rule_offsets,
    __global const int *rule_lengths,
    __global const uchar *bloom_filter,
    const int num_base_words,
    const int num_rules,
    __global char *found_rules,
    __global volatile int *found_count
) {{
    int gid = get_global_id(0);
    int total_items = num_base_words * num_rules;

    if (gid >= total_items) return;

    int word_idx = gid / num_rules;
    int rule_idx = gid % num_rules;

    // Get base word
    __global const unsigned char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];

    unsigned char input_word[MAX_WORD_LEN];
    for (int i = 0; i < word_len; i++) {{
        input_word[i] = word_ptr[i];
    }}
    input_word[word_len] = '\\0';

    // Get rule
    __global const unsigned char *rule_ptr = rules + rule_offsets[rule_idx];
    int rule_len = rule_lengths[rule_idx];

    unsigned char rule_str[MAX_RULE_LEN];
    for (int i = 0; i < rule_len; i++) {{
        rule_str[i] = rule_ptr[i];
    }}
    rule_str[rule_len] = '\\0';

    // Apply rule
    unsigned char output_word[MAX_OUTPUT_LEN];
    int output_len;
    int result = apply_gpu_rule(rule_str, rule_len, input_word, word_len, output_word, &output_len);

    if (result > 0 && output_len > 0) {{
        if (bloom_check(bloom_filter, output_word, output_len)) {{
            int idx = atomic_inc(found_count);
            if (idx < MAX_CHAINS_TO_FIND) {{
                __global char *output_ptr = found_rules + idx * MAX_CHAIN_STRING_LEN;
                for (int i = 0; i < rule_len && i < MAX_CHAIN_STRING_LEN - 1; i++) {{
                    output_ptr[i] = rule_str[i];
                }}
                output_ptr[rule_len] = '\\0';
            }}
        }}
    }}
}}

// ============================================================================
// KERNEL: FIND RULE CHAINS
// ============================================================================

__kernel void find_rule_chains_gpu(
    __global const unsigned char *base_words,
    __global const int *base_offsets,
    __global const int *base_lengths,
    __global const unsigned char *rules,
    __global const int *rule_offsets,
    __global const int *rule_lengths,
    __global const int *chain_sequences,
    __global const int *chain_depths,
    __global const uchar *bloom_filter,
    const int num_base_words,
    const int num_chains,
    const int max_chain_depth,
    __global char *found_chains,
    __global volatile int *found_count
) {{
    int gid = get_global_id(0);
    int total_items = num_base_words * num_chains;

    if (gid >= total_items) return;

    int word_idx = gid / num_chains;
    int chain_idx = gid % num_chains;

    // Get base word
    __global const unsigned char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];

    unsigned char current_word[MAX_OUTPUT_LEN];
    for (int i = 0; i < word_len; i++) {{
        current_word[i] = word_ptr[i];
    }}
    current_word[word_len] = '\\0';
    int current_len = word_len;

    int depth = chain_depths[chain_idx];
    if (depth < 1 || depth > max_chain_depth) return;

    unsigned char temp_word[MAX_OUTPUT_LEN];
    char chain_buffer[MAX_CHAIN_STRING_LEN];
    int chain_pos = 0;

    for (int d = 0; d < depth; d++) {{
        int rule_idx = chain_sequences[chain_idx * max_chain_depth + d];
        if (rule_idx < 0) break;

        __global const unsigned char *rule_ptr = rules + rule_offsets[rule_idx];
        int rule_len = rule_lengths[rule_idx];

        unsigned char rule_str[MAX_RULE_LEN];
        for (int i = 0; i < rule_len; i++) {{
            rule_str[i] = rule_ptr[i];
        }}
        rule_str[rule_len] = '\\0';

        for (int i = 0; i < rule_len && chain_pos < MAX_CHAIN_STRING_LEN - 2; i++) {{
            chain_buffer[chain_pos++] = rule_str[i];
        }}
        if (d < depth - 1 && chain_pos < MAX_CHAIN_STRING_LEN - 1) {{
            chain_buffer[chain_pos++] = ' ';
        }}

        int new_len;
        int result = apply_gpu_rule(rule_str, rule_len, current_word, current_len, temp_word, &new_len);

        if (result <= 0 || new_len == 0) {{
            return;
        }}

        for (int i = 0; i < new_len; i++) {{
            current_word[i] = temp_word[i];
        }}
        current_word[new_len] = '\\0';
        current_len = new_len;
    }}

    chain_buffer[chain_pos] = '\\0';

    if (bloom_check(bloom_filter, current_word, current_len)) {{
        int idx = atomic_inc(found_count);
        if (idx < MAX_CHAINS_TO_FIND) {{
            __global char *output_ptr = found_chains + idx * MAX_CHAIN_STRING_LEN;
            for (int i = 0; i < chain_pos && i < MAX_CHAIN_STRING_LEN - 1; i++) {{
                output_ptr[i] = chain_buffer[i];
            }}
            output_ptr[chain_pos] = '\\0';
        }}
    }}
}}
"""

# ====================================================================
# --- UTILITY FUNCTIONS ---
# ====================================================================

def load_wordlist_fast(filename):
    """Fast wordlist loading with statistics"""
    words = set()
    print(f"{blue('[LOAD]')} {bold('Loading:')} {filename}")

    try:
        with open(filename, 'r', encoding='latin-1', errors='ignore') as f:
            for line in tqdm(f, desc="Loading words"):
                word = line.strip()
                if word and len(word) <= MAX_WORD_LEN:
                    words.add(word)
    except FileNotFoundError:
        print(f"{red('[ERROR]')} {bold('FATAL ERROR:')} Wordlist not found: {filename}")
        sys.exit(1)

    words_list = list(words)

    if words_list:
        avg_len = sum(len(w) for w in words_list) / len(words_list)
        max_len = max(len(w) for w in words_list)
    else:
        avg_len = max_len = 0

    print(f"{green('[OK]')} {bold('Loaded:')} {cyan(f'{len(words_list):,}')} {bold('words')}")
    print(f"{blue('[INFO]')} {bold('Average length:')} {cyan(f'{avg_len:.1f}')}")
    print(f"{blue('[INFO]')} {bold('Max length:')} {cyan(f'{max_len}')}")

    return words_list

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"{bold('GPU-COMPATIBLE Hashcat Rules Engine with Dynamic Workload Processing (no global cap)')}",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('base_wordlist', nargs='?', help='Base wordlist path')
    parser.add_argument('target_wordlist', nargs='?', help='Target wordlist path')
    parser.add_argument('-d', '--max-depth', type=int, default=3,
                       choices=[1,2,3,4,5,6],
                       help='Max chain depth (1-6, default: 3)')
    parser.add_argument('-o', '--output', type=str, default='found_chains.txt',
                       help='Output file (default: found_chains.txt)')
    parser.add_argument('--max-chains', type=int, default=None,
                       help='Maximum chains to generate (default: unlimited)')
    parser.add_argument('--target-hours', type=float, default=0.5,
                       help='Target completion time in hours (default: 0.5)')
    # Device selection
    parser.add_argument('--list-devices', action='store_true',
                       help='List available OpenCL devices and exit')
    parser.add_argument('--device', type=str, default=None,
                       help='Device index or substring (e.g., "0" or "NVIDIA")')
    # Depth-specific overrides (up to depth 6)
    parser.add_argument('--depth2-chains', type=int, default=None,
                       help='Override dynamic limit for depth 2 chains')
    parser.add_argument('--depth3-chains', type=int, default=None,
                       help='Override dynamic limit for depth 3 chains')
    parser.add_argument('--depth4-chains', type=int, default=None,
                       help='Override dynamic limit for depth 4 chains')
    parser.add_argument('--depth5-chains', type=int, default=None,
                       help='Override dynamic limit for depth 5 chains')
    parser.add_argument('--depth6-chains', type=int, default=None,
                       help='Override dynamic limit for depth 6 chains')

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    # If not listing devices, we need both wordlists
    if args.base_wordlist is None or args.target_wordlist is None:
        parser.print_help()
        print(f"\n{red('[ERROR]')} Both base_wordlist and target_wordlist are required when not using --list-devices.")
        sys.exit(1)

    print(f"\n{bold(green('=' * 80))}")
    print(f"{bold('GPU-COMPATIBLE HASHCAT RULES ENGINE (DYNAMIC WORKLOAD, UNLIMITED GLOBAL CAP)')}")
    print(f"{bold(green('=' * 80))}{Colors.END}\n")

    print(f"{blue('[INIT]')} {bold('Loading data...')}")

    base_words = load_wordlist_fast(args.base_wordlist)
    target_words = load_wordlist_fast(args.target_wordlist)

    print(f"\n{blue('[ANALYSIS]')} {bold('Dataset Analysis:')}")
    print(f"  {cyan('[*]')} Base words: {len(base_words):,}")
    print(f"  {cyan('[*]')} Target words: {len(target_words):,}")
    print(f"  {cyan('[*]')} Processing ALL words")
    print(f"  {cyan('[*]')} Target completion: {args.target_hours} hours")

    start_time = time.time()

    # Device is handled inside GPUExtractor and GPUEngine
    extractor = GPUExtractor(len(base_words), len(target_words), args.max_depth, args.device, args.target_hours)

    if args.max_chains:
        extractor.params['MAX_CHAINS_TO_FIND'] = args.max_chains
        print(f"{blue('[OVERRIDE]')} {bold('Max chains set to:')} {cyan(args.max_chains)}")
    else:
        print(f"{blue('[OVERRIDE]')} {bold('Max chains:')} {cyan('unlimited')}")

    print(f"\n{blue('=' * 60)}")
    print(f"{bold('STARTING GPU-COMPATIBLE RULE EXTRACTION')}")
    print(f"{blue('=' * 60)}")

    chains = extractor.extract_rules(base_words, target_words,
                                     depth2_override=args.depth2_chains,
                                     depth3_override=args.depth3_chains,
                                     depth4_override=args.depth4_chains,
                                     depth5_override=args.depth5_chains,
                                     depth6_override=args.depth6_chains)

    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600

    print(f"\n{blue('[SAVE]')} {bold('Saving results...')}")

    final_chains = HashcatRuleValidator.validate_rules_for_gpu(chains)

    with open(args.output, 'w', encoding='latin-1') as f:
        for chain in final_chains:
            f.write(f"{chain}\n")

    print(f"\n{bold(green('=' * 80))}")
    print(f"{bold('FINAL RESULTS')}")
    print(f"{bold(green('=' * 80))}")
    print(f"{blue('[INFO]')} {bold('Base words:')} {cyan(f'{len(base_words):,}')}")
    print(f"{blue('[INFO]')} {bold('Target words:')} {cyan(f'{len(target_words):,}')}")
    print(f"{blue('[INFO]')} {bold('Max depth:')} {cyan(f'{args.max_depth}')}")
    print(f"{blue('[INFO]')} {bold('Total time:')} {cyan(f'{elapsed_hours:.2f} hours ({end_time - start_time:.2f}s)')}")
    print(f"{blue('[INFO]')} {bold('Target time:')} {cyan(f'{args.target_hours} hours')}")
    print(f"{green('[RESULT]')} {bold('GPU-compatible chains found:')} {cyan(f'{len(final_chains):,}')}")

    if final_chains:
        print(f"{blue('[SAMPLE]')} {bold('Sample chains (max 20):')}")
        for i, chain in enumerate(final_chains[:20]):
            depth = len(chain.split())
            print(f"  {cyan(f'{i+1:2d}.')} [{depth}] {chain}")

    print(f"{blue('[OUTPUT]')} {bold('Output saved to:')} {bold(args.output)}")
    print(f"{blue('[NOTE]')} {bold('All chains are GPU-compatible and syntactically valid for Hashcat')}")
    print(f"{bold(green('=' * 80))}{Colors.END}")
