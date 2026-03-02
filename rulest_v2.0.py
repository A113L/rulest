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

# Suppress compiler warnings
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

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
        """Validate rule for GPU compatibility"""
        line_len = len(rule_str)
        pos = 0
        cnt = 0
        
        # Max rules per line for GPU - significantly increased
        MAX_GPU_RULES = 10000  # Increased from 255
        
        while pos < line_len:
            c = rule_str[pos]
            
            if c == ' ':
                pos += 1
                continue
            
            # Hashcat rule commands with GPU compatibility check
            if c == ':':  # RULE_OP_MANGLE_NOOP
                pos += 1
            elif c == 'l':  # RULE_OP_MANGLE_LREST
                pos += 1
            elif c == 'u':  # RULE_OP_MANGLE_UREST
                pos += 1
            elif c == 'c':  # RULE_OP_MANGLE_LREST_UFIRST
                pos += 1
            elif c == 'C':  # RULE_OP_MANGLE_UREST_LFIRST
                pos += 1
            elif c == 't':  # RULE_OP_MANGLE_TREST
                pos += 1
            elif c == 'T':  # RULE_OP_MANGLE_TOGGLE_AT
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'r':  # RULE_OP_MANGLE_REVERSE
                pos += 1
            elif c == 'd':  # RULE_OP_MANGLE_DUPEWORD
                pos += 1
            elif c == 'p':  # RULE_OP_MANGLE_DUPEWORD_TIMES
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'f':  # RULE_OP_MANGLE_REFLECT
                pos += 1
            elif c == '{':  # RULE_OP_MANGLE_ROTATE_LEFT
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == '}':  # RULE_OP_MANGLE_ROTATE_RIGHT
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == '$':  # RULE_OP_MANGLE_APPEND
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '^':  # RULE_OP_MANGLE_PREPEND
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '[':  # RULE_OP_MANGLE_DELETE_FIRST
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == ']':  # RULE_OP_MANGLE_DELETE_LAST
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == 'D':  # RULE_OP_MANGLE_DELETE_AT
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'x':  # RULE_OP_MANGLE_EXTRACT
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'i':  # RULE_OP_MANGLE_INSERT
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == 'o':  # RULE_OP_MANGLE_OVERSTRIKE
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '\'':  # RULE_OP_MANGLE_TRUNCATE_AT
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 's':  # RULE_OP_MANGLE_REPLACE
                pos += 1
                if pos >= line_len: return False
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '@':  # RULE_OP_MANGLE_PURGECHAR
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == 'a':  # RULE_OP_MANGLE_TOGGLECASE_REC
                pos += 1
            elif c == 'z':  # RULE_OP_MANGLE_DUPECHAR_FIRST
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'Z':  # RULE_OP_MANGLE_DUPECHAR_LAST
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'q':  # RULE_OP_MANGLE_DUPECHAR_ALL
                pos += 1
            elif c == 'y':  # RULE_OP_MANGLE_DUPEBLOCK_FIRST
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'Y':  # RULE_OP_MANGLE_DUPEBLOCK_LAST
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'k':  # RULE_OP_MANGLE_SWITCH_FIRST
                pos += 1
            elif c == 'K':  # RULE_OP_MANGLE_SWITCH_LAST
                pos += 1
            elif c == '*':  # RULE_OP_MANGLE_SWITCH_AT
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'L':  # RULE_OP_MANGLE_CHR_SHIFTL
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'R':  # RULE_OP_MANGLE_CHR_SHIFTR
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == '+':  # RULE_OP_MANGLE_CHR_INCR
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == '-':  # RULE_OP_MANGLE_CHR_DECR
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == '.':  # RULE_OP_MANGLE_REPLACE_NP1
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == ',':  # RULE_OP_MANGLE_REPLACE_NM1
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'E':  # RULE_OP_MANGLE_TITLE
                pos += 1
            elif c == 'e':  # RULE_OP_MANGLE_TITLE_SEP
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == 'X':  # RULE_OP_MANGLE_EXTRACT_MEMORY
                # Memory rules NOT supported on GPU
                return False
            elif c == '4':  # RULE_OP_MANGLE_APPEND_MEMORY
                # Memory rules NOT supported on GPU
                return False
            elif c == '6':  # RULE_OP_MANGLE_PREPEND_MEMORY
                # Memory rules NOT supported on GPU
                return False
            elif c == 'M':  # RULE_OP_MEMORIZE_WORD
                # Memory rules NOT supported on GPU
                return False
            elif c == '<':  # RULE_OP_REJECT_LESS
                # Reject rules NOT supported on GPU
                return False
            elif c == '>':  # RULE_OP_REJECT_GREATER
                # Reject rules NOT supported on GPU
                return False
            elif c == '!':  # RULE_OP_REJECT_CONTAIN
                # Reject rules NOT supported on GPU
                return False
            elif c == '/':  # RULE_OP_REJECT_NOT_CONTAIN
                # Reject rules NOT supported on GPU
                return False
            elif c == '(':  # RULE_OP_REJECT_EQUAL_FIRST
                # Reject rules NOT supported on GPU
                return False
            elif c == ')':  # RULE_OP_REJECT_EQUAL_LAST
                # Reject rules NOT supported on GPU
                return False
            elif c == '=':  # RULE_OP_REJECT_EQUAL_AT
                # Reject rules NOT supported on GPU
                return False
            elif c == '%':  # RULE_OP_REJECT_CONTAINS
                # Reject rules NOT supported on GPU
                return False
            elif c == 'Q':  # RULE_OP_REJECT_MEMORY
                # Reject rules NOT supported on GPU
                return False
            elif c == 'v':  # RULE_OP_MANGLE_INSERT_MEMORY
                # Memory rules NOT supported on GPU
                return False
            elif c == '3':  # RULE_OP_MANGLE_TOGGLE_CASE_MEMORY
                # Memory rules NOT supported on GPU
                return False
            elif c == '?':  # RULE_OP_REJECT_EQUAL_MEMORY
                # Reject rules NOT supported on GPU
                return False
            else:
                # Unknown/invalid rule character
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
            # Remove any whitespace
            rule = rule.strip()
            
            # Skip empty rules
            if not rule:
                continue
            
            # Check if rule is valid for GPU
            if HashcatRuleValidator.validate_rule_for_gpu(rule):
                valid_rules.append(rule)
        
        return valid_rules

# ====================================================================
# --- DYNAMIC CONSTANTS CALCULATION ---
# ====================================================================

def calculate_dynamic_parameters(base_count, target_count, device=None, target_hours=0.5):
    """Calculate dynamic parameters based on input data size and GPU capabilities"""
    
    # Base multipliers
    BASE_BLOOM_SIZE = 1024 * 1024 * 8  # 8MB base bloom filter
    
    # Dynamic scaling factors
    bloom_scale = max(1.0, math.log10(base_count + target_count) / 2.0)
    words_scale = max(1.0, math.log10(base_count) / 2.0)
    rules_scale = max(1.0, math.log10(target_count) / 2.0)
    
    # Calculate dynamic parameters
    BLOOM_FILTER_SIZE = int(BASE_BLOOM_SIZE * bloom_scale)
    
    # Ensure power of 2 for better performance
    BLOOM_FILTER_SIZE = 1 << (BLOOM_FILTER_SIZE.bit_length() - 1)
    
    # Get GPU work group info if available
    if device:
        try:
            # Query device limits
            max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            max_work_item_sizes = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
            max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            global_mem = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            
            # Dynamic work group size (must be divisor of max)
            possible_sizes = [32, 64, 128, 256, 512, 1024]
            LOCAL_WORK_SIZE = max([s for s in possible_sizes if s <= max_work_group_size])
            
            # Calculate optimal global size based on compute units
            # Each compute unit can handle multiple work groups
            OPTIMAL_GLOBAL_MULTIPLIER = max_compute_units * 8  # Increased for better utilization
            
            # Estimate GPU processing speed (combinations per second)
            # Based on RTX 3060 Ti performance
            EST_COMBOS_PER_SEC = 50000000  # 50 million combinations per second estimate
            
            print(f"{blue('[GPU]')} {bold('Work Group Limits:')}")
            print(f"  {cyan('[*]')} Max work group size: {max_work_group_size}")
            print(f"  {cyan('[*]')} Compute units: {max_compute_units}")
            print(f"  {cyan('[*]')} Global memory: {global_mem // (1024**3)}GB")
            print(f"  {cyan('[*]')} Using work group size: {LOCAL_WORK_SIZE}")
            
        except:
            LOCAL_WORK_SIZE = 256
            OPTIMAL_GLOBAL_MULTIPLIER = 40
            EST_COMBOS_PER_SEC = 30000000
            max_compute_units = 40
    else:
        LOCAL_WORK_SIZE = 256
        OPTIMAL_GLOBAL_MULTIPLIER = 40
        EST_COMBOS_PER_SEC = 30000000
        max_compute_units = 40
    
    # Calculate target combinations based on time limit
    target_seconds = target_hours * 3600
    max_combinations_time_limit = int(EST_COMBOS_PER_SEC * target_seconds * 0.8)  # 80% efficiency
    
    # Calculate batch sizes for 30-minute completion
    total_combinations = base_count * 100000  # Rough estimate
    
    # Scale batch sizes aggressively for speed
    WORDS_PER_BATCH = min(50000, max(10000, base_count // 10))
    CHAINS_PER_BATCH = min(20000, max(5000, int(OPTIMAL_GLOBAL_MULTIPLIER * LOCAL_WORK_SIZE / 10)))
    
    # Max chains based on time limit
    MAX_CHAINS_TO_FIND = min(1000000, max_combinations_time_limit // 1000)
    
    # Chain generation limits for 30-minute completion
    if base_count > 1000000:
        CHAIN_GEN_LIMIT_2 = 150000
        CHAIN_GEN_LIMIT_3 = 75000
        CHAIN_GEN_LIMIT_4 = 35000
        CHAIN_GEN_LIMIT_5 = 15000
    elif base_count > 100000:
        CHAIN_GEN_LIMIT_2 = 100000
        CHAIN_GEN_LIMIT_3 = 50000
        CHAIN_GEN_LIMIT_4 = 25000
        CHAIN_GEN_LIMIT_5 = 10000
    else:
        CHAIN_GEN_LIMIT_2 = 50000
        CHAIN_GEN_LIMIT_3 = 25000
        CHAIN_GEN_LIMIT_4 = 10000
        CHAIN_GEN_LIMIT_5 = 5000
    
    # Chain depth
    MAX_CHAIN_DEPTH = 3  # Limit to depth 3 for speed
    
    print(f"\n{blue('[TIME]')} {bold(f'Target completion: {target_hours} hours')}")
    print(f"{blue('[PERF]')} {bold('Estimated processing speed:')} {cyan(f'{EST_COMBOS_PER_SEC:,}')} combos/sec")
    print(f"{blue('[PERF]')} {bold('Max combinations in time:')} {cyan(f'{max_combinations_time_limit:,}')}")
    
    return {
        'BLOOM_FILTER_SIZE': BLOOM_FILTER_SIZE,
        'WORDS_PER_BATCH': WORDS_PER_BATCH,
        'CHAINS_PER_BATCH': CHAINS_PER_BATCH,
        'MAX_CHAINS_TO_FIND': MAX_CHAINS_TO_FIND,
        'MAX_CHAIN_DEPTH': MAX_CHAIN_DEPTH,
        'LOCAL_WORK_SIZE': LOCAL_WORK_SIZE,
        'OPTIMAL_GLOBAL_MULTIPLIER': OPTIMAL_GLOBAL_MULTIPLIER,
        'CHAIN_GEN_LIMIT_2': CHAIN_GEN_LIMIT_2,
        'CHAIN_GEN_LIMIT_3': CHAIN_GEN_LIMIT_3,
        'CHAIN_GEN_LIMIT_4': CHAIN_GEN_LIMIT_4,
        'CHAIN_GEN_LIMIT_5': CHAIN_GEN_LIMIT_5,
        'EST_COMBOS_PER_SEC': EST_COMBOS_PER_SEC,
        'TARGET_SECONDS': target_seconds,
        'bloom_scale': bloom_scale,
        'words_scale': words_scale,
        'rules_scale': rules_scale
    }

# Initialize with default values
MAX_WORD_LEN = 256         
MAX_RULE_LEN = 16         
MAX_OUTPUT_LEN = 512      
MAX_CHAIN_STRING_LEN = 512

# --- COLOR CODES ---
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

# ====================================================================
# --- GPU-COMPATIBLE HASHCAT RULES GENERATION ---
# ====================================================================

class GPUCompatibleRulesGenerator:
    """Generate ONLY GPU-compatible Hashcat rules"""
    
    def __init__(self, slow_mode=False):
        self.slow_mode = slow_mode
        self.validator = HashcatRuleValidator()
    
    def generate_gpu_compatible_rules(self):
        """Generate GPU-compatible Hashcat rules only"""
        rules = set()  # Use set to avoid duplicates
        
        print(f"{blue('[SETUP]')} {bold('Generating GPU-compatible Hashcat rules...')}")
        
        # ===== CATEGORY 1: SIMPLE RULES (always GPU compatible) =====
        print(f"  {cyan('[*]')} Simple rules...")
        simple_rules = [
            'l', 'u', 'c', 'C', 't', 'r', 'd', 'f', 'p', 'z', 'Z', 'q', 'E',
            '{', '}', '[', ']', 'k', 'K', ':', 'a'
        ]
        rules.update(simple_rules)
        
        # ===== CATEGORY 2: POSITION-BASED RULES (GPU compatible) =====
        print(f"  {cyan('[*]')} Position-based rules (0-9 only for GPU)...")
        digits = '0123456789'
        
        position_cmds = ['T', 'D', 'L', 'R', '+', '-', '.', ',', '\'', 'z', 'Z']
        for cmd in position_cmds:
            for pos in digits:
                rules.add(f'{cmd}{pos}')
        
        # Two position rules with digits only
        for cmd in ['T', 'x', '*', 'K']:
            for pos1 in digits:
                for pos2 in digits:
                    rules.add(f'{cmd}{pos1}{pos2}')
        
        # ===== CATEGORY 3: PREFIX/SUFFIX =====
        print(f"  {cyan('[*]')} Prefix/suffix rules...")
        # All printable ASCII characters except control chars
        for i in range(32, 127):
            char = chr(i)
            rules.add(f'^{char}')
            rules.add(f'${char}')
            rules.add(f'@{char}')
        
        # ===== CATEGORY 4: SUBSTITUTIONS (sXY) =====
        print(f"  {cyan('[*]')} Substitution rules...")
        # Common leet substitutions
        leet_subs = [
            ('a', '@'), ('a', '4'), ('e', '3'), ('i', '1'), ('o', '0'), 
            ('s', '$'), ('s', '5'), ('t', '7'), ('l', '1'), ('g', '9'),
            ('b', '8'), ('z', '2')
        ]
        
        for orig, sub in leet_subs:
            rules.add(f's{orig}{sub}')
        
        # All possible substitutions (limited to avoid explosion)
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
                # Additional sanity check: rule length
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
        self.max_work_group_size = 256
        self.local_work_size = params.get('LOCAL_WORK_SIZE', 256)
        
    def initialize_gpu(self):
        """Initialize OpenCL with dynamic parameters"""
        try:
            platform = cl.get_platforms()[0]
            self.device = platform.get_devices(cl.device_type.GPU)[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            
            # Get GPU info
            global_mem = self.device.global_mem_size
            self.max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            max_compute_units = self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            
            print(f"{green('[GPU]')} {bold('GPU:')} {cyan(self.device.name.strip())}")
            print(f"{blue('[INFO]')} {bold('Global Memory:')} {cyan(f'{global_mem // (1024**3)}GB')}")
            print(f"{blue('[INFO]')} {bold('Max Work Group Size:')} {cyan(self.max_work_group_size)}")
            print(f"{blue('[INFO]')} {bold('Compute Units:')} {cyan(max_compute_units)}")
            
            # Adjust local work size based on device capabilities
            self.local_work_size = min(self.local_work_size, self.max_work_group_size)
            # Ensure it's a divisor of max (power of 2)
            while self.max_work_group_size % self.local_work_size != 0 and self.local_work_size > 32:
                self.local_work_size //= 2
            
            print(f"{blue('[INFO]')} {bold('Using Work Group Size:')} {cyan(self.local_work_size)}")
            
            return True
        except Exception as e:
            print(f"{red('[ERROR]')} {bold('GPU initialization failed:')} {e}")
            return False
    
    def compile_kernel(self):
        """Compile the fixed GPU kernel"""
        try:
            print(f"{blue('[SETUP]')} {bold('Compiling GPU-compatible kernel...')}")
            
            # Read the fixed kernel source
            kernel_source = GPU_COMPATIBLE_KERNEL
            
            # Replace constants with dynamic values
            kernel_source = kernel_source.replace(
                "BLOOM_FILTER_SIZE 8388608",
                f"BLOOM_FILTER_SIZE {self.params['BLOOM_FILTER_SIZE']}"
            )
            kernel_source = kernel_source.replace(
                "MAX_CHAINS_TO_FIND 10000000",
                f"MAX_CHAINS_TO_FIND {self.params['MAX_CHAINS_TO_FIND']}"
            )
            kernel_source = kernel_source.replace(
                "MAX_CHAIN_DEPTH 6",
                f"MAX_CHAIN_DEPTH {self.params['MAX_CHAIN_DEPTH']}"
            )
            
            self.program = cl.Program(self.context, kernel_source).build()
            print(f"{green('[OK]')} {bold('Kernel compiled successfully')}")
            return self.program
        except Exception as e:
            print(f"{red('[ERROR]')} {bold('Kernel compilation failed:')}")
            print(f"  {str(e)}")
            return None
    
    def generate_bloom_filter(self, target_words):
        """Generate Bloom filter with dynamic size - using ALL target words"""
        print(f"{blue('[SETUP]')} {bold('Generating comprehensive Bloom filter...')}")
        
        bloom_size_bytes = self.params['BLOOM_FILTER_SIZE'] // 8
        bloom_filter = np.zeros(bloom_size_bytes, dtype=np.uint8)
        
        # Use ALL target words for maximum accuracy
        print(f"  {cyan('[*]')} Bloom filter size: {bloom_size_bytes / 1024:.1f} KB")
        print(f"  {cyan('[*]')} Hashing ALL target words: {len(target_words):,}")
        
        for word in tqdm(target_words, desc="Building bloom filter", leave=False):
            word_bytes = word.encode('latin-1')
            
            # Use multiple hash functions
            hash_funcs = [
                hashlib.md5,
                hashlib.sha1,
                hashlib.sha256,
                hashlib.sha512
            ]
            
            for i, hash_func in enumerate(hash_funcs):
                hash_val = int(hash_func(word_bytes).hexdigest()[:8], 16)
                idx = hash_val % self.params['BLOOM_FILTER_SIZE']
                byte_idx = idx // 8
                bit_idx = idx % 8
                bloom_filter[byte_idx] |= (1 << bit_idx)
        
        # Calculate and display statistics
        bits_set = np.sum(np.unpackbits(bloom_filter))
        fill_ratio = bits_set / self.params['BLOOM_FILTER_SIZE']
        
        print(f"  {cyan('[*]')} Bloom filter fill ratio: {fill_ratio:.3%}")
        print(f"  {cyan('[*]')} False positive rate: {(fill_ratio**4):.6%}")
        
        return bloom_filter
    
    def prepare_batch_data(self, words, rules):
        """Prepare data for a single batch"""
        # Prepare words
        words_flat = []
        word_offsets = []
        word_lengths = []
        current_offset = 0
        
        for word in words:
            word_bytes = word.encode('latin-1')
            words_flat.extend(word_bytes)
            word_offsets.append(current_offset)
            word_lengths.append(len(word_bytes))
            current_offset += len(word_bytes)
        
        # Prepare rules
        rules_flat = []
        rule_offsets = []
        rule_lengths = []
        current_offset = 0
        
        for rule in rules:
            rule_bytes = rule.encode('latin-1')
            rules_flat.extend(rule_bytes)
            rule_offsets.append(current_offset)
            rule_lengths.append(len(rule_bytes))
            current_offset += len(rule_bytes)
        
        return {
            'words_flat': np.array(words_flat, dtype=np.uint8),
            'word_offsets': np.array(word_offsets, dtype=np.int32),
            'word_lengths': np.array(word_lengths, dtype=np.int32),
            'rules_flat': np.array(rules_flat, dtype=np.uint8),
            'rule_offsets': np.array(rule_offsets, dtype=np.int32),
            'rule_lengths': np.array(rule_lengths, dtype=np.int32),
            'num_words': len(words),
            'num_rules': len(rules)
        }
    
    def process_all_words_single_rule(self, base_words, target_words, rules):
        """Process ALL base words with single rules"""
        print(f"{blue('[GPU]')} {bold('Processing ALL words with single rules...')}")
        
        # Generate Bloom filter with ALL target words
        bloom_filter = self.generate_bloom_filter(target_words)
        
        # Compile kernel
        if not self.compile_kernel():
            return []
        
        # Validate rules for GPU compatibility
        gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(rules)
        
        print(f"{blue('[INFO]')} {bold('GPU-compatible rules:')} {len(gpu_rules):,}")
        
        all_found_rules = []
        
        # Calculate optimal batch size
        optimal_global_size = self.params['OPTIMAL_GLOBAL_MULTIPLIER'] * self.local_work_size
        items_per_combination = len(gpu_rules)
        optimal_words_per_batch = max(1000, optimal_global_size // items_per_combination)
        
        # Adjust batch size
        batch_size = min(
            self.params['WORDS_PER_BATCH'],
            optimal_words_per_batch,
            len(base_words)
        )
        
        num_batches = (len(base_words) + batch_size - 1) // batch_size
        
        print(f"{blue('[INFO]')} {bold('Processing ALL')} {len(base_words):,} {bold('words in')} {num_batches} {bold('batches')}")
        print(f"{blue('[INFO]')} {bold('Batch size:')} {batch_size:,} words")
        
        # Process batches
        with tqdm(total=num_batches, desc="Processing all words", unit="batch") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(base_words))
                batch_words = base_words[start_idx:end_idx]
                
                if not batch_words:
                    pbar.update(1)
                    continue
                
                # Process this batch
                batch_data = self.prepare_batch_data(batch_words, gpu_rules)
                batch_found = self.process_batch_single(batch_data, bloom_filter, gpu_rules)
                
                if batch_found:
                    all_found_rules.extend(batch_found)
                
                # Update progress
                pbar.set_postfix({
                    'found': len(all_found_rules),
                    'progress': f"{end_idx:,}/{len(base_words):,}"
                })
                pbar.update(1)
        
        # Remove duplicates and return
        all_found_rules = list(set(all_found_rules))
        print(f"\n{green('[OK]')} {bold('Total unique single rules found:')} {cyan(len(all_found_rules))}")
        
        return all_found_rules
    
    def process_batch_single(self, batch_data, bloom_filter, rules):
        """Process a single batch on GPU"""
        mf = cl.mem_flags
        
        try:
            # Create buffers
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
            
            bloom_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=bloom_filter)
            
            # Output buffers - dynamically sized
            max_output_size = min(self.params['MAX_CHAINS_TO_FIND'], 
                                 batch_data['num_words'] * batch_data['num_rules'])
            
            found_rules_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                      max_output_size * MAX_CHAIN_STRING_LEN)
            found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)
            
            # Initialize counter
            zero_count = np.array([0], dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count_buf, zero_count)
            
            # Run kernel
            kernel = self.program.find_single_rules_gpu
            
            # Calculate global size
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
            
            # Read results
            found_count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count, found_count_buf)
            
            num_found = min(found_count[0], max_output_size)
            
            batch_found = []
            if num_found > 0:
                # Read found rules
                found_data = np.zeros(num_found * MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, found_data, found_rules_buf)
                
                # Convert to strings
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
            # Cleanup all buffers
            try:
                base_buf.release()
                base_offsets_buf.release()
                base_lengths_buf.release()
                rules_buf.release()
                rule_offsets_buf.release()
                rule_lengths_buf.release()
                bloom_buf.release()
                found_rules_buf.release()
                found_count_buf.release()
            except:
                pass
    
    def process_all_words_chain_rules(self, base_words, target_words, rules, max_depth):
        """Process ALL base words with rule chains - optimized for speed"""
        print(f"{blue('[GPU]')} {bold('Processing ALL words with rule chains...')}")
        
        # Generate Bloom filter with ALL target words
        bloom_filter = self.generate_bloom_filter(target_words)
        
        # Compile kernel
        if not self.compile_kernel():
            return []
        
        # Validate rules for GPU compatibility
        gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(rules)
        
        # Generate chains from valid rules
        print(f"{blue('[SETUP]')} {bold('Generating rule chains...')}")
        chains = self.generate_valid_chains(gpu_rules, min(max_depth, 3))  # Limit to depth 3 for speed
        
        if not chains:
            return []
        
        print(f"{blue('[INFO]')} {bold('Total chains generated:')} {len(chains):,}")
        
        all_chains = []
        
        # Calculate optimal batch sizes for 30-minute completion
        optimal_global_size = self.params['OPTIMAL_GLOBAL_MULTIPLIER'] * self.local_work_size
        
        # Calculate words per batch based on target time
        total_combinations = len(base_words) * len(chains)
        target_batches = max(10, int(total_combinations / (optimal_global_size * 50)))
        
        # Adjust chain batch size for speed
        chain_batch_size = max(5000, min(20000, len(chains) // target_batches))
        
        # Ensure we don't have too many batches
        num_chain_batches = (len(chains) + chain_batch_size - 1) // chain_batch_size
        
        print(f"{blue('[INFO]')} {bold('Processing')} {len(chains):,} {bold('chains in')} {num_chain_batches} {bold('batches')}")
        print(f"{blue('[INFO]')} {bold('Chain batch size:')} {chain_batch_size:,}")
        print(f"{blue('[INFO]')} {bold('Total combinations:')} {total_combinations:,}")
        
        # Process chain batches
        with tqdm(total=num_chain_batches, desc="Chain batches", unit="batch") as chain_pbar:
            for chain_batch_idx in range(0, len(chains), chain_batch_size):
                chain_end = min(chain_batch_idx + chain_batch_size, len(chains))
                chain_batch = chains[chain_batch_idx:chain_end]
                
                # Process this chain batch with all words
                batch_chains = self._process_chain_batch(
                    base_words, gpu_rules, chain_batch, bloom_filter
                )
                
                if batch_chains:
                    all_chains.extend(batch_chains)
                
                chain_pbar.update(1)
                chain_pbar.set_postfix({
                    'found': len(all_chains),
                    'progress': f"{chain_end}/{len(chains)}"
                })
        
        # Remove duplicates and return
        all_chains = list(set(all_chains))
        print(f"\n{green('[OK]')} {bold('Total unique chains found:')} {cyan(len(all_chains))}")
        
        return all_chains
    
    def generate_valid_chains(self, rules, max_depth):
        """Generate valid rule chains up to max_depth - optimized for speed"""
        print(f"  {cyan('->')} Generating chains up to depth {max_depth}...")
        
        # Validate all rules first
        valid_rules = [r for r in rules if HashcatRuleValidator.validate_rule_for_gpu(r)]
        
        if not valid_rules:
            print(f"  {yellow('[WARN]')} No valid rules found")
            return []
        
        # Use all valid rules
        rules_to_use = valid_rules
        print(f"  {cyan('[*]')} Using all {len(rules_to_use)} rules for chain generation")
        
        chains = set()
        
        # Depth 1: Single rules
        print(f"  {cyan('->')} Depth 1 chains...")
        for rule in rules_to_use:
            chains.add(rule)
        
        # Generate chains for depth 2 and 3 only (for speed)
        for depth in range(2, max_depth + 1):
            print(f"  {cyan('->')} Depth {depth} chains...")
            
            import random
            random.seed(42)
            
            # Calculate theoretical maximum combinations
            max_combinations = len(rules_to_use) ** depth
            
            # Get depth-specific limit
            if depth == 2:
                target_combinations = self.params.get('CHAIN_GEN_LIMIT_2', 100000)
            elif depth == 3:
                target_combinations = self.params.get('CHAIN_GEN_LIMIT_3', 50000)
            else:
                target_combinations = self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 25000)
            
            # Adjust based on total combinations
            target_combinations = min(target_combinations, max_combinations)
            
            print(f"  {cyan('[*]')} Generating up to {target_combinations:,} chains...")
            
            chains_added = 0
            attempts = 0
            max_attempts = target_combinations * 5  # Reduced attempts for speed
            
            # Track generated patterns
            generated_patterns = set()
            
            while chains_added < target_combinations and attempts < max_attempts:
                attempts += 1
                
                # Generate chain parts
                chain_parts = [random.choice(rules_to_use) for _ in range(depth)]
                
                # Create pattern key
                pattern_key = ' '.join(chain_parts)
                
                # Check if valid
                valid = all(HashcatRuleValidator.validate_rule_for_gpu(rule) for rule in chain_parts)
                
                # Add if valid and not duplicate
                if valid and pattern_key not in generated_patterns:
                    chains.add(pattern_key)
                    generated_patterns.add(pattern_key)
                    chains_added += 1
        
        chains_list = list(chains)
        print(f"  {cyan('[*]')} Generated {len(chains_list):,} chains")
        
        return chains_list
    
    def _process_chain_batch(self, words, rules, chains, bloom_filter):
        """Process a single chain batch optimized for speed"""
        
        # Prepare chain sequences
        chain_sequences = []
        chain_depths = []
        
        for chain in chains:
            chain_rules = chain.split()
            depth = len(chain_rules)
            chain_depths.append(depth)
            
            rule_indices = []
            for rule in chain_rules:
                try:
                    idx = rules.index(rule)
                    rule_indices.append(idx)
                except ValueError:
                    rule_indices.append(-1)
            
            # Pad to max depth
            while len(rule_indices) < self.params['MAX_CHAIN_DEPTH']:
                rule_indices.append(-1)
            
            chain_sequences.extend(rule_indices)
        
        # Prepare data
        batch_data = self.prepare_batch_data(words, rules)
        
        mf = cl.mem_flags
        
        try:
            # Create buffers
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
            
            bloom_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=bloom_filter)
            
            # Output buffers
            max_output_size = min(self.params['MAX_CHAINS_TO_FIND'], 
                                 len(words) * len(chains))
            
            found_chains_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                       max_output_size * MAX_CHAIN_STRING_LEN)
            found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)
            
            # Initialize counter
            zero_count = np.array([0], dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count_buf, zero_count)
            
            # Run kernel
            kernel = self.program.find_rule_chains_gpu
            
            total_items = len(words) * len(chains)
            
            # Calculate global size
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
            
            # Read results
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
            # Cleanup all buffers
            try:
                base_buf.release()
                base_offsets_buf.release()
                base_lengths_buf.release()
                rules_buf.release()
                rule_offsets_buf.release()
                rule_lengths_buf.release()
                chain_seq_buf.release()
                chain_depth_buf.release()
                bloom_buf.release()
                found_chains_buf.release()
                found_count_buf.release()
            except:
                pass

# ====================================================================
# --- GPU EXTRACTOR ---
# ====================================================================

class GPUExtractor:
    """GPU-optimized extractor with complete processing"""
    
    def __init__(self, base_count, target_count, device=None, slow_mode=False, target_hours=0.5):
        self.base_count = base_count
        self.target_count = target_count
        self.slow_mode = slow_mode
        
        # Calculate dynamic parameters with GPU info and time target
        self.params = calculate_dynamic_parameters(base_count, target_count, device, target_hours)
        
        print(f"{blue('[CONFIG]')} {bold('GPU-Optimized Configuration:')}")
        print(f"  {cyan('[*]')} Bloom filter size: {self.params['BLOOM_FILTER_SIZE']:,} bits")
        print(f"  {cyan('[*]')} Words per batch: {self.params['WORDS_PER_BATCH']:,}")
        print(f"  {cyan('[*]')} Chains per batch: {self.params['CHAINS_PER_BATCH']:,}")
        print(f"  {cyan('[*]')} Max chains to find: {self.params['MAX_CHAINS_TO_FIND']:,}")
        print(f"  {cyan('[*]')} Max chain depth: {self.params['MAX_CHAIN_DEPTH']}")
        print(f"  {cyan('[*]')} Chain gen limits: D2={self.params['CHAIN_GEN_LIMIT_2']:,}, D3={self.params['CHAIN_GEN_LIMIT_3']:,}")
        
        # Initialize components
        self.rules_generator = GPUCompatibleRulesGenerator(slow_mode)
        self.gpu_engine = GPUEngine(self.params)
        self.validator = HashcatRuleValidator()
    
    def extract_rules(self, base_words, target_words, max_depth=3):
        """Extract GPU-compatible rules using complete processing"""
        print(f"{blue('[MAIN]')} {bold('Starting GPU-optimized rule extraction...')}")
        
        all_chains = []
        
        # Generate GPU-compatible rules
        rules = self.rules_generator.generate_gpu_compatible_rules()
        
        # Initialize GPU
        if not self.gpu_engine.initialize_gpu():
            print(f"{yellow('[WARN]')} {bold('GPU not available')}")
            return []
        
        # Phase 1: Single rules with ALL words
        print(f"\n{blue('=' * 60)}")
        print(f"{bold('PHASE 1: SINGLE RULE SEARCH (ALL WORDS)')}")
        print(f"{blue('=' * 60)}")
        
        single_chains = self.gpu_engine.process_all_words_single_rule(
            base_words, target_words, rules
        )
        all_chains.extend(single_chains)
        
        print(f"{green('[OK]')} {bold('Single rules found:')} {cyan(len(single_chains))}")
        
        # Phase 2: Rule chains with ALL words
        if max_depth > 1:
            print(f"\n{blue('=' * 60)}")
            print(f"{bold('PHASE 2: RULE CHAIN SEARCH (ALL WORDS)')}")
            print(f"{blue('=' * 60)}")
            
            chain_chains = self.gpu_engine.process_all_words_chain_rules(
                base_words, target_words, rules, max_depth
            )
            all_chains.extend(chain_chains)
            
            print(f"{green('[OK]')} {bold('Rule chains found:')} {cyan(len(chain_chains))}")
        
        # Final cleanup
        print(f"\n{blue('=' * 60)}")
        print(f"{bold('FINAL CLEANUP')}")
        print(f"{blue('=' * 60)}")
        
        # Validate for GPU compatibility
        final_chains = self.validator.validate_rules_for_gpu(all_chains)
        
        return final_chains

# ====================================================================
# --- FIXED GPU-COMPATIBLE KERNEL ---
# ====================================================================

GPU_COMPATIBLE_KERNEL = """
#define MAX_WORD_LEN 256
#define MAX_RULE_LEN 16
#define MAX_OUTPUT_LEN 512
#define MAX_CHAIN_STRING_LEN 512
#define MAX_CHAINS_TO_FIND 10000000
#define MAX_CHAIN_DEPTH 6
#define BLOOM_FILTER_SIZE 8388608
#define BLOOM_HASH_FUNCTIONS 4

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

inline int is_lower(unsigned char c) {
    return (c >= 'a' && c <= 'z');
}

inline int is_upper(unsigned char c) {
    return (c >= 'A' && c <= 'Z');
}

inline int is_digit(unsigned char c) {
    return (c >= '0' && c <= '9');
}

inline int is_hex_digit(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
}

inline int hex_to_int(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return 0;
}

inline unsigned char to_lower(unsigned char c) {
    if (is_upper(c)) return c + 32;
    return c;
}

inline unsigned char to_upper(unsigned char c) {
    if (is_lower(c)) return c - 32;
    return c;
}

inline unsigned char toggle_case(unsigned char c) {
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}

// ============================================================================
// BLOOM FILTER FUNCTIONS
// ============================================================================

uint bloom_hash(const unsigned char *str, int len, uint seed) {
    uint hash = seed;
    for (int i = 0; i < len; i++) {
        hash = (hash * 31) + str[i];
    }
    return hash;
}

int bloom_check(__global const uchar *bloom_filter, const unsigned char *word, int len) {
    uint h1 = bloom_hash(word, len, 0xDEADBEEF);
    uint h2 = bloom_hash(word, len, 0xCAFEBABE);
    
    for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {
        uint hash_val = (h1 + i * h2) % BLOOM_FILTER_SIZE;
        uint byte_idx = hash_val / 8;
        uint bit_idx = hash_val % 8;
        
        if (!(bloom_filter[byte_idx] & (1 << bit_idx))) {
            return 0;
        }
    }
    return 1;
}

// ============================================================================
// GPU-COMPATIBLE RULE APPLICATION (SAFE FOR OPENCL)
// ============================================================================

int apply_gpu_rule(
    const unsigned char *rule_str, int rule_len,
    const unsigned char *input_word, int input_len,
    unsigned char *output_word, int *output_len
) {
    // Initialize with input
    *output_len = input_len;
    for (int i = 0; i < input_len; i++) {
        output_word[i] = input_word[i];
    }
    output_word[input_len] = '\\0';
    
    if (rule_len == 0 || input_len == 0) return 1;
    
    unsigned char cmd = rule_str[0];
    int changed = 0;
    
    // ========================================================================
    // SINGLE CHARACTER RULES
    // ========================================================================
    
    if (rule_len == 1) {
        switch (cmd) {
            case 'l': // Lowercase all
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = to_lower(output_word[i]);
                }
                changed = 1;
                break;
                
            case 'u': // Uppercase all
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = to_upper(output_word[i]);
                }
                changed = 1;
                break;
                
            case 'c': // Capitalize first, lowercase rest
                if (*output_len > 0) {
                    output_word[0] = to_upper(output_word[0]);
                    for (int i = 1; i < *output_len; i++) {
                        output_word[i] = to_lower(output_word[i]);
                    }
                }
                changed = 1;
                break;
                
            case 'C': // Lowercase first, uppercase rest
                if (*output_len > 0) {
                    output_word[0] = to_lower(output_word[0]);
                    for (int i = 1; i < *output_len; i++) {
                        output_word[i] = to_upper(output_word[i]);
                    }
                }
                changed = 1;
                break;
                
            case 't': // Toggle case
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = toggle_case(output_word[i]);
                }
                changed = 1;
                break;
                
            case 'r': // Reverse
                for (int i = 0; i < *output_len / 2; i++) {
                    unsigned char temp = output_word[i];
                    output_word[i] = output_word[*output_len - 1 - i];
                    output_word[*output_len - 1 - i] = temp;
                }
                changed = 1;
                break;
                
            case 'd': // Duplicate
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {
                    for (int i = 0; i < *output_len; i++) {
                        output_word[*output_len + i] = output_word[i];
                    }
                    *output_len *= 2;
                }
                changed = 1;
                break;
                
            case 'f': // Reflect
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {
                    for (int i = 0; i < *output_len; i++) {
                        output_word[*output_len + i] = output_word[*output_len - 1 - i];
                    }
                    *output_len *= 2;
                }
                changed = 1;
                break;
                
            case '{': // Rotate left
                if (*output_len > 1) {
                    unsigned char first = output_word[0];
                    for (int i = 0; i < *output_len - 1; i++) {
                        output_word[i] = output_word[i + 1];
                    }
                    output_word[*output_len - 1] = first;
                }
                changed = 1;
                break;
                
            case '}': // Rotate right
                if (*output_len > 1) {
                    unsigned char last = output_word[*output_len - 1];
                    for (int i = *output_len - 1; i > 0; i--) {
                        output_word[i] = output_word[i - 1];
                    }
                    output_word[0] = last;
                }
                changed = 1;
                break;
                
            case '[': // Delete first char
                if (*output_len > 0) {
                    for (int i = 0; i < *output_len - 1; i++) {
                        output_word[i] = output_word[i + 1];
                    }
                    (*output_len)--;
                }
                changed = 1;
                break;
                
            case ']': // Delete last char
                if (*output_len > 0) {
                    (*output_len)--;
                }
                changed = 1;
                break;
                
            case 'k': // Swap first two
                if (*output_len >= 2) {
                    unsigned char temp = output_word[0];
                    output_word[0] = output_word[1];
                    output_word[1] = temp;
                }
                changed = 1;
                break;
                
            case 'K': // Swap last two
                if (*output_len >= 2) {
                    unsigned char temp = output_word[*output_len - 2];
                    output_word[*output_len - 2] = output_word[*output_len - 1];
                    output_word[*output_len - 1] = temp;
                }
                changed = 1;
                break;
                
            case ':': // No operation
                changed = 0;
                break;
                
            case 'a': // Toggle case recursively
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = toggle_case(output_word[i]);
                }
                changed = 1;
                break;
                
            case 'q': // Duplicate all chars
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {
                    int idx = 0;
                    unsigned char temp[MAX_OUTPUT_LEN];
                    for (int i = 0; i < *output_len; i++) {
                        temp[i] = output_word[i];
                    }
                    for (int i = 0; i < *output_len; i++) {
                        output_word[idx++] = temp[i];
                        output_word[idx++] = temp[i];
                    }
                    *output_len *= 2;
                }
                changed = 1;
                break;
                
            case 'E': // Title case
                if (*output_len > 0) {
                    int capitalize = 1;
                    for (int i = 0; i < *output_len; i++) {
                        if (capitalize && is_lower(output_word[i])) {
                            output_word[i] = to_upper(output_word[i]);
                            capitalize = 0;
                        }
                        if (output_word[i] == ' ' || output_word[i] == '-' || output_word[i] == '_') {
                            capitalize = 1;
                        }
                    }
                }
                changed = 1;
                break;
        }
    }
    
    // ========================================================================
    // TWO CHARACTER RULES
    // ========================================================================
    
    else if (rule_len == 2) {
        unsigned char param = rule_str[1];
        
        if (cmd == '^') { // Prepend
            if (*output_len + 1 <= MAX_OUTPUT_LEN) {
                for (int i = *output_len; i > 0; i--) {
                    output_word[i] = output_word[i - 1];
                }
                output_word[0] = param;
                (*output_len)++;
                changed = 1;
            }
        }
        else if (cmd == '$') { // Append
            if (*output_len + 1 <= MAX_OUTPUT_LEN) {
                output_word[*output_len] = param;
                (*output_len)++;
                changed = 1;
            }
        }
        else if (cmd == '@') { // Delete all instances
            int new_len = 0;
            for (int i = 0; i < *output_len; i++) {
                if (output_word[i] != param) {
                    output_word[new_len++] = output_word[i];
                } else {
                    changed = 1;
                }
            }
            *output_len = new_len;
        }
        else if (cmd == 'p') { // Purge character
            int new_len = 0;
            for (int i = 0; i < *output_len; i++) {
                if (output_word[i] != param) {
                    output_word[new_len++] = output_word[i];
                } else {
                    changed = 1;
                }
            }
            *output_len = new_len;
        }
        else if (cmd == 'T' && is_digit(param)) { // Toggle at position
            int pos = param - '0';
            if (pos < *output_len) {
                output_word[pos] = toggle_case(output_word[pos]);
                changed = 1;
            }
        }
        else if (cmd == 'D' && is_digit(param)) { // Delete at position
            int pos = param - '0';
            if (pos < *output_len) {
                for (int i = pos; i < *output_len - 1; i++) {
                    output_word[i] = output_word[i + 1];
                }
                (*output_len)--;
                changed = 1;
            }
        }
        else if (cmd == 'L' && is_digit(param)) { // Delete left
            int pos = param - '0';
            if (pos < *output_len) {
                int new_len = 0;
                for (int i = pos; i < *output_len; i++) {
                    output_word[new_len++] = output_word[i];
                }
                *output_len = new_len;
                changed = 1;
            }
        }
        else if (cmd == 'R' && is_digit(param)) { // Delete right
            int pos = param - '0';
            if (pos < *output_len) {
                *output_len = pos + 1;
                changed = 1;
            }
        }
        else if (cmd == '+' && is_digit(param)) { // Increment
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] < 255) {
                output_word[pos]++;
                changed = 1;
            }
        }
        else if (cmd == '-' && is_digit(param)) { // Decrement
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] > 0) {
                output_word[pos]--;
                changed = 1;
            }
        }
        else if (cmd == '.' && is_digit(param)) { // Replace with dot
            int pos = param - '0';
            if (pos < *output_len) {
                output_word[pos] = '.';
                changed = 1;
            }
        }
        else if (cmd == ',' && is_digit(param)) { // Replace with comma
            int pos = param - '0';
            if (pos < *output_len) {
                output_word[pos] = ',';
                changed = 1;
            }
        }
        else if (cmd == '\\'' && is_digit(param)) { // Increment (apostrophe)
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] < 255) {
                output_word[pos]++;
                changed = 1;
            }
        }
        else if (cmd == 'z' && is_digit(param)) { // Duplicate first
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {
                unsigned char first = output_word[0];
                for (int i = *output_len + n - 1; i >= n; i--) {
                    output_word[i] = output_word[i - n];
                }
                for (int i = 0; i < n; i++) {
                    output_word[i] = first;
                }
                *output_len += n;
                changed = 1;
            }
        }
        else if (cmd == 'Z' && is_digit(param)) { // Duplicate last
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {
                unsigned char last = output_word[*output_len - 1];
                for (int i = 0; i < n; i++) {
                    output_word[*output_len + i] = last;
                }
                *output_len += n;
                changed = 1;
            }
        }
        else if (cmd == 'p' && is_digit(param)) { // Duplicate word N times
            int n = param - '0';
            if (*output_len * (n + 1) <= MAX_OUTPUT_LEN) {
                int original_len = *output_len;
                for (int rep = 0; rep < n; rep++) {
                    for (int i = 0; i < original_len; i++) {
                        output_word[*output_len + i] = output_word[i];
                    }
                    *output_len += original_len;
                }
                changed = 1;
            }
        }
        else if (cmd == 'y' && is_digit(param)) { // Duplicate first block
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {
                for (int i = 0; i < n; i++) {
                    output_word[*output_len + i] = output_word[i];
                }
                *output_len += n;
                changed = 1;
            }
        }
        else if (cmd == 'Y' && is_digit(param)) { // Duplicate last block
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {
                for (int i = 0; i < n; i++) {
                    output_word[*output_len + i] = output_word[*output_len - n + i];
                }
                *output_len += n;
                changed = 1;
            }
        }
        else if (cmd == '{' && is_digit(param)) { // Rotate left N
            int n = param - '0';
            if (*output_len > 0 && n > 0) {
                n = n % *output_len;
                if (n > 0) {
                    unsigned char temp[MAX_OUTPUT_LEN];
                    for (int i = 0; i < *output_len; i++) {
                        temp[i] = output_word[i];
                    }
                    for (int i = 0; i < *output_len; i++) {
                        output_word[i] = temp[(i + n) % *output_len];
                    }
                    changed = 1;
                }
            }
        }
        else if (cmd == '}' && is_digit(param)) { // Rotate right N
            int n = param - '0';
            if (*output_len > 0 && n > 0) {
                n = n % *output_len;
                if (n > 0) {
                    unsigned char temp[MAX_OUTPUT_LEN];
                    for (int i = 0; i < *output_len; i++) {
                        temp[i] = output_word[i];
                    }
                    for (int i = 0; i < *output_len; i++) {
                        output_word[i] = temp[(i - n + *output_len) % *output_len];
                    }
                    changed = 1;
                }
            }
        }
        else if (cmd == '[' && is_digit(param)) { // Delete first N
            int n = param - '0';
            if (n >= *output_len) {
                *output_len = 0;
                changed = 1;
            } else if (n > 0) {
                for (int i = n; i < *output_len; i++) {
                    output_word[i - n] = output_word[i];
                }
                *output_len -= n;
                changed = 1;
            }
        }
        else if (cmd == ']' && is_digit(param)) { // Delete last N
            int n = param - '0';
            if (n >= *output_len) {
                *output_len = 0;
                changed = 1;
            } else if (n > 0) {
                *output_len -= n;
                changed = 1;
            }
        }
    }
    
    // ========================================================================
    // THREE CHARACTER RULES
    // ========================================================================
    
    else if (rule_len == 3) {
        unsigned char param1 = rule_str[1];
        unsigned char param2 = rule_str[2];
        
        if (cmd == 's') { // Substitute
            for (int i = 0; i < *output_len; i++) {
                if (output_word[i] == param1) {
                    output_word[i] = param2;
                    changed = 1;
                }
            }
        }
        else if (cmd == 'i' && is_digit(param1)) { // Insert at position
            int pos = param1 - '0';
            if (pos <= *output_len && *output_len + 1 <= MAX_OUTPUT_LEN) {
                for (int i = *output_len; i > pos; i--) {
                    output_word[i] = output_word[i - 1];
                }
                output_word[pos] = param2;
                (*output_len)++;
                changed = 1;
            }
        }
        else if (cmd == 'o' && is_digit(param1)) { // Overwrite at position
            int pos = param1 - '0';
            if (pos < *output_len) {
                output_word[pos] = param2;
                changed = 1;
            }
        }
        else if (cmd == 'e') { // Title case with separator
            unsigned char separator = param1;
            if (*output_len > 0) {
                int capitalize = 1;
                for (int i = 0; i < *output_len; i++) {
                    if (capitalize && is_lower(output_word[i])) {
                        output_word[i] = to_upper(output_word[i]);
                        capitalize = 0;
                    }
                    if (output_word[i] == separator) {
                        capitalize = 1;
                    }
                }
                changed = 1;
            }
        }
        else if (cmd == 'x' && is_digit(param1) && is_digit(param2)) { // Extract
            int n = param1 - '0';
            int m = param2 - '0';
            if (n > m) {
                int temp = n; n = m; m = temp;
            }
            if (n < *output_len) {
                int new_len = 0;
                for (int i = n; i <= m && i < *output_len; i++) {
                    output_word[new_len++] = output_word[i];
                }
                *output_len = new_len;
                changed = 1;
            }
        }
        else if (cmd == '*' && is_digit(param1) && is_digit(param2)) { // Swap positions
            int n = param1 - '0';
            int m = param2 - '0';
            if (n < *output_len && m < *output_len && n != m) {
                unsigned char temp = output_word[n];
                output_word[n] = output_word[m];
                output_word[m] = temp;
                changed = 1;
            }
        }
        else if (cmd == 'K' && is_digit(param1) && is_digit(param2)) { // Swap ranges
            int n = param1 - '0';
            int m = param2 - '0';
            if (n < *output_len && m < *output_len && n <= m) {
                for (int i = n; i <= (n + m) / 2 && i <= m; i++) {
                    int j = n + m - i;
                    if (j < *output_len) {
                        unsigned char temp = output_word[i];
                        output_word[i] = output_word[j];
                        output_word[j] = temp;
                    }
                }
                changed = 1;
            }
        }
    }
    
    output_word[*output_len] = '\\0';
    return changed ? 1 : 0;
}

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
) {
    int gid = get_global_id(0);
    int total_items = num_base_words * num_rules;
    
    if (gid >= total_items) return;
    
    int word_idx = gid / num_rules;
    int rule_idx = gid % num_rules;
    
    // Get base word
    __global const unsigned char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];
    
    unsigned char input_word[MAX_WORD_LEN];
    for (int i = 0; i < word_len; i++) {
        input_word[i] = word_ptr[i];
    }
    input_word[word_len] = '\\0';
    
    // Get rule
    __global const unsigned char *rule_ptr = rules + rule_offsets[rule_idx];
    int rule_len = rule_lengths[rule_idx];
    
    unsigned char rule_str[MAX_RULE_LEN];
    for (int i = 0; i < rule_len; i++) {
        rule_str[i] = rule_ptr[i];
    }
    rule_str[rule_len] = '\\0';
    
    // Apply rule
    unsigned char output_word[MAX_OUTPUT_LEN];
    int output_len;
    int result = apply_gpu_rule(rule_str, rule_len, input_word, word_len, output_word, &output_len);
    
    if (result > 0 && output_len > 0) {
        // Check bloom filter
        if (bloom_check(bloom_filter, output_word, output_len)) {
            int idx = atomic_inc(found_count);
            
            if (idx < MAX_CHAINS_TO_FIND) {
                __global char *output_ptr = found_rules + idx * MAX_CHAIN_STRING_LEN;
                
                // Store rule string
                for (int i = 0; i < rule_len && i < MAX_CHAIN_STRING_LEN - 1; i++) {
                    output_ptr[i] = rule_str[i];
                }
                output_ptr[rule_len] = '\\0';
            }
        }
    }
}

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
) {
    int gid = get_global_id(0);
    int total_items = num_base_words * num_chains;
    
    if (gid >= total_items) return;
    
    int word_idx = gid / num_chains;
    int chain_idx = gid % num_chains;
    
    // Get base word
    __global const unsigned char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];
    
    unsigned char current_word[MAX_OUTPUT_LEN];
    for (int i = 0; i < word_len; i++) {
        current_word[i] = word_ptr[i];
    }
    current_word[word_len] = '\\0';
    int current_len = word_len;
    
    // Get chain depth
    int depth = chain_depths[chain_idx];
    if (depth < 1 || depth > max_chain_depth) return;
    
    // Apply chain of rules
    unsigned char temp_word[MAX_OUTPUT_LEN];
    char chain_buffer[MAX_CHAIN_STRING_LEN];
    int chain_pos = 0;
    
    for (int d = 0; d < depth; d++) {
        int rule_idx = chain_sequences[chain_idx * max_chain_depth + d];
        if (rule_idx < 0) break;
        
        // Get rule
        __global const unsigned char *rule_ptr = rules + rule_offsets[rule_idx];
        int rule_len = rule_lengths[rule_idx];
        
        unsigned char rule_str[MAX_RULE_LEN];
        for (int i = 0; i < rule_len; i++) {
            rule_str[i] = rule_ptr[i];
        }
        rule_str[rule_len] = '\\0';
        
        // Store rule in chain buffer
        for (int i = 0; i < rule_len && chain_pos < MAX_CHAIN_STRING_LEN - 2; i++) {
            chain_buffer[chain_pos++] = rule_str[i];
        }
        if (d < depth - 1 && chain_pos < MAX_CHAIN_STRING_LEN - 1) {
            chain_buffer[chain_pos++] = ' ';
        }
        
        // Apply rule
        int new_len;
        int result = apply_gpu_rule(rule_str, rule_len, current_word, current_len, temp_word, &new_len);
        
        if (result <= 0 || new_len == 0) {
            return;
        }
        
        // Copy back
        for (int i = 0; i < new_len; i++) {
            current_word[i] = temp_word[i];
        }
        current_word[new_len] = '\\0';
        current_len = new_len;
    }
    
    chain_buffer[chain_pos] = '\\0';
    
    // Check bloom filter
    if (bloom_check(bloom_filter, current_word, current_len)) {
        int idx = atomic_inc(found_count);
        
        if (idx < MAX_CHAINS_TO_FIND) {
            __global char *output_ptr = found_chains + idx * MAX_CHAIN_STRING_LEN;
            
            // Store chain string
            for (int i = 0; i < chain_pos && i < MAX_CHAIN_STRING_LEN - 1; i++) {
                output_ptr[i] = chain_buffer[i];
            }
            output_ptr[chain_pos] = '\\0';
        }
    }
}
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
    
    # Calculate statistics
    avg_len = sum(len(w) for w in words_list) / len(words_list) if words_list else 0
    max_len = max(len(w) for w in words_list) if words_list else 0
    
    print(f"{green('[OK]')} {bold('Loaded:')} {cyan(f'{len(words_list):,}')} {bold('words')}")
    print(f"{blue('[INFO]')} {bold('Average length:')} {cyan(f'{avg_len:.1f}')}")
    print(f"{blue('[INFO]')} {bold('Max length:')} {cyan(f'{max_len}')}")
    
    return words_list

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"{bold('GPU-COMPATIBLE Hashcat Rules Engine with Dynamic Workload Processing')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('base_wordlist', help='Base wordlist path')
    parser.add_argument('target_wordlist', help='Target wordlist path')
    parser.add_argument('-d', '--depth', type=int, default=3, 
                       choices=[1, 2, 3],
                       help='Max chain depth (1-3 only for speed, default: 3)')
    parser.add_argument('-o', '--output', type=str, default='found_chains.txt', 
                       help='Output file (default: found_chains.txt)')
    parser.add_argument('--slow', action='store_true', 
                       help='Enable more comprehensive search (may take longer)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify all chains work before saving')
    parser.add_argument('--max-chains', type=int, default=None,
                       help='Maximum chains to generate (overrides automatic limits)')
    parser.add_argument('--target-hours', type=float, default=0.5,
                       help='Target completion time in hours (default: 0.5)')
    
    args = parser.parse_args()

    print(f"\n{bold(green('=' * 80))}")
    print(f"{bold('GPU-COMPATIBLE HASHCAT RULES ENGINE (DYNAMIC WORKLOAD)')}")
    print(f"{bold(green('=' * 80))}{Colors.END}\n")
    
    # Load data first
    print(f"{blue('[INIT]')} {bold('Loading data...')}")
    
    base_words = load_wordlist_fast(args.base_wordlist)
    target_words = load_wordlist_fast(args.target_wordlist)
    
    print(f"\n{blue('[ANALYSIS]')} {bold('Dataset Analysis:')}")
    print(f"  {cyan('[*]')} Base words: {len(base_words):,}")
    print(f"  {cyan('[*]')} Target words: {len(target_words):,}")
    print(f"  {cyan('[*]')} Processing ALL words")
    print(f"  {cyan('[*]')} Target completion: {args.target_hours} hours")
    
    start_time = time.time()
    
    # Get GPU device for parameter calculation
    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices(cl.device_type.GPU)[0]
    except:
        device = None
    
    # Initialize GPU extractor with time target
    extractor = GPUExtractor(len(base_words), len(target_words), device, args.slow, args.target_hours)
    
    # Override max chains if specified
    if args.max_chains:
        extractor.params['MAX_CHAINS_TO_FIND'] = args.max_chains
        print(f"{blue('[OVERRIDE]')} {bold('Max chains set to:')} {cyan(args.max_chains)}")
    
    # Run extraction
    print(f"\n{blue('=' * 60)}")
    print(f"{bold('STARTING GPU-COMPATIBLE RULE EXTRACTION')}")
    print(f"{blue('=' * 60)}")
    
    chains = extractor.extract_rules(base_words, target_words, args.depth)
    
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    
    # Save results
    print(f"\n{blue('[SAVE]')} {bold('Saving results...')}")
    
    # Validate for GPU one more time
    final_chains = HashcatRuleValidator.validate_rules_for_gpu(chains)
    
    with open(args.output, 'w', encoding='latin-1') as f:
        for chain in final_chains:
            f.write(f"{chain}\n")
    
    # Print summary
    print(f"\n{bold(green('=' * 80))}")
    print(f"{bold('FINAL RESULTS')}")
    print(f"{bold(green('=' * 80))}")
    print(f"{blue('[INFO]')} {bold('Base words:')} {cyan(f'{len(base_words):,}')}")
    print(f"{blue('[INFO]')} {bold('Target words:')} {cyan(f'{len(target_words):,}')}")
    print(f"{blue('[INFO]')} {bold('Max depth:')} {cyan(f'{args.depth}')}")
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

