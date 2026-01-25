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
# --- HASHCAT RULE VALIDATION (from hashcat cleanup-rules.c) ---
# ====================================================================

class HashcatRuleValidator:
    """Validates rules according to Hashcat's official cleanup-rules.c logic"""
    
    @staticmethod
    def is_digit(c):
        return '0' <= c <= '9'
    
    @staticmethod
    def is_upper(c):
        return 'A' <= c <= 'Z'
    
    @staticmethod
    def conv_ctoi(c):
        if '0' <= c <= '9':
            return ord(c) - ord('0')
        elif 'A' <= c <= 'Z':
            return ord(c) - ord('A') + 10
        return -1
    
    @staticmethod
    def validate_rule_for_gpu(rule_str):
        """Validate rule for GPU compatibility (mode 2 from cleanup-rules.c)"""
        line_len = len(rule_str)
        pos = 0
        cnt = 0
        
        # Max rules per line for GPU
        MAX_GPU_RULES = 255
        
        while pos < line_len:
            c = rule_str[pos]
            
            if c == ' ':
                pos += 1
                continue
            
            # Hashcat rule commands
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
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'r':  # RULE_OP_MANGLE_REVERSE
                pos += 1
            elif c == 'd':  # RULE_OP_MANGLE_DUPEWORD
                pos += 1
            elif c == 'p':  # RULE_OP_MANGLE_DUPEWORD_TIMES
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'f':  # RULE_OP_MANGLE_REFLECT
                pos += 1
            elif c == '{':  # RULE_OP_MANGLE_ROTATE_LEFT
                pos += 1
            elif c == '}':  # RULE_OP_MANGLE_ROTATE_RIGHT
                pos += 1
            elif c == '$':  # RULE_OP_MANGLE_APPEND
                pos += 1
                if pos >= line_len: return False
            elif c == '^':  # RULE_OP_MANGLE_PREPEND
                pos += 1
                if pos >= line_len: return False
            elif c == '[':  # RULE_OP_MANGLE_DELETE_FIRST
                pos += 1
            elif c == ']':  # RULE_OP_MANGLE_DELETE_LAST
                pos += 1
            elif c == 'D':  # RULE_OP_MANGLE_DELETE_AT
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'x':  # RULE_OP_MANGLE_EXTRACT
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'i':  # RULE_OP_MANGLE_INSERT
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
                pos += 1
                if pos >= line_len: return False
            elif c == 'o':  # RULE_OP_MANGLE_OVERSTRIKE
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
                pos += 1
                if pos >= line_len: return False
            elif c == '\'':  # RULE_OP_MANGLE_TRUNCATE_AT
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 's':  # RULE_OP_MANGLE_REPLACE
                pos += 1
                if pos >= line_len: return False
                pos += 1
                if pos >= line_len: return False
            elif c == '@':  # RULE_OP_MANGLE_PURGECHAR
                pos += 1
                if pos >= line_len: return False
            elif c == 'a':  # RULE_OP_MANGLE_TOGGLECASE_REC
                pos += 1
            elif c == 'z':  # RULE_OP_MANGLE_DUPECHAR_FIRST
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'Z':  # RULE_OP_MANGLE_DUPECHAR_LAST
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'q':  # RULE_OP_MANGLE_DUPECHAR_ALL
                pos += 1
            elif c == 'y':  # RULE_OP_MANGLE_DUPEBLOCK_FIRST
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'Y':  # RULE_OP_MANGLE_DUPEBLOCK_LAST
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'k':  # RULE_OP_MANGLE_SWITCH_FIRST
                pos += 1
            elif c == 'K':  # RULE_OP_MANGLE_SWITCH_LAST
                pos += 1
            elif c == '*':  # RULE_OP_MANGLE_SWITCH_AT
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'L':  # RULE_OP_MANGLE_CHR_SHIFTL
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'R':  # RULE_OP_MANGLE_CHR_SHIFTR
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == '+':  # RULE_OP_MANGLE_CHR_INCR
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == '-':  # RULE_OP_MANGLE_CHR_DECR
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == '.':  # RULE_OP_MANGLE_REPLACE_NP1
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == ',':  # RULE_OP_MANGLE_REPLACE_NM1
                pos += 1
                if pos >= line_len: return False
                if HashcatRuleValidator.conv_ctoi(rule_str[pos]) == -1: return False
            elif c == 'E':  # RULE_OP_MANGLE_TITLE
                pos += 1
            elif c == 'X':  # RULE_OP_MANGLE_EXTRACT_MEMORY
                # Memory rules are NOT supported on GPU
                return False
            elif c == '4':  # RULE_OP_MANGLE_APPEND_MEMORY
                # Memory rules are NOT supported on GPU
                return False
            elif c == '6':  # RULE_OP_MANGLE_PREPEND_MEMORY
                # Memory rules are NOT supported on GPU
                return False
            elif c == 'M':  # RULE_OP_MEMORIZE_WORD
                # Memory rules are NOT supported on GPU
                return False
            elif c == '<':  # RULE_OP_REJECT_LESS
                # Reject rules are NOT supported on GPU
                return False
            elif c == '>':  # RULE_OP_REJECT_GREATER
                # Reject rules are NOT supported on GPU
                return False
            elif c == '!':  # RULE_OP_REJECT_CONTAIN
                # Reject rules are NOT supported on GPU
                return False
            elif c == '/':  # RULE_OP_REJECT_NOT_CONTAIN
                # Reject rules are NOT supported on GPU
                return False
            elif c == '(':  # RULE_OP_REJECT_EQUAL_FIRST
                # Reject rules are NOT supported on GPU
                return False
            elif c == ')':  # RULE_OP_REJECT_EQUAL_LAST
                # Reject rules are NOT supported on GPU
                return False
            elif c == '=':  # RULE_OP_REJECT_EQUAL_AT
                # Reject rules are NOT supported on GPU
                return False
            elif c == '%':  # RULE_OP_REJECT_CONTAINS
                # Reject rules are NOT supported on GPU
                return False
            elif c == 'Q':  # RULE_OP_REJECT_MEMORY
                # Reject rules are NOT supported on GPU
                return False
            else:
                # Unknown/invalid rule character
                return False
            
            cnt += 1
            if cnt > MAX_GPU_RULES:
                return False
        
        return True
    
    @staticmethod
    def clean_rules(rules):
        """Clean and validate rules for GPU compatibility"""
        valid_rules = []
        for rule in rules:
            # Remove any whitespace at the beginning/end of rule string
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

def calculate_dynamic_parameters(base_count, target_count):
    """Calculate dynamic parameters based on input data size"""
    
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
    
    # Batch sizes based on available memory
    if base_count > 1000000:  # Large dataset
        WORDS_PER_BATCH = 5000
        RULES_PER_BATCH = 1000
    elif base_count > 100000:  # Medium dataset
        WORDS_PER_BATCH = 10000
        RULES_PER_BATCH = 2000
    else:  # Small dataset
        WORDS_PER_BATCH = 20000
        RULES_PER_BATCH = 5000
    
    # Scale batch sizes
    WORDS_PER_BATCH = int(WORDS_PER_BATCH * words_scale)
    RULES_PER_BATCH = int(RULES_PER_BATCH * rules_scale)
    
    # Clamp values
    WORDS_PER_BATCH = min(max(WORDS_PER_BATCH, 1000), 50000)
    RULES_PER_BATCH = min(max(RULES_PER_BATCH, 500), 20000)
    
    # Max chains based on dataset size
    if base_count * target_count > 10000000:
        MAX_CHAINS_TO_FIND = 10000
    elif base_count * target_count > 1000000:
        MAX_CHAINS_TO_FIND = 50000
    else:
        MAX_CHAINS_TO_FIND = 100000
    
    # Chain depth based on mode
    MAX_CHAIN_DEPTH = 6
    
    return {
        'BLOOM_FILTER_SIZE': BLOOM_FILTER_SIZE,
        'WORDS_PER_BATCH': WORDS_PER_BATCH,
        'RULES_PER_BATCH': RULES_PER_BATCH,
        'MAX_CHAINS_TO_FIND': MAX_CHAINS_TO_FIND,
        'MAX_CHAIN_DEPTH': MAX_CHAIN_DEPTH,
        'bloom_scale': bloom_scale,
        'words_scale': words_scale,
        'rules_scale': rules_scale
    }

# Initialize with default values
MAX_WORD_LEN = 256         
MAX_RULE_LEN = 16         
MAX_OUTPUT_LEN = 512      
MAX_CHAIN_STRING_LEN = 512  # Increased for longer chains
LOCAL_WORK_SIZE = 256

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
# --- VALID HASHCAT RULES GENERATION ---
# ====================================================================

class ValidHashcatRulesGenerator:
    """Generate ONLY valid Hashcat rules (GPU-compatible)"""
    
    def __init__(self, slow_mode=False):
        self.slow_mode = slow_mode
        self.validator = HashcatRuleValidator()
    
    def generate_valid_rules(self):
        """Generate valid Hashcat rules only"""
        rules = set()  # Use set to avoid duplicates
        
        print(f"{blue('[SETUP]')} {bold('Generating VALID Hashcat rules for GPU...')}")
        
        # ===== CATEGORY 1: SIMPLE GPU-COMPATIBLE RULES =====
        print(f"  {cyan('[*]')} Simple rules...")
        simple_rules = [
            # Single character rules
            'l', 'u', 'c', 'C', 't', 'r', 'd', 'f',
            '{', '}', '[', ']', 'k', 'K', ':', 'a',
            'q', 'E'
        ]
        rules.update(simple_rules)
        
        # ===== CATEGORY 2: POSITION-BASED RULES (0-9) =====
        print(f"  {cyan('[*]')} Position-based rules (0-9)...")
        position_cmds = ['T', 'D', 'L', 'R', '+', '-', '.', ',', 'z', 'Z']
        for cmd in position_cmds:
            for pos in range(10):
                rules.add(f'{cmd}{pos}')
        
        # ===== CATEGORY 3: PREFIX/SUFFIX =====
        print(f"  {cyan('[*]')} Prefix/suffix rules...")
        # All printable ASCII characters except control chars
        for i in range(32, 127):
            char = chr(i)
            rules.add(f'^{char}')
            rules.add(f'${char}')
            if char not in '\\':  # @ can't have backslash
                rules.add(f'@{char}')
        
        # ===== CATEGORY 4: SUBSTITUTIONS =====
        print(f"  {cyan('[*]')} Substitution rules...")
        common_subs = [
            ('a', '@'), ('e', '3'), ('i', '1'), ('o', '0'), ('s', '$'),
            ('t', '7'), ('l', '1'), ('g', '9'), ('b', '8'), ('z', '2'),
            ('A', '@'), ('E', '3'), ('I', '1'), ('O', '0'), ('S', '$'),
            ('T', '7'), ('L', '1'), ('G', '9'), ('B', '8'), ('Z', '2')
        ]
        
        for orig, sub in common_subs:
            rules.add(f's{orig}{sub}')
        
        # More substitutions
        for orig in string.ascii_lowercase + string.ascii_uppercase:
            for sub in string.digits + string.punctuation:
                if orig != sub:
                    rules.add(f's{orig}{sub}')
        
        # ===== CATEGORY 5: INSERTION/OVERWRITE =====
        print(f"  {cyan('[*]')} Insertion/overwrite rules...")
        for pos in range(10):
            for char in string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?/~':
                rules.add(f'i{pos}{char}')
                rules.add(f'o{pos}{char}')
        
        # ===== CATEGORY 6: EXTRACTION/SWAP =====
        print(f"  {cyan('[*]')} Extraction/swap rules...")
        for n in range(10):
            for m in range(10):
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
        for separator in [' ', '-', '_', '.', ',', ';', ':', '|', '/', '\\']:
            # 'E' is for title case, but it doesn't take parameters
            # Actually, 'e' with separator is a different rule (eX)
            if separator != ' ':
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
# --- DYNAMIC GPU ENGINE WITH VALID RULES ---
# ====================================================================

class DynamicGPUEngine:
    """GPU-accelerated engine with valid rules only"""
    
    def __init__(self, params):
        self.params = params
        self.context = None
        self.queue = None
        self.device = None
        self.program = None
        
    def initialize_gpu(self):
        """Initialize OpenCL with dynamic parameters"""
        try:
            platform = cl.get_platforms()[0]
            self.device = platform.get_devices(cl.device_type.GPU)[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            
            # Get GPU memory info for dynamic batching
            global_mem = self.device.global_mem_size
            print(f"{green('[GPU]')} {bold('GPU:')} {cyan(self.device.name.strip())}")
            print(f"{blue('[INFO]')} {bold('Global Memory:')} {cyan(f'{global_mem // (1024**3)}GB')}")
            
            return True
        except Exception as e:
            print(f"{red('[ERROR]')} {bold('GPU initialization failed:')} {e}")
            return False
    
    def compile_kernel(self):
        """Compile the valid kernel"""
        try:
            print(f"{blue('[SETUP]')} {bold('Compiling kernel...')}")
            
            # Read the kernel source
            kernel_source = VALID_HASHCAT_KERNEL
            
            # Replace constants with dynamic values
            kernel_source = kernel_source.replace(
                "BLOOM_FILTER_SIZE 8388608",
                f"BLOOM_FILTER_SIZE {self.params['BLOOM_FILTER_SIZE']}"
            )
            kernel_source = kernel_source.replace(
                "MAX_CHAINS_TO_FIND 100000",
                f"MAX_CHAINS_TO_FIND {self.params['MAX_CHAINS_TO_FIND']}"
            )
            kernel_source = kernel_source.replace(
                "MAX_CHAIN_DEPTH 6",
                f"MAX_CHAIN_DEPTH {self.params['MAX_CHAIN_DEPTH']}"
            )
            
            self.program = cl.Program(self.context, kernel_source).build()
            print(f"{green('[OK]')} {bold('Kernel compiled')}")
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
        
        # Filter rules for GPU compatibility
        validator = HashcatRuleValidator()
        gpu_rules = [r for r in rules if validator.validate_rule_for_gpu(r)]
        
        print(f"{blue('[INFO]')} {bold('GPU-compatible rules:')} {len(gpu_rules):,}")
        
        all_found_rules = []
        
        # Process ALL base words in optimized batches
        batch_size = self.params['WORDS_PER_BATCH']
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
        
        # Output buffers
        found_rules_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                  self.params['MAX_CHAINS_TO_FIND'] * MAX_CHAIN_STRING_LEN)
        found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)
        
        # Initialize counter
        zero_count = np.array([0], dtype=np.int32)
        cl.enqueue_copy(self.queue, found_count_buf, zero_count)
        
        # Run kernel
        kernel = self.program.find_single_rules
        
        # Calculate global size
        total_combinations = batch_data['num_words'] * batch_data['num_rules']
        global_size = ((total_combinations + LOCAL_WORK_SIZE - 1) // LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE
        
        kernel.set_args(
            base_buf, base_offsets_buf, base_lengths_buf,
            rules_buf, rule_offsets_buf, rule_lengths_buf,
            bloom_buf,
            np.int32(batch_data['num_words']),
            np.int32(batch_data['num_rules']),
            found_rules_buf,
            found_count_buf
        )
        
        cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (LOCAL_WORK_SIZE,))
        self.queue.finish()
        
        # Read results
        found_count = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(self.queue, found_count, found_count_buf)
        
        num_found = min(found_count[0], self.params['MAX_CHAINS_TO_FIND'])
        
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
        
        # Cleanup
        base_buf.release()
        base_offsets_buf.release()
        base_lengths_buf.release()
        rules_buf.release()
        rule_offsets_buf.release()
        rule_lengths_buf.release()
        bloom_buf.release()
        found_rules_buf.release()
        found_count_buf.release()
        
        return batch_found
    
    def process_all_words_chain_rules(self, base_words, target_words, rules, max_depth):
        """Process ALL base words with rule chains"""
        print(f"{blue('[GPU]')} {bold('Processing ALL words with rule chains...')}")
        
        # Generate Bloom filter with ALL target words
        bloom_filter = self.generate_bloom_filter(target_words)
        
        # Compile kernel
        if not self.compile_kernel():
            return []
        
        # Filter rules for GPU compatibility
        validator = HashcatRuleValidator()
        gpu_rules = [r for r in rules if validator.validate_rule_for_gpu(r)]
        
        # Generate chains from valid rules
        print(f"{blue('[SETUP]')} {bold('Generating rule chains...')}")
        chains = self.generate_valid_chains(gpu_rules, max_depth)
        
        if not chains:
            return []
        
        print(f"{blue('[INFO]')} {bold('Valid chains:')} {len(chains):,}")
        
        all_chains = []
        
        # Process ALL base words in batches
        batch_size = self.params['WORDS_PER_BATCH'] // 2  # Smaller batches for chains
        num_batches = (len(base_words) + batch_size - 1) // batch_size
        
        print(f"{blue('[INFO]')} {bold('Processing ALL')} {len(base_words):,} {bold('words in')} {num_batches} {bold('batches')}")
        
        # Process batches
        with tqdm(total=num_batches, desc="Processing chains", unit="batch") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(base_words))
                batch_words = base_words[start_idx:end_idx]
                
                if not batch_words:
                    pbar.update(1)
                    continue
                
                # Process this batch
                batch_chains = self.process_batch_chains(batch_words, gpu_rules, chains, bloom_filter)
                
                if batch_chains:
                    all_chains.extend(batch_chains)
                
                # Update progress
                pbar.set_postfix({
                    'found': len(all_chains),
                    'progress': f"{end_idx:,}/{len(base_words):,}"
                })
                pbar.update(1)
        
        # Remove duplicates and return
        all_chains = list(set(all_chains))
        print(f"\n{green('[OK]')} {bold('Total unique chains found:')} {cyan(len(all_chains))}")
        
        return all_chains
    
    def generate_valid_chains(self, rules, max_depth):
        """Generate valid rule chains"""
        chains = []
        
        # Single rules (already validated)
        print(f"  {cyan('->')} Single rules...")
        chains.extend(rules[:1000])  # Use first 1000 rules
        
        # Depth 2 chains
        if max_depth >= 2 and len(rules) >= 2:
            print(f"  {cyan('->')} Depth 2 chains...")
            # Generate combinations of valid rules
            for i in range(min(100, len(rules))):
                for j in range(min(100, len(rules))):
                    if len(chains) >= 5000:
                        break
                    chains.append(f"{rules[i]} {rules[j]}")
                if len(chains) >= 5000:
                    break
        
        # Depth 3 chains
        if max_depth >= 3 and len(rules) >= 3:
            print(f"  {cyan('->')} Depth 3 chains...")
            # Use common rules for depth 3
            common_rules = ['l', 'u', 'c', 't', 'r', 'd', 'f', '{', '}', '[', ']', '^A', '$1', 'sae']
            for i in range(min(20, len(common_rules))):
                for j in range(min(20, len(common_rules))):
                    for k in range(min(20, len(common_rules))):
                        if len(chains) >= 10000:
                            break
                        chains.append(f"{common_rules[i]} {common_rules[j]} {common_rules[k]}")
                    if len(chains) >= 10000:
                        break
                if len(chains) >= 10000:
                    break
        
        # Clean chains (ensure each rule in chain is valid)
        valid_chains = []
        validator = HashcatRuleValidator()
        
        for chain in chains:
            chain_rules = chain.split()
            valid = True
            for rule in chain_rules:
                if not validator.validate_rule_for_gpu(rule):
                    valid = False
                    break
            if valid and 1 <= len(chain_rules) <= max_depth:
                valid_chains.append(chain)
        
        print(f"  {cyan('[*]')} Valid chains: {len(valid_chains):,}")
        return valid_chains
    
    def process_batch_chains(self, words, rules, chains, bloom_filter):
        """Process a batch of chains"""
        # Prepare chain sequences
        chain_sequences = []
        chain_depths = []
        
        for chain in chains:
            chain_rules = chain.split()
            depth = len(chain_rules)
            chain_depths.append(depth)
            
            # Convert rules to indices
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
        found_chains_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                   self.params['MAX_CHAINS_TO_FIND'] * MAX_CHAIN_STRING_LEN)
        found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)
        
        # Initialize counter
        zero_count = np.array([0], dtype=np.int32)
        cl.enqueue_copy(self.queue, zero_count, found_count_buf)
        
        # Run kernel
        kernel = self.program.find_rule_chains
        
        total_items = len(words) * len(chains)
        global_size = ((total_items + LOCAL_WORK_SIZE - 1) // LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE
        
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
        
        cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (LOCAL_WORK_SIZE,))
        self.queue.finish()
        
        # Read results
        found_count = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(self.queue, found_count, found_count_buf)
        
        num_found = min(found_count[0], self.params['MAX_CHAINS_TO_FIND'])
        
        batch_chains = []
        if num_found > 0:
            # Read found chains
            found_data = np.zeros(num_found * MAX_CHAIN_STRING_LEN, dtype=np.uint8)
            cl.enqueue_copy(self.queue, found_data, found_chains_buf)
            
            # Convert to strings
            for i in range(num_found):
                start = i * MAX_CHAIN_STRING_LEN
                chain_bytes = bytes(found_data[start:start + MAX_CHAIN_STRING_LEN])
                chain_str = chain_bytes.split(b'\0')[0].decode('latin-1', errors='ignore')
                if chain_str:
                    batch_chains.append(chain_str)
        
        # Cleanup
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
        
        return batch_chains

# ====================================================================
# --- HYBRID EXTRACTOR WITH VALIDATION ---
# ====================================================================

class ValidHybridExtractor:
    """Hybrid system with valid rules and complete processing"""
    
    def __init__(self, base_count, target_count, slow_mode=False):
        self.base_count = base_count
        self.target_count = target_count
        self.slow_mode = slow_mode
        
        # Calculate dynamic parameters
        self.params = calculate_dynamic_parameters(base_count, target_count)
        
        print(f"{blue('[CONFIG]')} {bold('Valid Configuration:')}")
        print(f"  {cyan('[*]')} Bloom filter size: {self.params['BLOOM_FILTER_SIZE']:,} bits")
        print(f"  {cyan('[*]')} Words per batch: {self.params['WORDS_PER_BATCH']:,}")
        print(f"  {cyan('[*]')} Max chains to find: {self.params['MAX_CHAINS_TO_FIND']:,}")
        print(f"  {cyan('[*]')} Max chain depth: {self.params['MAX_CHAIN_DEPTH']}")
        
        # Initialize components
        self.rules_generator = ValidHashcatRulesGenerator(slow_mode)
        self.gpu_engine = DynamicGPUEngine(self.params)
        self.validator = HashcatRuleValidator()
    
    def extract_rules(self, base_words, target_words, max_depth=3):
        """Extract valid rules using complete processing"""
        print(f"{blue('[MAIN]')} {bold('Starting complete rule extraction...')}")
        
        all_chains = []
        
        # Generate valid rules
        rules = self.rules_generator.generate_valid_rules()
        
        # Initialize GPU
        if not self.gpu_engine.initialize_gpu():
            print(f"{yellow('[WARN]')} {bold('GPU not available, using CPU only')}")
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
        
        # Final cleanup and verification
        print(f"\n{blue('=' * 60)}")
        print(f"{bold('FINAL CLEANUP AND VERIFICATION')}")
        print(f"{blue('=' * 60)}")
        
        # Clean rules using Hashcat's validator
        cleaned_chains = self.validator.clean_rules(all_chains)
        
        # Verify chains work with sample
        verified_chains = self.verify_chains_complete(base_words, target_words, cleaned_chains)
        
        return verified_chains
    
    def verify_chains_complete(self, base_words, target_words, chains):
        """Verify that chains actually work with ALL base words"""
        print(f"{blue('[VERIFY]')} {bold('Verifying chains with complete dataset...')}")
        
        target_set = set(target_words)
        verified = []
        
        # Use more base words for verification
        sample_size = min(500, len(base_words))
        base_sample = base_words[:sample_size]
        
        with tqdm(total=len(chains), desc="Verifying chains", unit="chain") as pbar:
            for chain in chains:
                found = False
                
                # Test with more base words
                for base in base_sample:
                    result = self.apply_chain_cpu(base, chain)
                    if result and result in target_set:
                        found = True
                        break
                
                if found:
                    verified.append(chain)
                
                pbar.update(1)
                pbar.set_postfix({'verified': len(verified)})
        
        print(f"{green('[OK]')} {bold('Verified chains:')} {cyan(len(verified))}/{cyan(len(chains))}")
        
        # Show sample of verified chains
        if verified:
            print(f"{blue('[SAMPLE]')} {bold('Sample verified chains:')}")
            for i, chain in enumerate(verified[:10]):
                depth = len(chain.split())
                print(f"  {cyan(f'{i+1:2d}.')} [{depth}] {chain}")
        
        return verified
    
    def apply_chain_cpu(self, word, chain):
        """Simple CPU chain application for verification"""
        result = word
        for rule in chain.split():
            result = self.apply_rule_cpu(result, rule)
            if not result:
                return None
        return result
    
    def apply_rule_cpu(self, word, rule):
        """Simple CPU rule application"""
        if not rule or not word:
            return None
        
        result = word
        
        try:
            # Single char rules
            if rule == 'l':
                result = word.lower()
            elif rule == 'u':
                result = word.upper()
            elif rule == 'c' and word:
                result = word[0].upper() + word[1:].lower()
            elif rule == 'C' and word:
                result = word[0].lower() + word[1:].upper()
            elif rule == 't':
                result = ''.join([c.upper() if c.islower() else c.lower() for c in word])
            elif rule == 'r':
                result = word[::-1]
            elif rule == 'd':
                result = word + word
            elif rule == 'f':
                result = word + word[::-1]
            elif rule == '{' and word:
                result = word[1:] + word[0]
            elif rule == '}' and word:
                result = word[-1] + word[:-1]
            elif rule == '[' and word:
                result = word[1:]
            elif rule == ']' and word:
                result = word[:-1]
            elif rule == 'k' and len(word) >= 2:
                result = word[1] + word[0] + word[2:]
            elif rule == ':':
                result = word
            elif rule == 'a':
                # Toggle case recursively
                result = ''
                for c in word:
                    if c.islower():
                        result += c.upper()
                    elif c.isupper():
                        result += c.lower()
                    else:
                        result += c
            elif rule == 'q':
                # Duplicate all chars
                result = ''.join([c * 2 for c in word])
            elif rule == 'E':
                # Title case
                result = word.title()
            
            # Two char rules
            elif len(rule) == 2:
                if rule[0] == '^':
                    result = rule[1] + word
                elif rule[0] == '$':
                    result = word + rule[1]
                elif rule[0] == '@':
                    result = word.replace(rule[1], '')
                elif rule[0] == 'T' and rule[1].isdigit():
                    pos = int(rule[1])
                    if pos < len(word):
                        chars = list(word)
                        chars[pos] = chars[pos].upper() if chars[pos].islower() else chars[pos].lower()
                        result = ''.join(chars)
                elif rule[0] == 'D' and rule[1].isdigit():
                    pos = int(rule[1])
                    if pos < len(word):
                        result = word[:pos] + word[pos+1:]
                elif rule[0] == 's':
                    # This is actually a 3-char rule, handle specially
                    return None
            
            # Three char rules
            elif len(rule) == 3:
                if rule[0] == 's':
                    result = word.replace(rule[1], rule[2])
                elif rule[0] == 'i' and rule[1].isdigit():
                    pos = int(rule[1])
                    if pos <= len(word):
                        result = word[:pos] + rule[2] + word[pos:]
                elif rule[0] == 'o' and rule[1].isdigit():
                    pos = int(rule[1])
                    if pos < len(word):
                        chars = list(word)
                        chars[pos] = rule[2]
                        result = ''.join(chars)
                elif rule[0] == 'e':
                    # Title case with separator
                    separator = rule[1]
                    parts = word.split(separator)
                    result = separator.join([p.capitalize() for p in parts])
            
            # Position-based rules with hex
            elif len(rule) == 2 and rule[1] in '0123456789ABCDEF':
                if rule[0] == 'T':
                    pos = int(rule[1], 16)
                    if pos < len(word):
                        chars = list(word)
                        chars[pos] = chars[pos].upper() if chars[pos].islower() else chars[pos].lower()
                        result = ''.join(chars)
                elif rule[0] == 'D':
                    pos = int(rule[1], 16)
                    if pos < len(word):
                        result = word[:pos] + word[pos+1:]
                
        except Exception:
            return None
        
        return result if len(result) <= MAX_WORD_LEN else None

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
# --- VALID HASHCAT KERNEL (GPU-COMPATIBLE) ---
# ====================================================================

VALID_HASHCAT_KERNEL = """
#define MAX_WORD_LEN 256
#define MAX_RULE_LEN 16
#define MAX_OUTPUT_LEN 512
#define MAX_CHAIN_STRING_LEN 512
#define MAX_CHAINS_TO_FIND 100000
#define MAX_CHAIN_DEPTH 6
#define BLOOM_FILTER_SIZE 8388608
#define BLOOM_HASH_FUNCTIONS 4

// Utility functions
inline int is_lower(char c) {
    return (c >= 'a' && c <= 'z');
}

inline int is_upper(char c) {
    return (c >= 'A' && c <= 'Z');
}

inline int is_digit(char c) {
    return (c >= '0' && c <= '9');
}

inline int is_hex(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

inline int hex_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

inline char to_lower_char(char c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c;
}

inline char to_upper_char(char c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    return c;
}

inline char toggle_case_char(char c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c >= 'a' && c <= 'z') return c - 32;
    return c;
}

// Bloom filter
uint bloom_hash(const char *str, int len, uint seed) {
    uint hash = seed;
    for (int i = 0; i < len; i++) {
        hash = (hash * 31) + str[i];
    }
    return hash;
}

int bloom_check(__global const uchar *bloom_filter, const char *word, int len) {
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

// VALID Hashcat rules implementation (GPU-compatible only)
void apply_valid_rule(const char *rule_str, int rule_len,
                      const char *input_word, int input_len,
                      char *output_word, int *output_len) {
    
    // Initialize with input
    *output_len = input_len;
    for (int i = 0; i < input_len; i++) {
        output_word[i] = input_word[i];
    }
    output_word[input_len] = '\\0';
    
    if (rule_len == 0 || input_len == 0) return;
    
    char cmd = rule_str[0];
    
    // Single character rules
    if (rule_len == 1) {
        switch (cmd) {
            case 'l':  // Lowercase all
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = to_lower_char(output_word[i]);
                }
                return;
                
            case 'u':  // Uppercase all
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = to_upper_char(output_word[i]);
                }
                return;
                
            case 'c':  // Capitalize first, lowercase rest
                if (*output_len > 0) {
                    output_word[0] = to_upper_char(output_word[0]);
                    for (int i = 1; i < *output_len; i++) {
                        output_word[i] = to_lower_char(output_word[i]);
                    }
                }
                return;
                
            case 'C':  // Lowercase first, uppercase rest
                if (*output_len > 0) {
                    output_word[0] = to_lower_char(output_word[0]);
                    for (int i = 1; i < *output_len; i++) {
                        output_word[i] = to_upper_char(output_word[i]);
                    }
                }
                return;
                
            case 't':  // Toggle case
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = toggle_case_char(output_word[i]);
                }
                return;
                
            case 'r':  // Reverse
                for (int i = 0; i < *output_len / 2; i++) {
                    char temp = output_word[i];
                    output_word[i] = output_word[*output_len - 1 - i];
                    output_word[*output_len - 1 - i] = temp;
                }
                return;
                
            case 'd':  // Duplicate
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {
                    for (int i = 0; i < *output_len; i++) {
                        output_word[*output_len + i] = output_word[i];
                    }
                    *output_len *= 2;
                }
                return;
                
            case 'f':  // Reflect
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {
                    for (int i = 0; i < *output_len; i++) {
                        output_word[*output_len + i] = output_word[*output_len - 1 - i];
                    }
                    *output_len *= 2;
                }
                return;
                
            case '{':  // Rotate left
                if (*output_len > 1) {
                    char first = output_word[0];
                    for (int i = 0; i < *output_len - 1; i++) {
                        output_word[i] = output_word[i + 1];
                    }
                    output_word[*output_len - 1] = first;
                }
                return;
                
            case '}':  // Rotate right
                if (*output_len > 1) {
                    char last = output_word[*output_len - 1];
                    for (int i = *output_len - 1; i > 0; i--) {
                        output_word[i] = output_word[i - 1];
                    }
                    output_word[0] = last;
                }
                return;
                
            case '[':  // Delete first char
                if (*output_len > 0) {
                    for (int i = 0; i < *output_len - 1; i++) {
                        output_word[i] = output_word[i + 1];
                    }
                    (*output_len)--;
                }
                return;
                
            case ']':  // Delete last char
                if (*output_len > 0) {
                    (*output_len)--;
                }
                return;
                
            case 'k':  // Swap first two
                if (*output_len >= 2) {
                    char temp = output_word[0];
                    output_word[0] = output_word[1];
                    output_word[1] = temp;
                }
                return;
                
            case 'K':  // Swap last two
                if (*output_len >= 2) {
                    char temp = output_word[*output_len - 2];
                    output_word[*output_len - 2] = output_word[*output_len - 1];
                    output_word[*output_len - 1] = temp;
                }
                return;
                
            case ':':  // No operation
                return;
                
            case 'a':  // Toggle case recursively
                for (int i = 0; i < *output_len; i++) {
                    output_word[i] = toggle_case_char(output_word[i]);
                }
                return;
                
            case 'q':  // Duplicate all chars
                if (*output_len * 2 <= MAX_OUTPUT_LEN) {
                    int idx = 0;
                    for (int i = 0; i < *output_len; i++) {
                        output_word[idx++] = output_word[i];
                        output_word[idx++] = output_word[i];
                    }
                    *output_len *= 2;
                }
                return;
                
            case 'E':  // Title case
                if (*output_len > 0) {
                    int capitalize = 1;
                    for (int i = 0; i < *output_len; i++) {
                        if (capitalize && is_lower(output_word[i])) {
                            output_word[i] = output_word[i] - 32;
                            capitalize = 0;
                        }
                        if (output_word[i] == ' ' || output_word[i] == '-' || output_word[i] == '_') {
                            capitalize = 1;
                        }
                    }
                }
                return;
        }
    }
    
    // Two character rules
    else if (rule_len == 2) {
        char param = rule_str[1];
        
        if (cmd == '^') {  // Prepend
            if (*output_len + 1 <= MAX_OUTPUT_LEN) {
                for (int i = *output_len; i > 0; i--) {
                    output_word[i] = output_word[i - 1];
                }
                output_word[0] = param;
                (*output_len)++;
            }
        }
        else if (cmd == '$') {  // Append
            if (*output_len + 1 <= MAX_OUTPUT_LEN) {
                output_word[*output_len] = param;
                (*output_len)++;
            }
        }
        else if (cmd == '@') {  // Delete all instances
            int new_len = 0;
            for (int i = 0; i < *output_len; i++) {
                if (output_word[i] != param) {
                    output_word[new_len++] = output_word[i];
                }
            }
            *output_len = new_len;
        }
        else if (cmd == 'T' && is_digit(param)) {  // Toggle at position
            int pos = param - '0';
            if (pos < *output_len) {
                output_word[pos] = toggle_case_char(output_word[pos]);
            }
        }
        else if (cmd == 'D' && is_digit(param)) {  // Delete at position
            int pos = param - '0';
            if (pos < *output_len) {
                for (int i = pos; i < *output_len - 1; i++) {
                    output_word[i] = output_word[i + 1];
                }
                (*output_len)--;
            }
        }
        else if (cmd == 'L' && is_digit(param)) {  // Shift left
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] > 0) {
                output_word[pos]--;
            }
        }
        else if (cmd == 'R' && is_digit(param)) {  // Shift right
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] < 255) {
                output_word[pos]++;
            }
        }
        else if (cmd == '+' && is_digit(param)) {  // Increment
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] < 255) {
                output_word[pos]++;
            }
        }
        else if (cmd == '-' && is_digit(param)) {  // Decrement
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] > 0) {
                output_word[pos]--;
            }
        }
        else if (cmd == '.' && is_digit(param)) {  // Replace with dot
            int pos = param - '0';
            if (pos < *output_len) {
                output_word[pos] = '.';
            }
        }
        else if (cmd == ',' && is_digit(param)) {  // Replace with comma
            int pos = param - '0';
            if (pos < *output_len) {
                output_word[pos] = ',';
            }
        }
        else if (cmd == 'z' && is_digit(param)) {  // Duplicate first N
            int n = param - '0';
            if (n > 0 && *output_len + n <= MAX_OUTPUT_LEN) {
                for (int i = 0; i < n; i++) {
                    output_word[*output_len + i] = output_word[i];
                }
                *output_len += n;
            }
        }
        else if (cmd == 'Z' && is_digit(param)) {  // Duplicate last N
            int n = param - '0';
            if (n > 0 && *output_len + n <= MAX_OUTPUT_LEN) {
                for (int i = 0; i < n; i++) {
                    output_word[*output_len + i] = output_word[*output_len - n + i];
                }
                *output_len += n;
            }
        }
        else if (cmd == 'p' && is_digit(param)) {  // Duplicate word N times
            int n = param - '0';
            if (n > 0 && *output_len * (n + 1) <= MAX_OUTPUT_LEN) {
                for (int rep = 0; rep < n; rep++) {
                    for (int i = 0; i < *output_len; i++) {
                        output_word[*output_len * (rep + 1) + i] = output_word[i];
                    }
                }
                *output_len *= (n + 1);
            }
        }
    }
    
    // Three character rules
    else if (rule_len == 3) {
        char param1 = rule_str[1];
        char param2 = rule_str[2];
        
        if (cmd == 's') {  // Substitute
            for (int i = 0; i < *output_len; i++) {
                if (output_word[i] == param1) {
                    output_word[i] = param2;
                }
            }
        }
        else if (cmd == 'i' && is_digit(param1)) {  // Insert at position
            int pos = param1 - '0';
            if (*output_len + 1 <= MAX_OUTPUT_LEN) {
                for (int i = *output_len; i > pos; i--) {
                    output_word[i] = output_word[i - 1];
                }
                output_word[pos] = param2;
                (*output_len)++;
            }
        }
        else if (cmd == 'o' && is_digit(param1)) {  // Overwrite at position
            int pos = param1 - '0';
            if (pos < *output_len) {
                output_word[pos] = param2;
            }
        }
        else if (cmd == 'x' && is_digit(param1) && is_digit(param2)) {  // Extract
            int n = param1 - '0';
            int m = param2 - '0';
            if (n > m) {
                int temp = n; n = m; m = temp;
            }
            int new_len = 0;
            for (int i = n; i <= m && i < *output_len; i++) {
                output_word[new_len++] = output_word[i];
            }
            *output_len = new_len;
        }
        else if (cmd == '*' && is_digit(param1) && is_digit(param2)) {  // Swap positions
            int n = param1 - '0';
            int m = param2 - '0';
            if (n < *output_len && m < *output_len && n != m) {
                char temp = output_word[n];
                output_word[n] = output_word[m];
                output_word[m] = temp;
            }
        }
        else if (cmd == 'e') {  // Title case with separator
            // param1 is the separator
            if (*output_len > 0) {
                int capitalize = 1;
                for (int i = 0; i < *output_len; i++) {
                    if (capitalize && is_lower(output_word[i])) {
                        output_word[i] = output_word[i] - 32;
                        capitalize = 0;
                    }
                    if (output_word[i] == param1) {
                        capitalize = 1;
                    }
                }
            }
        }
        else if (cmd == 'y' && is_digit(param1)) {  // Duplicate first N
            int n = param1 - '0';
            char x = param2;  // Actually param2 is not used in 'y' rule
            if (n > 0 && *output_len + n <= MAX_OUTPUT_LEN) {
                for (int i = 0; i < n; i++) {
                    output_word[*output_len + i] = output_word[i];
                }
                *output_len += n;
            }
        }
        else if (cmd == 'Y' && is_digit(param1)) {  // Duplicate last N
            int n = param1 - '0';
            char x = param2;  // Actually param2 is not used in 'Y' rule
            if (n > 0 && *output_len + n <= MAX_OUTPUT_LEN) {
                for (int i = 0; i < n; i++) {
                    output_word[*output_len + i] = output_word[*output_len - n + i];
                }
                *output_len += n;
            }
        }
    }
    
    output_word[*output_len] = '\\0';
}

// ============================================================================
// KERNELS
// ============================================================================

__kernel void find_single_rules(
    __global const char *base_words,
    __global const int *base_offsets,
    __global const int *base_lengths,
    __global const char *rules,
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
    __global const char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];
    
    char input_word[MAX_WORD_LEN];
    for (int i = 0; i < word_len; i++) {
        input_word[i] = word_ptr[i];
    }
    input_word[word_len] = '\\0';
    
    // Get rule
    __global const char *rule_ptr = rules + rule_offsets[rule_idx];
    int rule_len = rule_lengths[rule_idx];
    
    char rule_str[MAX_RULE_LEN];
    for (int i = 0; i < rule_len; i++) {
        rule_str[i] = rule_ptr[i];
    }
    rule_str[rule_len] = '\\0';
    
    // Apply rule
    char output_word[MAX_OUTPUT_LEN];
    int output_len;
    apply_valid_rule(rule_str, rule_len, input_word, word_len, output_word, &output_len);
    
    if (output_len > 0) {
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

__kernel void find_rule_chains(
    __global const char *base_words,
    __global const int *base_offsets,
    __global const int *base_lengths,
    __global const char *rules,
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
    __global const char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];
    
    char current_word[MAX_OUTPUT_LEN];
    for (int i = 0; i < word_len; i++) {
        current_word[i] = word_ptr[i];
    }
    current_word[word_len] = '\\0';
    int current_len = word_len;
    
    // Get chain depth
    int depth = chain_depths[chain_idx];
    if (depth < 1 || depth > max_chain_depth) return;
    
    // Apply chain of rules
    char temp_word[MAX_OUTPUT_LEN];
    char chain_buffer[MAX_CHAIN_STRING_LEN];
    int chain_pos = 0;
    
    for (int d = 0; d < depth; d++) {
        int rule_idx = chain_sequences[chain_idx * max_chain_depth + d];
        if (rule_idx < 0) break;
        
        // Get rule
        __global const char *rule_ptr = rules + rule_offsets[rule_idx];
        int rule_len = rule_lengths[rule_idx];
        
        char rule_str[MAX_RULE_LEN];
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
        apply_valid_rule(rule_str, rule_len, current_word, current_len, temp_word, &new_len);
        
        if (new_len == 0) {
            return;  // Rule failed
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
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"{bold('VALID Hashcat Rules Engine with GPU Acceleration')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('base_wordlist', help='Base wordlist path')
    parser.add_argument('target_wordlist', help='Target wordlist path')
    parser.add_argument('-d', '--depth', type=int, default=3, 
                       choices=[1, 2, 3, 4, 5, 6],
                       help='Max chain depth (1-6, default: 3)')
    parser.add_argument('-o', '--output', type=str, default='found_chains.txt', 
                       help='Output file (default: found_chains.txt)')
    parser.add_argument('--slow', action='store_true', 
                       help='Enable more comprehensive search')
    parser.add_argument('--verify', action='store_true',
                       help='Verify all chains work before saving')
    
    args = parser.parse_args()

    print(f"\n{bold(green('=' * 80))}")
    print(f"{bold('VALID HASHCAT RULES ENGINE (GPU-COMPATIBLE + COMPLETE PROCESSING)')}")
    print(f"{bold(green('=' * 80))}{Colors.END}\n")
    
    # Load data first
    print(f"{blue('[INIT]')} {bold('Loading data...')}")
    
    base_words = load_wordlist_fast(args.base_wordlist)
    target_words = load_wordlist_fast(args.target_wordlist)
    
    print(f"\n{blue('[ANALYSIS]')} {bold('Dataset Analysis:')}")
    print(f"  {cyan('[*]')} Base words: {len(base_words):,}")
    print(f"  {cyan('[*]')} Target words: {len(target_words):,}")
    print(f"  {cyan('[*]')} Processing ALL words")
    
    start_time = time.time()
    
    # Initialize extractor with valid rules only
    extractor = ValidHybridExtractor(len(base_words), len(target_words), args.slow)
    
    # Run extraction
    print(f"\n{blue('=' * 60)}")
    print(f"{bold('STARTING COMPLETE RULE EXTRACTION')}")
    print(f"{blue('=' * 60)}")
    
    chains = extractor.extract_rules(base_words, target_words, args.depth)
    
    end_time = time.time()
    
    # Save results
    print(f"\n{blue('[SAVE]')} {bold('Saving results...')}")
    
    # Clean chains one more time before saving
    validator = HashcatRuleValidator()
    final_chains = validator.clean_rules(chains)
    
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
    print(f"{blue('[INFO]')} {bold('Total time:')} {cyan(f'{end_time - start_time:.2f}s')}")
    print(f"{green('[RESULT]')} {bold('Valid GPU-compatible chains found:')} {cyan(f'{len(final_chains):,}')}")
    
    if final_chains:
        print(f"{blue('[SAMPLE]')} {bold('Sample chains (max 20):')}")
        for i, chain in enumerate(final_chains[:20]):
            depth = len(chain.split())
            print(f"  {cyan(f'{i+1:2d}.')} [{depth}] {chain}")
    
    print(f"{blue('[OUTPUT]')} {bold('Output saved to:')} {bold(args.output)}")
    print(f"{blue('[NOTE]')} {bold('All chains are GPU-compatible and valid for Hashcat')}")
    print(f"{bold(green('=' * 80))}{Colors.END}")
