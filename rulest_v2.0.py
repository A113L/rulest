import os
import sys
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
from collections import defaultdict, deque
from tqdm import tqdm
import mmap
import time
import hashlib
import signal
import math
import zlib 
import random

# Suppress compiler warnings
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

# ====================================================================
# --- OPTIMIZED CONSTANTS ---
# ====================================================================
MAX_WORD_LEN = 32         
MAX_OUTPUT_LEN = 48       
MAX_CHAINS_TO_FIND = 2000000 
MAX_CHAIN_STRING_LEN = 64  # Increased for longer chains
BLOOM_FALSE_POSITIVE_RATE = 0.001

# Performance optimizations
LOCAL_WORK_SIZE = 256  # Much better GPU utilization
MAX_WORK_ITEMS_PER_LAUNCH = 50_000_000  # 50M work items max
WORDS_PER_BATCH = 1000  # Small word batches for memory efficiency

# Global placeholders
BLOOM_HASH_FUNCTIONS = 0 
BLOOM_FILTER_SIZE_BYTES = 0

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
# --- COMPREHENSIVE HASHCAT RULES IMPLEMENTATION ---
# ====================================================================

def generate_comprehensive_hashcat_rules(slow_mode=False):
    """Generate ALL Hashcat rules from the original implementation"""
    rules = []
    
    if slow_mode:
        print(f"{yellow('🔧')} {bold('Generating COMPREHENSIVE Hashcat rules (SLOW MODE)...')}")
    else:
        print(f"{green('🔧')} {bold('Generating optimized Hashcat rules...')}")
    
    # ===== CORE/BASIC RULES (Always included) =====
    basic_rules = [
        'l',      # Lowercase all
        'u',      # Uppercase all  
        'c',      # Capitalize (first upper, rest lower)
        'C',      # Invert capitalize (first lower, rest upper)
        't',      # Toggle case
        'r',      # Reverse
        'k',      # Duplicate word (password -> passwordpassword)
        ':',      # Duplicate word (same as 'k')
        'd0',     # Duplicate word (same as 'k' and ':')
        'f',      # Reflect (password -> passworddrowssap)
        '[',      # Delete first character
        ']',      # Delete last character
        '{',      # Rotate left (password -> asswordp)
        '}',      # Rotate right (password -> dpasswor)
        'p',      # Append duplicated last character (password -> passwordd)
        'z',      # Prepend duplicated first character (password -> ppassword)
        'q',      # Append duplicated first character (password -> passwordp)
        'x',      # Extract memory - needs parameters
        'y',      # Insert character - needs parameters
        'L',      # Delete all lowercase letters
        'R',      # Delete all uppercase letters
        '@',      # Purge all occurrences of character - needs parameter
        'X',      # Extract memory - needs parameters
        "'",      # Truncate at position - needs parameter
    ]
    rules.extend(basic_rules)
    
    # ===== POSITION-BASED RULES =====
    for i in range(16):  # Positions 0-15 (hex)
        pos_hex = hex(i)[2:]
        # Toggle case at position
        rules.append(f'T{pos_hex}')
        # Delete at position  
        rules.append(f'D{pos_hex}')
        # Insert character at position (needs char parameter)
        rules.append(f'I{pos_hex}')
        # Overwrite at position (needs char parameter)
        rules.append(f'O{pos_hex}')
        # Extract memory range (needs length parameter)
        rules.append(f'M{pos_hex}')
    
    # ===== PREFIX/SUFFIX/AT RULES =====
    common_chars = string.digits + "!@#$%^&*()_-+=[]{};:,.<>/?~`" + string.ascii_letters[0:15]
    for char in common_chars:
        if char not in [' ', '\n', '\t']:
            # Prefix character
            rules.append(f'^{char}')
            # Suffix character  
            rules.append(f'${char}')
            # Insert at position (needs position)
            rules.append(f'@{char}')
    
    # ===== INSERT/OVERWRITE/EXTRACT RULES =====
    if slow_mode:
        # SLOW MODE: Include all possible combinations
        insert_chars = string.digits + "!@#$%^&*()_-+=" + string.ascii_letters
        for pos in range(16):  # Positions 0-15
            pos_hex = hex(pos)[2:]
            for char in insert_chars[:20]:  # Limit to first 20 chars even in slow mode
                # Insert at position
                rules.append(f'i{pos_hex}{char}')
                # Overwrite at position
                rules.append(f'o{pos_hex}{char}')
                # Extract substring
                rules.append(f'y{pos_hex}{char}')
    else:
        # FAST MODE: Limited combinations
        insert_chars = "!@#$%^&*()_+-=1234567890"
        for pos in range(8):  # Only first 8 positions
            pos_hex = hex(pos)[2:]
            for char in insert_chars[:12]:  # Only first 12 chars
                rules.append(f'i{pos_hex}{char}')
                rules.append(f'o{pos_hex}{char}')
    
    # ===== SUBSTITUTION (LEETSPEAK) RULES =====
    leet_subs = ['@', '4', '3', '1', '0', '5', '$', '7', '+', '-', '*', '#', '(', ')', '!', '?', '8', '9']
    for orig_char in string.ascii_lowercase:
        for sub_char in leet_subs[:8] if not slow_mode else leet_subs:  # Limit in fast mode
            rules.append(f's{orig_char}{sub_char}')
    
    # ===== PURGE RULES =====
    if slow_mode:
        # SLOW MODE: Include purge rules for all alphanumeric
        purge_chars = string.ascii_letters + string.digits
        for char in purge_chars:
            rules.append(f'P{char}')
    else:
        # FAST MODE: Only common purge characters
        purge_chars = "aeiouAEIOU0123456789!@#$%"
        for char in purge_chars:
            rules.append(f'P{char}')
    
    # Remove duplicates and filter invalid rules
    rules = list(set(rules))
    
    # Filter out rules that require additional parameters we can't provide
    valid_rules = []
    for rule in rules:
        if len(rule) == 1:
            valid_rules.append(rule)
        elif len(rule) == 2:
            # Rules like T0, D1, etc. are valid
            if rule[0] in ['T', 'D', 'M'] and rule[1] in '0123456789abcdef':
                valid_rules.append(rule)
        elif len(rule) == 3:
            # Rules like i0a, sae, etc. are valid
            if rule[0] in ['i', 'o', 's', 'y']:
                valid_rules.append(rule)
        else:
            # For now, skip longer rules that need special handling
            if rule.startswith(('^', '$', '@', 'P')):
                valid_rules.append(rule)
    
    if slow_mode:
        print(f"{yellow('✅')} {bold('Generated COMPREHENSIVE rules:')} {cyan(f'{len(valid_rules):,}')} {bold('rules (SLOW MODE)')}")
    else:
        print(f"{green('✅')} {bold('Generated optimized rules:')} {cyan(f'{len(valid_rules):,}')} {bold('rules')}")
    
    return valid_rules

# ====================================================================
# --- FIXED RULE CHAIN GENERATION (PROPER CHAIN COUNTS) ---
# ====================================================================

def generate_rule_chains_flattened(rules, max_depth, max_chains_per_depth=None, slow_mode=False):
    """Pre-compute all rule chains - ELIMINATES NESTED LOOPS"""
    print(f"{blue('🔧')} {bold('Generating flattened rule chains...')}")
    
    all_chains = []
    all_chain_strings = []
    
    # REASONABLE LIMITS FOR BOTH MODES
    if slow_mode:
        # SLOW MODE: More comprehensive coverage
        depth_limits = {1: None, 2: 100000, 3: 200000}  # Higher limits for slow mode
    else:
        # FAST MODE: Still good coverage but faster
        depth_limits = {1: None, 2: 50000, 3: 100000}   
    
    for depth in range(1, max_depth + 1):
        print(f"  {blue('-')} Depth {depth}: generating chains...")
        
        max_combinations = depth_limits.get(depth)
        
        if depth == 1:
            # For depth 1, use ALL rules in both modes
            chains_at_depth = [(r,) for r in rules]
            print(f"  {green('✓')} Depth {depth}: {len(chains_at_depth):,} chains (all rules)")
            
        else:
            chains_at_depth = []
            
            if depth == 2:
                # For depth 2: use systematic sampling
                if slow_mode:
                    # SLOW MODE: More comprehensive sampling for depth 2
                    sample_size1 = min(400, len(rules))  # Sample 400 rules for first position
                    sample_size2 = min(400, len(rules))  # Sample 400 rules for second position
                    
                    # Create representative samples
                    indices1 = list(range(len(rules)))
                    indices2 = list(range(len(rules)))
                    
                    # If we have too many rules, sample strategically
                    if len(rules) > sample_size1:
                        step1 = max(1, len(rules) // sample_size1)
                        indices1 = indices1[::step1][:sample_size1]
                    
                    if len(rules) > sample_size2:
                        step2 = max(1, len(rules) // sample_size2)
                        indices2 = indices2[::step2][:sample_size2]
                    
                    # Generate combinations
                    for i in indices1:
                        for j in indices2:
                            if max_combinations and len(chains_at_depth) >= max_combinations:
                                break
                            chains_at_depth.append((rules[i], rules[j]))
                        if max_combinations and len(chains_at_depth) >= max_combinations:
                            break
                            
                else:
                    # FAST MODE: Smaller but still good sampling for depth 2
                    sample_size1 = min(200, len(rules))
                    sample_size2 = min(200, len(rules))
                    
                    indices1 = list(range(len(rules)))
                    indices2 = list(range(len(rules)))
                    
                    if len(rules) > sample_size1:
                        step1 = max(1, len(rules) // sample_size1)
                        indices1 = indices1[::step1][:sample_size1]
                    
                    if len(rules) > sample_size2:
                        step2 = max(1, len(rules) // sample_size2)
                        indices2 = indices2[::step2][:sample_size2]
                    
                    for i in indices1:
                        for j in indices2:
                            if max_combinations and len(chains_at_depth) >= max_combinations:
                                break
                            chains_at_depth.append((rules[i], rules[j]))
                        if max_combinations and len(chains_at_depth) >= max_combinations:
                            break
            
            else:  # depth >= 3
                # For depth 3+: use random sampling
                target_size = max_combinations if max_combinations else 50000
                
                # Generate random chains
                for _ in range(target_size):
                    chain = tuple(random.choices(rules, k=depth))
                    chains_at_depth.append(chain)
                    
                    if max_combinations and len(chains_at_depth) >= max_combinations:
                        break
            
            print(f"  {green('✓')} Depth {depth}: {len(chains_at_depth):,} chains")
        
        # Add chains to final list
        for chain in chains_at_depth:
            all_chains.append(chain)
            all_chain_strings.append(' '.join(chain))
    
    print(f"{green('✅')} {bold('Generated')} {cyan(f'{len(all_chains):,}')} {bold('total rule chains')}")
    return all_chains, all_chain_strings

# ====================================================================
# --- OPTIMIZED FILE LOADING ---
# ====================================================================

def estimate_word_count(path):
    """Fast word count estimation"""
    try:
        file_size = os.path.getsize(path)
        sample_size = min(10 * 1024 * 1024, file_size)
        
        with open(path, 'rb') as f:
            sample = f.read(sample_size)
            lines = sample.count(b'\n')
            
            if file_size <= sample_size:
                return lines
            else:
                avg_line_length = sample_size / max(lines, 1)
                return int(file_size / avg_line_length)
    except:
        return 1000000

def load_wordlist_fast(filename):
    """Fast wordlist loading"""
    words = set()
    print(f"{blue('📊')} {bold('Loading:')} {filename}")
    
    total_estimated = estimate_word_count(filename)
    
    try:
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line_bytes in tqdm(iter(mm.readline, b""), total=total_estimated, 
                                      desc="Loading words", leave=False):
                    line = line_bytes.strip()
                    if line and len(line) <= MAX_WORD_LEN:
                        try:
                            words.add(line.decode('latin-1', errors='ignore'))
                        except:
                            pass
    except FileNotFoundError:
        print(f"{red('❌')} {bold('FATAL ERROR:')} Wordlist not found: {filename}")
        sys.exit(1)

    words_list = list(words)
    print(f"{green('✅')} {bold('Loaded:')} {cyan(f'{len(words_list):,}')} {bold('words')}")
    return words_list

# ====================================================================
# --- OPTIMIZED BLOOM FILTER ---
# ====================================================================

def calculate_bloom_size(num_elements):
    """Calculate optimal Bloom Filter size"""
    if num_elements == 0:
        return 0, 0
    
    m = - (num_elements * math.log(BLOOM_FALSE_POSITIVE_RATE)) / (math.log(2) ** 2)
    k = math.ceil((m / num_elements) * math.log(2))
    k = max(2, min(k, 12)) 
    m = int(math.ceil(m))
    bloom_size_bytes = math.ceil(m / 8)

    global BLOOM_HASH_FUNCTIONS
    BLOOM_HASH_FUNCTIONS = k
    
    return bloom_size_bytes, k

def simple_hash_py(key_bytes, seed):
    """FNV-1a hash matching OpenCL kernel"""
    fnv_prime = 16777619 
    hash_val = seed
    for byte in key_bytes:
        hash_val ^= byte
        hash_val = (hash_val * fnv_prime) & 0xFFFFFFFF
    return hash_val

def generate_bloom_filter(target_words):
    """Generate Bloom Filter"""
    n = len(target_words)
    if n == 0:
        return np.array([0], dtype=np.uint8)
    
    global BLOOM_FILTER_SIZE_BYTES, BLOOM_HASH_FUNCTIONS
    m_bytes, k = calculate_bloom_size(n)
    BLOOM_FILTER_SIZE_BYTES = m_bytes
    
    bloom_filter = np.zeros(m_bytes, dtype=np.uint8)
    m_bits = m_bytes * 8

    print(f"{blue('🔧')} {bold('Generating Bloom Filter...')}")
    for word in tqdm(target_words, desc="Hashing targets", leave=False):
        word_bytes = word.encode('latin-1', errors='ignore')
        h1 = simple_hash_py(word_bytes, 0xDEADBEEF)
        h2 = simple_hash_py(word_bytes, 0xCAFEBABE)
        
        for i in range(k):
            index = (h1 + i * h2)
            bit_index_full = index % m_bits
            byte_index = bit_index_full // 8
            bit_shift = bit_index_full % 8
            bloom_filter[byte_index] |= (1 << bit_shift)

    return bloom_filter.tobytes()

# ====================================================================
# --- RULE MANAGEMENT ---
# ====================================================================

def load_rules_from_file(filename, slow_mode=False):
    """Load rules from file with proper error handling"""
    rules = []
    print(f"{blue('📊')} {bold('Loading rules from:')} {filename}")
    
    if not os.path.exists(filename):
        print(f"{red('❌')} {bold('ERROR:')} Rule file not found: {filename}")
        return []
    
    try:
        with open(filename, 'r', encoding='latin-1', errors='ignore') as f:
            for line in tqdm(f, desc="Reading rules", leave=False):
                rule = line.strip()
                # Skip empty lines, comments, and lines with spaces
                if rule and not rule.startswith('#') and ' ' not in rule: 
                    rules.append(rule)
        
        rules = list(set(rules))  # Remove duplicates
        
        # Filter valid rules
        valid_rules = []
        for rule in rules:
            if len(rule) == 1:
                valid_rules.append(rule)
            elif len(rule) == 2:
                if rule[0] in ['T', 'D', 'M'] and rule[1] in '0123456789abcdef':
                    valid_rules.append(rule)
            elif len(rule) == 3:
                if rule[0] in ['i', 'o', 's', 'y']:
                    valid_rules.append(rule)
            else:
                if rule.startswith(('^', '$', '@', 'P')):
                    valid_rules.append(rule)
        
        print(f"{green('✅')} {bold('Loaded:')} {cyan(f'{len(valid_rules):,}')} {bold('rules from file')}")
        
        if slow_mode and len(valid_rules) > 1000:
            print(f"{yellow('⚠️')} {bold('Large rule file detected in SLOW MODE - this may take a while!')}")
        
        return valid_rules
        
    except Exception as e:
        print(f"{red('❌')} {bold('ERROR loading rules:')} {e}")
        return []

# ====================================================================
# --- COMPREHENSIVE OPENCL KERNEL (ALL HASHCAT RULES) ---
# ====================================================================

RULE_CHAINING_KERNEL_COMPREHENSIVE = """
#define MAX_WORD_LEN 32
#define MAX_OUTPUT_LEN 48
#define MAX_CHAIN_STRING_LEN 64
#define NULL_BYTE 0
#define MAX_CHAINS_TO_FIND_LIMIT //MAX_CHAINS_PLACEHOLDER//

// Placeholders
#define BLOOM_HASH_FUNCTIONS_K //K_PLACEHOLDER// 
#define BLOOM_FILTER_SIZE_BITS (//M_BYTES_PLACEHOLDER// * 8) 

// Optimized hash function
uint simple_hash(const char *key, int len, uint seed) {
    const uint fnv_prime = 16777619U;
    uint hash = seed;
    #pragma unroll 4
    for (int i = 0; i < len; i++) {
        hash ^= (uint)key[i];
        hash *= fnv_prime;
    }
    return hash;
}

// Pre-compute bloom hashes once
inline void precompute_bloom_hashes(const char *word, int len, __private uint *h1, __private uint *h2) {
    *h1 = simple_hash(word, len, 0xDEADBEEFU);
    *h2 = simple_hash(word, len, 0xCAFEBABEU);
}

// Fast bloom check with early exit
bool bloom_check_fast(__global const uchar *filter, uint h1, uint h2) {
    #pragma unroll 4
    for (int i = 0; i < BLOOM_HASH_FUNCTIONS_K; i++) {
        uint bit_pos = (h1 + i * h2) % BLOOM_FILTER_SIZE_BITS;
        if (!(filter[bit_pos >> 3] & (1 << (bit_pos & 7)))) {
            return false;
        }
    }
    return true;
}

// Helper functions
char tolower_ocl(char c) {
    return (c >= 'A' && c <= 'Z') ? (c | 0x20) : c;
}

char toupper_ocl(char c) {
    return (c >= 'a' && c <= 'z') ? (c & ~0x20) : c;
}

int get_pos(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1; 
}

// Fast word copy
int copy_word(const char *src, char *dst, int max_len) {
    int len = 0;
    for (len = 0; len < max_len; len++) {
        dst[len] = src[len];
        if (src[len] == NULL_BYTE) break;
    }
    if (len == max_len) dst[max_len - 1] = NULL_BYTE;
    return len;
}

// COMPREHENSIVE RULE APPLICATION (ALL HASHCAT RULES)
int apply_rule(__global const char *rule_start, const char *input_word, int input_len, char *output_word) {
    int new_len = copy_word(input_word, output_word, MAX_OUTPUT_LEN);
    if (new_len == 0) return 0;
    
    char rule_op = rule_start[0];
    char p1 = rule_start[1];
    char p2 = rule_start[2];
    
    // COMPREHENSIVE RULE SWITCH - ALL HASHCAT RULES
    switch (rule_op) {
        case 'l': { // Lowercase all - BITWISE OPTIMIZED
            #pragma unroll 8
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                output_word[k] = (c >= 'A' && c <= 'Z') ? (c | 0x20) : c;
            }
            break;
        }
        case 'u': { // Uppercase all - BITWISE OPTIMIZED
            #pragma unroll 8
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                output_word[k] = (c >= 'a' && c <= 'z') ? (c & ~0x20) : c;
            }
            break;
        }
        case 'c': { // Capitalize (first upper, rest lower)
            if (new_len > 0) output_word[0] = toupper_ocl(output_word[0]);
            for (int k = 1; k < new_len; k++) output_word[k] = tolower_ocl(output_word[k]);
            break;
        }
        case 'C': { // Invert capitalize (first lower, rest upper)
            if (new_len > 0) output_word[0] = tolower_ocl(output_word[0]);
            for (int k = 1; k < new_len; k++) output_word[k] = toupper_ocl(output_word[k]);
            break;
        }
        case 't': { // Toggle case
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                if (c >= 'a' && c <= 'z') output_word[k] = c - 32;
                else if (c >= 'A' && c <= 'Z') output_word[k] = c + 32;
            }
            break;
        }
        case 'r': { // Reverse
            for (int k = 0; k < new_len / 2; k++) {
                char temp = output_word[k];
                output_word[k] = output_word[new_len - 1 - k];
                output_word[new_len - 1 - k] = temp;
            }
            break;
        }
        case 'k':   // Duplicate word
        case ':':   // Duplicate word
        case 'd': { // Duplicate word (d0)
            if (rule_op == 'd' && p1 != '0') return 0;
            if (new_len * 2 < MAX_WORD_LEN) {
                for (int k = 0; k < new_len; k++) output_word[new_len + k] = output_word[k];
                new_len *= 2;
            }
            break;
        }
        case 'f': { // Reflect (password -> passworddrowssap)
            if (new_len * 2 - 1 < MAX_WORD_LEN && new_len > 0) {
                for (int k = 0; k < new_len - 1; k++) output_word[new_len + k] = output_word[new_len - 2 - k];
                new_len = new_len * 2 - 1;
            }
            break;
        }
        case '[': { // Delete first character
            if (new_len > 0) {
                for(int k = 0; k < new_len; k++) output_word[k] = output_word[k+1];
                new_len--;
            }
            break;
        }
        case ']': { // Delete last character
            if (new_len > 0) { new_len--; }
            break;
        }
        case '{': { // Rotate left (password -> asswordp)
            if (new_len > 1) {
                char first = output_word[0];
                for (int k = 0; k < new_len - 1; k++) output_word[k] = output_word[k+1];
                output_word[new_len - 1] = first;
            }
            break;
        }
        case '}': { // Rotate right (password -> dpasswor)
            if (new_len > 1) {
                char last = output_word[new_len - 1];
                for (int k = new_len - 1; k > 0; k--) output_word[k] = output_word[k-1];
                output_word[0] = last;
            }
            break;
        }
        case 'p': { // Append duplicated last character (password -> passwordd)
            if (new_len > 0 && new_len < MAX_WORD_LEN) {
                output_word[new_len] = output_word[new_len - 1];
                new_len++;
            }
            break;
        }
        case 'z': { // Prepend duplicated first character (password -> ppassword)
            if (new_len > 0 && new_len < MAX_WORD_LEN) {
                for(int k = new_len; k > 0; k--) output_word[k] = output_word[k-1];
                output_word[0] = output_word[1]; 
                new_len++;
            }
            break;
        }
        case 'q': { // Append duplicated first character (password -> passwordp)
            if (new_len > 0 && new_len < MAX_WORD_LEN) {
                output_word[new_len] = output_word[0];
                new_len++;
            }
            break;
        }
        case 'L': { // Delete all lowercase letters
            int j = 0;
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                if (c < 'a' || c > 'z') { output_word[j] = c; j++; }
            }
            new_len = j;
            break;
        }
        case 'R': { // Delete all uppercase letters
            int j = 0;
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                if (c < 'A' || c > 'Z') { output_word[j] = c; j++; }
            }
            new_len = j;
            break;
        }
        case '$': { // Append character
            if (p1 != NULL_BYTE && new_len < MAX_WORD_LEN) {
                output_word[new_len] = p1; 
                new_len++; 
            }
            break;
        }
        case '^': { // Prepend character
            if (p1 != NULL_BYTE && new_len < MAX_WORD_LEN) {
                for(int k = new_len; k > 0; k--) output_word[k] = output_word[k-1];
                output_word[0] = p1; 
                new_len++; 
            }
            break;
        }
        case 'i': { // Insert character at position
            if (p2 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos <= new_len && new_len < MAX_WORD_LEN) {
                    for(int k = new_len; k > pos; k--) output_word[k] = output_word[k-1];
                    output_word[pos] = p2; 
                    new_len++;
                }
            }
            break;
        }
        case 'o': { // Overwrite character at position
            if (p2 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) { 
                    output_word[pos] = p2; 
                }
            }
            break;
        }
        case 's': { // Substitute character
            if (p2 != NULL_BYTE) {
                char orig = p1;
                char sub = p2;
                for(int k = 0; k < new_len; k++) { 
                    if (output_word[k] == orig) output_word[k] = sub; 
                }
            }
            break;
        }
        case 'T': { // Toggle case at position
            if (p1 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) {
                    char c = output_word[pos];
                    if (c >= 'a' && c <= 'z') output_word[pos] = c - 32;
                    else if (c >= 'A' && c <= 'Z') output_word[pos] = c + 32;
                }
            }
            break;
        }
        case 'D': { // Delete at position
            if (p1 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) {
                    for(int k = pos; k < new_len; k++) output_word[k] = output_word[k+1];
                    new_len--;
                }
            }
            break;
        }
        case '@':   // Purge all occurrences of character
        case 'P': { // Purge all occurrences of character
            if (p1 != NULL_BYTE) {
                char purge_char = p1;
                int j = 0;
                for (int k = 0; k < new_len; k++) {
                    if (output_word[k] != purge_char) { 
                        output_word[j] = output_word[k]; 
                        j++; 
                    }
                }
                new_len = j;
            }
            break;
        }
        case '\\'': { // Truncate at position
            if (p1 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) { 
                    new_len = pos; 
                }
            }
            break;
        }
        case 'x':   // Extract substring
        case 'X':   // Extract substring
        case 'M': { // Extract memory range
            if (p2 != NULL_BYTE) {
                int pos = get_pos(p1);
                int len_to_extract = get_pos(p2);
                if (pos >= 0 && len_to_extract > 0 && pos < new_len) {
                    int end = min(pos + len_to_extract, new_len);
                    int extracted_len = end - pos;
                    for (int k = 0; k < extracted_len; k++) output_word[k] = output_word[pos + k];
                    new_len = extracted_len;
                }
            }
            break;
        }
        case 'y': { // Insert character (alternative syntax)
            if (p2 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos <= new_len && new_len < MAX_WORD_LEN) {
                    for(int k = new_len; k > pos; k--) output_word[k] = output_word[k-1];
                    output_word[pos] = p2; 
                    new_len++;
                }
            }
            break;
        }
        default:
            return 0; // Unknown rule
    }
    
    if (new_len >= MAX_OUTPUT_LEN) new_len = MAX_OUTPUT_LEN - 1;
    output_word[new_len] = NULL_BYTE;
    return new_len;
}

// MAIN KERNEL - FLATTENED APPROACH (NO NESTED LOOPS!)
__kernel void rule_chaining_kernel_flattened(
    // Input Data
    __global const char *g_base_words_ptr,
    __global const int *g_word_offsets,
    __global const char *g_rule_chains_ptr,      // Pre-built chains
    __global const int *g_chain_offsets,
    __global const int *g_chain_lengths,         // Rules per chain
    __global const uchar *g_bloom_filter,
    
    // Configuration
    const int num_base_words,
    const int num_chains,
    
    // Output
    __global char *g_output_buffer,
    __global int *g_output_count 
) {
    int gid = get_global_id(0);
    
    // FLATTENED INDEXING - NO NESTED LOOPS!
    int base_word_idx = gid / num_chains;
    int chain_idx = gid % num_chains;
    
    if (base_word_idx >= num_base_words) return;
    
    // Get base word
    __global const char *base_word_start = g_base_words_ptr + g_word_offsets[base_word_idx];
    char current_word[MAX_OUTPUT_LEN];
    int current_len = copy_word(base_word_start, current_word, MAX_WORD_LEN);
    if (current_len == 0) return;
    
    // Get rule chain
    __global const char *chain_start = g_rule_chains_ptr + g_chain_offsets[chain_idx];
    int chain_length = g_chain_lengths[chain_idx];
    
    // Apply rules sequentially (NO NESTED LOOPS!)
    __global const char *current_rule = chain_start;
    for (int rule_idx = 0; rule_idx < chain_length; rule_idx++) {
        current_len = apply_rule(current_rule, current_word, current_len, current_word);
        if (current_len == 0) return; // Rule failed
        
        // Move to next rule in chain
        while (*current_rule != NULL_BYTE) current_rule++;
        current_rule++; // Skip null terminator
    }
    
    // Single bloom filter check at the end
    uint h1, h2;
    precompute_bloom_hashes(current_word, current_len, &h1, &h2);
    
    if (bloom_check_fast(g_bloom_filter, h1, h2)) {
        // RECORD CHAIN
        int output_index = atomic_inc((__global volatile int *)g_output_count);
        
        if (output_index < MAX_CHAINS_TO_FIND_LIMIT) {
            __global char *output_ptr = g_output_buffer + output_index * MAX_CHAIN_STRING_LEN;
            
            // Copy chain string
            __global const char *chain_str = g_rule_chains_ptr + g_chain_offsets[chain_idx];
            copy_word(chain_str, output_ptr, MAX_CHAIN_STRING_LEN);
        }
    }
}
"""

# ====================================================================
# --- OPTIMIZED GPU EXECUTION WITH TWO-LEVEL BATCHING ---
# ====================================================================

def fast_gpu_rule_chaining_flattened(base_words, target_words, rules, max_depth, slow_mode=False):
    """
    Optimized GPU execution with flattened approach and two-level batching
    """
    if not rules or not base_words: 
        return []
    
    # 1. Generate flattened rule chains (CRITICAL OPTIMIZATION)
    chains, chain_strings = generate_rule_chains_flattened(
        rules, max_depth, max_chains_per_depth=None, slow_mode=slow_mode
    )
    
    if not chains:
        print(f"{red('❌')} {bold('No rule chains generated')}")
        return []
    
    # 2. Prepare chain data for GPU
    print(f"{blue('🔧')} {bold('Preparing chain data for GPU...')}")
    
    # Pack chains into single buffer
    chains_bytes = b'\0'.join([chain_str.encode('latin-1') for chain_str in chain_strings]) + b'\0'
    chain_offsets = np.array([0] + [len(c) + 1 for c in chain_strings], dtype=np.int32).cumsum()[:-1]
    chain_lengths = np.array([len(chain) for chain in chains], dtype=np.int32)
    
    # 3. Generate Bloom Filter
    bloom_filter_bytes = generate_bloom_filter(target_words)
    bloom_filter_np = np.frombuffer(bloom_filter_bytes, dtype=np.uint8)
    
    # 4. Setup OpenCL
    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices(cl.device_type.GPU)[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        print(f"{green('🎮')} {bold('GPU:')} {cyan(device.name.strip())}")
        
    except Exception as e:
        print(f"{red('❌')} {bold('OpenCL init failed:')} {e}")
        return []
    
    # 5. Compile kernel
    kernel_source = RULE_CHAINING_KERNEL_COMPREHENSIVE
    kernel_source = kernel_source.replace("//K_PLACEHOLDER//", str(BLOOM_HASH_FUNCTIONS))
    kernel_source = kernel_source.replace("//M_BYTES_PLACEHOLDER//", str(BLOOM_FILTER_SIZE_BYTES))
    kernel_source = kernel_source.replace("//MAX_CHAINS_PLACEHOLDER//", str(MAX_CHAINS_TO_FIND))
    
    try:
        program = cl.Program(context, kernel_source).build()
        kernel = program.rule_chaining_kernel_flattened
    except Exception as e:
        print(f"{red('❌')} {bold('Kernel compilation failed:')} {e}")
        return []
    
    mf = cl.mem_flags
    
    # 6. Create static buffers
    chains_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chains_bytes)
    chain_offsets_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chain_offsets)
    chain_lengths_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chain_lengths)
    bloom_filter_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bloom_filter_np)
    
    output_buffer_bytes = MAX_CHAINS_TO_FIND * MAX_CHAIN_STRING_LEN
    output_buffer_g = cl.Buffer(context, mf.WRITE_ONLY, output_buffer_bytes)
    
    all_chains = set()
    total_chains_found = 0
    
    print(f"{blue('🚀')} {bold('Starting GPU processing...')}")
    start_time = time.time()
    
    # 7. TWO-LEVEL BATCHING (CRITICAL FOR MEMORY)
    num_chains = len(chains)
    
    # Adjust batch sizes based on slow mode
    if slow_mode:
        words_per_batch = 500  # Smaller batches for slow mode (more rules)
        chains_per_batch = MAX_WORK_ITEMS_PER_LAUNCH // words_per_batch
    else:
        words_per_batch = WORDS_PER_BATCH
        chains_per_batch = MAX_WORK_ITEMS_PER_LAUNCH // words_per_batch
    
    # Single progress bar for entire process
    total_work_items = len(base_words) * num_chains
    with tqdm(total=total_work_items, desc="GPU Processing", unit="items") as pbar:
        
        for word_start in range(0, len(base_words), words_per_batch):
            word_end = min(word_start + words_per_batch, len(base_words))
            word_batch = base_words[word_start:word_end]
            num_words_batch = len(word_batch)
            
            # Prepare word batch
            words_bytes = b'\0'.join([w.encode('latin-1', errors='ignore') for w in word_batch]) + b'\0'
            word_offsets = np.array([0] + [len(w) + 1 for w in word_batch], dtype=np.int32).cumsum()[:-1]
            
            words_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=words_bytes)
            word_offsets_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=word_offsets)
            
            for chain_start in range(0, num_chains, chains_per_batch):
                chain_end = min(chain_start + chains_per_batch, num_chains)
                num_chains_batch = chain_end - chain_start
                
                # Calculate work size
                global_size = num_words_batch * num_chains_batch
                if global_size == 0:
                    continue
                
                # Align to work group size
                global_size = ((global_size + LOCAL_WORK_SIZE - 1) // LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE
                
                # Output counter
                output_count = np.array([0], dtype=np.int32)
                output_count_g = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=output_count)
                
                # Set kernel arguments
                kernel.set_args(
                    words_g, word_offsets_g,
                    chains_g, chain_offsets_g, chain_lengths_g,
                    bloom_filter_g,
                    np.int32(num_words_batch),
                    np.int32(num_chains_batch),
                    output_buffer_g,
                    output_count_g
                )
                
                # Launch kernel
                cl.enqueue_nd_range_kernel(
                    queue, kernel, (global_size,), (LOCAL_WORK_SIZE,)
                ).wait()
                
                # Read results
                cl.enqueue_copy(queue, output_count, output_count_g).wait()
                chains_found = output_count[0]
                
                if chains_found > 0:
                    read_count = min(chains_found, MAX_CHAINS_TO_FIND)
                    read_size = read_count * MAX_CHAIN_STRING_LEN
                    
                    output_data = np.empty(read_size, dtype=np.byte)
                    cl.enqueue_copy(queue, output_data, output_buffer_g).wait()
                    
                    for j in range(read_count):
                        start = j * MAX_CHAIN_STRING_LEN
                        chain_bytes = output_data[start:start + MAX_CHAIN_STRING_LEN]
                        
                        try:
                            null_pos = chain_bytes.tolist().index(0)
                            chain_str = chain_bytes[:null_pos].tobytes().decode('latin-1').strip()
                        except:
                            chain_str = chain_bytes.tobytes().decode('latin-1', errors='ignore').strip('\x00').strip()
                        
                        if chain_str:
                            all_chains.add(chain_str)
                    
                    total_chains_found += chains_found
                
                # Update progress
                pbar.update(num_words_batch * num_chains_batch)
                
                # Cleanup
                output_count_g.release()
            
            # Cleanup word buffers
            words_g.release()
            word_offsets_g.release()
    
    # Cleanup static buffers
    chains_g.release()
    chain_offsets_g.release()
    chain_lengths_g.release()
    bloom_filter_g.release()
    output_buffer_g.release()
    
    end_time = time.time()
    
    mode_str = "SLOW MODE" if slow_mode else "FAST MODE"
    print(f"{green('✅')} {bold('GPU processing complete:')} {cyan(f'{len(all_chains):,}')} {bold('chains found in')} {cyan(f'{end_time - start_time:.2f}s')} {bold(f'({mode_str})')}")
    return list(all_chains)

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"{bold('🚀 GPU Rule Chaining (All Hashcat Rules + External Rule Files)')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('base_wordlist', help='Base wordlist path')
    parser.add_argument('target_wordlist', help='Target wordlist path')
    parser.add_argument('--rules', type=str, default=None, 
                       help='Path to external rule file (Hashcat .rule format)')
    parser.add_argument('-d', '--depth', type=int, default=2, 
                       help='Max chain depth (1-3, default: 2)')
    parser.add_argument('-o', '--output', type=str, default='found_chains.txt', 
                       help='Output file (default: found_chains.txt)')
    parser.add_argument('--slow', action='store_true', 
                       help='Enable SLOW mode with comprehensive rule coverage')
    
    args = parser.parse_args()

    print(f"\n{bold(green('=' * 70))}")
    print(f"{bold('🚀 GPU RULE CHAINING (ALL HASHCAT RULES + EXTERNAL RULE FILES)')}")
    print(f"{bold(green('=' * 70))}{Colors.END}\n")
    
    if args.slow:
        print(f"{yellow('⚠️')} {bold('SLOW MODE ENABLED - Comprehensive rule coverage')}")
        print(f"{yellow('⚠️')} {bold('Expect significantly longer processing times!')}")
    
    if args.rules:
        print(f"{blue('📁')} {bold('Using external rule file:')} {cyan(args.rules)}")
    
    print()
    
    # Load data
    base_words = load_wordlist_fast(args.base_wordlist)
    target_words = load_wordlist_fast(args.target_wordlist)
    
    if not base_words or not target_words:
        sys.exit(1)
    
    # Load/generate rules
    if args.rules:
        rules = load_rules_from_file(args.rules, slow_mode=args.slow)
        if not rules:
            print(f"{red('❌')} {bold('No valid rules found in file, falling back to generated rules')}")
            rules = generate_comprehensive_hashcat_rules(slow_mode=args.slow)
    else:
        print(f"{blue('🔧')} {bold('No external rule file specified, generating rules...')}")
        rules = generate_comprehensive_hashcat_rules(slow_mode=args.slow)
    
    if not rules:
        print(f"{red('❌')} {bold('No rules available. Exiting.')}")
        sys.exit(1)
    
    # Find chains
    chains = fast_gpu_rule_chaining_flattened(
        base_words, target_words, rules, min(args.depth, 3), slow_mode=args.slow
    )
    
    # Save results
    with open(args.output, 'w') as f:
        for chain in chains:
            f.write(f"{chain}\n")
    
    print(f"\n{bold(green('=' * 70))}")
    print(f"{bold('✨ FINAL RESULTS')}")
    print(f"{bold(green('=' * 70))}")
    print(f"{blue('📊')} {bold('Base words:')} {cyan(f'{len(base_words):,}')}")
    print(f"{blue('📊')} {bold('Target words:')} {cyan(f'{len(target_words):,}')}")
    print(f"{blue('🔧')} {bold('Rules:')} {cyan(f'{len(rules):,}')}")
    print(f"{blue('📏')} {bold('Max depth:')} {cyan(f'{min(args.depth, 3)}')}")
    print(f"{blue('⚡')} {bold('Mode:')} {cyan('SLOW' if args.slow else 'FAST')}")
    if args.rules:
        print(f"{blue('📁')} {bold('Rule source:')} {cyan('EXTERNAL FILE')}")
    else:
        print(f"{blue('📁')} {bold('Rule source:')} {cyan('GENERATED')}")
    print(f"{green('✅')} {bold('Chains found:')} {cyan(f'{len(chains):,}')}")
    print(f"{blue('💾')} {bold('Output:')} {bold(args.output)}")
    print(f"{bold(green('=' * 70))}{Colors.END}")
    print(f"{green('✅')} {bold('Chains found:')} {cyan(f'{len(chains):,}')}")
    print(f"{blue('💾')} {bold('Output:')} {bold(args.output)}")
    print(f"{bold(green('=' * 70))}{Colors.END}")
