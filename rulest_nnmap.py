import os
import sys
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
from collections import defaultdict, deque
from tqdm import tqdm # <- Used for batch progress bar
import mmap
import time
import hashlib
import signal
import math
import zlib 

# Suppress compiler warnings
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

# ====================================================================
# --- CONFIGURABLE CONSTANTS ---
# ====================================================================
MAX_WORD_LEN = 32         
LOCAL_WORK_SIZE = 1       
MAX_OUTPUT_LEN = 48       
MAX_CHAINS_TO_FIND = 2000000 # Max chains to find *per batch*
MAX_CHAIN_STRING_LEN = 16 
BLOOM_FALSE_POSITIVE_RATE = 0.001
DEFAULT_BATCH_SIZE = 100000 # New default batch size

# Global placeholders for calculated Bloom Filter stats
BLOOM_HASH_FUNCTIONS = 0 
BLOOM_FILTER_SIZE_BYTES = 0
MAX_DEPTH = 3 

# --- COLOR CODES FOR TERMINAL OUTPUT ---
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
def bold(text): return f"{Colors.BOLD}{text}{Colors.END}"
# ====================================================================


def load_wordlist_fast(filename, max_words=None):
    """Fast wordlist loading using mmap."""
    words = set()
    print(f"Loading: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                file_content_size = len(mm)
                # Estimate line count for tqdm
                estimated_lines = max(1, file_content_size // 10) 
                
                mm.seek(0)
                
                # TQDM progress bar for wordlist loading
                for line_bytes in tqdm(iter(mm.readline, b""), total=estimated_lines, desc="Loading words"):
                    line = line_bytes.strip()
                    if line and len(line) <= MAX_WORD_LEN:
                        try:
                            words.add(line.decode('latin-1', errors='ignore'))
                        except UnicodeDecodeError:
                            pass

    except FileNotFoundError:
        print(f"{red('FATAL ERROR:')} Wordlist file not found: {filename}")
        sys.exit(1)

    words_list = list(words)
    print(f"Loaded: {len(words_list):,} words")
    return words_list

# ====================================================================
# --- BLOOM FILTER IMPLEMENTATION (FIXED HASHING) ---
# ====================================================================

def calculate_bloom_size(num_elements):
    """Calculate optimal Bloom Filter size (m) in bytes given n and false positive rate p."""
    if num_elements == 0:
        return 0, 0
    
    m = - (num_elements * math.log(BLOOM_FALSE_POSITIVE_RATE)) / (math.log(2) ** 2)
    k = math.ceil((m / num_elements) * math.log(2))
    
    k = max(2, min(k, 12)) 
    m = int(math.ceil(m))
    bloom_size_bytes = math.ceil(m / 8)

    global BLOOM_HASH_FUNCTIONS
    BLOOM_HASH_FUNCTIONS = k
    
    print(f"\nBloom Filter Stats (n={num_elements:,}, p={BLOOM_FALSE_POSITIVE_RATE}):")
    print(f"  Hash functions (k): {k}")
    print(f"  Filter size (bytes): {bloom_size_bytes:,}")
    
    return bloom_size_bytes, k

def simple_hash_py(key_bytes, seed):
    """FNV-1a-like hash function to match the OpenCL kernel (16777619U)."""
    fnv_prime = 16777619 
    hash_val = seed
    for byte in key_bytes:
        hash_val ^= byte
        hash_val = (hash_val * fnv_prime) & 0xFFFFFFFF
    return hash_val

def generate_bloom_filter(target_words):
    """Creates a numpy array representing the Bloom Filter with consistent hashing."""
    n = len(target_words)
    if n == 0:
        return np.array([0], dtype=np.uint8)
    
    global BLOOM_FILTER_SIZE_BYTES, BLOOM_HASH_FUNCTIONS
    
    m_bytes, k = calculate_bloom_size(n)
    BLOOM_FILTER_SIZE_BYTES = m_bytes
    
    bloom_filter = np.zeros(m_bytes, dtype=np.uint8)
    m_bits = m_bytes * 8

    print("Populating Bloom Filter with CONSISTENT HASHING...")
    # TQDM progress bar for hashing
    for word in tqdm(target_words, desc="Hashing targets"):
        word_bytes = word.encode('latin-1', errors='ignore')
        
        h1 = simple_hash_py(word_bytes, 0xDEADBEEF)
        h2 = simple_hash_py(word_bytes, 0xCAFEBABE)
        
        for i in range(k):
            index = (h1 + i * h2)
            bit_index_full = index % m_bits
            
            byte_index = bit_index_full // 8
            bit_shift = bit_index_full % 8
            
            bloom_filter[byte_index] |= (1 << bit_shift)

    print("Bloom Filter generation complete.")
    return bloom_filter.tobytes()

# ====================================================================
# --- EXPANDED INTERNAL RULE GENERATOR ---
# ====================================================================

def generate_comprehensive_hashcat_rules():
    """Generate an extensive subset of the most common Hashcat rules (>2500 rules)."""
    rules = []
    print(f"{green('Generating EXTENSIVE Hashcat rules as default...')}")
    
    rules.extend([ 'l', 'u', 'c', 'C', 't', 'r', 'k', ':', 'd0', 'f', '[', ']', '{', '}', 'p', 'z', 'q', 'x', 'y', 'L', 'R', '@', 'X0', '\'' ])
    
    for i in range(16):
        pos_hex = hex(i)[2:]
        rules.extend([f'T{pos_hex}', f'D{pos_hex}', f'I{pos_hex}', f'O{pos_hex}', f'M{pos_hex}'])
        
    common_chars = string.digits + "!@#$%^&*()_-+=[]{};:,.<>/?~`" + string.ascii_letters[0:15]
    for char in common_chars:
        if char not in [' ', '\n', '\t']: 
            rules.extend([f'^{char}', f'${char}', f'@{char}'])

    insert_chars = string.digits + "!@#$%^&*()_-+=" + string.ascii_letters[0:8]
    for pos in range(16):
        pos_hex = hex(pos)[2:] 
        for char in insert_chars:
            rules.extend([f'i{pos_hex}{char}', f'o{pos_hex}{char}', f'y{pos_hex}{char}']) 

    leet_subs = ['@', '4', '3', '1', '0', '5', '$', '7', '+', '-', '*', '#', '(', ')', '!', '?', '8', '9'] 
    for orig_char in string.ascii_lowercase: 
        for sub_char in leet_subs:
            rules.append(f's{orig_char}{sub_char}')
            
    purge_chars = string.ascii_letters + string.digits
    for char in purge_chars:
        rules.append(f'P{char}')
        
    rules = list(set(rules))  
    
    print(f"Generated {len(rules):,} total unique rules.")
    return rules

def load_rules_from_file(filename):
    """Loads rules from an external Hashcat rule file."""
    rules = []
    print(f"\nLoading rules from: {filename}")
    try:
        with open(filename, 'r', encoding='latin-1') as f:
            # TQDM progress bar for reading rules
            for line in tqdm(f, desc="Reading rules"):
                rule = line.strip()
                if rule and not rule.startswith('#') and ' ' not in rule: 
                     rules.append(rule)
        
        rules = list(set(rules))
        print(f"Loaded: {len(rules):,} unique base rules")
        return rules
    except FileNotFoundError:
        print(f"{red('ERROR:')} Rule file not found: {filename}. Using default generated rules.")
        return []

def save_rules_to_file(rules, filename):
    """Saves rules to a file."""
    print(f"Saving {len(rules):,} rules to {filename}")
    try:
        with open(filename, 'w') as f:
            for rule in rules:
                f.write(f"{rule}\n")
    except Exception as e:
        print(f"{red('ERROR:')} Could not save rules to file: {e}")

# ====================================================================
# --- OPENCL KERNEL: RULE_CHAINING_KERNEL (SWITCH STATEMENT OPTIMIZATION) ---
# ====================================================================

RULE_CHAINING_KERNEL = """
#define MAX_WORD_LEN 32
#define MAX_OUTPUT_LEN 48
#define MAX_CHAIN_STRING_LEN 16 
#define NULL_BYTE 0
#define MAX_CHAINS_TO_FIND_LIMIT //MAX_CHAINS_PLACEHOLDER// 

// Placeholders replaced by Python host code
#define BLOOM_HASH_FUNCTIONS_K //K_PLACEHOLDER// 
#define BLOOM_FILTER_SIZE_BITS (//M_BYTES_PLACEHOLDER// * 8) 
#define MAX_DEPTH_RUNTIME //MAX_DEPTH_PLACEHOLDER//

// Simple non-cryptographic hash (modified FNV-1a for example)
uint simple_hash(const char *key, int len, uint seed) {
    const uint fnv_prime = 16777619U; // Matches Python
    uint hash = seed;
    for (int i = 0; i < len; i++) {
        hash ^= (uint)key[i];
        hash *= fnv_prime;
    }
    return hash;
}

void generate_bloom_hashes(const char *word, int len, __private uint *h1_out, __private uint *h2_out) {
    // Matches Python seeds: 0xDEADBEEF and 0xCAFEBABE
    *h1_out = simple_hash(word, len, 0xDEADBEEFU);
    *h2_out = simple_hash(word, len, 0xCAFEBABEU);
}

// Bloom Filter lookup logic
bool is_target_word_found_bloom(__global const uchar *g_bloom_filter, const char *word, int len) {
    // Using bitwise OR | for warning cleanup (len == 0 || BLOOM_FILTER_SIZE_BITS == 0)
    if (len == 0 | BLOOM_FILTER_SIZE_BITS == 0) return false; 

    uint h1, h2;
    generate_bloom_hashes(word, len, &h1, &h2);
    
    for (int i = 0; i < BLOOM_HASH_FUNCTIONS_K; i++) {
        uint index = (h1 + i * h2);
        uint bit_index_full = index % BLOOM_FILTER_SIZE_BITS;
        
        uint byte_index = bit_index_full / 8;
        uint bit_shift = bit_index_full % 8;
        
        if (!((g_bloom_filter[byte_index] >> bit_shift) & 1)) {
            return false; // Definitely not in the set
        }
    }
    
    return true; 
}


// --- Helper Functions ---
char tolower_ocl(char c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    return c;
}
char toupper_ocl(char c) {
    if (c >= 'a' && c <= 'z') return c - ('a' - 'A');
    return c;
}
int get_pos(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1; 
}

int copy_word(const char *src, char *dst, int max_len) {
    int len = 0;
    for (len = 0; len < max_len; len++) {
        dst[len] = src[len];
        if (src[len] == NULL_BYTE) break;
    }
    if (len == max_len) dst[max_len - 1] = NULL_BYTE;
    return len;
}

uint simple_rng(uint seed) {
    return (seed * 1103515245U + 12345U) & 0x7FFFFFFFU;
}
// --- End Helper Functions ---


// Helper to record a found chain with weighted sampling
void record_chain(__global char *g_output_buffer, __global int *g_output_count, 
                  const char *chain_to_record, int depth_index, __private uint *rng_seed) {
    
    bool do_record = false;
    
    // CRITICAL WEIGHTED SAMPLING LOGIC (Tune for depth)
    if (depth_index == 0) { // Depth 1 chains (1 rule): Sample 1 in 100
        *rng_seed = simple_rng(*rng_seed);
        if ((*rng_seed % 100) == 0) { do_record = true; }
    } else if (depth_index == 1) { // Depth 2 chains (2 rules): Sample 1 in 10
        *rng_seed = simple_rng(*rng_seed);
        if ((*rng_seed % 10) == 0) { do_record = true; }
    } else { // Depth 3+ chains (3+ rules): Sample 100%
        do_record = true;
    }

    if (do_record) {
        int output_index = atomic_inc((__global volatile int *)g_output_count);
        
        if (output_index < MAX_CHAINS_TO_FIND_LIMIT) {
            __global char *output_ptr = g_output_buffer + output_index * MAX_CHAIN_STRING_LEN;
            copy_word(chain_to_record, output_ptr, MAX_CHAIN_STRING_LEN);
        }
    }
}

// Helper function to build the rule chain string
void build_chain(const char *current_chain, __global const char *new_rule, char *output_chain) {
    int current_chain_len = copy_word(current_chain, output_chain, MAX_CHAIN_STRING_LEN);
    
    if (current_chain_len > 0 && current_chain[0] != NULL_BYTE) {
        if (current_chain_len < MAX_CHAIN_STRING_LEN - 1) {
            output_chain[current_chain_len] = ' ';
            current_chain_len++;
        }
    }
    
    int rule_len = 0;
    while(new_rule[rule_len] != NULL_BYTE) {
        if (current_chain_len < MAX_CHAIN_STRING_LEN - 1) {
            output_chain[current_chain_len] = new_rule[rule_len];
            current_chain_len++;
        }
        rule_len++;
    }
    output_chain[current_chain_len] = NULL_BYTE;
}


// CRITICAL: Rule application logic
int apply_rule(__global const char *rule_start, const char *input_word, int input_len, char *output_word) {
    int new_len = copy_word(input_word, output_word, MAX_OUTPUT_LEN);
    if (new_len == 0) return 0;
    
    char rule_op = rule_start[0];
    char p1 = rule_start[1];
    char p2 = rule_start[2];
    
    // --- START RULE APPLICATION LOGIC (Optimized with switch) ---
    switch (rule_op) {
        case 'l': { // Lowercase (l)
            for (int k = 0; k < new_len; k++) output_word[k] = tolower_ocl(output_word[k]); 
            break;
        }
        case 'u': { // Uppercase (u)
            for (int k = 0; k < new_len; k++) output_word[k] = toupper_ocl(output_word[k]); 
            break;
        }
        case 'c': { // Capitalize (c)
            if (new_len > 0) output_word[0] = toupper_ocl(output_word[0]);
            for (int k = 1; k < new_len; k++) output_word[k] = tolower_ocl(output_word[k]);
            break;
        }
        case 'C': { // Toggle case first (C)
            if (new_len > 0) output_word[0] = tolower_ocl(output_word[0]);
            for (int k = 1; k < new_len; k++) output_word[k] = toupper_ocl(output_word[k]);
            break;
        }
        case 't': { // Toggle case all (t)
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                if (c >= 'a' && c <= 'z') output_word[k] = toupper_ocl(c);
                else if (c >= 'A' && c <= 'Z') output_word[k] = tolower_ocl(c);
            }
            break;
        }
        case '$': { // Append char ($X)
            if (p1 != NULL_BYTE) {
                if (new_len < MAX_WORD_LEN) { output_word[new_len] = p1; new_len++; }
            }
            break;
        }
        case '^': { // Prepend char (^X)
            if (p1 != NULL_BYTE) {
                if (new_len < MAX_WORD_LEN) {
                    for(int k = new_len; k > 0; k--) output_word[k] = output_word[k-1];
                    output_word[0] = p1; new_len++; 
                }
            }
            break;
        }
        case 'i': { // Insert char at position (i#X)
            if (p2 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos <= new_len && new_len < MAX_WORD_LEN) {
                    for(int k = new_len; k > pos; k--) output_word[k] = output_word[k-1];
                    output_word[pos] = p2; new_len++;
                }
            }
            break;
        }
        case 'o': { // Overwrite char at position (o#X)
            if (p2 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) { output_word[pos] = p2; }
            }
            break;
        }
        case 's': { // Substitute (sXY)
            if (p2 != NULL_BYTE) {
                char orig = p1;
                char sub = p2;
                for(int k = 0; k < new_len; k++) { if (output_word[k] == orig) output_word[k] = sub; }
            }
            break;
        }
        case 'T': { // Toggle char at position (T#)
            if (p1 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) {
                    char c = output_word[pos];
                    if (c >= 'a' && c <= 'z') output_word[pos] = toupper_ocl(c);
                    else if (c >= 'A' && c <= 'Z') output_word[pos] = tolower_ocl(c);
                }
            }
            break;
        }
        case 'r': { // Reverse (r)
            for (int k = 0; k < new_len / 2; k++) {
                char temp = output_word[k];
                output_word[k] = output_word[new_len - 1 - k];
                output_word[new_len - 1 - k] = temp;
            }
            break;
        }
        case 'k': // Duplicate (k)
        case ':': { // Duplicate (:)
            if (new_len * 2 < MAX_WORD_LEN) {
                for (int k = 0; k < new_len; k++) output_word[new_len + k] = output_word[k];
                new_len *= 2;
            }
            break;
        }
        case 'd': { // Duplicate first char (d0)
            if (rule_start[1] == '0') {
                if (new_len * 2 < MAX_WORD_LEN) {
                    for (int k = 0; k < new_len; k++) output_word[new_len + k] = output_word[k];
                    new_len *= 2;
                }
            } else {
                return 0; // Unsupported 'd' rule
            }
            break;
        }
        case 'f': { // Reflect (f)
            if (new_len * 2 - 1 < MAX_WORD_LEN && new_len > 0) {
                for (int k = 0; k < new_len - 1; k++) output_word[new_len + k] = output_word[new_len - 2 - k];
                new_len = new_len * 2 - 1;
            }
            break;
        }
        case '[': { // Delete first char ([)
            if (new_len > 0) {
                for(int k = 0; k < new_len; k++) output_word[k] = output_word[k+1];
                new_len--;
            }
            break;
        }
        case ']': { // Delete last char (])
            if (new_len > 0) { new_len--; }
            break;
        }
        case 'D': { // Delete char at position (D#)
            if (p1 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) {
                    for(int k = pos; k < new_len; k++) output_word[k] = output_word[k+1];
                    new_len--;
                }
            }
            break;
        }
        case '@': // Purge/Delete all instances of a character (@X)
        case 'P': { // Purge all instances of a character (P#) - Alias of @
            if (p1 != NULL_BYTE) {
                char purge_char = p1;
                int j = 0;
                for (int k = 0; k < new_len; k++) {
                    if (output_word[k] != purge_char) { output_word[j] = output_word[k]; j++; }
                }
                new_len = j;
            }
            break;
        }
        case 'p': { // Duplicate last char (p)
            if (new_len > 0 && new_len < MAX_WORD_LEN) {
                output_word[new_len] = output_word[new_len - 1];
                new_len++;
            }
            break;
        }
        case 'z': { // Duplicate first char (z)
            if (new_len > 0 && new_len < MAX_WORD_LEN) {
                for(int k = new_len; k > 0; k--) output_word[k] = output_word[k-1];
                output_word[0] = output_word[1]; new_len++;
            }
            break;
        }
        case 'L': { // Delete all lowercase (L)
            int j = 0;
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                if (c < 'a' || c > 'z') { output_word[j] = c; j++; }
            }
            new_len = j;
            break;
        }
        case 'R': { // Delete all uppercase (R)
            int j = 0;
            for (int k = 0; k < new_len; k++) {
                char c = output_word[k];
                if (c < 'A' || c > 'Z') { output_word[j] = c; j++; }
            }
            new_len = j;
            break;
        }
        case '{': { // Rotate left ({)
            if (new_len > 1) {
                char first = output_word[0];
                for (int k = 0; k < new_len - 1; k++) output_word[k] = output_word[k+1];
                output_word[new_len - 1] = first;
            }
            break;
        }
        case '}': { // Rotate right (})
            if (new_len > 1) {
                char last = output_word[new_len - 1];
                for (int k = new_len - 1; k > 0; k--) output_word[k] = output_word[k-1];
                output_word[0] = last;
            }
            break;
        }
        case '\\'': { // Truncate/Chop at position ('#)
            if (p1 != NULL_BYTE) {
                int pos = get_pos(p1);
                if (pos >= 0 && pos < new_len) { new_len = pos; }
            }
            break;
        }
        case 'x': { // Extract character range (x#$)
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
        default:
            return 0; // Rule not supported or invalid
    }
    
    // --- END RULE APPLICATION LOGIC ---
    
    if (new_len >= MAX_OUTPUT_LEN) new_len = MAX_OUTPUT_LEN - 1;
    output_word[new_len] = NULL_BYTE;
    
    return new_len;
}


__kernel void rule_chaining_kernel(
    // Input Data Buffers
    __global const char *g_base_words_ptr,
    __global const int *g_word_offsets,
    __global const char *g_rules_ptr,
    __global const int *g_rule_offsets,
    __global const uchar *g_bloom_filter, 

    // Input Configuration
    const int num_base_words,
    const int num_rules,
    const int max_depth, 
    
    // Output Buffers
    __global char *g_output_buffer,
    __global int *g_output_count 
) {
    int gid = get_global_id(0);
    if (gid >= num_base_words) return;

    __global const char *base_word_start = g_base_words_ptr + g_word_offsets[gid];
    
    // --- STATE MANAGEMENT: Use private memory for chaining state ---
    char w0[MAX_OUTPUT_LEN]; 
    char c0[MAX_CHAIN_STRING_LEN] = {NULL_BYTE}; 
    int len0 = copy_word(base_word_start, w0, MAX_WORD_LEN);
    
    char w1[MAX_OUTPUT_LEN]; 
    char c1[MAX_CHAIN_STRING_LEN]; 
    char w2[MAX_OUTPUT_LEN];
    char c2[MAX_CHAIN_STRING_LEN]; 
    char w3[MAX_OUTPUT_LEN];
    char c3[MAX_CHAIN_STRING_LEN]; 

    // Use gid as seed base, but apply rule to keep it unique per work-item
    uint rng_seed = gid * 1234567U + 1; 

    // --- DEPTH 1 (RULE R1) ---
    for (int r1 = 0; r1 < num_rules; r1++) {
        __global const char *rule1_start = g_rules_ptr + g_rule_offsets[r1];
        
        int len1 = apply_rule(rule1_start, w0, len0, w1);
        
        if (len1 > 0) {
            build_chain(c0, rule1_start, c1);
            if (is_target_word_found_bloom(g_bloom_filter, w1, len1)) {
                 record_chain(g_output_buffer, g_output_count, c1, 0, &rng_seed); 
            }

            // --- DEPTH 2 (RULE R2) ---
            if (max_depth >= 2) {
                for (int r2 = 0; r2 < num_rules; r2++) {
                    __global const char *rule2_start = g_rules_ptr + g_rule_offsets[r2];

                    int len2 = apply_rule(rule2_start, w1, len1, w2);

                    if (len2 > 0) {
                        build_chain(c1, rule2_start, c2);
                        if (is_target_word_found_bloom(g_bloom_filter, w2, len2)) {
                            record_chain(g_output_buffer, g_output_count, c2, 1, &rng_seed); 
                        }

                        // --- DEPTH 3 (RULE R3) ---
                        if (max_depth >= 3) {
                            for (int r3 = 0; r3 < num_rules; r3++) {
                                __global const char *rule3_start = g_rules_ptr + g_rule_offsets[r3];

                                int len3 = apply_rule(rule3_start, w2, len2, w3);

                                if (len3 > 0) {
                                    build_chain(c2, rule3_start, c3);
                                    if (is_target_word_found_bloom(g_bloom_filter, w3, len3)) {
                                        record_chain(g_output_buffer, g_output_count, c3, 2, &rng_seed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

# ====================================================================
# --- HOST CODE: GPU EXECUTION FUNCTION (REFACTORED FOR BATCHING) ---
# ====================================================================

def fast_gpu_rule_chaining_all_paths(base_words, target_words, rules, max_depth, batch_size):
    """
    Sets up OpenCL and executes the kernel in batches to find rule chains,
    displaying a progress bar over the base words.
    """
    if not rules or not base_words: return []
    
    # --- 1. Preparation & Bloom Filter Generation ---
    bloom_filter_bytes = generate_bloom_filter(target_words)
    bloom_filter_np = np.frombuffer(bloom_filter_bytes, dtype=np.uint8)
    
    # Inject constants into the kernel source
    kernel_source = RULE_CHAINING_KERNEL
    kernel_source = kernel_source.replace("//K_PLACEHOLDER//", str(BLOOM_HASH_FUNCTIONS))
    kernel_source = kernel_source.replace("//M_BYTES_PLACEHOLDER//", str(BLOOM_FILTER_SIZE_BYTES))
    kernel_source = kernel_source.replace("//MAX_CHAINS_PLACEHOLDER//", str(MAX_CHAINS_TO_FIND))
    kernel_source = kernel_source.replace("//MAX_DEPTH_PLACEHOLDER//", str(max_depth))

    try:
        # 2. Setup Context and Queue
        platform = cl.get_platforms()[0]
        device = platform.get_devices(cl.device_type.GPU)[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
    except Exception as e:
        print(f"{red('FATAL ERROR:')} Could not initialize PyOpenCL. Error: {e}")
        return []

    # 3. Kernel Compilation
    try:
        print(f"\n{blue('Compiling OpenCL Kernel (Optimized Switch)...')}")
        program = cl.Program(context, kernel_source).build()
        kernel = program.rule_chaining_kernel
    except cl.LogicError as e:
        print(f"{red('FATAL KERNEL ERROR:')} OpenCL kernel compilation failed. Error: {e}")
        try:
            print("\n--- OPENCL BUILD LOG (ERROR) ---")
            print(program.get_build_log(device))
            print("--------------------------------")
        except:
             pass
        return []
    
    mf = cl.mem_flags
    
    # --- 4. Static Buffers (Rules and Bloom Filter) ---
    rules_bytes = b'\0'.join([r.encode('latin-1') for r in rules]) + b'\0'
    rule_offsets = np.array([0] + [len(r) + 1 for r in rules], dtype=np.int32).cumsum()[:-1]
    d_rules = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rules_bytes)
    d_rule_offsets = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rule_offsets)
    d_bloom_filter = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bloom_filter_np)
    
    # --- 5. Output Buffers (Reused Per Batch) ---
    output_buffer_bytes = MAX_CHAINS_TO_FIND * MAX_CHAIN_STRING_LEN
    d_output_buffer = cl.Buffer(context, mf.WRITE_ONLY, output_buffer_bytes)
    
    all_chains = set()
    num_total_base_words = len(base_words)
    num_rules_arg = np.int32(len(rules))
    max_depth_arg = np.int32(max_depth)

    print(f"\n{bold(blue('...Starting GPU Chaining Process (Batched)...'))}")
    print(f"Rules: {num_rules_arg}, Max Depth: {max_depth}, Batch Size: {batch_size:,}")
    start_time = time.time()
    
    # --- 6. Batch Processing Loop with TQDM ---
    with tqdm(total=num_total_base_words, desc="Base Words Processed", unit="words") as pbar:
        for i in range(0, num_total_base_words, batch_size):
            batch = base_words[i:i + batch_size]
            current_batch_size = len(batch)
            
            # --- Prepare Batch Inputs ---
            base_words_batch_bytes = b'\0'.join([w.encode('latin-1', errors='ignore') for w in batch]) + b'\0'
            word_offsets_batch = np.array([0] + [len(w) + 1 for w in batch], dtype=np.int32).cumsum()[:-1]
            
            # --- Batch-specific Input Buffers ---
            d_base_words = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=base_words_batch_bytes)
            d_word_offsets = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=word_offsets_batch)
            
            # --- Output Count Buffer (Reset per Batch) ---
            h_output_count = np.array([0], dtype=np.int32) 
            d_output_count = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_output_count) 
            
            # --- Execute Kernel for the Batch ---
            GLOBAL_SIZE = current_batch_size
            LOCAL_SIZE = LOCAL_WORK_SIZE 
            num_base_words_arg = np.int32(current_batch_size)
            
            kernel.set_args(
                d_base_words, d_word_offsets, d_rules, d_rule_offsets, d_bloom_filter, 
                num_base_words_arg, num_rules_arg, max_depth_arg, 
                d_output_buffer, d_output_count
            )
            
            cl.enqueue_nd_range_kernel(queue, kernel, (GLOBAL_SIZE,), (LOCAL_SIZE,)).wait()
            
            # --- Read Results Back ---
            cl.enqueue_copy(queue, h_output_count, d_output_count).wait()
            chains_found_in_batch = h_output_count[0]
            
            if chains_found_in_batch > 0:
                # Calculate the exact size needed to read back
                read_count = min(chains_found_in_batch, MAX_CHAINS_TO_FIND)
                read_size = read_count * MAX_CHAIN_STRING_LEN
                
                # Create a host buffer of exactly the required size.
                h_output_buffer_current = np.empty(read_size, dtype=np.byte)

                cl.enqueue_copy(queue, h_output_buffer_current, d_output_buffer).wait()
                
                for j in range(read_count):
                    start = j * MAX_CHAIN_STRING_LEN
                    chain_bytes = h_output_buffer_current[start:start + MAX_CHAIN_STRING_LEN]
                    
                    try:
                        # Extract the string before the null byte
                        null_pos = chain_bytes.tolist().index(0)
                        chain_str = chain_bytes[:null_pos].tobytes().decode('latin-1').strip()
                    except ValueError:
                        # Fallback in case null byte is missing or at the very end
                        chain_str = chain_bytes.tobytes().decode('latin-1', errors='ignore').strip('\x00').strip()
                        
                    if chain_str:
                        all_chains.add(chain_str)

            # Update progress bar and release memory
            pbar.update(current_batch_size)
            del d_base_words, d_word_offsets, d_output_count
            
    end_time = time.time()
    
    print(f"\n{green('Kernel Execution Complete.')} Time: {end_time - start_time:.2f}s.")
    return list(all_chains)

# ====================================================================
# --- MAIN EXECUTION BLOCK ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"{bold('GPU Rule Chaining Finder (Fixed Multi-Depth)')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('base_wordlist', help='Path to the wordlist containing BASE words (R1).')
    parser.add_argument('target_wordlist', help='Path to the wordlist containing TARGET words (B).')
    parser.add_argument('-r', '--rules', type=str, default=None,
                        help='Path to a Hashcat rule file. If not provided, a large internal set (~2500 rules) will be generated.')
    parser.add_argument('-d', '--depth', type=int, default=MAX_DEPTH,
                        help=f'Maximum rule chain depth (R1, R1 R2, R1 R2 R3...). Default is {MAX_DEPTH}. \n{red("WARNING:")} Depth 3 is extremely slow with large rule sets.')
    parser.add_argument('-b', '--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Number of base words processed in a single GPU call. Default: {DEFAULT_BATCH_SIZE:,}. Lower for small GPU memory.')
    parser.add_argument('-o', '--output', type=str, default='found_chains.txt',
                        help='Output file to save the found rule chains.')
    
    args = parser.parse_args()

    # --- Setup ---
    print(f"\n{bold(green('=' * 70))}")
    print(f"{bold('🔬 GPU RULE CHAINING FINDER')}")
    print(f"{bold(green('=' * 70))}{Colors.END}\n")
    
    if args.depth > 3:
        print(f"{yellow('NOTE:')} Max depth limited to 3 in the kernel for simplicity.")
        args.depth = 3
    
    # 1. Load Wordlists
    base_words = load_wordlist_fast(args.base_wordlist)
    target_words = load_wordlist_fast(args.target_wordlist)

    if not base_words or not target_words:
        print(f"{red('ERROR:')} One or both wordlists are empty. Exiting.")
        sys.exit(1)
        
    # 2. Load or Generate Rules
    if args.rules and os.path.exists(args.rules):
        rules = load_rules_from_file(args.rules)
    else:
        rules = generate_comprehensive_hashcat_rules()
        if not args.rules:
            rules_file = 'default_generated_rules.txt'
            save_rules_to_file(rules, rules_file)
    
    if not rules:
        print(f"{red('ERROR:')} No rules available. Exiting.")
        sys.exit(1)
    
    # 3. Find chains using comprehensive GPU approach
    chains = fast_gpu_rule_chaining_all_paths(
        base_words, 
        target_words, 
        rules, 
        max_depth=args.depth,
        batch_size=args.batch_size
    )
    
    # 4. Save and Report Results
    with open(args.output, 'w') as f:
        for chain in chains:
            f.write(f"{chain}\n")
    
    print(f"\n{bold(green('=' * 70))}")
    print(f"{bold('✨ FINAL RESULTS REPORT ✨')}")
    print(f"{bold(green('=' * 70))}")
    print(f"Total base words: {len(base_words):,}")
    print(f"Total target words: {len(target_words):,}") 
    print(f"Rules used: {len(rules):,}")
    print(f"Unique Chains Found: {bold(yellow(len(chains))):,}")
    print(f"Output saved to: {bold(args.output)}")
    print(f"{bold(green('=' * 70))}{Colors.END}")
