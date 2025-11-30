import os
import sys
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
from collections import Counter
from tqdm import tqdm
import mmap
import math
import time
import json
import signal
import csv
from typing import List, Tuple, Set, Dict, Optional

# Suppress compiler warnings
if 'PYOPENCL_COMPILER_OUTPUT' not in os.environ:
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

# --- Color formatting constants ---
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

# --- Constants for High Performance ---
MAX_WORD_LEN = 32
MAX_OUTPUT_LEN = MAX_WORD_LEN * 2
LOCAL_WORK_SIZE = 256
GLOBAL_HASH_MAP_BITS = 28

# --- Fast FNV-1a Hash Function ---
def fast_fnv1a_hash_32(data):
    """Optimized FNV-1a hash for bytes"""
    if isinstance(data, np.ndarray):
        hash_val = np.uint32(2166136261)
        for byte in data:
            hash_val ^= np.uint32(byte)
            hash_val *= np.uint32(16777619)
        return hash_val
    else:
        hash_val = 2166136261
        for byte in data:
            hash_val = (hash_val ^ byte) * 16777619 & 0xFFFFFFFF
        return hash_val

# --- Optimized File Loading ---
def load_wordlist_fast(filename, max_words=None):
    """Fast wordlist loading with progress"""
    words = []
    file_size = os.path.getsize(filename)
    
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = 0
            
            with tqdm(total=file_size, desc=f"Loading {os.path.basename(filename)}", unit="B", unit_scale=True) as pbar:
                while pos < file_size:
                    end_pos = mm.find(b'\n', pos)
                    if end_pos == -1:
                        end_pos = file_size
                    
                    line = mm[pos:end_pos].strip()
                    pos = end_pos + 1
                    pbar.update(pos - pbar.n)
                    
                    if line and len(line) <= MAX_WORD_LEN:
                        try:
                            word = line.decode('latin-1').strip()
                            if word:
                                words.append(word)
                                if max_words and len(words) >= max_words:
                                    break
                        except:
                            pass
    
    return words

def build_target_hash_map(target_words, hash_map_bits=GLOBAL_HASH_MAP_BITS):
    """Build GPU hash map for O(1) lookups"""
    hash_map_size = 1 << (hash_map_bits - 5)
    hash_map = np.zeros(hash_map_size, dtype=np.uint32)
    mask = (1 << (hash_map_bits - 5)) - 1
    
    print(f"{blue('📊')} {bold('Building target hash map:')} {cyan(f'{len(target_words):,}')} {bold('words')}")
    
    for word in tqdm(target_words, desc="Building hash map"):
        word_hash = fast_fnv1a_hash_32(word.encode('latin-1'))
        map_index = (word_hash >> 5) & mask
        bit_index = word_hash & 31
        hash_map[map_index] |= (1 << bit_index)
    
    return hash_map, mask

# --- Data Preparation for GPU ---

def generate_leetspeak_rules():
    """Generates a list of common leetspeak substitution rules."""
    leetspeak_rules = []
    # Most common leetspeak substitutions
    substitutions = {
        'a': '@', 'e': '3', 'i': '1', 'o': '0',
        's': '5', 't': '7', 'l': '1', 'z': '2'
    }
    
    for original, sub in substitutions.items():
        # Lowercase substitution (e.g., sa@)
        leetspeak_rules.append(f"s{original}{sub}")
        
        # Uppercase substitution (e.g., sA@)
        if original.isalpha():
            leetspeak_rules.append(f"s{original.upper()}{sub}")
            
        # Swap substitution (e.g., s@a)
        leetspeak_rules.append(f"s{sub}{original}")
        
    return leetspeak_rules

def generate_all_rules():
    """Generates the built-in, static set of rules including new Hashcat rules."""
    rules = []
    
    # Simple rules (ID 0-9): lowercase, uppercase, capitalize, inverse capitalize, toggle case, reverse, swap first two, identity, duplicate, reflect
    simple_rules = ['l', 'u', 'c', 'C', 't', 'r', 'k', ':', 'd', 'f']
    rules.extend(simple_rules)
    
    # T and D rules (ID 10-29): toggle case at position i, delete character at position i
    for i in range(10):
        rules.extend([f'T{i}', f'D{i}'])
    
    # s rules (ID 30-...): substitution rules
    chars_2arg = string.digits + string.ascii_lowercase
    for c1 in chars_2arg:
        for c2 in chars_2arg:
            rules.append(f's{c1}{c2}')
            
    # Add leetspeak rules
    rules.extend(generate_leetspeak_rules())

    # Group A rules (ID ...-...): prepend (^), append ($), delete all instances (@)
    chars = string.digits + string.ascii_letters + string.punctuation
    for c in chars:
        rules.extend([f'^{c}', f'${c}', f'@{c}'])
    
    # --- NEW GROUP B RULES ---
    # p, {, }, [, ], x, O, i, o, ', z, Z, q
    new_rules = []
    
    # 'pN' - Duplicate word N times
    for n in range(10):  # 0-9
        new_rules.append(f'p{n}')
    
    # '{' - Rotate left
    new_rules.append('{')
    # '}' - Rotate right  
    new_rules.append('}')
    # '[' - Delete first char
    new_rules.append('[')
    # ']' - Delete last char
    new_rules.append(']')
    
    # 'xNM' - Extract range (N=start, M=length)
    for n in range(10):
        for m in range(1, 10):  # length from 1-9
            new_rules.append(f'x{n}{m}')
    
    # 'ONM' - Omit range (N=start, M=length)
    for n in range(10):
        for m in range(1, 10):  # length from 1-9
            new_rules.append(f'O{n}{m}')
    
    # 'iNX' - Insert char X at position N
    for n in range(10):
        for x in string.digits + string.ascii_lowercase:
            new_rules.append(f'i{n}{x}')
    
    # 'oNX' - Overwrite char at position N with X
    for n in range(10):
        for x in string.digits + string.ascii_lowercase:
            new_rules.append(f'o{n}{x}')
    
    # "'N" - Truncate at position N
    for n in range(10):
        new_rules.append(f"'{n}")  # Single quote rules
    
    # 'zN' - Duplicate first char N times
    for n in range(10):
        new_rules.append(f'z{n}')
    
    # 'ZN' - Duplicate last char N times
    for n in range(10):
        new_rules.append(f'Z{n}')
    
    # 'q' - Duplicate all characters
    new_rules.append('q')
    
    rules.extend(new_rules)
    
    # --- NEW COMPREHENSIVE RULES ---
    comprehensive_rules = []
    
    # 'K' - Swap last two characters
    comprehensive_rules.append('K')
    
    # '*NM' - Swap character at position N with character at position M
    for n in range(10):
        for m in range(10):
            if n != m:  # No point swapping same position
                comprehensive_rules.append(f'*{n}{m}')
    
    # 'LN' - Bitwise shift left character @ N
    for n in range(10):
        comprehensive_rules.append(f'L{n}')
    
    # 'RN' - Bitwise shift right character @ N  
    for n in range(10):
        comprehensive_rules.append(f'R{n}')
    
    # '+N' - ASCII increment character @ N by 1
    for n in range(10):
        comprehensive_rules.append(f'+{n}')
    
    # '-N' - ASCII decrement character @ N by 1
    for n in range(10):
        comprehensive_rules.append(f'-{n}')
    
    # '.N' - Replace character @ N with value at @ N plus 1
    for n in range(10):
        comprehensive_rules.append(f'.{n}')
    
    # ',N' - Replace character @ N with value at @ N minus 1
    for n in range(10):
        comprehensive_rules.append(f',{n}')
    
    # 'yN' - Duplicate first N characters
    for n in range(1, 10):  # 1-9, 0 doesn't make sense
        comprehensive_rules.append(f'y{n}')
    
    # 'YN' - Duplicate last N characters
    for n in range(1, 10):  # 1-9, 0 doesn't make sense
        comprehensive_rules.append(f'Y{n}')
    
    # 'E' - Title case
    comprehensive_rules.append('E')
    
    # 'eX' - Title case with custom separator
    for x in ['-', '_', '.', ',', ';']:
        comprehensive_rules.append(f'e{x}')
    
    # '3NX' - Toggle case after Nth instance of separator char
    for n in range(1, 5):  # 1-4 instances
        for x in ['-', '_', '.', ',', ';', ' ']:
            comprehensive_rules.append(f'3{n}{x}')
    
    rules.extend(comprehensive_rules)
    
    # --- INSERT EVERY RULES (vNX) ---
    insert_rules = []
    # 'vNX' - Insert character X every N characters
    for n in range(1, 10):  # 1-9
        for x in string.digits + string.ascii_lowercase + string.punctuation:
            insert_rules.append(f'v{n}{x}')
    
    # 'vX' - Insert character X after every character (N=0)
    for x in string.digits + string.ascii_lowercase + string.punctuation:
        insert_rules.append(f'v0{x}')
    
    rules.extend(insert_rules)
    
    return rules

def prepare_data_for_gpu(words, rules, max_word_len):
    """
    Prepares words and rules for GPU transfer.
    Key: Maps rule string to its sequential ID.
    """
    
    rule_map = {}
    for i, r in enumerate(rules):
        rule_map[r] = i

    max_rule_len = max(len(rule.encode('latin-1')) for rule in rules) if rules else 0
    # +1 for null terminator, +1 for Rule ID (stored as uint16)
    max_rule_len_padded = max_rule_len + 1 
    
    # rules_padded stores rule ID (uint16) + rule bytes (uint16 * max_rule_len_padded)
    rules_padded = np.zeros((len(rules), max_rule_len_padded + 1), dtype=np.uint16)
    for i, rule in enumerate(rules):
        rule_bytes = rule.encode('latin-1')
        rule_id = rule_map.get(rule, 65535) 
        
        rules_padded[i, 0] = rule_id
        # Store rule bytes as a sequence of uint16 for alignment/ease of access in kernel
        rules_padded[i, 1:1+len(rule_bytes)] = np.frombuffer(rule_bytes, dtype=np.uint8)
    
    # Pad words for GPU: max_word_len includes null terminator
    words_padded = np.zeros((len(words), max_word_len), dtype=np.uint8)
    for i, word in enumerate(words):
        if not word: continue
        word_bytes = word.encode('latin-1')
        words_padded[i, :len(word_bytes)] = np.frombuffer(word_bytes, dtype=np.uint8)
        
    return words_padded, rules_padded, max_rule_len_padded

# --- Word Processing Functions ---
def prepare_words_for_gpu(words, max_word_len=MAX_WORD_LEN):
    """Convert words to GPU-friendly format"""
    words_array = np.zeros((len(words), max_word_len), dtype=np.uint8)
    for i, word in enumerate(words):
        word_bytes = word.encode('latin-1')
        words_array[i, :len(word_bytes)] = np.frombuffer(word_bytes, dtype=np.uint8)
    return words_array

def prepare_rules_for_gpu(rules_list, max_rule_len=64):
    """Prepare rules for GPU processing"""
    rules_padded = np.zeros((len(rules_list), max_rule_len + 1), dtype=np.uint16)
    rule_map = {}
    
    for i, rule in enumerate(rules_list):
        rule_bytes = rule.encode('latin-1')
        rules_padded[i, 0] = i  # rule ID
        if len(rule_bytes) > 0:
            rules_padded[i, 1:1+len(rule_bytes)] = np.frombuffer(rule_bytes, dtype=np.uint8)
        rule_map[i] = rule
    
    return rules_padded, rule_map

# --- Chain Depth Processing ---
def save_words_for_next_depth(words_with_chains, depth):
    """Save words with their rule chains for next depth"""
    temp_file = f"words_depth_{depth}.tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for word, chain in words_with_chains:
            f.write(f"{word}\t{chain}\n")
    return temp_file

def load_words_for_depth(depth):
    """Load words with rule chains for given depth"""
    temp_file = f"words_depth_{depth}.tmp"
    words_with_chains = []
    
    if os.path.exists(temp_file):
        print(f"{blue('📖')} {bold('Loading')} {cyan(temp_file)} {bold('for depth')} {cyan(depth)}")
        with open(temp_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    words_with_chains.append((parts[0], parts[1]))
    
    return words_with_chains

def cleanup_temp_files(max_depth):
    """Clean up temporary files"""
    for depth in range(1, max_depth + 2):
        temp_file = f"words_depth_{depth}.tmp"
        if os.path.exists(temp_file):
            os.remove(temp_file)

# --- Original BFS Kernel (Compatible with existing code) ---
def get_bfs_kernel_source(start_id_simple, num_simple_rules, start_id_TD, num_td_rules, 
                         start_id_s, num_s_rules, start_id_A, num_a_rules):
    """Get the original BFS kernel that works with the existing code"""
    return f"""
// Helper function to convert char digit/letter to int position
unsigned int char_to_pos(unsigned char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    // Return a value guaranteed to fail bounds checks
    return 0xFFFFFFFF; 
}}

__kernel void bfs_kernel(
    __global const unsigned char* base_words_in,
    __global const unsigned short* rules_in,
    __global unsigned char* result_buffer,
    const unsigned int num_words,
    const unsigned int num_rules,
    const unsigned int max_word_len,
    const unsigned int max_rule_len_padded,
    const unsigned int max_output_len_padded)
{{
    unsigned int global_id = get_global_id(0);
    unsigned int word_idx = global_id / num_rules;
    unsigned int rule_idx = global_id % num_rules;

    if (word_idx >= num_words) return;

    __global const unsigned char* current_word_ptr = base_words_in + word_idx * max_word_len;
    __global const unsigned short* rule_id_ptr = rules_in + rule_idx * (max_rule_len_padded + 1); 
    __global const unsigned char* rule_ptr = (__global const unsigned char*)rules_in + rule_idx * (max_rule_len_padded + 1) * sizeof(unsigned short) + sizeof(unsigned short);

    unsigned int rule_id = rule_id_ptr[0];

    __global unsigned char* result_ptr = result_buffer + global_id * max_output_len_padded;

    unsigned int word_len = 0;
    for (unsigned int i = 0; i < max_word_len; i++) {{
        if (current_word_ptr[i] == 0) {{
            word_len = i;
            break;
        }}
    }}
    
    unsigned int out_len = 0;
    bool changed_flag = false;
    
    // Zero out the result buffer for this thread
    for(unsigned int i = 0; i < max_output_len_padded; i++) {{
        result_ptr[i] = 0;
    }}

    // --- Unify rule ID blocks (Substituted from Python) ---
    unsigned int start_id_simple = {start_id_simple};
    unsigned int end_id_simple = start_id_simple + {num_simple_rules};
    unsigned int start_id_TD = {start_id_TD};
    unsigned int end_id_TD = start_id_TD + {num_td_rules};
    unsigned int start_id_s = {start_id_s};
    unsigned int end_id_s = start_id_s + {num_s_rules};
    unsigned int start_id_A = {start_id_A};
    unsigned int end_id_A = start_id_A + {num_a_rules};
    
    // --- Kernel Logic (Rule Transformation) ---
    
    if (rule_id >= start_id_simple && rule_id < end_id_simple) {{ // Simple rules (l, u, c, C, t, r, k, :, d, f)
        switch(rule_id - start_id_simple) {{
            case 0: {{ // 'l' (lowercase)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'A' && c <= 'Z') {{
                        result_ptr[i] = c + 32;
                        changed_flag = true;
                    }} else {{
                        result_ptr[i] = c;
                    }}
                }}
                break;
            }}
            case 1: {{ // 'u' (uppercase)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'a' && c <= 'z') {{
                        result_ptr[i] = c - 32;
                        changed_flag = true;
                    }} else {{
                        result_ptr[i] = c;
                    }}
                }}
                break;
            }}
            case 2: {{ // 'c' (capitalize)
                out_len = word_len;
                if (word_len > 0) {{
                    if (current_word_ptr[0] >= 'a' && current_word_ptr[0] <= 'z') {{
                        result_ptr[0] = current_word_ptr[0] - 32;
                        changed_flag = true;
                    }} else {{
                        result_ptr[0] = current_word_ptr[0];
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = current_word_ptr[i];
                        if (c >= 'A' && c <= 'Z') {{ // Ensure rest is lowercase
                            result_ptr[i] = c + 32;
                            changed_flag = true;
                        }} else {{
                            result_ptr[i] = c;
                        }}
                    }}
                }}
                break;
            }}
            case 3: {{ // 'C' (invert capitalize)
                out_len = word_len;
                if (word_len > 0) {{
                    if (current_word_ptr[0] >= 'A' && current_word_ptr[0] <= 'Z') {{
                        result_ptr[0] = current_word_ptr[0] + 32;
                        changed_flag = true;
                    }} else {{
                        result_ptr[0] = current_word_ptr[0];
                    }}
                    for (unsigned int i = 1; i < word_len; i++) {{
                        unsigned char c = current_word_ptr[i];
                        if (c >= 'a' && c <= 'z') {{ // Ensure rest is UPPERCASE
                            result_ptr[i] = c - 32;
                            changed_flag = true;
                        }} else {{
                            result_ptr[i] = c;
                        }}
                    }}
                }}
                break;
            }}
            case 4: {{ // 't' (toggle case)
                out_len = word_len;
                for (unsigned int i = 0; i < word_len; i++) {{
                    unsigned char c = current_word_ptr[i];
                    if (c >= 'a' && c <= 'z') {{
                        result_ptr[i] = c - 32;
                        changed_flag = true;
                    }} else if (c >= 'A' && c <= 'Z') {{
                        result_ptr[i] = c + 32;
                        changed_flag = true;
                    }} else {{
                        result_ptr[i] = c;
                    }}
                }}
                break;
            }}
            case 5: {{ // 'r' (reverse)
                out_len = word_len;
                if (word_len > 1) {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result_ptr[i] = current_word_ptr[word_len - 1 - i];
                    }}
                    // Check if word actually changed
                    for (unsigned int i = 0; i < word_len; i++) {{
                        if (result_ptr[i] != current_word_ptr[i]) {{
                            changed_flag = true;
                            break;
                        }}
                    }}
                }} else {{
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result_ptr[i] = current_word_ptr[i];
                    }}
                }}
                break;
            }}
            case 6: {{ // 'k' (swap first two chars)
                out_len = word_len;
                for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
                if (word_len >= 2) {{
                    result_ptr[0] = current_word_ptr[1];
                    result_ptr[1] = current_word_ptr[0];
                    changed_flag = true;
                }}
                break;
            }}
            case 7: {{ // ':' (identity/no change)
                out_len = word_len;
                for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
                changed_flag = false;
                break;
            }}
            case 8: {{ // 'd' (duplicate)
                out_len = word_len * 2;
                if (out_len >= max_output_len_padded) {{
                    out_len = 0;	
                    changed_flag = false;
                    break;
                }}
                for(unsigned int i=0; i<word_len; i++) {{
                    result_ptr[i] = current_word_ptr[i];
                    result_ptr[word_len+i] = current_word_ptr[i];
                }}
                changed_flag = true;
                break;
            }}
            case 9: {{ // 'f' (reflect: word + reverse(word))
                out_len = word_len * 2;
                if (out_len >= max_output_len_padded) {{
                    out_len = 0;
                    changed_flag = false;
                    break;
                }}
                for(unsigned int i=0; i<word_len; i++) {{
                    result_ptr[i] = current_word_ptr[i];
                    result_ptr[word_len+i] = current_word_ptr[word_len-1-i];
                }}
                changed_flag = true;
                break;
            }}
        }}
    }} else if (rule_id >= start_id_TD && rule_id < end_id_TD) {{ // T, D rules (Toggle at pos, Delete at pos)
        // Read position from the second byte of the rule (e.g., T1 -> byte '1')
        unsigned char operator_char = rule_ptr[0];
        unsigned char pos_char = rule_ptr[1];
        
        unsigned int pos_to_change = pos_char - '0';
        
        if (operator_char == 'T') {{ // 'T' (toggle case at pos)
            out_len = word_len;
            for (unsigned int i = 0; i < word_len; i++) {{
                result_ptr[i] = current_word_ptr[i];
            }}
            if (pos_to_change < word_len) {{
                unsigned char c = current_word_ptr[pos_to_change];
                if (c >= 'a' && c <= 'z') {{
                    result_ptr[pos_to_change] = c - 32;
                    changed_flag = true;
                }} else if (c >= 'A' && c <= 'Z') {{
                    result_ptr[pos_to_change] = c + 32;
                    changed_flag = true;
                }}
            }}
        }}
        else if (operator_char == 'D') {{ // 'D' (delete char at pos)
            unsigned int out_idx = 0;
            if (pos_to_change < word_len) {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    if (i != pos_to_change) {{
                        result_ptr[out_idx++] = current_word_ptr[i];
                    }} else {{
                        changed_flag = true;
                    }}
                }}
            }} else {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    result_ptr[i] = current_word_ptr[i];
                }}
                out_idx = word_len;
            }}
            out_len = out_idx;
        }}
    }}
    else if (rule_id >= start_id_s && rule_id < end_id_s) {{ // 's' rules (substitute char)
        out_len = word_len;
        for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
        
        unsigned char find = rule_ptr[0];
        unsigned char replace = rule_ptr[1];
        for(unsigned int i = 0; i < word_len; i++) {{
            if (current_word_ptr[i] == find) {{
                result_ptr[i] = replace;
                changed_flag = true;
            }}
        }}
    }} else if (rule_id >= start_id_A && rule_id < end_id_A) {{ // Group A rules (Prepend ^, Append $, Delete all @)
        out_len = word_len;
        for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
        
        unsigned char cmd = rule_ptr[0];
        unsigned char arg = rule_ptr[1];
        
        if (cmd == '^') {{ // Prepend
            if (word_len + 1 >= max_output_len_padded) {{
                out_len = 0;
                changed_flag = false;
            }} else {{
                // Shift all characters right
                for(unsigned int i=word_len; i>0; i--) {{
                    result_ptr[i] = result_ptr[i-1];
                }}
                result_ptr[0] = arg;
                out_len++;
                changed_flag = true;
            }}
        }} else if (cmd == '$') {{ // Append
            if (word_len + 1 >= max_output_len_padded) {{
                out_len = 0;
                changed_flag = false;
            }} else {{
                result_ptr[out_len] = arg;
                out_len++;
                changed_flag = true;
            }}
        }} else if (cmd == '@') {{ // Delete all instances of char
            unsigned int temp_idx = 0;
            for(unsigned int i=0; i<word_len; i++) {{
                if (result_ptr[i] != arg) {{
                    result_ptr[temp_idx++] = result_ptr[i];
                }} else {{
                    changed_flag = true;
                }}
            }}
            out_len = temp_idx;
        }}
    }}
    
    if (changed_flag) {{
        if (out_len < max_output_len_padded) {{
             result_ptr[out_len] = 0; // Null terminator
        }}
    }} else {{
        // If the word was not changed or rule execution failed, zero out the output
        for (unsigned int i = 0; i < max_output_len_padded; i++) {{
            result_ptr[i] = 0;
        }}
    }}
}}
"""

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(
        description=f'{Colors.CYAN}High Performance GPU Rule Extraction with Depth Processing{Colors.END}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-w', '--wordlist', required=True, help='Path to the target wordlist file.')
    parser.add_argument('-b', '--base-wordlist', help='Optional path to a base wordlist file. If not specified, --wordlist is used as the base.')
    parser.add_argument('-d', '--depth', type=int, default=1, help='Number of rules to chain together. Default: 1.')
    parser.add_argument('--batch-size', type=int, default=50000, help='Number of words to process in each GPU batch. Default: 50000.')
    parser.add_argument('-o', '--output', required=True, help='File to save the extracted rules to.')
    
    parser.add_argument('-r', '--rules_file', type=str,
                         help='Path to a file containing external rules (one per line). Filters the internal rule set.')
    
    args = parser.parse_args()

    # Print banner
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}           HIGH PERFORMANCE GPU RULE EXTRACTION TOOL{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

    # Initialize OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    print(f"{green('✅')} {bold('GPU:')} {device.name}")
    print(f"{blue('💾')} {bold('VRAM:')} {device.global_mem_size // (1024**3)}GB")
    print(f"{blue('🔗')} {bold('Chain Depth:')} {cyan(args.depth)}")

    # Load target words
    print(f"{blue('📁')} {bold('Loading target wordlist...')}")
    target_words = load_wordlist_fast(args.wordlist)
    word_set = set(target_words)

    # Generate Full Rule Set
    print(f"{yellow('🔧')} {bold('Generating full rule set...')}")
    all_rules_reference = generate_all_rules()
    
    # Calculate Block IDs for the OpenCL kernel
    num_simple_rules = 10 
    num_td_rules = 20
    num_s_rules = len(string.digits + string.ascii_lowercase)**2 + len(generate_leetspeak_rules())
    num_a_rules = len(string.digits + string.ascii_letters + string.punctuation) * 3
    
    start_id_simple = 0
    start_id_TD = num_simple_rules
    start_id_s = start_id_TD + num_td_rules
    start_id_A = start_id_s + num_s_rules
    
    # Create the reference map: rule string -> original ID
    rule_id_map_reference = {rule: i for i, rule in enumerate(all_rules_reference)}
    
    # Rule Filtering
    if args.rules_file:
        print(f"{blue('📁')} {bold('Loading and filtering rules from external file:')} '{args.rules_file}'")
        try:
            if not os.path.exists(args.rules_file):
                print(f"{red('❌')} {bold('Error: External rules file')} '{args.rules_file}' {bold('not found. Exiting.')}")
                sys.exit(1)
                
            with open(args.rules_file, 'r', encoding='latin-1') as f:
                external_rules = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Filter: only accept rules that are implemented
            all_rules = [r for r in external_rules if r in rule_id_map_reference]
            
            if not all_rules:
                print(f"{yellow('⚠️')} {bold('Warning: No valid implemented rules found in the external file. Using all internal rules.')}")
                all_rules = all_rules_reference
            else:
                print(f"{green('✅')} {bold('Filtered')} {cyan(f'{len(all_rules):,}')} {bold('valid rules for GPU testing.')}")
            
        except Exception as e:
            print(f"{red('❌')} {bold('Error loading external rules file:')} {e} {bold('Using all internal rules.')}")
            all_rules = all_rules_reference
    else:
        print(f"{blue('🔧')} {bold('Using internal static rule generation.')}")
        all_rules = all_rules_reference

    print(f"{green('✅')} {bold('Loaded:')} {cyan(f'{len(target_words):,}')} {bold('target words,')} {cyan(f'{len(all_rules):,}')} {bold('rules')}")

    # Prepare rules for GPU using original method
    _, host_rules_gpu_temp, max_rule_len_padded = prepare_data_for_gpu([], all_rules, 0)
    
    # Overwrite sequential IDs with the correct reference IDs for the OpenCL kernel
    for i, rule in enumerate(all_rules):
        correct_id = rule_id_map_reference[rule]
        host_rules_gpu_temp[i, 0] = correct_id
        
    host_rules_gpu = host_rules_gpu_temp

    # Build kernel
    kernel_source = get_bfs_kernel_source(
        start_id_simple, num_simple_rules, start_id_TD, num_td_rules,
        start_id_s, num_s_rules, start_id_A, num_a_rules
    )
    program = cl.Program(context, kernel_source).build()
    kernel = program.bfs_kernel

    # Create GPU buffers
    mf = cl.mem_flags
    
    # Rules buffer
    rules_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_rules_gpu)

    # Process for each depth level
    total_hits = Counter()
    overall_start_time = time.time()
    total_words_processed = 0

    for current_depth in range(1, args.depth + 1):
        print(f"\n{magenta(f'🔍 Processing depth {current_depth}/{args.depth}...')}")
        
        # Load words for current depth
        if current_depth == 1:
            # Load base wordlist
            base_wordlist = args.base_wordlist if args.base_wordlist else args.wordlist
            base_words = load_wordlist_fast(base_wordlist)
            words_to_process = [(word, "") for word in base_words]
        else:
            # Load from previous depth
            words_to_process = load_words_for_depth(current_depth - 1)
            if not words_to_process:
                print(f"{yellow('⚠️')} {bold('No words for depth')} {cyan(current_depth)}")
                break
        
        print(f"{blue('📊')} {bold('Processing')} {cyan(f'{len(words_to_process):,}')} {bold('words at depth')} {cyan(current_depth)}")
        
        # Extract just the words for processing
        base_words = [word for word, chain in words_to_process]
        word_to_chain_map = {word: chain for word, chain in words_to_process}
        
        # Process in batches
        batch_size = min(args.batch_size, len(base_words))
        depth_hits = Counter()
        
        start_time = time.time()
        
        # Process in batches
        num_batches = (len(base_words) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(base_words))
            batch_words = base_words[start_idx:end_idx]
            batch_size_actual = len(batch_words)
            
            if batch_size_actual == 0:
                continue
            
            # Prepare batch for GPU
            max_word_len = max([len(word.encode('latin-1')) for word in batch_words] + [0]) + 1
            max_output_len_padded = max_word_len + current_depth * 10
            
            host_base_words_gpu, _, _ = prepare_data_for_gpu(batch_words, [], max_word_len)
            
            # Create batch buffer
            words_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_base_words_gpu)
            
            # Allocate result buffer
            global_size = batch_size_actual * len(all_rules)
            result_buf = cl.Buffer(context, mf.WRITE_ONLY, global_size * max_output_len_padded)
            
            # Initialize result buffer to zero
            cl.enqueue_fill_buffer(queue, result_buf, np.uint8(0), 0, result_buf.size).wait()
            
            # Execute kernel
            kernel.set_arg(0, words_buf)
            kernel.set_arg(1, rules_buf)
            kernel.set_arg(2, result_buf)
            kernel.set_arg(3, np.uint32(batch_size_actual))
            kernel.set_arg(4, np.uint32(len(all_rules)))
            kernel.set_arg(5, np.uint32(max_word_len))
            kernel.set_arg(6, np.uint32(max_rule_len_padded + 1))
            kernel.set_arg(7, np.uint32(max_output_len_padded))
            
            cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), None).wait()
            
            # Get results back from GPU
            host_results_flat = np.zeros(global_size * max_output_len_padded, dtype=np.uint8)
            cl.enqueue_copy(queue, host_results_flat, result_buf).wait()
            
            # Process results
            for j in range(batch_size_actual * len(all_rules)):
                start_idx = j * max_output_len_padded
                word_bytes = host_results_flat[start_idx:start_idx + max_output_len_padded].tobytes().split(b'\0', 1)[0]
                
                try:
                    transformed_word = word_bytes.decode('latin-1', 'ignore')
                except UnicodeDecodeError:
                    transformed_word = None
                    
                if transformed_word and transformed_word in word_set:
                    base_word_idx = j // len(all_rules)
                    rule_idx = j % len(all_rules)
                    base_word = batch_words[base_word_idx]
                    new_rule = all_rules[rule_idx]

                    # Get the previous rule chain and create the full chain
                    previous_chain = word_to_chain_map.get(base_word, "")
                    
                    # Create the full chain for the found hit: R1 R2... Rn
                    if previous_chain:
                        full_chain = f"{previous_chain} {new_rule}"
                    else:
                        full_chain = new_rule
                    
                    # Condition: ensure the word actually changed
                    if transformed_word != base_word:
                        # Count only unique hits for the FULL CHAIN
                        depth_hits[full_chain] += 1
            
            total_words_processed += batch_size_actual
            elapsed_time = time.time() - start_time
            words_per_sec = total_words_processed / elapsed_time if elapsed_time > 0 else 0
            
            print(f"{green('📊')} {bold('Batch')} {cyan(f'{batch_idx + 1}/{num_batches}')} - "
                  f"{cyan(f'{words_per_sec:,.0f}')} words/sec - "
                  f"{cyan(f'{len(depth_hits):,}')} rules found")
            
            # Cleanup
            words_buf.release()
            result_buf.release()
        
        # Save results for this depth
        total_hits.update(depth_hits)
        
        # Prepare words for next depth if needed
        if current_depth < args.depth and depth_hits:
            next_depth_words = []
            # For each word and each rule that hit, create new combinations
            for word, chain in words_to_process:
                for rule in depth_hits:
                    new_chain = f"{chain} {rule}".strip() if chain else rule
                    next_depth_words.append((word, new_chain))
            
            if next_depth_words:
                save_words_for_next_depth(next_depth_words, current_depth)
                print(f"{blue('💾')} {bold('Saved')} {cyan(f'{len(next_depth_words):,}')} {bold('words for depth')} {cyan(current_depth + 1)}")

    total_time = time.time() - overall_start_time
    
    print(f"\n{green('='*60)}")
    print(f"{bold('🎉 EXTRACTION COMPLETE')}")
    print(f"{green('='*60)}")
    print(f"{blue('📊')} {bold('Total Rules Found:')} {cyan(f'{len(total_hits):,}')}")
    print(f"{blue('⚡')} {bold('Performance:')} {cyan(f'{total_words_processed/total_time:,.0f}')} words/sec")
    print(f"{blue('🔗')} {bold('Chain Depth:')} {cyan(args.depth)}")
    print(f"{green('='*60)}")
    
    # Save results
    with open(args.output, 'w') as f:
        for rule, count in total_hits.most_common():
            f.write(f"{rule}\n")
    
    print(f"{green('💾')} {bold('Results saved to:')} {args.output}")
    
    # Cleanup temporary files
    cleanup_temp_files(args.depth)

if __name__ == '__main__':
    main()
