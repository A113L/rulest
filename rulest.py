import os
import sys
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
from collections import Counter
from tqdm import tqdm

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
        new_rules.append(f"' {n}")  # Note: single quote needs special handling
    
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

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description='OpenCL GPU Wrapper for rule extraction. Supports external rule files and rule chaining (BFS).')
    parser.add_argument('-w', '--wordlist', required=True, help='Path to the single wordlist file (target dictionary).')
    parser.add_argument('-b', '--base-wordlist', help='Optional path to a base wordlist file. If not specified, --wordlist is used as the base.')
    parser.add_argument('-d', '--chain-depth', type=int, default=1, help='Number of rules to chain together. Default: 1.')
    parser.add_argument('--batch-size', type=int, default=5000, help='Number of words to process in each GPU batch. Default: 5000.')
    parser.add_argument('-o', '--output', help='File to save the extracted rules to.')
    
    parser.add_argument('-r', '--rules_file', type=str,
                         help='Path to a file containing external rules (one per line). Filters the internal rule set.')
    
    args = parser.parse_args()

    # --- Generate Full Rule Set and Calculate Block IDs ---
    # We ALWAYS generate the full set to know the correct reference IDs
    all_rules_reference = generate_all_rules()

    # Calculate Block IDs for the OpenCL kernel
    num_simple_rules = 10 
    num_td_rules = 20
    num_s_rules = len(string.digits + string.ascii_lowercase)**2 + len(generate_leetspeak_rules())
    num_a_rules = len(string.digits + string.ascii_letters + string.punctuation) * 3
    
    # Calculate new Group B rules count
    num_groupB_rules = (
        10 +  # pN (0-9)
        4 +   # {, }, [, ]
        90 +  # xNM (10*9)
        90 +  # ONM (10*9) 
        160 + # iNX (10*16)
        160 + # oNX (10*16)
        10 +  # 'N (0-9)
        10 +  # zN (0-9)
        10 +  # ZN (0-9)
        1     # q
    )
    
    # Calculate comprehensive rules count
    num_comprehensive_rules = (
        1 +   # K
        90 +  # *NM (10*10 - 10 same positions)
        10 +  # LN
        10 +  # RN
        10 +  # +N
        10 +  # -N
        10 +  # .N
        10 +  # ,N
        9 +   # yN (1-9)
        9 +   # YN (1-9)
        1 +   # E
        5 +   # eX
        30    # 3NX (5 separators * 6 N values)
    )
    
    start_id_simple = 0
    start_id_TD = num_simple_rules
    start_id_s = start_id_TD + num_td_rules
    start_id_A = start_id_s + num_s_rules
    start_id_groupB = start_id_A + num_a_rules
    start_id_comprehensive = start_id_groupB + num_groupB_rules
    
    # Create the reference map: rule string -> original ID
    rule_id_map_reference = {rule: i for i, rule in enumerate(all_rules_reference)}

    # --- Rule Filtering ---
    if args.rules_file:
        print(f"Loading and filtering rules from external file: '{args.rules_file}'")
        try:
            if not os.path.exists(args.rules_file):
                print(f"Error: External rules file '{args.rules_file}' not found. Exiting.")
                sys.exit(1)
                
            # FIX: Use 'latin-1' encoding instead of 'utf-8' to prevent decoding errors
            with open(args.rules_file, 'r', encoding='latin-1') as f:
                external_rules = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Filter: only accept rules that are implemented in the kernel
            all_rules = [r for r in external_rules if r in rule_id_map_reference]
            
            if not all_rules:
                print("Error: No valid implemented rules found in the external file. Falling back to all internal rules.")
                all_rules = all_rules_reference
            else:
                print(f"Filtered {len(all_rules)} valid rules for GPU testing.")
            
        except Exception as e:
            print(f"Error loading external rules file: {e}. Falling back to all internal rules.")
            all_rules = all_rules_reference
    else:
        print("Using internal static rule generation.")
        all_rules = all_rules_reference
    # --- End Rule Filtering ---

    # --- Prepare Rule Buffer for GPU ---
    # 1. Prepare data structures for the filtered set
    _, host_rules_gpu_temp, max_rule_len_padded = prepare_data_for_gpu([], all_rules, 0)
    
    # 2. Overwrite sequential IDs with the correct reference IDs for the OpenCL kernel
    for i, rule in enumerate(all_rules):
        correct_id = rule_id_map_reference[rule]
        host_rules_gpu_temp[i, 0] = correct_id # Set the ID expected by the OpenCL kernel
        
    host_rules_gpu = host_rules_gpu_temp # Final buffer to send

    # --- OpenCL Kernel Source (Implemented Hashcat Rules) ---
    kernel_source = f"""
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
    unsigned int start_id_groupB = {start_id_groupB};
    unsigned int end_id_groupB = start_id_groupB + {num_groupB_rules};
    unsigned int start_id_comprehensive = {start_id_comprehensive};
    unsigned int end_id_comprehensive = start_id_comprehensive + {num_comprehensive_rules};
    
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
        
        unsigned int pos_to_change = char_to_pos(pos_char);
        
        if (operator_char == 'T') {{ // 'T' (toggle case at pos)
            out_len = word_len;
            for (unsigned int i = 0; i < word_len; i++) {{
                result_ptr[i] = current_word_ptr[i];
            }}
            if (pos_to_change != 0xFFFFFFFF && pos_to_change < word_len) {{
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
            if (pos_to_change != 0xFFFFFFFF && pos_to_change < word_len) {{
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
    // --- START GROUP B RULES ---
    else if (rule_id >= start_id_groupB && rule_id < end_id_groupB) {{ 
        
        // Default to copying the word for modification
        for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
        out_len = word_len;

        unsigned char cmd = rule_ptr[0];
        unsigned int N = (rule_ptr[1] != 0) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
        unsigned int M = (rule_ptr[2] != 0) ? char_to_pos(rule_ptr[2]) : 0xFFFFFFFF;
        unsigned char X = (rule_ptr[2] != 0) ? rule_ptr[2] : 0; // for i/o rules

        if (cmd == 'p') {{ // 'p' (Duplicate N times)
            if (N != 0xFFFFFFFF) {{
                unsigned int num_dupes = N;
                unsigned int total_len = word_len * (num_dupes + 1); 

                if (total_len >= max_output_len_padded || num_dupes == 0) {{
                    out_len = 0; 
                }} else {{
                    for (unsigned int j = 1; j <= num_dupes; j++) {{
                        unsigned int offset = word_len * j;
                        for (unsigned int i = 0; i < word_len; i++) {{
                            result_ptr[offset + i] = current_word_ptr[i];
                        }}
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }}
            }}
        }} 
        
        else if (cmd == 'q') {{ // 'q' (Duplicate all characters)
            unsigned int total_len = word_len * 2;
            if (total_len >= max_output_len_padded) {{
                out_len = 0;
            }} else {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    result_ptr[i * 2] = current_word_ptr[i];
                    result_ptr[i * 2 + 1] = current_word_ptr[i];
                }}
                out_len = total_len;
                changed_flag = true;
            }}
        }}

        else if (cmd == '{{') {{ // '{{' (Rotate Left)
            if (word_len > 0) {{
                unsigned char first_char = current_word_ptr[0];
                for (unsigned int i = 0; i < word_len - 1; i++) {{
                    result_ptr[i] = current_word_ptr[i + 1];
                }}
                result_ptr[word_len - 1] = first_char;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == '}}') {{ // '}}' (Rotate Right)
            if (word_len > 0) {{
                unsigned char last_char = current_word_ptr[word_len - 1];
                for (unsigned int i = word_len - 1; i > 0; i--) {{
                    result_ptr[i] = current_word_ptr[i - 1];
                }}
                result_ptr[0] = last_char;
                changed_flag = true;
            }}
        }}
        
        else if (cmd == '[') {{ // '[' (Truncate Left / Delete first char)
            if (word_len > 0) {{
                for (unsigned int i = 0; i < word_len - 1; i++) {{
                    result_ptr[i] = current_word_ptr[i + 1];
                }}
                out_len = word_len - 1;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == ']') {{ // ']' (Truncate Right / Delete last char)
            if (word_len > 0) {{
                // Word already copied up to word_len - 1
                out_len = word_len - 1;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == 'x') {{ // 'xNM' (Extract range, N=start, M=length)
            unsigned int start = N;
            unsigned int length = M;
            
            if (start != 0xFFFFFFFF && length != 0xFFFFFFFF && start < word_len && length > 0) {{
                unsigned int end = start + length;
                if (end > word_len) end = word_len;
                
                out_len = 0;
                for (unsigned int i = start; i < end; i++) {{
                    result_ptr[out_len++] = current_word_ptr[i];
                }}
                changed_flag = true;
            }} else {{
                // Invalid range results in an empty word
                out_len = 0; 
            }}
        }}
        
        else if (cmd == 'O') {{ // 'ONM' (Omit range, N=start, M=length)
            unsigned int start = N;
            unsigned int length = M;
            
            if (start != 0xFFFFFFFF && length != 0xFFFFFFFF && length > 0) {{
                unsigned int skip_start = (start < word_len) ? start : word_len;
                unsigned int skip_end = (skip_start + length < word_len) ? skip_start + length : word_len;
                
                out_len = 0;
                for (unsigned int i = 0; i < word_len; i++) {{
                    if (i < skip_start || i >= skip_end) {{
                        result_ptr[out_len++] = current_word_ptr[i];
                    }} else {{
                        changed_flag = true;
                    }}
                }}
            }}
        }}

        else if (cmd == 'i') {{ // 'iNX' (Insert char X at position N)
            unsigned int pos = N;
            unsigned char insert_char = X;

            if (pos != 0xFFFFFFFF && word_len + 1 < max_output_len_padded) {{
                unsigned int final_pos = (pos > word_len) ? word_len : pos;
                out_len = word_len + 1;

                // Copy and shift
                unsigned int current_idx = 0;
                for (unsigned int i = 0; i < out_len; i++) {{
                    if (i == final_pos) {{
                        result_ptr[i] = insert_char;
                    }} else {{
                        result_ptr[i] = current_word_ptr[current_idx++];
                    }}
                }}
                changed_flag = true;
            }} else {{
                out_len = 0;
            }}
        }}

        else if (cmd == 'o') {{ // 'oNX' (Overwrite char at position N with X)
            unsigned int pos = N;
            unsigned char new_char = X;

            if (pos != 0xFFFFFFFF && pos < word_len) {{
                result_ptr[pos] = new_char;
                changed_flag = true;
            }}
        }}
        
        else if (cmd == '\\'') {{ // "'N" (Truncate at position N)
            unsigned int pos = N;
            
            if (pos != 0xFFFFFFFF && pos < word_len) {{
                out_len = pos;
                changed_flag = true;
            }} 
        }}

        else if (cmd == 'z') {{ // 'zN' (Duplicate first char N times)
            unsigned int num_dupes = N;
            if (num_dupes != 0xFFFFFFFF && num_dupes > 0) {{
                unsigned int total_len = word_len + num_dupes;
                if (total_len < max_output_len_padded) {{
                    unsigned char first_char = current_word_ptr[0];
                    unsigned int out_idx = 0;
                    
                    // 1. Write duplicates
                    for (unsigned int i = 0; i < num_dupes; i++) {{
                        result_ptr[out_idx++] = first_char;
                    }}
                    // 2. Append original word
                    for (unsigned int i = 0; i < word_len; i++) {{
                        result_ptr[out_idx++] = current_word_ptr[i];
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }} else {{
                    out_len = 0;
                }}
            }}
        }}

        else if (cmd == 'Z') {{ // 'ZN' (Duplicate last char N times)
            unsigned int num_dupes = N;
            if (num_dupes != 0xFFFFFFFF && num_dupes > 0) {{
                unsigned int total_len = word_len + num_dupes;
                if (total_len < max_output_len_padded) {{
                    unsigned char last_char = current_word_ptr[word_len - 1];
                    
                    // Copy original word first (it was already copied at the start of this block)
                    unsigned int out_idx = word_len;
                    
                    // Append duplicates
                    for (unsigned int i = 0; i < num_dupes; i++) {{
                        result_ptr[out_idx++] = last_char;
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }} else {{
                    out_len = 0;
                }}
            }}
        }}

    }}
    // --- END GROUP B RULES ---
    
    // --- START COMPREHENSIVE RULES ---
    else if (rule_id >= start_id_comprehensive && rule_id < end_id_comprehensive) {{ 
        
        // Default to copying the word for modification
        for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
        out_len = word_len;

        unsigned char cmd = rule_ptr[0];
        unsigned int N = (rule_ptr[1] != 0) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
        unsigned int M = (rule_ptr[2] != 0) ? char_to_pos(rule_ptr[2]) : 0xFFFFFFFF;
        unsigned char X = (rule_ptr[2] != 0) ? rule_ptr[2] : 0;
        unsigned char separator = rule_ptr[1];

        if (cmd == 'K') {{ // 'K' (Swap last two characters)
            if (word_len >= 2) {{
                result_ptr[word_len - 1] = current_word_ptr[word_len - 2];
                result_ptr[word_len - 2] = current_word_ptr[word_len - 1];
                changed_flag = true;
            }}
        }}
        else if (cmd == '*') {{ // '*NM' (Swap character at position N with character at position M)
            if (N != 0xFFFFFFFF && M != 0xFFFFFFFF && N < word_len && M < word_len && N != M) {{
                unsigned char temp = result_ptr[N];
                result_ptr[N] = result_ptr[M];
                result_ptr[M] = temp;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'L') {{ // 'LN' (Bitwise shift left character @ N)
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] << 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'R') {{ // 'RN' (Bitwise shift right character @ N)
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] >> 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '+') {{ // '+N' (ASCII increment character @ N by 1)
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] + 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '-') {{ // '-N' (ASCII decrement character @ N by 1)
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] - 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '.') {{ // '.N' (Replace character @ N with value at @ N plus 1)
            if (N != 0xFFFFFFFF && N + 1 < word_len) {{
                result_ptr[N] = current_word_ptr[N + 1];
                changed_flag = true;
            }}
        }}
        else if (cmd == ',') {{ // ',N' (Replace character @ N with value at @ N minus 1)
            if (N != 0xFFFFFFFF && N > 0 && N < word_len) {{
                result_ptr[N] = current_word_ptr[N - 1];
                changed_flag = true;
            }}
        }}
        else if (cmd == 'y') {{ // 'yN' (Duplicate first N characters)
            if (N != 0xFFFFFFFF && N > 0 && N <= word_len) {{
                unsigned int total_len = word_len + N;
                if (total_len < max_output_len_padded) {{
                    // Shift original word right by N positions
                    for (int i = word_len - 1; i >= 0; i--) {{
                        result_ptr[i + N] = result_ptr[i];
                    }}
                    // Duplicate first N characters at the beginning
                    for (unsigned int i = 0; i < N; i++) {{
                        result_ptr[i] = current_word_ptr[i];
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }}
            }}
        }}
        else if (cmd == 'Y') {{ // 'YN' (Duplicate last N characters)
            if (N != 0xFFFFFFFF && N > 0 && N <= word_len) {{
                unsigned int total_len = word_len + N;
                if (total_len < max_output_len_padded) {{
                    // Append last N characters
                    for (unsigned int i = 0; i < N; i++) {{
                        result_ptr[word_len + i] = current_word_ptr[word_len - N + i];
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }}
            }}
        }}
        else if (cmd == 'E') {{ // 'E' (Title case)
            // First lowercase everything
            for (unsigned int i = 0; i < word_len; i++) {{
                unsigned char c = current_word_ptr[i];
                if (c >= 'A' && c <= 'Z') {{
                    result_ptr[i] = c + 32;
                }} else {{
                    result_ptr[i] = c;
                }}
            }}
            
            // Then uppercase first letter and letters after spaces
            bool capitalize_next = true;
            for (unsigned int i = 0; i < word_len; i++) {{
                if (capitalize_next && result_ptr[i] >= 'a' && result_ptr[i] <= 'z') {{
                    result_ptr[i] = result_ptr[i] - 32;
                    changed_flag = true;
                }}
                capitalize_next = (result_ptr[i] == ' ');
            }}
            out_len = word_len;
        }}
        else if (cmd == 'e') {{ // 'eX' (Title case with custom separator)
            // First lowercase everything
            for (unsigned int i = 0; i < word_len; i++) {{
                unsigned char c = current_word_ptr[i];
                if (c >= 'A' && c <= 'Z') {{
                    result_ptr[i] = c + 32;
                }} else {{
                    result_ptr[i] = c;
                }}
            }}
            
            // Then uppercase first letter and letters after custom separator
            bool capitalize_next = true;
            for (unsigned int i = 0; i < word_len; i++) {{
                if (capitalize_next && result_ptr[i] >= 'a' && result_ptr[i] <= 'z') {{
                    result_ptr[i] = result_ptr[i] - 32;
                    changed_flag = true;
                }}
                capitalize_next = (result_ptr[i] == separator);
            }}
            out_len = word_len;
        }}
        else if (cmd == '3') {{ // '3NX' (Toggle case after Nth instance of separator char)
            unsigned int separator_count = 0;
            unsigned int target_count = N;
            unsigned char sep_char = X;
            
            if (target_count != 0xFFFFFFFF) {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    if (current_word_ptr[i] == sep_char) {{
                        separator_count++;
                        if (separator_count == target_count && i + 1 < word_len) {{
                            // Toggle the case of the character after the separator
                            unsigned char c = current_word_ptr[i + 1];
                            if (c >= 'a' && c <= 'z') {{
                                result_ptr[i + 1] = c - 32;
                                changed_flag = true;
                            }} else if (c >= 'A' && c <= 'Z') {{
                                result_ptr[i + 1] = c + 32;
                                changed_flag = true;
                            }}
                            break;
                        }}
                    }}
                }}
            }}
        }}
    }}
    // --- END COMPREHENSIVE RULES ---
    
    // Final output processing
    if (changed_flag && out_len > 0) {{
        if (out_len < max_output_len_padded) {{
                   result_ptr[out_len] = 0; // Null terminator
        }}
    }} else {{
        // If the word was not changed or rule execution failed/resulted in length 0, zero out the output
        for (unsigned int i = 0; i < max_output_len_padded; i++) {{
            result_ptr[i] = 0;
        }}
    }}
}}
"""
    # --- End OpenCL Kernel Source ---
    
    # [Rest of your existing main function remains unchanged...]
    # Check for OpenCL context and queue
    ctx = None
    try:
        platforms = cl.get_platforms()
        chosen_platform = platforms[0]
        chosen_device = chosen_platform.get_devices()[0]
        ctx = cl.Context([chosen_device])
        queue = cl.CommandQueue(ctx)
        print(f"Selected device: {chosen_device.name}")
    except (IndexError, cl.Error) as e:
        print(f"ERROR: Cannot create context for the selected device. Error: {e}")
        return

    def load_data(filename):
        """Loads words from a file using 'latin-1' encoding."""
        if not os.path.exists(filename):
            print(f"Error: The file '{filename}' does not exist.")
            return None
        try:
            # Using 'latin-1' for robust wordlist reading
            with open(filename, 'r', encoding='latin-1') as f:
                return [line.strip().split()[0] for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            print(f"An error occurred while loading the file '{filename}': {e}")
            return None

    wordlist = load_data(args.wordlist)
    if wordlist is None:
        return
    
    print(f"Loaded {len(wordlist)} words from '{args.wordlist}' as the target dictionary.")
    
    word_set = set(wordlist)
    
    # Send the prepared rules buffer to the GPU
    mf = cl.mem_flags
    rules_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_rules_gpu)
    
    prg = cl.Program(ctx, kernel_source).build()
    bfs_kernel_func = prg.bfs_kernel

    extracted_rules_with_hits = Counter()

    base_wordlist = load_data(args.base_wordlist if args.base_wordlist else args.wordlist)
    if base_wordlist is None:
        return
    
    try:
        if args.base_wordlist:
            print(f"Loaded {len(base_wordlist)} words from '{args.base_wordlist}' as the base wordlist.")
        else:
            print(f"Loaded {len(wordlist)} words from '{args.wordlist}' as the base wordlist.")
    except Exception as e:
        print(f"Error during wordlist loading: {e}")
        return

    initial_batch_size = args.batch_size
    current_batch_size = initial_batch_size
    
    # words_with_chains: list of (word, chain_of_rules_so_far)
    words_with_chains = [(word, "") for word in base_wordlist] 
    
    for current_depth in range(1, args.chain_depth + 1):
        print(f"\nProcessing depth {current_depth}/{args.chain_depth}...")
        
        words_to_process_file_in = f"words_to_process_d{current_depth}.tmp"
        words_to_process_file_out = f"words_to_process_d{current_depth+1}.tmp"
        
        # --- START MODIFIED WORD LOADING LOGIC FOR D > 1 ---
        if current_depth > 1 and os.path.exists(words_to_process_file_in):
            words_with_chains = [] # Reset for the new depth
            print(f"Loading words and previous chains from temporary file: {words_to_process_file_in}")
            try:
                with open(words_to_process_file_in, 'r', encoding='latin-1') as f:
                    for line in f:
                        parts = line.strip().split('\t', 1)
                        if len(parts) == 2 and parts[0]:
                            # parts[0] = transformed_word, parts[1] = previous_chain
                            words_with_chains.append((parts[0], parts[1]))
            except Exception as e:
                print(f"Error reading temporary file: {e}. Stopping.")
                break

        # Words to process in this batch (only words, GPU does not see chains)
        words_to_process = [word for word, chain in words_with_chains]
        # Map to combine chains AFTER GPU processing
        word_to_chain_map = {word: chain for word, chain in words_with_chains}

        # --- END MODIFIED WORD LOADING LOGIC ---
        
        if not words_to_process:
            if current_depth > 1:
                print("No words to process for the next depth. Stopping.")
            break

        num_words_total = len(words_to_process)
        # Max word length + 1 for null terminator
        max_word_len = max([len(word.encode('latin-1')) for word in words_to_process] + [0]) + 1
        # Max possible output length (e.g., from 'd' - duplicate) - adjusted for some growth
        max_output_len_padded = max_word_len + current_depth * 10
        
        num_rules = len(all_rules) 
        current_batch_size = initial_batch_size
        
        new_found_words = [] # Tracks unique words for the next depth
        unique_next_depth_words = set()

        with open(words_to_process_file_out, 'w', encoding='latin-1') as f_out, \
             tqdm(total=num_words_total, unit='words') as pbar:

            i = 0
            while i < num_words_total:
                batch_words = words_to_process[i:i + current_batch_size]
                num_words_batch = len(batch_words)
                global_size = num_words_batch * num_rules
                
                try:
                    host_base_words_gpu, _, _ = prepare_data_for_gpu(batch_words, [], max_word_len)
                    words_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_base_words_gpu)
                    # Allocate result buffer: global_size * max_output_len_padded
                    result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, global_size * max_output_len_padded)
                    
                    # Initialize result buffer to zero
                    cl.enqueue_fill_buffer(queue, result_buf, np.uint8(0), 0, result_buf.size).wait()
                    
                    # Launch kernel
                    bfs_kernel_func(queue, (global_size,), None,
                                     words_buf, rules_buf, result_buf,
                                     np.uint32(num_words_batch),
                                     np.uint32(num_rules), 
                                     np.uint32(max_word_len),
                                     np.uint32(max_rule_len_padded + 1), 
                                     np.uint32(max_output_len_padded))
                                     
                    # Get results back from GPU
                    host_results_flat = np.zeros(global_size * max_output_len_padded, dtype=np.uint8)
                    cl.enqueue_copy(queue, host_results_flat, result_buf).wait()
                    
                    for j in range(num_words_batch * num_rules):
                        # Process results and count hits
                        start_idx = j * max_output_len_padded
                        # Extract word bytes up to the null terminator
                        word_bytes = host_results_flat[start_idx:start_idx + max_output_len_padded].tobytes().split(b'\0', 1)[0]
                        
                        try:
                            transformed_word = word_bytes.decode('latin-1', 'ignore')
                        except UnicodeDecodeError:
                            transformed_word = None
                            
                        # --- START MODIFIED RESULT LOGGING LOGIC ---
                        if transformed_word and transformed_word in word_set:
                            base_word_idx = j // num_rules
                            rule_idx = j % num_rules
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
                                extracted_rules_with_hits[full_chain] += 1
                                
                                # Check if the word should be used in the next depth
                                if transformed_word not in unique_next_depth_words:
                                    unique_next_depth_words.add(transformed_word)
                                    # Write the word and RULE CHAIN to the temporary file
                                    # Format: transformed_word\tfull_chain_for_next_step
                                    f_out.write(f"{transformed_word}\t{full_chain}\n")
                        # --- END MODIFIED RESULT LOGGING LOGIC ---
                                        
                    words_buf.release()
                    result_buf.release()
                    pbar.update(num_words_batch)
                    i += num_words_batch
                    current_batch_size = initial_batch_size 
                        
                except cl.MemoryError:
                    print(f"\n[Warning] Memory allocation failed for batch size {current_batch_size}. Retrying with smaller batch size.")
                    current_batch_size //= 2
                    if current_batch_size == 0:
                        print("ERROR: Failed to allocate memory even for the smallest batch size. Stopping.")
                        return

        # Clean up temporary file from the current depth
        if current_depth > 1 and os.path.exists(words_to_process_file_in):
            os.remove(words_to_process_file_in)
            
        # Prepare words for processing at the next depth (they will be re-read at the start of the next loop)
        words_with_chains = [] # Will be reloaded at the start of the next loop
        
    print(f"\nGPU-based extraction finished.")
    
    sorted_rules = extracted_rules_with_hits.most_common()

    print(f"Total unique rules chains extracted: {len(sorted_rules)}")

    if args.output:
        print(f"\nSaving extracted rule chains to '{args.output}'...")
        try:
            # Save the final rule chains
            with open(args.output, 'w', encoding='utf-8') as f:
                for full_chain, count in sorted_rules:
                    f.write(f"{full_chain}\n")
            print("Done.")
        except Exception as e:
            print(f"Error: Could not save rules to file '{args.output}'. Error: {e}")

    # Remove all temporary files
    for i in range(1, args.chain_depth + 2):
        temp_file = f"words_to_process_d{i}.tmp"
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
