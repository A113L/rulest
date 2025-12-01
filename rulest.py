# OpenCL GPU-accelerated rule extraction tool using BFS chaining to find full rule sequences (R1 R2...) that transform base words into target dictionary words.
import os
import sys
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
import re
from collections import Counter
from tqdm import tqdm
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# --- Color Definitions ---
COLOR_TITLE = Fore.CYAN + Style.BRIGHT
COLOR_SUCCESS = Fore.GREEN
COLOR_WARNING = Fore.YELLOW
COLOR_ERROR = Fore.RED
COLOR_INFO = Fore.BLUE
COLOR_HIGHLIGHT = Fore.MAGENTA
COLOR_DEBUG = Fore.WHITE + Style.DIM

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

def generate_fast_rules():
    """Generates the FAST set of rules (~1,188 rules - original working version)."""
    rules = []
    
    # Simple rules (ID 0-9): lowercase, uppercase, capitalize, inverse capitalize, toggle case, reverse, swap first two, identity, duplicate, reflect
    simple_rules = ['l', 'u', 'c', 'C', 't', 'r', 'k', ':', 'd', 'f']
    rules.extend(simple_rules)
    
    # T and D rules (ID 10-29): toggle case at position i, delete character at position i
    for i in range(10):
        rules.extend([f'T{i}', f'D{i}'])
    
    # s rules (ID 30-...): substitution rules (digits + lowercase ONLY - like original)
    chars_2arg = string.digits + string.ascii_lowercase
    for c1 in chars_2arg:
        for c2 in chars_2arg:
            rules.append(f's{c1}{c2}')
            
    # Add leetspeak rules
    rules.extend(generate_leetspeak_rules())

    # Group A rules (ID ...-END): prepend (^), append ($), delete all instances (@)
    chars = string.digits + string.ascii_letters + string.punctuation
    for c in chars:
        rules.extend([f'^{c}', f'${c}', f'@{c}'])
    
    return rules

def generate_comprehensive_rules():
    """Generates the COMPREHENSIVE set of rules (~16,000+ rules)."""
    rules = []
    
    # =================== SIMPLE RULES (ID 0-9) ===================
    # l, u, c, C, t, r, k, :, d, f
    simple_rules = ['l', 'u', 'c', 'C', 't', 'r', 'k', ':', 'd', 'f']
    rules.extend(simple_rules)
    
    # =================== T/D RULES (ID 10-29) ===================
    # T0-T9, D0-D9
    for i in range(10):
        rules.append(f'T{i}')
        rules.append(f'D{i}')
    
    # =================== S RULES (ID 30-...) ===================
    # sXY substitution rules (all character combinations)
    chars_for_substitution = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation
    for c1 in chars_for_substitution:
        for c2 in chars_for_substitution:
            if c1 != c2:  # Skip identity substitutions
                rules.append(f's{c1}{c2}')
    
    # Add leetspeak rules as special s rules
    rules.extend(generate_leetspeak_rules())

    # =================== GROUP A RULES (ID ...) ===================
    # ^X, $X, @X
    chars_for_groupA = string.digits + string.ascii_letters + string.punctuation
    for c in chars_for_groupA:
        rules.extend([f'^{c}', f'${c}', f'@{c}'])
    
    # =================== GROUP B RULES (ID ...) ===================
    # p, {, }, [, ], x, O, i, o, ', z, Z, q
    groupB_rules = ['p', '{', '}', '[', ']']
    rules.extend(groupB_rules)
    
    # xNM rules (extract range)
    for i in range(10):
        for j in range(10):
            if i <= j:
                rules.append(f'x{i}{j}')
    
    # O, i, o rules with positions and characters
    for i in range(10):
        for c in chars_for_groupA:
            rules.extend([f'O{i}{c}', f'i{i}{c}', f'o{i}{c}'])
    
    # ' (increment) rules
    for i in range(10):
        rules.append(f"' {i}")  # Note: space after quote to differentiate
    
    # z, Z, q rules
    rules.extend(['z', 'Z', 'q'])
    
    # =================== NEW RULES (ID ...) ===================
    # K, *NM, LN, RN, +N, -N, .N, ,N, yN, YN, E, eX, 3NX
    rules.append('K')  # Swap last two
    
    # *NM swap rules
    for i in range(10):
        for j in range(10):
            if i != j:
                rules.append(f'*{i}{j}')
    
    # LN, RN, +N, -N, .N, ,N rules
    for i in range(10):
        rules.extend([f'L{i}', f'R{i}', f'+{i}', f'-{i}', f'.{i}', f',{i}'])
    
    # yN, YN rules (duplicate first/last N)
    for i in range(1, 11):  # 1-10
        rules.extend([f'y{i}', f'Y{i}'])
    
    # E rule
    rules.append('E')
    
    # eX rules (title case with separator)
    for c in chars_for_groupA:
        rules.append(f'e{c}')
    
    # 3NX rules (toggle case after Nth separator)
    for i in range(1, 11):  # 1-10
        for c in chars_for_groupA:
            rules.append(f'3{i}{c}')
    
    # =================== INSERT EVERY RULES (vNX) ===================
    # vNX rules (insert character X every N characters)
    for i in range(10):  # N = 0-9
        for c in chars_for_groupA:
            rules.append(f'v{i}{c}')
    
    # =================== ADDITIONAL HASHCAT RULES ===================
    # Memory rules: M, 4, 6, _, '
    rules.extend(['M', '4', '6', '_', "'"])
    
    # Rejection rules: !X, /X
    for c in chars_for_groupA:
        rules.extend([f'!{c}', f'/{c}'])
    
    # Length rules: <N, >N, (N, )N
    for i in range(1, 21):  # Lengths 1-20
        rules.extend([f'<{i}', f'>{i}', f'({i}', f'){i}'])
    
    # Toggle range: TIJ (where I < J)
    for i in range(10):
        for j in range(i+1, 10):
            rules.append(f'T{i}{j}')
    
    # Purge all: pX (different from pluralize 'p')
    for c in chars_for_groupA:
        rules.append(f'p{c}')
    
    # Logical rules: ?NX, =NX
    for i in range(10):
        for c in chars_for_groupA:
            rules.extend([f'?{i}{c}', f'={i}{c}'])
    
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

# --- Rule Statistics and Documentation ---
def print_rule_statistics(rules, mode="FAST"):
    """Print statistics about the implemented rules."""
    print(f"\n{COLOR_TITLE}=== RULE STATISTICS ({mode} MODE) ===")
    print(f"{COLOR_INFO}Total rules generated: {len(rules)}")
    
    if mode == "FAST":
        categories = {
            'Simple Rules': ['l', 'u', 'c', 'C', 't', 'r', 'k', ':', 'd', 'f'],
            'T/D Rules': [f'T{i}' for i in range(10)] + [f'D{i}' for i in range(10)],
            'Substitution Rules': [r for r in rules if r.startswith('s') and len(r) == 3],
            'Group A Rules': [r for r in rules if r[0] in '^$@' and len(r) == 2],
        }
    else:  # COMPREHENSIVE
        categories = {
            'Simple Rules': ['l', 'u', 'c', 'C', 't', 'r', 'k', ':', 'd', 'f'],
            'Position T/D': ['T[0-9]$', 'D[0-9]$'],
            'Substitution': ['s..'],
            'Append/Prepend/Delete': ['^.', '$.', '@.'],
            'Group B': ['p$', '{', '}', '\\[', '\\]', 'x..', 'O..', 'i..', 'o..', "'", 'z', 'Z', 'q'],
            'New Rules': ['K', '\\*..', 'L.', 'R.', '\\+.', '\\-.', '\\..', ',.', 'y.', 'Y.', 'E', 'e.', '3..'],
            'Insert Every': ['v..'],
            'Memory': ['M', '4', '6', '_', "'"],
            'Rejection': ['!.', '/.'],
            'Length': ['<.', '>.', '\\(.', '\\)\.'],
            'Toggle Range': ['T..'],
            'Purge': ['p.$'],
            'Logical': ['\\?..', '=..']
        }
    
    total_counted = 0
    for category, patterns in categories.items():
        if isinstance(patterns, list) and patterns and isinstance(patterns[0], str) and not patterns[0].endswith('.'):
            # Simple list of rules
            count = len([r for r in rules if r in patterns])
        else:
            # Regex patterns
            cat_rules = []
            for pattern in patterns:
                regex = re.compile(f'^{pattern}$')
                cat_rules.extend([r for r in rules if regex.match(r)])
            count = len(set(cat_rules))  # Remove duplicates
        total_counted += count
        print(f"  {COLOR_HIGHLIGHT}{category:25} {COLOR_SUCCESS}{count:5} rules")
    
    print(f"\n{COLOR_INFO}Memory estimate: {COLOR_WARNING}{len(rules) * 20 / 1024:.1f} KB rule data")

# --- Enhanced Help Display ---
def print_colored_help():
    """Print beautiful colored help message."""
    script_name = os.path.basename(sys.argv[0])
    
    print(f"\n{COLOR_TITLE}╔{'═'*70}╗")
    print(f"║{'GPU-ACCELERATED HASHCAT RULE EXTRACTOR':^70}║")
    print(f"╚{'═'*70}╝{Style.RESET_ALL}\n")
    
    print(f"{COLOR_INFO}DESCRIPTION:")
    print(f"  Blazing-fast OpenCL GPU rule extraction with BFS chaining.")
    print(f"  Finds rule sequences that transform base words into target words.\n")
    
    print(f"{COLOR_SUCCESS}USAGE:")
    print(f"  {script_name} -w WORDLIST [OPTIONS]\n")
    
    print(f"{COLOR_HIGHLIGHT}REQUIRED ARGUMENTS:")
    print(f"  {COLOR_SUCCESS}-w, --wordlist WORDLIST{COLOR_INFO}    Path to target wordlist/dictionary")
    
    print(f"\n{COLOR_HIGHLIGHT}OPTIONAL ARGUMENTS:")
    print(f"  {COLOR_SUCCESS}-h, --help{COLOR_INFO}               Show this help message")
    print(f"  {COLOR_SUCCESS}--list-rules{COLOR_INFO}             List all supported rules and exit")
    print(f"  {COLOR_SUCCESS}-c, --comprehensive{COLOR_INFO}      Use comprehensive mode (16,000+ rules)")
    print(f"  {COLOR_SUCCESS}-b, --base-wordlist FILE{COLOR_INFO} Base wordlist (default: target wordlist)")
    print(f"  {COLOR_SUCCESS}-d, --chain-depth N{COLOR_INFO}      Rules to chain together (default: 1)")
    print(f"  {COLOR_SUCCESS}--batch-size N{COLOR_INFO}           GPU batch size (default: 5000)")
    print(f"  {COLOR_SUCCESS}-o, --output FILE{COLOR_INFO}        Save extracted rules to file")
    print(f"  {COLOR_SUCCESS}-r, --rules_file FILE{COLOR_INFO}    Filter external rules file")
    
    print(f"\n{COLOR_SUCCESS}EXAMPLES:")
    print(f"  {COLOR_INFO}Fast extraction (1,188 rules):{COLOR_DEBUG}")
    print(f"    {script_name} -w rockyou.txt -o fast_rules.rule")
    print(f"  {COLOR_INFO}Comprehensive extraction (16,000+ rules):{COLOR_DEBUG}")
    print(f"    {script_name} -w rockyou.txt -c -o comp_rules.rule")
    print(f"  {COLOR_INFO}With chaining depth 4:{COLOR_DEBUG}")
    print(f"    {script_name} -w rockyou.txt -b base.txt -d 4 --batch-size 500 -o chains.rule")
    print(f"  {COLOR_INFO}Comprehensive with depth 2:{COLOR_DEBUG}")
    print(f"    {script_name} -w rockyou.txt -c -d 2 --batch-size 200 -o comp_depth2.rule")
    
    print(f"\n{COLOR_WARNING}PERFORMANCE TIPS:")
    print(f"  • {COLOR_HIGHLIGHT}FAST MODE:{COLOR_WARNING} 1,188 rules - use batch-size 5000-10000")
    print(f"  • {COLOR_HIGHLIGHT}COMPREHENSIVE MODE:{COLOR_WARNING} 16,000+ rules - use batch-size 200-500")
    print(f"  • Chain depth 1-2 for comprehensive, 3-4 for fast mode")
    print(f"  • Use --list-rules to see all implemented rules\n")

# --- Memory-Optimized Processing ---
def process_batch_gpu(queue, bfs_kernel_func, words_buf, rules_buf, result_buf,
                     num_words_batch, num_rules, max_word_len, 
                     max_rule_len_padded, max_output_len_padded):
    """Process a batch with error handling and memory optimization."""
    try:
        # Clear result buffer
        cl.enqueue_fill_buffer(queue, result_buf, np.uint8(0), 0, result_buf.size).wait()
        
        # Launch kernel
        bfs_kernel_func(queue, (num_words_batch * num_rules,), None,
                        words_buf, rules_buf, result_buf,
                        np.uint32(num_words_batch),
                        np.uint32(num_rules), 
                        np.uint32(max_word_len),
                        np.uint32(max_rule_len_padded + 1), 
                        np.uint32(max_output_len_padded))
        
        # Get results
        host_results_flat = np.zeros(num_words_batch * num_rules * max_output_len_padded, dtype=np.uint8)
        cl.enqueue_copy(queue, host_results_flat, result_buf).wait()
        
        return host_results_flat, True
    except cl.MemoryError:
        return None, False
    except cl.Error as e:
        print(f"{COLOR_WARNING}GPU error: {e}")
        return None, False

# --- Main Logic ---
def main():
    # Handle --help and --list-rules before argument parsing
    if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
        print_colored_help()
        sys.exit(0)
    
    if '--list-rules' in sys.argv:
        # Show both modes
        print(f"{COLOR_TITLE}=== RULE MODES AVAILABLE ===")
        
        fast_rules = generate_fast_rules()
        comp_rules = generate_comprehensive_rules()
        
        print(f"\n{COLOR_HIGHLIGHT}1. FAST MODE (default):")
        print(f"  {COLOR_SUCCESS}Rules: {len(fast_rules)}")
        print(f"  {COLOR_INFO}Performance: Blazing fast")
        print(f"  {COLOR_INFO}Batch size: 5000-10000")
        print(f"  {COLOR_INFO}Depth: Up to 4+ chains")
        
        print(f"\n{COLOR_HIGHLIGHT}2. COMPREHENSIVE MODE (-c flag):")
        print(f"  {COLOR_SUCCESS}Rules: {len(comp_rules)}")
        print(f"  {COLOR_INFO}Performance: Memory intensive")
        print(f"  {COLOR_INFO}Batch size: 200-500")
        print(f"  {COLOR_INFO}Depth: 1-2 chains recommended")
        
        print(f"\n{COLOR_WARNING}Use -c flag for comprehensive mode (all Hashcat rules)")
        sys.exit(0)
    
    # Regular argument parsing for execution
    parser = argparse.ArgumentParser(description='OpenCL GPU Wrapper for rule extraction. Supports external rule files and rule chaining (BFS).', add_help=False)
    
    parser.add_argument('-w', '--wordlist', required=True, help='Path to the single wordlist file (target dictionary).')
    parser.add_argument('-b', '--base-wordlist', help='Optional path to a base wordlist file. If not specified, --wordlist is used as the base.')
    parser.add_argument('-d', '--chain-depth', type=int, default=1, help='Number of rules to chain together. Default: 1.')
    parser.add_argument('--batch-size', type=int, default=5000, help='Number of words to process in each GPU batch. Default: 5000.')
    parser.add_argument('-o', '--output', help='File to save the extracted rules to.')
    parser.add_argument('-r', '--rules_file', type=str, help='Path to a file containing external rules (one per line). Filters the internal rule set.')
    parser.add_argument('-c', '--comprehensive', action='store_true', help='Use comprehensive mode with 16,000+ rules (memory intensive).')
    
    args = parser.parse_args()

    # --- Select Rule Generation Mode ---
    if args.comprehensive:
        print(f"{COLOR_TITLE}=== INITIALIZING COMPREHENSIVE MODE ===")
        print(f"{COLOR_WARNING}Warning: Using comprehensive mode with 16,000+ rules")
        print(f"{COLOR_INFO}Recommended batch size: 200-500 for memory safety")
        all_rules_reference = generate_comprehensive_rules()
        mode = "COMPREHENSIVE"
    else:
        print(f"{COLOR_TITLE}=== INITIALIZING FAST MODE ===")
        print(f"{COLOR_INFO}Using fast mode with ~1,200 rules")
        all_rules_reference = generate_fast_rules()
        mode = "FAST"
    
    print(f"{COLOR_INFO}Generating rule database...")
    
    # Calculate Block IDs for the OpenCL kernel (simplified for fast mode)
    if mode == "FAST":
        num_simple_rules = 10 
        num_td_rules = 20
        num_s_rules = len([r for r in all_rules_reference if r.startswith('s')])
        num_a_rules = len([r for r in all_rules_reference if r[0] in '^$@'])
        
        start_id_simple = 0
        start_id_TD = num_simple_rules
        start_id_s = start_id_TD + num_td_rules
        start_id_A = start_id_s + num_s_rules
    else:
        # Comprehensive mode uses more categories
        num_simple_rules = 10
        num_td_rules = 20
        num_s_rules = len([r for r in all_rules_reference if r.startswith('s')])
        num_a_rules = len([r for r in all_rules_reference if r[0] in '^$@'])
        # We'll use a simplified block structure for comprehensive mode
        start_id_simple = 0
        start_id_TD = num_simple_rules
        start_id_s = start_id_TD + num_td_rules
        start_id_A = start_id_s + num_s_rules
    
    # Create the reference map: rule string -> original ID
    rule_id_map_reference = {rule: i for i, rule in enumerate(all_rules_reference)}

    # --- Rule Filtering ---
    if args.rules_file:
        print(f"{COLOR_INFO}Loading external rules from: '{args.rules_file}'")
        try:
            if not os.path.exists(args.rules_file):
                print(f"{COLOR_ERROR}Error: External rules file '{args.rules_file}' not found. Exiting.")
                sys.exit(1)
                
            with open(args.rules_file, 'r', encoding='latin-1') as f:
                external_rules = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Filter: only accept rules that are implemented in the kernel
            all_rules = [r for r in external_rules if r in rule_id_map_reference]
            
            if not all_rules:
                print(f"{COLOR_WARNING}Warning: No valid implemented rules found in the external file.")
                print(f"{COLOR_INFO}Falling back to all internal rules.")
                all_rules = all_rules_reference
            else:
                print(f"{COLOR_SUCCESS}Filtered {len(all_rules)} valid rules for GPU testing.")
            
        except Exception as e:
            print(f"{COLOR_ERROR}Error loading external rules file: {e}")
            print(f"{COLOR_INFO}Falling back to all internal rules.")
            all_rules = all_rules_reference
    else:
        print(f"{COLOR_INFO}Using internal static rule generation.")
        all_rules = all_rules_reference
    
    print_rule_statistics(all_rules, mode)
    
    # Auto-adjust batch size for comprehensive mode
    if mode == "COMPREHENSIVE" and args.batch_size > 500:
        print(f"{COLOR_WARNING}Auto-adjusting batch size from {args.batch_size} to 500 for comprehensive mode")
        args.batch_size = 500
    
    # --- Prepare Rule Buffer for GPU ---
    print(f"{COLOR_INFO}Preparing GPU data structures...")
    _, host_rules_gpu_temp, max_rule_len_padded = prepare_data_for_gpu([], all_rules, 0)
    
    # Overwrite sequential IDs with the correct reference IDs for the OpenCL kernel
    for i, rule in enumerate(all_rules):
        correct_id = rule_id_map_reference[rule]
        host_rules_gpu_temp[i, 0] = correct_id
        
    host_rules_gpu = host_rules_gpu_temp

    # --- OpenCL Kernel Source (Unified for both modes) ---
    kernel_source = f"""
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

        // --- Unify rule ID blocks ---
        unsigned int start_id_simple = {start_id_simple};
        unsigned int end_id_simple = start_id_simple + {num_simple_rules};
        unsigned int start_id_TD = {start_id_TD};
        unsigned int end_id_TD = start_id_TD + {num_td_rules};
        unsigned int start_id_s = {start_id_s};
        unsigned int end_id_s = start_id_s + {num_s_rules};
        unsigned int start_id_A = {start_id_A};
        unsigned int end_id_A = start_id_A + {num_a_rules};
        
        // --- Kernel Logic (Rule Transformation) ---
        
        if (rule_id >= start_id_simple && rule_id < end_id_simple) {{
            switch(rule_id - start_id_simple) {{
                case 0: {{
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
                case 1: {{
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
                case 2: {{
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
                            if (c >= 'A' && c <= 'Z') {{
                                result_ptr[i] = c + 32;
                                changed_flag = true;
                            }} else {{
                                result_ptr[i] = c;
                            }}
                        }}
                    }}
                    break;
                }}
                case 3: {{
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
                            if (c >= 'a' && c <= 'z') {{
                                result_ptr[i] = c - 32;
                                changed_flag = true;
                            }} else {{
                                result_ptr[i] = c;
                            }}
                        }}
                    }}
                    break;
                }}
                case 4: {{
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
                case 5: {{
                    out_len = word_len;
                    if (word_len > 1) {{
                        for (unsigned int i = 0; i < word_len; i++) {{
                            result_ptr[i] = current_word_ptr[word_len - 1 - i];
                        }}
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
                case 6: {{
                    out_len = word_len;
                    for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
                    if (word_len >= 2) {{
                        result_ptr[0] = current_word_ptr[1];
                        result_ptr[1] = current_word_ptr[0];
                        changed_flag = true;
                    }}
                    break;
                }}
                case 7: {{
                    out_len = word_len;
                    for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
                    changed_flag = false;
                    break;
                }}
                case 8: {{
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
                case 9: {{
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
        }} else if (rule_id >= start_id_TD && rule_id < end_id_TD) {{
            unsigned char operator_char = rule_ptr[0];
            unsigned char pos_char = rule_ptr[1];
            
            unsigned int pos_to_change = pos_char - '0';
            
            if (operator_char == 'T') {{
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
            else if (operator_char == 'D') {{
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
        else if (rule_id >= start_id_s && rule_id < end_id_s) {{
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
        }} else if (rule_id >= start_id_A && rule_id < end_id_A) {{
            out_len = word_len;
            for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
            
            unsigned char cmd = rule_ptr[0];
            unsigned char arg = rule_ptr[1];
            
            if (cmd == '^') {{
                if (word_len + 1 >= max_output_len_padded) {{
                    out_len = 0;
                    changed_flag = false;
                }} else {{
                    for(unsigned int i=word_len; i>0; i--) {{
                        result_ptr[i] = result_ptr[i-1];
                    }}
                    result_ptr[0] = arg;
                    out_len++;
                    changed_flag = true;
                }}
            }} else if (cmd == '$') {{
                if (word_len + 1 >= max_output_len_padded) {{
                    out_len = 0;
                    changed_flag = false;
                }} else {{
                    result_ptr[out_len] = arg;
                    out_len++;
                    changed_flag = true;
                }}
            }} else if (cmd == '@') {{
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
                 result_ptr[out_len] = 0;
            }}
        }} else {{
            for (unsigned int i = 0; i < max_output_len_padded; i++) {{
                result_ptr[i] = 0;
            }}
        }}
    }}
    """
    
    # Check for OpenCL context and queue
    ctx = None
    try:
        platforms = cl.get_platforms()
        chosen_platform = platforms[0]
        chosen_device = chosen_platform.get_devices()[0]
        ctx = cl.Context([chosen_device])
        queue = cl.CommandQueue(ctx)
        print(f"{COLOR_SUCCESS}Selected device: {chosen_device.name}")
    except (IndexError, cl.Error) as e:
        print(f"{COLOR_ERROR}ERROR: Cannot create context for the selected device. Error: {e}")
        return

    def load_data(filename):
        """Loads words from a file using 'latin-1' encoding."""
        if not os.path.exists(filename):
            print(f"{COLOR_ERROR}Error: The file '{filename}' does not exist.")
            return None
        try:
            with open(filename, 'r', encoding='latin-1') as f:
                return [line.strip().split()[0] for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            print(f"{COLOR_ERROR}An error occurred while loading the file '{filename}': {e}")
            return None

    wordlist = load_data(args.wordlist)
    if wordlist is None:
        return
    
    print(f"{COLOR_SUCCESS}Loaded {len(wordlist)} words from '{args.wordlist}' as the target dictionary.")
    
    word_set = set(wordlist)
    
    # Send the prepared rules buffer to the GPU
    mf = cl.mem_flags
    print(f"{COLOR_INFO}Transferring {len(all_rules)} rules to GPU...")
    rules_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_rules_gpu)
    
    # Build kernel
    print(f"{COLOR_INFO}Building OpenCL kernel...")
    try:
        prg = cl.Program(ctx, kernel_source).build()
        bfs_kernel_func = prg.bfs_kernel
        print(f"{COLOR_SUCCESS}Kernel built successfully!")
    except cl.Error as e:
        print(f"{COLOR_ERROR}Error building kernel: {e}")
        return

    extracted_rules_with_hits = Counter()

    # Load base wordlist
    base_wordlist = load_data(args.base_wordlist if args.base_wordlist else args.wordlist)
    if base_wordlist is None:
        return
    
    try:
        if args.base_wordlist:
            print(f"{COLOR_SUCCESS}Loaded {len(base_wordlist)} words from '{args.base_wordlist}' as the base wordlist.")
        else:
            print(f"{COLOR_SUCCESS}Using {len(wordlist)} words from target wordlist as base.")
    except Exception as e:
        print(f"{COLOR_ERROR}Error during wordlist loading: {e}")
        return

    initial_batch_size = args.batch_size
    current_batch_size = initial_batch_size
    
    # words_with_chains: list of (word, chain_of_rules_so_far)
    words_with_chains = [(word, "") for word in base_wordlist] 
    
    print(f"{COLOR_INFO}Starting BFS rule extraction with depth {args.chain_depth}...")
    print(f"{COLOR_INFO}Mode: {mode} | Rules: {len(all_rules)} | Batch size: {initial_batch_size}")
    
    for current_depth in range(1, args.chain_depth + 1):
        print(f"\n{COLOR_TITLE}=== Processing depth {current_depth}/{args.chain_depth} ===")
        
        words_to_process_file_in = f"words_to_process_d{current_depth}.tmp"
        words_to_process_file_out = f"words_to_process_d{current_depth+1}.tmp"
        
        if current_depth > 1 and os.path.exists(words_to_process_file_in):
            words_with_chains = []
            print(f"{COLOR_INFO}Loading words and previous chains from temporary file...")
            try:
                with open(words_to_process_file_in, 'r', encoding='latin-1') as f:
                    for line in f:
                        parts = line.strip().split('\t', 1)
                        if len(parts) == 2 and parts[0]:
                            words_with_chains.append((parts[0], parts[1]))
            except Exception as e:
                print(f"{COLOR_ERROR}Error reading temporary file: {e}. Stopping.")
                break

        words_to_process = [word for word, chain in words_with_chains]
        word_to_chain_map = {word: chain for word, chain in words_with_chains}

        if not words_to_process:
            if current_depth > 1:
                print(f"{COLOR_WARNING}No words to process for the next depth. Stopping.")
            break

        num_words_total = len(words_to_process)
        max_word_len = max([len(word.encode('latin-1')) for word in words_to_process] + [0]) + 1
        max_output_len_padded = max_word_len + current_depth * 10
        
        num_rules = len(all_rules) 
        current_batch_size = initial_batch_size
        
        unique_next_depth_words = set()
        total_hits_this_depth = 0
        
        # Calculate dynamic batch size based on available memory
        if mode == "COMPREHENSIVE" and current_depth > 1:
            # Reduce batch size for deeper chains in comprehensive mode
            recommended_size = max(50, initial_batch_size // (2 * (current_depth - 1)))
            if current_batch_size > recommended_size:
                print(f"{COLOR_INFO}Reducing batch size to {recommended_size} for depth {current_depth}")
                current_batch_size = recommended_size
        
        bar_format = f'{{l_bar}}{COLOR_SUCCESS}{{bar}}{COLOR_INFO}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}]'

        with open(words_to_process_file_out, 'w', encoding='latin-1') as f_out, \
             tqdm(total=num_words_total, unit='words', bar_format=bar_format, 
                  ncols=80, desc=f"{COLOR_INFO}Depth {current_depth}") as pbar:

            i = 0
            while i < num_words_total:
                batch_words = words_to_process[i:i + current_batch_size]
                num_words_batch = len(batch_words)
                global_size = num_words_batch * num_rules
                
                # Memory check for comprehensive mode
                if mode == "COMPREHENSIVE":
                    estimated_memory = (global_size * max_output_len_padded) / (1024*1024)
                    if estimated_memory > 100:  # More than 100MB
                        print(f"{COLOR_WARNING}Estimated memory: {estimated_memory:.1f}MB - reducing batch size")
                        current_batch_size = max(10, current_batch_size // 2)
                        continue
                
                try:
                    host_base_words_gpu, _, _ = prepare_data_for_gpu(batch_words, [], max_word_len)
                    words_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_base_words_gpu)
                    result_buf = cl.Buffer(ctx, mf.WRITE_ONLY, global_size * max_output_len_padded)
                    
                    # Process batch
                    results, success = process_batch_gpu(
                        queue, bfs_kernel_func, words_buf, rules_buf, result_buf,
                        num_words_batch, num_rules, max_word_len,
                        max_rule_len_padded + 1, max_output_len_padded
                    )
                    
                    if not success:
                        # Memory error - reduce batch size
                        print(f"\n{COLOR_WARNING}Memory error, reducing batch size from {current_batch_size} to {max(10, current_batch_size // 2)}")
                        current_batch_size = max(10, current_batch_size // 2)
                        words_buf.release()
                        result_buf.release()
                        continue
                    
                    # Process results
                    batch_hits = 0
                    for j in range(num_words_batch * num_rules):
                        start_idx = j * max_output_len_padded
                        word_bytes = results[start_idx:start_idx + max_output_len_padded].tobytes().split(b'\0', 1)[0]
                        
                        if word_bytes:
                            try:
                                transformed_word = word_bytes.decode('latin-1', 'ignore')
                            except UnicodeDecodeError:
                                transformed_word = None
                                
                            if transformed_word and transformed_word in word_set:
                                base_word_idx = j // num_rules
                                rule_idx = j % num_rules
                                base_word = batch_words[base_word_idx]
                                new_rule = all_rules[rule_idx]

                                previous_chain = word_to_chain_map.get(base_word, "")
                                
                                if previous_chain:
                                    full_chain = f"{previous_chain} {new_rule}"
                                else:
                                    full_chain = new_rule
                                
                                if transformed_word != base_word:
                                    extracted_rules_with_hits[full_chain] += 1
                                    batch_hits += 1
                                    
                                    if transformed_word not in unique_next_depth_words:
                                        unique_next_depth_words.add(transformed_word)
                                        f_out.write(f"{transformed_word}\t{full_chain}\n")
                    
                    total_hits_this_depth += batch_hits
                    
                    words_buf.release()
                    result_buf.release()
                    pbar.update(num_words_batch)
                    i += num_words_batch
                    current_batch_size = min(initial_batch_size, current_batch_size * 2)  # Gradually increase if successful
                        
                except cl.MemoryError:
                    print(f"\n{COLOR_WARNING}Memory allocation failed for batch size {current_batch_size}.")
                    current_batch_size = max(10, current_batch_size // 2)
                    if current_batch_size < 10:
                        print(f"{COLOR_ERROR}Cannot proceed with batch size < 10. Stopping.")
                        break

        # Clean up
        if current_depth > 1 and os.path.exists(words_to_process_file_in):
            os.remove(words_to_process_file_in)
            
        words_with_chains = []
        
        print(f"{COLOR_SUCCESS}Depth {current_depth} complete. Found {total_hits_this_depth} new rule chains.")
        
        # Stop if no new chains found
        if total_hits_this_depth == 0 and current_depth > 1:
            print(f"{COLOR_WARNING}No new rule chains found at depth {current_depth}. Stopping.")
            break
            
    print(f"\n{COLOR_TITLE}=== GPU EXTRACTION COMPLETE ===")
    
    sorted_rules = extracted_rules_with_hits.most_common()

    print(f"{COLOR_SUCCESS}Total unique rule chains extracted: {len(sorted_rules)}")

    if args.output:
        print(f"\n{COLOR_INFO}Saving extracted rule chains to '{args.output}'...")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                for full_chain, count in sorted_rules:
                    f.write(f"{full_chain}\n")
            print(f"{COLOR_SUCCESS}Done. Saved {len(sorted_rules)} rule chains to '{args.output}'")
        except Exception as e:
            print(f"{COLOR_ERROR}Error: Could not save rules to file '{args.output}'. Error: {e}")
    
    # Display top 10 rule chains
    if sorted_rules:
        print(f"\n{COLOR_TITLE}=== TOP 10 RULE CHAINS ===")
        for i, (rule_chain, count) in enumerate(sorted_rules[:10]):
            print(f"{COLOR_HIGHLIGHT}{i+1:2d}. {COLOR_SUCCESS}{rule_chain:<40} {COLOR_INFO}({count} hits)")

    # Remove all temporary files
    for i in range(1, args.chain_depth + 2):
        temp_file = f"words_to_process_d{i}.tmp"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"\n{COLOR_TITLE}=== FINISHED ===")
    print(f"{COLOR_SUCCESS}Rule extraction completed successfully!")
    print(f"{COLOR_INFO}Mode: {mode}")
    print(f"{COLOR_INFO}Rules processed: {len(all_rules)}")
    print(f"{COLOR_INFO}Max chain depth reached: {min(args.chain_depth, current_depth)}")
    print(f"{COLOR_INFO}Total rule chains found: {len(sorted_rules)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{COLOR_WARNING}\nProcess interrupted by user. Cleaning up...")
        for i in range(1, 10):
            temp_file = f"words_to_process_d{i}.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print(f"{COLOR_INFO}Temporary files cleaned up. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n{COLOR_ERROR}Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
