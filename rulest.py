# OpenCL GPU-accelerated rule extraction tool using BFS chaining to find full rule sequences (R1 R2...) that transform base words into target dictionary words.
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
    """Generates the built-in, static set of rules."""
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

    # Group A rules (ID ...-END): prepend (^), append ($), delete all instances (@)
    chars = string.digits + string.ascii_letters + string.punctuation
    for c in chars:
        rules.extend([f'^{c}', f'${c}', f'@{c}'])
    
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
    
    start_id_simple = 0
    start_id_TD = num_simple_rules
    start_id_s = start_id_TD + num_td_rules
    start_id_A = start_id_s + num_s_rules
    
    # Create the reference map: rule string -> original ID
    rule_id_map_reference = {rule: i for i, rule in enumerate(all_rules_reference)}

    # --- Rule Filtering ---
    if args.rules_file:
        print(f"Loading and filtering rules from external file: '{args.rules_file}'")
        try:
            if not os.path.exists(args.rules_file):
                print(f"Error: External rules file '{args.rules_file}' not found. Exiting.")
                sys.exit(1)
                
            # Use 'latin-1' encoding instead of 'utf-8' to prevent decoding errors
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

    # --- OpenCL Kernel Source (Unchanged) ---
    # The kernel source is large and remains unchanged, using f-string interpolation for block IDs.
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
        // rule_id_ptr[0] contains the rule ID (uint16). 
        // This ID refers to the original full list, so the logic remains the same.
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
    # --- End OpenCL Kernel Source ---
    
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

    def load_target_dictionary(filename):
        """Loads the entire target dictionary into a set for fast lookup (O(1))."""
        if not os.path.exists(filename):
            print(f"Error: The target dictionary file '{filename}' does not exist.")
            return None
        try:
            with open(filename, 'r', encoding='latin-1') as f:
                # Store as a set for O(1) lookups
                return set(line.strip().split()[0] for line in f if line.strip() and not line.startswith('#'))
        except Exception as e:
            print(f"An error occurred while loading the target dictionary '{filename}': {e}")
            return None

    word_set = load_target_dictionary(args.wordlist)
    if word_set is None:
        return
    
    print(f"Loaded {len(word_set)} words from '{args.wordlist}' as the target dictionary (in memory).")
    
    # Send the prepared rules buffer to the GPU
    mf = cl.mem_flags
    rules_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_rules_gpu)
    
    prg = cl.Program(ctx, kernel_source).build()
    bfs_kernel_func = prg.bfs_kernel

    extracted_rules_with_hits = Counter()

    initial_batch_size = args.batch_size
    current_batch_size = initial_batch_size
    
    num_rules = len(all_rules)
    
    # Generator for lazy-loading words and chains
    def lazy_load_words_and_chains(current_depth, base_filename, temp_filename):
        """
        A generator that yields (word, previous_chain) tuples,
        reading either the base wordlist or a temporary file.
        """
        if current_depth == 1:
            # Depth 1: Read the BASE wordlist
            if not os.path.exists(base_filename):
                print(f"Error: Base wordlist '{base_filename}' not found.")
                return
            
            print(f"Streaming base wordlist from '{base_filename}'.")
            with open(base_filename, 'r', encoding='latin-1') as f:
                for line in f:
                    word = line.strip().split()[0]
                    if word and not line.startswith('#'):
                        # Word, empty previous chain
                        yield (word, "") 
        else:
            # Depth > 1: Read transformed words/chains from the temporary file
            if not os.path.exists(temp_filename):
                return
                
            print(f"Streaming words and previous chains from temporary file: {temp_filename}")
            with open(temp_filename, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2 and parts[0]:
                        # parts[0] = transformed_word, parts[1] = previous_chain
                        yield (parts[0], parts[1])
    
    # --- BFS Chaining Loop ---
    for current_depth in range(1, args.chain_depth + 1):
        print(f"\nProcessing depth {current_depth}/{args.chain_depth}...")
        
        # Determine input/output files
        base_filename = args.base_wordlist if args.base_wordlist else args.wordlist
        words_to_process_file_in = f"words_to_process_d{current_depth}.tmp"
        words_to_process_file_out = f"words_to_process_d{current_depth+1}.tmp"
        
        # Get the iterator for this depth
        word_chain_iterator = lazy_load_words_and_chains(
            current_depth, 
            base_filename, 
            words_to_process_file_in
        )
        
        # Check if the iterator yielded anything (if file was empty or not found)
        # Note: We cannot easily determine the total number of words without a full pass. 
        # Tqdm total=None is used for streaming input.
        
        temp_word_chain_iterator, word_chain_iterator = itertools.tee(word_chain_iterator)
        try:
            next(temp_word_chain_iterator)
        except StopIteration:
            if current_depth > 1:
                print("No words were generated by the previous depth. Stopping BFS.")
            break
        # Reset iterator
        word_chain_iterator = lazy_load_words_and_chains(
            current_depth, 
            base_filename, 
            words_to_process_file_in
        )

        # Max output length adjustment (heuristic based on max possible growth: 'd' or 'f' rules)
        max_word_len_current = 0 # Will be updated dynamically per batch
        max_output_len_padded = 1024 # Initial safe estimate
        
        words_processed_count = 0

        with open(words_to_process_file_out, 'w', encoding='latin-1') as f_out, \
             tqdm(unit='words', desc=f"Depth {current_depth}", total=None) as pbar: # total=None for streaming

            while True:
                # Fetch the next batch of (word, chain) tuples from the iterator
                batch_data = list(itertools.islice(word_chain_iterator, current_batch_size))
                
                if not batch_data:
                    break
                    
                batch_words = [word for word, chain in batch_data]
                # Map for combining chains ONLY for the current batch
                word_to_chain_map = {word: chain for word, chain in batch_data}
                
                num_words_batch = len(batch_words)
                
                # --- Dynamic Padding Adjustment (Critical for GPU efficiency) ---
                # Recalculate max word len for the current batch and set max output size
                max_word_len_batch = max([len(word.encode('latin-1')) for word in batch_words] + [1]) + 1
                max_word_len = max(max_word_len_batch, max_word_len)
                
                # Max output len must accommodate current max word len + max growth (e.g., from 'd' rules)
                # We use a heuristic: max_word_len * 2 (from 'd' rule) + 1 for null
                max_output_len_padded = max(max_word_len * 2 + 1, max_output_len_padded) 
                
                global_size = num_words_batch * num_rules
                
                try:
                    # Prepare input data for the current batch
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
                    
                    # Process results and count hits
                    for j in range(num_words_batch * num_rules):
                        start_idx = j * max_output_len_padded
                        # Extract word bytes up to the null terminator
                        word_bytes = host_results_flat[start_idx:start_idx + max_output_len_padded].tobytes().split(b'\0', 1)[0]
                        
                        try:
                            transformed_word = word_bytes.decode('latin-1', 'ignore')
                        except UnicodeDecodeError:
                            transformed_word = None
                            
                        # --- Result Registration Logic ---
                        if transformed_word and transformed_word in word_set:
                            base_word_idx = j // num_rules
                            rule_idx = j % num_rules
                            base_word = batch_words[base_word_idx]
                            new_rule = all_rules[rule_idx]

                            previous_chain = word_to_chain_map.get(base_word, "")
                            
                            # Create the full chain for the found hit: R1 R2... Rn
                            if previous_chain:
                                full_chain = f"{previous_chain} {new_rule}"
                            else:
                                full_chain = new_rule
                                
                            # Condition: ensure the word was actually changed
                            if transformed_word != base_word:
                                # Count unique hits for the FULL CHAIN
                                extracted_rules_with_hits[full_chain] += 1
                                
                                # Save the word and the RULE CHAIN to the temporary file for the next depth
                                # We don't check uniqueness here, as duplicates are filtered naturally
                                # when reloading the set of words for the next depth.
                                # However, to reduce disk I/O, a simple in-memory set check is re-introduced:
                                
                                # Check if the word *as a new starting point* should be used in the next depth
                                # The uniqueness check is applied to the combination of (transformed_word, full_chain)
                                # but since the word itself determines the next state, we check unique words only
                                
                                # Only write if we haven't already written this word as a base word for the next level
                                if transformed_word not in unique_next_depth_words:
                                     unique_next_depth_words.add(transformed_word)
                                     # Format: transformed_word\tfull_chain_for_next_step
                                     f_out.write(f"{transformed_word}\t{full_chain}\n")
                                     
                    
                    words_buf.release()
                    result_buf.release()
                    
                    pbar.update(num_words_batch)
                    words_processed_count += num_words_batch
                    current_batch_size = initial_batch_size 
                        
                except cl.MemoryError:
                    print(f"\n[Warning] Memory allocation failed for batch size {current_batch_size}. Retrying with smaller batch size.")
                    current_batch_size //= 2
                    if current_batch_size == 0:
                        print("ERROR: Failed to allocate memory even for the smallest batch size. Stopping.")
                        return
                    # The `break` here means the loop will restart at the top of the `while True`
                    # but since `islice` has consumed the items, the memory error handling is imperfect
                    # for generators. For simplicity in this implementation, we will stop on critical failure.
                    if current_batch_size < 100:
                         print("ERROR: Batch size too small. Stopping.")
                         return

        # Clean up temporary file from the current depth (if it's not the first depth)
        if current_depth > 1 and os.path.exists(words_to_process_file_in):
            os.remove(words_to_process_file_in)
            
        # The temporary file for the next depth (f"words_to_process_d{current_depth+1}.tmp")
        # will be the input for the next loop iteration (current_depth + 1).

        # If no words were generated for the next depth, we stop the BFS early
        if not unique_next_depth_words and current_depth < args.chain_depth:
            print(f"Depth {current_depth} yielded no new words. Stopping BFS early.")
            break
            
    print(f"\nGPU-based extraction finished. Total words processed: {words_processed_count}")
    
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
