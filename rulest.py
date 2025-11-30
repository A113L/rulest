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
from time import time
import signal

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

# --- Memory Management Constants ---
MEMORY_REDUCTION_FACTOR = 0.7
MAX_ALLOCATION_RETRIES = 5
VRAM_SAFETY_MARGIN = 0.15
LOCAL_WORK_SIZE = 256
DEFAULT_BATCH_SIZE = 10000

# --- Global variables for interrupt handling ---
interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C interrupt signal"""
    global interrupted
    print(f"\n{Colors.YELLOW}⚠️ Interrupt received! Cleaning up...{Colors.END}")
    interrupted = True

# --- Memory-Mapped File Reader ---
class MappedWordlistReader:
    """Memory-mapped wordlist reader for efficient large file processing"""
    
    def __init__(self, filename, encoding='latin-1'):
        self.filename = filename
        self.encoding = encoding
        self.file_size = os.path.getsize(filename)
        self.fd = None
        self.mm = None
        
    def __enter__(self):
        self.fd = open(self.filename, 'rb')
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mm:
            self.mm.close()
        if self.fd:
            self.fd.close()
            
    def estimate_word_count(self):
        """Fast word count estimation using sampling"""
        sample_size = min(10 * 1024 * 1024, self.file_size)  # 10MB or file size
        sample = self.mm[:sample_size]
        lines = sample.count(b'\n')
        
        if self.file_size <= sample_size:
            return lines
        else:
            avg_line_length = sample_size / max(lines, 1)
            return int(self.file_size / avg_line_length)
    
    def read_words_batch(self, batch_size, max_word_len, start_pos=0):
        """Read words in batches using memory mapping"""
        words = []
        current_pos = start_pos
        
        while current_pos < self.file_size and len(words) < batch_size and not interrupted:
            # Find next newline
            end_pos = self.mm.find(b'\n', current_pos)
            if end_pos == -1:
                end_pos = self.file_size
            
            # Extract line
            line_bytes = self.mm[current_pos:end_pos].strip()
            if line_bytes and len(line_bytes) <= max_word_len:
                try:
                    word = line_bytes.decode(self.encoding, 'ignore')
                    words.append(word)
                except UnicodeDecodeError:
                    pass
            
            current_pos = end_pos + 1
            if current_pos >= self.file_size:
                break
        
        return words, current_pos

# --- Optimized Data Preparation for GPU ---
def generate_leetspeak_rules():
    """Generates a list of common leetspeak substitution rules."""
    leetspeak_rules = []
    substitutions = {
        'a': '@', 'e': '3', 'i': '1', 'o': '0',
        's': '5', 't': '7', 'l': '1', 'z': '2'
    }
    
    for original, sub in substitutions.items():
        leetspeak_rules.append(f"s{original}{sub}")
        if original.isalpha():
            leetspeak_rules.append(f"s{original.upper()}{sub}")
        leetspeak_rules.append(f"s{sub}{original}")
            
    return leetspeak_rules

def generate_all_rules():
    """Generates the built-in, static set of rules including new Hashcat rules."""
    rules = []
    
    # Simple rules
    simple_rules = ['l', 'u', 'c', 'C', 't', 'r', 'k', ':', 'd', 'f']
    rules.extend(simple_rules)
    
    # T and D rules
    for i in range(10):
        rules.extend([f'T{i}', f'D{i}'])
    
    # s rules
    chars_2arg = string.digits + string.ascii_lowercase
    for c1 in chars_2arg:
        for c2 in chars_2arg:
            rules.append(f's{c1}{c2}')
            
    # Add leetspeak rules
    rules.extend(generate_leetspeak_rules())

    # Group A rules
    chars = string.digits + string.ascii_letters + string.punctuation
    for c in chars:
        rules.extend([f'^{c}', f'${c}', f'@{c}'])
    
    # Group B rules
    new_rules = []
    for n in range(10):
        new_rules.append(f'p{n}')
    
    new_rules.extend(['{', '}', '[', ']'])
    
    for n in range(10):
        for m in range(1, 10):
            new_rules.append(f'x{n}{m}')
    
    for n in range(10):
        for m in range(1, 10):
            new_rules.append(f'O{n}{m}')
    
    for n in range(10):
        for x in string.digits + string.ascii_lowercase:
            new_rules.append(f'i{n}{x}')
    
    for n in range(10):
        for x in string.digits + string.ascii_lowercase:
            new_rules.append(f'o{n}{x}')
    
    for n in range(10):
        new_rules.append(f"' {n}")
    
    for n in range(10):
        new_rules.append(f'z{n}')
    
    for n in range(10):
        new_rules.append(f'Z{n}')
    
    new_rules.append('q')
    rules.extend(new_rules)
    
    # Comprehensive rules
    comprehensive_rules = []
    comprehensive_rules.append('K')
    
    for n in range(10):
        for m in range(10):
            if n != m:
                comprehensive_rules.append(f'*{n}{m}')
    
    for n in range(10):
        comprehensive_rules.append(f'L{n}')
    
    for n in range(10):
        comprehensive_rules.append(f'R{n}')
    
    for n in range(10):
        comprehensive_rules.append(f'+{n}')
    
    for n in range(10):
        comprehensive_rules.append(f'-{n}')
    
    for n in range(10):
        comprehensive_rules.append(f'.{n}')
    
    for n in range(10):
        comprehensive_rules.append(f',{n}')
    
    for n in range(1, 10):
        comprehensive_rules.append(f'y{n}')
    
    for n in range(1, 10):
        comprehensive_rules.append(f'Y{n}')
    
    comprehensive_rules.append('E')
    
    for x in ['-', '_', '.', ',', ';']:
        comprehensive_rules.append(f'e{x}')
    
    for n in range(1, 5):
        for x in ['-', '_', '.', ',', ';', ' ']:
            comprehensive_rules.append(f'3{n}{x}')
    
    rules.extend(comprehensive_rules)
    
    return rules

def prepare_data_for_gpu(words, rules, max_word_len):
    """Prepares words and rules for GPU transfer with optimized memory layout."""
    
    rule_map = {}
    for i, r in enumerate(rules):
        rule_map[r] = i

    max_rule_len = max(len(rule.encode('latin-1')) for rule in rules) if rules else 0
    max_rule_len_padded = max_rule_len + 1 
    
    # Optimized rule storage: rule ID + rule bytes
    rules_padded = np.zeros((len(rules), max_rule_len_padded + 1), dtype=np.uint16)
    for i, rule in enumerate(rules):
        rule_bytes = rule.encode('latin-1')
        rule_id = rule_map.get(rule, 65535) 
        
        rules_padded[i, 0] = rule_id
        rules_padded[i, 1:1+len(rule_bytes)] = np.frombuffer(rule_bytes, dtype=np.uint8)
    
    # Optimized word storage with padding
    words_padded = np.zeros((len(words), max_word_len), dtype=np.uint8)
    for i, word in enumerate(words):
        if not word: continue
        word_bytes = word.encode('latin-1')
        words_padded[i, :len(word_bytes)] = np.frombuffer(word_bytes, dtype=np.uint8)
        
    return words_padded, rules_padded, max_rule_len_padded

def get_gpu_memory_info(device):
    """Get GPU memory information for optimal batch sizing"""
    try:
        total_memory = device.global_mem_size
        available_memory = int(total_memory * (1 - VRAM_SAFETY_MARGIN))
        return total_memory, available_memory
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️ Warning: Could not query GPU memory: {e}{Colors.END}")
        return 4 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024  # Conservative defaults

def calculate_optimal_batch_size(available_vram, max_word_len, max_output_len, num_rules):
    """Calculate optimal batch size based on available VRAM"""
    # Memory per word: input + output + rule processing overhead
    memory_per_word = (
        max_word_len +  # input word
        max_output_len +  # output word  
        (num_rules * 8)  # rule processing overhead (estimated)
    )
    
    max_batch_by_memory = int(available_vram / memory_per_word)
    optimal_batch = min(DEFAULT_BATCH_SIZE, max_batch_by_memory)
    
    # Ensure batch size is multiple of local work size for better performance
    optimal_batch = (optimal_batch // LOCAL_WORK_SIZE) * LOCAL_WORK_SIZE
    
    return max(1000, optimal_batch)  # Minimum batch size

def create_opencl_buffers_with_retry(context, buffer_specs, max_retries=MAX_ALLOCATION_RETRIES):
    """Create OpenCL buffers with retry logic for memory allocation failures"""
    buffers = {}
    current_reduction = 1.0
    
    for retry in range(max_retries + 1):
        try:
            for name, spec in buffer_specs.items():
                flags = spec['flags']
                size = int(spec['size'] * current_reduction)
                
                if 'hostbuf' in spec:
                    buffers[name] = cl.Buffer(context, flags, size, hostbuf=spec['hostbuf'])
                else:
                    buffers[name] = cl.Buffer(context, flags, size)
            
            return buffers
            
        except cl.MemoryError as e:
            if "MEM_OBJECT_ALLOCATION_FAILURE" in str(e) and retry < max_retries:
                print(f"{Colors.YELLOW}⚠️ Memory allocation failed, reducing memory usage...{Colors.END}")
                current_reduction *= MEMORY_REDUCTION_FACTOR
                # Clean up any partially allocated buffers
                for buf in buffers.values():
                    try:
                        buf.release()
                    except:
                        pass
                buffers = {}
            else:
                raise e
                
    raise cl.MemoryError(f"Failed to allocate buffers after {max_retries} retries")

# --- Optimized Main Logic ---
def main():
    # Setup interrupt handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description=f'{Colors.CYAN}OpenCL GPU Wrapper for rule extraction. Supports external rule files and rule chaining (BFS).{Colors.END}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-w', '--wordlist', required=True, help='Path to the single wordlist file (target dictionary).')
    parser.add_argument('-b', '--base-wordlist', help='Optional path to a base wordlist file. If not specified, --wordlist is used as the base.')
    parser.add_argument('-d', '--chain-depth', type=int, default=1, help='Number of rules to chain together. Default: 1.')
    parser.add_argument('--batch-size', type=int, default=0, help='Number of words to process in each GPU batch. Default: auto-calculate.')
    parser.add_argument('-o', '--output', help='File to save the extracted rules to.')
    parser.add_argument('-r', '--rules_file', type=str, help='Path to a file containing external rules (one per line).')
    
    # New performance options
    parser.add_argument('--max-memory', type=int, default=0, help='Maximum GPU memory to use in MB. Default: auto-detect.')
    parser.add_argument('--work-size', type=int, default=LOCAL_WORK_SIZE, help=f'OpenCL local work size. Default: {LOCAL_WORK_SIZE}.')

    args = parser.parse_args()

    # Print banner
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}                    GPU RULE EXTRACTION TOOL (OPTIMIZED){Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

    # --- OpenCL Initialization with Memory Detection ---
    try:
        platforms = cl.get_platforms()
        chosen_platform = platforms[0]
        chosen_device = chosen_platform.get_devices()[0]
        ctx = cl.Context([chosen_device])
        queue = cl.CommandQueue(ctx)
        
        # Get GPU memory info
        total_vram, available_vram = get_gpu_memory_info(chosen_device)
        if args.max_memory > 0:
            available_vram = min(available_vram, args.max_memory * 1024 * 1024)
            
        print(f"{Colors.GREEN}✅ Selected device: {Colors.BOLD}{chosen_device.name}{Colors.END}")
        print(f"{Colors.BLUE}💾 GPU Memory: {Colors.BOLD}{total_vram / (1024**3):.1f} GB total, {available_vram / (1024**3):.1f} GB available{Colors.END}")
        
    except (IndexError, cl.Error) as e:
        print(f"{Colors.RED}❌ ERROR: Cannot create context for the selected device. Error: {e}{Colors.END}")
        return

    # --- Generate Full Rule Set ---
    print(f"{Colors.YELLOW}🔧 Generating full rule set...{Colors.END}")
    all_rules_reference = generate_all_rules()

    # Calculate Block IDs for the OpenCL kernel
    num_simple_rules = 10 
    num_td_rules = 20
    num_s_rules = len(string.digits + string.ascii_lowercase)**2 + len(generate_leetspeak_rules())
    num_a_rules = len(string.digits + string.ascii_letters + string.punctuation) * 3
    
    num_groupB_rules = (
        10 + 4 + 90 + 90 + 160 + 160 + 10 + 10 + 10 + 1
    )
    
    num_comprehensive_rules = (
        1 + 90 + 10 + 10 + 10 + 10 + 10 + 10 + 9 + 9 + 1 + 5 + 30
    )
    
    start_id_simple = 0
    start_id_TD = num_simple_rules
    start_id_s = start_id_TD + num_td_rules
    start_id_A = start_id_s + num_s_rules
    start_id_groupB = start_id_A + num_a_rules
    start_id_comprehensive = start_id_groupB + num_groupB_rules

    # Create the reference map
    rule_id_map_reference = {rule: i for i, rule in enumerate(all_rules_reference)}

    # --- Rule Filtering ---
    if args.rules_file:
        print(f"{Colors.BLUE}📁 Loading and filtering rules from external file: '{args.rules_file}'{Colors.END}")
        try:
            if not os.path.exists(args.rules_file):
                print(f"{Colors.RED}❌ Error: External rules file '{args.rules_file}' not found. Exiting.{Colors.END}")
                sys.exit(1)
                
            with open(args.rules_file, 'r', encoding='latin-1') as f:
                external_rules = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            all_rules = [r for r in external_rules if r in rule_id_map_reference]
            
            if not all_rules:
                print(f"{Colors.YELLOW}⚠️ Warning: No valid implemented rules found in the external file. Falling back to all internal rules.{Colors.END}")
                all_rules = all_rules_reference
            else:
                print(f"{Colors.GREEN}✅ Filtered {len(all_rules)} valid rules for GPU testing.{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading external rules file: {e}. Falling back to all internal rules.{Colors.END}")
            all_rules = all_rules_reference
    else:
        print(f"{Colors.BLUE}🔧 Using internal static rule generation.{Colors.END}")
        all_rules = all_rules_reference

    # --- Prepare Rule Buffer for GPU ---
    _, host_rules_gpu_temp, max_rule_len_padded = prepare_data_for_gpu([], all_rules, 0)
    
    for i, rule in enumerate(all_rules):
        correct_id = rule_id_map_reference[rule]
        host_rules_gpu_temp[i, 0] = correct_id
        
    host_rules_gpu = host_rules_gpu_temp

    # --- Complete OpenCL Kernel Source with All Rule Implementations ---
    kernel_source = f"""
// Helper function to convert char digit/letter to int position
unsigned int char_to_pos(unsigned char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return 0xFFFFFFFF; 
}}

__kernel __attribute__((reqd_work_group_size({args.work_size}, 1, 1)))
void bfs_kernel(
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
    
    // Zero out the result buffer for this thread using local work group
    unsigned int local_id = get_local_id(0);
    unsigned int local_size = get_local_size(0);
    for(unsigned int i = local_id; i < max_output_len_padded; i += local_size) {{
        result_ptr[i] = 0;
    }}
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Unify rule ID blocks ---
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
    
    if (rule_id >= start_id_simple && rule_id < end_id_simple) {{
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
    }} else if (rule_id >= start_id_TD && rule_id < end_id_TD) {{
        unsigned char operator_char = rule_ptr[0];
        unsigned char pos_char = rule_ptr[1];
        unsigned int pos_to_change = char_to_pos(pos_char);
        
        if (operator_char == 'T') {{
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
        else if (operator_char == 'D') {{
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
    // --- START GROUP B RULES ---
    else if (rule_id >= start_id_groupB && rule_id < end_id_groupB) {{ 
        
        for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
        out_len = word_len;

        unsigned char cmd = rule_ptr[0];
        unsigned int N = (rule_ptr[1] != 0) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
        unsigned int M = (rule_ptr[2] != 0) ? char_to_pos(rule_ptr[2]) : 0xFFFFFFFF;
        unsigned char X = (rule_ptr[2] != 0) ? rule_ptr[2] : 0;

        if (cmd == 'p') {{
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
        
        else if (cmd == 'q') {{
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

        else if (cmd == '{{') {{
            if (word_len > 0) {{
                unsigned char first_char = current_word_ptr[0];
                for (unsigned int i = 0; i < word_len - 1; i++) {{
                    result_ptr[i] = current_word_ptr[i + 1];
                }}
                result_ptr[word_len - 1] = first_char;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == '}}') {{
            if (word_len > 0) {{
                unsigned char last_char = current_word_ptr[word_len - 1];
                for (unsigned int i = word_len - 1; i > 0; i--) {{
                    result_ptr[i] = current_word_ptr[i - 1];
                }}
                result_ptr[0] = last_char;
                changed_flag = true;
            }}
        }}
        
        else if (cmd == '[') {{
            if (word_len > 0) {{
                for (unsigned int i = 0; i < word_len - 1; i++) {{
                    result_ptr[i] = current_word_ptr[i + 1];
                }}
                out_len = word_len - 1;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == ']') {{
            if (word_len > 0) {{
                out_len = word_len - 1;
                changed_flag = true;
            }}
        }} 
        
        else if (cmd == 'x') {{
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
                out_len = 0; 
            }}
        }}
        
        else if (cmd == 'O') {{
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

        else if (cmd == 'i') {{
            unsigned int pos = N;
            unsigned char insert_char = X;

            if (pos != 0xFFFFFFFF && word_len + 1 < max_output_len_padded) {{
                unsigned int final_pos = (pos > word_len) ? word_len : pos;
                out_len = word_len + 1;

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

        else if (cmd == 'o') {{
            unsigned int pos = N;
            unsigned char new_char = X;

            if (pos != 0xFFFFFFFF && pos < word_len) {{
                result_ptr[pos] = new_char;
                changed_flag = true;
            }}
        }}
        
        else if (cmd == '\\'') {{
            unsigned int pos = N;
            
            if (pos != 0xFFFFFFFF && pos < word_len) {{
                out_len = pos;
                changed_flag = true;
            }} 
        }}

        else if (cmd == 'z') {{
            unsigned int num_dupes = N;
            if (num_dupes != 0xFFFFFFFF && num_dupes > 0) {{
                unsigned int total_len = word_len + num_dupes;
                if (total_len < max_output_len_padded) {{
                    unsigned char first_char = current_word_ptr[0];
                    unsigned int out_idx = 0;
                    
                    for (unsigned int i = 0; i < num_dupes; i++) {{
                        result_ptr[out_idx++] = first_char;
                    }}
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

        else if (cmd == 'Z') {{
            unsigned int num_dupes = N;
            if (num_dupes != 0xFFFFFFFF && num_dupes > 0) {{
                unsigned int total_len = word_len + num_dupes;
                if (total_len < max_output_len_padded) {{
                    unsigned char last_char = current_word_ptr[word_len - 1];
                    
                    unsigned int out_idx = word_len;
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
        
        for(unsigned int i=0; i<word_len; i++) result_ptr[i] = current_word_ptr[i];
        out_len = word_len;

        unsigned char cmd = rule_ptr[0];
        unsigned int N = (rule_ptr[1] != 0) ? char_to_pos(rule_ptr[1]) : 0xFFFFFFFF;
        unsigned int M = (rule_ptr[2] != 0) ? char_to_pos(rule_ptr[2]) : 0xFFFFFFFF;
        unsigned char X = (rule_ptr[2] != 0) ? rule_ptr[2] : 0;
        unsigned char separator = rule_ptr[1];

        if (cmd == 'K') {{
            if (word_len >= 2) {{
                result_ptr[word_len - 1] = current_word_ptr[word_len - 2];
                result_ptr[word_len - 2] = current_word_ptr[word_len - 1];
                changed_flag = true;
            }}
        }}
        else if (cmd == '*') {{
            if (N != 0xFFFFFFFF && M != 0xFFFFFFFF && N < word_len && M < word_len && N != M) {{
                unsigned char temp = result_ptr[N];
                result_ptr[N] = result_ptr[M];
                result_ptr[M] = temp;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'L') {{
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] << 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == 'R') {{
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] >> 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '+') {{
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] + 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '-') {{
            if (N != 0xFFFFFFFF && N < word_len) {{
                result_ptr[N] = current_word_ptr[N] - 1;
                changed_flag = true;
            }}
        }}
        else if (cmd == '.') {{
            if (N != 0xFFFFFFFF && N + 1 < word_len) {{
                result_ptr[N] = current_word_ptr[N + 1];
                changed_flag = true;
            }}
        }}
        else if (cmd == ',') {{
            if (N != 0xFFFFFFFF && N > 0 && N < word_len) {{
                result_ptr[N] = current_word_ptr[N - 1];
                changed_flag = true;
            }}
        }}
        else if (cmd == 'y') {{
            if (N != 0xFFFFFFFF && N > 0 && N <= word_len) {{
                unsigned int total_len = word_len + N;
                if (total_len < max_output_len_padded) {{
                    for (int i = word_len - 1; i >= 0; i--) {{
                        result_ptr[i + N] = result_ptr[i];
                    }}
                    for (unsigned int i = 0; i < N; i++) {{
                        result_ptr[i] = current_word_ptr[i];
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }}
            }}
        }}
        else if (cmd == 'Y') {{
            if (N != 0xFFFFFFFF && N > 0 && N <= word_len) {{
                unsigned int total_len = word_len + N;
                if (total_len < max_output_len_padded) {{
                    for (unsigned int i = 0; i < N; i++) {{
                        result_ptr[word_len + i] = current_word_ptr[word_len - N + i];
                    }}
                    out_len = total_len;
                    changed_flag = true;
                }}
            }}
        }}
        else if (cmd == 'E') {{
            for (unsigned int i = 0; i < word_len; i++) {{
                unsigned char c = current_word_ptr[i];
                if (c >= 'A' && c <= 'Z') {{
                    result_ptr[i] = c + 32;
                }} else {{
                    result_ptr[i] = c;
                }}
            }}
            
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
        else if (cmd == 'e') {{
            for (unsigned int i = 0; i < word_len; i++) {{
                unsigned char c = current_word_ptr[i];
                if (c >= 'A' && c <= 'Z') {{
                    result_ptr[i] = c + 32;
                }} else {{
                    result_ptr[i] = c;
                }}
            }}
            
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
        else if (cmd == '3') {{
            unsigned int separator_count = 0;
            unsigned int target_count = N;
            unsigned char sep_char = X;
            
            if (target_count != 0xFFFFFFFF) {{
                for (unsigned int i = 0; i < word_len; i++) {{
                    if (current_word_ptr[i] == sep_char) {{
                        separator_count++;
                        if (separator_count == target_count && i + 1 < word_len) {{
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
            result_ptr[out_len] = 0;
        }}
    }} else {{
        for (unsigned int i = local_id; i < max_output_len_padded; i += local_size) {{
            result_ptr[i] = 0;
        }}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
}}
"""

    # --- Memory-Mapped File Loading ---
    def load_words_mapped(filename, batch_size, max_word_len):
        """Load words using memory-mapped files for efficiency"""
        with MappedWordlistReader(filename) as reader:
            words = []
            current_pos = 0
            
            while current_pos < reader.file_size and not interrupted:
                batch_words, current_pos = reader.read_words_batch(batch_size, max_word_len, current_pos)
                if batch_words:
                    words.extend(batch_words)
                else:
                    break
                    
            return words

    # --- Determine Optimal Batch Size ---
    print(f"{Colors.BLUE}📊 Analyzing wordlists for optimal batch sizing...{Colors.END}")
    
    with MappedWordlistReader(args.wordlist) as reader:
        estimated_words = reader.estimate_word_count()
        print(f"{Colors.GREEN}✅ Estimated words in target dictionary: {estimated_words:,}{Colors.END}")

    # Calculate max word length for memory allocation
    test_words = load_words_mapped(args.wordlist, 10000, 256)  # Sample to find max length
    max_word_len = max([len(word.encode('latin-1')) for word in test_words] + [0]) + 1
    max_output_len = max_word_len * 2  # Account for duplication rules
    
    # Auto-calculate batch size if not specified
    if args.batch_size == 0:
        args.batch_size = calculate_optimal_batch_size(
            available_vram, max_word_len, max_output_len, len(all_rules)
        )
        print(f"{Colors.GREEN}✅ Auto-calculated batch size: {args.batch_size:,}{Colors.END}")

    # --- Load Data with Memory Mapping ---
    print(f"{Colors.BLUE}📖 Loading target dictionary with memory mapping...{Colors.END}")
    wordlist = load_words_mapped(args.wordlist, args.batch_size, max_word_len)
    if not wordlist:
        print(f"{Colors.RED}❌ Error: No words loaded from target dictionary.{Colors.END}")
        return

    word_set = set(wordlist)
    
    # Send the prepared rules buffer to the GPU
    mf = cl.mem_flags
    
    # Create buffers with retry logic
    buffer_specs = {
        'rules_buf': {
            'flags': mf.READ_ONLY | mf.COPY_HOST_PTR,
            'size': host_rules_gpu.nbytes,
            'hostbuf': host_rules_gpu
        }
    }
    
    try:
        buffers = create_opencl_buffers_with_retry(ctx, buffer_specs)
        rules_buf = buffers['rules_buf']
    except cl.MemoryError as e:
        print(f"{Colors.RED}❌ ERROR: Could not allocate GPU memory for rules: {e}{Colors.END}")
        return

    prg = cl.Program(ctx, kernel_source).build()
    bfs_kernel_func = prg.bfs_kernel

    extracted_rules_with_hits = Counter()

    # Load base wordlist
    print(f"{Colors.BLUE}📖 Loading base wordlist with memory mapping...{Colors.END}")
    base_wordlist = load_words_mapped(args.base_wordlist if args.base_wordlist else args.wordlist, 
                                     args.batch_size, max_word_len)
    if not base_wordlist:
        print(f"{Colors.RED}❌ Error: No words loaded from base wordlist.{Colors.END}")
        return

    print(f"{Colors.GREEN}✅ Loaded {len(base_wordlist):,} words from base wordlist.{Colors.END}")

    words_with_chains = [(word, "") for word in base_wordlist] 
    
    # --- Bulk Processing with Work Groups ---
    for current_depth in range(1, args.chain_depth + 1):
        if interrupted:
            break
            
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}🔍 Processing depth {current_depth}/{args.chain_depth}...{Colors.END}")
        
        # Load words for current depth
        if current_depth > 1:
            words_with_chains = []
            temp_file = f"words_to_process_d{current_depth}.tmp"
            if os.path.exists(temp_file):
                print(f"{Colors.BLUE}📖 Loading words from previous depth...{Colors.END}")
                with open(temp_file, 'r', encoding='latin-1') as f:
                    for line in f:
                        parts = line.strip().split('\t', 1)
                        if len(parts) == 2 and parts[0]:
                            words_with_chains.append((parts[0], parts[1]))

        if not words_with_chains:
            print(f"{Colors.YELLOW}⚠️ No words to process for depth {current_depth}.{Colors.END}")
            break

        words_to_process = [word for word, chain in words_with_chains]
        word_to_chain_map = {word: chain for word, chain in words_with_chains}

        num_words_total = len(words_to_process)
        max_output_len_padded = max_word_len + current_depth * 10
        
        num_rules = len(all_rules) 

        new_found_words = []
        unique_next_depth_words = set()

        output_file = f"words_to_process_d{current_depth+1}.tmp"
        
        with open(output_file, 'w', encoding='latin-1') as f_out, \
             tqdm(total=num_words_total, unit='words', bar_format='{l_bar}%s{bar}%s{r_bar}' % (Colors.CYAN, Colors.END)) as pbar:

            i = 0
            while i < num_words_total and not interrupted:
                batch_words = words_to_process[i:i + args.batch_size]
                num_words_batch = len(batch_words)
                
                # Prepare batch data for GPU
                host_base_words_gpu, _, _ = prepare_data_for_gpu(batch_words, [], max_word_len)
                
                # Create buffers for this batch
                batch_buffer_specs = {
                    'words_buf': {
                        'flags': mf.READ_ONLY | mf.COPY_HOST_PTR,
                        'size': host_base_words_gpu.nbytes,
                        'hostbuf': host_base_words_gpu
                    },
                    'result_buf': {
                        'flags': mf.WRITE_ONLY,
                        'size': num_words_batch * num_rules * max_output_len_padded
                    }
                }
                
                try:
                    batch_buffers = create_opencl_buffers_with_retry(ctx, batch_buffer_specs)
                    words_buf = batch_buffers['words_buf']
                    result_buf = batch_buffers['result_buf']
                    
                    # Initialize result buffer
                    cl.enqueue_fill_buffer(queue, result_buf, np.uint8(0), 0, result_buf.size).wait()
                    
                    # Launch kernel with work groups
                    global_size = num_words_batch * num_rules
                    global_size_aligned = (int(math.ceil(global_size / args.work_size)) * args.work_size,)
                    
                    bfs_kernel_func(queue, global_size_aligned, (args.work_size,),
                                   words_buf, rules_buf, result_buf,
                                   np.uint32(num_words_batch),
                                   np.uint32(num_rules), 
                                   np.uint32(max_word_len),
                                   np.uint32(max_rule_len_padded + 1), 
                                   np.uint32(max_output_len_padded))
                                     
                    # Get results
                    host_results_flat = np.zeros(num_words_batch * num_rules * max_output_len_padded, dtype=np.uint8)
                    cl.enqueue_copy(queue, host_results_flat, result_buf).wait()
                    
                    # Process results in bulk
                    for j in range(num_words_batch * num_rules):
                        start_idx = j * max_output_len_padded
                        word_bytes = host_results_flat[start_idx:start_idx + max_output_len_padded].tobytes().split(b'\0', 1)[0]
                        
                        if word_bytes:
                            try:
                                transformed_word = word_bytes.decode('latin-1', 'ignore')
                            except UnicodeDecodeError:
                                continue
                                
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
                                    
                                    if transformed_word not in unique_next_depth_words:
                                        unique_next_depth_words.add(transformed_word)
                                        f_out.write(f"{transformed_word}\t{full_chain}\n")
                    
                    # Clean up batch buffers
                    words_buf.release()
                    result_buf.release()
                    
                    pbar.update(num_words_batch)
                    i += num_words_batch
                        
                except cl.MemoryError as e:
                    print(f"\n{Colors.YELLOW}⚠️ Memory allocation failed for batch. Skipping batch...{Colors.END}")
                    i += num_words_batch  # Skip this batch
                    continue

        # Clean up temporary file from current depth
        if current_depth > 1:
            prev_temp_file = f"words_to_process_d{current_depth}.tmp"
            if os.path.exists(prev_temp_file):
                os.remove(prev_temp_file)

    # Clean up final temporary file
    if args.chain_depth >= 1:
        final_temp_file = f"words_to_process_d{args.chain_depth+1}.tmp"
        if os.path.exists(final_temp_file):
            os.remove(final_temp_file)

    if interrupted:
        print(f"\n{Colors.YELLOW}⚠️ Processing was interrupted.{Colors.END}")
        return

    print(f"\n{Colors.BOLD}{Colors.GREEN}✅ GPU-based extraction finished.{Colors.END}")
    
    sorted_rules = extracted_rules_with_hits.most_common()

    print(f"{Colors.BOLD}{Colors.CYAN}📊 Total unique rules chains extracted: {Colors.WHITE}{len(sorted_rules)}{Colors.END}")

    if args.output:
        print(f"\n{Colors.BLUE}💾 Saving extracted rule chains to '{args.output}'...{Colors.END}")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                for full_chain, count in sorted_rules:
                    f.write(f"{full_chain}\n")
            print(f"{Colors.GREEN}✅ Done.{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error: Could not save rules to file '{args.output}'. Error: {e}{Colors.END}")

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}                    EXTRACTION COMPLETE{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

if __name__ == '__main__':
    main()
