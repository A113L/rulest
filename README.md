**GPU-Accelerated Rule Mining for Hash Cracking Wordlists (Full Chain Extraction)**

[![rulest.png](https://i.postimg.cc/cLC9vwHG/rulest.png)](https://postimg.cc/FYtgqf3T)

This project implements a highly optimized tool for extracting effective word-transformation rule chains by leveraging PyOpenCL to run the transformation logic on a Graphics Processing Unit (GPU). It uses a Breadth-First Search (BFS) approach to efficiently explore multi-step rule combinations (chains) that convert base dictionary words into target dictionary words.

💡 **Overview**
Rule-based attacks are highly effective against passwords. This utility automates the process of discovering the most successful rule chains (e.g., l $1 T0 - lowercase, append '1', toggle case at position 0) that have high hit rates against a specific target wordlist, accelerating rule-set generation for tools like Hashcat.

A high-performance OpenCL-based tool for extracting and testing Hashcat rules using GPU acceleration. Perfect for password cracking research and rule optimization.

🚀 **Features**

- GPU Accelerated: Uses OpenCL for massive parallel rule processing
- Memory Optimized: Implements memory-mapped I/O for large files (v2.0)
- Smart Batching: Automatic batch size calculation based on GPU memory
- Rule Chaining: Supports multi-depth rule chaining (BFS approach)
- Flexible Input: Works with internal rules or external rule files
- Progress Tracking: Real-time progress bars and statistics



⚙️ **Requirements**
- Python 3.x
- numpy
- pyopencl (Requires OpenCL runtime drivers for your GPU vendor: NVIDIA, AMD, or Intel).
- tqdm (For progress bar display)

```pip install numpy pyopencl tqdm```

```
python3 rulest.py -h

╔══════════════════════════════════════════════════════════════════════╗
║                GPU-ACCELERATED HASHCAT RULE EXTRACTOR                ║
╚══════════════════════════════════════════════════════════════════════╝

DESCRIPTION:
  Blazing-fast OpenCL GPU rule extraction with BFS chaining.
  Finds rule sequences that transform base words into target words.

USAGE:
  rulest.py -w WORDLIST [OPTIONS]

REQUIRED ARGUMENTS:
  -w, --wordlist WORDLIST    Path to target wordlist/dictionary

OPTIONAL ARGUMENTS:
  -h, --help               Show this help message
  --list-rules             List all supported rules and exit
  -c, --comprehensive      Use comprehensive mode (16,000+ rules)
  -b, --base-wordlist FILE Base wordlist (default: target wordlist)
  -d, --chain-depth N      Rules to chain together (default: 1)
  --batch-size N           GPU batch size (default: 5000)
  -o, --output FILE        Save extracted rules to file
  -r, --rules_file FILE    Filter external rules file

EXAMPLES:
  Fast extraction (1,188 rules):
    rulest.py -w rockyou.txt -o fast_rules.rule
  Comprehensive extraction (16,000+ rules):
    rulest.py -w rockyou.txt -c -o comp_rules.rule
  With chaining depth 4:
    rulest.py -w rockyou.txt -b base.txt -d 4 --batch-size 500 -o chains.rule
  Comprehensive with depth 2:
    rulest.py -w rockyou.txt -c -d 2 --batch-size 200 -o comp_depth2.rule

PERFORMANCE TIPS:
  • FAST MODE: 1,188 rules - use batch-size 5000-10000
  • COMPREHENSIVE MODE: 16,000+ rules - use batch-size 200-500
  • Chain depth 1-2 for comprehensive, 3-4 for fast mode
  • Use --list-rules to see all implemented rules
```

🎯 **Performance Tips**

- Use SSD Storage - Faster file reading
- Adjust Batch Size - Larger batches for more GPU memory
- Filter Rules - Use -r to test specific rule sets
- Chain Depth - Start with depth 1, increase gradually

🔧 **Technical Details**

*Memory Management*

- Automatic GPU memory detection (v2.0)
- Memory-mapped file I/O for large wordlists (v2.0)
- Smart batch sizing with safety margins
- Retry logic for memory allocation failures

*GPU Optimization*

- Work group parallelization
- Bulk processing with aligned memory
- Optimized kernel with barrier synchronization
- Efficient data transfer patterns

**Website**

https://hcrt.pages.dev/rulest.static_workflow

**Credits:**

https://github.com/synacktiv/rulesfinder
