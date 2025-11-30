**GPU-Accelerated Rule Mining for Hash Cracking Wordlists (Full Chain Extraction)**

[![rulest.png](https://i.postimg.cc/cLC9vwHG/rulest.png)](https://postimg.cc/FYtgqf3T)

This project implements a highly optimized tool for extracting effective word-transformation rule chains by leveraging PyOpenCL to run the transformation logic on a Graphics Processing Unit (GPU). It uses a Breadth-First Search (BFS) approach to efficiently explore multi-step rule combinations (chains) that convert base dictionary words into target dictionary words.

💡 **Overview**
Rule-based attacks are highly effective against passwords. This utility automates the process of discovering the most successful rule chains (e.g., l $1 T0 - lowercase, append '1', toggle case at position 0) that have high hit rates against a specific target wordlist, accelerating rule-set generation for tools like Hashcat.

A high-performance OpenCL-based tool for extracting and testing Hashcat rules using GPU acceleration. Perfect for password cracking research and rule optimization.

🚀 **Features**

- GPU Accelerated: Uses OpenCL for massive parallel rule processing
- Memory Optimized: Implements memory-mapped I/O for large files
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
-w, --wordlist	Target dictionary file (required)
-b, --base-wordlist	Base wordlist for rule application
-r, --rules_file	External rules file to test
-d, --chain-depth	Rule chaining depth (default: 1)
-o, --output	Output file for extracted rules
```
**Examples**

1. Extracting Single Rules (Depth 1)
Find all single rules that transform a base word into a target word, saving the results to my_rules_d1.txt.

```python rulest.py -w target_passwords.txt -b base_words.txt -d 1 -o my_rules_d1.txt```

2. Extracting Two-Step Rule Chains (Depth 2)
Find rule chains of length two (R1 R2) that lead to a hit.

```python rulest.py -w target_passwords.txt -b base_words.txt -d 2 -o my_rules_d2.txt```

3. Using a Filtered Rule Set
Only test rules specified in common_leetspeak.rule for a three-step chain.

```python rulest.py -w target.txt -b base.txt -d 3 -r common_leetspeak.rule -o leet_chains_d3.txt```


🎯 **Performance Tips**

- Use SSD Storage - Faster file reading
- Adjust Batch Size - Larger batches for more GPU memory
- Filter Rules - Use -r to test specific rule sets
- Chain Depth - Start with depth 1, increase gradually
- Monitor Memory - Use --max-memory on limited GPUs

🔧 **Technical Details**

*Memory Management*

- Automatic GPU memory detection
- Memory-mapped file I/O for large wordlists
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
https://github.com/hashcat/hashcat-utils/blob/master/src/cleanup-rules.c

