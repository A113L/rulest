**GPU-Accelerated Rule Mining for Hash Cracking Wordlists (Full Chain Extraction)**

[![rulest.png](https://i.postimg.cc/vDRkTBrs/rulest.png)](https://postimg.cc/1VHJYy7C)

This project implements a highly optimized tool for extracting effective word-transformation rule chains by leveraging PyOpenCL to run the transformation logic on a Graphics Processing Unit (GPU). It uses a Breadth-First Search (BFS) approach to efficiently explore multi-step rule combinations (chains) that convert base dictionary words into target dictionary words.

💡 **Overview**
Rule-based attacks are highly effective against passwords. This utility automates the process of discovering the most successful rule chains (e.g., l $1 T0 - lowercase, append '1', toggle case at position 0) that have high hit rates against a specific target wordlist, accelerating rule-set generation for tools like Hashcat.

A high-performance OpenCL-based tool for extracting and testing Hashcat rules using GPU acceleration. Perfect for password cracking research and rule optimization.

🚀 **Features**

- GPU Accelerated: Uses OpenCL for massive parallel rule processing
- Smart Batching: Automatic batch size calculation based on GPU memory
- Rule Chaining: Supports multi-depth rule chaining (BFS approach); MITM IN V2.0
- Flexible Input: Works with internal rules or external rule files
- Progress Tracking: Real-time progress bars and statistics



⚙️ **Requirements**
- Python 3.x
- numpy
- pyopencl (Requires OpenCL runtime drivers for your GPU vendor: NVIDIA, AMD, or Intel).
- tqdm (For progress bar display)

```pip install numpy pyopencl tqdm```

```
 python3 rulest-2.0.py -h
usage: rulest-2.0.py [-h] [-d {1,2,3}] [-o OUTPUT] [--slow] [--verify] [--max-chains MAX_CHAINS] [--target-hours TARGET_HOURS] base_wordlist target_wordlist

GPU-COMPATIBLE Hashcat Rules Engine with Dynamic Workload Processing

positional arguments:
  base_wordlist         Base wordlist path
  target_wordlist       Target wordlist path

optional arguments:
  -h, --help            show this help message and exit
  -d {1,2,3}, --depth {1,2,3}
                        Max chain depth (1-3 only for speed, default: 3)
  -o OUTPUT, --output OUTPUT
                        Output file (default: found_chains.txt)
  --slow                Enable more comprehensive search (may take longer)
  --verify              Verify all chains work before saving
  --max-chains MAX_CHAINS
                        Maximum chains to generate (overrides automatic limits)
  --target-hours TARGET_HOURS
                        Target completion time in hours (default: 0.5)


```

🔧 **Technical Details**

*Memory Management*

- Automatic GPU memory detection (v2.0)
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
