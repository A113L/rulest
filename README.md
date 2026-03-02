**GPU-Accelerated Rule Mining for Hash Cracking Wordlists (Full Chain Extraction)**

[![rulest.png](https://i.postimg.cc/vDRkTBrs/rulest.png)](https://postimg.cc/1VHJYy7C)

This project implements a highly optimized tool for extracting effective word-transformation rule chains by leveraging PyOpenCL to run the transformation logic on a Graphics Processing Unit (GPU).

💡 **Overview**
Rule-based attacks are highly effective against passwords. This utility automates the process of discovering the most successful rule chains (e.g., l $1 T0 - lowercase, append '1', toggle case at position 0) that have high hit rates against a specific target wordlist, accelerating rule-set generation for tools like Hashcat.

A high-performance OpenCL-based tool for extracting and testing Hashcat rules using GPU acceleration. Perfect for password cracking research and rule optimization.

**What does the script do?**

- Generates all possible GPU-compatible rules
- Creates random rule chains of limited depth
- Checks them in parallel on the GPU for all base words
- Filters the results through a bloom filter (checks if the generated word exists in the target wordlist)



⚙️ **Requirements**
- Python 3.x
- numpy
- pyopencl (Requires OpenCL runtime drivers for your GPU vendor: NVIDIA, AMD, or Intel).
- tqdm (For progress bar display)

```pip install numpy pyopencl tqdm```

```
python3 rulest-v2.0.py -h
usage: rulest-v2.0.py [-h] [-d {1,2,3,4,5,6}] [-o OUTPUT] [--slow] [--verify] [--max-chains MAX_CHAINS] [--target-hours TARGET_HOURS] [--mitm] [--mitm-candidates MITM_CANDIDATES] base_wordlist target_wordlist

GPU-COMPATIBLE Hashcat Rules Engine with OPTIMIZED MITM Support

positional arguments:
  base_wordlist         Base wordlist path
  target_wordlist       Target wordlist path

optional arguments:
  -h, --help            show this help message and exit
  -d {1,2,3,4,5,6}, --depth {1,2,3,4,5,6}
                        Max chain depth (1-6, default: 6)
  -o OUTPUT, --output OUTPUT
                        Output file (default: found_chains.txt)
  --slow                Enable more comprehensive search (may take longer)
  --verify              Verify all chains work before saving
  --max-chains MAX_CHAINS
                        Maximum chains to generate (overrides automatic limits)
  --target-hours TARGET_HOURS
                        Target completion time in hours (default: 1.0)
  --mitm                Enable Meet-in-the-Middle for depths 4-6 (default: True)
  --mitm-candidates MITM_CANDIDATES
                        Maximum MITM candidates before GPU testing (default: 200000)
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
