**GPU-Accelerated Rule Mining for Hash Cracking Wordlists (Full Chain Extraction)**

This project implements a highly optimized tool for extracting effective word-transformation rule chains by leveraging PyOpenCL to run the transformation logic on a Graphics Processing Unit (GPU). It uses a Breadth-First Search (BFS) approach to efficiently explore multi-step rule combinations (chains) that convert base dictionary words into target dictionary words.

💡 **Overview**
Rule-based attacks are highly effective against passwords. This utility automates the process of discovering the most successful rule chains (e.g., l $1 T0 - lowercase, append '1', toggle case at position 0) that have high hit rates against a specific target wordlist, accelerating rule-set generation for tools like Hashcat.

**Key Features**
- OpenCL Acceleration: Word transformations are processed in parallel on the GPU for massive performance gains.

- Rule Chaining (BFS): Supports exploring rule chains up to a configurable depth (e.g., -d 3).

- Full Chain Extraction: Outputs the complete sequence of rules (R1 R2 ... Rn) responsible for a hit.

- Optimized Data Handling: Uses np.uint8 and np.uint16 buffers for minimal memory overhead and fast GPU transfers.

- Rule Filtering: Allows using an external file to limit the internal rule set, focusing the search space.

⚙️ **Requirements**
- Python 3.x

- numpy

- pyopencl (Requires OpenCL runtime drivers for your GPU vendor: NVIDIA, AMD, or Intel).

- tqdm (For progress bar display)

```pip install numpy pyopencl tqdm```

```
Argument,        Short Description
--wordlist,      -w,  (Required) Path to the target dictionary (words to match).
--base-wordlist, -b,  Optional path to the base wordlist (words to apply rules to).
--chain-depth,   -d,  Number of rules to chain together.
--batch-size,         Number of words to process per GPU batch.
--output,        -o,  File to save the extracted rule chains to.
--rules_file,    -r,  Path to an external file to filter the internal rule set.,Internal set
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



