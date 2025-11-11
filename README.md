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
--rules_file,    -r,  Path to an external file to filter the internal rule set.
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


**Rules supported by script**

*Simple Rules* (10 rules)

- l - Lowercase all letters

- u - Uppercase all letters

- c - Capitalize first letter, lowercase rest

- C - Lowercase first letter, uppercase rest

- t - Toggle case of all letters

- r - Reverse the word

- k - Swap first two characters

- : - Identity/no change

- d - Duplicate word

- f - Reflect word (word + reverse)

*Toggle/Delete Rules* (20 rules)

- T0-T9 - Toggle case at position 0-9

- D0-D9 - Delete character at position 0-9

*Substitution Rules* (1,296 + leetspeak rules)

- sXY - Substitute character X with Y

- X,Y ∈ (0-9, a-z) = 36 × 36 = 1,296 rules

*Leetspeak substitutions* (24 rules):

- sa@, sA@, s@a, se3, sE3, s3e, si1, sI1, s1i, so0, sO0, s0o, ss5, sS5, s5s, st7, sT7, s7t, sl1, sL1, s1l, sz2, sZ2, s2z

*Group A Rules* (282 rules)

- ^X - Prepend character X (94 rules)

- $X - Append character X (94 rules)

- @X - Delete all instances of character X (94 rules)

Where X ∈ (0-9, a-z, A-Z, punctuation) = 94 characters

*Group B - New Hashcat Rules* (545 rules)

*Duplication Rules*

- p0-p9 - Duplicate word N times (10 rules)

- q - Duplicate all characters (1 rule)

*Rotation Rules*

- { - Rotate left (1 rule)

- } - Rotate right (1 rule)

*Truncation Rules*

- [ - Delete first character (1 rule)

- ] - Delete last character (1 rule)

*Extraction Rules*

- xNM - Extract range starting at N, length M (90 rules)

- N ∈ (0-9), M ∈ (1-9) = 10 × 9 = 90 rules

*Omission Rules*

- ONM - Omit range starting at N, length M (90 rules)

- N ∈ (0-9), M ∈ (1-9) = 10 × 9 = 90 rules

*Insertion Rules*

- iNX - Insert character X at position N (160 rules)

- N ∈ (0-9), X ∈ (0-9, a-z) = 10 × 16 = 160 rules

*Overwrite Rules*

- oNX - Overwrite character at position N with X (160 rules)

- N ∈ (0-9), X ∈ (0-9, a-z) = 10 × 16 = 160 rules

*Truncate Position Rules*

- '0-'9 - Truncate at position N (10 rules)

*Character Duplication Rules*

- z0-z9 - Duplicate first character N times (10 rules)

- Z0-Z9 - Duplicate last character N times (10 rules)

*Comprehsive rules*

- K - Swap last two characters

- *NM - Swap characters at positions N and M

- LN - Bitwise shift left

- RN - Bitwise shift right

- +N - ASCII increment

- -N - ASCII decrement

- .N - Replace with next character

- ,N - Replace with previous character

- yN - Duplicate first N characters

- YN - Duplicate last N characters

- E - Title case

- eX - Title case with custom separator

- 3NX - Toggle case after Nth separator

https://hcrt.pages.dev/rulest.static_workflow




