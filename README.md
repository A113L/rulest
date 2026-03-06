**GPU-Accelerated Rule Mining for Hash Cracking Wordlists (Full Chain Extraction)**

💡 **Overview**

This Python script leverages the computational power of graphics cards (via OpenCL) to automatically discover Hashcat rules that transform a given base wordlist into target passwords. Its operation is based on a two‑phase process: first, a Bloom filter is generated from all target words; then, in the initial phase, individual rules are tested against every base word. Rules that produce a hit become "hot," and in the second phase they are combined into chains (up to depth three), with 60% of the combinations containing at least one proven rule. The script dynamically adjusts the work‑group size and buffer allocation according to the GPU's capabilities, and it estimates the runtime based on a user‑specified time limit. The final set of rules is filtered to ensure full GPU compatibility (memory‑based and reject rules are removed). The output can be used directly in rule‑based attacks within Hashcat, significantly accelerating hash cracking in scenarios where example plaintext‑hash pairs are available.

**What does the script do?**

- Generates possible GPU-compatible rules
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
usage: rulest-v2.0.py [-h] [-d {1,2,3}] [-o OUTPUT] [--max-chains MAX_CHAINS] [--target-hours TARGET_HOURS] [--depth2-chains DEPTH2_CHAINS]
                      [--depth3-chains DEPTH3_CHAINS]
                      base_wordlist target_wordlist

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
  --max-chains MAX_CHAINS
                        Maximum chains to generate (overrides automatic limits)
  --target-hours TARGET_HOURS
                        Target completion time in hours (default: 0.5)
  --depth2-chains DEPTH2_CHAINS
                        Override dynamic limit for depth 2 chains
  --depth3-chains DEPTH3_CHAINS
                        Override dynamic limit for depth 3 chains

```

**Credits:**

https://github.com/synacktiv/rulesfinder
