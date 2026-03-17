**GPU-Accelerated Rule Mining for Hash Cracking Wordlists (Full Chain Extraction)**

💡 **Overview**

This Python script leverages the computational power of graphics cards (via OpenCL) to automatically discover Hashcat rules that transform a given base wordlist into target passwords. Its operation is based on a two‑phase process: first, a Bloom filter is generated from all target words; then, in the initial phase, individual rules are tested against every base word. Rules that produce a hit become "hot," and in the second phase they are combined into chains, with 60% of the combinations containing at least one proven rule. The script dynamically adjusts the work‑group size and buffer allocation according to the GPU's capabilities, and it estimates the runtime based on a user‑specified time limit. The final set of rules is filtered to ensure full GPU compatibility (memory‑based and reject rules are removed). The output can be used directly in rule‑based attacks within Hashcat, significantly accelerating hash cracking in scenarios where example plaintext‑hash pairs are available.

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
python3 rulest_v2.0.py -h
usage: rulest_v2.0.py [-h] [-d MAX_DEPTH] [-o OUTPUT] [--max-chains MAX_CHAINS] [--target-hours TARGET_HOURS] [--seed-rules SEED_RULES] [--list-devices]
                [--device DEVICE] [--depth2-chains DEPTH2_CHAINS] [--depth3-chains DEPTH3_CHAINS] [--depth4-chains DEPTH4_CHAINS]
                [--depth5-chains DEPTH5_CHAINS] [--depth6-chains DEPTH6_CHAINS] [--depth7-chains DEPTH7_CHAINS] [--depth8-chains DEPTH8_CHAINS]
                [--depth9-chains DEPTH9_CHAINS] [--depth10-chains DEPTH10_CHAINS]
                [base_wordlist] [target_wordlist]

GPU-COMPATIBLE Hashcat Rules Engine with Hit Counting and Seed Support

positional arguments:
  base_wordlist         Base wordlist path
  target_wordlist       Target wordlist path

optional arguments:
  -h, --help            show this help message and exit
  -d MAX_DEPTH, --max-depth MAX_DEPTH
                        Max chain depth (>=1, default: 3). Now supports any depth.
  -o OUTPUT, --output OUTPUT
                        Output file (default: found_chains.txt)
  --max-chains MAX_CHAINS
                        Maximum TOTAL number of chains to generate (default: unlimited)
  --target-hours TARGET_HOURS
                        Target completion time in hours (default: 0.5)
  --seed-rules SEED_RULES
                        File containing proven rules/chains to use as seeds for deeper generation
  --list-devices        List available OpenCL devices and exit
  --device DEVICE       Device index or substring (e.g., "0" or "NVIDIA")
  --depth2-chains DEPTH2_CHAINS
                        Override dynamic limit for depth 2 chains
  --depth3-chains DEPTH3_CHAINS
                        Override dynamic limit for depth 3 chains
  --depth4-chains DEPTH4_CHAINS
                        Override dynamic limit for depth 4 chains
  --depth5-chains DEPTH5_CHAINS
                        Override dynamic limit for depth 5 chains
  --depth6-chains DEPTH6_CHAINS
                        Override dynamic limit for depth 6 chains
  --depth7-chains DEPTH7_CHAINS
                        Override dynamic limit for depth 7 chains
  --depth8-chains DEPTH8_CHAINS
                        Override dynamic limit for depth 8 chains
  --depth9-chains DEPTH9_CHAINS
                        Override dynamic limit for depth 9 chains
  --depth10-chains DEPTH10_CHAINS
                        Override dynamic limit for depth 10 chains

```

**Credits:**

https://github.com/synacktiv/rulesfinder
