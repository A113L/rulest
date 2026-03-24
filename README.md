# GPU-Optimized Hashcat Rule Extractor

A high-performance, GPU-accelerated tool for discovering effective Hashcat rules and rule chains using OpenCL.

This project analyzes base and target wordlists to automatically extract transformation rules that convert base words into target words, optimized for large-scale cracking workflows.

---

## Features

- GPU-accelerated processing using OpenCL
- Automatic rule discovery
- Multi-depth rule chain generation
- GPU-compatible Hashcat rule validation
- Bloom filter optimization for fast lookups
- Dynamic VRAM and workload scaling
- Hot-rule biased chain generation
- Custom seed rule support

---

## Requirements

- Python 3.8 or higher
- OpenCL-compatible GPU (NVIDIA or AMD recommended)

Install dependencies:

```bash
pip install numpy pyopencl tqdm
```

---

## Usage

```bash
python rulest_v2.py base.txt target.txt \
  --max-depth 3 \
  --device 0 \
  --target-hours 0.5
```

---

## Arguments

### Positional Arguments

| Argument          | Description            |
|------------------|------------------------|
| base_wordlist     | Base wordlist path     |
| target_wordlist   | Target wordlist path   |

### Optional Arguments

| Argument            | Description |
|--------------------|------------|
| -d, --max-depth     | Maximum rule chain depth (>=1, default: 3) |
| -o, --output        | Output file (default: found_chains.txt) |
| --max-chains        | Maximum total number of chains to generate |
| --target-hours      | Target runtime in hours (default: 0.5) |
| --seed-rules        | File containing predefined rules or chains |
| --list-devices      | List available OpenCL devices and exit |
| --device            | Device index or substring (e.g., "0" or "NVIDIA") |

### Depth Override Arguments

| Argument              | Description |
|----------------------|------------|
| --depth2-chains       | Override limit for depth 2 |
| --depth3-chains       | Override limit for depth 3 |
| --depth4-chains       | Override limit for depth 4 |
| --depth5-chains       | Override limit for depth 5 |
| --depth6-chains       | Override limit for depth 6 |
| --depth7-chains       | Override limit for depth 7 |
| --depth8-chains       | Override limit for depth 8 |
| --depth9-chains       | Override limit for depth 9 |
| --depth10-chains      | Override limit for depth 10 |

---

## How It Works

### Phase 1: Single Rule Search

- Applies all GPU-compatible rules to base words
- Uses a Bloom filter to quickly check matches in the target set
- Tracks rule effectiveness based on hit counts

### Phase 2: Rule Chain Search

- Builds multi-step rule chains (depth >= 2)
- Prioritizes rules discovered in Phase 1
- Dynamically allocates compute budget per depth

---

## Key Components

### GPU Engine

Handles:
- OpenCL kernel execution
- Batch processing
- Memory-safe buffer allocation

### Rule Generator

- Generates only GPU-compatible Hashcat rules
- Validates rules against Hashcat constraints

### Bloom Filter

- Provides fast membership checking for target words
- Reduces expensive string comparisons

---

## Performance

Optimized for GPUs such as RTX 3060 Ti:

- Approximately 120 million combinations per second (baseline)
- Dynamic scaling based on:
  - VRAM
  - Compute units
  - Target runtime

---

## Output

- Produces rules and rule chains with hit counts
- Output can be used directly in Hashcat rule files

---

## Example Workflow

1. Provide base and target wordlists
2. Run the extractor
3. Export the best rules
4. Use them in Hashcat

```bash
hashcat -r extracted.rules ...
```
---

## License

MIT License

---

## Credits

Developed for advanced password research and rule optimization workflows.
https://github.com/synacktiv/rulesfinder
