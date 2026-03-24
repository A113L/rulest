# 🔐 GPU-Accelerated Hashcat Rule Extractor

> Extract and chain Hashcat-compatible rules from wordlists using OpenCL GPU acceleration.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Scripts](#-scripts)
  - [rulest.py — v1 (BFS, Legacy)](#rulestypy--v1-bfs-legacy)
  - [rulest\_v2.py — v2 (Recommended)](#rulest_v2py--v2-recommended)
- [Why v2 Supersedes v1](#-why-v2-supersedes-v1)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [rulest\_v2.py — Full Reference](#rulest_v2py--full-reference)
- [Architecture (v2)](#-architecture-v2)
- [Output Format](#-output-format)
- [Performance Tuning](#-performance-tuning)
- [Examples](#-examples)

---

## 🔍 Overview

This toolkit analyzes two wordlists — a **base** (source) wordlist and a **target** (dictionary) wordlist — and reverse-engineers the **Hashcat rules** that transform words from the base into words in the target. Rules are discovered via GPU-parallel transformation and validated for direct compatibility with Hashcat's GPU engine.

The result is a `.rule` file you can load directly into Hashcat (`-r rules.txt`), ordered by effectiveness (hit count).

---

## 📁 Scripts

### `rulest.py` — v1 (BFS, Legacy)

A first-generation implementation using a **Breadth-First Search (BFS)** chaining strategy executed on the GPU via a monolithic OpenCL kernel.

**Approach:**
- Generates a static, hard-coded rule set (simple rules, T/D positional, `s`-substitution, Group A)
- Chains rules across depths using temporary disk files to pass state between BFS layers
- No rule validation against Hashcat's GPU compatibility specification
- No Bloom filter — lookups performed directly against a Python `set`
- Single device selection (first available platform/device)
- No hit counting or frequency-based ranking
- Fixed batch size; halves on `MemoryError`

**When to use:** Historical reference only. v2 is strictly superior in every dimension.

---

### `rulest_v2.py` — v2 (Recommended)

A complete redesign built around GPU efficiency, Hashcat compatibility, and intelligent search strategy.

**Key capabilities:**
- ✅ Full **Hashcat GPU rule validation** (max 255 ops, correct argument types)
- ✅ **Bloom filter** on-GPU for fast membership testing with configurable false-positive rate
- ✅ **Two-phase extraction**: single-rule sweep → informed chain generation
- ✅ **Dynamic VRAM-aware** batch and budget sizing
- ✅ **Hot-rule biased** chain generation using Phase 1 results
- ✅ **Seed rules** support to guide chain exploration
- ✅ Per-depth chain budget overrides (depths 2–10)
- ✅ Unlimited result cap (no global ceiling)
- ✅ Full **hit counting** and frequency-ranked output
- ✅ Multi-device listing and explicit device selection
- ✅ Color-coded terminal output with live progress bars

---

## ⚡ Why v2 Supersedes v1

| Aspect | v1 (`rulest.py`) | v2 (`rulest_v2.py`) |
|---|---|---|
| **Rule validation** | None — invalid rules passed to Hashcat | Full `HashcatRuleValidator` against GPU spec (max 255 ops) |
| **Rule set size** | ~2,700 static rules | 100,000+ GPU-validated Hashcat rules across 9 categories |
| **Search strategy** | Naive BFS — every rule applied blindly | Phase 1 single-rule sweep → Phase 2 hot-biased chain generation |
| **Target lookup** | Python `set` (host RAM, per-result) | 64 MB Bloom filter uploaded once to GPU VRAM (FNV-1a, 4 hash functions) |
| **Chain state** | Temp `.tmp` files on disk per depth | In-memory, GPU buffer-based with proper release and `gc.collect()` |
| **Memory management** | Halve batch on OOM, no VRAM awareness | Dynamic sizing based on actual free VRAM estimate + safety factor |
| **Hit counting** | ❌ Not implemented | ✅ Full `Counter`-based frequency tracking, sorted output |
| **Device selection** | First platform, first device | `--list-devices`, `--device` by index or name substring |
| **Seed rules** | ❌ Not supported | ✅ `--seed-rules` file; seeds used to extend chains to deeper depths |
| **Per-depth budget** | ❌ Not supported | ✅ `--depth2-chains` through `--depth10-chains` overrides |
| **Output** | Unsorted, no metadata | Sorted by frequency; header with total hits and rule count |
| **Rule categories** | Simple, T/D, s, Group A | + `i`, `o`, `x`, `*`, `O`, `e`, `3`, `p`, `y`, `Y`, `z`, `Z`, `L`, `R`, `+`, `-`, `.`, `,`, `'`, `E`, `k`, `K`, `{`, `}`, `[`, `]`, `q` |

### BFS vs. Informed Chain Generation

The core algorithmic difference matters at scale:

**v1 BFS:** Every word × every rule at each depth level. At depth 2 with 2,700 rules and 100,000 base words: **270 million combinations per depth**, with no prioritization. State must be written to disk between depths, creating an I/O bottleneck. Rules that never produce hits are retried at every depth.

**v2 Informed Generation:** Phase 1 identifies which individual rules ("hot rules") actually hit the target dictionary. Phase 2 then generates chains **biased 60% toward hot rules** (configurable via `HOT_RULE_RATIO`). An additional 30% of the budget extends known-good seed chains. This dramatically reduces wasted GPU cycles and finds effective multi-rule sequences far faster than exhaustive BFS.

---

## 📦 Requirements

```
Python >= 3.8
numpy
pyopencl
tqdm
```

An OpenCL-capable GPU (NVIDIA, AMD, or Intel) is required. CPU fallback via OpenCL is supported but will be slow.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gpu-rule-extractor.git](https://github.com/A113L/rulest.git
cd rulest

# Install dependencies
pip install numpy pyopencl tqdm

# Verify OpenCL is available
python -c "import pyopencl; print(pyopencl.get_platforms())"
```

> **Windows users:** Install the appropriate OpenCL runtime for your GPU vendor. NVIDIA users typically have this via the CUDA toolkit or standard driver. AMD users should install ROCm or the AMD APP SDK.

---

## 🚀 Usage

### `rulest_v2.py` — Full Reference

```
usage: rulest_v2.py [options] base_wordlist target_wordlist
```

#### Positional Arguments

| Argument | Description |
|---|---|
| `base_wordlist` | Source wordlist — words to transform *from* |
| `target_wordlist` | Target dictionary — words to transform *to* |

#### Optional Arguments

| Flag | Default | Description |
|---|---|---|
| `-d`, `--max-depth` | `3` | Maximum rule chain depth (1–12; depths >12 warned) |
| `-o`, `--output` | `found_chains.txt` | Output file path |
| `--max-chains` | unlimited | Hard cap on total chains generated |
| `--target-hours` | `0.5` | Time budget in hours; controls chain generation budget |
| `--seed-rules` | None | File with known-good rules/chains to use as generation seeds |
| `--list-devices` | — | Print all available OpenCL devices and exit |
| `--device` | best GPU | Device index (e.g. `0`) or name substring (e.g. `NVIDIA`) |
| `--depth2-chains` | dynamic | Override chain generation limit for depth 2 |
| `--depth3-chains` | dynamic | Override chain generation limit for depth 3 |
| `--depth4-chains` through `--depth10-chains` | dynamic | Per-depth overrides up to depth 10 |

#### Legacy v1 Reference

```
usage: rulest.py -w WORDLIST [-b BASE_WORDLIST] [-d CHAIN_DEPTH]
                 [--batch-size N] [-o OUTPUT] [-r RULES_FILE]
```

---

## 🏗 Architecture (v2)

```
┌─────────────────────────────────────────────────────┐
│                  GPUExtractor                       │
│  ┌───────────────┐     ┌─────────────────────────┐  │
│  │  Rules        │     │  Dynamic Parameters     │  │
│  │  Generator    │────▶│  (VRAM-aware sizing)    │  │
│  └───────────────┘     └────────────┬────────────┘  │
│                                     │               │
│  ┌──────────────────────────────────▼────────────┐  │
│  │                 GPUEngine                     │  │
│  │                                               │  │
│  │  ┌─────────────┐    ┌───────────────────────┐ │  │
│  │  │ Bloom Filter│    │  OpenCL Kernel        │ │  │
│  │  │ (64MB VRAM) │    │  ┌─────────────────┐  │ │  │
│  │  └─────────────┘    │  │find_single_rules│  │ │  │
│  │                     │  ├─────────────────┤  │ │  │
│  │  Phase 1 ────────▶  │  │find_rule_chains │  │ │  │
│  │  (all words ×       │  └─────────────────┘  │ │  │
│  │   single rules)     └───────────────────────┘ │  │
│  │                                               │  │
│  │  Phase 2 ────────▶  Informed chain generation │  │
│  │  (hot-biased,        + seed extension         │  │
│  │   VRAM-budgeted)                              │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         │
         ▼
  HashcatRuleValidator  →  GPU-safe output (.rule file)
```

### Two-Phase Processing

**Phase 1 — Single Rule Sweep**
All base words are processed against every GPU-compatible single rule in parallel. The Bloom filter (built from the entire target wordlist and uploaded once) allows near-zero-cost hit detection on-device. Results feed a `Counter` of rule → hit frequency.

**Phase 2 — Informed Chain Generation**
Using Phase 1 hit data, chains are generated with a bias toward rules that already demonstrated effectiveness. The remaining time budget (total `--target-hours` minus Phase 1 duration) is split evenly across requested depths, with each depth's per-chain work scaled by word count and depth length. Seed rules from `--seed-rules` are extended to deeper depths automatically.

---

## 📄 Output Format

`found_chains.txt` (or your specified `-o` path):

```
# Generated by rulest_v2.0.py (seeded)
# Total unique rules: 4821
# Total hits (sum of frequencies): 2193047
:
c
$1
u
l $1
c $!
sa@ $0
...
```

- The identity rule (`:`) is always written first for Hashcat compatibility
- Rules are sorted by hit frequency (descending), then alphabetically
- All rules are guaranteed GPU-valid (max 255 ops, correct argument syntax)

---

## 🎛 Performance Tuning

| Goal | Recommendation |
|---|---|
| Maximize coverage in fixed time | Increase `--target-hours` |
| Reduce VRAM pressure | Lower `--max-chains` or use `--depth2-chains` / `--depth3-chains` |
| Force deep chain exploration | Set `--depth4-chains 50000 --depth5-chains 10000` explicitly |
| Use a specific GPU | `--device 1` or `--device "RTX 4090"` |
| Bootstrap from prior results | Pass previous output to `--seed-rules` for iterative refinement |
| Limit total combinations | `--max-chains 500000` to cap generation before scaling |

### VRAM Scaling

v2 automatically scales batch sizes, Bloom filter allocation, and chain budgets based on detected free VRAM. Baseline targets 8 GB; systems with less VRAM will receive proportionally smaller batches while maintaining correctness. The Bloom filter floor is 16 MB (32 MB on <4 GB VRAM), and the batch floor prevents starvation on very constrained devices.

---

## 💡 Examples

**Basic single-depth extraction:**
```bash
python rulest_v2.py rockyou.txt target_hashes_plain.txt -d 1 -o single_rules.txt
```

**Deep chain search with a 2-hour budget:**
```bash
python rulest_v2.py rockyou.txt target.txt -d 4 --target-hours 2.0 -o chains_deep.txt
```

**Use a specific GPU and seed from a previous run:**
```bash
python rulest_v2.py base.txt target.txt \
  --device "RTX 3080" \
  --seed-rules single_rules.txt \
  -d 3 --target-hours 1.0 \
  -o refined_chains.txt
```

**List available OpenCL devices:**
```bash
python rulest_v2.py --list-devices
```

**Override chain budget for specific depths:**
```bash
python rulest_v2.py base.txt target.txt -d 5 \
  --depth2-chains 200000 \
  --depth3-chains 100000 \
  --depth4-chains 30000 \
  --depth5-chains 5000 \
  -o custom_budget.txt
```

---

## 📝 License

MIT

## Credits

https://github.com/synacktiv/rulesfinder
