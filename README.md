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
  - [Extraction Pipeline](#extraction-pipeline)
- [Built-in Seed Families](#-built-in-seed-families)
- [Functional Minimization](#-functional-minimization)
- [Rule Categories](#-rule-categories)
- [GPU Command Support](#-gpu-command-support)
- [Configuration Constants](#-configuration-constants)
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
- ✅ Full **Hashcat GPU rule validation** (max 31 ops, correct argument types)
- ✅ **Bloom filter** on-GPU for fast membership testing with configurable false-positive rate
- ✅ **Three-phase extraction**: single-rule sweep (Phase 1) → built-in numeric seed pass (Phase S) → informed chain generation (Phase 2)
- ✅ **Built-in numeric seed families** (A–E): five deterministically generated seed families covering numeric prefixes/suffixes, mixed prepend/append, transform+digit combos, and date patterns — always run as a dedicated extraction pass, independent of `--max-depth` and the random-chain time budget
- ✅ **Signature-based functional minimization**: removes functionally equivalent rules post-GPU using a deterministic probe set, keeping only the highest-frequency representative per equivalence class
- ✅ **Dynamic VRAM-aware** batch and budget sizing (scales with available VRAM; baseline 8 GB)
- ✅ **Hot-rule biased** chain generation using Phase 1 results (60% hot-rule bias, configurable via `HOT_RULE_RATIO`)
- ✅ **User seed rules** support via `--seed-rules` to guide chain exploration (30% budget allocated to extending seeds)
- ✅ Per-depth chain budget overrides (depths 2–10)
- ✅ Unlimited result cap (no global ceiling)
- ✅ Full **hit counting** and frequency-ranked output
- ✅ Multi-device listing and explicit device selection by index or name substring
- ✅ Color-coded terminal output with live progress bars
- ✅ Configurable verbosity via `VERBOSE` flag

---

## ⚡ Why v2 Supersedes v1

| Aspect | v1 (`rulest.py`) | v2 (`rulest_v2.py`) |
|---|---|---|
| **Rule validation** | None — invalid rules passed to Hashcat | Full `HashcatRuleValidator` against GPU spec (max 31 ops) |
| **Functional minimization** | ❌ Not implemented | ✅ Signature-based deduplication via `minimize_by_signature`; removes 20–60% of raw candidates |
| **Rule set size** | ~2,700 static rules | 13,000+ GPU-validated Hashcat single rules across 9 categories |
| **Search strategy** | Naive BFS — every rule applied blindly | Phase 1 single-rule sweep → Phase S built-in seed extraction → Phase 2 hot-biased chain generation |
| **Built-in seed families** | ❌ Not implemented | ✅ Five families (A–E): numeric prepend/append, mixed, transform+digit, date patterns; always run as a dedicated pass independent of `--max-depth` |
| **Target lookup** | Python `set` (host RAM, per-result) | 16–64 MB Bloom filter uploaded once to GPU VRAM (FNV-1a, 4 hash functions) |
| **Chain state** | Temp `.tmp` files on disk per depth | In-memory, GPU buffer-based with proper release and `gc.collect()` |
| **Memory management** | Halve batch on OOM, no VRAM awareness | Dynamic sizing based on actual free VRAM estimate + 55% usage safety factor |
| **Hit counting** | ❌ Not implemented | ✅ Full `Counter`-based frequency tracking, sorted output |
| **Device selection** | First platform, first device | `--list-devices`, `--device` by index or name substring |
| **User seed rules** | ❌ Not supported | ✅ `--seed-rules` file; single seeds → Phase 1 + Phase 2 atoms; chain seeds → Phase 2 direct candidates |
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
git clone https://github.com/A113L/rulest.git
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
| `-d`, `--max-depth` | `2` | Maximum rule chain depth (1–12; depths >12 warned) |
| `-o`, `--output` | `rulest_output.txt` | Output file path |
| `--max-chains` | unlimited | Hard cap on total chains generated |
| `--target-hours` | `0.5` | Time budget in hours; controls chain generation budget |
| `--seed-rules` | None | File of user-supplied rules/chains. Single-rule seeds are injected into Phase 1 **and** used as Phase 2 chain atoms; multi-rule chain seeds are tested directly in Phase 2. Does not affect the built-in seed families (Phase S). |
| `--list-devices` | — | Print all available OpenCL devices and exit |
| `--device` | best GPU | Device index (e.g. `0`) or name substring (e.g. `NVIDIA`) |
| `--depth2-chains` | dynamic | Override chain generation limit for depth 2 |
| `--depth3-chains` | dynamic | Override chain generation limit for depth 3 |
| `--depth4-chains` through `--depth10-chains` | dynamic | Per-depth overrides up to depth 10 |
| `--bloom-mb` | dynamic | Override Bloom filter size (MB); 0 = auto-scale |
| `--sig-words` | `21` | Number of probe words used for functional deduplication (see [Functional Minimization](#-functional-minimization)) |
| `--min-word-len` | `10` | Minimum character length for probe words used in signature computation |
| `--allow-reject-rules` | off | Include rejection rules (normally excluded as GPU-incompatible) |
| `--debug` | off | Enable verbose output (sets `VERBOSE = True` at runtime) |

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
│  │  │ (16–64 MB   │    │  ┌─────────────────┐  │ │  │
│  │  │  VRAM)      │    │  │find_single_rules│  │ │  │
│  │  └─────────────┘    │  ├─────────────────┤  │ │  │
│  │                     │  │find_rule_chains │  │ │  │
│  │  Phase 1 ────────▶  │  └─────────────────┘  │ │  │
│  │  (all words ×       └───────────────────────┘ │  │
│  │   single rules)                               │  │
│  │                                               │  │
│  │  Phase S ────────▶  Built-in seed families   │  │
│  │  (Families A–E;      direct extraction pass,  │  │
│  │   always runs,       depth 2–9 seeds)         │  │
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

### Extraction Pipeline

**Phase 1 — Single Rule Sweep**
All base words are processed against every GPU-compatible single rule in parallel. The Bloom filter (built from the entire target wordlist and uploaded once) allows near-zero-cost hit detection on-device using FNV-1a hashing with 4 independent hash functions. Results feed a `Counter` of rule → hit frequency.

**Phase S — Built-in Numeric Seed Extraction**
A dedicated extraction pass that runs the five built-in seed families (A–E) through the GPU chain kernel. This phase always runs, regardless of `--max-depth` and the random-chain time budget. Depth-1 seeds are skipped (already covered by Phase 1); all multi-rule seed chains at depths 2 and above are tested directly against the Bloom filter. The prebuilt seed families are then forwarded to Phase 2 as scaffolding atoms to avoid regeneration and double-counting. See [Built-in Seed Families](#-built-in-seed-families) for a full description.

**Phase 2 — Informed Chain Generation**
Using Phase 1 hit data, chains are generated with a bias toward rules that already demonstrated effectiveness:
- **60%** of generated chains use hot rules from Phase 1 (`HOT_RULE_RATIO = 0.6`)
- **30%** of the budget extends known-good seed chains (`EXTENSION_RATIO = 0.3`)
- **10%** is allocated to random exploration

The remaining time budget (total `--target-hours` minus Phase 1 + Phase S duration) is split evenly across requested depths. User seed rules from `--seed-rules` are extended to deeper depths automatically.

### Bloom Filter

The on-GPU Bloom filter uses FNV-1a hashing with two seeds (`0xDEADBEEF` and `0xCAFEBABE`) and 4 hash functions, sized between **16 MB** (low-VRAM devices < 4 GB) and **64 MB** (default max). Size scales logarithmically with combined wordlist size.

### VRAM Management

Free VRAM is estimated as **55%** of total global memory (`VRAM_USAGE_FACTOR = 0.55`). All batch sizes, Bloom filter allocation, and chain budgets scale proportionally based on this estimate relative to an 8 GB baseline. Devices with fewer than 4 GB cap the Bloom filter at 32 MB; the batch floor prevents starvation on very constrained hardware.

---

## 🌱 Built-in Seed Families

`rulest_v2.py` ships with five deterministically generated seed families (A–E) that are built unconditionally at startup and run as **Phase S** — a dedicated GPU extraction pass that sits between Phase 1 and Phase 2. This pass is fully independent of `--max-depth` and the random-chain time budget: numeric and date-pattern chains are always tested, even when `--max-depth 1` is specified.

Depth-1 seeds (single-rule entries) are skipped in Phase S because Phase 1 already covers them. All multi-rule chains at depths 2 and above are submitted directly to the GPU chain kernel and checked against the Bloom filter. The prebuilt families are then forwarded to Phase 2 as scaffolding atoms so they can be used in chain extension without being re-tested or re-generated.

### Family A — Pure Prepend

Chains that prepend multi-digit numbers to a word by issuing one `^digit` operator per digit (right-to-left, so the number reads correctly). For example, prepending `12` produces the chain `^2 ^1`.

- Depths covered: 1–4 (10, 100, 1 000, 10 000 chains respectively)
- Example chains: `^0`, `^1 ^2`, `^9 ^8 ^7`, `^2 ^0 ^2 ^4`

### Family B — Pure Append

Chains that append multi-digit numbers to a word by issuing one `$digit` operator per digit (left-to-right). For example, appending `1990` produces `$1 $9 $9 $0`.

- Depths covered: 1–4 (same counts as Family A)
- Example chains: `$1`, `$1 $2`, `$1 $9 $9`, `$2 $0 $2 $4`

### Family C — Mixed Prepend/Append

All combinations of `^` and `$` operators with digits across every position. This covers patterns where numbers are split between prefix and suffix (e.g., `^1 $!` style numeric bookends).

- Depths covered: 1–4
- Total candidates: all `{^d, $d}^depth × digits^depth` combinations per depth

### Family D — Transform + Digit/Bracket

A single case-/position-transformation operator at position 1 (one of `l u c C t r d f E k K { } [ ]`), followed by 1–4 digit operators (`^d`, `$d`) or bracket operators (`[`, `]`).

| Depth | Structure | Examples |
|---|---|---|
| 2 | transform + 1 op | `u $1`, `l ^7`, `c [`, `C ]` |
| 3 | transform + 2 ops | `u ^1 $9`, `l $0 $8`, `c [ ]`, `t [ [` |
| 4 | transform + 3 ops | `u ^1 $2 ^9`, `c [ ] [`, `l $0 $5 $2` |
| 5 | transform + 4 ops | `u [ [ [ [`, `l ] ] ] ]`, `c [ ] [ ]`, `r ^1 $9 [ ]` |

All candidates are validated by `HashcatRuleValidator` before being added. Depths 2–5 are covered (depth 5 is included so that up to four `[`/`]` operators can follow a transform).

### Family E — Date Patterns

Date-pattern chains that cover the most common numeric date formats found in real passwords, in both append and prepend orientations. The date ranges used are:

- Days: `01`–`31`
- Months: `01`–`12`
- 2-digit years: `60`–`99` (1960s–1990s) and `00`–`30` (2000s–2030s)
- 4-digit years: `1960`–`2030`

| Format | Depth | Orientation |
|---|---|---|
| DDMM, MMDD, YYYY | 4 | append and prepend |
| Transform + 4-digit date | 5 | transform variant of every depth-4 date |
| DDMMYY, MMDDYY | 6 | append and prepend |
| 2–4 brackets + 4-digit date | 6, 7, 8 | bracket-prefix append/prepend |
| 1–2 brackets + 6-digit date | 7, 8 | bracket-prefix append/prepend |
| DDMMYYYY, MMDDYYYY | 8 | append and prepend |
| 1 bracket + 8-digit date | 9 | bracket-prefix append/prepend |

**Transform variants** (depth 5) apply every transform operator from Family D as a leading rule before the 4-digit date chain, e.g., `u $1 $9 $9 $0`, `c ^0 ^9 ^9 ^1`.

**Bracket-prefix variants** prepend 1–4 `[` or `]` operators before any date chain, allowing date extraction to succeed even when the base word has leading or trailing characters that need to be stripped.

> The seed families are always built with `max_seed_depth=4` in Phase S (capped internally regardless of `--max-depth`), so the maximum seed depth tested is 4 for Families A–D and up to 9 for Family E (date formats).

---

## 🔬 Functional Minimization

After GPU extraction, `rulest_v2.py` applies a **signature-based functional minimization** pass before writing the final output. This post-processing step removes redundant rules — rules that are syntactically different but produce identical outputs on real words — so the resulting ruleset is as compact as possible without losing coverage.

### Algorithm

1. **Build a probe set.** A fixed set of `--sig-words` words (default: 21) is sampled deterministically from the base wordlist, preferring words of length ≥ `--min-word-len` (default: 10). The selection uses a fixed random seed so results are reproducible across runs. If fewer long words are available than requested, shorter words are used to fill the set (with a warning).

2. **Compute each rule's signature.** Every candidate rule (or chain) is applied to every probe word using a pure-Python interpreter (`py_apply_chain`). The signature is the resulting tuple of transformed strings — one per probe word. Rules containing opcodes that cannot be emulated in Python are assigned the sentinel signature `('__UNSUPPORTED__',)` and are bucketed together.

3. **Group by signature.** Rules sharing an identical signature are considered **functionally equivalent** on the probe set. Only one representative survives from each group.

4. **Select the best representative.** Within each signature group, the rule with the **highest GPU hit-count** is kept. Ties are broken by preferring shorter chain depth, then lexicographic order.

5. **Write the minimized ruleset.** Surviving rules are written to the output file sorted by GPU frequency (descending). The file header records the number of probe words, the minimum word length, and how many equivalent rules were removed.

### Why It Matters

GPU Bloom filter screening (Phase 1 & 2) can yield thousands of candidates where many are functionally identical — for example, `c` (capitalize first letter) and a chain `l c` applied to an already-lowercase word produce the same output. Without minimization, the output file contains duplicate work that inflates Hashcat's rule-testing time without adding new candidate passwords.

Minimization typically removes **20–60% of raw candidates** depending on chain depth and wordlist diversity, leaving a tighter, faster ruleset with no reduction in theoretical coverage.

### Tuning

| Goal | Recommendation |
|---|---|
| More precise deduplication | Increase `--sig-words` (more probe words → fewer false equivalences) |
| Faster minimization pass | Decrease `--sig-words` (fewer probe words to evaluate per rule) |
| Bias toward longer probe words | Increase `--min-word-len` (longer words exercise more positional opcodes) |
| Reproduce a prior minimization exactly | Keep `--sig-words` and `--min-word-len` identical between runs (seed is fixed at 42) |

> **Note:** Signature equivalence is probabilistic — two rules might match on all 21 probe words yet differ on others. Increasing `--sig-words` reduces this false-equivalence rate but never eliminates it entirely. For production rulesets, values of 30–50 are reasonable; above 100 yields diminishing returns.

---

## 📚 Rule Categories

`GPUCompatibleRulesGenerator` generates rules across 9 categories, all pre-validated by `HashcatRuleValidator`:

| # | Category | Commands | Notes |
|---|---|---|---|
| 1 | **Simple rules** | `l u c C t r d f p z Z q E { } [ ] k K :` | No arguments |
| 2 | **Position-based (single digit)** | `T D L R + - . , ' z Z y Y` | Digit 0–9 |
| 3 | **Position-based (two digits)** | `x * O` | Two digits 0–9 each |
| 4 | **Prefix / Suffix / Delete-char** | `^ $ @` | Full printable ASCII (chars 32–126) |
| 5 | **Substitutions** | `s` | Leet-speak + alpha→digit/punctuation cross-product |
| 6 | **Insertion / Overwrite** | `i o` | Positions 0–9 × printable character set |
| 7 | **Extraction / Swap** | `x *` (non-equal positions) + `O` | Two-digit combos |
| 8 | **Duplication** | `p y Y z Z` + digit 1–9 | Word/char repetition variants |
| 9 | **Title case with separator** | `e` | Separator-triggered title casing |

> The identity rule (`:`) is always included and written first in the output for Hashcat compatibility.

---

## 🚫 GPU Command Support

The following commands are **not supported** on Hashcat's GPU engine and are automatically excluded during validation:

| Command(s) | Reason |
|---|---|
| `X` `4` `6` `M` | Memory operations — not available on GPU |
| `v` (three-char) | Not supported on GPU |
| `Q` | Quit rule — not GPU-compatible |
| `< > ! / ( ) = % ?` | Rejection rules — not GPU-compatible |
| `_` | Reject-if-length — not GPU-compatible |

Any rule exceeding **31 operations** is also rejected regardless of individual command validity.

---

## ⚙️ Configuration Constants

These constants are defined at the top of `rulest_v2.py` and can be tuned for advanced use:

| Constant | Default | Description |
|---|---|---|
| `VERBOSE` | `False` | Print per-rule validation messages and category counts; set at runtime via `--debug` |
| `VRAM_USAGE_FACTOR` | `0.55` | Fraction of device global memory to treat as free VRAM |
| `BLOOM_HASH_FUNCTIONS` | `4` | Number of FNV-1a hash functions in Bloom filter |
| `BLOOM_FILTER_MAX_MB` | `256` | Maximum Bloom filter allocation (MB); override at runtime with `--bloom-mb` |
| `HOT_RULE_RATIO` | `0.6` | Fraction of Phase 2 chains biased toward hot rules |
| `EXTENSION_RATIO` | `0.3` | Fraction of Phase 2 budget allocated to seed extension |
| `TIME_SAFETY_FACTOR` | `0.9` | Multiplier applied to time-budget combo estimates |
| `MAX_GPU_RULES` | `31` | Maximum operations allowed per rule chain |
| `BASELINE_COMBOS_PER_SEC` | `120,000,000` | Estimated throughput on a capable GPU |
| `LOW_END_COMBOS_PER_SEC` | `40,000,000` | Throughput fallback for devices with < 20 compute units |
| `MAX_WORD_LEN` | `256` | Maximum word length accepted from wordlists |
| `MAX_RULE_LEN` | `16` | Maximum single rule string length in GPU buffers |
| `MAX_OUTPUT_LEN` | `512` | Maximum transformed word output length in GPU buffers |
| `MAX_CHAIN_STRING_LEN` | `128` | Maximum chained rule string length in GPU buffers |
| `MAX_HASHCAT_CHAIN` | `31` | Maximum number of rules in a single Hashcat chain |
| `DEFAULT_SIG_WORDS` | `21` | Default number of probe words for signature-based minimization |
| `DEFAULT_MIN_WORD_LEN` | `10` | Default minimum word length for probe word selection |

---

## 📄 Output Format

`rulest_output.txt` (or your specified `-o` path):

```
# rulest — GPU-Compatible Hashcat Rules Engine
# Generated      : 2025-08-01 14:32:07
# Base           : rockyou.txt
# Target         : target_plain.txt
# Depth          : 1–3
# Bloom          : 256 MB
#
# GPU raw candidates      : 9,214  (bloom hits, includes false positives)
# Post-processing         : signature-based minimization
#   Probe words           : 21  (min length 10)
#   Equiv. rules removed  : 4,393
#
# Rules kept     : 4,821  (d1:3104  d2:1512  d3:205)
# Sorted by      : GPU frequency (descending, UTF-8)
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
- Rules are sorted by hit frequency (descending), then by chain depth, then alphabetically
- The header records both the raw Bloom candidate count and the post-minimization count, so you can see exactly how many equivalent rules were removed
- All rules are guaranteed GPU-valid (max 31 ops, correct argument syntax)
- Encoding is `utf-8`

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
| Reduce terminal noise | Set `VERBOSE = False` in the script header or omit `--debug` |
| Increase hot-rule aggressiveness | Raise `HOT_RULE_RATIO` toward `1.0` (reduces random exploration) |

### VRAM Scaling Reference

| Available VRAM | Scale Factor | Bloom Filter Cap |
|---|---|---|
| < 4 GB | 0.25–0.5× | 32 MB |
| 4–8 GB | 0.5–1.0× | 64 MB |
| 8 GB+ | 1.0× (full) | 64 MB |

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

**Iterative refinement workflow:**
```bash
# Pass 1 — fast sweep for single rules
python rulest_v2.py rockyou.txt target.txt -d 1 --target-hours 0.25 -o pass1.txt

# Pass 2 — chain from pass 1 results
python rulest_v2.py rockyou.txt target.txt -d 3 --target-hours 1.0 \
  --seed-rules pass1.txt -o pass2.txt

# Pass 3 — deep dive seeded from pass 2
python rulest_v2.py rockyou.txt target.txt -d 5 --target-hours 4.0 \
  --seed-rules pass2.txt -o pass3_final.txt
```

---

## 📝 License

MIT

## Credits

https://github.com/synacktiv/rulesfinder
