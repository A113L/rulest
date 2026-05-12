# 🔐 GPU-Accelerated Hashcat Rule Extractor
 
> Extract and chain Hashcat-compatible rules from wordlists using OpenCL GPU acceleration.
 
---
 
## 📋 Table of Contents
 
- [Overview](#-overview)
- [Scripts](#-scripts)
  - [rulest.py — v1 (BFS, Legacy)](#rulestpy--v1-bfs-legacy)
  - [rulest\_v2.py — v2 (Recommended)](#rulest_v2py--v2-recommended)
- [What's New in v2 — Phase 3 GA Fixes](#-whats-new-in-v2--phase-3-ga-fixes)
- [Why v2 Supersedes v1](#-why-v2-supersedes-v1)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [rulest\_v2.py — Full Reference](#rulest_v2py--full-reference)
- [Architecture (v2)](#-architecture-v2)
  - [Extraction Pipeline](#extraction-pipeline)
- [Built-in Seed Families (A–M)](#-built-in-seed-families)
- [Phase 0 — Token-Strip Rule Extraction](#-phase-0--token-strip-rule-extraction)
- [Phase 3 — Genetic Algorithm](#-phase-3--genetic-algorithm)
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
- ✅ **Phase 0 — Token-Strip Rule Extraction** (`--token-strip`): optional CPU pre-pass running **6 extraction modes** across every target word with full multiprocessing support (`--token-strip-workers`); decomposes passwords into base stem + rule chain using letter, digit, reverse, delete-edge, dup/fold modes (original five) plus insert (`iNX`); single-rule discoveries feed Phase 1; multi-rule chains are injected into Phase S and Phase 2
- ✅ **Four-phase extraction**: token-strip pre-pass (Phase 0) → single-rule sweep (Phase 1) → built-in seed pass (Phase S) → informed chain generation (Phase 2)
- ✅ **Built-in seed families** (A–M): thirteen deterministically generated seed families covering numeric prefixes/suffixes, mixed prepend/append, transform+digit combos, date patterns, special-character append/prepend/transform/combo patterns, leet substitutions, double-transform chains, special-before-digit patterns, and leet+transform combos — run by default as a dedicated extraction pass, independent of `--max-depth` and the random-chain time budget; can be skipped with `--no-builtin-seeds`
- ✅ **Signature-based functional minimization**: removes functionally equivalent rules post-GPU using a deterministic probe set, keeping only the highest-frequency representative per equivalence class
- ✅ **Dynamic VRAM-aware** batch and budget sizing (scales with available VRAM; baseline 8 GB)
- ✅ **Hot-rule biased** chain generation using Phase 1 results (60% hot-rule bias, configurable via `HOT_RULE_RATIO`)
- ✅ **User seed rules** support via `--seed-rules` to guide chain exploration (30% budget allocated to extending seeds)
- ✅ **Phase 3 Genetic Algorithm** (`--genetic`): optional evolutionary search that runs after Phase 2, guided by novelty-weighted bloom-filter coverage — breeds chains that Phase 1/S/2 have not yet found; dedicated time reservation (20 % of `--target-hours`, min 120 s) guarantees the GA always runs; stagnation guard refreshes the bottom 30 % of the population after 5 flat generations
- ✅ Per-depth chain budget overrides (depths 2–10)
- ✅ Unlimited result cap (no global ceiling)
- ✅ Full **hit counting** and frequency-ranked output
- ✅ Multi-device listing and explicit device selection by index or name substring
- ✅ Color-coded terminal output with live progress bars
- ✅ Configurable verbosity via `VERBOSE` flag
 
---
 
## 🆕 What's New
 
Three root-cause bugs that caused Phase 3 to produce identical results with and without `--genetic` were identified and fixed:
 
| Change | Description |
|---|---|
| **Phase 0 — Token-Strip** | New optional CPU pre-pass (`--token-strip`) that runs before Phase 1. Reverse-engineers exact rule chains from target passwords by splitting each word into a base stem + boundary tokens. Single-rule discoveries are injected into Phase 1; multi-rule chains are injected into Phase S and Phase 2. Supersedes ad-hoc leet/case rules with ground-truth verified chains. |
| **Novelty-weighted fitness** | Chains not yet discovered by Phase 1/S/2 receive a **2x fitness bonus** during GA selection. Previously fitness was raw bloom hits, so already-known high-scorers dominated every generation and the GA just rediscovered Phase 2 results. |
| **Unexplored-seed initial population** | The 40 % Phase-S fill slot now prefers seeds **absent from `known_rules`**. Previously it used top-hit Phase-S chains — all already in `all_counts` — so the GA started with a population of mostly-known rules. Depth-3+ bias (70 %) is also applied to seeded and fill portions when `--max-depth >= 3`. |
| **Dedicated time reservation** | Phase 3 now reserves **20 % of `--target-hours`** (min 120 s) from Phase 2's budget before Phase 2 begins. Previously the GA ran on leftover time; with the default `--target-hours 0.5` Phase 2 consumed everything and Phase 3 was silently skipped. |
| **Stagnation guard** | If best fitness does not improve for 5 consecutive generations, the bottom 30 % of the population is replaced with fresh random chains (depth-3+ biased). |
| **Depth-2 warning** | When `--max-depth 2`, the GA emits a warning: Phase 2 already exhaustively covers depth-2, so the GA cannot add new rules at that depth. Use `--max-depth 3` or higher. |
 
---
 
## ⚡ Why v2 Supersedes v1
 
| Aspect | v1 (`rulest.py`) | v2 (`rulest_v2.py`) |
|---|---|---|
| **Rule validation** | None — invalid rules passed to Hashcat | Full `HashcatRuleValidator` against GPU spec (max 31 ops) |
| **Functional minimization** | ❌ Not implemented | ✅ Signature-based deduplication via `minimize_by_signature`; removes 20–60% of raw candidates |
| **Rule set size** | ~2,700 static rules | 5,600+ GPU-validated Hashcat single rules across 9 categories |
| **Search strategy** | Naive BFS — every rule applied blindly | Phase 1 single-rule sweep → Phase S built-in seed extraction → Phase 2 hot-biased chain generation → Phase 3 GA (optional) |
| **Built-in seed families** | ❌ Not implemented | ✅ Thirteen families (A–M): numeric prepend/append, mixed, transform+digit, date patterns, special-char append/prepend/transform/combo (F–I), leet substitutions (J), double-transform chains (K), special-before-digit (L), leet+transform (M); run by default as a dedicated pass independent of `--max-depth`; disable with `--no-builtin-seeds` |
| **Token-strip pre-pass** | ❌ Not implemented | ✅ Phase 0 (`--token-strip`): **6 extraction modes** with multiprocessing; new flags: `--token-strip-workers`, `--token-strip-chunk-size` |
| **Genetic algorithm** | ❌ Not implemented | ✅ Optional Phase 3 (`--genetic`): novelty-weighted evolutionary search (2× fitness bonus for chains not yet in known_rules) with dedicated time reservation (20 % of `--target-hours`, min 120 s) and stagnation-triggered population refresh; discovers deep-chain patterns that random Phase 2 sampling misses |
| **Target lookup** | Python `set` (host RAM, per-result) | 16–256 MB Bloom filter uploaded once to GPU VRAM (FNV-1a, 4 hash functions) |
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
 
**Phase 3 GA (optional):** Where Phase 2 still samples randomly within the hot-rule-biased pool, the genetic algorithm *evolves* chains by recombining and mutating the highest-scoring ones each generation. A **2× novelty bonus** is applied to chains not already discovered by Phase 1/S/2, so selection pressure drives the population toward genuinely new territory rather than re-converging on known results. A **20 % time reservation** (minimum 120 s) ensures the GA always receives a meaningful budget rather than whatever Phase 2 leaves behind. This is particularly effective at depth ≥ 3 where the search space (`|pool|^depth`) is too large for exhaustive or purely random coverage.
 
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
| `--max-depth` | `2` | Maximum rule chain depth (1–31; depths >31 capped with a warning) |
| `-o`, `--output` | `rulest_output.txt` | Output file path |
| `--max-chains` | unlimited | Hard cap on total chains generated |
| `--target-hours` | `0.5` | Time budget in hours; controls chain generation budget for Phase 2 and Phase 3 |
| `--seed-rules` | None | File of user-supplied rules/chains. Single-rule seeds are injected into Phase 1 **and** used as Phase 2 chain atoms; multi-rule chain seeds are tested directly in Phase 2. Does not affect the built-in seed families (Phase S). |
| `--list-devices` | — | Print all available OpenCL devices and exit |
| `--device` | best GPU | Device index (e.g. `0`) or name substring (e.g. `NVIDIA`) |
| `--depth2-chains` | dynamic | Override chain generation limit for depth 2 |
| `--depth3-chains` | dynamic | Override chain generation limit for depth 3 |
| `--depth4-chains` through `--depth10-chains` | dynamic | Per-depth overrides up to depth 10 |
| `--bloom-mb` | dynamic | Override Bloom filter size (MB); 0 = auto-scale |
| `--allow-reject-rules` | off | Include rejection rules (normally excluded as GPU-incompatible) |
| `--no-builtin-seeds` | off | Disable the built-in seed families (Phase S). By default Phase S always runs; pass this flag to skip it entirely and rely solely on Phase 1 atomic rules and Phase 2 random chains. Useful for faster runs or when supplying all seeds via `--seed-rules`. Skips all thirteen families (A–M): numeric, date-pattern, special-character, leet substitution, double-transform, special-before-digit, and leet+transform. |
| `--debug` | off | Enable verbose output (sets `VERBOSE = True` at runtime) |

#### Phase 3 — Genetic Algorithm Arguments

| Flag | Default | Description |
|---|---|---|
| `--genetic` | off | Enable Phase 3 genetic algorithm rule evolution. A dedicated time budget of 20 % of `--target-hours` (min 120 s) is reserved for Phase 3 before Phase 2 begins. Has no effect at `--max-depth 1` (chains require at least depth 2); emits a warning at `--max-depth 2` since Phase 2 already covers depth-2 exhaustively — use `--max-depth 3` or higher for meaningful GA output. |
| `--genetic-generations` | `50` | Maximum number of GA generations. Each generation performs a full GPU fitness evaluation of the entire population, so larger values extend runtime proportionally. |
| `--genetic-pop` | `200` | GA population size — number of rule chains evaluated per generation. Larger populations improve search coverage at the cost of more GPU evaluations per generation. |
| `--genetic-elite` | `0.15` | Fraction of top-scoring individuals carried unchanged into the next generation (elitism). Must be strictly between 0.0 and 1.0. Higher values stabilise convergence; lower values increase diversity. |

#### Phase 0 — Token Strip Arguments

| Flag | Default | Description |
|---|---|---|
| `--token-strip` | off | Enable Phase 0: empirical CPU-only rule extraction by decomposing target passwords into stem + transform rules. Discovered rules are injected into the Phase 1 atomic pool (single-rule) and Phase S `sbd` (multi-rule chains) before any GPU work begins. |
| `--token-strip-min-stem` | `4` | Minimum stem length after token decoding. Shorter stems produce noisy rules and are discarded. |
| `--token-strip-max-prefix` | `4` | Maximum number of boundary characters to strip from the start of a target word. These become prepend (`^`) rules. |
| `--token-strip-max-suffix` | `4` | Maximum number of boundary characters to strip from the end of a target word. These become append (`$`) rules. |
| `--token-strip-min-leet-amb` | `3` | Maximum number of ambiguous leet positions per word. A position is ambiguous when its leet char maps to more than one base letter (e.g. `1` → `i` or `l`). Higher values allow more combinations but increase CPU time. |
| `--token-strip-workers` | `0` | Number of worker processes for Phase 0. `0` = all CPU cores. Set to `1` to disable multiprocessing. |
| `--token-strip-chunk-size` | `0` | Words per worker chunk. `0` = auto-sized to `max(500, n_words / (workers × 4))`. Smaller values improve progress resolution; larger values reduce dispatch overhead. |
 
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
│  │  Phase 0 ────────▶  TokenStripExtractor       │  │
│  │  (--token-strip;     CPU-only  ·  optional;   │  │
│  │   default off;       6 modes, MP workers;    │  │
│  │   runs pre-GPU)      singles → Phase 1 atoms; │  │
│  │                      chains  → Ph.S + Ph.2    │  │
│  │                                               │  │
│  │  ┌─────────────┐    ┌───────────────────────┐ │  │
│  │  │ Bloom Filter│    │  OpenCL Kernel        │ │  │
│  │  │ (16–256 MB  │    │  ┌─────────────────┐  │ │  │
│  │  │  VRAM)      │    │  │find_single_rules│  │ │  │
│  │  └─────────────┘    │  ├─────────────────┤  │ │  │
│  │                     │  │find_rule_chains │  │ │  │
│  │  Phase 1 ────────▶  │  └─────────────────┘  │ │  │
│  │  (all words ×       └───────────────────────┘ │  │
│  │   single rules)                               │  │
│  │                                               │  │
│  │  Phase S ────────▶  Built-in seed families    │  │
│  │  (Families A–M;      direct extraction pass,  │  │
│  │   default on;        depth 2–9 seeds)         │  │
│  │                                               │  │
│  │  Phase 2 ────────▶  Informed chain generation │  │
│  │  (hot-biased,        + seed extension         │  │
│  │   VRAM-budgeted)                              │  │
│  │                                               │  │
│  │  Phase 3 ────────▶  GeneticRuleEvolver        │  │
│  │  (--genetic;         novelty-weighted fitness │  │
│  │   reserved 20 %      (2× bonus for new chains)│  │
│  │   time budget;       tournament select +      │  │
│  │   stagnation guard)  crossover + mutation     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         │
         ▼
  HashcatRuleValidator  →  minimize_by_signature  →  GPU-safe output (.rule file)
```
 
### Extraction Pipeline
 
**Phase 0 — Token-Strip Rule Extraction** *(optional, `--token-strip`)*

A CPU-only pre-pass running **14 extraction modes** with multiprocessing. Seven inverted indexes are built once in the main process and inherited by all workers via CoW fork (zero serialization on Linux/macOS; passed via `initargs` on Windows/spawn).

Original 5 modes: letter (case + leet + boundary ops), digit (digit-stem with dynamic boundary), reverse (`r` prefix), delete-edge (`[`/`]` prefix), dup/fold (`d`/`f`), insert (`iNX`).

Validated single-rule discoveries are merged into Phase 1's atomic rule pool; multi-rule chains are injected into Phase S's seed-by-depth table and forwarded to Phase 2. Use `--token-strip-no-new-modes` to restrict to the original 5 modes; `--token-strip-workers` controls parallelism.

See [Phase 0 — Token-Strip Rule Extraction](#-phase-0--token-strip-rule-extraction) for full details.

**Phase 1 — Single Rule Sweep**
All base words are processed against every GPU-compatible single rule in parallel. The Bloom filter (built from the entire target wordlist and uploaded once) allows near-zero-cost hit detection on-device using FNV-1a hashing with 4 independent hash functions. Results feed a `Counter` of rule → hit frequency.
 
**Phase S — Built-in Seed Extraction**
A dedicated extraction pass that runs the thirteen built-in seed families (A–M) through the GPU chain kernel. This phase runs by default, regardless of `--max-depth` and the random-chain time budget; it can be disabled with `--no-builtin-seeds`. Depth-1 seeds are skipped (already covered by Phase 1); all multi-rule seed chains at depths 2 and above are tested directly against the Bloom filter. The prebuilt seed families are then forwarded to Phase 2 as scaffolding atoms to avoid regeneration and double-counting. See [Built-in Seed Families (A–M)](#-built-in-seed-families) for a full description.
 
**Phase 2 — Informed Chain Generation**
Using Phase 1 hit data, chains are generated with a bias toward rules that already demonstrated effectiveness:
- **60%** of generated chains use hot rules from Phase 1 (`HOT_RULE_RATIO = 0.6`)
- **30%** of the budget extends known-good seed chains (`EXTENSION_RATIO = 0.3`)
- **10%** is allocated to random exploration
 
The remaining time budget (total `--target-hours` minus Phase 1 + Phase S duration) is split evenly across requested depths. User seed rules from `--seed-rules` are extended to deeper depths automatically.
 
**Phase 3 — Genetic Algorithm** *(optional, `--genetic`)*
An evolutionary search that runs after Phase 2. A dedicated budget of **20 % of `--target-hours`** (minimum 120 s) is reserved for Phase 3 before Phase 2 begins, so the GA is always guaranteed a meaningful run. Fitness is novelty-weighted: chains not already present in the Phase 1/S/2 results receive a 2× multiplier, driving the population toward genuinely new discoveries. A stagnation guard replaces the bottom 30 % of the population with fresh random chains if the best score does not improve for 5 consecutive generations. See [Phase 3 — Genetic Algorithm](#-phase-3--genetic-algorithm) for full details.
 
**Post-processing — Signature-Based Minimization**
After all phases complete, every candidate rule is applied to the built-in probe set in pure Python. Rules producing identical outputs on all probe words are grouped; only the highest-GPU-hit representative per group survives. See [Functional Minimization](#-functional-minimization) for details.
 
### Bloom Filter
 
The on-GPU Bloom filter uses FNV-1a hashing with two seeds (`0xDEADBEEF` and `0xCAFEBABE`) and 4 hash functions, sized between **16 MB** (low-VRAM devices < 4 GB) and **256 MB** (default max; override with `--bloom-mb`). Size scales logarithmically with combined wordlist size.
 
### VRAM Management
 
Free VRAM is estimated as **55%** of total global memory (`VRAM_USAGE_FACTOR = 0.55`). All batch sizes, Bloom filter allocation, and chain budgets scale proportionally based on this estimate relative to an 8 GB baseline. Devices with fewer than 4 GB cap the Bloom filter at 32 MB; the batch floor prevents starvation on very constrained hardware.
 
---
 
## 🌱 Built-in Seed Families (A–M)
 
`rulest_v2.py` ships with thirteen deterministically generated seed families (A–M) that are built at startup and run as **Phase S** — a dedicated GPU extraction pass that sits between Phase 1 and Phase 2. This pass runs by default and is fully independent of `--max-depth` and the random-chain time budget: numeric, date-pattern, special-character, leet-substitution, and transform chains are tested even when `--max-depth 1` is specified. To skip Phase S entirely, pass `--no-builtin-seeds`; this is useful for faster runs when you are supplying all seeds yourself via `--seed-rules` or when benchmarking the contribution of the built-in families.
 
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
 
> The seed families are always built with `max_seed_depth=4` in Phase S (capped internally regardless of `--max-depth`), so the maximum seed depth tested is 4 for Families A–D and F–M, and up to 9 for Family E (date formats).
 
### Family F — Pure Append Special Chars
 
Chains that append one, two, or three special characters from the top-15 set to a word using `$char` operators. Depth 3 was added in v2.1 to cover three-char suffixes such as `!!!` or `!@#` that appear in older forced-complexity passwords.
 
- Depths covered: 1–3
- Special chars: `! @ # $ % ^ & * ? . - _ + ( )`
- Seed counts: 15 (d1) · 225 (d2) · 3 375 (d3)
- Example chains: `$!`, `$@ $#`, `$! $! $!`, `$! $@ $#`
 
### Family G — Pure Prepend Special Chars
 
Chains that prepend one, two, or three special characters from the top-15 set to a word using `^char` operators (right-to-left order so the final string reads left-to-right). Depth 3 mirrors the Family F extension.
 
- Depths covered: 1–3
- Special chars: `! @ # $ % ^ & * ? . - _ + ( )`
- Seed counts: 15 (d1) · 225 (d2) · 3 375 (d3)
- Example chains: `^!`, `^@ ^!`, `^# ^@ ^!`
 
### Family H — Transform + Special Char
 
A single case-/position-transformation operator followed by one or two special-character append or prepend operators. Covers patterns like `capitalize + append !`.
 
- Depths covered: 2–3
- Transform operators: same set as Family D (`l u c C t r d f E k K { } [ ]`)
- Special chars: top-15 set
- Example chains: `u $!`, `c ^@`, `l $! $@`, `r ^# $%`
 
### Family I — Digit(s) + Special Char
 
Chains combining one or more digit operators with a special character from the core-7 set. Covers the ubiquitous `word123!` / `!word123` patterns — digits first, special char last.
 
- Depths covered: 2–4
- Core-7 special chars: `! @ # $ % * ?`
- Example chains: `$1 $!`, `^! ^1`, `$1 $2 $3 $!`, `^@ ^3 ^2 ^1`
 
### Family J — Leet Substitutions
 
The ten most common character→character leet-speak substitutions used in real passwords, applied as `sXY` opcode rules.
 
**Core leet pairs** (ordered by real-world frequency):
 
| Rule | Substitution | Example |
|---|---|---|
| `sa@` | a → @ | `password` → `p@ssword` |
| `se3` | e → 3 | `secret` → `s3cr3t` |
| `so0` | o → 0 | `football` → `f00tball` |
| `si1` | i → 1 | `login` → `log1n` |
| `sl1` | l → 1 | `leet` → `1eet` |
| `ss5` | s → 5 | `pass` → `pa55` |
| `ss$` | s → $ | `pass` → `pa$$` |
| `st7` | t → 7 | `test` → `7es7` |
| `sa4` | a → 4 | `admin` → `4dmin` |
| `si!` | i → ! | `bitcoin` → `b!tco!n` |
 
**Depth breakdown:**
- **Depth 1 (~10 seeds):** Pure substitution. These are also seen in Phase 1 as atomic rules; explicit seeding guarantees they are never missed.
- **Depth 2a (~340 seeds):** Each leet op combined with one digit or special-char append/prepend. Catches `p@ssword1`, `p@ssword!`, `1p@ssword`, etc.
- **Depth 2b (~90 seeds):** Two distinct leet ops chained. Catches multi-substitution passwords like `p@ssw0rd` (= `sa@` + `so0`) and `s3cur1ty` (= `se3` + `si1`), which previously relied on random Phase 2 discovery.
 
### Family K — Double-Transform Chains
 
All ordered pairs of pure structural transformation operators (no digit or special-char appends). None of these chains are generated by any other family — Family D always pairs a transform with a digit or bracket op, never with a second transform — so Family K adds entirely new coverage.
 
- Depth covered: 2 only
- Seed count: 15 × 15 = **225 chains**
- Transform ops: `l u c C t r d f E k K { } [ ]`
 
| Example chain | Effect |
|---|---|
| `c r` | Capitalize then reverse → `"password"` → `"drowssaP"` |
| `u d` | Uppercase then duplicate → `"abc"` → `"ABCABC"` |
| `t f` | Toggle case then fold → `"Hello"` → `"hELLOolleh"` |
| `E l` | Title-case then lowercase (no-op on plain words; meaningful after prior transforms) |
| `l ]` | Lowercase then drop last char |
| `c {` | Capitalize then rotate left |
 
### Family L — Special-before-Digit Patterns
 
The reverse orientation of Family I. Family I covers `word<digits><sp>` (digits first, then special char). Family L covers `word<sp><digits>` (special char first, then digits) — patterns like `word!12` and `!12word`.
 
- Depths covered: 2–3
- Core-7 special chars: `! @ # $ % * ?`
- Append orientation: `$sp $d₁ … $dₙ` → `word!12`
- Prepend orientation: `^dₙ … ^d₁ ^sp` → `12!word`
 
| Depth | Seeds | Orientation | Example |
|---|---|---|---|
| 2 | 70 append + 70 prepend = **140** | 7 sp × 10 d | `$! $1`, `^1 ^!` |
| 3 | 700 append + 700 prepend = **1 400** | 7 sp × 100 dd | `$! $1 $2`, `^2 ^1 ^!` |
 
> **Prepend ordering note:** to produce `12!word`, hashcat prepend ops are applied right-to-left. The chain `^2 ^1 ^!` reads: first prepend `!` → `!word`, then `1` → `1!word`, then `2` → `12!word`. Family L constructs chains accordingly.
 
### Family M — Leet + Transform Chains
 
Every leet substitution op paired with every structural transform op in both orderings. This closes the gap between leet-only (Family J) and transform-only (Family K) chains, covering patterns where a word is both transformed in case/structure *and* leet-substituted.
 
- Depth covered: 2 only
- Seed count: 10 leet × 15 transforms × 2 orderings = **300 chains** (≈280–295 unique after dedup)
 
| Chain | Effect | Example |
|---|---|---|
| `sa@ c` | leet then capitalize | `password` → `p@ssword` → `P@ssword` |
| `c sa@` | capitalize then leet | `password` → `Password` → `P@ssword` |
| `so0 u` | leet then uppercase | `password` → `passw0rd` → `PASSW0RD` |
| `u sa@` | uppercase then leet | `password` → `PASSWORD` → `PASSWORD`* |
| `sl1 r` | leet then reverse | `leet` → `1eet` → `tee1` |
 
> \* `u sa@` on a word with no remaining lowercase `a` after uppercasing is a no-op for the substitution step — this is correct hashcat behavior and is handled transparently.
 
---
 
> **Special-character sets:**  
> Top-15 (Families F/G/H): `! @ # $ % ^ & * ? . - _ + ( )`  
> Core-7 (Families I/L): `! @ # $ % * ?`
 
---
 
### Phase S seed count summary (at `--max-depth 4`)
 
| Family | Description | New in | Approx. seeds (d≥2) |
|---|---|---|---|
| A | Pure prepend digits | v2 | 11 100 |
| B | Pure append digits | v2 | 11 100 |
| C | Mixed prepend/append | v2 | ~168 000 |
| D | Transform + digit/bracket | v2 | ~167 000 |
| E | Date patterns | v2 | ~varies |
| F | Append special chars | v2 | 3 600 |
| G | Prepend special chars | v2 | 3 600 |
| H | Transform + special char | v2 | 13 950 |
| I | Digit(s) + special char | v2 | 15 540 |
| J | Leet substitutions | v2 | ~520 |
| K | Double-transform | v2 | 225 |
| L | Special-before-digit | v2 | 1 540 |
| M | Leet + transform | v2 | ~300 |
 
---

## 🔤 Phase 0 — Token-Strip Rule Extraction

Phase 0 is an optional **CPU-only pre-pass** activated by `--token-strip`. It runs before Phase 1, reverse-engineering exact Hashcat rule chains by decomposing each target password into its constituent token classes and verifying that the derived chain correctly reconstructs the original password when applied to a stem from the base wordlist.

### Extraction Modes

Phase 0 runs **14 extraction modes** per target word. The original five modes are always active; the nine v1.1 modes run by default and can be disabled with `--token-strip-no-new-modes`.

#### Original 5 modes

| Mode | Hashcat rules | Trigger | Description |
|---|---|---|---|
| **letter** | case ops + leet `sXY` + `^X $X` | letters dominate (or tie) | Boundary = digits+specials; stem = lowercase letters + leet decoding |
| **digit** | `^X $X` | digits dominate | Boundary = letters+specials; stem = pure digit sequence |
| **reverse** | `r` + combinations | depth ≥ 2 | Prepends `r`; remainder handled by letter/digit modes |
| **delete-edge** | `[` or `]` + combinations | depth ≥ 2 | Strips one boundary char from start or end first |
| **dup/fold** | `d`, `f` | word ≥ 2× min-stem | Detects passwords formed by duplicating or folding a base word |

#### New v1.1 modes

| Mode | Hashcat rules | Example |
|---|---|---|
| **insert** | `iNX` | `paassword` → `password` via `i1a` |
| **swap** | `k`, `K`, `*MN` | `apssword` → `password` via `k` |
| **range** | `xNM`, `ONM` | `sswor` → `password` via `x26`; `passd` via `O43` |
| **char-dup** | `zN`, `ZN`, `q` | `ppassword` → `password` via `z1` |
| **ascii-transform** | `+N`, `-N`, `LN`, `RN` | `qbttxpse` → `password` via `+0 +1 … +7` |
| **truncate** | `'N` | `passwor` → `password` via `'6` |
| **purge** | `@X` | `pssword` → `password` via `@a` |
| **repeat** | `pN` | `passwordpassword` → `password` via `p1` |
| **sep-title** | `eX` | `Pass-Word` → `pass-word` via `e-` |

### Multiprocessing

Phase 0 fans out work across all CPU cores using `multiprocessing.Pool`. Seven inverted indexes are built once in the main process before the pool is created:

| Index | Used by | Lookup cost |
|---|---|---|
| `base_by_len` | insert, ascii-transform | O(1) |
| `base_lower_idx` | sep-title | O(1) |
| `purge_idx` | purge | O(1) |
| `substr_idx` | range `xNM` | O(1) |
| `omit_idx` | range `ONM` | O(1) |
| `prefix_idx` | truncate | O(1) |
| `ascii_idx` | ascii-transform | O(1) |

On Linux/macOS (fork context) workers inherit all indexes via copy-on-write — zero serialization overhead. On Windows (spawn context) they are passed via `initargs`. Control parallelism with `--token-strip-workers` (default: all cores) and `--token-strip-chunk-size` (default: auto).

### Token Model

Each character in a target password is classified into one of four roles:

| Character type | Role | Handling |
|---|---|---|
| Lowercase letters | Base stem characters | Passed through unchanged |
| Uppercase letters | Stem chars (lowercased) + case op | Encoded as `u` / `c` / `C` / `t` / `E`; per-position `TN` used only as fallback |
| Leet substitutions | Stem chars (reversed) + `sXY` op | e.g. `@` → `a` (rule `sa@`), `3` → `e` (rule `se3`) |
| Boundary digits / specials | Prepend `^` / append `$` ops | Must form a contiguous prefix and/or suffix |

### Case-Op Optimization

When the cased middle segment can be expressed by a single compact operator (`u`, `c`, `C`, `t`, `E`), Phase 0 uses that operator exclusively. Per-position `TN` ops (e.g. `T0 T1 T2 T3 T4 T5 T6 T7 T8`) are only emitted when no such single-op equivalent exists — for example, irregular mixed-case patterns like `pAsSwOrD`. This prevents long depth-9 `TN` chains from dominating Phase 0 output and avoids bloating Phase S with chains that minimization cannot collapse (since `T0..T8` and `u` have different signatures on probe words longer than 9 characters).

### Decomposition Examples

| Target password | Base stem | Rule chain |
|---|---|---|
| `12password!` | `password` | `^2 ^1 $!` |
| `Password!` | `password` | `c $!` |
| `P@ssw0rd` | `password` | `c sa@ so0` |
| `ADMIN123` | `admin` | `u $1 $2 $3` |
| `p@SSW0RD` | `password` | `C sa@ so0` |
| `L33t!` | `leet` | `c se3 $!` |
| `paassword` | `password` | `i1a` |
| `Pass-Word` | `pass-word` | `e-` |
| `ppassword` | `password` | `z1` |
| `pssword` | `password` | `@a` |

### Algorithm

For each target word, every `(prefix_len, suffix_len)` split within the configured bounds is tried (letter/digit/reverse/delete-edge modes). The v1.1 modes use index lookups instead of exhaustive splits:

1. The boundary prefix/suffix must consist entirely of digits and common special characters. Scanning stops as soon as a non-boundary character is encountered.
2. The middle segment is leet-decoded: ambiguous chars like `1` (→ `i` or `l`) are branched exhaustively up to `--token-strip-min-leet-amb` positions.
3. Uppercase letters are normalised to lowercase; the case pattern is encoded as the shortest matching op (`u` / `c` / `C` / `t` / `E`). Per-position `TN` ops are used only when no single-op equivalent covers the pattern.
4. The resulting all-lowercase stem is looked up in the base wordlist set.
5. The full rule chain (case + leet + prepend + append) is verified via `py_apply_chain` — must reproduce the original target word exactly.
6. The chain is validated by `HashcatRuleValidator` (max 31 ops, GPU-compatible syntax).

### Injection into Later Phases

| Discovery type | Injected into |
|---|---|
| Single-rule chains (depth 1) | Phase 1 atomic rule pool (deduplicated) |
| Multi-rule chains (depth ≥ 2) | Phase S seed-by-depth table + Phase 2 seed chains |

When both `--token-strip` and Phase S are active (default), Phase 0 multi-rule chains are injected into Phase S's `sbd` pool before the GPU seed extraction pass, giving them the same full GPU coverage sweep as families A–M. If a Phase 0 chain falls at a depth that Phase S never initialized (e.g. depth 10), the depth slot is created automatically.

---

## 🧬 Phase 3 — Genetic Algorithm

Phase 3 is an optional evolutionary search activated by `--genetic`. It runs **after Phase 2**, with a **dedicated time reservation of 20 % of `--target-hours`** (minimum 120 s) that is subtracted from Phase 2's budget before Phase 2 begins — guaranteeing the GA always runs with meaningful time rather than only leftover scraps. All results are merged into the global hit counter before signature minimization.
 
### Why a Genetic Algorithm
 
Phase 2 samples rule chains from a hot-rule-biased pool, but the sampling is still **random**: at chain depth 3 with 5 000 atomic rules the search space is `5 000³ = 125 billion` candidates. Even a well-biased random sampler cannot explore that space thoroughly in a fixed time budget.
 
A genetic algorithm solves this by **directing** the search. Chains that produce many bloom-filter hits ("high-fitness individuals") are preferentially recombined and mutated, so successive generations concentrate probability mass on high-coverage regions of the rule space. The GA reuses the existing GPU chain kernel for fitness evaluation — no new OpenCL code is required.
 
### Algorithm
 
```
Initial population
  30 % — depth-2 combos of top-50 hot Phase-1 rules
  30 % — seeded deeper chains (1 hot rule + random atoms, biased depth 3+ when max_depth ≥ 3)
  40 % — unexplored Phase-S chains (NOT already in known_rules); fallback to random when
          --no-builtin-seeds is set or all Phase-S seeds are already known

For each generation:
  1. Evaluate fitness   — _run_chain_kernel raw hit count per chain (GPU batch)
  2. Apply novelty bonus — chains NOT in known_rules (Phase 1/S/2 results) score × 2
  3. Merge raw hits     — raw (un-multiplied) counts added to all_counts
  4. Sort by eff. score — build ranked (chain, effective_score) list
  5. Stagnation check  — if best score unchanged for 5 gens: replace bottom 30 % with
                          fresh random chains (depth 3+ biased), restart stagnation counter
  6. Elitism            — top elite_frac % copied unchanged to next generation
  7. Tournament select  — draw k=4 contenders; highest effective score wins
  8. One-point crossover — random cut on each parent's token list (p=0.80)
  9. Mutation           — replace / insert / delete one token (60/20/20 %)
 10. Diversity fill     — duplicate chains replaced by random individuals (depth 3+ biased)
 11. Repeat until generations exhausted or wall-clock budget reached
```
 
### Population Initialisation
 
The initial population is seeded from Phase 1 results to give the GA a strong starting point:
 
| Slice | Strategy | Rationale |
|---|---|---|
| 30 % | Depth-2 combos of the top-50 hot rules | These pairs are already "known good" atoms — often produce immediate hits in generation 0 |
| 30 % | Seeded deeper chains: 1 hot rule + random pool atoms; **70 % probability of depth 3+** when `--max-depth ≥ 3` | Biases toward depth regions that Phase 2's random sampling undercovers, while maintaining structural variety |
| 40 % | **Unexplored** Phase-S chains (NOT already in `known_rules`), sorted by hit count; known seeds used as fallback; pads with depth-3+-biased random chains if pool is exhausted; falls back to pure random when `--no-builtin-seeds` is set | Prioritises genuinely new starting points; avoids re-seeding the GA with rules Phase 2 already found |
 
### Crossover
 
One-point crossover exchanges rule-token sub-sequences between two parents at independently chosen cut points:
 
```
parent 1:  [ c ]  [ $1 ]  [ $! ]  [ r ]
                    ↑ cut1
parent 2:  [ u ]  [ sa@ ]  [ so0 ]
                              ↑ cut2

child 1:   [ c ]  [ $1 ] + [ so0 ]          →  "c $1 so0"
child 2:   [ u ]  [ sa@ ] + [ $! ]  [ r ]   →  "u sa@ $! r"
```
 
Both offspring are clamped to `[2, --max-depth]` tokens. If the crossover probability draw fails (20% of the time) both parents are passed unchanged, ensuring elitism is not undermined.
 
### Mutation
 
Each offspring undergoes exactly one mutation drawn from three operators with normalised weights (default 60/20/20):
 
| Operator | Weight | Action | Constraint |
|---|---|---|---|
| **replace** | 60 % | Swap one random token with a rule drawn from the pool | Always applicable |
| **insert** | 20 % | Insert one random rule at a random position | Only if `len < max_depth`; falls back to replace |
| **delete** | 20 % | Remove one random token | Only if `len > 2`; falls back to replace |
 
### Fitness, Novelty Weighting, and Hit Merging

Fitness used for GA selection is a **novelty-weighted bloom-filter hit count**:

| Chain status | Fitness multiplier | Rationale |
|---|---|---|
| **Not yet in `known_rules`** (Phase 1/S/2) | **x 2** (novelty bonus) | Drives selection toward unexplored territory |
| **Already in `known_rules`** | x 1 (raw hits) | Still valid as crossover/mutation scaffolding |

The raw (un-multiplied) hit count is what gets stored in `all_counts` and written to the output file, so the frequency column reflects true GPU hits rather than inflated fitness scores.

For each chain the **maximum raw hit count seen across all generations** is retained, so a chain that scores highly in a later generation after mutation still contributes its best score to the final output.

The `known_rules` set is built from all rules already discovered by Phase 1, Phase S, and Phase 2 immediately before the GA starts. Chains discovered by the GA itself are **not** added to `known_rules` mid-run — the novelty bonus persists for any rule that was genuinely new at Phase 3 start.

The terminal log and final summary report both the total GA hits and the **genuinely novel** count (chains absent from `known_rules` at start), making it easy to confirm Phase 3 is adding new coverage rather than rediscovering Phase 2 results.

### Time Budget

Phase 3 uses a **dedicated reserved budget** rather than only leftover time:

```
reserved_for_ga = max(120 s,  target_hours x 3600 x 0.20)
phase2_budget   = target_hours x 3600  -  elapsed_p1_ps  -  reserved_for_ga
ga_budget       = reserved_for_ga  +  max(0, unused_phase2_time)
```

The 20 % reservation is subtracted from Phase 2's budget before Phase 2 begins, so Phase 3 is always guaranteed at least 120 s (or 20 % of the total budget, whichever is larger). Any Phase 2 time that goes unused (e.g. because all chains were tested quickly) is added on top.

This eliminates the previous failure mode where Phase 2 consumed the entire time budget and Phase 3 was silently skipped with 0 seconds remaining.

### Stagnation Guard

If the best effective fitness score does not improve for **5 consecutive generations**, the stagnation guard fires:

1. The bottom 30 % of the population (beyond the elite) is discarded.
2. Fresh random chains fill those slots, biased toward depth 3+ when `--max-depth >= 3`.
3. The stagnation counter resets.

This prevents the GA from burning its remaining budget re-evaluating the same elites with no escape from a local optimum.

### Interaction with Other Phases

| Setting | Effect on Phase 3 |
|---|---|
| `--max-depth 1` | Phase 3 is silently skipped (chains require depth >= 2) |
| `--max-depth 2` | Phase 3 emits a warning: Phase 2 already covers depth-2 exhaustively; use `--max-depth 3+` for meaningful GA output |
| `--no-builtin-seeds` | Phase S is skipped; Phase 3's 40 % fill slot falls back to depth-3+-biased random chains |
| `--seed-rules` | Seed singles appear in the rule pool available to the GA; chain seeds do not affect Phase 3 |
| `--genetic-pop 400 --genetic-generations 100` | Doubles population and generation count; roughly doubles Phase 3 GPU time |
| `--target-hours 3.0` | Increases the total budget; the 20 % reservation scales proportionally (min 120 s floor) |
| Short `--target-hours` (e.g. `0.5`) | Reservation is `max(120 s, 108 s) = 120 s`; Phase 2 runs on the remaining ~240 s |


---
 
## 🔬 Functional Minimization
 
After GPU extraction, `rulest_v2.py` applies a **signature-based functional minimization** pass before writing the final output. This post-processing step removes redundant rules — rules that are syntactically different but produce identical outputs on real words — so the resulting ruleset is as compact as possible without losing coverage.
 
### Algorithm
 
1. **Use the built-in probe set.** A fixed, hand-curated set of probe words is always used (no external file or CLI flag required). The set is designed to exercise every class of hashcat opcode: very short words (edge cases for `k`, `K`, `{`, `}`, `[`, `]`), short alphanumeric base words, typical password base words of lengths 7–9, longer words for truncation and repeat ops, mixed-case words, words with embedded digits, words with special characters, words with repeated characters, and dedicated leet-substitution targets (`master`, `leet`, `elite`, `access`). Because the probe set is deterministic and ships with the script, minimization results are fully reproducible across runs with no configuration needed.
 
2. **Compute each rule's signature.** Every candidate rule (or chain) is applied to every probe word using a pure-Python interpreter (`py_apply_chain`). The signature is the resulting tuple of transformed strings — one per probe word. Rules containing opcodes that cannot be emulated in Python are assigned the sentinel signature `('__UNSUPPORTED__',)` and are bucketed together.
 
3. **Group by signature.** Rules sharing an identical signature are considered **functionally equivalent** on the probe set. Only one representative survives from each group.
 
4. **Select the best representative.** Within each signature group, the rule with the **highest GPU hit-count** is kept. Ties are broken by preferring shorter chain depth, then lexicographic order.
 
5. **Write the minimized ruleset.** Surviving rules are written to the output file sorted by GPU frequency (descending). The file header records the probe word count and how many equivalent rules were removed.
 
### Large-Scale Minimization (Disk-Backed Path)

For large runs that produce more than **500,000 raw candidates** (`MINIMIZE_DISK_THRESHOLD`), `minimize_by_signature` automatically switches from the in-memory algorithm to a **SQLite-backed path** to avoid OOM conditions:

| Condition | Path | Peak heap |
|---|---|---|
| ≤ 500,000 rules | `_minimize_mem` — in-memory dict of signature tuples | ~2.5 KB × rule count |
| > 500,000 rules | `_minimize_disk` — SQLite temp DB with SHA-1-hashed signatures | O(batch size) — a few hundred KB regardless of rule count |

The disk path computes each rule's signature, immediately SHA-1-hashes it (replacing the ~2.5 KB tuple with a 160-bit digest), and writes rows in batches of `MINIMIZE_DISK_BATCH_SIZE` (default 10,000) to a temporary SQLite database using an `INSERT … ON CONFLICT DO UPDATE` so only one row per equivalence class is ever stored. Survivors are read back in a single `SELECT` scan and the temp file is deleted. Both paths apply identical tie-breaking: highest GPU hit count → shortest chain depth → lexicographic rule order. The path used is recorded in the output file header (`# Path : in-memory` or `# Path : disk-backed SQLite`).

### Why It Matters
 
GPU Bloom filter screening (Phase 1, 2 & 3) can yield thousands of candidates where many are functionally identical — for example, `c` (capitalize first letter) and a chain `l c` applied to an already-lowercase word produce the same output. Without minimization, the output file contains duplicate work that inflates Hashcat's rule-testing time without adding new candidate passwords.
 
Minimization typically removes **20–60% of raw candidates** depending on chain depth and wordlist diversity, leaving a tighter, faster ruleset with no reduction in theoretical coverage.
 
### Built-in Probe Set Design
 
The probe set covers several distinct word classes to minimise false equivalences (two rules appearing identical on the probe set when they differ on real words):
 
| Word class | Purpose | Examples |
|---|---|---|
| Very short (len 2–4) | Edge cases for `k K { } [ ]` | `ab`, `abc`, `abcd` |
| Short alphanumeric (len 4–6) | Common base words | `pass`, `root`, `admin` |
| Typical password words (len 7–9) | Core coverage | `letmein`, `password`, `sunshine` |
| Longer words (len 10+) | Truncation and repeat ops | `qwertyuiop`, `monkey12345` |
| Mixed-case words | `l u c C t E T k K` ops | `Password`, `AdminUser`, `HelloWorld` |
| Words with embedded digits | `s o @ T` ops | `pass123`, `admin2024` |
| Words with special chars | `@` removal, `s` substitution | `p@ssw0rd`, `s3cur1ty` |
| Leet-substitution targets | Family J / M coverage | `master`, `leet`, `elite`, `access` |
| Repeated-char words | `q z Z` ops | `aaaa`, `bbbb` |
 
> Signature equivalence is probabilistic — two rules might match on all probe words yet differ on others. The hand-curated set is tuned to keep the false-equivalence rate very low for the rule patterns generated by rulest. If you observe unexpected merging in the output, you can increase probe coverage by passing a `--seed-rules` file containing probe-sensitive rules to force Phase 2 to preserve them.
 
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
| `MAX_GPU_RULES` | `255` | Maximum number of operations allowed in a single GPU rule string (validated by `HashcatRuleValidator`); separate from `MAX_HASHCAT_CHAIN` which caps chain length |
| `BASELINE_COMBOS_PER_SEC` | `120,000,000` | Estimated throughput on a capable GPU |
| `LOW_END_COMBOS_PER_SEC` | `40,000,000` | Throughput fallback for devices with < 20 compute units |
| `MAX_WORD_LEN` | `256` | Maximum word length accepted from wordlists |
| `MAX_RULE_LEN` | `16` | Maximum single rule string length in GPU buffers |
| `MAX_OUTPUT_LEN` | `512` | Maximum transformed word output length in GPU buffers |
| `MAX_CHAIN_STRING_LEN` | `128` | Maximum chained rule string length in GPU buffers |
| `MAX_HASHCAT_CHAIN` | `31` | Maximum number of rules in a single Hashcat chain |
| `MINIMIZE_DISK_THRESHOLD` | `500,000` | Rule count above which `minimize_by_signature` switches from in-memory to the disk-backed SQLite path |
| `MINIMIZE_DISK_BATCH_SIZE` | `10,000` | Rows per `executemany` batch in the SQLite minimization path |
 
> Phase 3 GA parameters (`--genetic-pop`, `--genetic-generations`, `--genetic-elite`) are CLI-only and have no corresponding module-level constants.
 
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
# Phase 0 TS     : enabled  modes=14  workers=16  min-stem=4  leet-amb=3
# Phase 3 GA     : enabled  pop=200  gen=50  elite=15%
#
# GPU raw candidates      : 9,214  (bloom hits, includes false positives)
# Post-processing         : signature-based minimization
#   Path                  : in-memory  (threshold 500,000)
#   Probe words           : 37  (built-in)
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
 
- The `# Phase 3 GA` header line is only written when `--genetic` is active
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
| Skip built-in seed families | Pass `--no-builtin-seeds` to skip Phase S entirely; useful when supplying all seeds via `--seed-rules` or benchmarking Phase S contribution (families A–M) |
| Reduce VRAM pressure | Lower `--max-chains` or use `--depth2-chains` / `--depth3-chains` |
| Force deep chain exploration | Set `--depth4-chains 50000 --depth5-chains 10000` explicitly |
| Use a specific GPU | `--device 1` or `--device "RTX 4090"` |
| Bootstrap from prior results | Pass previous output to `--seed-rules` for iterative refinement |
| Limit total combinations | `--max-chains 500000` to cap generation before scaling |
| Reduce terminal noise | Set `VERBOSE = False` in the script header or omit `--debug` |
| Increase hot-rule aggressiveness | Raise `HOT_RULE_RATIO` toward `1.0` (reduces random exploration) |
| Enable token-strip pre-pass | Add `--token-strip` — most effective when the target wordlist contains structured transformations (leet speak, capitalized words, boundary digits) |
| Maximize Phase 0 coverage | Leave `--token-strip-no-new-modes` off (default) to run all 14 modes; the 9 v1.1 modes (insert, swap, range, char-dup, ascii-transform, truncate, purge, repeat, sep-title) find patterns the original 5 miss entirely |
| Speed up Phase 0 | Add `--token-strip-no-new-modes` to restrict to the original 5 modes; or lower `--token-strip-max-prefix` / `--token-strip-max-suffix` |
| Tune Phase 0 parallelism | `--token-strip-workers N` — default 0 uses all cores; set to 1 to force single-process; tune `--token-strip-chunk-size` for progress granularity vs dispatch overhead |
| Tune token-strip leet scope | Raise `--token-strip-min-leet-amb` (e.g. `5`) to decode more ambiguous leet chars at the cost of more branching |
| Enable evolutionary search | Add `--genetic` with `--max-depth 3` or higher — Phase 3 adds no value at depth 2 since Phase 2 already covers it exhaustively |
| Guarantee GA time budget | The 20 % reservation is automatic; raise `--target-hours` to give both Phase 2 and Phase 3 more headroom |
| Speed up GA per generation | Lower `--genetic-pop` (e.g. `100`) — fewer GPU evaluations per generation |
| Improve GA convergence quality | Raise `--genetic-pop` (e.g. `500`) and `--genetic-generations` (e.g. `100`) |
| Reduce GA premature convergence | Lower `--genetic-elite` (e.g. `0.05`) for more diversity each generation; stagnation guard fires after 5 flat gens automatically |
| Stabilise GA on narrow targets | Raise `--genetic-elite` (e.g. `0.25`) to preserve top chains longer |
| Maximise novel rule discovery | Use `--max-depth 4` or higher with GA — novelty bonus (2×) is most impactful in large unexplored depth regions |
| GA with a very short time budget | Even with the 120 s floor, reduce `--genetic-generations` (e.g. 10) to run more focused generations within the reserved time |
 
### VRAM Scaling Reference
 
| Available VRAM | Scale Factor | Bloom Filter Cap |
|---|---|---|
| < 4 GB | 0.25–0.5× | 32 MB |
| 4–8 GB | 0.5–1.0× | 128 MB |
| 8 GB+ | 1.0× (full) | 256 MB |
 
---
 
## 💡 Examples
 
**Basic single-depth extraction:**
```bash
python rulest_v2.py rockyou.txt target_hashes_plain.txt --max-depth 1 -o single_rules.txt
```
 
**Deep chain search with a 2-hour budget:**
```bash
python rulest_v2.py rockyou.txt target.txt --max-depth 4 --target-hours 2.0 -o chains_deep.txt
```
 
**Use a specific GPU and seed from a previous run:**
```bash
python rulest_v2.py base.txt target.txt \
  --device "RTX 3080" \
  --seed-rules single_rules.txt \
  --max-depth 3 --target-hours 1.0 \
  -o refined_chains.txt
```
 
**List available OpenCL devices:**
```bash
python rulest_v2.py --list-devices
```
 
**Override chain budget for specific depths:**
```bash
python rulest_v2.py base.txt target.txt --max-depth 5 \
  --depth2-chains 200000 \
  --depth3-chains 100000 \
  --depth4-chains 30000 \
  --depth5-chains 5000 \
  -o custom_budget.txt
```
 
**Genetic algorithm — minimal invocation:**
```bash
# Enable Phase 3 with defaults (pop=200, gen=50, elite=15%)
# 20 % of --target-hours (18 min) is reserved for Phase 3 before Phase 2 begins
# Check the "[GA] genuinely new" line in the output to verify novel rule discovery
python rulest_v2.py base.txt target.txt \
  --max-depth 3 \
  --target-hours 1.5 \
  --genetic \
  -o evolved_rules.txt
```
 
**Genetic algorithm — tuned for deep chain discovery:**
```bash
# Larger population and more generations; needs extra time budget
python rulest_v2.py rockyou.txt target.txt \
  --max-depth 4 \
  --target-hours 4.0 \
  --genetic \
  --genetic-pop 400 \
  --genetic-generations 100 \
  --genetic-elite 0.10 \
  -o evolved_deep.txt
```
 
**Genetic algorithm — fast iteration on a small target:**
```bash
# Small pop + few generations when time is tight
python rulest_v2.py base.txt target.txt \
  --max-depth 3 \
  --target-hours 0.5 \
  --genetic \
  --genetic-pop 100 \
  --genetic-generations 20 \
  -o quick_ga.txt
```
 
**Iterative refinement workflow:**
```bash
# Pass 1 — fast sweep for single rules
python rulest_v2.py rockyou.txt target.txt --max-depth 1 --target-hours 0.25 -o pass1.txt
 
# Pass 2 — chain from pass 1 results
python rulest_v2.py rockyou.txt target.txt --max-depth 3 --target-hours 1.0 \
  --seed-rules pass1.txt -o pass2.txt
 
# Pass 3 — deep dive with GA seeded from pass 2
python rulest_v2.py rockyou.txt target.txt --max-depth 5 --target-hours 4.0 \
  --seed-rules pass2.txt \
  --genetic --genetic-generations 75 --genetic-pop 300 \
  -o pass3_final.txt
```
 
**Token-strip pre-pass — extract rules from structured target wordlist:**
```bash
# Phase 0 scans the target using all 14 modes (default)
# and injects verified chains into Phase 1, Phase S, and Phase 2
python rulest_v2.py rockyou.txt target.txt \
  --token-strip \
  --max-depth 4 --target-hours 1.0 \
  -o with_token_strip.txt
```

**Token-strip — use all CPU cores, default 6-mode extraction:**
```bash
# Skip v1.1 modes (insert, swap, range, char-dup, ascii, truncate, purge, repeat, sep-title)
python rulest_v2.py rockyou.txt target.txt \
  --token-strip \
  --token-strip-workers 8 \
  --max-depth 4 --target-hours 1.0 \
  -o fast_ts.txt
```

**Token-strip with extended leet ambiguity and explicit chunk size:**
```bash
# Allow up to 5 ambiguous leet positions per word (default 3)
# Useful for targets with heavy leet encoding like "1337sp34k"
python rulest_v2.py rockyou.txt target.txt \
  --token-strip \
  --token-strip-min-leet-amb 5 \
  --token-strip-max-prefix 6 \
  --token-strip-max-suffix 6 \
  --token-strip-chunk-size 500 \
  --max-depth 4 --target-hours 1.5 \
  -o extended_leet.txt
```

**Benchmark Phase 0 contribution and mode impact:**
```bash
# Without token-strip — baseline
python rulest_v2.py base.txt target.txt --max-depth 3 --target-hours 1.0 -o no_ts.txt

# With token-strip, 14 modes (default) — compare rule count and depth distribution
python rulest_v2.py base.txt target.txt --max-depth 3 --target-hours 1.0 \
  --token-strip -o with_ts_14modes.txt

# With token-strip, 5 modes only — isolate v1.1 mode contribution
python rulest_v2.py base.txt target.txt --max-depth 3 --target-hours 1.0 \
  --token-strip --token-strip-no-new-modes -o with_ts_5modes.txt
```

**Skip built-in seed families (Phase S disabled):**
```bash
# Faster run when you supply all seeds yourself and don't need families A–M
python rulest_v2.py base.txt target.txt --max-depth 3 --target-hours 1.0 \
  --seed-rules my_seeds.txt \
  --no-builtin-seeds \
  -o no_phase_s.txt
```
 
**Benchmark Phase S contribution:**
```bash
# With built-in seeds (default — families A–M)
python rulest_v2.py base.txt target.txt --max-depth 2 -o with_seeds.txt
 
# Without built-in seeds — compare output sizes to measure Phase S value
python rulest_v2.py base.txt target.txt --max-depth 2 --no-builtin-seeds -o without_seeds.txt
```
 
**Benchmark Phase 3 GA contribution:**
```bash
# Without GA — baseline
python rulest_v2.py base.txt target.txt --max-depth 3 --target-hours 2.0 -o no_ga.txt
 
 
# With GA — compare rule count, depth distribution, and "genuinely novel" count in the log
# Note: GA reserves 20 % of total time (24 min) before Phase 2 begins
python rulest_v2.py base.txt target.txt --max-depth 3 --target-hours 2.0 \
  --genetic -o with_ga.txt
```
 
---
 
## 📝 License
 
MIT
 
## Credits

https://github.com/synacktiv/rulesfinder
