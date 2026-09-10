# 🔐 GPU-Accelerated Hashcat Rule Extractor

> Extract and chain Hashcat-compatible rules from wordlists using OpenCL GPU acceleration.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Versions](#-versions)
  - [rulest.py — v1 (BFS, Legacy)](#rulestpy--v1-bfs-legacy)
  - [rulest_v2.py — v2 (Legacy, single-file)](#rulest_v2py--v2-legacy-single-file)
  - [rulest v3 — Package + Greedy Selection](#-rulest-v3--package--greedy-selection-recommended)
- [What's New in v3](#-whats-new-in-v3)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [run_rulest.py — Full Reference](#run_rulestpy--full-reference)
- [Architecture (v3)](#-architecture-v3)
  - [Package Layout](#package-layout)
  - [Extraction Pipeline](#extraction-pipeline)
- [Multi-GPU Execution](#-multi-gpu-execution)
- [Bloom Filter & Sharding](#-bloom-filter--sharding)
- [Built-in Seed Families (A–M)](#-built-in-seed-families)
- [Stage 0 — Token-Strip Rule Extraction](#-stage-0--token-strip-rule-extraction)
- [Stage 3 — Genetic Algorithm](#-stage-3--genetic-algorithm)
- [Rule Selection — Frequency vs. Greedy (CELF)](#-rule-selection--frequency-vs-greedy-celf)
- [Functional Minimization](#-functional-minimization)
- [Rule Categories](#-rule-categories)
- [GPU Command Support](#-gpu-command-support)
- [Configuration Constants](#-configuration-constants)
- [Output Format](#-output-format)
- [Performance Tuning](#-performance-tuning)
- [Testing](#-testing)
- [Examples](#-examples)

---

## 🔍 Overview

This toolkit analyzes two wordlists — a **base** (source) wordlist and a **target** (dictionary) wordlist — and reverse-engineers the **Hashcat rules** that transform words from the base into words in the target. Rules are discovered via GPU-parallel transformation and validated for direct compatibility with Hashcat's GPU engine.

The result is a `.rule` file you can load directly into Hashcat (`-r rules.txt`), ordered either by raw hit frequency or by marginal contribution to overall target coverage (see [Rule Selection](#-rule-selection--frequency-vs-greedy-celf)).

---

## 📁 Versions

### `rulest.py` — v1 (BFS, Legacy)

A first-generation implementation using a **Breadth-First Search (BFS)** chaining strategy executed on the GPU via a monolithic OpenCL kernel. No rule validation, no Bloom filter, single device only, no hit counting. Historical reference only.

### `rulest_v2.py` — v2 (Legacy, single-file)

A ~210 KB single-file rewrite adding Hashcat GPU rule validation, a GPU-only Bloom filter (with automatic sharding above ~512 MB), staged extraction (single-rule sweep → built-in seed families → informed chain generation → optional genetic algorithm), token-strip pre-extraction, multi-GPU execution with process-isolated workers, and frequency-ranked output. Still supported for compatibility, but superseded by v3's package structure and greedy rule selection.

### 🚀 `rulest` v3 — Package + Greedy Selection (Recommended)

v3 takes the entire v2 engine — Stages 0–3, the Bloom filter (including sharding), token-strip, multi-GPU execution, and the genetic algorithm — and:

1. **Splits it into a proper Python package** (`rulest/`) instead of one large script, with a thin `run_rulest.py` entry point.
2. **Adds a new post-processing stage**: greedy marginal-coverage rule selection (CELF), as an alternative to plain frequency sorting.

Multi-GPU support (process-isolated workers, adaptive load balancing) and Bloom filter sharding were already present in v2; v3 carries them over into the new package layout unchanged. Stage 0–3 behavior and the default (frequency-sorted) output are also unchanged from v2 — v3 is a strict superset.

---

## 🆕 What's New in v3

| Change | Description |
|---|---|
| **Modular package** | The single 210 KB `rulest_v2.py` file is now the `rulest/` package (14 modules — see [Package Layout](#package-layout)) with `run_rulest.py` as the entry point. Behavior of Stages 0–3 and frequency-based output is unchanged in default mode. |
| **Greedy marginal-coverage selection (CELF)** (`rulest/selection.py`) | New optional post-processing stage: `--select-mode greedy` evaluates every candidate rule's *exact* set of recovered target words (GPU bloom prefilter + exact verification kernel) and runs a lazy-greedy (Cost-Effective Lazy Forward) set-cover selection, so the kept ruleset maximizes the union of recovered passwords for a given rule budget instead of just summing per-rule hit counts. |
| **Budget sweeps** | `--select-budgets 64,250,1500,...` cuts the same ordering (greedy or frequency) at multiple sizes in one run, writing `<output>.<budget>.txt` for each — useful for plotting recovery-vs-budget without re-running coverage evaluation. |
| **Exact recovery reporting** | `--exact-recovery` reports the exact count of distinct target words recovered by the selected ruleset against the real target wordlist (not bloom-filter estimates). |
| **Cost-aware selection** | `--select-cost-alpha` lets greedy selection discount deeper (more expensive) chains: `gain(r) = new_recovery(r) / depth(r)**alpha`. |
| **Unit test suite** | `tests/` (pytest) now covers rule-pool synchronization across GPUs, selection/CELF logic, minimization, common utilities, I/O helpers, and exact multi-GPU output parity. |

---

## 📦 Requirements

```
Python >= 3.8
numpy
pyopencl
tqdm
```

Dev/test extras (`requirements-dev.txt`):

```
pytest
```

An OpenCL-capable GPU (NVIDIA, AMD, or Intel) is required. CPU fallback via OpenCL is supported but will be slow.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/A113L/rulest.git
cd rulest

# Install dependencies
pip install -r requirements.txt

# (optional) install test dependencies
pip install -r requirements-dev.txt

# Verify OpenCL is available
python -c "import pyopencl; print(pyopencl.get_platforms())"
```

> **Windows users:** Install the appropriate OpenCL runtime for your GPU vendor. NVIDIA users typically have this via the CUDA toolkit or standard driver. AMD users should install ROCm or the AMD APP SDK.

---

## 🚀 Usage

### `run_rulest.py` — Full Reference

```
usage: run_rulest.py [options] base_wordlist target_wordlist
```

#### Positional Arguments

| Argument | Description |
|---|---|
| `base_wordlist` | Source wordlist — words to transform *from* |
| `target_wordlist` | Target dictionary — words to transform *to* |

#### Core Arguments

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output` | `rulest_output.txt` | Output file path |
| `--device` | best GPU | Device index, name substring, `"all"` to run every detected GPU in parallel, or a comma-separated index list (e.g. `"0,1"`) to run only those GPUs in parallel |
| `--list-devices` | — | Print all available OpenCL devices and exit |
| `--max-depth` | `2` | Maximum rule chain depth (1–31; depths >31 capped with a warning) |
| `--target-hours` | `0.5` | Time budget in hours; controls chain generation budget for Stage 2 and Stage 3 |
| `--max-chains` | unlimited | Hard cap on total chains generated |
| `--bloom-mb` | dynamic | Override Bloom filter size (MB); 0 = auto-scale |
| `--bloom-no-shard` | off | Force a single unsharded Bloom buffer even above the ~512 MB auto-shard threshold |
| `--seed-rules` | None | File of user-supplied rules/chains |
| `--depth2-chains` through `--depth10-chains` | dynamic | Per-depth chain budget overrides |
| `--allow-reject-rules` | off | Include rejection rules (normally excluded as GPU-incompatible) |
| `--no-builtin-seeds` | off | Disable the built-in seed families (Stage S) |
| `--debug` | off | Enable verbose output |

#### Stage 0 — Token-Strip Arguments

| Flag | Default | Description |
|---|---|---|
| `--token-strip` | off | Enable Stage 0 CPU exact-match pre-pass |
| `--token-strip-min-stem` | `4` | Minimum stem length after token decoding |
| `--token-strip-max-prefix` | `4` | Maximum boundary characters stripped from the start |
| `--token-strip-max-suffix` | `4` | Maximum boundary characters stripped from the end |
| `--token-strip-min-leet-amb` | `3` | Maximum ambiguous leet positions per word |
| `--token-strip-workers` | `0` | Worker processes for Stage 0 (`0` = all cores) |
| `--token-strip-chunk-size` | `0` | Words per worker chunk (`0` = auto) |

#### Genetic Algorithm Arguments

| Flag | Default | Description |
|---|---|---|
| `--genetic` | off | Enable Stage 3 genetic algorithm rule evolution |
| `--genetic-generations` | `50` | Maximum GA generations |
| `--genetic-pop` | `200` | GA population size |
| `--genetic-elite` | `0.15` | Fraction of top individuals carried unchanged each generation (must be strictly between 0 and 1) |

#### Rule Selection (post-processing) Arguments — 🆕 v3

| Flag | Default | Description |
|---|---|---|
| `--select-mode` | `frequency` | `frequency`: sort by raw GPU hit count (v2 behaviour). `greedy`: lazy-greedy (CELF) marginal-coverage selection against the real target wordlist — maximizes the union of recovered words per rule budget instead of summed per-rule score. |
| `--select-budget` | `0` | Max rules to keep with `--select-mode greedy` (`0` = run to saturation, i.e. until no remaining rule adds new coverage) |
| `--select-budgets` | None | Comma-separated budgets (e.g. `"64,250,1500,10000,25000,50000,150000"`); emits extra files `<output>.<budget>.txt` cut from the same ordering, for either select mode |
| `--select-cost-alpha` | `0.0` | `gain(r) = new_recovery(r) / depth(r)**alpha`. `0` (default) = pure marginal coverage; `>0` favors shorter/cheaper chains among rules with similar marginal gain |
| `--exact-recovery` | off | Report exact distinct target-word recovery after greedy selection (reporting only — greedy coverage is already exact, resolved through the verification kernel before CELF selection) |

---

## 🏗 Architecture (v3)

### Package Layout

The single-file v2 engine is now the `rulest` package:

| Module | Responsibility |
|---|---|
| `rulest/state.py` | Runtime flags |
| `rulest/common.py` | Logging, colors, constants, rule validator, CPU rule applicator |
| `rulest/minimize.py` | Signature-based functional minimization |
| `rulest/token_strip.py` | Stage 0 — CPU exact-match rule extraction |
| `rulest/devices.py` | OpenCL device discovery and dynamic parameter sizing |
| `rulest/kernel_source.py` | OpenCL C kernel template |
| `rulest/gpu_engine.py` | Single-GPU extraction engine |
| `rulest/gpu_worker.py` | Process-isolated GPU worker (`ProcessEngineProxy`) for multi-GPU mode |
| `rulest/multi_gpu.py` | Multi-GPU orchestration, adaptive load balancing, work splitting |
| `rulest/genetic.py` | Stage 3 genetic algorithm |
| `rulest/extractor.py` | Pipeline orchestrator (`GPUExtractor`) |
| `rulest/selection.py` | 🆕 Coverage evaluation + greedy (CELF) rule selection |
| `rulest/io_utils.py` | Wordlist loading |
| `rulest/worker_state.py` | Per-worker-process runtime state |
| `rulest/cli.py` | Argument parsing, pipeline invocation, output/report writing |
| `run_rulest.py` | Entry point (`python run_rulest.py ...`) |

### Extraction Pipeline

The four-stage GPU pipeline is unchanged from v2:

```
Stage 0 (--token-strip, optional, CPU, pre-GPU)
   → decomposes target passwords into stem + rule chain (6 modes)
   → single-rule discoveries feed Stage 1; multi-rule chains feed Stage S + Stage 2

Stage 1 — single-rule GPU sweep (all base words × every GPU-valid single rule)
Stage S — built-in seed families A–M (default on, independent of --max-depth)
Stage 2 — informed chain generation (60% hot-rule bias, 30% seed extension, 10% random)
Stage 3 (--genetic, optional) — novelty-weighted genetic algorithm, 20% reserved time budget
   ↓
minimize_by_signature() — signature-based functional minimization
   ↓
SELECT — 🆕 frequency sort (default) OR greedy CELF marginal-coverage selection
   ↓
GPU-safe, Hashcat-compatible .rule output file(s)
```

On a multi-GPU run (`--device all` or an explicit index list), Stages 1, S, 2, and 3 are each split across the available `GPUEngine` / `ProcessEngineProxy` instances by `MultiGPUEngine` and merged before minimization and selection, which remain single-process steps.

---

## 🖥 Multi-GPU Execution

> Carried over from v2 — the mechanics below are unchanged, just reorganized into `rulest/multi_gpu.py` and `rulest/gpu_worker.py`.

`--device` accepts:

| Value | Behavior |
|---|---|
| *(unset)* | Auto-select the best single GPU (v2 behaviour) |
| index (e.g. `0`) | Use that specific device |
| name substring (e.g. `"RTX 4090"`) | Use the first matching device |
| `all` | Run every detected OpenCL device concurrently via `MultiGPUEngine` |
| comma-separated indices (e.g. `0,1,3`) | Run exactly those devices concurrently, skipping any others |

**Work splitting:**
- **Stage 1** splits `base_words` across GPUs (word-dominated workload).
- **Stage S and Stage 2** split chain candidates across GPUs — each GPU tests its chain slice against all base words (chain-dominated workload).
- **Stage 3 (genetic)** also splits each generation's population across GPUs, reusing the same `MultiGPUEngine` instance so it benefits from load-balancing data gathered in earlier stages.

**Fault isolation:** every non-primary GPU runs inside its own OS process (`ProcessEngineProxy` in `rulest/gpu_worker.py`), talked to over a command/result queue with a background heartbeat. A hung OpenCL driver call blocks a Python *thread* and cannot be killed from within the same process — but a *process* can be terminated, so if a device goes silent for longer than the hang timeout, only that device's process is killed; the rest of the run continues.

**Adaptive load balancing:** `MultiGPUEngine` maintains an exponential moving average of items/sec per `(engine, work-kind)` pair. Early splits default to equal shares; as real throughput measurements come in, future splits shift toward proportional-to-speed shares — so pairing a fast discrete GPU with a slow integrated GPU doesn't bottleneck the whole run to the slow device's pace. A fixed vendor-ratio split (`vendor_ratio_split`) is also available for Stage S, matching devices by a name substring (e.g. `"NVIDIA"`, `"INTEL"`) to a fixed weight, with an automatic fallback to the adaptive split if devices can't be uniquely matched.

> ⚠️ **Careful with slow/mismatched devices in an explicit list.** Even with adaptive balancing, a very slow or driver-unstable device included in `--device 0,1,3` adds overhead and process-management risk. Prefer comparable, fast devices where possible.

---

## 🌸 Bloom Filter & Sharding

> Carried over from v2 — sharding already existed there; only the module location (`rulest/gpu_engine.py`) has changed.

The Bloom filter is still always built **on the GPU** using the `atomic_or`-based kernel, with the buffer kept in VRAM and a CPU fallback on any GPU build error, exactly as in v2.

**Sharding:** once the required filter size exceeds roughly 512 MB, the filter is automatically split into multiple shards (`BLOOM_NUM_SHARDS`) instead of requiring one oversized OpenCL allocation, which some drivers reject or handle poorly. Each shard is built and queried independently; hit results are merged transparently. Use `--bloom-no-shard` to force a single unsharded buffer (useful for debugging or when you know your driver handles large single allocations fine).

Filter size is still selected automatically based on available VRAM (override with `--bloom-mb`):

| Available VRAM | Auto bloom size |
|---|---|
| ≥ 8 GB | 512 MB |
| ≥ 4 GB | 128 MB |
| < 4 GB | 64 MB |

---

## 🌱 Built-in Seed Families (A–M)

Unchanged from v2. Thirteen deterministically generated seed families run as **Stage S**, a dedicated GPU extraction pass between Stage 1 and Stage 2, independent of `--max-depth` and the random-chain time budget. Disable with `--no-builtin-seeds`.

| Family | Description |
|---|---|
| A | Pure prepend digits |
| B | Pure append digits |
| C | Mixed prepend/append digits |
| D | Transform + digit/bracket |
| E | Date patterns (DDMM, MMDD, YYYY, DDMMYY, DDMMYYYY, etc.) |
| F | Pure append special chars (top-15 set) |
| G | Pure prepend special chars (top-15 set) |
| H | Transform + special char |
| I | Digit(s) + special char (core-7 set) |
| J | Leet substitutions (10 core pairs) |
| K | Double-transform chains |
| L | Special-before-digit patterns |
| M | Leet + transform chains |

Depth-1 seeds are skipped (already covered by Stage 1). All multi-rule seed chains at depths ≥2 are tested directly against the Bloom filter, then forwarded to Stage 2 as scaffolding atoms.

---

## 🔤 Stage 0 — Token-Strip Rule Extraction

Unchanged from v2. An optional CPU-only pre-pass (`--token-strip`) that runs **6 extraction modes** (letter, digit, reverse, delete-edge, dup/fold, insert) per target word, reverse-engineering exact Hashcat rule chains by decomposing each password into a base stem plus boundary/transform tokens, and verifying the chain reproduces the original word via `py_apply_chain`. Single-rule discoveries feed Stage 1; multi-rule chains feed Stage S and Stage 2.

---

## 🧬 Stage 3 — Genetic Algorithm

Unchanged from v2. An optional evolutionary search (`--genetic`) that runs after Stage 2, with a dedicated time reservation of 20% of `--target-hours` (minimum 120 s). Fitness is novelty-weighted (2× bonus for chains not already found by Stage 1/S/2), with tournament selection, one-point crossover, mutation, and a stagnation guard that refreshes the bottom 30% of the population after 5 flat generations. In multi-GPU runs, each generation's population evaluation is split across GPUs by the same `MultiGPUEngine` used for Stages 1/S/2.

---

## 🎯 Rule Selection — Frequency vs. Greedy (CELF)

This is the main new capability in v3, implemented in `rulest/selection.py`.

### The problem with frequency sorting

v2 (and v3's default `--select-mode frequency`) sorts candidate rules purely by raw GPU hit count. High-frequency rules often recover **overlapping** sets of target passwords — keeping the top N by frequency can waste rule budget on redundant rules that recover almost the same passwords as rules already kept, while a lower-frequency but orthogonal rule that would have added new coverage is left out.

### Greedy marginal-coverage selection

`--select-mode greedy` replaces frequency sorting with a two-step process:

1. **Coverage evaluation** (`evaluate_full_coverage`) — every candidate rule is evaluated against the real target wordlist. The GPU Bloom filter is used as a cheap prefilter, but every bloom-positive `(rule, base_word)` pair is then resolved through an **exact verification kernel**, mapping it to a specific target-word id. This produces `C(r)`: the exact set of distinct target words rule `r` actually recovers — with Bloom false positives excluded, so the optimization objective matches real recovery.

   Candidates longer than the GPU kernel's configured `MAX_CHAIN_DEPTH` are excluded up front (not silently truncated), since evaluating a truncated prefix would misattribute coverage to the wrong rule string.

2. **Lazy greedy / CELF selection** (`select_rules_by_marginal_coverage`) — rules are picked one at a time by the largest **new** coverage they add to the running union, using the Cost-Effective Lazy Forward heuristic to avoid recomputing every rule's marginal gain at every step. This maximizes the union of recovered target passwords for a given rule budget — the classic (1 − 1/e) near-optimal guarantee for submodular set-cover greedy selection.

### Budget control

| Flag | Effect |
|---|---|
| `--select-budget N` | Stop greedy selection after `N` rules (0 = run to saturation — keep going until no remaining rule adds any new coverage) |
| `--select-budgets "64,250,1500,..."` | In addition to the main output, cut the **same** ordering (greedy or frequency) at each listed size and write `<output>.<budget>.txt` for each, without re-running coverage evaluation per size |
| `--select-cost-alpha A` | Reweight marginal gain as `new_recovery(r) / depth(r)**A`; `0` (default) is pure coverage, `>0` favors shorter/cheaper chains when gains are similar |
| `--exact-recovery` | Add an exact recovery count/percentage to the output header — reporting only, since greedy coverage is already computed exactly |

### Compatibility

`--select-mode frequency` (the default) reproduces v2's exact ordering and output format. Nothing about Stages 0–3 changes based on `--select-mode` — selection only affects which rules from the already-extracted, already-minimized pool are written to the final file(s).

---

## 🔬 Functional Minimization

Unchanged from v2. After extraction, a signature-based pass applies every candidate rule to a fixed, built-in probe set via a pure-Python interpreter (`py_apply_chain`), groups rules with identical output signatures, and keeps only the highest-hit representative per group. Runs beyond `MINIMIZE_DISK_THRESHOLD` (500,000 raw candidates) automatically switch from the in-memory path to a SQLite-backed disk path to avoid OOM.

Minimization runs **before** rule selection — `--select-mode greedy` selects from the already-deduplicated candidate pool.

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

Defined in `rulest/common.py` (and `rulest/gpu_engine.py` for GPU-sizing constants):

| Constant | Default | Description |
|---|---|---|
| `VERBOSE` | `False` | Print per-rule validation messages and category counts; set at runtime via `--debug` |
| `VRAM_USAGE_FACTOR` | `0.55` | Fraction of device global memory to treat as free VRAM |
| `BLOOM_HASH_FUNCTIONS` | `4` | Number of FNV-1a hash functions in the Bloom filter |
| `BLOOM_FILTER_MAX_MB` | `256` | Maximum Bloom filter allocation (MB); override at runtime with `--bloom-mb` |
| `HOT_RULE_RATIO` | `0.6` | Fraction of Stage 2 chains biased toward hot rules |
| `EXTENSION_RATIO` | `0.3` | Fraction of Stage 2 budget allocated to seed extension |
| `TIME_SAFETY_FACTOR` | `0.9` | Multiplier applied to time-budget combo estimates |
| `MAX_GPU_RULES` | `255` | Maximum number of operations allowed in a single GPU rule string |
| `MAX_WORD_LEN` | `256` | Maximum word length accepted from wordlists |
| `MAX_RULE_LEN` | `16` | Maximum single rule string length in GPU buffers |
| `MAX_OUTPUT_LEN` | `512` | Maximum transformed word output length in GPU buffers |
| `MAX_CHAIN_STRING_LEN` | `128` | Maximum chained rule string length in GPU buffers |
| `MAX_HASHCAT_CHAIN` | `31` | Maximum number of rules in a single Hashcat chain |
| `MINIMIZE_DISK_THRESHOLD` | `500,000` | Rule count above which minimization switches to the disk-backed SQLite path |
| `MINIMIZE_DISK_BATCH_SIZE` | `10,000` | Rows per batch in the SQLite minimization path |
| `BLOOM_NUM_SHARDS` | auto | Number of Bloom filter shards (carried over from v2); auto-computed above the ~512 MB single-buffer threshold, forced to 1 with `--bloom-no-shard` |

> Stage 3 GA parameters (`--genetic-pop`, `--genetic-generations`, `--genetic-elite`) and Stage-selection parameters (`--select-budget`, `--select-cost-alpha`) are CLI-only and have no corresponding module-level constants.

---

## 📄 Output Format

`rulest_output.txt` (or your specified `-o` path). Header format for `--select-mode frequency` (default):

```
# rulest — GPU-Compatible Hashcat Rules Engine (optimized)
# Generated      : 2026-01-14 14:32:07
# Base           : rockyou.txt
# Target         : target_plain.txt
# Depth          : 1-3
# Bloom          : auto MB  (sharding: auto)
# STAGE 0        : core + insert mode  GPU-verified candidates
# STAGE 3 GA     : pop=200  gen=50  elite=15%
#
# GPU raw candidates      : 9,214
# Minimization            : in-memory  (threshold 500,000)
#   Probe words           : 37  (built-in)
#   Equiv. rules removed  : 4,393
#
# Rules kept     : 4,821
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

With `--select-mode greedy`, the header additionally includes the selection mode/budget and target recovery percentage:

```
# Rules kept     : 1,500
# Sorted by      : greedy marginal coverage (CELF, cost_alpha=0.0)
# SELECT mode     : greedy (budget=1500)
# Target recovery : 84,213/100,000 (84.21%)  (exact distinct target coverage)
```

- The `# STAGE 3 GA` header line is only written when `--genetic` is active.
- The identity rule (`:`) is always written first for Hashcat compatibility.
- With `--select-budgets`, additional files `<output>.<budget>.txt` are written for each requested size, each with its own header.
- Encoding is `utf-8`.

---

## 🎛 Performance Tuning

| Goal | Recommendation |
|---|---|
| Maximize coverage in fixed time | Increase `--target-hours` |
| Use every GPU in the machine | `--device all` |
| Use specific fast GPUs only | `--device 0,1,3` (avoid mixing in slow/mismatched devices) |
| Avoid Bloom sharding overhead | `--bloom-no-shard` (only if your driver handles large single allocations well) |
| Get the most diverse ruleset for a fixed rule count | `--select-mode greedy --select-budget N` |
| Compare recovery at several rule-budget sizes without re-running | `--select-mode greedy --select-budgets "100,1000,10000,50000"` |
| Prefer shorter/cheaper chains in greedy selection | Raise `--select-cost-alpha` (e.g. `0.5`) |
| Confirm real recovery % (not bloom estimate) | Add `--exact-recovery` |
| Skip built-in seed families | `--no-builtin-seeds` |
| Reduce VRAM pressure | Lower `--max-chains` or use `--depth2-chains` / `--depth3-chains` |
| Force deep chain exploration | Set `--depth4-chains 50000 --depth5-chains 10000` explicitly |
| Bootstrap from prior results | Pass previous output to `--seed-rules` for iterative refinement |
| Enable token-strip pre-pass | Add `--token-strip` — most effective for structured targets (leet speak, capitalized words, boundary digits) |
| Enable evolutionary search | Add `--genetic` with `--max-depth 3` or higher |
| Speed up GA per generation | Lower `--genetic-pop` |
| Improve GA convergence quality | Raise `--genetic-pop` and `--genetic-generations` |

---

## ✅ Testing

Unit tests are included under `tests/` (pytest), covering rule-pool synchronization across GPUs, greedy/CELF selection logic, functional minimization, common utilities, I/O helpers, and exact-output parity for multi-GPU runs vs. single-GPU runs.

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 💡 Examples

**Classic v2-equivalent behaviour (default, single GPU, frequency sort):**
```bash
python run_rulest.py rockyou.txt target.txt --max-depth 2 -o rules.txt
```

**Run on every detected GPU:**
```bash
python run_rulest.py rockyou.txt target.txt --device all --max-depth 3 --target-hours 2.0 -o rules.txt
```

**Run on two specific GPUs only:**
```bash
python run_rulest.py rockyou.txt target.txt --device 0,2 --max-depth 3 -o rules.txt
```

**Greedy selection, keep the best 50k rules:**
```bash
python run_rulest.py base.txt target.txt -o rules.txt \
    --select-mode greedy --select-budget 50000
```

**Greedy selection with a budget sweep and exact recovery reporting:**
```bash
python run_rulest.py base.txt target.txt -o rules.txt \
    --select-mode greedy \
    --select-budgets 64,250,1500,10000,25000,50000,150000 \
    --exact-recovery
```

**Cost-aware greedy selection favoring shorter chains:**
```bash
python run_rulest.py base.txt target.txt -o rules.txt \
    --select-mode greedy --select-budget 20000 --select-cost-alpha 0.5
```

**Token-strip + multi-GPU + genetic algorithm, all combined:**
```bash
python run_rulest.py rockyou.txt target.txt \
    --device all \
    --token-strip \
    --max-depth 4 --target-hours 2.0 \
    --genetic --genetic-pop 300 --genetic-generations 75 \
    --select-mode greedy --select-budget 100000 \
    -o full_pipeline.txt
```

**Large target — bloom sharding handled automatically (behavior carried over from v2):**
```bash
python run_rulest.py base.txt large_target.txt \
    --bloom-mb 1024 \
    --max-depth 4 --target-hours 1.0 \
    -o large_bloom.txt
# Force a single unsharded buffer instead:
python run_rulest.py base.txt large_target.txt \
    --bloom-mb 1024 --bloom-no-shard \
    --max-depth 4 --target-hours 1.0 \
    -o large_bloom_unsharded.txt
```

**List available OpenCL devices:**
```bash
python run_rulest.py --list-devices
```

### 💡 Pro Tips

- Use a **smaller, high-quality base wordlist** (10k–150k entries) — often outperforms massive dictionaries.
- `--select-mode greedy` is most valuable when you need a **small, deployable** ruleset (e.g. for repeated cracking runs) rather than the largest possible candidate pool — it trades exhaustive coverage for maximum coverage-per-rule.
- Start with `--select-budgets` during experimentation to see the recovery curve before committing to a single budget.
- On mixed-vendor multi-GPU systems, let the adaptive load balancer calibrate for a stage or two before judging split quality — the first stage after startup defaults to equal shares.

---

## 📝 License

MIT

## Credits

@Shooter3k for testing script on multi-GPU system

https://github.com/synacktiv/rulesfinder
