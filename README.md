# 🔐 GPU-Accelerated Hashcat Rule Extractor
 
> Extract and chain Hashcat-compatible rules from wordlists using OpenCL GPU acceleration.
 
---
 
## 📋 Table of Contents
 
- [Overview](#-overview)
- [Scripts](#-scripts)
  - [rulest.py — v1 (BFS, Legacy)](#rulestpy--v1-bfs-legacy)
  - [rulest\_v2.py — v2 (Recommended)](#rulest_v2py--v2-recommended)
- [Why v2 Supersedes v1](#-why-v2-supersedes-v1)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
  - [rulest\_v2.py — Full Reference](#rulest_v2py--full-reference)
- [Architecture (v2)](#-architecture-v2)
  - [Extraction Pipeline](#extraction-pipeline)
- [Built-in Seed Families (A–M)](#-built-in-seed-families)
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
- ✅ **Three-phase extraction**: single-rule sweep (Phase 1) → built-in seed pass (Phase S) → informed chain generation (Phase 2)
- ✅ **Built-in seed families** (A–M): thirteen deterministically generated seed families covering numeric prefixes/suffixes, mixed prepend/append, transform+digit combos, date patterns, special-character append/prepend/transform/combo patterns, leet substitutions, double-transform chains, special-before-digit patterns, and leet+transform combos — run by default as a dedicated extraction pass, independent of `--max-depth` and the random-chain time budget; can be skipped with `--no-builtin-seeds`
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
| **Rule set size** | ~2,700 static rules | 5,600+ GPU-validated Hashcat single rules across 9 categories |
| **Search strategy** | Naive BFS — every rule applied blindly | Phase 1 single-rule sweep → Phase S built-in seed extraction → Phase 2 hot-biased chain generation |
| **Built-in seed families** | ❌ Not implemented | ✅ Thirteen families (A–M): numeric prepend/append, mixed, transform+digit, date patterns, special-char append/prepend/transform/combo (F–I), leet substitutions (J), double-transform chains (K), special-before-digit (L), leet+transform (M); run by default as a dedicated pass independent of `--max-depth`; disable with `--no-builtin-seeds` |
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
| `--target-hours` | `0.5` | Time budget in hours; controls chain generation budget |
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
│  │  │ (16–256 MB   │    │  ┌─────────────────┐  │ │  │
│  │  │  VRAM)      │    │  │find_single_rules│  │ │  │
│  │  └─────────────┘    │  ├─────────────────┤  │ │  │
│  │                     │  │find_rule_chains │  │ │  │
│  │  Phase 1 ────────▶  │  └─────────────────┘  │ │  │
│  │  (all words ×       └───────────────────────┘ │  │
│  │   single rules)                               │  │
│  │                                               │  │
│  │  Phase S ────────▶  Built-in seed families   │  │
│  │  (Families A–M;      direct extraction pass,  │  │
│  │   default on;        depth 2–9 seeds;         │  │
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
 
**Phase S — Built-in Seed Extraction**
A dedicated extraction pass that runs the thirteen built-in seed families (A–M) through the GPU chain kernel. This phase runs by default, regardless of `--max-depth` and the random-chain time budget; it can be disabled with `--no-builtin-seeds`. Depth-1 seeds are skipped (already covered by Phase 1); all multi-rule seed chains at depths 2 and above are tested directly against the Bloom filter. The prebuilt seed families are then forwarded to Phase 2 as scaffolding atoms to avoid regeneration and double-counting. See [Built-in Seed Families (A–M)](#-built-in-seed-families) for a full description.
 
**Phase 2 — Informed Chain Generation**
Using Phase 1 hit data, chains are generated with a bias toward rules that already demonstrated effectiveness:
- **60%** of generated chains use hot rules from Phase 1 (`HOT_RULE_RATIO = 0.6`)
- **30%** of the budget extends known-good seed chains (`EXTENSION_RATIO = 0.3`)
- **10%** is allocated to random exploration
 
The remaining time budget (total `--target-hours` minus Phase 1 + Phase S duration) is split evenly across requested depths. User seed rules from `--seed-rules` are extended to deeper depths automatically.
 
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
| G | Prepend special chars | v2| 3 600 |
| H | Transform + special char | v2 | 13 950 |
| I | Digit(s) + special char | v2 | 15 540 |
| J | Leet substitutions | v2 | ~520 |
| K | Double-transform | v2 | 225 |
| L | Special-before-digit |v2 | 1 540 |
| M | Leet + transform | v2 | ~300 |
 
---
 
## 🔬 Functional Minimization
 
After GPU extraction, `rulest_v2.py` applies a **signature-based functional minimization** pass before writing the final output. This post-processing step removes redundant rules — rules that are syntactically different but produce identical outputs on real words — so the resulting ruleset is as compact as possible without losing coverage.
 
### Algorithm
 
1. **Use the built-in probe set.** A fixed, hand-curated set of probe words is always used (no external file or CLI flag required). The set is designed to exercise every class of hashcat opcode: very short words (edge cases for `k`, `K`, `{`, `}`, `[`, `]`), short alphanumeric base words, typical password base words of lengths 7–9, longer words for truncation and repeat ops, mixed-case words, words with embedded digits, words with special characters, words with repeated characters, and dedicated leet-substitution targets (`master`, `leet`, `elite`, `access`). Because the probe set is deterministic and ships with the script, minimization results are fully reproducible across runs with no configuration needed.
 
2. **Compute each rule's signature.** Every candidate rule (or chain) is applied to every probe word using a pure-Python interpreter (`py_apply_chain`). The signature is the resulting tuple of transformed strings — one per probe word. Rules containing opcodes that cannot be emulated in Python are assigned the sentinel signature `('__UNSUPPORTED__',)` and are bucketed together.
 
3. **Group by signature.** Rules sharing an identical signature are considered **functionally equivalent** on the probe set. Only one representative survives from each group.
 
4. **Select the best representative.** Within each signature group, the rule with the **highest GPU hit-count** is kept. Ties are broken by preferring shorter chain depth, then lexicographic order.
 
5. **Write the minimized ruleset.** Surviving rules are written to the output file sorted by GPU frequency (descending). The file header records the probe word count and how many equivalent rules were removed.
 
### Why It Matters
 
GPU Bloom filter screening (Phase 1 & 2) can yield thousands of candidates where many are functionally identical — for example, `c` (capitalize first letter) and a chain `l c` applied to an already-lowercase word produce the same output. Without minimization, the output file contains duplicate work that inflates Hashcat's rule-testing time without adding new candidate passwords.
 
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
| `MAX_GPU_RULES` | `31` | Maximum operations allowed per rule chain |
| `BASELINE_COMBOS_PER_SEC` | `120,000,000` | Estimated throughput on a capable GPU |
| `LOW_END_COMBOS_PER_SEC` | `40,000,000` | Throughput fallback for devices with < 20 compute units |
| `MAX_WORD_LEN` | `256` | Maximum word length accepted from wordlists |
| `MAX_RULE_LEN` | `16` | Maximum single rule string length in GPU buffers |
| `MAX_OUTPUT_LEN` | `512` | Maximum transformed word output length in GPU buffers |
| `MAX_CHAIN_STRING_LEN` | `128` | Maximum chained rule string length in GPU buffers |
| `MAX_HASHCAT_CHAIN` | `31` | Maximum number of rules in a single Hashcat chain |
 
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
#   Probe words           : 38  (built-in)
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
| Skip built-in seed families | Pass `--no-builtin-seeds` to skip Phase S entirely; useful when supplying all seeds via `--seed-rules` or benchmarking Phase S contribution (families A–M) |
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
| 4–8 GB | 0.5–1.0× | 128 MB |
| 8 GB+ | 1.0× (full) | 256 MB |
 
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
**Skip built-in seed families (Phase S disabled):**
```bash
# Faster run when you supply all seeds yourself and don't need families A–M
python rulest_v2.py base.txt target.txt -d 3 --target-hours 1.0 \
  --seed-rules my_seeds.txt \
  --no-builtin-seeds \
  -o no_phase_s.txt
```
 
**Benchmark Phase S contribution:**
```bash
# With built-in seeds (default — families A–M)
python rulest_v2.py base.txt target.txt -d 2 -o with_seeds.txt
 
# Without built-in seeds — compare output sizes to measure Phase S value
python rulest_v2.py base.txt target.txt -d 2 --no-builtin-seeds -o without_seeds.txt 
---
 
## 📝 License
 
MIT
 
## Credits
 
MIT
 
## Credits
https://github.com/synacktiv/rulesfinder
