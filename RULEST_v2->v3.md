# rulest — v2 → v3 changelog

`rulest` is a GPU (OpenCL) engine for generating and selecting hashcat
rules from a base wordlist / target wordlist pair (e.g. a leaked password
list). Below is a summary of the differences between `rulest_v2.py` and
`rulest_v3.py`, followed by a benchmark of `rulest.freq*`/`rulest.greedy*`
rules against the community comparison spreadsheet.

## Summary

v3 adds a **second rule-selection mode** after extraction: instead of
always sorting the final rule pool by raw GPU hit frequency, you can now
select a subset of rules using **lazy-greedy (CELF) marginal-coverage
set-cover** — this maximizes the actual union coverage of the target
wordlist per rule budget, instead of summing individual rule scores. v3
also adds an optional exact (non-bloom) verification pass and a few GPU
stability fixes.

## Default-behavior change

None — `--select-mode` still defaults to `frequency`, i.e. **exactly** the
old raw-GPU-frequency sort. The new selection path is fully opt-in.

---

# `rulest.freq*` vs `rulest.greedy*` — benchmark

**Important interpretive caveat:** `rulest.freq*` and `rulest.greedy*` are
outputs of a fully **automated pipeline** (raw GPU generation + one of two
mechanical selection modes — frequency sort or CELF), not the result of
manual, expensive tuning/debugging like most other rulesets in the
comparison [spreadsheet](https://docs.google.com/spreadsheets/d/1ctT9o-hdMaJMe9ZM7e6t9x0uDmGphvOixe0FHnHGCEI/edit?gid=1513384572#gid=1513384572) these were pulled from. Keep that in mind when
reading the numbers below — there's no external, hand-curated ruleset in
this table, only the two `rulest` selection modes compared against each
other at matched budgets.

### `rulest.freq*` vs `rulest.greedy*` at matched budgets

| Budget (~) | File | Recovered | Rules | Crack rate (/h) |
|---|---|---:|---:|---:|
| 64 | `rulest.greedy.64.rule` | **1.45%** | 64 | 1 618 |
| 64 | `rulest.freq.64.rule` | 0.79% | 64 | 881 |
| 100 | `rulest.greedy.100.rule` | **1.66%** | 100 | 1 853 |
| 100 | `rulest.freq.100.rule` | 1.00% | 100 | 1 111 |
| 500 | `rulest.greedy.500.rule` | **2.84%** | 500 | 3 160 |
| 500 | `rulest.freq.500.rule` | 2.33% | 500 | 2 600 |
| 1253 | `rulest.freq.1200.rule` | **4.07%** | 1 253 | 4 536 |
| 1237 | `rulest.greedy.1200.rule` | 3.50% | 1 237 | 3 902 |

### Takeaways

- At the smaller budgets (64, 100, 500), `greedy` (CELF) consistently beats
  `freq`: it recovers roughly 1.7–1.85x more at 64 rules (1.45% vs 0.79%)
  and 100 rules (1.66% vs 1.00%), and a solid margin at 500 (2.84% vs
  2.33%). This is exactly what marginal-coverage selection is meant to do —
  at a tight budget, avoiding rules that overlap in what they recover
  matters more than picking rules with the highest raw hit count.
- At ~1200 rules the relationship **flips** — `freq` (4.07%) beats `greedy`
  (3.50%). This lines up with the earlier saturation observation: CELF on
  this candidate pool exhausted its useful marginal gains around 1219
  rules (every rule after that added zero new coverage), so a `greedy`
  selection at budget 1200 is essentially running to saturation and
  stopping there — while `freq` keeps adding rules with real (if
  overlapping) hits past that point, which is enough to edge ahead at this
  particular budget size.
- Net effect: `--select-mode greedy` is the better default for small rule
  budgets against `rulest`-generated candidates, but its advantage
  shrinks — and can invert — once the budget approaches the candidate
  pool's coverage-saturation point. If you're targeting a large ruleset,
  it's worth checking where that saturation point lands for your specific
  base/target wordlist pair before assuming `greedy` is strictly better.

## New CLI flags (`Rule Selection` group)

| Flag | Default | Description |
|---|---|---|
| `--select-mode {frequency,greedy}` | `frequency` | Final rule-selection mode for the output file. `frequency` — identical to v2: descending sort by raw GPU hit count (ignoring coverage overlap between rules). `greedy` — lazy-greedy (CELF) marginal-coverage set-cover: builds a full coverage matrix (`evaluate_full_coverage`) for every candidate rule against the real `target_wordlist`, then repeatedly picks the rule with the largest **marginal** gain in unique coverage relative to what's already selected — maximizes total union coverage, not the sum of individual rule scores. |
| `--select-budget N` | `0` (= run to saturation) | Maximum number of rules to keep with `--select-mode greedy`. `0` means unbounded — the CELF loop stops on its own once no remaining rule adds any new coverage (`gain <= 0`), which can happen well before the candidate pool is exhausted (see the saturation observation at ~1200 rules). |
| `--select-budgets "100,1000,10000,..."` | `None` | Comma-separated list of budgets — for each size, writes an additional `<output>.<budget>.txt` file that is a prefix of **the same** ordering (greedy or frequency, depending on `--select-mode`), without re-computing coverage. Lets you compare recovery/size for many budgets in a single GPU run (this is exactly how the `rulest.freq.64/100/500/1200` and `rulest.greedy.64/100/500/1200` pairs in the benchmark above were produced). In `frequency` mode the sweep files are size-matched only (no % recovery — frequency mode never builds the coverage matrix, since that's expensive and only `greedy` needs it). |
| `--select-cost-alpha F` | `0.0` | Cost modifier in the CELF objective: `gain(r) = new_recovery(r) / depth(r)**alpha`. `0.0` = pure marginal-coverage greedy (rules weighted purely by coverage gain). Values `> 0` start favoring shorter/cheaper rule chains at similar coverage gain — useful when you also care about the computational cost of the final ruleset on hashcat (shorter rules = higher crack rate/h), not just raw % recovered. |
| `--exact-recovery` | off | After `greedy` selection finishes, runs one extra, one-shot GPU pass (`compute_exact_recovery` + the new `verify_pairs_gpu` kernel) over the **already-selected**, small set of (rule, base_word) pairs flagged as hits — no bloom filter, so no false positives (FPR). Returns the true percentage of distinct `target_words` entries actually cracked, as opposed to the default selection result, which is only a proxy computed relative to `base_words` and may include bloom-filter false positives. Costs extra GPU time, but is cheap since it only touches the final, small ruleset — not the whole candidate pool. |
