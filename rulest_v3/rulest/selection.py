"""GPU target-table-backed memory-light CELF selection.

Fast batched GPU scans produce admissible upper bounds (one count per matching
base-word output, so duplicate outputs may overcount but never undercount
unique active targets). Singleton scans use a per-candidate bitmap and return
exact distinct target IDs. CELF only selects a candidate after exact scoring.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
import time

import numpy as np
# Keep the custom heartbeat-driven bar (not plain tqdm): CELF rounds can go
# many seconds between .update() calls (one exact GPU word-sweep per round),
# and this bar keeps elapsed/ETA live via a background refresh thread instead
# of looking stalled. Styling below is normalized to match the other phases'
# real-tqdm bars (green desc, cyan postfix, same bar glyphs/format).
from .progress import tqdm, ActivityHeartbeat  # CELF_PROGRESS_DIAGNOSTICS_PATCH_V2

from .common import (
    MAX_CHAIN_STRING_LEN,
    MAX_HASHCAT_CHAIN,
    bold,
    cyan,
    green,
    log_info,
    log_section,
    log_warn,
)
# RULEST_TARGET_DEDUPE_INPLACE_SORT_V9
def _rulest_dedupe_targets_with_progress(target_words):
    """Memory-bounded exact dedupe by in-place sort + compaction.

    This deliberately avoids a Python set/dict: with millions of mostly-unique
    target strings, a set plus an output list adds large hash-table/pointer
    overhead while the original list is still live. Target ordering is not
    semantically meaningful to CELF; target IDs are assigned after sorting.

    If `target_words` is already a list, it is sorted and compacted in place.
    This preserves the list object and string objects and uses only sort's
    temporary pointer array (rather than another O(n) set plus list).

    Progress is reported through the same single-line ActivityHeartbeat used
    for "[CELF] Building heap" and "[CELF GPU] Building target hash map" right
    before and after this runs, instead of this function's own unstyled,
    unthrottled `[HASHMAP] ...` print() -- which used to emit a brand new
    scrolling line at least once a second and looked like a different tool
    entirely in the middle of the selection stage's output.
    """
    import time as _dedupe_time
    if not isinstance(target_words, list):
        target_words = list(target_words)
    total = len(target_words)
    started = _dedupe_time.monotonic()
    with ActivityHeartbeat(green(f"[CELF] Target dedupe ({total:,} strings)")) as heartbeat:
        target_words.sort()
        sorted_at = _dedupe_time.monotonic()
        if total == 0:
            return target_words

        write = 1
        for read in range(1, total):
            value = target_words[read]
            if value != target_words[write - 1]:
                if write != read:
                    target_words[write] = value
                write += 1
            now = _dedupe_time.monotonic()
            # Last row always re-renders (force=True) so the heartbeat's own
            # "done" line, printed right after this block exits, reflects the
            # final unique count rather than a stale mid-sweep one.
            heartbeat.set_desc(
                green(
                    f"[CELF] Target dedupe {read + 1:,}/{total:,} "
                    f"({100.0*(read+1)/max(total,1):.1f}%, unique={write:,}, "
                    f"{(read+1)/max(now-sorted_at,1e-9):,.0f} rows/s)"
                ),
                force=(read + 1 == total),
            )
        del target_words[write:]
    return target_words



# CELF_ADAPTIVE_BATCH_PATCH_V4
def _celf_auto_rescore_batch(requested, target_count, gpu_engine=None):
    """Conservative rescore batch: VRAM is not the only cost driver.

    `requested` is the available/preallocated ceiling. `target_count` captures
    per-rule workload; large target universes use smaller speculative batches to
    avoid rescoring hundreds of contenders that CELF may not need.
    """
    try:
        requested = max(1, min(1024, int(requested)))
    except (TypeError, ValueError):
        requested = 32
    try:
        target_count = max(1, int(target_count))
    except (TypeError, ValueError):
        target_count = 1_000_000  # conservative when the count isn't available

    device = getattr(gpu_engine, "device", None) if gpu_engine is not None else None
    if device is None and gpu_engine is not None:
        queue = getattr(gpu_engine, "queue", None)
        device = getattr(queue, "device", None)
    if device is None and gpu_engine is not None:
        context = getattr(gpu_engine, "context", None)
        try:
            devices = getattr(context, "devices", ())
            device = devices[0] if devices else None
        except Exception:
            device = None

    vram = 0
    if device is not None:
        try:
            vram = int(getattr(device, "global_mem_size", 0) or 0)
        except Exception:
            vram = 0
        if not vram and hasattr(device, "get_info"):
            try:
                vram = int(device.get_info(0x101F))  # CL_DEVICE_GLOBAL_MEM_SIZE
            except Exception:
                vram = 0

    gib = 1024 ** 3
    if not vram:
        vram_cap = 32
        vram_label = "unknown"
    elif vram < 4 * gib:
        vram_cap, vram_label = 16, f"{vram / gib:.1f} GiB"
    elif vram < 8 * gib:
        vram_cap, vram_label = 32, f"{vram / gib:.1f} GiB"
    elif vram < 16 * gib:
        vram_cap, vram_label = 64, f"{vram / gib:.1f} GiB"
    elif vram < 24 * gib:
        vram_cap, vram_label = 128, f"{vram / gib:.1f} GiB"
    else:
        vram_cap, vram_label = 256, f"{vram / gib:.1f} GiB"

    # Observed behaviour: on million-entry targets, huge speculative groups
    # (512) were slower than smaller CELF rescores. Keep the cap workload-aware.
    if target_count >= 1_000_000:
        work_cap = 32
    elif target_count >= 100_000:
        work_cap = 32
    elif target_count >= 25_000:
        work_cap = 64
    else:
        work_cap = 128

    chosen = max(1, min(requested, vram_cap, work_cap))
    override = os.environ.get("RULEST_CELF_RESCORE_BATCH_MAX")
    if override:
        try:
            chosen = max(1, min(chosen, int(override)))
        except ValueError:
            pass
    if not getattr(_celf_auto_rescore_batch, "_reported", False):
        print(
            f"[CELF AUTO] rescore batch={chosen} (workspace ceiling={requested}, "
            f"targets={target_count:,}, VRAM={vram_label}, "
            f"caps: VRAM={vram_cap}, workload={work_cap})",
            flush=True,
        )
        _celf_auto_rescore_batch._reported = True
    return chosen


_FNV1A64_OFFSET = 14695981039346656037
_FNV1A64_PRIME = 1099511628211
_UINT64_MASK = (1 << 64) - 1


def _fingerprint64(text: str) -> int:
    """Stable 64-bit FNV-1a fingerprint over Rulest's byte encoding (Latin-1)."""
    value = _FNV1A64_OFFSET
    for byte in text.encode("latin-1", errors="strict"):
        value = ((value ^ byte) * _FNV1A64_PRIME) & _UINT64_MASK
    return value


def _splitmix64(value: int) -> int:
    """SplitMix64 finalizer, matching the OpenCL slot mixer."""
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


class _TargetFingerprintIndex:
    """Small CPU helper retained for collision-focused unit tests."""

    def __init__(self, targets, fingerprint_fn=_fingerprint64):
        self.targets = list(targets)
        self.fingerprint_fn = fingerprint_fn
        buckets = {}
        for target_id, target in enumerate(self.targets):
            buckets.setdefault(fingerprint_fn(target), []).append(target_id)
        self.buckets = buckets

    def lookup(self, candidate):
        for target_id in self.buckets.get(self.fingerprint_fn(candidate), ()):
            if self.targets[target_id] == candidate:
                return target_id
        return None


def _rule_cost(rule, cost_alpha):
    if cost_alpha <= 0:
        return 1.0
    return max(1, len(rule.split())) ** float(cost_alpha)


class _ConstantCosts:
    """costs[index] == value for any index, without allocating a len(rules)
    list — used for the (default) cost_alpha<=0 case where every rule's
    cost is the same 1.0."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self.value


def _celf_select_from_upper_bounds(
    rules, upper_bounds, target_count, score_fn, budget=None, cost_alpha=0.0,
    on_select=None, score_batch_fn=None, rescore_batch_size=32,
    on_scan_progress=None,
):
    """Heap CELF with fast admissible bounds and exact singleton winners.

    Batch scoring intentionally counts matching base-word outputs, so duplicates
    may inflate a bound but can never make it smaller than true distinct-target
    marginal coverage. Only singleton scoring (which deduplicates target IDs) is
    allowed to establish the round winner. This is the same cheap bound contract
    used by ranker's GPU CELF and avoids comparing two different scoring modes as
    though both were exact.
    """
    if len(rules) != len(upper_bounds):
        raise ValueError("rules and upper_bounds must have equal lengths")
    if target_count < 0:
        raise ValueError("target_count cannot be negative")

    limit = len(rules) if budget is None else min(len(rules), max(0, int(budget)))
    covered = np.zeros(int(target_count), dtype=bool)

    # Building `costs` and the heap used to be two plain-Python loops over
    # every candidate with zero progress feedback — on multi-million-rule
    # pools this is exactly the silent gap between the (fast, GPU) upper-bound
    # pass finishing and the first tick of the "CELF GPU recompute" bar.
    # Report it with the same heartbeat spinner used for GPU setup, and skip
    # the per-rule Python cost loop entirely in the common cost_alpha<=0 case
    # (cost is always 1.0, so priority == bound and no per-rule division or
    # function call is needed at all).
    with ActivityHeartbeat(f"[CELF] Building heap ({len(rules):,} candidates)"):
        bounds_arr = np.maximum(0, np.asarray(upper_bounds, dtype=np.int64))
        positive = np.flatnonzero(bounds_arr)
        if cost_alpha <= 0:
            # costs[index] is still read later (priority()/push() index into
            # it by full rule index, including rules that start at bound=0
            # but get re-pushed with a tightened bound later), so this needs
            # to stay a full-length, O(1)-indexable sequence either way — it
            # just doesn't need per-rule Python work when cost is always 1.0.
            costs = _ConstantCosts(1.0)
        else:
            costs = [_rule_cost(rule, cost_alpha) for rule in rules]
        heap = [
            (-float(bounds_arr[i]) / costs[int(i)], int(i), int(bounds_arr[i]))
            for i in positive.tolist()
        ]
        heapq.heapify(heap)
        # Every rule this round can ever select comes from a heap entry, and
        # the heap is only ever seeded from (and re-pushed among) these
        # `positive` indices -- a candidate with bound<=0 never enters it and
        # push() only re-adds indices that were already popped from it. So
        # len(positive) is a true, closed upper bound on how many rules this
        # run could possibly select, even with no --budget. It is normally a
        # loose one (saturation -- overlapping coverage -- almost always
        # stops the run well before every positive-bound candidate gets
        # picked), so an ETA based on it will run noticeably optimistic
        # relative to wall time, but it beats no ETA at all for an unbounded
        # run.
        approx_total_without_budget = int(len(positive)) or None

    batch_size = max(1, int(rescore_batch_size))
    selected = []
    exact_rescored = 0
    batch_rescored = 0
    recovered = 0
    current_batch_size = 1
    progress_enabled = bool(sys.stderr.isatty())
    last_refresh = [0.0]
    t0 = time.monotonic()

    # RULEST_CELF_INDETERMINATE_PROGRESS_V1
    # leave/dynamic_ncols/ncols/bar_format used to be passed here as if this
    # were real tqdm, but rulest's own ProgressBar (see progress.py) builds
    # its line -- bar, counts, elapsed<ETA, rate -- and recomputes terminal
    # width itself on every render; it never reads those fields, so passing
    # them was dead weight. `disable` is the one that mattered: it silently
    # did nothing until progress.py was fixed to honor it, so a non-tty
    # stderr (redirected to a log, piped, cron) still got this bar's raw
    # \r/\x1b[K control bytes -- ETA included -- written straight into it.
    celf_bar_desc = "[CELF] GPU recompute" if budget else "[CELF] GPU recompute (approx)"
    with tqdm(
        total=budget if budget else approx_total_without_budget,
        desc=green(celf_bar_desc), unit="rule",
        disable=not progress_enabled, mininterval=0.5,
    ) as progress:
        def report_scan(done=None, total=None):
            if not progress_enabled:
                return
            # Order matters: progress.py now drops whole postfix fields from
            # the right when the line doesn't fit the terminal, instead of
            # slicing mid-field — so put the numbers people actually watch
            # (recovered, selected) first, and the noisier internal counters
            # last, so those are the first to go on a narrow terminal.
            postfix = {
                "recovered": cyan(f"{recovered:,}"),
                "selected": cyan(str(len(selected))),
                "batch": cyan(str(current_batch_size)),
                "exact scans": cyan(f"{exact_rescored:,}"),
                "batch bounds": cyan(f"{batch_rescored:,}"),
            }
            if done is not None and total is not None:
                postfix["scan words"] = cyan(f"{int(done):,}/{int(total):,}")
            progress.set_postfix(postfix, refresh=False)
            now = time.monotonic()
            if now - last_refresh[0] >= 0.5:
                progress.refresh()
                last_refresh[0] = now

        def exact_score(index):
            nonlocal exact_rescored
            exact_rescored += 1
            if on_scan_progress is None:
                ids = score_fn(index, covered)
            else:
                ids = score_fn(index, covered, report_scan)
            ids = {
                int(target_id) for target_id in ids
                if 0 <= int(target_id) < target_count and not covered[int(target_id)]
            }
            report_scan()
            return len(ids), ids

        def priority(bound, index):
            return max(0, int(bound)) / costs[int(index)]

        def can_beat(bound, index, best_gain, best_index):
            p = priority(bound, index)
            best_p = priority(best_gain, best_index)
            return p > best_p or (p == best_p and int(index) < int(best_index))

        def push(index, bound):
            bound = max(0, int(bound))
            if bound:
                heapq.heappush(heap, (-priority(bound, index), int(index), bound))

        while heap and len(selected) < limit:
            # Start each round with the strongest outstanding bound, but never
            # assume that its upper bound is the exact gain.
            _neg_seed_priority, seed_index, _seed_bound = heapq.heappop(heap)
            seed_index = int(seed_index)
            best_gain, best_ids = exact_score(seed_index)
            if best_gain <= 0 or not best_ids:
                # Marginal gain can only shrink; zero-gain candidates are dead.
                continue
            best_index = seed_index
            fresh_this_round = set()

            while heap:
                _neg_top, top_index, top_bound = heap[0]
                top_index = int(top_index)
                if not can_beat(top_bound, top_index, best_gain, best_index):
                    break

                # A freshly batch-scored contender still has an upper bound, not
                # an exact gain. Evaluate it singly before it can replace winner.
                if top_index in fresh_this_round:
                    heapq.heappop(heap)
                    fresh_this_round.remove(top_index)
                    gain, ids = exact_score(top_index)
                    if gain > best_gain or (gain == best_gain and top_index < best_index):
                        push(best_index, best_gain)  # previous exact winner remains a bound
                        best_index, best_gain, best_ids = top_index, gain, ids
                    else:
                        push(top_index, gain)
                    continue

                # Refresh only stale bounds that might still beat the exact best.
                batch_entries = []
                while heap and len(batch_entries) < batch_size:
                    _neg_p, index, old_bound = heap[0]
                    index = int(index)
                    if not can_beat(old_bound, index, best_gain, best_index):
                        break
                    if index in fresh_this_round:
                        break
                    heapq.heappop(heap)
                    batch_entries.append((index, int(old_bound)))
                if not batch_entries:
                    # The top should now be a fresh contender; next loop exact-scores it.
                    continue

                batch_indices = [index for index, _ in batch_entries]
                current_batch_size = len(batch_indices)
                batch_rescored += len(batch_indices)
                if score_batch_fn is None:
                    # CPU-only/test fallback has exact scores; keep IDs so it does
                    # not re-run the same candidate as a speculative singleton.
                    for index, _old_bound in batch_entries:
                        gain, ids = exact_score(index)
                        if gain > best_gain or (gain == best_gain and index < best_index):
                            push(best_index, best_gain)
                            best_index, best_gain, best_ids = index, gain, ids
                        else:
                            push(index, gain)
                else:
                    if on_scan_progress is None:
                        raw_bounds = score_batch_fn(batch_indices, covered)
                    else:
                        raw_bounds = score_batch_fn(batch_indices, covered, report_scan)
                    if len(raw_bounds) != len(batch_entries):
                        raise RuntimeError("CELF batch scorer returned the wrong number of upper bounds")

                    for (index, old_bound), raw_bound in zip(batch_entries, raw_bounds):
                        # Monotonicity guarantees an updated bound cannot need to be
                        # larger than its previous value; min protects against a bad
                        # later batch while keeping the value admissible.
                        tightened = min(old_bound, max(0, int(raw_bound)))
                        push(index, tightened)
                        if tightened and can_beat(tightened, index, best_gain, best_index):
                            fresh_this_round.add(index)
                report_scan()

            # The winner was scored exactly in this round. Clear by its target IDs,
            # never by the last transient hit buffer (which another scan may replace).
            covered[np.fromiter(best_ids, dtype=np.int64)] = True
            recovered += len(best_ids)
            if on_select is not None:
                on_select(best_ids)
            selected.append((rules[best_index], len(best_ids)))
            progress.update(1)
            report_scan()

    if sum(gain for _, gain in selected) != int(covered.sum()):
        raise RuntimeError("CELF gain accounting disagrees with covered bitmap")
    return selected, covered

class _GpuExactScorer:
    """Thin adapter around the GPU-resident target table and exact-match kernels."""

    def __init__(self, gpu_engine, base_words, target_words, pair_batch_size, candidate_rules):
        self.gpu_engine = gpu_engine
        self.base_words = base_words
        self.targets = target_words if isinstance(target_words, list) else list(target_words)
        self.pair_batch_size = max(1, int(pair_batch_size))
        self.gpu_engine.set_base_words(base_words)
        self.gpu_engine.celf_initialize_target_table(self.targets)
        # Upload the atomic rule pool and every candidate's sequence once.
        self.gpu_engine.celf_prepare_candidate_pool(candidate_rules)

    def upper_bound_batch(self, rules):
        """Return fast admissible upper bounds (raw matching base-word outputs)."""
        if not rules:
            return []
        gains, _slots = self.gpu_engine.celf_score_chains_gpu(rules, record_hits=False)
        return [int(gain) for gain in gains]

    def score_gains_batch(self, rules, progress_callback=None):
        """Return fast admissible upper bounds; duplicates may overcount by design."""
        if not rules:
            return []
        gains, _slots = self.gpu_engine.celf_score_chains_gpu(
            rules, record_hits=False, progress_callback=progress_callback
        )
        return [int(gain) for gain in gains]

    def score_one_with_targets(self, rule, covered_mask=None, progress_callback=None):
        # If the heap popped the candidate just scored as a singleton, consume
        # its GPU-recorded slots instead of rescanning the complete wordlist.
        cached = self.gpu_engine.celf_take_cached_single_score(rule)
        if cached is None:
            gains, slots = self.gpu_engine.celf_score_chains_gpu(
                [rule], record_hits=True, progress_callback=progress_callback
            )
            gain = int(gains[0]) if len(gains) else 0
        else:
            gain, slots = cached
        target_ids = self.gpu_engine.celf_target_ids_for_slots(slots)
        if covered_mask is not None:
            target_ids = {target_id for target_id in target_ids if not covered_mask[target_id]}
        if len(target_ids) != gain:
            raise RuntimeError(
                f"GPU CELF hit-recording mismatch: gain={gain}, distinct target IDs={len(target_ids)}"
            )
        return gain, target_ids

    def mark_selected(self, target_ids):
        # Clear the exact target IDs returned by the winner scan. Do not infer
        # the winner from the last scratch hit-slot buffer: later speculative
        # singleton scans may have overwritten it.
        self.gpu_engine.celf_clear_target_ids(target_ids)

    def close(self):
        self.gpu_engine.celf_release_target_table()


def _widen_depth(gpu_engine, rules, requested_depth):
    """Try widening the kernel depth; return (restore_depth, effective_depth)."""
    params = getattr(gpu_engine, "params", None)
    if not isinstance(params, dict):
        return None, max(1, int(requested_depth))
    original = params.get("MAX_CHAIN_DEPTH")
    if not isinstance(original, int) or original <= 0:
        return None, max(1, int(requested_depth))
    needed = max((len(rule.split()) for rule in rules), default=1)
    effective = min(max(int(requested_depth), needed), MAX_HASHCAT_CHAIN)
    if effective <= original:
        return None, original
    try:
        params["MAX_CHAIN_DEPTH"] = effective
        gpu_engine.params = params
        if gpu_engine.compile_kernel(force=True):
            return original, effective
    except Exception as exc:
        log_warn(f"[CELF] Could not widen GPU chain depth: {exc}")
    try:
        params["MAX_CHAIN_DEPTH"] = original
        gpu_engine.params = params
        gpu_engine.compile_kernel(force=True)
    except Exception as exc:
        log_warn(f"[CELF] Failed to restore original GPU depth: {exc}")
    return None, original


def _restore_depth(gpu_engine, restore_depth):
    if restore_depth is None:
        return
    try:
        gpu_engine.params["MAX_CHAIN_DEPTH"] = restore_depth
        gpu_engine.params = gpu_engine.params
        gpu_engine.compile_kernel(force=True)
    except Exception as exc:
        log_warn(f"[CELF] Could not restore GPU chain depth: {exc}")


def select_rules_by_marginal_coverage(
    gpu_engine, base_words, target_words, rules, max_depth, budget=None,
    cost_alpha=0.0, pair_batch_size=100_000,
):
    """One GPU upper-bound pass followed by exact batched lazy-greedy CELF recompute."""
    log_section("SELECTION — CELF (GPU hash table + exact batched lazy recompute)")
    unique_targets = _rulest_dedupe_targets_with_progress(target_words)
    target_count = len(unique_targets)
    empty_mask = np.zeros(target_count, dtype=bool)
    if not rules or not unique_targets or not base_words:
        return [], empty_mask

    rules = list(dict.fromkeys(rule for rule in rules if rule and rule != ":" and rule.split()))
    rules = [
        rule for rule in rules
        if sum(len(token) for token in rule.split()) + max(0, len(rule.split()) - 1)
        <= MAX_CHAIN_STRING_LEN - 2
        and len(rule.split()) <= MAX_HASHCAT_CHAIN
    ]
    if not rules:
        log_warn("[CELF] No candidate rules fit the GPU chain limits")
        return [], empty_mask

    restore_depth, effective_depth = _widen_depth(gpu_engine, rules, max_depth)
    rules = [rule for rule in rules if len(rule.split()) <= effective_depth]
    if not rules:
        _restore_depth(gpu_engine, restore_depth)
        return [], empty_mask

    scorer = None
    try:
        scorer = _GpuExactScorer(
            gpu_engine, base_words, unique_targets, pair_batch_size, rules
        )
        upper_bounds = [0] * len(rules)
        seen_words = int(gpu_engine._celf_seen_words)
        scratch_limit = max(4, int(gpu_engine.params.get("CELF_SEEN_SCRATCH_MB", 32))) * 1024 * 1024
        max_by_scratch = max(1, scratch_limit // max(4, seen_words * 4))
        configured_batch = max(1, int(gpu_engine.params.get("CELF_RULE_BATCH_SIZE", 32)))
        rule_batch_size = min(
            configured_batch, max_by_scratch,
            max(1, int(getattr(gpu_engine, "_celf_batch_capacity", configured_batch))),
        )

        # See the comment on the CELF recompute bar above: leave/dynamic_ncols/
        # ncols/bar_format were dead (rulest's ProgressBar ignores them and
        # computes its own width/format), and `disable` now actually works.
        with tqdm(
            total=len(rules), desc=green("[CELF] Upper-bound pass (GPU target-table scan)"),
            unit="rule", disable=not sys.stderr.isatty(), mininterval=0.5,
        ) as progress:
            for start in range(0, len(rules), rule_batch_size):
                chunk = rules[start:start + rule_batch_size]
                bounds = scorer.upper_bound_batch(chunk)
                upper_bounds[start:start + len(chunk)] = bounds
                progress.update(len(chunk))

        # The exact-rescore batch inside the CELF heap loop stays at a fixed
        # default of 32 — unlike the upper-bound pass's rule_batch_size (and
        # its scratch buffer), which are sized from VRAM in devices.py. The
        # rescore batch only groups candidates for one exact-match kernel
        # launch per heap round; it doesn't benefit from VRAM headroom the
        # way the upper-bound pass's much larger one-shot scan does, so we
        # don't scale it with rule_batch_size any more.
        # RULEST_GREEDY_RESCORE_BATCH still overrides explicitly.
        recompute_batch_size = max(1, int(os.environ.get("RULEST_GREEDY_RESCORE_BATCH", "32")))
        recompute_batch_size = min(recompute_batch_size, rule_batch_size)
        recompute_batch_size = _celf_auto_rescore_batch(recompute_batch_size, target_count, getattr(scorer, 'gpu_engine', None))  # CELF_ADAPTIVE_BATCH_PATCH_V4
        log_info(
            f"[CELF] Candidates: {bold(str(len(rules)))}; targets: {bold(str(target_count))}; "
            f"fast raw-hit upper bounds (duplicates may overcount); lazy heap batch={recompute_batch_size}; "
            f"candidate sequences cached on GPU; SplitMix64 table load <= 50%; coverage matrix: {green('none')}"
        )

        def scan_progress(done, total):
            # Updates the persistent CELF progress bar; no nested progress bars.
            return None

        def score_one(rule_index, covered_mask, progress_callback=None):
            _gain, target_ids = scorer.score_one_with_targets(
                rules[rule_index], covered_mask=covered_mask,
                progress_callback=progress_callback or scan_progress,
            )
            return target_ids

        def score_many(rule_indices, covered_mask, progress_callback=None):
            # Fast batched counter intentionally returns upper bounds, not exact gains.
            return scorer.score_gains_batch(
                [rules[index] for index in rule_indices],
                progress_callback=progress_callback or scan_progress,
            )

        selected, covered = _celf_select_from_upper_bounds(
            rules, upper_bounds, target_count, score_one,
            budget=budget, cost_alpha=cost_alpha,
            on_select=scorer.mark_selected,
            score_batch_fn=score_many,
            rescore_batch_size=recompute_batch_size,
            on_scan_progress=scan_progress,
        )
        log_info(
            f"[CELF] Selected {bold(str(len(selected)))} rules; exact recovery "
            f"{bold(cyan(f'{int(covered.sum()):,}/{target_count:,}'))}"
        )
        return selected, covered
    finally:
        if scorer is not None:
            try:
                scorer.gpu_engine.celf_report_profile()
            except Exception as exc:
                log_warn(f"[CELF] Could not report GPU profile: {exc}")
            try:
                scorer.close()
            except Exception as exc:
                log_warn(f"[CELF] Could not release GPU target table: {exc}")
        _restore_depth(gpu_engine, restore_depth)
