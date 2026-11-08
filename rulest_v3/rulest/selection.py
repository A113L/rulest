"""GPU target-table-backed memory-light CELF selection.

Fast batched GPU scans produce admissible upper bounds (one count per matching
base-word output, so duplicate outputs may overcount but never undercount
unique active targets). Singleton scans use a per-candidate bitmap and return
exact distinct target IDs. CELF only selects a candidate after exact scoring.
"""
from __future__ import annotations

import heapq
import math
from contextlib import nullcontext
import os
from collections import deque
import sys
import time

import numpy as np
# Keep the custom heartbeat-driven bar (not plain tqdm): CELF rounds can go
# many seconds between .update() calls (one exact GPU word-sweep per round),
# and this bar keeps elapsed/ETA live via a background refresh thread instead
# of looking stalled. Styling below is normalized to match the other phases'
# real-tqdm bars (green desc, cyan postfix, same bar glyphs/format).
from .progress import tqdm, ActivityHeartbeat  # CELF_PROGRESS_DIAGNOSTICS_PATCH_V2
from .profiling import stage_profiler

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


def _celf_log_interval():
    """Return a bounded heartbeat interval for server and terminal logging."""
    try:
        value = float(os.environ.get("RULEST_CELF_LOG_INTERVAL", "0.5"))
    except (TypeError, ValueError):
        value = 0.5
    return min(10.0, max(0.25, value))


def _rolling_rules_rate(samples, count, now, window=5.0):
    """Smoothed recent candidate/rule throughput, avoiding per-batch spikes."""
    count = int(count)
    if not samples or count > samples[-1][1]:
        samples.append((float(now), count))
    cutoff = float(now) - max(1.0, float(window))
    while len(samples) > 2 and samples[1][0] <= cutoff:
        samples.popleft()
    first_time, first_count = samples[0]
    elapsed = float(now) - first_time
    if elapsed <= 1e-9:
        return 0.0
    return max(0.0, (count - first_count) / elapsed)


def _format_progress_bar(done, total, width=24):
    """Render an exact fixed-width progress bar for newline-based web logs."""
    total = max(1, int(total))
    done = min(total, max(0, int(done)))
    fraction = done / total
    filled = min(width, int(fraction * width))
    return f"[{('█' * filled)}{('·' * (width - filled))}] {int(round(fraction * 100.0)):3d}%"


def _run_initial_upper_bound_pass(scorer, rules, rule_batch_size):
    """Score every candidate once and stream an exact progress/rate heartbeat.

    Terminal sessions get the normal tqdm bar. In the web server, stderr is a
    pipe, so a newline-based ActivityHeartbeat reports completed candidate
    batches, the exact scanned/total count, percentage, speed, average, recent
    throughput, batch number and ETA to the Live Activity log.
    """
    total = len(rules)
    upper_bounds = [0] * total
    if total == 0:
        return upper_bounds

    batch_size = max(1, int(rule_batch_size))
    is_tty = bool(sys.stderr.isatty())
    started = time.monotonic()
    samples = deque([(started, 0)])
    batch_total = (total + batch_size - 1) // batch_size
    heartbeat_context = (
        nullcontext(None) if is_tty else ActivityHeartbeat(
            "[CELF] Upper-bound pass | initializing",
            interval=_celf_log_interval(),
            line_mode=True,
        )
    )

    with heartbeat_context as heartbeat, tqdm(
        total=total,
        desc=green("[CELF] Upper-bound pass (GPU target-table scan)"),
        unit="rules",
        disable=not is_tty,
        mininterval=0.5,
    ) as progress:
        for start in range(0, total, batch_size):
            chunk = rules[start:start + batch_size]
            batch_number = start // batch_size + 1
            batch_started = time.monotonic()
            index_scorer = getattr(scorer, "upper_bound_index_batch", None)
            bounds = (
                index_scorer(range(start, start + len(chunk)))
                if index_scorer is not None else scorer.upper_bound_batch(chunk)
            )
            if len(bounds) != len(chunk):
                raise RuntimeError(
                    "CELF upper-bound scorer returned the wrong number of candidates "
                    f"for batch {batch_number}/{batch_total}"
                )
            upper_bounds[start:start + len(chunk)] = [max(0, int(v)) for v in bounds]
            batch_elapsed = max(time.monotonic() - batch_started, 1e-9)
            processed = start + len(chunk)
            now = time.monotonic()
            batch_rate = len(chunk) / batch_elapsed
            avg_rate = processed / max(now - started, 1e-9)
            recent_rate = _rolling_rules_rate(samples, processed, now, window=5.0)
            eta = (total - processed) / avg_rate if avg_rate > 0 else float("inf")
            eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "calculating"
            bar = _format_progress_bar(processed, total)
            status = (
                f"[CELF] Upper-bound pass | {bar} | scanned={processed:,}/{total:,} "
                f"| speed={batch_rate:,.2f} rules/s | avg={avg_rate:,.2f} rules/s "
                f"| recent={recent_rate:,.2f} rules/s | batch={batch_number:,}/{batch_total:,} "
                f"({len(chunk):,} rules) | ETA={eta_text} | elapsed={now - started:.1f}s"
            )
            if heartbeat is not None:
                heartbeat.set_desc(status)
            else:
                progress.update(len(chunk))
                progress.set_postfix(
                    {
                        "speed": f"{batch_rate:,.0f} rules/s",
                        "avg": f"{avg_rate:,.0f} rules/s",
                        "scanned": f"{processed:,}/{total:,}",
                    },
                    refresh=False,
                )

        finished = time.monotonic()
        avg_rate = total / max(finished - started, 1e-9)
        final_status = (
            f"[CELF] Upper-bound pass complete | {_format_progress_bar(total, total)} "
            f"| scanned={total:,}/{total:,} | avg={avg_rate:,.2f} rules/s "
            f"| elapsed={finished - started:.1f}s"
        )
        if heartbeat is not None:
            heartbeat.set_desc(final_status, force=True)
        log_info(final_status)

    return upper_bounds


_FNV1A64_OFFSET = 14695981039346656037
_FNV1A64_PRIME = 1099511628211
_UINT64_MASK = (1 << 64) - 1


class _CelfExactBatchMismatch(RuntimeError):
    """Signals that GPU batched exact gains failed winner-scan verification.

    The partially completed greedy run must be discarded and restarted from an
    empty active-target bitmap with batched exact rescoring disabled. Continuing
    with the suspect gains could violate CELF's admissible-bound invariant.
    """



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
    on_scan_progress=None, score_exact_batch_fn=None,
    materialize_exact_ids_fn=None,
):
    """Heap CELF with admissible bounds and optional batched exact rescoring.

    Fast batch scoring counts matching base-word outputs, so duplicates may inflate
    a bound but cannot undercount unique active targets. If `score_exact_batch_fn`
    is supplied, contenders whose tightened bound can still win are exact-rescored
    together using a per-candidate bitmap. When available, the actual winner's
    IDs are materialized from that already-computed bitmap instead of launching
    a second full base-word scan solely to record hits. Legacy/CPU callers keep
    the original verified singleton-score fallback.
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
    try:
        exact_microbatch_target = int(os.environ.get("RULEST_CELF_EXACT_MICROBATCH", "4"))
    except (TypeError, ValueError):
        exact_microbatch_target = 4
    exact_microbatch_target = max(1, min(batch_size, exact_microbatch_target))
    if score_exact_batch_fn is not None:
        log_info(
            f"[CELF] Exact challenger microbatch target={exact_microbatch_target} "
            f"(workspace ceiling={batch_size}; set RULEST_CELF_EXACT_MICROBATCH=1 to disable coalescing)"
        )
    selected = []
    exact_rescored = 0
    exact_gain_only = 0
    exact_batch_candidates = 0
    exact_batches = 0
    exact_scans = 0
    exact_record_scans = 0
    batch_rescored = 0
    upper_bound_candidates = 0
    upper_bound_batches = 0
    recovered = 0
    current_batch_size = 1
    progress_enabled = bool(sys.stderr.isatty())
    last_refresh = [0.0]
    t0 = time.monotonic()
    selected_rate_samples = deque([(t0, 0)])
    upper_rate_samples = deque([(t0, 0)])
    last_selected_rate = [0.0]
    last_upper_rate = [0.0]

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
    try:
        heartbeat_interval = float(os.environ.get("RULEST_CELF_LOG_INTERVAL", "3"))
    except (TypeError, ValueError):
        heartbeat_interval = 3.0
    heartbeat_interval = min(30.0, max(1.0, heartbeat_interval))
    heartbeat_context = (
        ActivityHeartbeat(
            "[CELF] Greedy selection | initializing",
            interval=heartbeat_interval,
            line_mode=True,
        )
        if not progress_enabled else nullcontext(None)
    )
    with heartbeat_context as heartbeat:
        with tqdm(
            total=budget if budget else approx_total_without_budget,
            desc=green(celf_bar_desc), unit="rule",
            disable=not progress_enabled, mininterval=0.5,
        ) as progress:
            def _status_line():
                now = time.monotonic()
                elapsed = max(now - t0, 1e-9)
                selected_pct = 100.0 * len(selected) / max(1, limit)
                coverage_pct = 100.0 * recovered / max(1, target_count)
                last_selected_rate[0] = _rolling_rules_rate(
                    selected_rate_samples, len(selected), now, window=5.0
                )
                last_upper_rate[0] = _rolling_rules_rate(
                    upper_rate_samples, upper_bound_candidates, now, window=5.0
                )
                average_selected_rate = len(selected) / elapsed
                average_upper_rate = upper_bound_candidates / elapsed
                return (
                    f"[CELF] Greedy selection | selected={len(selected):,}/{limit:,} "
                    f"({selected_pct:.1f}%) | speed={last_selected_rate[0]:,.2f} rules/s "
                    f"(avg={average_selected_rate:,.2f} rules/s) | recovered={recovered:,}/{target_count:,} "
                    f"({coverage_pct:.1f}%) | heap={len(heap):,} | exact_scans={exact_scans:,} "
                    f"| exact_batches={exact_batches:,} | exact_record_scans={exact_record_scans:,} "
                    f"| exact_gain_candidates={exact_gain_only + exact_batch_candidates:,} "
                    f"| lazy_upper_bound_candidates={upper_bound_candidates:,} "
                    f"| lazy_upper_bound_batches={upper_bound_batches:,} "
                    f"| lazy_upper_bound_speed={last_upper_rate[0]:,.2f} rules/s "
                    f"(avg={average_upper_rate:,.2f} rules/s) | batch={current_batch_size} "
                    f"| elapsed={elapsed:.1f}s"
                )

            def report_scan(done=None, total=None):
                if progress_enabled:
                    # Order matters: progress.py now drops whole postfix fields
                    # from the right when the line doesn't fit the terminal, so
                    # keep the useful coverage/selection counters first.
                    postfix = {
                        "selected": cyan(str(len(selected))),
                        "recovered": cyan(f"{recovered:,}"),
                        "speed": cyan(f"{last_selected_rate[0]:,.1f} rules/s"),
                        "upper": cyan(f"{upper_bound_candidates:,}"),
                        "exact": cyan(f"{exact_scans:,}"),
                        "batch": cyan(str(current_batch_size)),
                    }
                    if done is not None and total is not None:
                        postfix["scan words"] = cyan(f"{int(done):,}/{int(total):,}")
                    progress.set_postfix(postfix, refresh=False)
                    now = time.monotonic()
                    if now - last_refresh[0] >= 0.5:
                        progress.refresh()
                        last_refresh[0] = now
                elif heartbeat is not None:
                    # In the web server, stderr is a pipe rather than a TTY,
                    # which intentionally disables terminal progress bars.
                    # Keep a clean, timestamped heartbeat flowing to stdout so
                    # the GUI can show real CELF activity while GPU scoring runs.
                    heartbeat.set_desc(_status_line())

            def exact_score(index):
                # Compatibility fallback: a full exact scan WITH hit-slot
                # recording (one GPU wordlist pass plus synchronous target-ID
                # readback). The normal GPU path avoids calling this for losing
                # candidates by using exact batches and winner bitmaps.
                nonlocal exact_rescored, exact_scans, exact_record_scans
                exact_rescored += 1
                exact_scans += 1
                exact_record_scans += 1
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

            def exact_gain(index):
                """Exact marginal gain for ONE candidate, preferring the cheap
                batched-exact kernel (score_mode=1: deduplicates target IDs but
                never transfers/records them) over the expensive recording scan.

                Returns (gain, ids_or_None). In the GPU exact-batch path, IDs
                remain None until the candidate wins; the winner's bitmap row is
                then read back without repeating the full base-word scan. Callers
                without bitmap support retain the verified recording-scan fallback.

                Seed-pop and compatibility-mode fresh-contender checks use this
                path only when no batched exact scorer exists. In the normal GPU
                path, exact gains are batched and winner IDs are materialized from
                the exact bitmap, so losing candidates do not require hit recording.
                """
                nonlocal exact_gain_only, exact_scans, exact_batches
                if score_exact_batch_fn is None:
                    return exact_score(index)
                exact_gain_only += 1
                exact_scans += 1
                exact_batches += 1
                if on_scan_progress is None:
                    gains = score_exact_batch_fn([index], covered)
                else:
                    gains = score_exact_batch_fn([index], covered, report_scan)
                if len(gains) != 1:
                    raise RuntimeError("CELF exact batch scorer returned the wrong number of gains")
                report_scan()
                return max(0, int(gains[0])), None

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
                # Start each round with the strongest outstanding bound(s), but
                # never assume an upper bound is the exact gain.
                #
                # `covered` does not change until a winner is actually recorded
                # at the bottom of this round, so the exact gain of every
                # candidate popped here is independent of the others popped in
                # the same group -- there is no reason to score them one at a
                # time. Popping up to `batch_size` of them and exact-scoring the
                # whole group in a single GPU call (the same score_exact_batch_fn
                # already used below for stale-bound rescoring) collapses what
                # used to be one GPU round-trip per seed into one every
                # `batch_size` seeds. Keeping this initial exact check batched
                # avoids one launch per seed; later small challenger groups are
                # coalesced below before exact rescoring.
                seed_entries = []
                while heap and len(seed_entries) < batch_size:
                    _neg_p, index, bound = heapq.heappop(heap)
                    seed_entries.append((int(index), int(bound)))

                # Picking the best of a group must use the same cost_alpha-aware
                # priority() ordering as every other comparison in this function
                # (can_beat()), NOT a plain gain comparison -- with cost_alpha>0
                # a lower-gain, cheaper-depth candidate can legitimately outrank
                # a higher-gain, more-expensive one. Comparing raw gains here
                # would silently pick the wrong winner whenever cost_alpha>0.
                best_index = best_gain = None
                best_ids = None
                def _take_if_better(index, gain, ids):
                    nonlocal best_index, best_gain, best_ids
                    if best_index is None or can_beat(gain, index, best_gain, best_index):
                        if best_index is not None:
                            push(best_index, best_gain)
                        best_index, best_gain, best_ids = index, gain, ids
                    else:
                        push(index, gain)

                def materialize_best_ids():
                    """Resolve IDs for the current exact winner without rescoring it.

                    The GPU exact-batch kernel already populated one compact
                    target-ID bitmap per candidate. Reading the winning row is
                    cheaper than launching another full base-word scan in
                    record_hits mode. Callers without that capability retain
                    the old verified recording-scan fallback.
                    """
                    nonlocal best_ids
                    if best_index is None or best_gain is None or best_gain <= 0 or best_ids is not None:
                        return
                    if materialize_exact_ids_fn is not None and score_exact_batch_fn is not None:
                        raw_ids = materialize_exact_ids_fn(best_index)
                        ids = {
                            int(target_id) for target_id in raw_ids
                            if 0 <= int(target_id) < target_count and not covered[int(target_id)]
                        }
                        if len(ids) != int(best_gain):
                            raise _CelfExactBatchMismatch(
                                "CELF exact-batch winner bitmap mismatch for "
                                f"candidate {best_index}: batch={best_gain}, bitmap={len(ids)}"
                            )
                        best_ids = ids
                        return
                    recorded_gain, best_ids = exact_score(best_index)
                    if recorded_gain != best_gain:
                        raise _CelfExactBatchMismatch(
                            "CELF exact-batch/recording score mismatch for "
                            f"candidate {best_index}: batch={best_gain}, recorded={recorded_gain}"
                        )
                if score_exact_batch_fn is None or len(seed_entries) == 1:
                    # CPU/test fallback (no batched-exact kernel) or nothing to
                    # batch this round: original one-call-per-candidate path.
                    for index, _bound in seed_entries:
                        gain, ids = exact_gain(index)
                        _take_if_better(index, gain, ids)
                else:
                    seed_indices = [index for index, _bound in seed_entries]
                    if on_scan_progress is None:
                        gains = score_exact_batch_fn(seed_indices, covered)
                    else:
                        gains = score_exact_batch_fn(seed_indices, covered, report_scan)
                    if len(gains) != len(seed_entries):
                        raise RuntimeError("CELF exact batch scorer returned the wrong number of gains")
                    batch_rescored += len(seed_entries)
                    exact_batch_candidates += len(seed_entries)
                    exact_scans += len(seed_entries)
                    exact_batches += 1
                    current_batch_size = len(seed_entries)
                    report_scan()
                    for (index, _bound), raw_gain in zip(seed_entries, gains):
                        # best_ids stays None here (gain-only, no recording scan
                        # yet) -- materialized below once the round's actual
                        # winner is final, exactly as the stale-bound
                        # batched-exact path already does.
                        _take_if_better(index, max(0, int(raw_gain)), None)

                # Capture this batch's winner before another GPU call resets
                # the reusable per-candidate `seen` bitmap workspace.
                materialize_best_ids()

                if best_gain <= 0:
                    # Marginal gain can only shrink; zero-gain candidates are dead.
                    continue
                fresh_this_round = set()
                pending_exact_entries = []

                def score_pending_exact_entries(force=False):
                    """Score already-tightened challengers together when the group is small.

                    A raw upper-bound batch can often leave only one or two exact
                    challengers. Scoring each such remainder immediately creates
                    avoidable one-candidate GPU launches. Keep those contenders in
                    a per-round queue until a small microbatch is available (or no
                    more admissible bounds remain). The queue is capped by the same
                    batch size as the resident GPU bitmap workspace.
                    """
                    nonlocal best_index, best_gain, best_ids, exact_gain_only, exact_batch_candidates
                    nonlocal exact_scans, exact_batches, current_batch_size
                    if not pending_exact_entries:
                        return False
                    if not force and len(pending_exact_entries) < exact_microbatch_target:
                        return False

                    take = min(batch_size, len(pending_exact_entries))
                    exact_entries = pending_exact_entries[:take]
                    del pending_exact_entries[:take]
                    exact_indices = [index for index, _bound in exact_entries]
                    if on_scan_progress is None:
                        exact_gains = score_exact_batch_fn(exact_indices, covered)
                    else:
                        exact_gains = score_exact_batch_fn(exact_indices, covered, report_scan)
                    if len(exact_gains) != len(exact_entries):
                        raise RuntimeError("CELF exact batch scorer returned the wrong number of gains")

                    exact_batch_candidates += len(exact_entries)
                    exact_scans += len(exact_entries)
                    exact_batches += 1
                    current_batch_size = len(exact_entries)
                    for (index, _bound), raw_exact_gain in zip(exact_entries, exact_gains):
                        tightened_exact_gain = max(0, int(raw_exact_gain))
                        if tightened_exact_gain and can_beat(tightened_exact_gain, index, best_gain, best_index):
                            # The exact score is a valid bound for every later round.
                            push(best_index, best_gain)
                            best_index, best_gain, best_ids = index, tightened_exact_gain, None
                        else:
                            push(index, tightened_exact_gain)

                    # A winner may have displaced the previous best. Fetch its IDs
                    # now, before another exact batch overwrites the reusable bitmap.
                    materialize_best_ids()

                    # Exact scoring may have raised the best priority. Candidates
                    # still waiting in the queue whose tightened upper bound can no
                    # longer win are returned to the heap without another GPU scan.
                    still_pending = []
                    for index, bound in pending_exact_entries:
                        if can_beat(bound, index, best_gain, best_index):
                            still_pending.append((index, bound))
                        else:
                            push(index, bound)
                    pending_exact_entries[:] = still_pending
                    report_scan()
                    return True

                while heap or pending_exact_entries:
                    # Flush a full/sufficient microbatch before scanning more bounds.
                    if score_pending_exact_entries(force=False):
                        continue

                    if not heap:
                        if score_pending_exact_entries(force=True):
                            continue
                        break

                    _neg_top, top_index, top_bound = heap[0]
                    top_index = int(top_index)
                    if not can_beat(top_bound, top_index, best_gain, best_index):
                        # No remaining heap bound can win. Any small pending group
                        # is the final set of plausible challengers for this round.
                        if score_pending_exact_entries(force=True):
                            continue
                        break

                    # Legacy non-batched exact mode keeps its original singleton
                    # recording path. The default GPU exact-batch mode never adds
                    # entries to fresh_this_round.
                    if top_index in fresh_this_round:
                        heapq.heappop(heap)
                        fresh_this_round.remove(top_index)
                        gain, ids = exact_gain(top_index)
                        if gain > best_gain or (gain == best_gain and top_index < best_index):
                            push(best_index, best_gain)
                            best_index, best_gain, best_ids = top_index, gain, ids
                        else:
                            push(top_index, gain)
                        continue

                    # Refresh stale bounds that might still beat the exact best.
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
                        continue

                    batch_indices = [index for index, _ in batch_entries]
                    current_batch_size = len(batch_indices)
                    batch_rescored += len(batch_indices)
                    if score_batch_fn is None:
                        # CPU-only fallback has exact scores; preserve its verified IDs.
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
                        upper_bound_candidates += len(batch_entries)
                        upper_bound_batches += 1
                        upper_rate_samples.append((time.monotonic(), upper_bound_candidates))
                        now = time.monotonic()
                        while len(upper_rate_samples) > 2 and upper_rate_samples[1][0] <= now - 5.0:
                            upper_rate_samples.popleft()

                        if score_exact_batch_fn is None:
                            # Compatibility path for callers that only provide bounds.
                            for (index, old_bound), raw_bound in zip(batch_entries, raw_bounds):
                                tightened = min(old_bound, max(0, int(raw_bound)))
                                push(index, tightened)
                                if tightened and can_beat(tightened, index, best_gain, best_index):
                                    fresh_this_round.add(index)
                        else:
                            # Collect small groups of exact challengers from adjacent
                            # upper-bound batches. This preserves every admissible
                            # bound while reducing one-candidate GPU launches.
                            for (index, old_bound), raw_bound in zip(batch_entries, raw_bounds):
                                tightened = min(old_bound, max(0, int(raw_bound)))
                                if tightened and can_beat(tightened, index, best_gain, best_index):
                                    pending_exact_entries.append((index, tightened))
                                else:
                                    push(index, tightened)
                            score_pending_exact_entries(force=False)
                    report_scan()

                # Exact-batch winners are recovered from their already-computed
                # bitmap. Legacy callers still use the verified recording scan.
                if best_ids is None:
                    materialize_best_ids()

                # Clear by the winner's exact target IDs, never by a transient buffer.
                covered[np.fromiter(best_ids, dtype=np.int64)] = True
                recovered += len(best_ids)
                if on_select is not None:
                    on_select(best_ids)
                selected.append((rules[best_index], len(best_ids)))
                selection_now = time.monotonic()
                selected_rate_samples.append((selection_now, len(selected)))
                while len(selected_rate_samples) > 2 and selected_rate_samples[1][0] <= selection_now - 5.0:
                    selected_rate_samples.popleft()
                progress.update(1)
                report_scan()

    if sum(gain for _, gain in selected) != int(covered.sum()):
        raise RuntimeError("CELF gain accounting disagrees with covered bitmap")
    return selected, covered

def _resolve_single_engine(gpu_engine):
    """CELF selection runs on exactly ONE GPU -- unlike Stage 1/2/Seed
    extraction, it never fans a round's work out across several engines
    (there is only one active hash table / target bitmap, and CELF's
    round-to-round state has to live in one place). A MultiGPUEngine from
    `--device all`/an explicit multi-device list is accepted transparently
    by using its first surviving engine, rather than requiring the caller
    to know to unwrap it -- and if every device became unresponsive during
    an earlier stage (so MultiGPUEngine.engines is empty), this raises a
    clear, actionable RuntimeError here instead of letting a MultiGPUEngine
    fall through to .set_base_words()/.celf_*() calls it never implements,
    which previously surfaced as a confusing
    `AttributeError: 'MultiGPUEngine' object has no attribute 'set_base_words'`
    three frames below the real cause.
    """
    engines = getattr(gpu_engine, 'engines', None)
    if engines is None:
        return gpu_engine  # already a single GPUEngine/ProcessEngineProxy
    if not engines:
        raise RuntimeError(
            "[CELF]  No GPU device is available for selection -- every GPU "
            "became unresponsive during an earlier stage. Re-run with a "
            "single, healthy --device."
        )
    if len(engines) > 1:
        primary = engines[0]
        name = primary.device.name.strip() if getattr(primary, 'device', None) else "device 0"
        log_warn(
            f"[CELF]  Selection runs on a single GPU ({name}); the other "
            f"{len(engines) - 1} device(s) stay idle for this stage."
        )
    return engines[0]


class _GpuExactScorer:
    """Thin adapter around the GPU-resident target table and exact-match kernels."""

    def __init__(self, gpu_engine, base_words, target_words, pair_batch_size, candidate_rules):
        self.gpu_engine = _resolve_single_engine(gpu_engine)
        self.base_words = base_words
        self.targets = target_words if isinstance(target_words, list) else list(target_words)
        self.candidate_rules = list(candidate_rules)
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

    def upper_bound_index_batch(self, candidate_indices):
        """Score a candidate batch by pool index (compact worker RPC payload)."""
        indices = [int(index) for index in candidate_indices]
        if not indices:
            return []
        method = getattr(self.gpu_engine, "celf_score_candidate_indices_gpu", None)
        if method is None:
            return self.upper_bound_batch([self.candidate_rules[index] for index in indices])
        gains, _slots = method(indices, record_hits=False)
        return [int(gain) for gain in gains]

    def score_gains_batch(self, rules, progress_callback=None):
        """Return fast admissible upper bounds; duplicates may overcount by design."""
        if not rules:
            return []
        gains, _slots = self.gpu_engine.celf_score_chains_gpu(
            rules, record_hits=False, progress_callback=progress_callback
        )
        return [int(gain) for gain in gains]

    def score_gains_index_batch(self, candidate_indices, progress_callback=None):
        indices = [int(index) for index in candidate_indices]
        if not indices:
            return []
        method = getattr(self.gpu_engine, "celf_score_candidate_indices_gpu", None)
        if method is None:
            return self.score_gains_batch(
                [self.candidate_rules[index] for index in indices], progress_callback
            )
        gains, _slots = method(indices, record_hits=False)
        return [int(gain) for gain in gains]

    def score_exact_gains_batch(self, rules, progress_callback=None):
        """Return exact distinct-target marginal gains for a small candidate batch.

        Unlike the raw-hit upper-bound scan, this mode uses one compact seen
        bitmap per candidate but does not transfer/record target slots. It is
        reserved for candidates whose upper bounds still threaten the winner.
        """
        if not rules:
            return []
        gains, _slots = self.gpu_engine.celf_score_chains_gpu(
            rules, record_hits=False, exact_counts=True,
            progress_callback=progress_callback,
        )
        return [int(gain) for gain in gains]

    def score_exact_gains_index_batch(self, candidate_indices, progress_callback=None):
        indices = [int(index) for index in candidate_indices]
        if not indices:
            return []
        method = getattr(self.gpu_engine, "celf_score_candidate_indices_gpu", None)
        if method is None:
            return self.score_exact_gains_batch(
                [self.candidate_rules[index] for index in indices], progress_callback
            )
        gains, _slots = method(indices, record_hits=False, exact_counts=True)
        return [int(gain) for gain in gains]

    def materialize_exact_targets(self, candidate_index):
        """Fetch the exact target IDs for a candidate from the last exact batch."""
        method = getattr(self.gpu_engine, "celf_seen_target_ids_for_candidate", None)
        if method is None:
            raise RuntimeError("GPU engine does not support CELF winner bitmap materialization")
        return method(int(candidate_index))

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

    # CELF maintains one resident target table/bitmap, so select one engine
    # once and use it consistently for depth changes, scoring, and cleanup.
    # In multi-GPU mode this is a ProcessEngineProxy; it explicitly exposes
    # the CELF RPC surface rather than pretending to be a local GPUEngine.
    selection_engine = _resolve_single_engine(gpu_engine)
    restore_depth, effective_depth = _widen_depth(selection_engine, rules, max_depth)
    rules = [rule for rule in rules if len(rule.split()) <= effective_depth]
    if not rules:
        _restore_depth(selection_engine, restore_depth)
        return [], empty_mask

    scorer = None
    try:
        scorer = _GpuExactScorer(
            selection_engine, base_words, unique_targets, pair_batch_size, rules
        )
        celf_engine = scorer.gpu_engine
        seen_words = int(celf_engine._celf_seen_words)
        scratch_limit = max(4, int(celf_engine.params.get("CELF_SEEN_SCRATCH_MB", 32))) * 1024 * 1024
        max_by_scratch = max(1, scratch_limit // max(4, seen_words * 4))
        configured_batch = max(1, int(celf_engine.params.get("CELF_RULE_BATCH_SIZE", 32)))
        rule_batch_size = min(
            configured_batch, max_by_scratch,
            max(1, int(getattr(celf_engine, "_celf_batch_capacity", configured_batch))),
        )

        # Keep the existing VRAM-sized rule batch, but report exact throughput
        # and ETA in both terminals and web-server log streams.
        upper_bounds = _run_initial_upper_bound_pass(scorer, rules, rule_batch_size)

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
        exact_batch_enabled = os.environ.get("RULEST_CELF_EXACT_BATCH", "1").strip().lower() not in {"0", "false", "no", "off"}
        try:
            exact_microbatch = max(1, min(
                recompute_batch_size,
                int(os.environ.get("RULEST_CELF_EXACT_MICROBATCH", "4")),
            ))
        except (TypeError, ValueError):
            exact_microbatch = min(recompute_batch_size, 4)
        bitmap_enabled = bool(
            exact_batch_enabled and callable(getattr(scorer, "materialize_exact_targets", None))
        )
        log_info(
            f"[CELF] Candidates: {bold(str(len(rules)))}; targets: {bold(str(target_count))}; "
            f"upper-bound GPU batch={rule_batch_size}; exact seed/challenger workspace={recompute_batch_size}; "
            f"challenger coalescing threshold={exact_microbatch}; "
            f"batched exact scoring={'enabled' if exact_batch_enabled else 'disabled'}; "
            f"winner bitmap readback={'enabled' if bitmap_enabled else 'fallback recording scan'}; "
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
            index_scorer = getattr(scorer, "score_gains_index_batch", None)
            if index_scorer is not None:
                return index_scorer(rule_indices, progress_callback=progress_callback or scan_progress)
            return scorer.score_gains_batch(
                [rules[index] for index in rule_indices], progress_callback=progress_callback or scan_progress,
            )

        def score_exact_many(rule_indices, covered_mask, progress_callback=None):
            index_scorer = getattr(scorer, "score_exact_gains_index_batch", None)
            if index_scorer is not None:
                return index_scorer(rule_indices, progress_callback=progress_callback or scan_progress)
            return scorer.score_exact_gains_batch(
                [rules[index] for index in rule_indices],
                progress_callback=progress_callback or scan_progress,
            )

        try:
            selected, covered = _celf_select_from_upper_bounds(
                rules, upper_bounds, target_count, score_one,
                budget=budget, cost_alpha=cost_alpha,
                on_select=scorer.mark_selected,
                score_batch_fn=score_many,
                rescore_batch_size=recompute_batch_size,
                on_scan_progress=scan_progress,
                score_exact_batch_fn=score_exact_many if exact_batch_enabled else None,
                materialize_exact_ids_fn=(
                    getattr(scorer, "materialize_exact_targets", None)
                    if exact_batch_enabled else None
                ),
            )
        except _CelfExactBatchMismatch as exc:
            if not exact_batch_enabled:
                raise
            # Never continue with possibly undercounted/overcounted speculative
            # gains: that could silently alter the greedy ordering. Reset the
            # GPU active bitmap to the original universe and rerun the *entire*
            # selection with the older, verified singleton exact-rescore path.
            log_warn(
                f"[CELF] {exc}. Discarding this partial selection, resetting GPU "
                "coverage, and retrying with batched exact rescoring disabled."
            )
            try:
                celf_engine.celf_reset_active_targets()
            except Exception as reset_exc:
                raise RuntimeError(
                    "CELF batched exact scoring failed verification and the GPU "
                    "active-target bitmap could not be reset safely; refusing to "
                    f"continue with potentially incorrect coverage ({reset_exc}). "
                    "Retry with RULEST_CELF_EXACT_BATCH=0."
                ) from reset_exc
            selected, covered = _celf_select_from_upper_bounds(
                rules, upper_bounds, target_count, score_one,
                budget=budget, cost_alpha=cost_alpha,
                on_select=scorer.mark_selected,
                score_batch_fn=score_many,
                rescore_batch_size=recompute_batch_size,
                on_scan_progress=scan_progress,
                score_exact_batch_fn=None,
                materialize_exact_ids_fn=None,
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
            # Fold the same GPU-event breakdown into the unified per-stage
            # profiling table's "Selection (CELF greedy)" row (see cli.py,
            # which wraps this whole function in stage_profiler.track(...))
            # instead of only ever appearing in its own separate block above.
            # Gated on `detailed` since building/formatting this string costs
            # a little extra and is only useful when profiling was asked for.
            if stage_profiler.detailed:
                try:
                    summary = scorer.gpu_engine.celf_profile_summary()
                    if summary:
                        stage_profiler.note("Selection (CELF greedy)", extra=summary)
                except Exception as exc:
                    log_warn(f"[CELF] Could not fold GPU profile into stage table: {exc}")
            try:
                scorer.close()
            except Exception as exc:
                log_warn(f"[CELF] Could not release GPU target table: {exc}")
        _restore_depth(selection_engine, restore_depth)
