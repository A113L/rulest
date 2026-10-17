"""Unified wall-clock stage profiling for the rulest pipeline. RULEST_STAGE_PROFILE_V1

Every phase of the pipeline — STAGE 0 through STAGE 3, bloom-filter build,
minimization, and rule selection/CELF — gets wrapped in
``with stage_profiler.track("STAGE 0 — Token-Strip"):``. That records the
phase's wall-clock duration (accumulated across re-entries, since e.g.
STAGE 2 can be invoked more than once in multi-GPU retry paths) and, when
the caller supplies a word/chain/candidate count via ``set_count()``, a
throughput figure too.

At the end of a run, ``stage_profiler.report()`` prints one aligned table:
every tracked stage, its share of total wall time, and throughput where
known, sorted slowest-first so the dominant cost is the first thing you
see. This is the basis for tuning decisions (which stage actually eats the
runtime for a given workload) and is deliberately always-on and effectively
free — a handful of ``time.perf_counter()`` calls — unlike the heavier
per-call GPU-event accounting in gpu_engine.py's CELF profiler (gated
behind RULEST_CELF_PROFILE because it costs a bit more per call).

Profiling is opt-in, controlled by a single switch: the ``--profile`` CLI
flag (see cli.py), or the RULEST_PROFILE / RULEST_CELF_PROFILE env vars for
callers that don't go through argparse. Any one of those turns on both the
stage table and the finer CELF GPU-event accounting together -- see
`StageProfiler.configure()`.
"""
from __future__ import annotations
import os
import time
from collections import OrderedDict

from .common import log_info, bold, cyan, green, dim
from .progress import _duration


class _StageRecord:
    __slots__ = ("name", "seconds", "count", "unit", "extra", "calls")

    def __init__(self, name):
        self.name = name
        self.seconds = 0.0
        self.count = 0
        self.unit = ""
        self.extra = None
        self.calls = 0


class _StageTimer:
    """Context manager returned by `StageProfiler.track()`; accumulates one
    entry's elapsed time (and optional throughput count) into its stage's
    running total on exit, even if the tracked block raises."""

    def __init__(self, profiler, name, count, unit):
        self.profiler = profiler
        self.name = name
        self.count = count
        self.unit = unit
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def set_count(self, count, unit=None):
        """Supply (or correct) the throughput count after the tracked work
        has actually run — most stages don't know how many words/chains/
        candidates they processed until the wrapped call returns, so this
        is usually called right before falling out of the `with` block
        rather than passed to `track()` up front."""
        self.count = count
        if unit is not None:
            self.unit = unit

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is None:
            return False
        elapsed = time.perf_counter() - self._t0
        rec = self.profiler._record(self.name)
        rec.seconds += elapsed
        rec.calls += 1
        if self.count:
            rec.count += self.count
            if self.unit:
                rec.unit = self.unit
        return False


class StageProfiler:
    """Accumulates wall-clock time per named stage across a whole run."""

    def __init__(self):
        self._records = OrderedDict()
        self._run_t0 = time.perf_counter()
        # Opt-in by default: a bare `import rulest` (tests, library use, the
        # GUI wrapper) should never start printing profiling tables. cli.py
        # calls configure() right after parsing args to turn this on from
        # --profile or the env-var fallbacks; a standalone script that sets
        # RULEST_PROFILE/RULEST_CELF_PROFILE before import gets it for free
        # here too, without needing to call configure() itself.
        env_on = os.environ.get("RULEST_PROFILE") == "1" or os.environ.get("RULEST_CELF_PROFILE") == "1"
        self.enabled = env_on
        self.detailed = env_on

    def configure(self, enabled):
        """Single switch: turns the per-stage table on/off. Also flips
        `detailed`, which callers (selection.py) check before paying for
        the extra GPU-event CELF breakdown string -- so one flag controls
        both the cheap wall-clock table and the pricier per-call GPU
        accounting together, instead of two separately-remembered knobs."""
        self.enabled = bool(enabled)
        self.detailed = bool(enabled)

    def _record(self, name):
        rec = self._records.get(name)
        if rec is None:
            rec = _StageRecord(name)
            self._records[name] = rec
        return rec

    def begin_run(self):
        """Reset the run clock (used for the Untracked/overhead row and the
        Total row) without discarding any stage totals already recorded —
        call this once, right when the real pipeline starts, so import-time
        delay before `main()` runs never gets counted as overhead."""
        self._run_t0 = time.perf_counter()

    def reset(self):
        """Clear all recorded stages and restart the run clock. Mainly for
        tests that construct more than one pipeline run in-process."""
        self._records.clear()
        self._run_t0 = time.perf_counter()

    def track(self, name, *, count=None, unit=""):
        """Returns a context manager; time spent inside is added to the
        named stage's running total. Re-using the same name (e.g. a stage
        that legitimately runs twice) accumulates rather than overwrites."""
        return _StageTimer(self, name, count, unit)

    def note(self, name, *, extra):
        """Attach a free-form postfix string (e.g. a GPU/CPU split, a
        skip reason) to a stage's row without touching its timing — for
        context that doesn't reduce to a single throughput number."""
        self._record(name).extra = str(extra)

    def report(self, stream=None):
        if not self.enabled or not self._records:
            return
        total_wall = max(1e-9, time.perf_counter() - self._run_t0)
        rows = list(self._records.values())
        tracked = sum(r.seconds for r in rows)
        other = max(0.0, total_wall - tracked)
        name_w = max([len(r.name) for r in rows] + [len("Untracked / overhead")])
        log_info("")
        log_info(bold(cyan("[PROFILE] Per-stage wall-clock breakdown (for tuning)")))
        for r in sorted(rows, key=lambda r: -r.seconds):
            pct = 100.0 * r.seconds / total_wall
            rate = f"  ({r.count / r.seconds:,.1f} {r.unit}/s)" if r.count and r.seconds > 0 and r.unit else ""
            calls = f"  x{r.calls}" if r.calls > 1 else ""
            # Short extras (a skip reason) stay inline in brackets; a long one
            # (the CELF GPU-event breakdown folded in from selection.py) gets
            # its own indented continuation line instead of stretching the
            # main row past the terminal width.
            inline_extra, wrapped_extra = "", None
            if r.extra:
                if len(r.extra) <= 40:
                    inline_extra = f"  [{r.extra}]"
                else:
                    wrapped_extra = r.extra
            log_info(f"  {r.name.ljust(name_w)}  {_duration(r.seconds):>8}  {pct:5.1f}%{rate}{calls}{inline_extra}")
            if wrapped_extra:
                log_info(f"  {dim('    ' + wrapped_extra)}")
        pct_other = 100.0 * other / total_wall
        log_info(f"  {dim('Untracked / overhead'.ljust(name_w))}  {_duration(other):>8}  {pct_other:5.1f}%")
        log_info(f"  {bold('Total wall time'.ljust(name_w))}  {_duration(total_wall):>8}  {green('100.0%')}")


# One profiler per process; cli.py calls begin_run() at the top of main()
# and report() once at the very end, and extractor.py/selection.py/minimize
# call track() around each stage they own.
stage_profiler = StageProfiler()


class GpuPhaseProfile:
    """Accumulates the same host/GPU-event phase buckets the CELF scorer's
    `_celf_profile` dict already used (host alloc, host enqueue, GPU kernel,
    H2D, D2H, synchronization wait) -- but reusable for *any* GPU-heavy
    stage, not just CELF. STAGE 1/S/2/3 and the bloom-filter build all run
    the same host-enqueue-then-blocking-finish pattern per batch, so one
    accumulator shape works for all of them and every stage's tuning story
    reads the same way.

    Only ever constructed when RULEST_CELF_PROFILE=1 (which `--profile`
    sets): the per-call event bookkeeping costs a little extra, so callers
    gate construction on that env var and pass `_profile=None` otherwise,
    which every instrumented gpu_engine.py call site treats as "skip all of
    this cheaply"."""
    __slots__ = ("calls", "items", "host_alloc_s", "host_enqueue_s",
                 "kernel_gpu_s", "h2d_gpu_s", "d2h_gpu_s", "wait_wall_s")

    def __init__(self):
        self.calls = 0
        self.items = 0
        self.host_alloc_s = 0.0
        self.host_enqueue_s = 0.0
        self.kernel_gpu_s = 0.0
        self.h2d_gpu_s = 0.0
        self.d2h_gpu_s = 0.0
        self.wait_wall_s = 0.0

    @staticmethod
    def event_seconds(events):
        """Sum pyopencl Event.profile.end - .start across `events`, the same
        way gpu_engine.py's own CELF `_event_seconds` helper does. Silently
        skips any event whose profiling info isn't available (e.g. the
        queue wasn't created with PROFILING_ENABLE) rather than raising."""
        total = 0.0
        for ev in events:
            try:
                total += max(0, int(ev.profile.end) - int(ev.profile.start)) / 1e9
            except Exception:
                pass
        return total

    def add_call(self, *, items=0, host_alloc=0.0, host_enqueue=0.0, wait=0.0,
                 kernel_events=(), h2d_events=(), d2h_events=()):
        self.calls += 1
        self.items += items
        self.host_alloc_s += host_alloc
        self.host_enqueue_s += host_enqueue
        self.wait_wall_s += wait
        self.kernel_gpu_s += self.event_seconds(kernel_events)
        self.h2d_gpu_s += self.event_seconds(h2d_events)
        self.d2h_gpu_s += self.event_seconds(d2h_events)

    def summary(self):
        """Format like `celf_profile_summary()` does, so every stage's
        detailed row in the unified table looks the same. None if no calls
        were ever recorded (e.g. the stage was skipped or ran 0 batches)."""
        if not self.calls:
            return None
        return (
            "gpu calls={calls:,}; host alloc={alloc:.3f}s; host enqueue={enqueue:.3f}s; "
            "GPU kernel={kernel:.3f}s; H2D={h2d:.3f}s; D2H={d2h:.3f}s; "
            "synchronization wait={wait:.3f}s; avg kernel/call={avgk:.2f}ms"
        ).format(
            calls=self.calls, alloc=self.host_alloc_s, enqueue=self.host_enqueue_s,
            kernel=self.kernel_gpu_s, h2d=self.h2d_gpu_s, d2h=self.d2h_gpu_s,
            wait=self.wait_wall_s, avgk=1000.0 * self.kernel_gpu_s / max(1, self.calls),
        )
