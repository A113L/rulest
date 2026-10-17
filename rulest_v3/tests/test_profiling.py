"""Unit tests for rulest.profiling.StageProfiler: the single-flag, per-stage
wall-clock profiling table added to the extraction pipeline (STAGE 0-3,
bloom build, minimize, selection/CELF) and gpu_engine.py's CELF summary
string that gets folded into it.

Covers:
  - opt-in by default (no output unless enabled via configure() or the
    RULEST_PROFILE / RULEST_CELF_PROFILE env vars)
  - accumulation across repeated `track()` calls for the same stage name
  - throughput (count/unit) reporting and `set_count()` called after the
    tracked work finishes
  - `note()` attaching a skip reason or a folded-in CELF summary without
    disturbing timing
  - the report() output actually appearing (or not) on stdout
"""
import importlib
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest.profiling import StageProfiler, GpuPhaseProfile  # noqa: E402


class _FakeEvent:
    """Stand-in for a pyopencl Event with profiling enabled: .profile.start
    and .profile.end are nanosecond timestamps, matching what
    GpuPhaseProfile.event_seconds() reads from a real GPU event."""
    def __init__(self, start_ns, end_ns):
        self.profile = types.SimpleNamespace(start=start_ns, end=end_ns)


class _BrokenEvent:
    """A pyopencl Event accessed without PROFILING_ENABLE on the queue
    raises when .profile is touched -- event_seconds() must swallow that
    per-event rather than blow up the whole summary."""
    @property
    def profile(self):
        raise RuntimeError("CL_PROFILING_INFO_NOT_AVAILABLE")


class TestGpuPhaseProfileEvents:
    def test_event_seconds_sums_nanosecond_deltas(self):
        events = [_FakeEvent(0, 1_000_000), _FakeEvent(5_000_000, 7_500_000)]
        assert GpuPhaseProfile.event_seconds(events) == pytest.approx(0.0035)

    def test_event_seconds_ignores_negative_or_broken_events(self):
        events = [_FakeEvent(100, 50), _BrokenEvent()]
        assert GpuPhaseProfile.event_seconds(events) == 0.0

    def test_event_seconds_empty_list_is_zero(self):
        assert GpuPhaseProfile.event_seconds([]) == 0.0


class TestGpuPhaseProfileAccumulation:
    def test_summary_none_before_any_call(self):
        gp = GpuPhaseProfile()
        assert gp.summary() is None

    def test_add_call_accumulates_host_and_event_time(self):
        gp = GpuPhaseProfile()
        gp.add_call(items=1000, host_alloc=0.01, host_enqueue=0.02, wait=0.03,
                    kernel_events=[_FakeEvent(0, 50_000_000)],
                    h2d_events=[_FakeEvent(0, 1_000_000)],
                    d2h_events=[_FakeEvent(0, 2_000_000)])
        gp.add_call(items=500, host_alloc=0.005, host_enqueue=0.01, wait=0.015,
                    kernel_events=[_FakeEvent(0, 25_000_000)])
        assert gp.calls == 2
        assert gp.items == 1500
        assert gp.host_alloc_s == pytest.approx(0.015)
        assert gp.host_enqueue_s == pytest.approx(0.03)
        assert gp.wait_wall_s == pytest.approx(0.045)
        assert gp.kernel_gpu_s == pytest.approx(0.075)
        assert gp.h2d_gpu_s == pytest.approx(0.001)
        assert gp.d2h_gpu_s == pytest.approx(0.002)

    def test_summary_formats_all_phase_buckets(self):
        gp = GpuPhaseProfile()
        gp.add_call(items=100, host_alloc=0.01, host_enqueue=0.02, wait=0.03,
                    kernel_events=[_FakeEvent(0, 10_000_000)])
        s = gp.summary()
        for fragment in ("gpu calls=1", "host alloc=0.010s", "host enqueue=0.020s",
                          "GPU kernel=0.010s", "synchronization wait=0.030s",
                          "avg kernel/call="):
            assert fragment in s

    def test_summary_can_be_folded_into_stage_table_via_note(self, profiler, capsys):
        """The real integration point: gpu_engine.py / genetic.py build a
        GpuPhaseProfile, extractor.py reads .summary() and hands it to
        stage_profiler.note() for STAGE 1/S/2/3 and the bloom build, exactly
        like selection.py already does for CELF."""
        gp = GpuPhaseProfile()
        gp.add_call(items=100, host_alloc=0.01, host_enqueue=0.02, wait=0.03,
                    kernel_events=[_FakeEvent(0, 10_000_000)])
        with profiler.track("STAGE 1 — Single Rule Search"):
            time.sleep(0.001)
        profiler.note("STAGE 1 — Single Rule Search", extra=gp.summary())
        profiler.report()
        out = capsys.readouterr().out
        assert "GPU kernel=0.010s" in out


@pytest.fixture
def profiler():
    p = StageProfiler()
    p.configure(True)
    p.begin_run()
    return p


class TestOptIn:
    def test_default_construction_is_disabled(self):
        """A bare StageProfiler() (e.g. the module-level singleton created
        at import time, before cli.py has parsed --profile) must not emit
        anything -- profiling is opt-in, not opt-out."""
        p = StageProfiler()
        assert p.enabled is False
        assert p.detailed is False

    def test_report_prints_nothing_when_disabled(self, capsys):
        p = StageProfiler()
        with p.track("STAGE 1"):
            time.sleep(0.001)
        p.report()
        out = capsys.readouterr().out
        assert out == ""

    def test_report_prints_nothing_when_no_stages_tracked(self, capsys):
        """Even if enabled, an empty profiler (nothing ever tracked) should
        stay silent rather than print a table with zero rows."""
        p = StageProfiler()
        p.configure(True)
        p.report()
        assert capsys.readouterr().out == ""

    def test_configure_toggles_both_enabled_and_detailed(self):
        p = StageProfiler()
        p.configure(True)
        assert p.enabled is True
        assert p.detailed is True
        p.configure(False)
        assert p.enabled is False
        assert p.detailed is False

    @pytest.mark.parametrize("env_var", ["RULEST_PROFILE", "RULEST_CELF_PROFILE"])
    def test_env_var_fallback_enables_without_configure(self, env_var, monkeypatch):
        """A script that sets RULEST_PROFILE=1 or RULEST_CELF_PROFILE=1
        before constructing/importing the profiler (no argparse, no
        explicit configure() call) should still get both switches on --
        this is the documented fallback for non-CLI callers."""
        monkeypatch.setenv(env_var, "1")
        p = StageProfiler()
        assert p.enabled is True
        assert p.detailed is True


class TestTrackAccumulation:
    def test_single_stage_duration_is_recorded(self, profiler, capsys):
        with profiler.track("STAGE 1"):
            time.sleep(0.02)
        profiler.report()
        out = capsys.readouterr().out
        assert "STAGE 1" in out

    def test_repeated_track_same_name_accumulates(self, profiler):
        """STAGE 2 (or any stage) can legitimately be entered more than once
        in one run (e.g. multi-GPU retry paths) -- durations must sum, not
        overwrite, and call count must reflect every entry."""
        with profiler.track("STAGE 2"):
            time.sleep(0.01)
        with profiler.track("STAGE 2"):
            time.sleep(0.01)
        rec = profiler._records["STAGE 2"]
        assert rec.calls == 2
        assert rec.seconds >= 0.02

    def test_exception_inside_track_still_records_time(self, profiler):
        """A stage that raises partway through must not silently lose its
        elapsed time -- __exit__ always accumulates before propagating."""
        with pytest.raises(ValueError):
            with profiler.track("STAGE 3"):
                time.sleep(0.01)
                raise ValueError("boom")
        rec = profiler._records["STAGE 3"]
        assert rec.calls == 1
        assert rec.seconds > 0.0

    def test_distinct_stage_names_tracked_independently(self, profiler):
        with profiler.track("STAGE 0 — Token-Strip"):
            time.sleep(0.005)
        with profiler.track("STAGE 1 — Single Rule Search"):
            time.sleep(0.005)
        assert set(profiler._records.keys()) == {
            "STAGE 0 — Token-Strip", "STAGE 1 — Single Rule Search",
        }


class TestThroughput:
    def test_set_count_after_block_records_rate(self, profiler, capsys):
        with profiler.track("STAGE 1", unit="combo") as t:
            time.sleep(0.05)
            t.set_count(1000)
        profiler.report()
        out = capsys.readouterr().out
        assert "combo/s" in out

    def test_count_supplied_up_front_also_works(self, profiler, capsys):
        with profiler.track("Bloom filter build", count=500, unit="word"):
            time.sleep(0.02)
        profiler.report()
        out = capsys.readouterr().out
        assert "word/s" in out

    def test_zero_or_missing_count_shows_no_rate(self, profiler, capsys):
        with profiler.track("STAGE S — Seed Extraction"):
            time.sleep(0.01)
        profiler.report()
        out = capsys.readouterr().out
        assert "/s)" not in out

    def test_set_count_accumulates_across_repeated_entries(self, profiler):
        with profiler.track("STAGE 2", unit="chain") as t:
            t.set_count(100)
        with profiler.track("STAGE 2", unit="chain") as t:
            t.set_count(50)
        rec = profiler._records["STAGE 2"]
        assert rec.count == 150


class TestNote:
    def test_note_attaches_skip_reason_without_timing(self, profiler, capsys):
        profiler.note("STAGE 2 — Rule Chain Search", extra="skipped: --max-depth 1")
        profiler.report()
        out = capsys.readouterr().out
        assert "STAGE 2 — Rule Chain Search" in out
        assert "skipped: --max-depth 1" in out
        rec = profiler._records["STAGE 2 — Rule Chain Search"]
        assert rec.seconds == 0.0
        assert rec.calls == 0

    def test_long_extra_wraps_onto_its_own_line(self, profiler, capsys):
        """A long note (the CELF GPU-event breakdown folded in from
        selection.py) must not be squeezed inline in brackets -- it gets an
        indented continuation line so the main row stays scannable."""
        long_summary = "score calls=10 (batch=5, singleton=5), " + "x" * 60
        with profiler.track("Selection (CELF greedy)"):
            time.sleep(0.001)
        profiler.note("Selection (CELF greedy)", extra=long_summary)
        profiler.report()
        out = capsys.readouterr().out
        assert "Selection (CELF greedy)" in out
        assert long_summary in out
        # The row line itself (ending right after the stage's stats) should
        # not also contain the long summary -- it must be on its own line.
        row_line = next(l for l in out.splitlines() if l.strip().startswith("Selection"))
        assert long_summary not in row_line

    def test_short_extra_stays_inline(self, profiler, capsys):
        profiler.note("STAGE S — Seed Extraction", extra="skipped: --no-builtin-seeds")
        profiler.report()
        out = capsys.readouterr().out
        line = next(l for l in out.splitlines() if "STAGE S" in l)
        assert "[skipped: --no-builtin-seeds]" in line

    def test_note_before_track_is_preserved_after_track_runs(self, profiler, capsys):
        """selection.py's cleanup (which calls note()) runs *inside* the
        `with stage_profiler.track(...)` block in cli.py, before that
        block's __exit__ fires -- note() must create the record early and
        track()'s later __exit__ must reuse the same record rather than
        clobbering the extra."""
        name = "Selection (CELF greedy)"
        with profiler.track(name, unit="candidate") as t:
            profiler.note(name, extra="some detail")
            time.sleep(0.005)
            t.set_count(10)
        rec = profiler._records[name]
        assert rec.extra == "some detail"
        assert rec.calls == 1
        assert rec.count == 10


class TestReportShape:
    def test_report_includes_untracked_and_total_rows(self, profiler, capsys):
        with profiler.track("STAGE 1"):
            time.sleep(0.01)
        profiler.report()
        out = capsys.readouterr().out
        assert "Untracked" in out
        assert "Total wall time" in out

    def test_rows_sorted_slowest_first(self, profiler, capsys):
        with profiler.track("fast stage"):
            time.sleep(0.005)
        with profiler.track("slow stage"):
            time.sleep(0.03)
        profiler.report()
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        slow_idx = next(i for i, l in enumerate(lines) if "slow stage" in l)
        fast_idx = next(i for i, l in enumerate(lines) if "fast stage" in l)
        assert slow_idx < fast_idx

    def test_reset_clears_records_and_restarts_clock(self, profiler):
        with profiler.track("STAGE 1"):
            time.sleep(0.01)
        assert profiler._records
        profiler.reset()
        assert profiler._records == {}
