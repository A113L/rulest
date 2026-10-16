"""Regression tests for rulest.progress.ProgressBar / tqdm wrapper.

These pin down a real bug: the wrapper used to accept (and silently drop,
via **_ignored) tqdm-compatible kwargs like `disable` and `mininterval`.
Every CELF progress bar in selection.py passes `disable=not <is-a-tty>` so
that redirected/piped/non-interactive runs don't get raw \\r/\\x1b[K control
bytes -- including the ETA field -- written into a log file. Because
`disable` was ignored, that suppression never actually happened.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rulest"))

from progress import ProgressBar, tqdm, ActivityHeartbeat  # noqa: E402


class _PipeStream:
    """A non-tty, file-descriptor-backed stream (what stderr looks like
    when redirected to a file or piped) that os.write()-based _write()
    can target directly."""
    def __init__(self, write_fd):
        self._write_fd = write_fd

    def fileno(self):
        return self._write_fd

    def isatty(self):
        return False


def _drain(read_fd):
    data = b""
    while True:
        try:
            chunk = os.read(read_fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        data += chunk
    return data


class TestDisableKwarg:
    def test_disable_true_suppresses_all_output(self):
        r, w = os.pipe()
        try:
            bar = tqdm(total=10, desc="celf", unit="rule", disable=True,
                       stream=_PipeStream(w))
            bar.update(5)
            bar.close()
            os.close(w)
            w = -1
            assert _drain(r) == b""
        finally:
            if w != -1:
                os.close(w)
            os.close(r)

    def test_disable_false_still_renders(self):
        r, w = os.pipe()
        try:
            bar = tqdm(total=10, desc="celf", unit="rule", disable=False,
                       stream=_PipeStream(w))
            bar.update(5)
            bar.close()
            os.close(w)
            w = -1
            out = _drain(r).decode()
            assert "celf" in out
            assert "5/10" in out
        finally:
            if w != -1:
                os.close(w)
            os.close(r)

    def test_disable_overrides_enabled_true(self):
        # A caller that (redundantly) passes both must still end up
        # suppressed -- disable is the more specific/explicit intent.
        r, w = os.pipe()
        try:
            bar = ProgressBar(total=10, desc="celf", unit="rule",
                               enabled=True, disable=True, stream=_PipeStream(w))
            bar.update(1)
            bar.close()
            os.close(w)
            w = -1
            assert _drain(r) == b""
        finally:
            if w != -1:
                os.close(w)
            os.close(r)

    def test_disable_omitted_defaults_to_enabled(self):
        # Callers that never pass `disable` (most of the codebase, which
        # uses the real tqdm package instead of this wrapper) must see
        # unchanged behavior: enabled by default.
        r, w = os.pipe()
        try:
            bar = tqdm(total=10, desc="celf", unit="rule", stream=_PipeStream(w))
            bar.update(1)
            bar.close()
            os.close(w)
            w = -1
            assert _drain(r) != b""
        finally:
            if w != -1:
                os.close(w)
            os.close(r)


class TestMinintervalAlias:
    def test_mininterval_kwarg_sets_min_interval(self):
        # The real tqdm's throttle kwarg is `mininterval`; it used to be
        # swallowed by **_ignored, silently keeping the 0.5s default no
        # matter what a caller asked for.
        bar = ProgressBar(total=10, desc="celf", unit="rule", enabled=False,
                           mininterval=2.5)
        assert bar.min_interval == 2.5

    def test_min_interval_kwarg_still_works_directly(self):
        bar = ProgressBar(total=10, desc="celf", unit="rule", enabled=False,
                           min_interval=1.25)
        assert bar.min_interval == 1.25


class TestEtaFieldPresent:
    def test_eta_field_renders_with_total(self):
        r, w = os.pipe()
        try:
            bar = tqdm(total=10, desc="celf", unit="rule", disable=False,
                       stream=_PipeStream(w))
            bar.update(1)
            bar.close()
            os.close(w)
            w = -1
            out = _drain(r).decode()
            # format is "<elapsed><ETA" inside brackets, e.g. "[00:00<00:04"
            assert "<" in out and "rule/s" in out
        finally:
            if w != -1:
                os.close(w)
            os.close(r)

    def test_unknown_tqdm_kwargs_are_still_tolerated(self):
        # bar_format/leave/dynamic_ncols/ncols no longer need to be passed
        # (selection.py's call sites were cleaned up), but a stray/future
        # tqdm-style kwarg must not crash construction.
        bar = ProgressBar(total=10, desc="celf", unit="rule", enabled=False,
                           leave=True, dynamic_ncols=True, ncols=80,
                           bar_format="{l_bar}{bar}")
        bar.close()


class TestActivityHeartbeatSetDesc:
    def test_set_desc_updates_label_and_renders(self):
        r, w = os.pipe()
        try:
            hb = ActivityHeartbeat("phase", stream=_PipeStream(w), interval=10)
            hb.started = hb.started  # keep default; interval is large so the
            # background thread won't tick during this test
            hb.set_desc("phase 5/10", force=True)
            hb.stop()
            os.close(w)
            w = -1
            out = _drain(r).decode()
            assert "phase 5/10" in out
            assert "done" in out
        finally:
            if w != -1:
                os.close(w)
            os.close(r)

    def test_set_desc_is_throttled_without_force(self):
        r, w = os.pipe()
        try:
            hb = ActivityHeartbeat("phase", stream=_PipeStream(w), interval=10)
            hb.start()
            # start() already rendered once; a huge interval (10s) means an
            # un-forced set_desc called right after must be swallowed by the
            # same throttle the background spinner uses, rather than always
            # re-rendering on every call.
            hb.set_desc("phase 1/1000")
            hb._closed = True  # stop the background thread without a real
            hb._stop.set()     # stop() call, so there's no extra "done" line
            os.close(w)
            w = -1
            out = _drain(r).decode()
            assert "1/1000" not in out
        finally:
            if w != -1:
                os.close(w)
            os.close(r)

    def test_set_desc_after_close_is_a_noop(self):
        r, w = os.pipe()
        try:
            hb = ActivityHeartbeat("phase", stream=_PipeStream(w), interval=10)
            hb.stop()
            hb.set_desc("should not appear", force=True)
            os.close(w)
            w = -1
            out = _drain(r).decode()
            assert "should not appear" not in out
        finally:
            if w != -1:
                os.close(w)
            os.close(r)


class TestDynamicBarWidth:
    """The bar used to hardcode bar_width=24 regardless of terminal size --
    every other phase's bars (plain tqdm, dynamic_ncols=True) stretch to
    fill the terminal, so CELF's bars looked visibly narrower/inconsistent
    next to them. The bar now stretches like real tqdm's does, with
    bar_width as a floor rather than a fixed size."""

    def _line_at_width(self, monkeypatch, columns, **kwargs):
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((columns, 20))
        )
        bar = ProgressBar(enabled=False, heartbeat=False, **kwargs)
        return bar._line()

    def test_bar_stretches_to_fill_a_wide_terminal(self, monkeypatch):
        bar_narrow = self._line_at_width(
            monkeypatch, 80, total=100, desc="d", unit="rule", bar_width=24)
        bar_wide = self._line_at_width(
            monkeypatch, 200, total=100, desc="d", unit="rule", bar_width=24)
        assert len(bar_wide) > len(bar_narrow)
        assert len(bar_wide) <= 200
        # Within a few chars of the requested width (postfix-less core).
        assert len(bar_wide) >= 190

    def test_bar_width_is_a_floor_not_fixed(self, monkeypatch):
        # A very narrow terminal (or very long desc/counts) must never push
        # the bar thinner than the caller's requested minimum.
        bar = ProgressBar(enabled=False, heartbeat=False, total=100, desc="d",
                           unit="rule", bar_width=24)
        bar.postfix = "x" * 500  # force available width deeply negative
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((80, 20))
        )
        line = bar._line()
        assert line.count("█") + line.count(" ") >= 24  # bar cells, loosely

    def test_indeterminate_bar_also_stretches(self, monkeypatch):
        # total=None path (no known stopping point) builds its bar
        # differently from the determinate path -- must get the same
        # dynamic-width treatment, not just the percent/ETA path.
        narrow = self._line_at_width(
            monkeypatch, 80, total=None, desc="d", unit="rule", bar_width=24)
        wide = self._line_at_width(
            monkeypatch, 200, total=None, desc="d", unit="rule", bar_width=24)
        assert len(wide) > len(narrow)
        assert len(wide) <= 200

    def test_postfix_is_reserved_space_before_bar_stretches_over_it(self, monkeypatch):
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((160, 20))
        )
        bar = ProgressBar(enabled=False, heartbeat=False, total=100, desc="d",
                           unit="rule", bar_width=24)
        bar.n = 100
        bar.postfix = "recovered=1,890,521"
        line = bar._line()
        assert "recovered=1,890,521" in line
        assert len(line) <= 160


class TestAnsiAwareWidth:
    """The CELF bars route desc/postfix through cyan()/green() (see
    common.py), which wraps visible text in ANSI escape codes. Measuring
    those strings with plain len() counts the invisible escape bytes as if
    they occupied columns, which under-filled the bar/postfix budget by
    exactly that phantom amount -- a colored bar like "[CELF] GPU
    recompute" never reached full terminal width the way an uncolored one
    (e.g. "Upper-bound pass", no postfix colors) did."""

    def test_visible_len_strips_ansi_codes(self):
        from progress import _visible_len
        from rulest.common import cyan, green  # noqa
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rulest"))
        colored = "\x1b[96m1,890,521\x1b[0m"
        assert _visible_len(colored) == len("1,890,521")

    def test_colored_postfix_does_not_undershoot_bar_width(self, monkeypatch):
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((160, 20))
        )
        CYAN, END = "\x1b[96m", "\x1b[0m"

        plain_bar = ProgressBar(enabled=False, heartbeat=False, total=100,
                                 desc="d", unit="rule", bar_width=24)
        plain_bar.n = 50
        plain_bar.postfix = "recovered=1,890,521"
        plain_line = plain_bar._line()

        colored_bar = ProgressBar(enabled=False, heartbeat=False, total=100,
                                   desc="d", unit="rule", bar_width=24)
        colored_bar.n = 50
        colored_bar.postfix = f"recovered={CYAN}1,890,521{END}"
        colored_line = colored_bar._line()

        # Same visible content, same budget -> same number of bar cells,
        # regardless of the ANSI wrapper around the postfix value.
        assert plain_line.count("█") == colored_line.count("█")

    def test_colored_line_never_exceeds_terminal_visible_width(self, monkeypatch):
        import re
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((160, 20))
        )
        CYAN, GREEN, END = "\x1b[96m", "\x1b[92m", "\x1b[0m"
        bar = ProgressBar(enabled=False, heartbeat=False, total=100000,
                           desc=f"{GREEN}[CELF] GPU recompute (approx){END}",
                           unit="rule", bar_width=24)
        bar.n = 3319
        bar.postfix = " | ".join(
            f"{k}={CYAN}{v}{END}" for k, v in [
                ("recovered", "1,890,521"), ("selected", "3319"), ("batch", "32"),
                ("exact scans", "89,935"), ("batch bounds", "12,003"),
            ]
        )
        line = bar._line()
        visible = len(re.sub(r"\x1b\[[0-9;]*m", "", line))
        assert visible <= 160

    def test_colored_bar_fills_width_as_well_as_an_uncolored_one(self, monkeypatch):
        # The actual regression this was reported against: a colored CELF
        # bar should get just as close to full terminal width as a plain
        # one with equivalent visible content -- not undershoot it.
        import re
        monkeypatch.setattr(
            shutil, "get_terminal_size", lambda *_a, **_k: os.terminal_size((160, 20))
        )
        CYAN, END = "\x1b[96m", "\x1b[0m"

        plain = ProgressBar(enabled=False, heartbeat=False, total=100,
                             desc="x", unit="rule", bar_width=24)
        plain.n = 50
        plain.postfix = "a=1 | b=2"
        plain_visible = len(re.sub(r"\x1b\[[0-9;]*m", "", plain._line()))

        colored = ProgressBar(enabled=False, heartbeat=False, total=100,
                               desc="x", unit="rule", bar_width=24)
        colored.n = 50
        colored.postfix = f"a={CYAN}1{END} | b={CYAN}2{END}"
        colored_visible = len(re.sub(r"\x1b\[[0-9;]*m", "", colored._line()))

        assert colored_visible == plain_visible
