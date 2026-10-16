"""Ranker-style one-line terminal progress and CELF activity heartbeat. CELF_PROGRESS_DIAGNOSTICS_PATCH_V2"""
from __future__ import annotations
from collections import deque
import math
import os
import re
import shutil
import sys
import threading
import time

# Matches ANSI SGR color/style codes (e.g. the cyan()/green()/bold() wrappers
# from common.py: "\x1b[96m...\x1b[0m"). Used to measure how many columns a
# string actually occupies on screen, as opposed to len(), which counts the
# invisible escape bytes as real characters too.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text):
    return len(_ANSI_RE.sub("", text))

try:
    # Reuse the exact same ANSI helpers as every other phase's progress bars
    # so CELF's heartbeat/status text matches their color convention.
    from .common import cyan as _cyan, green as _green
except Exception:  # pragma: no cover - keep this module usable standalone
    def _cyan(t): return str(t)
    def _green(t): return str(t)


def _duration(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    n = int(seconds)
    h, rem = divmod(n, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _write(stream, text):
    try:
        os.write(stream.fileno(), text.encode("utf-8", errors="replace"))
    except (AttributeError, OSError, ValueError):
        stream.write(text)
        stream.flush()


class ProgressBar:  # RULEST_CELF_PROGRESS_RATE_FIX_V1
    """Low-noise, ranker-style tqdm-compatible single-line progress bar."""
    def __init__(self, total=None, desc="", unit="it", *, stream=None,
                 min_interval=0.5, bar_width=24, enabled=True, heartbeat=True,
                 iterable=None, initial=0, disable=None, mininterval=None,
                 **_ignored):
        self.stream = stream or sys.stdout
        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                # A lengthless iterable, like total=None, is indeterminate.
                total = None
        self.total = max(1, int(total)) if total is not None else None
        self.desc = str(desc)
        self.unit = str(unit).strip() or "it"
        # `mininterval` is accepted as a tqdm-compatible alias of
        # `min_interval` -- callers written against the real tqdm API pass
        # this name, and it used to fall into **_ignored and silently keep
        # the 0.5s default no matter what was asked for.
        self.min_interval = max(0.1, float(mininterval if mininterval is not None else min_interval))
        self.bar_width = max(10, int(bar_width))
        # `disable` is tqdm's name for "don't render"; honor it the same way
        # real tqdm does. It used to land in **_ignored, so passing
        # disable=True (e.g. stderr is not a tty) had no effect at all: the
        # bar -- ETA included -- kept writing raw \r/\x1b[K control bytes
        # into whatever the stream was redirected to (a log file, a pipe).
        # `disable` wins over `enabled` when both are given.
        self.enabled = (
            bool(enabled) and not bool(disable)
            and os.environ.get("RULEST_NO_PROGRESS") != "1"
        )
        self.n = max(0, int(initial))
        self.postfix = ""
        self.start_time = time.monotonic()
        self._rate_window_seconds = 15.0
        self._rate_samples = deque([(self.start_time, self.n)])
        self.last_render = 0.0
        self.closed = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        if self.enabled:
            self.refresh(force=True)
            if heartbeat:
                self._thread = threading.Thread(target=self._heartbeat_loop,
                    name="rulest-progress-heartbeat", daemon=True)
                self._thread.start()

    def _heartbeat_loop(self):
        while not self._stop.wait(self.min_interval):
            self.refresh()

    def _recent_rate(self, now=None):
        """Return (units/second, sample span) over the trailing rate window."""
        if now is None:
            now = time.monotonic()
        samples = self._rate_samples
        cutoff = now - self._rate_window_seconds
        # Keep one anchor sample at or before the window cutoff.
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.popleft()
        if len(samples) < 2:
            return 0.0, 0.0
        first_t, first_n = samples[0]
        last_t, last_n = samples[-1]
        if last_t < cutoff or last_n <= first_n:
            return 0.0, max(0.0, last_t - first_t)
        # `first_n` is the last known count at or before the cutoff. If no
        # update happened near the cutoff (for example, CELF spent 90 s
        # revalidating before its first selection), measure over the trailing
        # window, not over the entire startup stall.
        window_start = max(first_t, cutoff)
        span = last_t - window_start
        if span <= 0.0:
            return 0.0, 0.0
        return (last_n - first_n) / span, span

    def _dynamic_bar_width(self, terminal_width, non_bar_len):
        """Stretch the bar to fill the line, the same way real tqdm's
        dynamic_ncols does for every other phase's bars (STAGE 3's
        evolution bar, MINIMIZE's dedup bar) -- those are plain tqdm, which
        auto-sizes `{bar}` to the terminal; this bar used to hardcode 24
        cells regardless of how wide the terminal actually was, which is
        why it rendered noticeably narrower and looked like a different,
        less-finished tool next to them.

        `self.bar_width` (the constructor's bar_width=, default 24) becomes
        the floor this stretches up from, never the fixed size -- so a
        caller that explicitly asked for a minimum still gets at least
        that many cells, but a wide terminal fills the rest.

        `non_bar_len` is the *visible* length of everything else on the line
        (desc, percent, counts, ETA, rate -- everything except the bar
        itself; callers must pass this pre-measured with `_visible_len`,
        since `self.desc` can carry a color wrapper like green()). The
        postfix's current visible length is reserved too, so stretching the
        bar doesn't immediately push every postfix field (where
        "recovered", "selected", "exact scans" live) off the end and into
        `_fit_postfix`'s drop-from-the-right fallback.

        Every length here is *visible* columns, not Python string length:
        this bar's desc/postfix are routinely wrapped in cyan()/green()
        (see common.py), whose ANSI escape bytes add up to 10+ invisible
        characters per colored field without occupying a single column. Set
        bar width (or decide to drop a postfix field) from raw len() and
        the bar renders visibly narrower than the terminal actually allows
        -- the colored CELF bars undershot a plain-text bar like
        "Upper-bound pass" by exactly that phantom amount, never reaching a
        full-width 100% line despite having the room for it.
        """
        if terminal_width <= 20:
            return self.bar_width
        postfix_len = _visible_len(" | " + self.postfix) if self.postfix else 0
        available = terminal_width - non_bar_len - postfix_len
        return max(self.bar_width, available)

    def _line(self):
        now = time.monotonic()
        elapsed = max(0.0, now - self.start_time)
        avg_rate = self.n / elapsed if elapsed > 0 else 0.0
        current_rate, sample_span = self._recent_rate(now)
        # One rate figure, not two: pick whichever is the more informative
        # one (trailing-window once it has enough samples, else the overall
        # average) instead of always printing both "now" and "avg". The old
        # two-rate text was the single biggest reason the core line (before
        # postfix is even appended) already overflowed an 80/100-col
        # terminal, so ETA and everything after it — including the whole
        # postfix, where "recovered" lives — got silently chopped off by
        # the hard width-based slice below.
        eta_rate = current_rate if current_rate > 0 and sample_span >= min(5.0, self._rate_window_seconds / 3.0) else avg_rate
        rate_text = f"{eta_rate:,.1f} {self.unit}/s" if eta_rate > 0 else f"-- {self.unit}/s"

        try:
            width = shutil.get_terminal_size((120, 20)).columns
        except (OSError, ValueError):
            width = 120

        if self.total is None:
            # No known stopping point: deliberately omit percent, total, and ETA.
            plural = "s" if self.n != 1 else ""
            prefix = f"{self.desc}: ["
            suffix = f"] {self.n:,} {self.unit}{plural} [{_duration(elapsed)}, {rate_text}]"
            bar_width = self._dynamic_bar_width(width, _visible_len(prefix) + _visible_len(suffix))
            pos = int(elapsed / 0.18) % bar_width
            cells = [" "] * bar_width
            cells[pos] = "█"
            cells[(pos - 1) % bar_width] = "█"
            core = prefix + "".join(cells) + suffix
        else:
            frac = min(1.0, max(0.0, self.n / self.total))
            eta = (self.total - self.n) / eta_rate if eta_rate > 0 else float("nan")
            prefix = f"{self.desc}: {frac * 100:3.0f}%|"
            suffix = (f"| {self.n:,}/{self.total:,} [{_duration(elapsed)}<{_duration(eta)}, "
                      f"{rate_text}]")
            bar_width = self._dynamic_bar_width(width, _visible_len(prefix) + _visible_len(suffix))
            filled = max(0, min(bar_width, int(round(bar_width * frac))))
            bar = "█" * filled + " " * (bar_width - filled)
            core = prefix + bar + suffix

        return self._fit_postfix(core, width)

    def _fit_postfix(self, core, width=None):
        """Append postfix fields (desc-order), dropping whole trailing fields
        — never slicing mid-field — once the line stops fitting the terminal
        width. The core (bar/counts/ETA/rate) always renders in full; only
        postfix fields are ever sacrificed, and the caller is expected to put
        the fields that must stay visible (e.g. "recovered") first.

        Width comparisons use _visible_len, not len(): core/postfix text is
        routinely colorized (cyan()/green() from common.py), and measuring
        raw string length here would count invisible ANSI escape bytes as
        occupying columns -- undercounting how many actual characters still
        fit and dropping fields (or truncating core) well before the
        terminal is really full.
        """
        if width is None:
            try:
                width = shutil.get_terminal_size((120, 20)).columns
            except (OSError, ValueError):
                width = 120
        if not self.postfix or width <= 20:
            return core if width <= 20 or _visible_len(core) <= width else core[:width - 1]
        fields = self.postfix.split(" | ")
        line = core
        for field in fields:
            candidate = line + " | " + field
            if width > 20 and _visible_len(candidate) > width:
                break
            line = candidate
        return line

    def refresh(self, force=False):
        if not self.enabled: return
        with self._lock:
            if self.closed: return
            now = time.monotonic()
            if not force and now - self.last_render < self.min_interval: return
            line = self._line()  # already fitted to terminal width, field-aware
            self.last_render = now
            _write(self.stream, "\r" + line + "\x1b[K")

    def update(self, value=1, *, refresh=True):
        with self._lock:
            if self.closed:
                return
            self.n += max(0, int(value))
            now = time.monotonic()
            self._rate_samples.append((now, self.n))
            cutoff = now - self._rate_window_seconds
            while len(self._rate_samples) > 1 and self._rate_samples[1][0] <= cutoff:
                self._rate_samples.popleft()
        if refresh:
            self.refresh()

    def set_total(self, total, refresh=True):
        with self._lock:
            if self.closed:
                return
            self.total = max(1, int(total), self.n) if total is not None else None
        if refresh:
            self.refresh(force=True)

    def set_postfix(self, values, refresh=True):
        self.postfix = (" | ".join(f"{k}={v}" for k, v in values.items())
                        if isinstance(values, dict) else str(values))
        if refresh: self.refresh(force=True)

    def set_postfix_str(self, text, refresh=True):
        self.postfix = str(text)
        if refresh: self.refresh(force=True)

    def set_description(self, desc, refresh=True):
        self.desc = str(desc)
        if refresh: self.refresh(force=True)

    def close(self):
        if not self.enabled: return
        with self._lock:
            if self.closed: return
            self.closed = True
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.5, self.min_interval * 2))
        line = self._line()  # already fitted to terminal width, field-aware
        _write(self.stream, "\r" + line + "\x1b[K\n")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

    def __iter__(self):
        if self.iterable is None: return iter(())
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __len__(self):
        return self.total if self.total is not None else self.n


def tqdm(iterable=None, *args, **kwargs):
    if args and "total" not in kwargs and isinstance(args[0], int):
        kwargs["total"] = args[0]
    return ProgressBar(iterable=iterable, **kwargs)


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)


class ActivityHeartbeat:
    """Indeterminate, timestamped heartbeat for long CPU/GPU setup phases.

    Renders as a single line, overwritten in place (\\r ... \\x1b[K), the same
    convention as every other phase's status/progress output -- never a new
    scrolling print per tick. `set_desc` lets a caller attach live progress
    text (a running count, a rate, a percentage) to that one line instead of
    emitting its own separate, unthrottled print() calls -- which is what
    the selection stage's target-dedupe pass used to do (a full new
    `[HASHMAP] ...` line printed at least once a second), the noisiest part
    of this phase's output and visually inconsistent with the `[CELF] ...`
    heartbeats immediately before and after it.
    """
    def __init__(self, desc, *, stream=None, interval=0.5):
        self.desc = str(desc)
        self.stream = stream or sys.stdout
        self.interval = max(0.25, float(interval))
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._closed = False
        self._spinner = 0
        self._last_render = 0.0

    def start(self):
        self._last_render = time.monotonic()
        _write(self.stream, f"\r{self.desc} | {_cyan('working')} | elapsed 00:00\x1b[K")
        self._thread = threading.Thread(target=self._run, name="celf-hashmap-heartbeat", daemon=True)
        self._thread.start()
        return self

    def set_desc(self, desc, *, force=False):
        """Update the heartbeat's label (e.g. with a live count/rate) and
        re-render the line, rather than waiting for the next spinner tick --
        so a caller driving this from its own throttled loop sees its update
        land promptly instead of looking stuck for up to `interval` seconds.

        Re-renders are themselves throttled to `interval` (same as the
        background spinner) unless `force=True`, so a caller that forgets to
        throttle its own call site can't turn this back into a per-row print
        storm -- the whole point of routing dedupe progress through here."""
        with self._lock:
            if self._closed:
                return
            self.desc = str(desc)
            now = time.monotonic()
            if not force and now - self._last_render < self.interval:
                return
            self._last_render = now
            elapsed = now - self.started
            frame = "|/-\\"[self._spinner % 4]
            _write(self.stream, f"\r{self.desc} | {frame} {_cyan('working')} | elapsed {_cyan(_duration(elapsed))}\x1b[K")

    def _run(self):
        frames = "|/-\\"
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._closed: return
                elapsed = time.monotonic() - self.started
                frame = frames[self._spinner % len(frames)]
                self._spinner += 1
                self._last_render = time.monotonic()
                _write(self.stream, f"\r{self.desc} | {frame} {_cyan('working')} | elapsed {_cyan(_duration(elapsed))}\x1b[K")

    def stop(self):
        with self._lock:
            if self._closed: return
            self._closed = True
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0.6, self.interval * 2))
        _write(self.stream, f"\r{self.desc} | {_green('done')} | elapsed {_cyan(_duration(time.monotonic() - self.started))}\x1b[K\n")

    def __enter__(self): return self.start()
    def __exit__(self, exc_type, exc, tb): self.stop()
