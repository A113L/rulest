"""Regression tests for the "quiet down hashmap building" fix.

`build_indexed_open_addressing_table` and `fnv1a64_packed_records_chunked`
used to print their own unthrottled `[HASHMAP] ...` lines unconditionally
(the latter) or always-on-by-default (the former) while gpu_engine.py's CELF
target-table build already runs under a single-line ActivityHeartbeat -- the
two progress reporters interleaved and the heartbeat's one overwritten line
kept getting interrupted by a wall of separate scrolling prints. Both
functions can now be told to stay quiet via progress=False, which is what
gpu_engine.py's single call site now passes.
"""
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rulest"))

import numpy as np
from hashing import build_indexed_open_addressing_table  # noqa: E402


class TestBuildIndexedOpenAddressingTableProgress:
    def test_progress_false_prints_nothing(self):
        keys = np.arange(1, 500, dtype=np.uint64) * 2654435761 % (2**63)
        buf = io.StringIO()
        with redirect_stdout(buf):
            build_indexed_open_addressing_table(keys, progress=False)
        assert buf.getvalue() == ""

    def test_progress_true_is_still_the_default_and_prints(self):
        # Backward compatibility: any other/future caller that doesn't pass
        # progress= at all must see unchanged behavior.
        keys = np.arange(1, 500, dtype=np.uint64) * 2654435761 % (2**63)
        buf = io.StringIO()
        with redirect_stdout(buf):
            build_indexed_open_addressing_table(keys)
        assert "[HASHMAP]" in buf.getvalue()

    def test_progress_false_result_matches_progress_true_result(self):
        # Quieting the function must not change what it computes.
        keys = np.arange(1, 500, dtype=np.uint64) * 2654435761 % (2**63)
        quiet = build_indexed_open_addressing_table(keys, progress=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            loud = build_indexed_open_addressing_table(keys, progress=True)
        for a, b in zip(quiet, loud):
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b)
            else:
                assert a == b
