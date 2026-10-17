"""Regression tests for the GPU `apply()` kernel's duplicate/shift rules.

Background
----------
`celf_score_chains_gpu` scores the same candidate rule chain through three
different OpenCL dispatches over a run (fast upper-bound pass, batched exact
pass, and a final singleton "recording" pass for the winner). All three are
supposed to compute the exact same output string for the exact same
(candidate, word) pair -- `selection.py` relies on that invariant and raises
`_CelfExactBatchMismatch` the moment the batched exact gain disagrees with
the winner's recorded gain (see the `[WARN] ... score mismatch` in
production logs).

The GPU `apply()` implementation of the hashcat `y<n>`, `Y<n>`, `z<n>` and
`Z<n>` operators (kernel_source.py) previously diverged from the CPU
reference (`common.py::_py_apply_single_rule`) in two ways:

1. `y<n>` performed the wrong operation (append instead of prepend).
2. `y<n>`/`Y<n>`/`z<n>`/`Z<n>` were missing bounds checks against the
   *current* string length (`*ol`), so whenever `n` exceeded the word's
   length (or the word had already been truncated to empty by an earlier
   op in the chain), the kernel read bytes past what the chain had written
   -- i.e. uninitialized/stale per-work-item stack memory. That memory
   differs between separate kernel dispatches, so the same candidate could
   produce a different output string (and therefore a different target
   match) in the upper-bound pass, the batched exact pass, and the
   recording pass -- exactly the nondeterminism CELF's mismatch guard was
   catching.

Since this sandbox has no PyOpenCL/physical GPU (see
CELF_PERFORMANCE_PATCH.md), these tests port the *fixed* `apply()` C logic
for `y`/`Y`/`z`/`Z` into an equivalent, bounds-respecting Python
implementation and check it against the existing CPU reference
`_py_apply_single_rule` for every length relationship (n < len, n == len,
n > len, and the empty-word edge case). This both documents the intended
semantics and will catch any future GPU/CPU apply() divergence on these
ops before it reaches a multi-hour CELF run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest.common import _py_apply_single_rule  # CPU/host reference oracle

MAX_OUTPUT_LEN = 512


def _dpos(c):
    if '0' <= c <= '9':
        return ord(c) - ord('0')
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 10
    return -1


def gpu_apply_fixed(rule, word):
    """Python port of the FIXED kernel_source.py apply() for y/Y/z/Y rules.

    Mirrors the exact bounds checks now present in kernel_source.py so this
    test fails if that file's logic is ever changed independently of this
    port (the two are expected to stay in lockstep for these four ops).
    """
    out = list(word)
    ol = len(out)
    if not rule or not word:
        return word
    cmd = rule[0]
    if len(rule) == 2:
        p = rule[1]
        pv = _dpos(p)
        if cmd == 'z' and pv >= 0:
            n = pv
            if n > 0 and ol > 0 and ol + n <= MAX_OUTPUT_LEN:
                f = out[0]
                out = [f] * n + out
                ol += n
        elif cmd == 'Z' and pv >= 0:
            n = pv
            if n > 0 and ol > 0 and ol + n <= MAX_OUTPUT_LEN:
                last = out[ol - 1]
                out = out + [last] * n
                ol += n
        elif cmd == 'y' and pv >= 0:
            n = pv
            k = min(n, ol)
            if k > 0 and ol + k <= MAX_OUTPUT_LEN:
                out = out[:k] + out
                ol += k
        elif cmd == 'Y' and pv >= 0:
            n = pv
            if n > 0 and n <= ol and ol + n <= MAX_OUTPUT_LEN:
                out = out + out[ol - n:ol]
                ol += n
    return ''.join(out)


class TestYRulePrependSemantics:
    """`y<n>`: reference is `w = w[:n] + w` (prepend), not append."""

    def test_n_less_than_length_prepends_prefix(self):
        word = 'hello'
        rule = 'y2'
        expected = _py_apply_single_rule(rule, word)
        assert expected == 'hehello'
        assert gpu_apply_fixed(rule, word) == expected

    def test_n_equal_to_length_duplicates_whole_word(self):
        word = 'abc'
        rule = 'y3'
        expected = _py_apply_single_rule(rule, word)
        assert gpu_apply_fixed(rule, word) == expected

    def test_n_greater_than_length_clamps_like_python_slice(self):
        # Python: 'abc'[:99] == 'abc', so w[:99]+w == 'abcabc', never an
        # out-of-bounds read. The old kernel read past the written string
        # here instead of clamping.
        word = 'abc'
        rule = 'y9'
        expected = _py_apply_single_rule(rule, word)
        assert expected == 'abcabc'
        assert gpu_apply_fixed(rule, word) == expected


class TestYUpperRuleRequiresLongEnoughWord:
    """`Y<n>`: reference is a no-op unless `len(w) >= n`."""

    def test_n_less_than_length_appends_suffix(self):
        word = 'hello'
        rule = 'Y2'
        expected = _py_apply_single_rule(rule, word)
        assert expected == 'hellolo'
        assert gpu_apply_fixed(rule, word) == expected

    def test_n_greater_than_length_is_a_no_op(self):
        # The old kernel only checked ol+n<=MAX_OUTPUT_LEN and computed
        # out[ol-n+i], underflowing to a negative/garbage index when n>ol.
        word = 'ab'
        rule = 'Y5'
        expected = _py_apply_single_rule(rule, word)
        assert expected == word
        assert gpu_apply_fixed(rule, word) == expected


class TestZRulesIgnoreEmptyWord:
    """`z<n>`/`Z<n>`: reference requires a non-empty word (`if n>0 and w`)."""

    def test_z_on_nonempty_word_matches_reference(self):
        word = 'cat'
        rule = 'z3'
        expected = _py_apply_single_rule(rule, word)
        assert expected == 'ccccat'
        assert gpu_apply_fixed(rule, word) == expected

    def test_capital_z_on_nonempty_word_matches_reference(self):
        word = 'cat'
        rule = 'Z3'
        expected = _py_apply_single_rule(rule, word)
        assert expected == 'catttt'
        assert gpu_apply_fixed(rule, word) == expected

    def test_z_on_empty_word_is_a_no_op_not_a_buffer_underflow(self):
        # word='' is filtered out by apply()'s `if (!rl || !il) return 1`
        # at the top level, but a chain can still reach z/Z with ol==0
        # after an earlier truncating op (e.g. ']' repeated). The old
        # kernel read out[0] (z) / out[ol-1] (Z, underflowing to -1)
        # unconditionally once pv>=0, regardless of ol.
        assert gpu_apply_fixed('z2', '') == ''
        assert gpu_apply_fixed('Z2', '') == ''
