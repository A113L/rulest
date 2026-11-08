import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rulest.tokenstrip_hashset import FlatHashSet64


def _words(n):
    return [f"word{i}" for i in range(n)] + ["password", "P@ssw0rd", "zaq1@WSX", "aaaa"]


def test_membership_exact():
    words = _words(200_000)
    s = FlatHashSet64(words)
    assert len(s) == len(set(words))
    for w in words:
        assert w in s
    for miss in ("", "definitely-not-there", "word200000", "a" * 64):
        assert miss not in s


def test_empty_and_small():
    s = FlatHashSet64([])
    assert len(s) == 0 and "x" not in s
    s = FlatHashSet64(["x"])
    assert "x" in s and "y" not in s


def test_memory_ratio():
    import sys as _sys
    words = _words(300_000)
    flat_bytes = len(FlatHashSet64(words).table) * 8
    set_bytes = sum(_sys.getsizeof(w) + 64 for w in set(words))
    assert flat_bytes < set_bytes // 3
