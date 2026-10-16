import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest.multi_gpu import MultiGPUEngine


class FakeEngine:
    def __init__(self, idx):
        self.idx = idx
        self.device = SimpleNamespace(name=f"GPU-{idx}")
        self.disabled = False
        self.calls = []

    def verify_pairs_exact(self, base_words, selected_chains, pairs, batch_size=200000):
        self.calls.append(list(pairs))
        # Encode the pair so the test can verify exact global ordering.
        return [f"out:{c}:{w}:gpu{self.idx}" for c, w in pairs]


def test_verify_pairs_exact_uses_all_gpus_and_preserves_order():
    m = MultiGPUEngine({})
    m.engines = [FakeEngine(0), FakeEngine(1), FakeEngine(2)]
    pairs = [(i % 4, i) for i in range(30)]

    # Fixed equal split makes this deterministic and exercises all workers.
    result = m.verify_pairs_exact(["a"] * 30, ["x", "y", "z", "w"], pairs)

    assert len(result) == len(pairs)
    assert all(result[i] == f"out:{c}:{w}:gpu{result[i].split("gpu")[-1]}" for i, (c, w) in enumerate(pairs))
    assert all(engine.calls for engine in m.engines)

    flattened = [pair for engine in m.engines for call in engine.calls for pair in call]
    assert sorted(flattened) == sorted(pairs)


def test_verify_pairs_exact_empty_is_noop():
    m = MultiGPUEngine({})
    m.engines = [FakeEngine(0), FakeEngine(1)]
    assert m.verify_pairs_exact([], [], []) == []
    assert all(not e.calls for e in m.engines)


if __name__ == "__main__":
    import pytest
    print(f"Running {__file__} directly — use `pytest {__file__}` for full output.\n")
    raise SystemExit(pytest.main([__file__, "-v"]))
