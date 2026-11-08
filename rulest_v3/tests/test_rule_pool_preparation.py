"""
Regression tests for GPUEngine and MultiGPUEngine rule-pool preparation.

Bug recap: gpu_engine.verify_pairs_exact() intentionally narrows
gpu_rules/rule_index down to just the atomic tokens of whatever chains it
was last called with (see its docstring in gpu_engine.py). The batch loop
in evaluate_full_coverage() calls verify_pairs_exact() once per batch,
*after* evaluate_population() for that same batch. On the next batch,
evaluate_population() -> _compute_chain_seqs() looks tokens up in
rule_index.get(r, -1); any token that wasn't part of the *previous*
batch's narrowed set silently resolves to -1. The GPU chain kernel still
appends that token's separator space to the found-string buffer before
noticing the -1 and bailing out, so it reports a truncated, space-padded
chain string (e.g. "^t ") that no longer matches the original chain
string used as the dict key in genetic.py's hit_words — raising
KeyError('^t ') deep inside a multi-hour GPU run.

The fix is GPUEngine.prepare_rule_pool()/MultiGPUEngine.prepare_rule_pool(),
called once per batch in selection.py right before evaluate_population().
These current engine-facing checks do not require a real GPU.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest.gpu_engine import GPUEngine
from rulest.multi_gpu import MultiGPUEngine
from rulest import selection


class TestGPUEnginePrepareRulePool:
    def test_rebuilds_rule_index_from_chain_tokens(self):
        eng = GPUEngine({})
        eng.prepare_rule_pool(["^t X", "l"])
        assert set(eng.rule_index.keys()) == {"^t", "X", "l"}
        assert set(eng.gpu_rules) == {"^t", "X", "l"}

    def test_second_call_replaces_not_merges(self):
        """This is the exact shape of the bug: a later, narrower call must
        not leave stale tokens from an earlier call lying around, and must
        not silently keep only their intersection either — it fully
        replaces the index to match the chains it's given right now."""
        eng = GPUEngine({})
        eng.prepare_rule_pool(["^t X", "l"])
        eng.prepare_rule_pool(["u c"])
        assert set(eng.rule_index.keys()) == {"u", "c"}
        assert "^t" not in eng.rule_index
        assert "X" not in eng.rule_index
        assert "l" not in eng.rule_index

    def test_rule_index_values_are_dense_indices(self):
        eng = GPUEngine({})
        eng.prepare_rule_pool(["b a"])
        # atomic tokens are sorted, so indices are deterministic
        assert eng.rule_index == {"a": 0, "b": 1}
        assert eng.gpu_rules == ["a", "b"]

    def test_covers_every_token_across_a_full_batch(self):
        """The actual failure mode: evaluate_full_coverage() must hand the
        WHOLE batch's chains to prepare_rule_pool(), not just the ones with
        bloom hits, so evaluate_population()'s _compute_chain_seqs() never
        sees a -1 for any token appearing anywhere in the batch."""
        eng = GPUEngine({})
        chunk = ["^t", "^t X", "l u c", "3 2 s"]
        eng.prepare_rule_pool(chunk)
        expected_tokens = {tok for c in chunk for tok in c.split()}
        assert set(eng.rule_index.keys()) == expected_tokens
        for c in chunk:
            for tok in c.split():
                assert eng.rule_index[tok] >= 0


class FakeRulePoolEngine:
    def __init__(self, idx):
        self.idx = idx
        self.device = SimpleNamespace(name=f"GPU-{idx}")
        self.disabled = False
        self.rule_index = {}
        self.calls = []

    def prepare_rule_pool(self, chains):
        self.calls.append(list(chains))
        self.rule_index = {tok: i for i, tok in
                            enumerate(sorted({t for c in chains for t in c.split()}))}
        return True


class TestMultiGPUEnginePrepareRulePool:
    def test_broadcasts_to_every_active_engine(self):
        m = MultiGPUEngine({})
        m.engines = [FakeRulePoolEngine(0), FakeRulePoolEngine(1), FakeRulePoolEngine(2)]
        chunk = ["^t X", "l u"]

        ok = m.prepare_rule_pool(chunk)

        assert ok is True
        for eng in m.engines:
            assert eng.calls == [chunk]
            assert set(eng.rule_index.keys()) == {"^t", "X", "l", "u"}

    def test_skips_disabled_engines(self):
        m = MultiGPUEngine({})
        dead = FakeRulePoolEngine(0)
        dead.disabled = True
        alive = FakeRulePoolEngine(1)
        m.engines = [dead, alive]

        m.prepare_rule_pool(["a b"])

        assert dead.calls == []
        assert alive.calls == [["a b"]]

    def test_every_engine_gets_the_full_batch_not_a_shard(self):
        """evaluate_population() may later shard this same batch's chains
        across engines by weighted_split(); prepare_rule_pool() must give
        every engine the FULL batch's token universe up front, since any
        engine could end up evaluating any chain in the batch."""
        m = MultiGPUEngine({})
        m.engines = [FakeRulePoolEngine(0), FakeRulePoolEngine(1)]
        chunk = [f"r{i} s{i}" for i in range(10)]

        m.prepare_rule_pool(chunk)

        full_universe = {tok for c in chunk for tok in c.split()}
        for eng in m.engines:
            assert set(eng.rule_index.keys()) == full_universe


