"""
Regression tests for the stale-rule_index bug in the coverage-evaluation
batch loop (selection.py::evaluate_full_coverage).

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
These tests cover the two engine-facing pieces without needing a real GPU.
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


class _FakeEvolverNoRulePoolBypass:
    """Stand-in for GeneticRuleEvolver that fails exactly like the real GPU
    kernel did: it raises the same KeyError-shaped symptom whenever a
    chain's tokens aren't (yet) present in gpu_engine.rule_index, instead
    of silently succeeding. This lets the test assert on *why* a batch
    would have failed without the fix, using the same failure signature
    ("<token> " with a stray trailing space) as the real bug, rather than
    re-deriving the GPU kernel's byte-level corruption behavior."""

    def __init__(self, gpu_engine, base_words, rule_pool, max_depth):
        self.gpu_engine = gpu_engine
        self.base_words = base_words
        self._hit_cache = {}
        self.active_words = base_words

    def evaluate_population(self, population):
        hit_words = {}
        for tok in population:
            cs = ' '.join(tok)
            missing = [t for t in tok if t not in self.gpu_engine.rule_index]
            if missing:
                # Mirrors the real GPU kernel's corrupted-string symptom:
                # the chain gets truncated right after the last token that
                # *was* resolved, with a trailing separator space, and that
                # string doesn't match any key genetic.py expects.
                good = tok[:tok.index(missing[0])]
                raise KeyError(' '.join(good) + ' ')
            hit_words[cs] = {0}
        return hit_words


class TestEvaluateFullCoverageResyncsRulePoolPerBatch:
    """End-to-end (minus the real GPU) reproduction of the original crash:
    batch 2 introduces tokens that weren't part of batch 1, and batch 1's
    verify_pairs_exact() narrowed rule_index down to just batch 1's tokens.
    Without evaluate_full_coverage() calling prepare_rule_pool(chunk) before
    evaluate_population() each batch, this raises KeyError('b ') exactly
    like the production crash. With the fix, it doesn't."""

    def _fake_verify_pairs_exact(self, gpu_engine):
        def _verify(base_words, selected_chains, pairs, batch_size=200000):
            # Mirrors the real verify_pairs_exact(): narrows rule_index to
            # exactly this call's chains, as a side effect, before returning.
            gpu_engine.prepare_rule_pool(selected_chains)
            return ["" for _ in pairs]
        return _verify

    def test_batch_two_sees_its_own_tokens_not_stale_ones(self, monkeypatch):
        monkeypatch.setattr(selection, "GeneticRuleEvolver", _FakeEvolverNoRulePoolBypass)

        gpu_engine = GPUEngine({})
        gpu_engine.verify_pairs_exact = self._fake_verify_pairs_exact(gpu_engine)

        # After filtering/sorting by depth in evaluate_full_coverage:
        # depth-1 rules ("a", "d") come first, then depth-2 ("b c", "e f") —
        # batch size 2 puts them in separate batches with disjoint tokens.
        rules = ["a", "d", "b c", "e f"]
        base_words = ["w0"]
        target_words = ["dummy_target"]

        # Must not raise — this is the actual regression assertion.
        coverage = selection.evaluate_full_coverage(
            gpu_engine, base_words, target_words, rules, max_depth=2, batch_size=2
        )
        assert set(coverage.keys()) == {"a", "d", "b c", "e f"}

    def test_reproduces_original_crash_without_the_fix(self, monkeypatch):
        """Sanity-check the reproduction itself: if evaluate_full_coverage
        skips the prepare_rule_pool() resync (simulating the pre-fix code),
        batch 2 must fail with the same KeyError shape as production."""
        monkeypatch.setattr(selection, "GeneticRuleEvolver", _FakeEvolverNoRulePoolBypass)

        gpu_engine = GPUEngine({})
        gpu_engine.verify_pairs_exact = self._fake_verify_pairs_exact(gpu_engine)
        # Pre-seed rule_index exactly as it would be after batch 1's own
        # (successful) narrowing, then neutralize prepare_rule_pool so
        # nothing resyncs it before/after that — reproducing "no fix".
        gpu_engine.rule_index = {"a": 0, "d": 1}
        gpu_engine.gpu_rules = ["a", "d"]
        monkeypatch.setattr(gpu_engine, "prepare_rule_pool", lambda chains: True)

        rules = ["a", "d", "b c", "e f"]
        base_words = ["w0"]
        target_words = ["dummy_target"]

        import pytest
        with pytest.raises(KeyError):
            selection.evaluate_full_coverage(
                gpu_engine, base_words, target_words, rules, max_depth=2, batch_size=2
            )


class TestEvaluateFullCoverageExcludesOverLengthCandidates:
    """Regression test for the second production crash:
    KeyError("] $3 $0 $1 $0 $2 $0 $3") — a candidate longer than max_depth
    got silently truncated by the GPU kernel to its first max_depth tokens,
    and that truncated prefix string didn't match the full candidate string
    used as the hit_words dict key. evaluate_full_coverage() must exclude
    over-length candidates up front instead of letting them reach the GPU
    at all."""

    def test_over_length_candidates_never_reach_the_evolver(self, monkeypatch):
        seen_populations = []

        class _RecordingEvolver:
            def __init__(self, gpu_engine, base_words, rule_pool, max_depth):
                self._hit_cache = {}
                self.active_words = base_words

            def evaluate_population(self, population):
                seen_populations.append([tuple(t) for t in population])
                return {' '.join(tok): set() for tok in population}

        monkeypatch.setattr(selection, "GeneticRuleEvolver", _RecordingEvolver)

        gpu_engine = GPUEngine({})
        gpu_engine.params = {'MAX_CHAIN_DEPTH': 8}
        gpu_engine.verify_pairs_exact = lambda *a, **k: []

        short_rule = "a b c"                       # 3 tokens, fine
        exact_rule = "a b c d e f g h"              # 8 tokens, fine (== max_depth)
        over_rule = "a b c d e f g h i"             # 9 tokens, exceeds max_depth
        rules = [short_rule, exact_rule, over_rule]

        coverage = selection.evaluate_full_coverage(
            gpu_engine, ["w0"], ["dummy_target"], rules, max_depth=8, batch_size=10
        )

        all_seen_chains = {' '.join(tok) for pop in seen_populations for tok in pop}
        assert over_rule not in all_seen_chains
        assert short_rule in all_seen_chains
        assert exact_rule in all_seen_chains
        # Excluded candidates never make it into the returned coverage dict.
        assert over_rule not in coverage

    def test_uses_engine_reported_max_chain_depth_when_stricter(self, monkeypatch):
        """If the caller passes a larger max_depth than what the compiled
        kernel actually supports (gpu_engine.params['MAX_CHAIN_DEPTH']), the
        kernel's real limit wins — that's the buffer size that actually
        matters on GPU."""
        seen = []

        class _RecordingEvolver:
            def __init__(self, gpu_engine, base_words, rule_pool, max_depth):
                self._hit_cache = {}
                self.active_words = base_words

            def evaluate_population(self, population):
                seen.extend(' '.join(tok) for tok in population)
                return {' '.join(tok): set() for tok in population}

        monkeypatch.setattr(selection, "GeneticRuleEvolver", _RecordingEvolver)

        gpu_engine = GPUEngine({})
        gpu_engine.params = {'MAX_CHAIN_DEPTH': 4}  # kernel only supports depth 4
        gpu_engine.verify_pairs_exact = lambda *a, **k: []

        five_token_rule = "a b c d e"
        selection.evaluate_full_coverage(
            gpu_engine, ["w0"], ["dummy_target"], [five_token_rule],
            max_depth=8,  # caller asks for 8, but the kernel is only compiled for 4
            batch_size=10,
        )
        assert five_token_rule not in seen


class TestEvalOnEngineSurvivesUnrecognizedKeys:
    """Regression test at the genetic.py boundary: even if some future edge
    case still produces a GPU-reported chain string that isn't one of the
    requested chains, _eval_on_engine() must warn and discard rather than
    crash the whole run with KeyError."""

    def test_discards_unknown_key_without_raising(self):
        from rulest.genetic import GeneticRuleEvolver

        class _FakeEngine:
            def evaluate_chains_for_ga(self, chains):
                return {
                    "a b": {0, 1},
                    "] $3 $0 $1 $0 $2 $0 $3": {2},  # not one of `chains`
                }

        evolver = GeneticRuleEvolver.__new__(GeneticRuleEvolver)
        hit_words = {"a b": set(), "c d": set()}

        # Must not raise.
        evolver._eval_on_engine(_FakeEngine(), ["a b", "c d"], hit_words)

        assert hit_words["a b"] == {0, 1}
        assert hit_words["c d"] == set()

    def test_known_keys_still_merge_normally(self):
        from rulest.genetic import GeneticRuleEvolver

        class _FakeEngine:
            def evaluate_chains_for_ga(self, chains):
                return {"a b": {5}, "c d": {6, 7}}

        evolver = GeneticRuleEvolver.__new__(GeneticRuleEvolver)
        hit_words = {"a b": {1}, "c d": set()}

        evolver._eval_on_engine(_FakeEngine(), ["a b", "c d"], hit_words)

        assert hit_words["a b"] == {1, 5}
        assert hit_words["c d"] == {6, 7}


if __name__ == "__main__":
    import pytest
    print(f"Running {__file__} directly — use `pytest {__file__}` for full output.\n")
    raise SystemExit(pytest.main([__file__, "-v"]))
