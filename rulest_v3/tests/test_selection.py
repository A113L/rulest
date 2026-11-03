import pytest

import numpy as np
from rulest.selection import (
    _TargetFingerprintIndex,
    _GpuExactScorer,
    _celf_select_from_upper_bounds,
    _fingerprint64,
    _splitmix64,
    _resolve_single_engine,
)


class _FakeSingleEngine:
    """Stands in for a plain GPUEngine/ProcessEngineProxy: no `.engines`
    attribute at all, so _resolve_single_engine must pass it through
    unchanged."""
    pass


class _FakeMultiGPUEngine:
    """Stands in for MultiGPUEngine: duck-typed by having an `.engines`
    list, the only thing _resolve_single_engine actually looks at."""
    def __init__(self, engines):
        self.engines = engines


class _FakeDevice:
    def __init__(self, name):
        self.name = name


class _FakeSubEngine:
    def __init__(self, name):
        self.device = _FakeDevice(name)


def test_resolve_single_engine_passes_through_a_plain_engine_unchanged():
    eng = _FakeSingleEngine()
    assert _resolve_single_engine(eng) is eng


def test_resolve_single_engine_unwraps_multi_gpu_to_its_first_engine():
    primary = _FakeSubEngine("NVIDIA GeForce RTX 3060 Ti")
    other = _FakeSubEngine("Intel(R) UHD Graphics 630")
    multi = _FakeMultiGPUEngine([primary, other])
    assert _resolve_single_engine(multi) is primary


def test_resolve_single_engine_raises_a_clear_error_when_no_gpu_survived():
    # This is the scenario from the field crash: every device became
    # unresponsive during an earlier stage and was pruned, leaving
    # MultiGPUEngine.engines empty. Previously this fell through to
    # AttributeError: 'MultiGPUEngine' object has no attribute
    # 'set_base_words' deep inside _GpuExactScorer.__init__ instead of a
    # clear, actionable message right here.
    multi = _FakeMultiGPUEngine([])
    with pytest.raises(RuntimeError, match="No GPU device is available"):
        _resolve_single_engine(multi)


def test_fingerprint_collision_requires_exact_string_match():
    index = _TargetFingerprintIndex(["alpha", "beta"], fingerprint_fn=lambda _: 1)
    assert index.lookup("alpha") == 0
    assert index.lookup("beta") == 1
    assert index.lookup("gamma") is None


def test_celf_emits_live_selection_heartbeat_when_stderr_is_not_a_tty(monkeypatch):
    import rulest.selection as selection

    events = []

    class FakeHeartbeat:
        def __init__(self, desc, interval=3.0, **kwargs):
            self.desc = desc
            self.interval = interval
            events.append(desc)

        def __enter__(self):
            return self

        def set_desc(self, desc, force=False):
            self.desc = desc
            events.append(desc)

        def __exit__(self, exc_type, exc, tb):
            events.append(self.desc + " | done")

    monkeypatch.setattr(selection.sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(selection, "ActivityHeartbeat", FakeHeartbeat)

    rules = ["A", "B", "C"]
    gains = [{0, 1}, {1, 2}, {2}]
    selected, covered = _celf_select_from_upper_bounds(
        rules, [2, 2, 1], 3,
        lambda index, current: gains[index] - set(np.flatnonzero(current)),
    )

    assert selected == [("A", 2), ("B", 1)]
    assert int(covered.sum()) == 3
    assert any("[CELF] Greedy selection" in event for event in events)
    assert any("selected=0/3" in event for event in events)
    assert any("selected=1/3" in event for event in events)
    assert any("recovered=2/3" in event for event in events)


def test_basic_greedy_coverage_and_accounting():
    rules = ["A", "B", "C"]
    sets = [{0, 1}, {1, 2}, {2}]
    selected, covered = _celf_select_from_upper_bounds(
        rules, [2, 2, 1], 3,
        lambda i, covered: sets[i] - set(np.flatnonzero(covered)),
    )
    assert selected == [("A", 2), ("B", 1)]
    assert int(covered.sum()) == 3


def test_batch_scores_are_exact_distinct_target_marginals():
    rules = ["A", "B", "C", "D"]
    sets = [{0, 1, 2, 5}, {0, 1, 2, 3, 4}, {0, 1, 3}, {4, 5}]
    calls = []
    def score_one(i, covered):
        return sets[i] - set(np.flatnonzero(covered))
    def score_bounds(indices, covered):
        calls.append(tuple(indices))
        # GPU scratch bitmap: each distinct active target counts once per rule.
        return [len(sets[i] - set(np.flatnonzero(covered))) for i in indices]
    selected, covered = _celf_select_from_upper_bounds(
        rules, [4, 5, 3, 2], 6, score_one, budget=2,
        score_batch_fn=score_bounds, rescore_batch_size=2,
    )
    assert selected == [("B", 5), ("A", 1)]
    assert int(covered.sum()) == 6
    assert calls
    assert all(1 <= len(batch) <= 2 for batch in calls)


def test_exact_batch_winner_ids_are_materialized_without_a_second_score_scan():
    rules = ["A", "B", "C"]
    sets = [{0, 1, 2}, {0, 1, 3}, {3, 4}]
    last_exact_batch = {}
    materialized = []

    def exact_batch(indices, covered):
        already = set(np.flatnonzero(covered))
        last_exact_batch.clear()
        last_exact_batch.update({i: sets[i] - already for i in indices})
        return [len(last_exact_batch[i]) for i in indices]

    def materialize(index):
        materialized.append(index)
        return set(last_exact_batch[index])

    def score_one(_index, _covered):
        raise AssertionError("winner should be recovered from the exact batch bitmap, not rescored")

    selected, covered = _celf_select_from_upper_bounds(
        rules,
        [3, 3, 2],
        5,
        score_one,
        budget=2,
        rescore_batch_size=3,
        score_exact_batch_fn=exact_batch,
        materialize_exact_ids_fn=materialize,
    )

    assert selected == [("A", 3), ("C", 2)]
    assert set(np.flatnonzero(covered)) == {0, 1, 2, 3, 4}
    assert materialized == [0, 2]


def test_gpu_scorer_prefers_compact_candidate_index_api():
    class IndexEngine:
        def __init__(self):
            self.calls = []

        def set_base_words(self, _words):
            return None

        def celf_initialize_target_table(self, _targets):
            return None

        def celf_prepare_candidate_pool(self, _rules):
            return None

        def celf_score_candidate_indices_gpu(self, indices, record_hits=False, exact_counts=False):
            self.calls.append((tuple(indices), record_hits, exact_counts))
            return ([10 + int(index) for index in indices], [])

        def celf_seen_target_ids_for_candidate(self, candidate_index):
            return {int(candidate_index), int(candidate_index) + 100}

        def celf_release_target_table(self):
            return None

    engine = IndexEngine()
    scorer = _GpuExactScorer(engine, ["base"], ["target"], 100, ["A", "B", "C"])
    assert scorer.upper_bound_index_batch([0, 2]) == [10, 12]
    assert scorer.score_gains_index_batch([1, 2]) == [11, 12]
    assert scorer.score_exact_gains_index_batch([2, 0]) == [12, 10]
    assert scorer.materialize_exact_targets(2) == {2, 102}
    assert engine.calls == [
        ((0, 2), False, False),
        ((1, 2), False, False),
        ((2, 0), False, True),
    ]


def test_ties_are_deterministic():
    rules = ["first", "second"]
    sets = [{0}, {1}]
    selected, _ = _celf_select_from_upper_bounds(
        rules, [1, 1], 2,
        lambda i, covered: sets[i] - set(np.flatnonzero(covered)), budget=1,
    )
    assert selected == [("first", 1)]


def test_cost_alpha_adjusts_priority_but_reports_raw_gain():
    rules = ["long chain", "short"]
    sets = [{0, 1, 2}, {3, 4}]
    selected, covered = _celf_select_from_upper_bounds(
        rules, [3, 2], 5,
        lambda i, covered: sets[i] - set(np.flatnonzero(covered)),
        budget=1, cost_alpha=1.0,
    )
    assert selected == [("short", 2)]
    assert int(covered.sum()) == 2


def test_splitmix64_is_deterministic_and_mixes_keys():
    assert _splitmix64(1) == _splitmix64(1)
    assert _splitmix64(1) != _splitmix64(2)
    assert _splitmix64(0) != _splitmix64(1 << 32)
    assert _fingerprint64("hello") != _fingerprint64("hello!")


def test_ranker_heap_prunes_batch_that_cannot_beat_exact_seed():
    rules = ["A", "B", "C", "D"]
    sets = [{0, 1, 2}, {3, 4, 5, 6}, {3, 4, 5}, {7}]
    calls = []
    def score_one(i, covered):
        return sets[i] - set(np.flatnonzero(covered))
    def score_batch(indices, covered):
        calls.append(tuple(indices))
        return [len(sets[i] - set(np.flatnonzero(covered))) for i in indices]
    selected, covered = _celf_select_from_upper_bounds(
        rules, [3, 4, 3, 1], 8, score_one, budget=2,
        score_batch_fn=score_batch, rescore_batch_size=2,
    )
    assert selected == [("B", 4), ("A", 3)]
    assert calls == []
    assert int(covered.sum()) == 7


def test_raw_upper_bounds_may_overcount_but_exact_winner_stays_correct():
    rules = ["A", "B", "C", "D"]
    sets = [{0, 1, 2, 3}, {0, 1, 4}, {2, 3, 4, 5}, {0, 5}]
    multiplicity = [
        {0: 5, 1: 3, 2: 2, 3: 4},
        {0: 4, 1: 2, 4: 7},
        {2: 3, 3: 2, 4: 6, 5: 4},
        {0: 8, 5: 3},
    ]
    initial_bounds = [sum(multiplicity[i][t] for t in sets[i]) for i in range(len(rules))]

    def score_one(i, covered):
        return sets[i] - set(np.flatnonzero(covered))

    def score_batch(indices, covered):
        return [sum(multiplicity[i][t] for t in sets[i] if not covered[t]) for i in indices]

    selected, covered = _celf_select_from_upper_bounds(
        rules, initial_bounds, 6, score_one, budget=4,
        score_batch_fn=score_batch, rescore_batch_size=2,
    )
    expected = []
    covered_ref = set()
    while len(expected) < 4:
        choices = [(len(sets[i] - covered_ref), i) for i in range(len(rules)) if rules[i] not in {r for r, _ in expected}]
        gain, idx = max(choices, key=lambda item: (item[0], -item[1]))
        if gain <= 0:
            break
        expected.append((rules[idx], gain))
        covered_ref |= sets[idx]
    assert selected == expected
    assert set(np.flatnonzero(covered)) == covered_ref


def test_randomized_lazy_heap_matches_exact_greedy_under_duplicate_overcount_bounds():
    import random

    for seed in range(60):
        rng = random.Random(seed)
        n_rules, n_targets = 9, 17
        rules = [f"R{i}" for i in range(n_rules)]
        sets = [{t for t in range(n_targets) if rng.random() < 0.32} for _ in rules]
        multiplicity = [{t: rng.randint(1, 6) for t in sets[i]} for i in range(n_rules)]
        initial = [sum(multiplicity[i].values()) for i in range(n_rules)]

        def score_one(i, covered, _sets=sets):
            return _sets[i] - set(np.flatnonzero(covered))

        def score_batch(indices, covered, _sets=sets, _multiplicity=multiplicity):
            return [sum(_multiplicity[i][t] for t in _sets[i] if not covered[t]) for i in indices]

        selected, covered = _celf_select_from_upper_bounds(
            rules, initial, n_targets, score_one, budget=7,
            score_batch_fn=score_batch, rescore_batch_size=4,
        )
        expected = []
        covered_ref = set()
        remaining = set(range(n_rules))
        while remaining and len(expected) < 7:
            idx = min(remaining, key=lambda i: (-len(sets[i] - covered_ref), i))
            gain = len(sets[idx] - covered_ref)
            if gain <= 0:
                break
            expected.append((rules[idx], gain))
            covered_ref |= sets[idx]
            remaining.remove(idx)
        assert selected == expected, f"seed={seed}: {selected} != {expected}"
        assert set(np.flatnonzero(covered)) == covered_ref


def test_budgetless_run_gets_approximate_total_for_eta(monkeypatch):
    """With no --budget, total used to be None -- progress.py deliberately
    omits percent/total/ETA for an indeterminate bar (see progress.py's
    "No known stopping point" branch), so the CELF bar never showed an ETA
    at all for an unbounded run. len(positive-bound candidates) is a true
    upper bound on how many rules a run can ever select (every winner comes
    from a heap entry, and the heap is only ever seeded/repushed from that
    same set), so passing it as `total` lets progress.py compute a (loose,
    optimistic) ETA instead of none."""
    import rulest.selection as sel

    captured = {}
    real_tqdm = sel.tqdm

    def spy_tqdm(*args, **kwargs):
        captured["total"] = kwargs.get("total")
        captured["desc"] = kwargs.get("desc")
        return real_tqdm(*args, **{**kwargs, "disable": True})

    monkeypatch.setattr(sel, "tqdm", spy_tqdm)

    rules = ["A", "B", "C", "D"]
    sets = [{0, 1}, {1, 2}, {2}, set()]  # "D" has upper_bound 0 -> never in heap
    upper_bounds = [2, 2, 1, 0]
    selected, covered = _celf_select_from_upper_bounds(
        rules, upper_bounds, 3,
        lambda i, covered: sets[i] - set(np.flatnonzero(covered)),
        budget=None,
    )
    # 3 of the 4 rules have a positive upper bound ("D" does not), so that's
    # the approximate total -- not None, and not len(rules).
    assert captured["total"] == 3
    assert "approx" in captured["desc"]


def test_budgeted_run_uses_the_real_budget_as_total(monkeypatch):
    """A caller-supplied --budget is an exact, known stopping point, so it
    must still win outright over the approximate positive-bound count, and
    the bar's description must not get the "(approx)" caveat appended."""
    import rulest.selection as sel

    captured = {}
    real_tqdm = sel.tqdm

    def spy_tqdm(*args, **kwargs):
        captured["total"] = kwargs.get("total")
        captured["desc"] = kwargs.get("desc")
        return real_tqdm(*args, **{**kwargs, "disable": True})

    monkeypatch.setattr(sel, "tqdm", spy_tqdm)

    rules = ["A", "B", "C"]
    sets = [{0, 1}, {1, 2}, {2}]
    selected, covered = _celf_select_from_upper_bounds(
        rules, [2, 2, 1], 3,
        lambda i, covered: sets[i] - set(np.flatnonzero(covered)),
        budget=1,
    )
    assert captured["total"] == 1
    assert "approx" not in captured["desc"]


def test_dedupe_targets_no_longer_spams_raw_hashmap_prints(capsys):
    """The sort+compact dedupe pass used to print a brand new `[HASHMAP]
    ...` line via bare print() at least once a second; it now reports
    through the same single-line ActivityHeartbeat the rest of this stage
    uses (see progress.py's `set_desc`), which writes `\\r`-overwritten
    status to the stream it was given, not scrolling stdout lines captured
    by capsys."""
    from rulest.selection import _rulest_dedupe_targets_with_progress

    words = [f"word{i % 37}" for i in range(500)]
    result = _rulest_dedupe_targets_with_progress(list(words))

    assert sorted(result) == sorted(set(words))
    out = capsys.readouterr().out
    assert "[HASHMAP]" not in out
    assert "print" not in out  # sanity: nothing scrolling landed on stdout


def test_dedupe_targets_handles_empty_input():
    from rulest.selection import _rulest_dedupe_targets_with_progress
    assert _rulest_dedupe_targets_with_progress([]) == []


def test_greedy_selection_reads_celf_state_from_the_selected_single_gpu(monkeypatch):
    """The MultiGPUEngine itself has no CELF workspace fields; its primary
    ProcessEngineProxy does. Selection must use that same primary proxy for
    depth handling, workspace sizing, scoring, and cleanup."""
    import rulest.selection as selection_module

    class PrimaryEngine:
        def __init__(self):
            self.device = _FakeDevice("NVIDIA test GPU")
            self.params = {
                "MAX_CHAIN_DEPTH": 8,
                "CELF_SEEN_SCRATCH_MB": 4,
                "CELF_RULE_BATCH_SIZE": 4,
            }
            self._celf_seen_words = 2
            self._celf_batch_capacity = 1

        def compile_kernel(self, force=False):
            return True

        def celf_report_profile(self):
            return None

        def celf_profile_summary(self):
            return None

    primary = PrimaryEngine()
    secondary = _FakeSubEngine("second GPU")

    class FakeMultiEngine:
        def __init__(self):
            self.engines = [primary, secondary]

    class FakeScorer:
        def __init__(self, gpu_engine, base_words, target_words, pair_batch_size, candidate_rules):
            self.gpu_engine = gpu_engine

        def upper_bound_batch(self, rules):
            return [1] * len(rules)

        def upper_bound_index_batch(self, indices):
            return [1] * len(list(indices))

        def score_one_with_targets(self, rule, covered_mask=None, progress_callback=None):
            return 1, {0}

        def score_gains_batch(self, rules, progress_callback=None):
            return [1] * len(rules)

        def score_gains_index_batch(self, indices, progress_callback=None):
            return [1] * len(list(indices))

        def score_exact_gains_batch(self, rules, progress_callback=None):
            # Exercised by the seed-pop's cheap exact-gain-only check (see
            # exact_gain() in selection.py), which now runs before any
            # expensive recording scan, even for a single-candidate round.
            return [1] * len(rules)

        def score_exact_gains_index_batch(self, indices, progress_callback=None):
            return [1] * len(list(indices))

        def materialize_exact_targets(self, candidate_index):
            assert candidate_index == 0
            return {0}

        def mark_selected(self, target_ids):
            return None

        def close(self):
            return None

    monkeypatch.setattr(selection_module, "_GpuExactScorer", FakeScorer)
    selected, covered = selection_module.select_rules_by_marginal_coverage(
        FakeMultiEngine(), ["base"], ["target"], ["l"], 8, budget=1
    )
    assert selected == [("l", 1)]
    assert int(covered.sum()) == 1


def test_batched_exact_rescoring_preserves_greedy_results_and_reduces_singleton_calls():
    """Exact-batch winner bitmaps remove the redundant final singleton scan."""
    rules = ["A", "B", "C", "D", "E"]
    sets = [
        {0},
        {1, 2},
        {1, 2, 3},
        {3, 4},
        {5, 6},
    ]
    multiplicity = [
        {0: 10},
        {1: 5, 2: 5},
        {1: 4, 2: 4, 3: 4},
        {3: 2, 4: 2},
        {5: 1, 6: 1},
    ]
    singleton_calls = []
    exact_batches = []
    latest_exact_ids = {}

    def score_one(i, covered):
        singleton_calls.append(i)
        return sets[i] - set(np.flatnonzero(covered))

    def score_bounds(indices, covered):
        covered_ids = set(np.flatnonzero(covered))
        return [sum(v for t, v in multiplicity[i].items() if t not in covered_ids) for i in indices]

    def score_exact_batch(indices, covered):
        exact_batches.append(tuple(indices))
        covered_ids = set(np.flatnonzero(covered))
        latest_exact_ids.clear()
        latest_exact_ids.update({i: sets[i] - covered_ids for i in indices})
        return [len(latest_exact_ids[i]) for i in indices]

    def materialize_exact_ids(index):
        return set(latest_exact_ids[index])

    selected, covered = _celf_select_from_upper_bounds(
        rules, [20, 10, 12, 4, 2], 7, score_one, budget=4,
        score_batch_fn=score_bounds, score_exact_batch_fn=score_exact_batch,
        rescore_batch_size=3, materialize_exact_ids_fn=materialize_exact_ids,
    )

    expected = []
    remaining = set(range(len(rules)))
    covered_ref = set()
    while remaining and len(expected) < 4:
        idx = min(remaining, key=lambda i: (-len(sets[i] - covered_ref), i))
        gain = len(sets[idx] - covered_ref)
        if gain <= 0:
            break
        expected.append((rules[idx], gain))
        covered_ref |= sets[idx]
        remaining.remove(idx)

    assert selected == expected
    assert set(np.flatnonzero(covered)) == covered_ref
    assert exact_batches, "the optimized exact-batch path should be exercised"
    # The exact bitmap from the batch supplies winner IDs; no full winner
    # recording scan is needed.
    assert singleton_calls == []


def test_dead_seeds_use_the_cheap_exact_batch_path_not_a_recording_scan():
    """Regression test for the "selected=0 while exact scans climb into the
    thousands" symptom: when `score_exact_batch_fn` is available, a losing
    round seed (and a losing "fresh_this_round" re-check) must be resolved
    through the cheap batched-exact kernel, never through the full
    recording-scan `score_fn`/`score_one` path. Before this fix, the seed
    and the fresh-contender re-check always called the expensive singleton
    recording path directly, so a heap full of duplicate-inflated raw upper
    bounds that mostly resolve to zero exact gain burned thousands of full
    GPU wordlist scans with hit recording before anything was ever selected.
    """
    # Five candidates whose raw (duplicate-inflated) upper bounds all look
    # promising, but whose TRUE exact marginal gain is zero for four of
    # them and small for the fifth -- i.e. a duplicate-heavy pool exactly
    # like the one described in the bug report.
    rules = ["dead1", "dead2", "dead3", "dead4", "winner"]
    true_sets = [set(), set(), set(), set(), {0}]
    upper_bounds = [50, 40, 30, 20, 1]

    singleton_calls = []
    exact_batch_calls = []

    def score_one(i, covered):
        # The expensive path: must be called at most once, for the actual
        # winner's recording scan.
        singleton_calls.append(i)
        return true_sets[i] - set(np.flatnonzero(covered))

    def score_bounds(indices, covered):
        # Deliberately return the raw (inflated) bounds unchanged, so every
        # dead candidate still looks like it could beat the eventual winner
        # after cheap tightening, forcing the exact-batch path to resolve
        # each one.
        return [upper_bounds[i] for i in indices]

    def score_exact_batch(indices, covered):
        exact_batch_calls.append(tuple(indices))
        covered_ids = set(np.flatnonzero(covered))
        return [len(true_sets[i] - covered_ids) for i in indices]

    selected, covered = _celf_select_from_upper_bounds(
        rules, upper_bounds, 1, score_one, budget=None,
        score_batch_fn=score_bounds, score_exact_batch_fn=score_exact_batch,
        rescore_batch_size=4,
    )

    assert selected == [("winner", 1)]
    assert int(covered.sum()) == 1
    assert exact_batch_calls, "dead/losing candidates must be resolved via the cheap exact-batch path"
    # Exactly one recording scan: the real winner's. None of the four dead
    # seeds/contenders should ever reach the expensive singleton path.
    assert singleton_calls == [4]


def test_batched_exact_rescoring_matches_reference_on_randomized_coverage():
    import random

    for seed in range(35):
        rng = random.Random(seed + 1000)
        n_rules, n_targets = 11, 23
        rules = [f"R{i}" for i in range(n_rules)]
        sets = [{t for t in range(n_targets) if rng.random() < 0.3} for _ in rules]
        multiplicity = [{t: rng.randint(1, 7) for t in sets[i]} for i in range(n_rules)]
        initial_bounds = [sum(multiplicity[i].values()) + (n_rules - i) for i in range(n_rules)]
        latest_exact_ids = {}

        def score_one(i, covered, _sets=sets):
            return _sets[i] - set(np.flatnonzero(covered))

        def score_bounds(indices, covered, _sets=sets, _mult=multiplicity):
            covered_ids = set(np.flatnonzero(covered))
            return [sum(_mult[i][t] for t in _sets[i] if t not in covered_ids) for i in indices]

        def score_exact(indices, covered, _sets=sets):
            covered_ids = set(np.flatnonzero(covered))
            latest_exact_ids.clear()
            latest_exact_ids.update({i: _sets[i] - covered_ids for i in indices})
            return [len(latest_exact_ids[i]) for i in indices]

        def materialize_exact_ids(index):
            return set(latest_exact_ids[index])

        selected, covered = _celf_select_from_upper_bounds(
            rules, initial_bounds, n_targets, score_one, budget=8,
            score_batch_fn=score_bounds, score_exact_batch_fn=score_exact,
            rescore_batch_size=4, materialize_exact_ids_fn=materialize_exact_ids,
        )

        expected = []
        covered_ref = set()
        remaining = set(range(n_rules))
        while remaining and len(expected) < 8:
            idx = min(remaining, key=lambda i: (-len(sets[i] - covered_ref), i))
            gain = len(sets[idx] - covered_ref)
            if gain <= 0:
                break
            expected.append((rules[idx], gain))
            covered_ref |= sets[idx]
            remaining.remove(idx)

        assert selected == expected, f"seed={seed}: {selected} != {expected}"
        assert set(np.flatnonzero(covered)) == covered_ref


def test_gpu_exact_batch_mismatch_restarts_on_safe_singleton_path(monkeypatch):
    """A GPU batch/record disagreement must restart, never keep suspect gains."""
    import rulest.selection as selection_module

    class FakeEngine:
        def __init__(self):
            self.params = {"CELF_RULE_BATCH_SIZE": 2, "CELF_SEEN_SCRATCH_MB": 4}
            self._celf_seen_words = 1
            self._celf_batch_capacity = 2
            self.device = None
            self.reset_count = 0

        def celf_reset_active_targets(self):
            self.reset_count += 1

        def celf_report_profile(self):
            return None

        def celf_profile_summary(self):
            return None

        def celf_release_target_table(self):
            return None

    class FakeScorer:
        def __init__(self, gpu_engine, base_words, target_words, pair_batch_size, candidate_rules):
            self.gpu_engine = gpu_engine
            self.bounds = {"A": 10, "B": 9}
            self.exact_batch_calls = 0

        def upper_bound_batch(self, rules):
            return [self.bounds[r] for r in rules]

        def score_gains_batch(self, rules, progress_callback=None):
            return [self.bounds[r] for r in rules]

        def score_exact_gains_batch(self, rules, progress_callback=None):
            self.exact_batch_calls += 1
            # Simulate the reported hardware discrepancy on B: speculative exact
            # batch says 2, while its recording scan confirms only 1 target.
            return [2 if r == "B" else 1 for r in rules]

        def score_one_with_targets(self, rule, covered_mask=None, progress_callback=None):
            return 1, ({"A": {0}, "B": {1}}[rule])

        def mark_selected(self, target_ids):
            return None

        def close(self):
            return None

    engine = FakeEngine()
    monkeypatch.setattr(selection_module, "_GpuExactScorer", FakeScorer)
    selected, covered = selection_module.select_rules_by_marginal_coverage(
        engine, ["base"], ["target0", "target1"], ["A", "B"],
        max_depth=8, budget=1,
    )

    assert selected == [("A", 1)]
    assert int(covered.sum()) == 1
    assert engine.reset_count == 1


def test_celf_coalesces_small_exact_challenger_groups_across_bound_batches(monkeypatch):
    """Avoid one-candidate exact GPU launches when adjacent bound batches
    each leave only a small number of plausible challengers.
    """
    rules = [f"R{i}" for i in range(20)]
    target_sets = {
        0: set(range(10)),
        1: {10}, 2: {11}, 3: {12},
        4: set(range(13, 22)),
        8: set(range(22, 30)),
        12: set(range(30, 37)),
        16: set(range(37, 43)),
    }
    latest_ids = {}
    exact_batches = []
    bound_batches = []

    def exact_batch(indices, covered):
        covered_ids = set(np.flatnonzero(covered))
        latest_ids.clear()
        latest_ids.update({int(i): target_sets.get(int(i), set()) - covered_ids for i in indices})
        exact_batches.append(tuple(int(i) for i in indices))
        return [len(latest_ids[int(i)]) for i in indices]

    def materialize(index):
        return set(latest_ids[int(index)])

    def score_one(index, covered):
        return target_sets.get(int(index), set()) - set(np.flatnonzero(covered))

    def score_bounds(indices, covered):
        bound_batches.append(tuple(int(i) for i in indices))
        # Exactly one candidate in each 4-entry stale batch survives the cheap
        # upper-bound pass. The survivors are all smaller than R0's exact gain.
        winners = {4, 8, 12, 16}
        return [20 if int(i) in winners else 0 for i in indices]

    monkeypatch.setenv("RULEST_CELF_EXACT_MICROBATCH", "4")
    selected, covered = _celf_select_from_upper_bounds(
        rules,
        [100 - i for i in range(len(rules))],
        50,
        score_one,
        budget=1,
        score_batch_fn=score_bounds,
        score_exact_batch_fn=exact_batch,
        materialize_exact_ids_fn=materialize,
        rescore_batch_size=4,
    )

    assert selected == [("R0", 10)]
    assert int(covered.sum()) == 10
    assert bound_batches == [(4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19)]
    assert exact_batches == [(0, 1, 2, 3), (4, 8, 12, 16)]


def test_celf_exact_microbatch_preserves_cost_alpha_greedy_order(monkeypatch):
    import random

    monkeypatch.setenv("RULEST_CELF_EXACT_MICROBATCH", "4")
    for seed in range(12):
        rng = random.Random(7000 + seed)
        n_rules, n_targets = 13, 29
        rules = [f"R{i}" + (" x" * (i % 4)) for i in range(n_rules)]
        sets = [{t for t in range(n_targets) if rng.random() < 0.32} for _ in rules]
        multiplicity = [{t: rng.randint(1, 6) for t in sets[i]} for i in range(n_rules)]
        bounds = [sum(multiplicity[i].values()) + n_rules - i for i in range(n_rules)]
        latest_ids = {}
        cost_alpha = 0.7

        def score_one(index, covered):
            return sets[index] - set(np.flatnonzero(covered))

        def score_bounds(indices, covered):
            covered_ids = set(np.flatnonzero(covered))
            return [
                sum(value for target, value in multiplicity[index].items() if target not in covered_ids)
                for index in indices
            ]

        def score_exact(indices, covered):
            covered_ids = set(np.flatnonzero(covered))
            latest_ids.clear()
            latest_ids.update({index: sets[index] - covered_ids for index in indices})
            return [len(latest_ids[index]) for index in indices]

        def materialize(index):
            return set(latest_ids[index])

        selected, covered = _celf_select_from_upper_bounds(
            rules, bounds, n_targets, score_one, budget=9, cost_alpha=cost_alpha,
            score_batch_fn=score_bounds, score_exact_batch_fn=score_exact,
            materialize_exact_ids_fn=materialize, rescore_batch_size=4,
        )

        ref_selected = []
        ref_covered = set()
        remaining = set(range(n_rules))
        while remaining and len(ref_selected) < 9:
            def ref_key(index):
                gain = len(sets[index] - ref_covered)
                cost = max(1, len(rules[index].split())) ** cost_alpha
                return (-(gain / cost), index)
            winner = min(remaining, key=ref_key)
            gain = len(sets[winner] - ref_covered)
            if gain <= 0:
                break
            ref_selected.append((rules[winner], gain))
            ref_covered.update(sets[winner])
            remaining.remove(winner)

        assert selected == ref_selected, f"seed={seed}: {selected} != {ref_selected}"
        assert set(np.flatnonzero(covered)) == ref_covered
