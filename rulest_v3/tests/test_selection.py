import pytest

import numpy as np
from rulest.selection import (
    _TargetFingerprintIndex,
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

        def score_one_with_targets(self, rule, covered_mask=None, progress_callback=None):
            return 1, {0}

        def score_gains_batch(self, rules, progress_callback=None):
            return [1] * len(rules)

        def score_exact_gains_batch(self, rules, progress_callback=None):
            # Exercised by the seed-pop's cheap exact-gain-only check (see
            # exact_gain() in selection.py), which now runs before any
            # expensive recording scan, even for a single-candidate round.
            return [1] * len(rules)

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
    """Contenders that can still win are exact-scored as a batch, while only the
    seed and final winner need target-ID recording scans."""
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

    def score_one(i, covered):
        singleton_calls.append(i)
        return sets[i] - set(np.flatnonzero(covered))

    def score_bounds(indices, covered):
        covered_ids = set(np.flatnonzero(covered))
        return [sum(v for t, v in multiplicity[i].items() if t not in covered_ids) for i in indices]

    def score_exact_batch(indices, covered):
        exact_batches.append(tuple(indices))
        covered_ids = set(np.flatnonzero(covered))
        return [len(sets[i] - covered_ids) for i in indices]

    selected, covered = _celf_select_from_upper_bounds(
        rules, [20, 10, 12, 4, 2], 7, score_one, budget=4,
        score_batch_fn=score_bounds, score_exact_batch_fn=score_exact_batch,
        rescore_batch_size=3,
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
    # A batch may need a recording pass when it displaces the seed, but not one
    # full singleton score per competing candidate.
    assert len(singleton_calls) < 2 * len(selected) + 1


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


def test_immediate_winner_costs_a_single_one_candidate_seed_call():
    """Regression test for the "200 rules/s ceiling, phase ~5 minutes
    longer" symptom: on an ORDINARY (non-duplicate-heavy) pool, where the
    single top-of-heap candidate is already each round's true winner, each
    round must cost exactly one `score_exact_batch_fn` call scoring exactly
    ONE candidate -- not a full `rescore_batch_size`-sized call.

    An earlier version of the seed-batching optimization always probed a
    full `rescore_batch_size` candidates every round regardless of whether
    the top candidate alone would have won immediately. That is a pure win
    on a duplicate-heavy pool (see the dead-run test below) but a pure loss
    here: it turned one cheap 1-candidate GPU dispatch per round into a
    wasted `rescore_batch_size`-candidate one on every round of an otherwise
    healthy run, which is exactly what the reported slowdown and 200
    rules/s ceiling match.
    """
    # Five candidates with distinct, non-overlapping true coverage and raw
    # upper bounds that already rank them in true-gain order -- i.e. no
    # duplicate-inflation at all, so the heap's top candidate is always
    # immediately correct.
    rules = [f"r{i}" for i in range(5)]
    true_sets = [{10 * i} for i in range(5)]
    upper_bounds = [50, 40, 30, 20, 10]

    exact_batch_calls = []
    singleton_calls = []

    def score_one(i, covered):
        singleton_calls.append(i)
        return true_sets[i] - set(np.flatnonzero(covered))

    def score_bounds(indices, covered):
        return [upper_bounds[i] for i in indices]

    def score_exact_batch(indices, covered):
        exact_batch_calls.append(tuple(indices))
        covered_ids = set(np.flatnonzero(covered))
        return [len(true_sets[i] - covered_ids) for i in indices]

    selected, covered = _celf_select_from_upper_bounds(
        rules, upper_bounds, 41, score_one, budget=None,
        score_batch_fn=score_bounds, score_exact_batch_fn=score_exact_batch,
        rescore_batch_size=32,  # a large configured batch ceiling
    )

    assert [name for name, _ in selected] == rules
    # Round 1's seed probe is a single candidate (the point of this test).
    # The remaining four candidates' raw bounds (40/30/20/10) still look
    # like they could beat that round's winner (gain 1), so the PRE-EXISTING
    # stale-bound-refresh path (unrelated to seed probing, already present
    # before this fix) legitimately tightens and exact-scores all of them
    # together in that same round -- a real, useful batched call, not
    # wasted work. After that, every candidate's bound is already exact, so
    # every later round's seed probe goes back to costing exactly one call.
    assert exact_batch_calls == [(0,), (1, 2, 3, 4), (1,), (2,), (3,), (4,)], exact_batch_calls
    # Crucially: never a 32-candidate probe. Only the first round pays for
    # more than one candidate, and that one call is the legitimate
    # stale-bound-refresh pass, not a seed over-fetch.
    assert all(len(call) <= 4 for call in exact_batch_calls)


def test_long_dead_run_is_cleared_in_logarithmically_many_probes():
    """Companion to the immediate-winner test above: a long run of dead
    (zero true gain) candidates at the top of the heap must still be
    cleared efficiently, via a doubling probe size, rather than reverting
    to one GPU call per dead candidate.
    """
    # Eight candidates with inflated (duplicate-heavy) raw upper bounds, all
    # of which happen to have zero true gain, followed by one real winner
    # with a much lower raw bound -- a long dead run at the top of the heap.
    rules = [f"dead{i}" for i in range(8)] + ["winner"]
    true_sets = [set() for _ in range(8)] + [{0}]
    upper_bounds = [100 - 10 * i for i in range(8)] + [1]

    exact_batch_calls = []
    singleton_calls = []

    def score_one(i, covered):
        singleton_calls.append(i)
        return true_sets[i] - set(np.flatnonzero(covered))

    def score_bounds(indices, covered):
        return [upper_bounds[i] for i in indices]

    def score_exact_batch(indices, covered):
        exact_batch_calls.append(tuple(indices))
        covered_ids = set(np.flatnonzero(covered))
        return [len(true_sets[i] - covered_ids) for i in indices]

    selected, covered = _celf_select_from_upper_bounds(
        rules, upper_bounds, 1, score_one, budget=None,
        score_batch_fn=score_bounds, score_exact_batch_fn=score_exact_batch,
        rescore_batch_size=8,
    )

    assert selected == [("winner", 1)]
    assert int(covered.sum()) == 1
    # Doubling probe (1, 2, 4, then whatever remains) instead of either 8
    # single-candidate calls or one fixed 8-candidate call every time.
    assert exact_batch_calls == [(0,), (1, 2), (3, 4, 5, 6), (7, 8)], exact_batch_calls
    assert len(exact_batch_calls) < 8, "must clear the dead run in fewer than one call per dead candidate"
    # Still exactly one recording scan: the real winner's.
    assert singleton_calls == [8]


def test_batched_exact_rescoring_matches_reference_on_randomized_coverage():
    import random

    for seed in range(35):
        rng = random.Random(seed + 1000)
        n_rules, n_targets = 11, 23
        rules = [f"R{i}" for i in range(n_rules)]
        sets = [{t for t in range(n_targets) if rng.random() < 0.3} for _ in rules]
        multiplicity = [{t: rng.randint(1, 7) for t in sets[i]} for i in range(n_rules)]
        initial_bounds = [sum(multiplicity[i].values()) + (n_rules - i) for i in range(n_rules)]

        def score_one(i, covered, _sets=sets):
            return _sets[i] - set(np.flatnonzero(covered))

        def score_bounds(indices, covered, _sets=sets, _mult=multiplicity):
            covered_ids = set(np.flatnonzero(covered))
            return [sum(_mult[i][t] for t in _sets[i] if t not in covered_ids) for i in indices]

        def score_exact(indices, covered, _sets=sets):
            covered_ids = set(np.flatnonzero(covered))
            return [len(_sets[i] - covered_ids) for i in indices]

        selected, covered = _celf_select_from_upper_bounds(
            rules, initial_bounds, n_targets, score_one, budget=8,
            score_batch_fn=score_bounds, score_exact_batch_fn=score_exact,
            rescore_batch_size=4,
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
