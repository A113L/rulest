import numpy as np
from rulest.selection import (
    _TargetFingerprintIndex,
    _celf_select_from_upper_bounds,
    _fingerprint64,
    _splitmix64,
)


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
