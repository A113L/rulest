import numpy as np


def test_initial_upper_bound_pass_emits_exact_progress_bar_and_rules_rate(monkeypatch):
    import rulest.selection as selection

    events = []

    class FakeHeartbeat:
        def __init__(self, desc, interval=0.5, **kwargs):
            self.desc = str(desc)
            self.interval = interval
            self.kwargs = kwargs
            events.append(self.desc)

        def __enter__(self):
            events.append("HEARTBEAT START")
            return self

        def set_desc(self, desc, force=False):
            self.desc = str(desc)
            events.append(self.desc)

        def __exit__(self, exc_type, exc, tb):
            events.append(self.desc + " | done")

    class FakeScorer:
        def __init__(self):
            self.calls = []

        def upper_bound_batch(self, rules):
            self.calls.append(tuple(rules))
            return [len(rule) + 1 for rule in rules]

    monkeypatch.setattr(selection.sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(selection, "ActivityHeartbeat", FakeHeartbeat)
    scorer = FakeScorer()
    rules = ["A", "BB", "CCC", "DDDD", "EEEEE"]

    bounds = selection._run_initial_upper_bound_pass(scorer, rules, rule_batch_size=2)

    assert bounds == [2, 3, 4, 5, 6]
    assert len(scorer.calls) == 3
    upper_lines = [line for line in events if "[CELF] Upper-bound pass" in line]
    assert any("scanned=2/5" in line and "rules/s" in line for line in upper_lines)
    assert any("scanned=4/5" in line and "ETA=" in line for line in upper_lines)
    assert any("scanned=5/5" in line and "100%" in line for line in upper_lines)
    assert any("[" in line and "█" in line and "scanned=" in line for line in upper_lines)


def test_greedy_heartbeat_reports_selected_rate_and_lazy_upper_bounds(monkeypatch):
    import rulest.selection as selection

    events = []

    class FakeHeartbeat:
        def __init__(self, desc, interval=0.5, **kwargs):
            self.desc = str(desc)
            events.append(self.desc)

        def __enter__(self):
            return self

        def set_desc(self, desc, force=False):
            self.desc = str(desc)
            events.append(self.desc)

        def __exit__(self, exc_type, exc, tb):
            events.append(self.desc + " | done")

    monkeypatch.setattr(selection.sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(selection, "ActivityHeartbeat", FakeHeartbeat)

    target_sets = [{0, 1}, {1, 2}, {2, 3}, {3}]
    multiplicities = [{t: 2 for t in ids} for ids in target_sets]
    last_exact = {}

    def score_one(index, covered):
        covered_ids = set(np.flatnonzero(covered))
        return target_sets[index] - covered_ids

    def score_bounds(indices, covered):
        covered_ids = set(np.flatnonzero(covered))
        return [sum(1 for t in target_sets[i] if t not in covered_ids) for i in indices]

    def score_exact(indices, covered):
        covered_ids = set(np.flatnonzero(covered))
        last_exact.clear()
        last_exact.update({i: target_sets[i] - covered_ids for i in indices})
        return [len(last_exact[i]) for i in indices]

    selected, covered = selection._celf_select_from_upper_bounds(
        ["A", "B", "C", "D"],
        [4, 3, 3, 2],
        4,
        score_one,
        budget=2,
        score_batch_fn=score_bounds,
        score_exact_batch_fn=score_exact,
        rescore_batch_size=3,
    )

    assert selected
    assert int(covered.sum()) == sum(gain for _rule, gain in selected)
    greedy_lines = [line for line in events if "[CELF] Greedy selection" in line]
    assert greedy_lines
    assert any("rules/s" in line and "(avg=" in line for line in greedy_lines)
    assert any("exact_scans=" in line and "exact_record_scans=" in line for line in greedy_lines)
    assert any("lazy_upper_bound_candidates=" in line for line in greedy_lines)
    assert any("lazy_upper_bound_speed=" in line and "rules/s" in line for line in greedy_lines)
