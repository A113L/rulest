"""Regression tests for CELF calls forwarded through a process-isolated GPU."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rulest.gpu_worker import ProcessEngineProxy


def test_process_proxy_exposes_celf_methods_and_workspace_metadata(monkeypatch):
    proxy = ProcessEngineProxy({"MAX_CHAIN_DEPTH": 4}, "0")
    calls = []

    def fake_call(method, *args, **kwargs):
        calls.append((method, args, kwargs))
        if method == "__getattr__":
            return {"_celf_seen_words": 17, "_celf_batch_capacity": 23}[args[0]]
        if method == "celf_score_chains_gpu":
            return ([5] * len(args[0]), [])
        if method == "celf_target_ids_for_slots":
            return {2, 4}
        if method == "celf_profile_summary":
            return "profile-summary"
        return None

    monkeypatch.setattr(proxy, "_call", fake_call)
    assert proxy._celf_seen_words == 17
    assert proxy._celf_batch_capacity == 23

    proxy.celf_initialize_target_table(["target"])
    proxy.celf_prepare_candidate_pool(["rule"])
    callback = lambda done, total: None
    assert proxy.celf_score_chains_gpu(["rule"], record_hits=True,
                                       progress_callback=callback) == ([5], [])
    assert proxy.celf_target_ids_for_slots([10, 11]) == {2, 4}
    proxy.celf_clear_target_ids({2, 4})
    proxy.celf_release_target_table()
    proxy.celf_report_profile()
    assert proxy.celf_profile_summary() == "profile-summary"

    score_call = next(call for call in calls if call[0] == "celf_score_chains_gpu")
    assert score_call[1] == (["rule"],)
    assert score_call[2] == {"record_hits": True, "_raise_on_error": True}
    assert all("progress_callback" not in kwargs for _method, _args, kwargs in calls)

    assert proxy.celf_score_chains_gpu(["rule-a", "rule-b"], exact_counts=True) == ([5, 5], [])
    exact_call = [call for call in calls if call[0] == "celf_score_chains_gpu"][-1]
    assert exact_call[2] == {"record_hits": False, "_raise_on_error": True, "exact_counts": True}


def test_strict_rpc_surfaces_worker_errors_instead_of_swallowing_them():
    import pytest

    proxy = ProcessEngineProxy({}, "0")

    class FakeQueue:
        def put(self, item):
            self.sent = item

        def get(self, timeout):
            call_id = proxy._call_id
            return (call_id, "err", "simulated CELF setup failure")

    class FakeProcess:
        def is_alive(self):
            return True

    proxy._proc = FakeProcess()
    proxy._cmd_q = FakeQueue()
    proxy._res_q = FakeQueue()
    proxy._last_heartbeat = 1e20  # prevent heartbeat timeout during this unit test
    with pytest.raises(RuntimeError, match="simulated CELF setup failure"):
        proxy._call("celf_initialize_target_table", [], _raise_on_error=True)
