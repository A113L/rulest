from pathlib import Path


def test_celf_kernel_uses_pre_workgroup_atomic_gain_path():
    root = Path(__file__).resolve().parents[1]
    kernel_source = (root / "rulest" / "kernel_source.py").read_text(encoding="utf-8")
    start = kernel_source.index("__kernel void celf_score_chains_gpu(")
    end = kernel_source.index("__kernel void celf_clear_active_slots(", start)
    kernel = kernel_source[start:end]

    # The experimental local reduction added barriers and a per-group local
    # atomic. Roll back to the previously used one-dimensional work-item layout.
    assert "int gid = (int)get_global_id(0);" in kernel
    assert "int ci = gid / nw;" in kernel
    assert "atomic_inc((volatile __global int *)(gains + ci));" in kernel
    assert "__local int group_gain;" not in kernel
    assert "barrier(CLK_LOCAL_MEM_FENCE);" not in kernel


def test_celf_score_kernel_launch_uses_flat_word_candidate_ndrange():
    root = Path(__file__).resolve().parents[1]
    engine_source = (root / "rulest" / "gpu_engine.py").read_text(encoding="utf-8")
    assert "total = num_words * n_rules" in engine_source
    assert "(global_size,), (self.local_work_size,)" in engine_source
    assert "(global_words, n_rules), (self.local_work_size, 1)" not in engine_source


def test_celf_web_heartbeat_reports_rules_per_second_and_is_responsive():
    root = Path(__file__).resolve().parents[1]
    selection_source = (root / "rulest" / "selection.py").read_text(encoding="utf-8")
    assert 'speed={last_selected_rate[0]:,.2f} rules/s' in selection_source
    assert '(avg={average_selected_rate:,.2f} rules/s) | recovered=' in selection_source
    assert 'os.environ.get("RULEST_CELF_LOG_INTERVAL", "0.5")' in selection_source
    assert 'unit="rules"' in selection_source
