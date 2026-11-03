from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def reset_stage_state():
    with app.LOCK:
        app.STATE.update({
            "status": "running",
            "stage": "Starting engine",
            "stage_detail": "Launching the engine process…",
            "stage_started_at": None,
            "progress": None,
            "paused": False,
            "pause_requested": False,
        })


def test_startup_enabled_disabled_summary_does_not_claim_active_phase():
    assert app.detect_active_stage("STAGE 0 : enabled — CPU exact-match") is None
    assert app.detect_active_stage("STAGE 1 : enabled — single-rule GPU sweep") is None
    assert app.detect_active_stage("STAGE 3 : disabled — genetic algorithm") is None


def test_real_phase_section_headings_are_detected():
    assert app.detect_active_stage("│ STAGE 0 — TOKEN-STRIP RULE EXTRACTION") == "Stage 0 · Token Strip"
    assert app.detect_active_stage("│ STAGE S — SEED EXTRACTION") == "Stage S · Built-in Seeds"
    assert app.detect_active_stage("│ STAGE 1 — SINGLE RULE SEARCH") == "Stage 1 · Single-rule GPU sweep"
    assert app.detect_active_stage("│ STAGE 2 — RULE CHAIN SEARCH") == "Stage 2 · Rule-chain search"
    assert app.detect_active_stage("│ STAGE 3 — GENETIC ALGORITHM RULE EVOLUTION") == "Stage 3 · Genetic Algorithm"


def test_live_subphase_markers_are_detected():
    assert app.detect_active_stage("[GPU] Building bloom filter on GPU …") == "GPU Setup · Bloom Filter"
    assert app.detect_active_stage("[CELF] GPU recompute | recovered=2,000") == "Rule Selection · CELF"
    assert app.detect_active_stage("POST-PROCESSING — Signature-Based Functional Minimization") == "Minimization"
    assert app.detect_active_stage("[MINIMIZE] Candidates : 20,000") == "Minimization"
    assert app.detect_active_stage("[S3] pop=200 max_gen=50 elite=15%") == "Stage 3 · Genetic Algorithm"


def test_minimization_done_message_is_not_misread_as_run_completion():
    assert app.detect_active_stage("[MINIMIZE] Done") == "Minimization"
    assert app.detect_active_stage("DONE  rulest finished in 32.1s") == "Completed"


def test_progress_percent_updates_active_phase_and_detail():
    reset_stage_state()
    line = "  STAGE 2:  42%|██████████ | 42/100 [00:10<00:12, 4.2 batch/s]"
    app.update_stage(line)
    assert app.STATE["stage"] == "Stage 2 · Rule-chain search"
    assert app.STATE["progress"] == 42
    assert "42%" in app.STATE["stage_detail"]
    assert app.STATE["stage_started_at"] is not None


def test_celf_selection_heartbeat_updates_the_live_phase_detail():
    reset_stage_state()
    line = (
        "\x1b[96m[CELF] Greedy selection | selected=5/50 (10.0%) "
        "| recovered=2,000/12,000 (16.7%) | heap=40,000 "
        "| exact_scans=12 | elapsed=32s\x1b[0m"
    )
    app.update_stage(line)
    assert app.STATE["stage"] == "Rule Selection · CELF"
    assert app.STATE["stage_detail"].startswith("[CELF] Greedy selection")
    assert "selected=5/50" in app.STATE["stage_detail"]


def test_progress_count_can_be_used_when_percent_is_absent():
    reset_stage_state()
    app.update_stage("[CELF] GPU recompute | 25/100 [00:20<01:00]")
    assert app.STATE["stage"] == "Rule Selection · CELF"
    assert app.STATE["progress"] == 25


def test_terminal_cleaner_preserves_full_long_line_and_removes_ansi():
    long_text = "Important counter: " + ("1234567890" * 80)
    raw = "\x1b[96m" + long_text + "\x1b[0m\x1b[K"
    assert app.clean_terminal_text(raw) == long_text


def test_gui_has_wrap_expand_and_live_phase_detail_controls():
    web = Path(app.__file__).resolve().parent / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    js = (web / "app.js").read_text(encoding="utf-8")
    css = (web / "styles.css").read_text(encoding="utf-8")
    assert 'id="current-stage-detail"' in html
    assert 'id="wrap-log"' in html
    assert 'id="expand-log"' in html
    assert 'id="terminal-window"' in html
    assert "current-stage-detail" in js
    assert "terminal-window').classList.toggle('expanded')" in js
    assert ".log-message" in css and ".terminal-window.expanded" in css
