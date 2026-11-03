import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from rulest.common import KeyboardController


def test_build_command_maps_all_cli_options(tmp_path, monkeypatch):
    uploads = {
        "base": {"path": tmp_path / "base.txt", "name": "base.txt", "size": 10},
        "target": {"path": tmp_path / "target.txt", "name": "target.txt", "size": 10},
        "seed": {"path": tmp_path / "seeds.rule", "name": "seeds.rule", "size": 5},
    }
    monkeypatch.setattr(app, "UPLOADS_META", uploads)
    options = {
        "output_name": "rules.txt", "device": "0,1", "max_depth": 4,
        "target_hours": 1.5, "max_chains": 200, "bloom_mb": 512,
        "bloom_no_shard": True, "allow_reject_rules": True, "no_builtin_seeds": True,
        "token_strip": True, "token_strip_min_stem": 5, "token_strip_max_prefix": 2,
        "token_strip_max_suffix": 3, "token_strip_min_leet_amb": 4,
        "token_strip_workers": 6, "token_strip_chunk_size": 10000,
        "genetic": True, "genetic_generations": 70, "genetic_pop": 300,
        "genetic_elite": 0.2, "select_mode": "greedy", "select_budget": 800,
        "select_budgets": "100,500", "select_cost_alpha": 0.5,
        "exact_recovery": True, "debug": True, "profile": True,
        "depth2_chains": 12, "depth10_chains": 99,
    }
    command, output = app.build_command(options, tmp_path)
    assert output == "rules.txt"
    expected = [
        "--device", "0,1", "--max-depth", "4", "--target-hours", "1.5",
        "--max-chains", "200", "--bloom-mb", "512", "--seed-rules",
        str(uploads["seed"]["path"]), "--depth2-chains", "12", "--depth10-chains", "99",
        "--select-mode", "greedy", "--select-budget", "800", "--select-budgets", "100,500",
    ]
    for token in expected:
        assert token in command
    for flag in [
        "--bloom-no-shard", "--allow-reject-rules", "--no-builtin-seeds", "--token-strip",
        "--genetic", "--exact-recovery", "--debug", "--profile",
    ]:
        assert flag in command


def test_output_control_filename_is_reserved(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOADS_META", {
        "base": {"path": tmp_path / "base", "name": "base", "size": 1},
        "target": {"path": tmp_path / "target", "name": "target", "size": 1},
        "seed": None,
    })
    try:
        app.build_command({"output_name": "control.txt"}, tmp_path)
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("control.txt must be reserved")


def test_keyboard_controller_accepts_external_pause_resume_quit(tmp_path, monkeypatch):
    control_file = tmp_path / "control.txt"
    control_file.write_text("", encoding="ascii")
    monkeypatch.setenv("RULEST_CONTROL_FILE", str(control_file))
    controller = KeyboardController()
    controller.start()
    try:
        with control_file.open("a", encoding="ascii") as stream:
            stream.write("p\n")
        deadline = time.time() + 2
        while time.time() < deadline and not controller._paused:
            time.sleep(0.02)
        assert controller._paused

        with control_file.open("a", encoding="ascii") as stream:
            stream.write("r\n")
        deadline = time.time() + 2
        while time.time() < deadline and controller._paused:
            time.sleep(0.02)
        assert not controller._paused

        with control_file.open("a", encoding="ascii") as stream:
            stream.write("q\n")
        deadline = time.time() + 2
        while time.time() < deadline and not controller.quit_requested:
            time.sleep(0.02)
        assert controller.quit_requested
    finally:
        controller.stop()


def test_optional_stage_switches_are_explicit_in_the_gui():
    web_dir = Path(app.__file__).resolve().parent / "web"
    html = (web_dir / "index.html").read_text(encoding="utf-8")
    css = (web_dir / "styles.css").read_text(encoding="utf-8")
    js = (web_dir / "app.js").read_text(encoding="utf-8")

    assert 'id="token_strip"' in html
    assert 'aria-label="Enable Stage 0 Token Strip"' in html
    assert "Enable Stage 0" in html
    assert 'id="genetic"' in html
    assert 'aria-label="Enable Genetic Algorithm"' in html
    assert "Enable Genetic" in html
    assert 'id="token-state"' in html and 'id="genetic-state"' in html
    assert ".stage-enable-toggle input:checked" in css
    assert "state.dataset.enabled = String(enabled)" in js
    assert "Genetic Algorithm requires Maximum depth" in js


def test_build_command_omits_token_strip_and_genetic_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOADS_META", {
        "base": {"path": tmp_path / "base.txt", "name": "base.txt", "size": 1},
        "target": {"path": tmp_path / "target.txt", "name": "target.txt", "size": 1},
        "seed": None,
    })
    command, _ = app.build_command({
        "output_name": "rules.txt", "token_strip": False, "genetic": False,
    }, tmp_path)
    assert "--token-strip" not in command
    assert "--genetic" not in command


def test_live_activity_navigation_targets_the_live_log_panel():
    web_dir = Path(app.__file__).resolve().parent / "web"
    html = (web_dir / "index.html").read_text(encoding="utf-8")
    css = (web_dir / "styles.css").read_text(encoding="utf-8")
    js = (web_dir / "app.js").read_text(encoding="utf-8")

    assert 'href="#live-log"' in html
    assert 'id="live-log"' in html
    assert 'class="panel log-panel" id="live-log"' in html
    assert "target.scrollIntoView({ behavior: 'smooth', block: 'start' })" in js
    assert "window.addEventListener('popstate'" in js
    assert "#live-log { scroll-margin-top: 18px; }" in css
    assert "auth-password-1" in html


def test_cleanup_uploaded_files_removes_inputs_and_keeps_run_outputs(tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    base = uploads_dir / "base.txt"
    target = uploads_dir / "target.txt"
    seed = uploads_dir / "seed.rule"
    for path in (base, target, seed):
        path.write_text("temporary input", encoding="utf-8")
    run_dir = tmp_path / "run-123"
    run_dir.mkdir()
    output = run_dir / "rules.txt"
    output.write_text("generated rules", encoding="utf-8")

    monkeypatch.setattr(app, "UPLOADS", uploads_dir)
    monkeypatch.setattr(app, "UPLOADS_META", {
        "base": {"path": base, "name": "base.txt", "size": 15},
        "target": {"path": target, "name": "target.txt", "size": 15},
        "seed": {"path": seed, "name": "seed.rule", "size": 15},
    })
    monkeypatch.setattr(app, "UPLOADS_ACTIVE", set())

    removed = app.cleanup_uploaded_files(log_message=False)

    assert len(removed) == 3
    assert not base.exists() and not target.exists() and not seed.exists()
    assert all(value is None for value in app.UPLOADS_META.values())
    assert output.read_text(encoding="utf-8") == "generated rules"


def test_run_worker_cleans_uploaded_files_when_engine_cannot_start(tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    base = uploads_dir / "base.txt"
    target = uploads_dir / "target.txt"
    base.write_text("base", encoding="utf-8")
    target.write_text("target", encoding="utf-8")
    run_dir = tmp_path / "run-456"
    run_dir.mkdir()
    control_file = run_dir / "control.txt"
    control_file.write_text("", encoding="ascii")

    monkeypatch.setattr(app, "UPLOADS", uploads_dir)
    monkeypatch.setattr(app, "UPLOADS_META", {
        "base": {"path": base, "name": "base.txt", "size": 4},
        "target": {"path": target, "name": "target.txt", "size": 6},
        "seed": None,
    })
    monkeypatch.setattr(app, "UPLOADS_ACTIVE", set())
    monkeypatch.setattr(app, "PROCESS", None)
    monkeypatch.setattr(app, "RUN_THREAD", None)
    monkeypatch.setattr(app, "STATE", {
        "status": "running", "paused": False, "pause_requested": False,
        "pid": None, "stage": "Starting engine", "stage_detail": "Launching",
        "stage_started_at": None, "progress": None, "exit_code": None,
        "logs": [], "log_seq": 0, "outputs": [], "error": None,
        "control_file": str(control_file), "run_dir": str(run_dir),
        "job_id": "run-456", "started_at": time.time(), "finished_at": None,
    })

    def fail_to_start(*args, **kwargs):
        raise OSError("simulated engine launch failure")

    monkeypatch.setattr(app.subprocess, "Popen", fail_to_start)
    app.run_worker(["python", "missing.py"], "run-456", run_dir, control_file)

    assert not base.exists() and not target.exists()
    assert all(value is None for value in app.UPLOADS_META.values())
    assert app.STATE["status"] == "failed"
    assert "simulated engine launch failure" in app.STATE["error"]


def test_upload_ui_clears_cards_when_server_reports_temporary_files_removed():
    web_dir = Path(app.__file__).resolve().parent / "web"
    js = (web_dir / "app.js").read_text(encoding="utf-8")
    html = (web_dir / "index.html").read_text(encoding="utf-8")

    assert "function clearUploadField(field)" in js
    assert "!upload && !uploadInProgress[field] && uploadState[field]" in js
    assert "Temporary server copies are automatically deleted" in html


def test_run_worker_cleans_uploaded_files_after_normal_completion(tmp_path, monkeypatch):
    import io

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    base = uploads_dir / "base.txt"
    target = uploads_dir / "target.txt"
    base.write_text("base", encoding="utf-8")
    target.write_text("target", encoding="utf-8")
    run_dir = tmp_path / "run-789"
    run_dir.mkdir()
    control_file = run_dir / "control.txt"
    control_file.write_text("", encoding="ascii")
    output = run_dir / "rules.txt"
    output.write_text("rule data", encoding="utf-8")

    monkeypatch.setattr(app, "UPLOADS", uploads_dir)
    monkeypatch.setattr(app, "UPLOADS_META", {
        "base": {"path": base, "name": "base.txt", "size": 4},
        "target": {"path": target, "name": "target.txt", "size": 6},
        "seed": None,
    })
    monkeypatch.setattr(app, "UPLOADS_ACTIVE", set())
    monkeypatch.setattr(app, "PROCESS", None)
    monkeypatch.setattr(app, "RUN_THREAD", None)
    monkeypatch.setattr(app, "STATE", {
        "status": "running", "paused": False, "pause_requested": False,
        "pid": None, "stage": "Starting engine", "stage_detail": "Launching",
        "stage_started_at": None, "progress": None, "exit_code": None,
        "logs": [], "log_seq": 0, "outputs": [], "error": None,
        "control_file": str(control_file), "run_dir": str(run_dir),
        "job_id": "run-789", "started_at": time.time(), "finished_at": None,
    })

    class FakeProcess:
        pid = 12345
        stdout = io.BytesIO(b"RULEST FINISHED\n")
        def wait(self):
            return 0

    monkeypatch.setattr(app.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    app.run_worker(["python", "run_rulest.py"], "run-789", run_dir, control_file)

    assert not base.exists() and not target.exists()
    assert all(value is None for value in app.UPLOADS_META.values())
    assert app.STATE["status"] == "completed"
    assert {item["name"] for item in app.STATE["outputs"]} >= {"rules.txt", "engine.log"}


def test_cleanup_refuses_to_delete_while_upload_is_active(tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    base = uploads_dir / "base.txt"
    base.write_text("still uploading", encoding="utf-8")
    monkeypatch.setattr(app, "UPLOADS", uploads_dir)
    monkeypatch.setattr(app, "UPLOADS_META", {
        "base": {"path": base, "name": "base.txt", "size": 14},
        "target": None,
        "seed": None,
    })
    monkeypatch.setattr(app, "UPLOADS_ACTIVE", {"target"})

    assert app.cleanup_uploaded_files(log_message=False) == []
    assert base.exists()
    assert app.UPLOADS_META["base"] is not None


def test_clear_log_history_truncates_engine_log_and_preserves_results(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-clear-test"
    run_dir.mkdir()
    engine_log = run_dir / "engine.log"
    engine_log.write_text("old engine output\nsecond line\n", encoding="utf-8")
    rules = run_dir / "rules.txt"
    rules.write_text("generated rules stay here\n", encoding="utf-8")

    monkeypatch.setattr(app, "STATE", {
        "status": "running", "logs": [
            {"id": 1, "text": "old engine output", "time": time.time()},
            {"id": 2, "text": "second line", "time": time.time()},
        ],
        "log_seq": 2, "log_generation": 4, "run_dir": str(run_dir),
        "outputs": [
            {"name": "engine.log", "size": engine_log.stat().st_size},
            {"name": "rules.txt", "size": rules.stat().st_size},
        ],
    })

    result = app.clear_log_history()

    assert engine_log.read_text(encoding="utf-8") == ""
    assert app.STATE["logs"] == []
    assert app.STATE["log_seq"] == 2  # Sequence stays monotonic after a clear.
    assert result == {"log_seq": 2, "log_generation": 5}
    assert app.STATE["outputs"][0]["size"] == 0
    assert rules.read_text(encoding="utf-8") == "generated rules stay here\n"


def test_log_clear_ui_calls_server_and_tracks_log_generations():
    web_dir = Path(app.__file__).resolve().parent / "web"
    html = (web_dir / "index.html").read_text(encoding="utf-8")
    js = (web_dir / "app.js").read_text(encoding="utf-8")

    assert 'id="clear-log"' in html and '<span>Clear log history</span>' in html
    assert 'title="Clear visible logs and truncate the saved engine.log"' in html
    assert "async function clearLogHistoryFromUi()" in js
    assert "apiPost('/api/logs/clear', {})" in js
    assert "window.confirm('Clear the displayed log history" in js
    assert "logGeneration = result.log_generation" in js
    assert "generation=${generationParam}" in js
    assert "data.log_generation !== logGeneration" in js



def test_log_clear_control_is_prominent_and_branding_is_updated():
    web_dir = Path(app.__file__).resolve().parent / "web"
    html = (web_dir / "index.html").read_text(encoding="utf-8")
    css = (web_dir / "styles.css").read_text(encoding="utf-8")
    js = (web_dir / "app.js").read_text(encoding="utf-8")

    # The clear action is a standalone toolbar row, not a tiny utility button
    # mixed into the optional wrap/export controls.
    assert html.count('id="clear-log"') == 1
    assert 'class="log-clear-row"' in html
    assert 'class="clear-logs-action"' in html
    assert '<span>Clear log history</span>' in html
    assert '.clear-logs-action' in css
    assert "$('clear-log').addEventListener('click', clearLogHistoryFromUi)" in js

    assert "Ready when you are" not in html
    assert "Rule engine standing by" in html
    assert 'class="brand-mark"' in html and '<svg viewBox="0 0 44 44"' in html
    assert (web_dir / "favicon.svg").is_file()


def test_favicon_is_served_by_http_server():
    import threading
    import urllib.request

    server = app.RulestHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/favicon.svg", timeout=3) as response:
            payload = response.read().decode("utf-8")
            assert response.status == 200
            assert response.headers.get_content_type() == "image/svg+xml"
            assert "#b8f36b" in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

def test_clear_logs_http_endpoint_resets_other_browser_sessions(tmp_path, monkeypatch):
    import json
    import threading
    import urllib.request

    run_dir = tmp_path / "run-http-clear"
    run_dir.mkdir()
    engine_log = run_dir / "engine.log"
    engine_log.write_text("historic entry\n", encoding="utf-8")
    monkeypatch.setattr(app, "STATE", {
        "status": "running", "paused": False, "pause_requested": False,
        "started_at": time.time(), "finished_at": None, "stage_started_at": time.time(),
        "stage": "Stage 1", "stage_detail": "Working", "progress": None,
        "pid": 123, "exit_code": None,
        "logs": [{"id": 20, "text": "historic entry", "time": time.time()}],
        "log_seq": 20, "log_generation": 2, "outputs": [],
        "error": None, "control_file": None, "run_dir": str(run_dir),
        "job_id": "http-clear",
    })
    monkeypatch.setattr(app, "AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr(app, "SESSIONS", {})
    monkeypatch.setattr(app, "LOGIN_FAILURES", {})
    server = app.RulestHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        setup_request = urllib.request.Request(
            base_url + "/api/auth/setup", data=json.dumps({"password": "test-password-123456"}).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(setup_request, timeout=3) as response:
            setup = json.loads(response.read().decode("utf-8"))
            cookie = response.headers.get("Set-Cookie").split(";", 1)[0]
        request = urllib.request.Request(
            base_url + "/api/logs/clear", data=b"{}", method="POST",
            headers={"Content-Type": "application/json", "Cookie": cookie, "X-CSRF-Token": setup["csrf_token"]},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            cleared = json.loads(response.read().decode("utf-8"))
        assert cleared["ok"] is True
        assert cleared["log_generation"] == 3
        assert engine_log.read_text(encoding="utf-8") == ""

        # A second browser still on generation 2 must see the reset, not stale lines.
        status_request = urllib.request.Request(base_url + "/api/status?since=20&generation=2", headers={"Cookie": cookie})
        with urllib.request.urlopen(status_request, timeout=3) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert status["log_generation"] == 3
        assert status["logs"] == []

        app.add_log("fresh line after clear")
        status_request = urllib.request.Request(base_url + "/api/status?since=20&generation=3", headers={"Cookie": cookie})
        with urllib.request.urlopen(status_request, timeout=3) as response:
            status = json.loads(response.read().decode("utf-8"))
        assert [entry["text"] for entry in status["logs"]] == ["fresh line after clear"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
