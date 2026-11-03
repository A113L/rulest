import json
import sys
import threading
import urllib.error
import urllib.request
from http.cookies import SimpleCookie
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def request_json(url, payload=None, cookie=None, csrf=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if cookie:
        headers["Cookie"] = cookie
    if csrf:
        headers["X-CSRF-Token"] = csrf
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    body = response.read().decode("utf-8")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = {"body": body}
    return response, parsed


def start_server(monkeypatch, auth_file):
    monkeypatch.setattr(app, "AUTH_FILE", auth_file)
    monkeypatch.setattr(app, "SESSIONS", {})
    monkeypatch.setattr(app, "LOGIN_FAILURES", {})
    server = app.RulestHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_password_record_stores_only_salted_pbkdf2_hash(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    monkeypatch.setattr(app, "AUTH_FILE", auth_file)
    record = app.make_password_record("a-long-example-passphrase")
    app.save_password_record(record)

    saved = auth_file.read_text(encoding="utf-8")
    assert "a-long-example-passphrase" not in saved
    assert record["algorithm"] == "pbkdf2_sha256"
    assert record["iterations"] >= 600_000
    assert app.verify_password("a-long-example-passphrase")
    assert not app.verify_password("wrong-password")


def test_first_visit_setup_protects_api_and_sets_http_only_session(tmp_path, monkeypatch):
    server, thread, base = start_server(monkeypatch, tmp_path / "auth.json")
    try:
        response, me = request_json(base + "/api/auth/me")
        assert me["authenticated"] is False
        assert me["setup_required"] is True

        response, rejected = request_json(base + "/api/status")
        assert response.code == 401
        assert rejected["auth_required"] is True

        response, setup = request_json(base + "/api/auth/setup", {"password": "correct-horse-battery-staple"})
        assert response.code == 200 and setup["ok"] is True
        cookie_header = response.headers.get("Set-Cookie")
        assert cookie_header and "HttpOnly" in cookie_header and "SameSite=Strict" in cookie_header
        cookie = SimpleCookie(); cookie.load(cookie_header)
        session_cookie = f"rulest_session={cookie['rulest_session'].value}"

        response, me = request_json(base + "/api/auth/me", cookie=session_cookie)
        assert me["authenticated"] is True
        assert me["csrf_token"]

        response, status = request_json(base + "/api/status", cookie=session_cookie)
        assert response.code == 200
        assert "status" in status

        auth_text = (tmp_path / "auth.json").read_text(encoding="utf-8")
        assert "correct-horse-battery-staple" not in auth_text
    finally:
        stop_server(server, thread)


def test_protected_posts_require_csrf_and_logout_revokes_session(tmp_path, monkeypatch):
    server, thread, base = start_server(monkeypatch, tmp_path / "auth.json")
    try:
        response, setup = request_json(base + "/api/auth/setup", {"password": "another-strong-password-123"})
        cookie = "rulest_session=" + SimpleCookie(response.headers["Set-Cookie"])["rulest_session"].value
        csrf = setup["csrf_token"]

        response, denied = request_json(base + "/api/logs/clear", {}, cookie=cookie)
        assert response.code == 403
        assert "security token" in denied["error"].lower()

        response, logged_out = request_json(base + "/api/auth/logout", {}, cookie=cookie, csrf=csrf)
        assert response.code == 200 and logged_out["ok"] is True
        response, denied = request_json(base + "/api/status", cookie=cookie)
        assert response.code == 401
    finally:
        stop_server(server, thread)


def test_login_and_change_password_rotate_sessions(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    server, thread, base = start_server(monkeypatch, auth_file)
    try:
        response, setup = request_json(base + "/api/auth/setup", {"password": "first-secure-password-987"})
        cookie = "rulest_session=" + SimpleCookie(response.headers["Set-Cookie"])["rulest_session"].value
        csrf = setup["csrf_token"]

        response, changed = request_json(
            base + "/api/auth/change-password",
            {"current_password": "first-secure-password-987", "new_password": "second-secure-password-654"},
            cookie=cookie, csrf=csrf,
        )
        assert response.code == 200 and changed["ok"] is True
        new_cookie = "rulest_session=" + SimpleCookie(response.headers["Set-Cookie"])["rulest_session"].value
        assert new_cookie != cookie

        # The old session is revoked and the new password works for a fresh login.
        response, old_session = request_json(base + "/api/status", cookie=cookie)
        assert response.code == 401
        response, bad_login = request_json(base + "/api/auth/login", {"password": "first-secure-password-987"})
        assert response.code == 401
        response, login = request_json(base + "/api/auth/login", {"password": "second-secure-password-654"})
        assert response.code == 200 and login["ok"] is True
    finally:
        stop_server(server, thread)


def test_setup_rejects_weak_password(tmp_path, monkeypatch):
    server, thread, base = start_server(monkeypatch, tmp_path / "auth.json")
    try:
        response, data = request_json(base + "/api/auth/setup", {"password": "short"})
        assert response.code == 400
        assert "12 characters" in data["error"]
        assert not (tmp_path / "auth.json").exists()
    finally:
        stop_server(server, thread)


def test_login_page_and_main_assets_are_separated_by_authentication(tmp_path, monkeypatch):
    server, thread, base = start_server(monkeypatch, tmp_path / "auth.json")
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            page = response.read().decode("utf-8")
            assert response.status == 200
            assert "Create your password" in page or "Create your password." in page
            assert "first person" not in page.lower()  # Message is displayed after setup-state detection.
        response, setup = request_json(base + "/api/auth/setup", {"password": "private-lan-password-123"})
        cookie = "rulest_session=" + SimpleCookie(response.headers["Set-Cookie"])["rulest_session"].value
        request = urllib.request.Request(base + "/", headers={"Cookie": cookie})
        with urllib.request.urlopen(request, timeout=5) as response:
            page = response.read().decode("utf-8")
            assert "Run workspace" in page
            assert "PASSWORD SESSION" in page
            assert "Change password" in page
        request = urllib.request.Request(base + "/app.js", headers={"Cookie": cookie})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.headers.get_content_type() == "text/javascript"
    finally:
        stop_server(server, thread)


def test_login_rate_limiter_blocks_after_five_incorrect_attempts(tmp_path, monkeypatch):
    server, thread, base = start_server(monkeypatch, tmp_path / "auth.json")
    try:
        record = app.make_password_record("rate-limit-test-password")
        app.save_password_record(record)
        monkeypatch.setattr(app, "verify_password", lambda _password: False)
        for _ in range(app.LOGIN_MAX_FAILURES):
            response, _ = request_json(base + "/api/auth/login", {"password": "wrong-password"})
            assert response.code == 401
        response, locked = request_json(base + "/api/auth/login", {"password": "rate-limit-test-password"})
        assert response.code == 429
        assert "too many failed attempts" in locked["error"].lower()
    finally:
        stop_server(server, thread)
