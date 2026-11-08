#!/usr/bin/env python3
"""Local web interface for rulest. Uses only the Python standard library."""
from __future__ import annotations

import json
import os
import codecs
import re
import sys
import time
import uuid
import shutil
import signal
import threading
import subprocess
import urllib.parse
import ipaddress
import socket
import secrets
import hashlib
import hmac
import base64
from http.cookies import SimpleCookie
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gpu_diagnostics import collect_gpu_diagnostics

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DATA = ROOT / "runs"
UPLOADS = DATA / "uploads"
AUTH_FILE = ROOT / "auth.json"
DATA.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

# Passwords are never stored in clear text. Sessions live in memory, so a server
# restart automatically signs out every browser.
PASSWORD_ITERATIONS = 600_000
SESSION_TTL_SECONDS = 12 * 60 * 60
PASSWORD_MIN_LENGTH = 12
LOGIN_WINDOW_SECONDS = 5 * 60
LOGIN_MAX_FAILURES = 5
LOGIN_LOCK_SECONDS = 5 * 60
SESSIONS = {}       # token -> {csrf, expires}
LOGIN_FAILURES = {} # client IP -> {count, first, locked_until}

LOCK = threading.RLock()
STATE = {
    "status": "idle", "paused": False, "pause_requested": False, "job_id": None, "started_at": None, "finished_at": None,
    "pid": None, "stage": "Waiting for a run", "stage_detail": "Start a run to see the active processing phase.",
    "stage_started_at": None, "progress": None,
    "exit_code": None, "logs": [], "log_seq": 0, "log_generation": 0, "outputs": [],
    "error": None, "control_file": None, "run_dir": None,
}
UPLOADS_META = {"base": None, "target": None, "seed": None}
UPLOADS_ACTIVE = set()
PROCESS = None
RUN_THREAD = None
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB per selected file

BOOL_OPTIONS = {
    "bloom_no_shard": "--bloom-no-shard",
    "allow_reject_rules": "--allow-reject-rules",
    "no_builtin_seeds": "--no-builtin-seeds",
    "token_strip": "--token-strip",
    "genetic": "--genetic",
    "exact_recovery": "--exact-recovery",
    "debug": "--debug",
    "profile": "--profile",
}
VALUE_OPTIONS = {
    "device": "--device", "max_depth": "--max-depth", "target_hours": "--target-hours",
    "max_chains": "--max-chains", "bloom_mb": "--bloom-mb",
    "token_strip_min_stem": "--token-strip-min-stem",
    "token_strip_max_prefix": "--token-strip-max-prefix",
    "token_strip_max_suffix": "--token-strip-max-suffix",
    "token_strip_min_leet_amb": "--token-strip-min-leet-amb",
    "token_strip_workers": "--token-strip-workers",
    "token_strip_chunk_size": "--token-strip-chunk-size",
    "token_strip_bloom_mb": "--token-strip-bloom-mb",
    "genetic_generations": "--genetic-generations", "genetic_pop": "--genetic-pop",
    "genetic_elite": "--genetic-elite", "select_mode": "--select-mode",
    "select_budget": "--select-budget", "select_budgets": "--select-budgets",
    "select_cost_alpha": "--select-cost-alpha",
}


def auth_configured() -> bool:
    return AUTH_FILE.is_file()


def _derive_password_hash(password: str, salt: bytes, iterations: int = PASSWORD_ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)


def make_password_record(password: str) -> dict:
    salt = secrets.token_bytes(32)
    digest = _derive_password_hash(password, salt)
    return {
        "algorithm": "pbkdf2_sha256",
        "iterations": PASSWORD_ITERATIONS,
        "salt": base64.b64encode(salt).decode("ascii"),
        "password_hash": base64.b64encode(digest).decode("ascii"),
        "created_at": int(time.time()),
    }


def save_password_record(record: dict):
    """Atomically persist the password verifier with owner-only file permissions."""
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUTH_FILE.with_name(AUTH_FILE.name + ".tmp")
    payload = json.dumps(record, indent=2) + "\n"
    # Create the temporary verifier with owner-only permissions from the outset.
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass  # Windows ACLs govern access there; the file still stores only a hash.
    temporary.replace(AUTH_FILE)
    try:
        os.chmod(AUTH_FILE, 0o600)
    except OSError:
        pass


def load_password_record() -> dict:
    try:
        record = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        if record.get("algorithm") != "pbkdf2_sha256":
            raise ValueError("Unsupported password hash algorithm in auth.json.")
        return record
    except FileNotFoundError:
        raise ValueError("No application password has been configured yet.")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"The authentication file is invalid: {exc}") from exc


def verify_password(password: str) -> bool:
    if not isinstance(password, str) or len(password) > 1024 or not auth_configured():
        return False
    try:
        record = load_password_record()
        salt = base64.b64decode(record["salt"], validate=True)
        expected = base64.b64decode(record["password_hash"], validate=True)
        iterations = int(record["iterations"])
        if iterations < 100_000 or iterations > 2_000_000 or len(salt) < 16 or len(expected) != 32:
            return False
        actual = _derive_password_hash(password, salt, iterations)
        return hmac.compare_digest(actual, expected)
    except (ValueError, KeyError, TypeError, base64.binascii.Error):
        return False


def _cookie_token(handler) -> str | None:
    try:
        cookie = SimpleCookie()
        cookie.load(handler.headers.get("Cookie", ""))
        morsel = cookie.get("rulest_session")
        return morsel.value if morsel else None
    except Exception:
        return None


def get_session(handler):
    token = _cookie_token(handler)
    if not token:
        return None
    now = time.time()
    with LOCK:
        for key in [key for key, value in SESSIONS.items() if value.get("expires", 0) <= now]:
            SESSIONS.pop(key, None)
        session = SESSIONS.get(token)
        if session and session.get("expires", 0) > now:
            return token, session
    return None


def create_session(handler):
    token = secrets.token_urlsafe(36)
    csrf_token = secrets.token_urlsafe(32)
    with LOCK:
        SESSIONS[token] = {"csrf": csrf_token, "expires": time.time() + SESSION_TTL_SECONDS}
    # Queue the cookie until send_response() has emitted the HTTP status line.
    handler._auth_set_cookie = (
        f"rulest_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_TTL_SECONDS}"
    )
    return csrf_token


def clear_session_cookie(handler):
    handler.send_header(
        "Set-Cookie",
        "rulest_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
    )


def request_ip(handler) -> str:
    try:
        return str(handler.client_address[0])
    except Exception:
        return "unknown"


def login_lock_remaining(client_ip: str) -> int:
    now = time.time()
    with LOCK:
        item = LOGIN_FAILURES.get(client_ip)
        if not item:
            return 0
        if item.get("locked_until", 0) > now:
            return max(1, int(item["locked_until"] - now))
        if item.get("locked_until", 0) and item.get("locked_until", 0) <= now:
            LOGIN_FAILURES.pop(client_ip, None)
            return 0
        if now - item.get("first", now) > LOGIN_WINDOW_SECONDS:
            LOGIN_FAILURES.pop(client_ip, None)
        return 0


def record_login_failure(client_ip: str):
    now = time.time()
    with LOCK:
        item = LOGIN_FAILURES.get(client_ip)
        if not item or now - item.get("first", now) > LOGIN_WINDOW_SECONDS:
            item = {"count": 0, "first": now, "locked_until": 0}
        item["count"] += 1
        if item["count"] >= LOGIN_MAX_FAILURES:
            item["locked_until"] = now + LOGIN_LOCK_SECONDS
        LOGIN_FAILURES[client_ip] = item


def reset_login_failures(client_ip: str):
    with LOCK:
        LOGIN_FAILURES.pop(client_ip, None)


def safe_name(value: str, fallback="output.txt") -> str:
    value = Path(str(value or fallback)).name
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" .")
    return value or fallback


def cleanup_uploaded_files(log_message: bool = True):
    """Delete server-side copies of uploaded inputs and clear their metadata.

    Run output files live under runs/<run-id>/ and are deliberately untouched.
    The caller must only invoke this when no upload request is in progress.
    """
    removed = []
    with LOCK:
        if UPLOADS_ACTIVE:
            return removed
        for field in UPLOADS_META:
            UPLOADS_META[field] = None
        try:
            for candidate in UPLOADS.iterdir():
                # Uploaded files are always direct children of runs/uploads.
                # Unlink symlinks themselves; never follow them.
                if candidate.is_file() or candidate.is_symlink():
                    try:
                        candidate.unlink()
                        removed.append(candidate.name)
                    except OSError:
                        pass
        except FileNotFoundError:
            UPLOADS.mkdir(parents=True, exist_ok=True)
    if removed and log_message:
        add_log(f"[WEB] Deleted {len(removed)} temporary uploaded input file(s).")
    return removed


def cleanup_stale_uploaded_files():
    """Remove upload leftovers from a previous unclean server shutdown."""
    return cleanup_uploaded_files(log_message=False)


ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\)|[@-Z\\-_])"
)
PROGRESS_PERCENT_RE = re.compile(r"(?<!\d)(\d{1,3})%\s*\|")
PROGRESS_COUNT_RE = re.compile(r"\|\s*([\d,]+)\s*/\s*([\d,]+)\s*\[")


def clean_terminal_text(line: str) -> str:
    """Remove terminal control sequences without dropping printable log text."""
    clean = ANSI_ESCAPE_RE.sub("", str(line))
    clean = clean.replace("\x08", "")
    clean = "".join(ch for ch in clean if ch in "\t" or ord(ch) >= 32)
    return clean.strip()


def detect_active_stage(line: str):
    """Return a phase only for real activity markers, never startup option summaries."""
    text = clean_terminal_text(line)
    upper = text.upper()
    # CLI prints a configuration overview near startup. Those lines say
    # `STAGE N : enabled/disabled` and must not masquerade as active phases.
    if re.search(r"\bSTAGE\s*[0-3S]\s*:\s*(?:ENABLED|DISABLED)\b", upper):
        return None

    # Section headings are emitted exactly when a phase starts. Progress bars
    # also contain the stage name and a percentage marker.
    if "DONE" in upper and "RULEST FINISHED" in upper:
        return "Completed"
    if "POST-PROCESSING" in upper or "[MINIMIZE]" in upper or "[MIN]" in upper or "MINIMIZING" in upper:
        return "Minimization"
    if "SELECTION — CELF" in upper or "SELECTION - CELF" in upper or "[CELF]" in upper or "[SELECT]" in upper:
        return "Rule Selection · CELF"
    if "BLOOM FILTER" in upper or "BUILDING BLOOM" in upper:
        return "GPU Setup · Bloom Filter"
    if re.search(r"\bSTAGE\s*0\b", upper) or "[S0]" in upper:
        if "STAGE 0" in upper or "[S0]" in upper:
            return "Stage 0 · Token Strip"
    if re.search(r"\bSTAGE\s*S\b", upper) or "[SEED]" in upper:
        return "Stage S · Built-in Seeds"
    if re.search(r"\bSTAGE\s*1\b", upper) or "[S1]" in upper:
        return "Stage 1 · Single-rule GPU sweep"
    if re.search(r"\bSTAGE\s*2\b", upper) or "[S2]" in upper:
        return "Stage 2 · Rule-chain search"
    if re.search(r"\bSTAGE\s*3\b", upper) or "[S3]" in upper:
        return "Stage 3 · Genetic Algorithm"
    if "[OUT]" in upper:
        return "Writing output"
    if "[GPU]" in upper and any(word in upper for word in ("INITIAL", "COMPILE", "DEVICE", "CONTEXT")):
        return "GPU Setup · Device initialization"
    return None


def clear_log_history():
    """Clear the current browser log buffer and truncate the persisted engine log.

    A generation counter lets every connected browser discard stale entries,
    including entries that were already rendered before another client cleared
    the log. Sequence numbers remain monotonic within a run.
    """
    with LOCK:
        run_dir = STATE.get("run_dir")
        log_path = Path(run_dir) / "engine.log" if run_dir else None
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("", encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(f"Could not clear the saved engine log: {exc}") from exc

        STATE["logs"] = []
        STATE["log_generation"] = int(STATE.get("log_generation", 0)) + 1
        for output in STATE.get("outputs", []):
            if output.get("name") == "engine.log":
                output["size"] = 0
        return {
            "log_seq": int(STATE.get("log_seq", 0)),
            "log_generation": STATE["log_generation"],
        }


def add_log(line: str):
    clean = clean_terminal_text(line)
    if not clean:
        return
    with LOCK:
        STATE["log_seq"] += 1
        event_time = time.time()
        STATE["logs"].append({"id": STATE["log_seq"], "text": clean, "time": event_time})
        if len(STATE["logs"]) > 5000:
            STATE["logs"] = STATE["logs"][-4000:]
        run_dir = STATE.get("run_dir")
        if run_dir:
            try:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event_time))
                with (Path(run_dir) / "engine.log").open("a", encoding="utf-8") as log_file:
                    log_file.write(f"[{stamp}] {clean}\n")
            except OSError:
                pass


def update_stage(line: str):
    clean = clean_terminal_text(line)
    if not clean:
        return
    stage = detect_active_stage(clean)
    pct_match = PROGRESS_PERCENT_RE.search(clean)
    count_match = PROGRESS_COUNT_RE.search(clean)
    progress = None
    if pct_match:
        progress = min(100, int(pct_match.group(1)))
    elif count_match:
        total = int(count_match.group(2).replace(",", ""))
        done = int(count_match.group(1).replace(",", ""))
        if total > 0:
            progress = min(100, round(done * 100 / total))

    with LOCK:
        if "RESUMED" in clean.upper():
            STATE["paused"] = False
            STATE["pause_requested"] = False
        elif "PAUSED" in clean.upper():
            STATE["paused"] = True
            STATE["pause_requested"] = False
        if stage:
            if stage != STATE["stage"]:
                STATE["progress"] = None
                STATE["stage_started_at"] = time.time()
            STATE["stage"] = stage
            if stage == "Completed":
                STATE["progress"] = 100
                STATE["stage_detail"] = "All processing phases completed successfully."
            else:
                # Keep the most recent meaningful engine message as the live
                # sub-step, while retaining the complete line in the log.
                STATE["stage_detail"] = clean[:320]
        elif clean and STATE.get("status") in ("running", "stopping"):
            # Show current work details from the same phase without allowing
            # the startup configuration list to change the active phase.
            if STATE.get("stage") not in ("Waiting for a run", "Starting engine"):
                STATE["stage_detail"] = clean[:320]
        if progress is not None:
            STATE["progress"] = progress


def build_command(options: dict, run_dir: Path):
    base = UPLOADS_META.get("base")
    target = UPLOADS_META.get("target")
    if not base or not target:
        raise ValueError("Upload both BASE and TARGET wordlists before starting a run.")
    output_name = safe_name(options.get("output_name", "rulest_output.txt"))
    if output_name.lower() == "control.txt":
        raise ValueError("The output filename 'control.txt' is reserved for run controls.")
    output_path = run_dir / output_name
    cmd = [sys.executable, "-u", str(ROOT / "run_rulest.py"), str(base["path"]), str(target["path"]), "--output", str(output_path)]
    for key, flag in BOOL_OPTIONS.items():
        if options.get(key):
            cmd.append(flag)
    for key, flag in VALUE_OPTIONS.items():
        value = options.get(key)
        if value is None or value == "":
            continue
        cmd.extend([flag, str(value)])
    # The engine's --seed option fixes Python/NumPy RNG state and pins
    # PYTHONHASHSEED via its early re-exec path. Keep it optional so existing
    # clients retain the engine's default non-deterministic behaviour.
    rng_seed_raw = options.get("rng_seed")
    if rng_seed_raw not in (None, ""):
        if isinstance(rng_seed_raw, bool):
            raise ValueError("RNG seed must be an integer from 0 to 4294967295.")
        rng_seed_text = str(rng_seed_raw).strip()
        if not re.fullmatch(r"[0-9]{1,10}", rng_seed_text):
            raise ValueError("RNG seed must be an integer from 0 to 4294967295.")
        rng_seed = int(rng_seed_text)
        if rng_seed > 0xFFFFFFFF:
            raise ValueError("RNG seed must be between 0 and 4294967295.")
        cmd.extend(["--seed", str(rng_seed)])
    seed = UPLOADS_META.get("seed")
    if seed:
        cmd.extend(["--seed-rules", str(seed["path"])])
    for depth in range(2, 11):
        value = options.get(f"depth{depth}_chains")
        if value not in (None, ""):
            cmd.extend([f"--depth{depth}-chains", str(value)])
    return cmd, output_name


def run_worker(cmd, run_id, run_dir, control_file):
    global PROCESS, RUN_THREAD
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Give progress renderers a wide virtual terminal so they do not discard
    # useful counters/postfix fields before the GUI can wrap the full line.
    env["COLUMNS"] = "200"
    env["LINES"] = "40"
    env["RULEST_CONTROL_FILE"] = str(control_file)
    creationflags = 0
    popen_kwargs = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    start = time.time()
    proc = None
    code = None
    error = None
    outputs = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, creationflags=creationflags, **popen_kwargs,
        )
        with LOCK:
            PROCESS = proc
            STATE["pid"] = proc.pid
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            decoded = decoder.decode(chunk)
            # rulest progress bars redraw with CR; split them into live log lines.
            pending += decoded
            parts = re.split(r"\r\n|\n|\r", pending)
            pending = parts.pop() if parts else ""
            for line in parts:
                add_log(line)
                update_stage(line)
        pending += decoder.decode(b"", final=True)
        if pending.strip():
            add_log(pending)
            update_stage(pending)
        code = proc.wait()
        add_log(f"[WEB] Process finished with exit code {code} in {time.time() - start:.1f}s.")
    except Exception as exc:
        error = str(exc)
        add_log(f"[WEB ERROR] {exc}")
    finally:
        # Keep status as running/stopping until cleanup finishes so a new upload
        # cannot race with deletion of the previous run's inputs.
        with LOCK:
            was_stopped = STATE.get("status") == "stopping"
        cleanup_uploaded_files(log_message=True)
        if error is None:
            for item in sorted(run_dir.iterdir()):
                if item.is_file() and item.name != "control.txt":
                    outputs.append({"name": item.name, "size": item.stat().st_size})
        with LOCK:
            PROCESS = None
            RUN_THREAD = None
            STATE["pid"] = None
            STATE["paused"] = False
            STATE["pause_requested"] = False
            STATE["finished_at"] = time.time()
            STATE["outputs"] = outputs
            if error is not None:
                STATE["status"] = "failed"
                STATE["error"] = error
                STATE["stage"] = "Process error"
                STATE["stage_detail"] = error[:320]
                STATE["exit_code"] = code
            elif was_stopped:
                STATE["status"] = "stopped"
                STATE["stage"] = "Stopped"
                STATE["stage_detail"] = "Run stopped. Partial results, if any, were preserved."
                STATE["exit_code"] = code
            else:
                STATE["status"] = "completed" if code == 0 else "failed"
                STATE["exit_code"] = code
                STATE["stage"] = "Completed" if code == 0 else "Process exited with an error"
                STATE["stage_detail"] = (
                    "Run completed successfully." if code == 0
                    else f"Engine process exited with code {code}. Check the log for the failure details."
                )
                STATE["progress"] = 100 if code == 0 else STATE.get("progress")


class RulestHTTPServer(ThreadingHTTPServer):
    # Wait for in-flight upload requests before the final temporary-file cleanup.
    daemon_threads = False
    block_on_close = True


class Handler(BaseHTTPRequestHandler):
    server_version = "RulestGUI/1.4"

    def log_message(self, fmt, *args):
        # Keep the terminal clean; application activity is in the browser log.
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        queued_cookie = getattr(self, "_auth_set_cookie", None)
        if queued_cookie:
            self.send_header("Set-Cookie", queued_cookie)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Login/setup endpoints and the branded favicon are intentionally public.
        if path == "/api/auth/me":
            self.handle_auth_me()
            return
        if path == "/favicon.svg":
            self.serve_static("favicon.svg", "image/svg+xml; charset=utf-8")
            return
        if path in ("/", "/index.html", "/login"):
            if get_session(self):
                if path == "/login":
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                else:
                    self.serve_static("index.html", "text/html; charset=utf-8")
            else:
                self.serve_static("login.html", "text/html; charset=utf-8")
            return

        if not self.require_authenticated(api=path.startswith("/api/")):
            return

        if path == "/app.js":
            self.serve_static("app.js", "text/javascript; charset=utf-8")
        elif path == "/styles.css":
            self.serve_static("styles.css", "text/css; charset=utf-8")
        elif path == "/api/devices":
            try:
                diagnostics = collect_gpu_diagnostics()
                devices = [
                    {"index": str(item["index"]), "name": item["name"], "type": item["type"],
                     "vendor": item["vendor"], "platform": item["platform"]}
                    for item in diagnostics["devices"]
                ]
                self.send_json({
                    "ok": diagnostics["gpu_count"] > 0,
                    "devices": devices,
                    "gpu_devices": [item for item in devices if item["type"] == "GPU"],
                    "message": diagnostics["summary"],
                    "diagnostics": diagnostics,
                })
            except Exception as exc:
                self.send_json({
                    "ok": False, "devices": [], "gpu_devices": [],
                    "message": f"GPU diagnostics failed: {type(exc).__name__}: {exc}",
                }, 500)
        elif path == "/api/status":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                since = int(qs.get("since", ["0"])[0])
            except ValueError:
                since = 0
            try:
                client_generation = int(qs.get("generation", ["-1"])[0])
            except ValueError:
                client_generation = -1
            with LOCK:
                elapsed = None
                if STATE["started_at"]:
                    end = STATE["finished_at"] or time.time()
                    elapsed = max(0, end - STATE["started_at"])
                data = {k: v for k, v in STATE.items() if k not in ("logs", "control_file", "run_dir")}
                data["elapsed"] = elapsed
                data["stage_elapsed"] = max(0, time.time() - STATE["stage_started_at"]) if STATE.get("stage_started_at") else None
                current_generation = int(STATE.get("log_generation", 0))
                data["log_generation"] = current_generation
                data["logs"] = (
                    list(STATE["logs"]) if client_generation != current_generation
                    else [entry for entry in STATE["logs"] if entry["id"] > since]
                )
                data["uploads"] = {k: ({"name": v["name"], "size": v["size"]} if v else None) for k, v in UPLOADS_META.items()}
            self.send_json(data)
        elif path.startswith("/api/download/"):
            name = safe_name(urllib.parse.unquote(path.rsplit("/", 1)[-1]))
            with LOCK:
                run_dir = Path(STATE["run_dir"]) if STATE.get("run_dir") else None
                allowed = {x["name"] for x in STATE.get("outputs", [])}
            target = run_dir / name if run_dir else None
            if not target or name not in allowed or not target.is_file():
                self.send_error(404, "Output not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            with target.open("rb") as f:
                shutil.copyfileobj(f, self.wfile)
        else:
            self.send_error(404)

    def serve_static(self, filename, content_type):
        path = WEB / filename
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store" if filename == "index.html" else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/auth/setup":
            self.handle_auth_setup()
            return
        if parsed.path == "/api/auth/login":
            self.handle_auth_login()
            return
        if not self.require_authenticated(api=True, csrf=True):
            return
        if parsed.path == "/api/auth/logout":
            self.handle_auth_logout()
        elif parsed.path == "/api/auth/change-password":
            self.handle_auth_change_password()
        elif parsed.path.startswith("/api/upload/"):
            self.handle_upload(parsed)
        elif parsed.path == "/api/run":
            self.handle_run()
        elif parsed.path == "/api/control":
            self.handle_control()
        elif parsed.path == "/api/logs/clear":
            self.handle_clear_logs()
        else:
            self.send_error(404)

    def require_authenticated(self, api=False, csrf=False):
        session = get_session(self)
        if not session:
            if api:
                self.send_json({"error": "Authentication required. Please sign in again.", "auth_required": True}, 401)
            else:
                self.send_response(302)
                self.send_header("Location", "/")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            return False
        if csrf:
            supplied = self.headers.get("X-CSRF-Token", "")
            if not supplied or not hmac.compare_digest(supplied, session[1].get("csrf", "")):
                self.send_json({"error": "Invalid or missing request security token. Refresh the page and try again."}, 403)
                return False
        return True

    def read_json_body(self, maximum=1024 * 1024):
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > maximum:
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return None

    def handle_auth_me(self):
        session = get_session(self)
        self.send_json({
            "authenticated": bool(session),
            "setup_required": not auth_configured(),
            "csrf_token": session[1]["csrf"] if session else None,
            "session_expires_in": max(0, int(session[1]["expires"] - time.time())) if session else 0,
        })

    def handle_auth_setup(self):
        if auth_configured():
            self.send_json({"error": "An application password is already configured. Sign in instead."}, 409)
            return
        data = self.read_json_body(16 * 1024)
        password = data.get("password") if isinstance(data, dict) else None
        if not isinstance(password, str) or len(password) < PASSWORD_MIN_LENGTH or len(password) > 1024:
            self.send_json({"error": f"Choose a password with at least {PASSWORD_MIN_LENGTH} characters."}, 400)
            return
        with LOCK:
            if auth_configured():
                self.send_json({"error": "An application password was configured by another session. Sign in instead."}, 409)
                return
            try:
                save_password_record(make_password_record(password))
            except OSError as exc:
                self.send_json({"error": f"Could not save auth.json: {exc}"}, 500)
                return
        csrf = create_session(self)
        self.send_json({"ok": True, "csrf_token": csrf, "message": "Password configured. You are signed in."})

    def handle_auth_login(self):
        client_ip = request_ip(self)
        remaining = login_lock_remaining(client_ip)
        if remaining:
            self.send_json({"error": f"Too many failed attempts. Try again in {remaining} seconds."}, 429)
            return
        if not auth_configured():
            self.send_json({"error": "Create an application password before signing in."}, 409)
            return
        data = self.read_json_body(16 * 1024)
        password = data.get("password") if isinstance(data, dict) else None
        if not isinstance(password, str) or len(password) > 1024 or not verify_password(password):
            record_login_failure(client_ip)
            self.send_json({"error": "Incorrect password."}, 401)
            return
        reset_login_failures(client_ip)
        csrf = create_session(self)
        self.send_json({"ok": True, "csrf_token": csrf, "message": "Signed in successfully."})

    def handle_auth_logout(self):
        token = _cookie_token(self)
        with LOCK:
            if token:
                SESSIONS.pop(token, None)
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        clear_session_cookie(self)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def handle_auth_change_password(self):
        client_ip = request_ip(self)
        remaining = login_lock_remaining(client_ip)
        if remaining:
            self.send_json({"error": f"Too many failed attempts. Try again in {remaining} seconds."}, 429)
            return
        data = self.read_json_body(16 * 1024)
        if not isinstance(data, dict):
            self.send_json({"error": "Invalid password-change request."}, 400)
            return
        current = data.get("current_password")
        new_password = data.get("new_password")
        if not isinstance(current, str) or not verify_password(current):
            record_login_failure(request_ip(self))
            self.send_json({"error": "Current password is incorrect."}, 401)
            return
        if not isinstance(new_password, str) or len(new_password) < PASSWORD_MIN_LENGTH or len(new_password) > 1024:
            self.send_json({"error": f"New password must contain at least {PASSWORD_MIN_LENGTH} characters."}, 400)
            return
        try:
            save_password_record(make_password_record(new_password))
        except OSError as exc:
            self.send_json({"error": f"Could not update auth.json: {exc}"}, 500)
            return
        # Revoke every session and issue a fresh one to this browser.
        with LOCK:
            SESSIONS.clear()
            LOGIN_FAILURES.clear()
        csrf = create_session(self)
        self.send_json({"ok": True, "csrf_token": csrf, "message": "Password changed. Other sessions have been signed out."})

    def handle_clear_logs(self):
        try:
            result = clear_log_history()
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, 500)
            return
        self.send_json({"ok": True, **result})

    def handle_upload(self, parsed):
        field = parsed.path.rsplit("/", 1)[-1]
        if field not in ("base", "target", "seed"):
            self.send_error(404)
            return
        with LOCK:
            if STATE["status"] in ("running", "stopping"):
                self.send_json({"error": "Files cannot be replaced while a run is active."}, 409)
                return
            if field in UPLOADS_ACTIVE:
                self.send_json({"error": f"A {field.upper()} upload is already in progress."}, 409)
                return
            UPLOADS_ACTIVE.add(field)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            with LOCK:
                UPLOADS_ACTIVE.discard(field)
            self.send_json({"error": "File is empty or exceeds the 20 GiB upload limit."}, 413)
            return
        qs = urllib.parse.parse_qs(parsed.query)
        filename = safe_name(qs.get("filename", [f"{field}.txt"])[0], f"{field}.txt")
        stored = UPLOADS / f"{uuid.uuid4().hex}_{filename}"
        remaining = length
        try:
            with stored.open("wb") as out:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("Upload ended before all bytes arrived")
                    out.write(chunk)
                    remaining -= len(chunk)
            meta = {"path": stored, "name": filename, "size": length}
            with LOCK:
                old = UPLOADS_META.get(field)
                UPLOADS_META[field] = meta
                UPLOADS_ACTIVE.discard(field)
            if old and Path(old["path"]).exists():
                try:
                    Path(old["path"]).unlink()
                except OSError:
                    pass
            self.send_json({"ok": True, "field": field, "name": filename, "size": length})
        except Exception as exc:
            with LOCK:
                UPLOADS_ACTIVE.discard(field)
            try:
                stored.unlink(missing_ok=True)
            except OSError:
                pass
            self.send_json({"error": f"Upload failed: {exc}"}, 400)

    def handle_run(self):
        global PROCESS, RUN_THREAD
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 1024 * 1024:
            self.send_json({"error": "Configuration payload is too large."}, 413)
            return
        try:
            options = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self.send_json({"error": "Invalid JSON configuration."}, 400)
            return
        with LOCK:
            if STATE["status"] in ("running", "stopping"):
                self.send_json({"error": "A run is already in progress."}, 409)
                return
            if UPLOADS_ACTIVE:
                self.send_json({"error": "Wait for the active file upload to finish before starting."}, 409)
                return
            if not UPLOADS_META["base"] or not UPLOADS_META["target"]:
                self.send_json({"error": "Upload both BASE and TARGET wordlists first."}, 400)
                return
            run_id = uuid.uuid4().hex[:10]
            run_dir = DATA / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            control_file = run_dir / "control.txt"
            control_file.write_text("", encoding="ascii")
            try:
                cmd, output_name = build_command(options, run_dir)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            STATE.update({
                "status": "running", "paused": False, "pause_requested": False, "job_id": run_id, "started_at": time.time(), "finished_at": None,
                "pid": None, "stage": "Starting engine", "stage_detail": "Launching the engine process…",
                "stage_started_at": time.time(), "progress": None, "exit_code": None,
                "logs": [], "log_seq": 0,
                "log_generation": int(STATE.get("log_generation", 0)) + 1,
                "outputs": [], "error": None,
                "control_file": str(control_file), "run_dir": str(run_dir),
            })
            add_log("[WEB] Starting rulest engine…")
            add_log("[WEB] Uploaded files are mounted as BASE / TARGET inputs.")
            add_log("[WEB] " + " ".join(cmd[2:]))
        thread = threading.Thread(target=run_worker, args=(cmd, run_id, run_dir, control_file), daemon=True)
        with LOCK:
            RUN_THREAD = thread
        thread.start()
        self.send_json({"ok": True, "job_id": run_id, "output": output_name})

    def handle_control(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            action = data.get("action")
        except Exception:
            self.send_json({"error": "Invalid control request."}, 400)
            return
        command = {"pause": "p", "resume": "r", "stop": "q"}.get(action)
        if command is None:
            self.send_json({"error": "Unsupported control action."}, 400)
            return
        with LOCK:
            if STATE["status"] not in ("running", "stopping") or not STATE.get("control_file"):
                self.send_json({"error": "There is no active run to control."}, 409)
                return
            path = Path(STATE["control_file"])
            with path.open("a", encoding="ascii") as f:
                f.write(command + "\n")
            if action == "pause":
                STATE["pause_requested"] = True
            elif action == "resume":
                STATE["pause_requested"] = False
            elif action == "stop":
                STATE["status"] = "stopping"
                STATE["pause_requested"] = False
        labels = {"pause": "Pause requested", "resume": "Resume requested", "stop": "Partial-save exit requested"}
        add_log(f"[WEB] {labels[action]}.")
        self.send_json({"ok": True, "action": action})


def stop_active_run_for_shutdown(timeout: float = 15.0):
    """Ask an active engine to stop, then force-stop it if the server exits."""
    global PROCESS, RUN_THREAD
    with LOCK:
        proc = PROCESS
        worker = RUN_THREAD
        active = STATE.get("status") in ("running", "stopping")
        control_file = STATE.get("control_file")
        if active and control_file:
            try:
                with Path(control_file).open("a", encoding="ascii") as stream:
                    stream.write("q\n")
                STATE["status"] = "stopping"
                STATE["pause_requested"] = False
            except OSError:
                pass
    if active:
        add_log("[WEB] Web server is shutting down; stopping the active RULEST run.")
    if proc is not None and proc.poll() is None:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=8,
                    )
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                try:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
    if worker is not None and worker.is_alive():
        worker.join(timeout=10)
    # If no worker is running (for example after a previous crash), clean up
    # any remaining temporary files. Never delete inputs while the child lives.
    if worker is None or not worker.is_alive():
        cleanup_uploaded_files(log_message=False)


def discover_lan_ipv4_addresses():
    """Discover private IPv4 addresses suitable for opening the UI on a LAN."""
    found = set()
    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("192.0.2.1", 80))
        found.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        if probe is not None:
            probe.close()

    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            found.add(item[4][0])
    except OSError:
        pass

    private = []
    for address in found:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.version == 4 and parsed.is_private and not parsed.is_loopback and not parsed.is_link_local:
            private.append(str(parsed))
    return sorted(set(private))


def build_access_urls(host, port, lan_addresses=None):
    """Return browser URLs corresponding to the server bind address."""
    if host in ("0.0.0.0", ""):
        addresses = ["127.0.0.1"]
        addresses.extend(discover_lan_ipv4_addresses() if lan_addresses is None else lan_addresses)
        return [f"http://{address}:{port}" for address in dict.fromkeys(addresses)]
    return [f"http://{host}:{port}"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rulest web frontend (LAN-accessible)")
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0 for local-network access; use 127.0.0.1 for this PC only)",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    cleanup_stale_uploaded_files()
    server = RulestHTTPServer((args.host, args.port), Handler)
    actual_port = server.server_address[1]
    print()
    print("  RULEST  /  CONTROL CENTER")
    print(f"  Listening on: {args.host}:{actual_port}")
    for url in build_access_urls(args.host, actual_port):
        label = "This PC" if "127.0.0.1" in url else "LAN"
        print(f"  {label}: {url}")
    if args.host in ("0.0.0.0", ""):
        print()
        print("  NETWORK ACCESS / SECURITY")
        print("  A password is required before the panel or API can be used.")
        print("  First visit: create the application password in your browser.")
        print("  Complete first-time setup promptly; the first person to set it becomes admin.")
        print("  HTTP traffic is not encrypted; use a trusted LAN or a VPN/TLS tunnel.")
        print("  Never port-forward this development server to the public internet.")
        print("  For this PC only, start with: python app.py --host 127.0.0.1")
        print("  On Windows, allow Python through the firewall on Private networks only.")
    print()
    print("  Press Ctrl+C to stop the web interface.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("Stopping web interface...")
    finally:
        stop_active_run_for_shutdown()
        server.server_close()
        cleanup_uploaded_files(log_message=False)


if __name__ == "__main__":
    main()
