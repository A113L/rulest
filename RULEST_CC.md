# Rulest Control Center

A dark Cleanlooks-inspired web frontend for the updated Rulest GPU-Compatible Hashcat Rules Engine. This bundle includes the engine from the newly supplied `rulest_v3_latest` archive, together with the graphical control panel, direct file uploads, complete CLI configuration, live phase/log monitoring, safe-checkpoint pause/resume/partial-stop controls, and generated-file downloads.

## Requirements

- Python 3.9 or newer
- Dependencies from `requirements.txt` (`numpy`, `pyopencl`, `tqdm`)
- A working OpenCL runtime and compatible GPU driver for GPU acceleration

## Install and start

### Windows

1. Extract this ZIP.
2. Open the project folder.
3. Install dependencies once:

   ```bat
   python -m pip install -r requirements.txt
   ```

4. Double-click `start_gui.bat`.
5. The terminal displays the local URL and any detected LAN URL, for example `http://192.168.1.25:8765`.

### Linux / macOS

```bash
python3 -m pip install -r requirements.txt
chmod +x start_gui.sh
./start_gui.sh
```

The terminal prints the local URL and discovered private IPv4 LAN address(es).

On the first visit, open the local URL on the computer running RULEST and create your application password (minimum 12 characters). The first person to complete this one-time setup becomes the administrator. Once the password is set, other devices can sign in using the same password.

## Open from another device on the same network

1. Start the server on the computer running RULEST.
2. Keep that computer and the other device connected to the same trusted Wi-Fi/LAN.
3. Read the `LAN` address printed in the server terminal, such as `http://192.168.1.25:8765`.
4. Open that exact address in the other device's browser.

Do not use `127.0.0.1` from the other device; that address points back to the device itself. If Windows Firewall asks, allow Python on **Private networks** only. If the LAN address is unreachable, check that the computer and client are on the same subnet and that the firewall allows inbound TCP port `8765`.

## Password authentication and network security

The entire Control Center is password-protected, including its APIs, file uploads, run controls, GPU diagnostics, logs and downloads.

- On the first visit, create an application password (minimum 12 characters). There is no default password.
- The password is stored as a salted PBKDF2-HMAC-SHA256 verifier in `auth.json`; the plain-text password is never written to disk.
- Sign-in is rate-limited after five incorrect attempts from the same client address (five-minute lockout).
- Sessions use HttpOnly, SameSite=Strict cookies, expire after 12 hours, and are cleared when the server restarts.
- Use **Change password** in the top bar to update the password; other active sessions are revoked.
- Use **Log out** in the top bar to end the current browser session.
- Keep `auth.json` private and include it in your backups. To reset the password manually, stop the server and remove `auth.json`; the next visit will show the first-time password setup screen.

The server binds to `0.0.0.0` by default so devices on the local network can connect. **Important:** this built-in server uses plain HTTP, so a password does not encrypt traffic on the network. Use it only on a trusted LAN, or place it behind a trusted VPN/TLS reverse proxy/SSH tunnel. Do not port-forward port `8765` or expose it directly to the public internet.

For this PC only, start with:

```bash
python app.py --host 127.0.0.1 --port 8765
```

To use another port:

```bash
python app.py --host 0.0.0.0 --port 9000
```

On Windows, the launch script respects `RULEST_HOST` if you want to override the default, e.g. set `RULEST_HOST=127.0.0.1` before launching.

## Using the interface

1. Upload BASE and TARGET wordlists in **Input files**. A custom seed-rules file is optional.
2. Configure engine, Stage 0, genetic algorithm, rule selection and diagnostic settings. Use **Enable Stage 0** and **Enable Genetic** to turn the optional stages on or off.
3. Click **Launch engine**. The phase card follows actual engine output; the live log is saved to `runs/<run-id>/engine.log`.
4. Use **Pause**, **Resume**, or **Save & stop** to control a running job at safe engine checkpoints.
5. During CELF selection, the live log now emits a periodic non-TTY heartbeat with selected-rule count, target coverage, remaining heap size, rescoring counters, batch size and elapsed time—even when profiling is disabled.
6. Use the log wrap/expand controls and download generated outputs when the run ends.
7. Uploaded BASE, TARGET and optional seed files are temporary server-side copies. They are automatically deleted when a run completes, fails or is stopped. Leftover uploads from an unexpected server shutdown are removed the next time the server starts. Generated outputs and `engine.log` are preserved.

## GPU diagnostics

The panel checks OpenCL and reports the Python interpreter, PyOpenCL status, platforms and devices. If PyOpenCL is missing, install it into the same interpreter used by the app:

```bash
python -m pip install --upgrade pyopencl
```

Installing PyOpenCL does not install the GPU manufacturer's OpenCL driver/runtime. If no OpenCL platform or GPU appears, install or repair the official graphics driver and the manufacturer's OpenCL runtime, then restart and check again.

### Clear logs from the web interface

Open **Live activity** and click **Clear log history**. After confirmation, the Control Center clears the visible log history and truncates the current run's saved `engine.log`. New engine messages continue to stream normally. Generated ruleset files are not deleted. Other browser sessions connected to the same server also reset their displayed history automatically.
