#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
RULEST_HOST="${RULEST_HOST:-0.0.0.0}"
exec python3 app.py --host "$RULEST_HOST" --port 8765
