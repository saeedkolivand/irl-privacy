#!/bin/sh
# One command to get running, for macOS, Linux and Git Bash. Creates the venv and fetches the
# models on first run, then relays. Windows PowerShell has start.ps1 instead.
#
#   ./start.sh             # relay only
#   ./start.sh --upload    # relay, plus the phone upload page in the background
#   ./start.sh --selftest  # the self-checks, no stream
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3.14 -m venv .venv 2>/dev/null || py -3.14 -m venv .venv 2>/dev/null || python3 -m venv .venv
    fresh=1
fi
# Git Bash on Windows puts the venv interpreter in Scripts/, everywhere else it is bin/
if [ -d .venv/Scripts ]; then py=.venv/Scripts/python.exe; else py=.venv/bin/python; fi

if [ -n "$fresh" ]; then
    "$py" -m pip install -r requirements.txt
    # No CUDA wheels exist for macOS; faster-whisper's plain onnxruntime is what runs there
    case "$(uname -s)" in
        Darwin) ;;
        *) "$py" -m pip install "onnxruntime-gpu[cuda,cudnn]" ;;   # after the CPU build, so it wins the import path
    esac
    "$py" -m pip install ultralytics   # only fetch_models.py needs it, and only once
fi
"$py" fetch_models.py   # every step self-skips if the file is already there

case "$1" in
    --selftest) for m in redact speech upload relay; do "$py" "$m.py" --selftest; done; exit ;;
    --upload)   "$py" upload.py & ;;
esac
exec "$py" relay.py   # --listen defaults to this PC's Tailscale IP
