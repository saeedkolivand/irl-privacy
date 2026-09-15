# One command to get running. Creates the venv and fetches the models on first run, then relays.
#
#   .\start.ps1             # relay only
#   .\start.ps1 -Upload     # relay, plus the phone upload page in a second window
#   .\start.ps1 -Selftest   # the self-checks, no stream
param([switch]$Upload, [switch]$Selftest)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    py -3.14 -m venv .venv
    & $py -m pip install -r requirements.txt
    & $py -m pip install "onnxruntime-gpu[cuda,cudnn]"   # after the CPU build, so it wins the import path
    & $py -m pip install ultralytics                     # only fetch_models.py needs it, and only once
}
& $py fetch_models.py   # every step self-skips if the file is already there

if ($Selftest) {
    foreach ($m in "redact", "speech", "upload", "relay") { & $py "$m.py" --selftest }
    exit
}
if ($Upload) { Start-Process $py -ArgumentList "upload.py" }
& $py relay.py   # --listen defaults to this PC's Tailscale IP
