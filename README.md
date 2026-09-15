# irl-privacy

[![License: MIT](https://img.shields.io/github/license/saeedkolivand/irl-privacy)](LICENSE)
[![selftest](https://img.shields.io/github/actions/workflow/status/saeedkolivand/irl-privacy/selftest.yml?label=selftest)](.github/workflows/selftest.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue)](.python-version)

Redacts third-party personal information out of a first-person IRL livestream before any viewer
sees it. Built for Ray-Ban Meta glasses streaming to Kick/Twitch, where the glasses and phone are
far too weak to run detection and the phone apps that can stream have no redaction at all.

```
Glasses ──DAT 720x1280──▶ iPhone (StreamHand) ──RTMP over Tailscale──▶ relay.py
                                                                          │
                                       Privacy Buffer (3 s, RAM only)     │
                                       detect → blur → Backfill           │
                                       transcribe → silence names/numbers │
                                                                          ▼
                                            OBS (scenes, alerts) ──▶ Kick / Twitch
```

Everything runs on one PC with an NVIDIA GPU. The phone never holds a platform stream key, so if
the link dies the stream shows a BRB scene rather than an unredacted feed.

Built for Windows. It starts and self-checks on macOS and Linux too, but without an NVIDIA GPU
detection falls back to the CPU (far below 30 fps) and audio is silenced wholesale, because
`speech.py` only builds a CUDA transcriber.

## Contents

- [What it redacts](#what-it-redacts)
- [Why a buffer](#why-a-buffer)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Panic](#panic)
- [Turning redaction off](#turning-redaction-off)
- [Checking it](#checking-it)
- [Known gaps](#known-gaps)
- [Tuning](#tuning)
- [Contributing](#contributing)
- [License](#license)

## What it redacts

| | How |
|---|---|
| Faces | YuNet (MIT) |
| Text: addresses, house numbers, signs, labels, screens | PP-OCRv3 DB (Apache-2.0) |
| Licence plates | via the text detector — plate characters are text |
| Barcodes / QR | YOLOv8s barcode model (AGPL-3.0) |
| Spoken numbers, names, postcodes | faster-whisper + word rules (`speech.py`) |

Nothing is ever read to decide: text is located and blurred, never transcribed to judge whether it
is private. Speech is transcribed only to pick which milliseconds to silence, and the transcript is
discarded (`docs/adr/0002-raw-footage-ram-only.md`).

## Why a buffer

The stream airs ~3 seconds behind reality. That delay is the whole design: when a target is first
detected, its blur is painted **backwards** over frames that have not aired yet, so an object is
covered from the first frame it appeared in rather than the frame it was recognised in. It also
means a panic blanks the three seconds that already happened — usually the part you are reacting to.

See `docs/adr/0001-fail-closed-lookahead-before-obs.md`.

## Prerequisites

- A PC with an NVIDIA GPU (CUDA) — this is where detection, blur, and encode all run
- Python 3.14 (see `.python-version`)
- [Tailscale](https://tailscale.com/), signed into the same account on the PC and the phone
- A phone RTMP app with no built-in delay, e.g. StreamHand or Streamlabs Mobile
- [OBS](https://obsproject.com/), to receive the Clean Feed and forward it to Kick/Twitch

## Quick start

```powershell
.\start.ps1             # first run builds the venv and fetches the models, then relays
.\start.ps1 -Upload     # also serves the phone -> PC upload page for test footage
.\start.ps1 -Selftest   # the self-checks, no stream
```

macOS, Linux, and Git Bash on Windows get the same thing as a shell script:

```bash
./start.sh              # first run builds the venv and fetches the models, then relays
./start.sh --upload     # also serves the phone -> PC upload page for test footage
./start.sh --selftest   # the self-checks, no stream
```

It prints the RTMP URL to point the phone at — `--listen` and `--host` default to this PC's
Tailscale IP, so there is nothing to look up. The first run installs ~7 GB of CUDA and torch
wheels and takes a while; later runs start straight away.

Then on the phone: Tailscale signed into the same account, and StreamHand (or Streamlabs Mobile)
pointed at the URL the relay printed, with any stream key.

In OBS: a Media Source with *local file* unchecked and input `udp://127.0.0.1:9000`, plus a BRB
scene. Set the Streamlabs alert delay to 3 s so alerts land with the delayed video.

<details><summary>Or by hand, without the script</summary>

```powershell
py -3.14 -m venv .venv                                              # python3.14 on macOS/Linux
.venv\Scripts\python -m pip install -r requirements.txt             # .venv/bin/python elsewhere
.venv\Scripts\python -m pip install "onnxruntime-gpu[cuda,cudnn]"   # after the CPU build, to win
.venv\Scripts\python -m pip install ultralytics                     # export-time only, for the barcode model
.venv\Scripts\python fetch_models.py
.venv\Scripts\python relay.py
```

Skip the `onnxruntime-gpu` line on macOS — there are no CUDA wheels for it.

</details>

## Panic

Latched blackout, engaged by voice or phone, applied to everything still in the buffer.

- engage: say **"privacy"** or **"blackout"**, or `GET :8765/panic`
- release: say **"all clear"** (two adjacent words), or `GET :8765/clear`

Releasing is deliberately harder than engaging. Engaging by accident costs a grey stream; releasing
by accident un-hides the thing you panicked about. If both are heard at once, engaging wins.

Both endpoints bind to the Tailscale address and have no auth of their own — reaching them already
means being on the tailnet. See `SECURITY.md`.

## Turning redaction off

Sometimes there is nobody else in frame and nothing to hide. `redaction off` stops all of it —
no blur, no Bleep, and no Auto Trigger either, so a dark or blurry frame airs as it is.

- off: say **"redaction off"** (two adjacent words), or `GET :8765/raw`
- on: say **"redact"**, or `GET :8765/redact`

The friction runs the opposite way to a Panic, for the same reason. Turning it off is the only
command here that can expose somebody else, so it takes a phrase; going back to redacted is one
word. If both are heard at once, redacting wins, and a Panic cancels it outright — so "all clear"
never drops you back into an unredacted stream. While it is on the status line reads `** RAW **`.

## Checking it

```powershell
.\start.ps1 -Selftest             # every self-check, no stream (./start.sh --selftest elsewhere)
python relay.py --selftest        # runs relay + redact + speech checks
python redact.py bench            # per-detector ms
python redact.py clip IN.mp4 OUT.mp4
```

The self-checks are mostly regressions for bugs that only appeared on real footage — a queue stall
that blacked out a whole stream, a box-merge that cascaded into a frame-sized blur, contractions
being silenced as if they were surnames. The same commands run in CI on every push.

## Known gaps

- **A bare barcode with no printed text beside it is not blurred.** The barcode model scores ~0 on
  plain code stickers (ultralytics' own runtime agrees, so it is not a wiring fault). Real carrier
  labels carry an address block, which the text detector finds and the padding then spreads over.
- **A misheard name can be spoken aloud.** Name detection keys off capitalisation, so a surname
  transcribed as ordinary lowercase words is not silenced. Low confidence catches many but not all.
- **Over-muting is normal.** "a busy one today" loses a word because "one" is a number.
- English only, until `speech.Speech(language=...)` is changed.

## Tuning

Thresholds in `redact.py` were set against a real glasses feed, not stock footage, and the
difference mattered: the feed runs at luma 19-27, so a "too dark to trust" threshold of 32 blacked
out every frame in ordinary room light. If you change any of them, re-run the self-checks — several
exist purely to stop an old mistake coming back.

## Contributing

PRs welcome — see `CONTRIBUTING.md` for the self-check convention this repo uses instead of a test
suite, and the terminology defined in `CONTEXT.md`.

## License

MIT, see `LICENSE`. The barcode detector model fetched by `fetch_models.py` is AGPL-3.0 and is not
vendored in this repo — see `.gitignore`.
