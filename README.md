# irl-privacy

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

## Why a buffer

The stream airs ~3 seconds behind reality. That delay is the whole design: when a target is first
detected, its blur is painted **backwards** over frames that have not aired yet, so an object is
covered from the first frame it appeared in rather than the frame it was recognised in. It also
means a panic blanks the three seconds that already happened — usually the part you are reacting to.

See `docs/adr/0001-fail-closed-lookahead-before-obs.md`.

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

## Setup

```bash
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install "onnxruntime-gpu[cuda,cudnn]"   # after the CPU build, to win
.venv\Scripts\python fetch_models.py
```

Then on the phone: Tailscale signed into the same account, and StreamHand (or Streamlabs Mobile)
pointed at `rtmp://<your-tailscale-ip>:1935/live` with any stream key.

In OBS: a Media Source with *local file* unchecked and input `udp://127.0.0.1:9000`, plus a BRB
scene. Set the Streamlabs alert delay to 3 s so alerts land with the delayed video.

```bash
.venv\Scripts\python relay.py            # --listen defaults to a Tailscale IP: change it to yours
.venv\Scripts\python upload.py           # optional: phone -> PC upload page for test footage
```

## Panic

Latched blackout, engaged by voice or phone, applied to everything still in the buffer.

- engage: say **"privacy"** or **"blackout"**, or `GET :8765/panic`
- release: say **"all clear"** (two adjacent words), or `GET :8765/clear`

Releasing is deliberately harder than engaging. Engaging by accident costs a grey stream; releasing
by accident un-hides the thing you panicked about. If both are heard at once, engaging wins.

Both endpoints bind to the Tailscale address and have no auth of their own — reaching them already
means being on the tailnet.

## Checking it

```bash
python relay.py --selftest        # runs relay + redact + speech checks
python redact.py bench            # per-detector ms
python redact.py clip IN.mp4 OUT.mp4
```

The self-checks are mostly regressions for bugs that only appeared on real footage — a queue stall
that blacked out a whole stream, a box-merge that cascaded into a frame-sized blur, contractions
being silenced as if they were surnames.

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
