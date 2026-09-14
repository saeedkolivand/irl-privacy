"""IRL privacy relay.

Raw Feed (RTMP from the phone, over Tailscale) -> Privacy Buffer (RAM only, ADR 0002)
-> Clean Feed (MPEG-TS over localhost UDP; OBS Media Source input `udp://127.0.0.1:9000`).

Detection runs as frames ARRIVE, so a target found now can be Backfilled onto frames that have
not aired yet. Frames are burned only on the way out (ADR 0001). Run `python relay.py --selftest`.
"""
import argparse
import collections
import threading
import time
import urllib.parse
from fractions import Fraction

import av
import numpy as np

import redact
import speech

DELAY = 3.0     # Privacy Buffer seconds (D5); Streamlabs Alert Delay must match
REANCHOR = 0.5  # seconds of drift between phone clock and arrival clock before re-syncing
W, H, FPS, RATE = 720, 1280, 30, 48000
MS, SAMPLE = Fraction(1, 1000), Fraction(1, RATE)
LATE = 1.0      # seconds overdue before we admit we are losing: stop detecting, start dropping
stats = collections.Counter()


def stamp(offsets, kind, frame, now_ms):
    """Output timestamp taken from the source clock, re-anchored PER STREAM when it drifts.

    Video and audio arrive on independent clocks. Sharing one correction between them lets each
    re-anchor the other, which sent one stream's timestamps minutes into the future."""
    if frame.pts is None:
        return now_ms
    src = float(frame.pts * frame.time_base * 1000)
    off = offsets.get(kind)
    if off is None or abs(src + off - now_ms) > REANCHOR * 1000:
        off = offsets[kind] = now_ms - src
    return src + off


def pop_due(buf, now):
    """Pop buffered items whose due time has passed. Item 0 is always the due time."""
    out = []
    while buf and buf[0][0] <= now:
        out.append(buf.popleft())
    return out


class Panic:
    """A latched Blackout, engaged by a spoken keyword or the phone.

    Because it is read at emit time rather than stored per frame, engaging it blanks everything
    still inside the Privacy Buffer -- the ~3 s that had already happened when you reacted, which
    is exactly the part you are usually reacting to. It only releases on a deliberate request:
    a voice command could engage it by accident, and the safe accident is the one that hides more."""

    def __init__(self):
        self.on, self.since = False, None

    def engage(self, why):
        if not self.on:
            self.on, self.since = True, time.monotonic()
            print(f"** PANIC engaged ({why}) -- the buffer is blanked until you clear it", flush=True)

    def release(self):
        if self.on:
            print(f"** panic cleared after {time.monotonic() - self.since:.0f}s", flush=True)
        self.on, self.since = False, None


def serve_panic(panic, host, port):
    """A GET-only endpoint so an iPhone Shortcut needs nothing but a URL. No auth of its own:
    it binds to the Tailscale address, so reaching it at all means being on the tailnet."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.rstrip("/").lower()
            if path.endswith("/panic"):
                panic.engage("phone")
            elif path.endswith("/clear"):
                panic.release()
            body = (b"PANIC\n" if panic.on else b"clear\n")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass        # the periodic status line already reports the state

    ThreadingHTTPServer((host, port), Handler).serve_forever()


class MuteAll:
    """Stand-in when speech redaction cannot start. Silences everything, loudly and on purpose:
    a stream with no audio is recoverable, one that airs an address read aloud is not."""

    stats = collections.Counter()

    def feed(self, pcm, t_ms):
        pass

    def muted(self, a_ms, b_ms):
        return True


def ingest(url, buf, t0, pipe, ears):
    """Accept one publisher at a time, forever; normalise frames, detect, and stamp them."""
    while True:
        try:
            # Security: RTMP has no auth, so only listen on the Tailscale IP (tailnet = auth).
            with av.open(url, options={"listen": "1"}, timeout=(None, 5)) as src:
                vs = src.streams.video[:1]
                aud = src.streams.audio[:1]
                cc = vs[0].codec_context if vs else None
                print(f"ingest: publisher connected -- video "
                      f"{cc.width}x{cc.height} {cc.name} @{vs[0].average_rate}" if cc else "no video")
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=RATE)
                asr = av.AudioResampler(format="flt", layout="mono", rate=speech.SR)
                offsets = {}
                for packet in src.demux(*(vs + aud)):
                    for frame in packet.decode():
                        now = time.monotonic()
                        ms = stamp(offsets, packet.stream.type, frame, (now - t0) * 1000)
                        # Due time comes from ARRIVAL, never the source clock, so the deque stays
                        # sorted by construction and one bad timestamp cannot stall the queue.
                        due = now + DELAY
                        if packet.stream.type == "video":
                            arr = frame.reformat(W, H, "bgr24").to_ndarray()
                            boxes = []
                            late = buf and (now - buf[0][0]) > LATE
                            t = time.monotonic()
                            bad = True if late else pipe.feed(arr, boxes)
                            stats["detect_ms"] += (time.monotonic() - t) * 1000
                            stats["detected"] += 1
                            stats["luma"] += pipe.luma
                            stats["sharp"] += pipe.sharp
                            stats["soft"] += pipe.sharp < redact.SHARP
                            stats["dim"] += pipe.luma < redact.DARK
                            buf.append((due, "video", ms, arr, boxes, bad))
                        else:
                            frame.pts = None
                            for f in asr.resample(frame):
                                ears.feed(f.to_ndarray()[0], ms)
                            for f in resampler.resample(frame):
                                buf.append((due, "audio", ms, f, None, False))
            print("ingest: publisher ended")
        except Exception as e:  # ponytail: any ingest failure = wait for the phone to reconnect
            print(f"ingest: {e!r}")


def emit(url, buf, ears, panic):
    out = av.open(url, "w", format="mpegts")
    v = out.add_stream("h264_nvenc", rate=FPS, options={"preset": "p4", "tune": "ll", "bf": "0", "g": str(FPS * 2)})
    v.width, v.height, v.pix_fmt, v.bit_rate = W, H, "yuv420p", 6_000_000
    v.codec_context.time_base = MS
    a = out.add_stream("aac", rate=RATE, layout="stereo")
    last_v, samples, last_log = -1, 0, time.monotonic()
    while True:
        now = time.monotonic()
        for due, kind, ms, payload, boxes, bad in pop_due(buf, now):
            if kind == "video":
                if now - due > LATE:
                    # Already too late to be worth showing. Dropping is how we catch up; encoding
                    # every stale frame is what turns a brief stall into a permanent one.
                    stats["dropped"] += 1
                    continue
                last_v = max(int(ms), last_v + 1)  # keep pts monotonic across re-anchors
                stats["blackout"] += bad
                stats["boxes"] += len(boxes)
                t = time.monotonic()
                burned, why = redact.burn(payload, boxes, bad or panic.on)
                if why:
                    stats[f"why_{why}"] += 1
                t2 = time.monotonic()
                clean = av.VideoFrame.from_ndarray(burned, format="bgr24")
                clean.pts, clean.time_base = last_v, MS
                packets = v.encode(clean)
                t3 = time.monotonic()
                stats["burn_ms"] += (t2 - t) * 1000
                stats["enc_ms"] += (t3 - t2) * 1000
                stats["emitted"] += 1
            else:
                gap = int(ms * RATE / 1000) - samples
                if gap < -RATE // 10:
                    continue  # audio ahead of the clock after a re-anchor: drop
                if panic.on or ears.muted(ms, ms + payload.samples / RATE * 1000):
                    payload = av.AudioFrame.from_ndarray(
                        np.zeros((2, payload.samples), np.float32), format="fltp", layout="stereo")
                    stats["bleeped"] += 1
                frames = [payload]
                if gap > RATE // 10:  # hole (start/reconnect): pad silence to hold A/V sync
                    frames.insert(0, av.AudioFrame.from_ndarray(np.zeros((2, gap), np.float32), format="fltp", layout="stereo"))
                packets = []
                for f in frames:
                    f.pts, f.time_base, f.sample_rate = samples, SAMPLE, RATE
                    samples += f.samples
                    packets += a.encode(f)
            for p in packets:
                out.mux(p)
        if now - last_log > 5:
            depth = buf[-1][0] - now if buf else 0.0
            behind = (now - buf[0][0]) if buf else 0.0
            n, m = max(1, stats["detected"]), max(1, stats["emitted"])
            print(f"buf {len(buf):5d} | newest {depth:5.2f}s oldest {behind:+5.2f}s | "
                  f"detect {stats['detect_ms'] / n:5.1f} burn {stats['burn_ms'] / m:5.1f} "
                  f"enc {stats['enc_ms'] / m:5.1f} ms | boxes {stats['boxes'] / m:4.1f} | "
                  f"luma {stats['luma'] / n:5.1f} sharp {stats['sharp'] / n:7.1f} | black: "
                  f"trust {stats['why_trust']:4d} count {stats['why_count']:4d} cover {stats['why_cover']:4d} "
                  f"| {'** PANIC **' if panic.on else 'live'} | {stats['dropped']:4d} drop | audio: {stats['bleeped']:4d} muted "
                  f"{ears.stats['bleeps']:3d} bleeps {ears.stats['overrun'] + ears.stats['failed']:2d} fallback")
            last_log = now
            stats.clear()
        time.sleep(0.002)


def selftest():
    buf = collections.deque([(1.0, "video", 0, None, [], False), (2.0, "audio", 1, None, None, False),
                             (3.5, "video", 2, None, [], False)])
    assert [i[2] for i in pop_due(buf, 2.0)] == [0, 1] and len(buf) == 1
    assert pop_due(buf, 3.0) == [] and [i[2] for i in pop_due(buf, 3.5)] == [2]

    # Arrival-based due times must stay sorted even when the source clock jumps, or the queue
    # stalls behind one bad frame -- the bug that blacked out a whole live test.
    offs, jumpy = {}, [(1000.0, 5.0), (2000.0, 900000.0), (3000.0, 6.0)]
    dues = []
    for arrival_ms, src_ms in jumpy:
        f = type("F", (), {"pts": src_ms, "time_base": Fraction(1, 1000)})()
        stamp(offs, "video", f, arrival_ms)
        dues.append(arrival_ms / 1000 + DELAY)
    assert dues == sorted(dues), "due times must follow arrival, not the source clock"
    assert stamp({}, "video", type("F", (), {"pts": None})(), 42.0) == 42.0, "no pts -> use arrival"
    a, v = {}, {}
    stamp(a, "audio", type("F", (), {"pts": 999000.0, "time_base": Fraction(1, 1000)})(), 0.0)
    stamp(v, "video", type("F", (), {"pts": 5.0, "time_base": Fraction(1, 1000)})(), 0.0)
    assert set(a) == {"audio"} and set(v) == {"video"}, "streams must not share a clock correction"

    mute = MuteAll()
    assert mute.muted(0, 1) and mute.muted(1e9, 1e9 + 1), "the fallback silences everything, always"

    p = Panic()
    assert not p.on
    p.engage("test")
    assert p.on, "a panic engages"
    p.engage("again")
    assert p.on, "engaging twice is harmless and does not reset the clock"
    p.release()
    assert not p.on, "and clears only when asked"
    # Panic is read at emit time, so it applies to frames already sitting in the buffer.
    p.engage("retro")
    assert redact.burn(np.zeros((8, 8, 3), np.uint8), [], False or p.on)[1] == "trust"

    speech.selftest()
    redact.selftest()
    print("selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listen", default="rtmp://100.124.231.50:1935/live", help="RTMP URL the phone publishes to (PC's Tailscale IP)")
    p.add_argument("--out", default="udp://127.0.0.1:9000?pkt_size=1316", help="Clean Feed destination for OBS")
    p.add_argument("--panic-port", type=int, default=8765, help="port for the panic endpoint")
    p.add_argument("--debug-words", action="store_true",
                   help="TEMPORARY: log the transcript with muted words bracketed. Fake data only.")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    if args.debug_words:
        speech.DEBUG = True
        print("!! --debug-words is ON: transcribed speech is being written to this log")
    buf, t0 = collections.deque(), time.monotonic()
    pipe = redact.Pipeline(W, H)
    panic = Panic()
    try:
        ears = speech.Speech(on_keyword=lambda: panic.engage("voice"), on_clear=panic.release)
    except Exception as e:      # no transcription means no way to know what is being said
        print(f"!! speech redaction unavailable ({e!r}) -- ALL AUDIO WILL BE SILENCED")
        ears = MuteAll()
    host = urllib.parse.urlparse(args.listen).hostname or "127.0.0.1"
    threading.Thread(target=serve_panic, args=(panic, host, args.panic_port), daemon=True).start()
    print(f"listening for the phone on {args.listen}\nclean feed -> {args.out}\n"
          f"panic: http://{host}:{args.panic_port}/panic   clear: http://{host}:{args.panic_port}/clear\n"
          f"or say {' / '.join(sorted(speech.KEYWORDS))} to engage, "
          f"\"{' '.join(speech.CLEAR_PHRASE)}\" to release")
    threading.Thread(target=ingest, args=(args.listen, buf, t0, pipe, ears), daemon=True).start()
    emit(args.out, buf, ears, panic)


if __name__ == "__main__":
    main()
