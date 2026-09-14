"""Video redaction: find Redaction Targets, blur them, Backfill new ones into the Privacy Buffer.

Detection runs on the GPU through onnxruntime (CUDA): YuNet face (MIT), PP-OCRv3 DB text
(Apache-2.0), YOLOv8s barcode/QR (AGPL-3.0). Never reads text, only locates it (ADR 0001, D3).

    python redact.py bench                    # per-detector ms on one 720x1280 frame
    python redact.py clip IN.mp4 OUT.mp4      # redact a clip, same logic as the live buffer
    python redact.py --selftest
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).parent / "models"
FACE_MULT = 64           # YuNet needs both input dims divisible by 64
CODE_MULT = 32           # YOLOv8 stride
TEXT_SIZE = (544, 960)   # DB input, multiples of 32, ~portrait aspect
FACE_CONF, CODE_CONF, NMS = 0.55, 0.35, 0.4
DB_THRESH, DB_UNCLIP = 0.25, 2.2

PAD = 0.28              # box grown by this fraction per side before blurring
NEAR = 0.30             # a box spanning this much of the frame counts as "close"
# Padding is applied to EVERY side, so p grows a box to (1 + 2p) times its width. A parcel is
# roughly 2-3x its label, so ~0.45 is the whole-parcel grow; 1.20 made one close label cover the
# entire frame, which then tripped the coverage Blackout on essentially every frame.
NEAR_PAD = 0.45         # extra padding at full proximity, to swallow the whole parcel (D3)
MAX_PAD = 0.15          # but never grow a box by more than this share of the frame, per side
BACKFILL = 45           # frames a newly-seen target is painted backwards (1.5 s at 30 fps)
EVERY = 1               # detect on every Nth frame; the GPU stack affords every frame
CODE_EVERY = 3          # the barcode model is the slowest and weakest -- run it at a third rate
# Blackout is for "detection cannot be trusted", NOT for "the targets we found are large". A close
# parcel legitimately fills the frame; blurring it is the right answer, greying the whole view is
# not -- and once the targets are blurred individually, a Blackout adds no privacy at all, only the
# remaining sliver of background. So this fires only when the frame is essentially all target, where
# a Blackout is simply the cheaper way to draw the same picture. On the real glasses feed 0.35 and
# then 0.75 both blacked out nearly every frame with only 3-4 boxes present.
COVER = 0.92            # boxes over this share of the frame -> Blackout
# This existed to bound merge()'s O(n^2) cost. merge() is gone and blur_boxes is linear, so the
# only reason left is a sanity ceiling. At 60 a busy shopping street (76 shop signs) greyed out
# entirely, when blurring the signs individually leaves the road, sky and people perfectly visible.
MAX_BOXES = 250         # more separate targets than this -> Blackout rather than blur each
# The glasses run dim: the live feed averages luma 30-37 indoors and detection still works there.
DARK = 15               # mean luma below this -> Blackout
SHARP = 14              # Laplacian variance below this -> Blackout (motion blur)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def coverage(boxes, w, h, s=8):
    """Exact fraction of the frame under the boxes, measured on a 1/8-scale mask.

    Summing box areas double-counts overlaps, and merging them into bounding rectangles inflates
    far worse: unions cascade until a few scattered detections become one frame-sized box."""
    if not boxes or not w or not h:
        return 0.0
    m = np.zeros((max(1, h // s), max(1, w // s)), np.uint8)
    for x, y, bw, bh in boxes:
        m[max(0, y // s):(y + bh) // s, max(0, x // s):(x + bw) // s] = 1
    return float(m.mean())


def near_pad(box, w, h):
    """A target filling much of the frame is close, so the blur widens to swallow whatever it is
    printed on -- the whole parcel in your hands, just the label on a doorstep across the yard."""
    rel = max(box[2] / float(w), box[3] / float(h))
    return PAD + NEAR_PAD * min(1.0, rel / NEAR)


def pad_clip(box, w, h, pad=PAD):
    x, y, bw, bh = box
    # Proportional padding is right for a small label but absurd for a target that already fills
    # much of the view -- it needs no help covering what it is printed on. Cap the grow in absolute
    # terms, or one large detection expands over the entire frame on its own.
    dx = min(bw * pad, w * MAX_PAD)
    dy = min(bh * pad, h * MAX_PAD)
    x0, y0 = max(0, int(x - dx)), max(0, int(y - dy))
    x1, y1 = min(w, int(x + bw + dx)), min(h, int(y + bh + dy))
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def poly_box(points, sx=1.0, sy=1.0):
    """Axis-aligned box around an Nx2 polygon, scaled back to frame coordinates."""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return (x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)


def pad_to(frame, mult, cache={}):
    """Grow the canvas to a multiple of `mult` without resampling. These models are fully
    convolutional, so padding keeps every original pixel where downscaling loses small faces.
    The canvas is reused between frames -- at 30 fps the allocations alone cost real milliseconds."""
    h, w = frame.shape[:2]
    hh, ww = -(-h // mult) * mult, -(-w // mult) * mult
    if (hh, ww) == (h, w):
        return frame
    key = (hh, ww, frame.dtype.str)
    out = cache.get(key)
    if out is None:
        out = cache[key] = np.zeros((hh, ww, 3), frame.dtype)
    out[:h, :w] = frame
    return out


def nms(boxes, scores):
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes([list(map(float, b)) for b in boxes], list(map(float, scores)), 0.0, NMS)
    return [boxes[i] for i in np.array(keep).flatten()]


def _session(name):
    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(MODELS / name), so,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


class Detectors:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.face_s = _session("face_yunet_dyn.onnx")
        self.text_s = _session("text_ppocrv3.onnx")
        self.code_s = _session("barcode_yolov8s.onnx")
        self.n, self.held_codes = 0, []

    def faces(self, frame):
        """YuNet is anchor-free: each cell of the stride-8/16/32 grids predicts one box."""
        img = pad_to(frame, FACE_MULT)
        blob = cv2.dnn.blobFromImage(img)     # C-speed BGR->NCHW float32; no scaling, as YuNet wants
        out = self.face_s.run(None, {self.face_s.get_inputs()[0].name: blob})
        r = dict(zip([o.name for o in self.face_s.get_outputs()], out))
        boxes, scores = [], []
        for s in (8, 16, 32):
            score = np.sqrt(np.clip(r[f"cls_{s}"][0, :, 0], 0, 1) * np.clip(r[f"obj_{s}"][0, :, 0], 0, 1))
            idx = np.nonzero(score > FACE_CONF)[0]
            if not len(idx):
                continue
            b, cols = r[f"bbox_{s}"][0][idx], blob.shape[3] // s
            cx = (idx % cols + b[:, 0]) * s
            cy = (idx // cols + b[:, 1]) * s
            bw, bh = np.exp(b[:, 2]) * s, np.exp(b[:, 3]) * s
            boxes += [(cx[i] - bw[i] / 2, cy[i] - bh[i] / 2, bw[i], bh[i]) for i in range(len(idx))]
            scores += list(score[idx])
        return nms(boxes, scores)

    def text_regions(self, frame):
        """DB emits a text-probability map; every blob in it is a Redaction Target."""
        # One C call for resize + mean + scale + NCHW. The three ImageNet stds differ by ~1%, so a
        # single scalar stands in for them; validate_port.py confirms recall is unchanged.
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 57.63, TEXT_SIZE, (123.675, 116.28, 103.53))
        prob = self.text_s.run(None, {self.text_s.get_inputs()[0].name: blob})[0]
        mask = (np.squeeze(prob) > DB_THRESH).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = self.w / TEXT_SIZE[0], self.h / TEXT_SIZE[1]
        grow = (DB_UNCLIP - 1) * 0.5      # stand-in for DB's polygon unclip; we pad again later
        out = []
        for c in cnts:
            if cv2.contourArea(c) < 6:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            out.append(((x - bw * grow) * sx, (y - bh * grow) * sy,
                        bw * (1 + 2 * grow) * sx, bh * (1 + 2 * grow) * sy))
        return out

    def codes(self, frame):
        """Barcodes and QR codes carry the tracking number and the address; cv2's own detectors
        missed them on real parcels, so this is a model trained for exactly that."""
        img = pad_to(frame, CODE_MULT)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (0, 0), (0, 0, 0), swapRB=True)
        o = self.code_s.run(None, {self.code_s.get_inputs()[0].name: blob})[0][0].T   # (n, 4+nc)
        score = o[:, 4:].max(1)
        idx = np.nonzero(score > CODE_CONF)[0]
        boxes = [(o[i, 0] - o[i, 2] / 2, o[i, 1] - o[i, 3] / 2, o[i, 2], o[i, 3]) for i in idx]
        return nms(boxes, score[idx])

    def detect(self, frame):
        # Codes are stickers on static objects, so a third of the rate costs nothing in practice
        # and Backfill covers the gaps. It is also the weakest detector -- see PHASE2 notes.
        self.n += 1
        if self.n % CODE_EVERY == 0:
            self.held_codes = self.codes(frame)
        boxes = self.faces(frame) + self.text_regions(frame) + self.held_codes
        return [pad_clip(b, self.w, self.h, near_pad(b, self.w, self.h))
                for b in boxes if b[2] > 1 and b[3] > 1]


def quality(frame):
    """Brightness and sharpness. Reported as numbers so thresholds can be tuned against a real
    feed without ever writing a Raw Feed frame to disk (ADR 0002)."""
    gray = cv2.cvtColor(cv2.resize(frame, (180, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_bad(frame):
    """Auto Trigger: too dark or too motion-blurred for detection to be trusted (D7)."""
    luma, sharp = quality(frame)
    return bool(luma < DARK or sharp < SHARP)


def blur_boxes(frame, boxes):
    """Mosaic down, smooth back up. The downscale is what destroys the detail -- it averages pixels
    away and cannot be inverted. The smooth upscale is only what makes it look like a blur.

    A GaussianBlur on top of this cost 124 ms/frame (sigma 12 over near_pad-sized boxes, 47 of them)
    and added no privacy whatsoever, because the information was already gone."""
    for x, y, w, h in boxes:
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        small = cv2.resize(roi, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
        frame[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame


def blackout(frame):
    """Must survive a close-up: big glyphs stay legible through a gentle whole-frame blur."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 90), max(1, h // 90)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def burn(frame, boxes, bad):
    """Apply a frame's accumulated redaction, the last thing to touch a frame before it airs.

    Returns (frame, reason) -- "" when the frame came through normally. The caller needs the reason
    because a stream that is quietly grey most of the time is a problem, even though it is a safe
    one, and three different rules can cause it."""
    if bad:
        return blackout(frame), "trust"          # too dark, too blurred, or we fell behind
    if len(boxes) > MAX_BOXES:
        return blackout(frame), "count"          # a scene so dense with print that boxes stop meaning anything
    h, w = frame.shape[:2]
    if coverage(boxes, w, h) > COVER:
        return blackout(frame), "cover"
    return blur_boxes(frame, boxes), ""


class Backfill:
    """Paints a newly-seen target backwards over frames still in the Privacy Buffer."""

    def __init__(self, frames=BACKFILL):
        self.prev = []
        self.recent = collections.deque(maxlen=frames)

    def feed(self, boxes, sink):
        """sink: the mutable box list of the current frame. Returns it."""
        new = [b for b in boxes if all(iou(b, p) < 0.2 for p in self.prev)]
        if new:
            for past in self.recent:
                past.extend(new)
        sink.extend(boxes)
        self.recent.append(sink)
        self.prev = boxes
        return sink


class Pipeline:
    """Detect / hold / Backfill over a stream of frames. Shared by the live relay and run_clip."""

    def __init__(self, w, h, every=EVERY):
        self.det = Detectors(w, h)
        self.back = Backfill()
        self.every, self.n, self.held = every, 0, []
        self.luma, self.sharp = 0.0, 0.0   # last frame's metrics, for the relay's log

    def feed(self, frame, sink):
        """Fill `sink` (a frame's mutable box list). Returns True when the frame needs a Blackout."""
        self.n += 1
        self.luma, self.sharp = quality(frame)
        bad = self.luma < DARK or self.sharp < SHARP
        if bad:
            self.back.feed([], sink)
            return True
        if self.n % self.every == 0:
            self.held = self.det.detect(frame)
        self.back.feed(self.held, sink)
        return False


def run_clip(src, dst, every=EVERY):
    # PyAV + NVENC rather than cv2.VideoCapture/VideoWriter: OpenCV does both decode and encode on
    # the CPU, which left the GPU at 12% while the machine worked hard for no reason.
    import av
    from fractions import Fraction

    sc = av.open(src)
    vs = sc.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    rate = vs.average_rate or Fraction(30, 1)
    pipe = Pipeline(w, h, every)
    dc = av.open(dst, "w")
    ov = dc.add_stream("h264_nvenc", rate=rate, options={"preset": "p4"})
    ov.width, ov.height, ov.pix_fmt, ov.bit_rate = w, h, "yuv420p", 6_000_000

    def write(img):
        for p in ov.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            dc.mux(p)

    pending = collections.deque()  # (frame, boxes, bad) still inside the simulated Privacy Buffer
    stats = collections.Counter()
    t0 = time.monotonic()
    for vframe in sc.decode(video=0):
        frame = vframe.to_ndarray(format="bgr24")
        sink = []
        bad = pipe.feed(frame, sink)
        stats["frames"] += 1
        stats["blackout"] += bad
        pending.append((frame, sink, bad))
        while len(pending) > BACKFILL:          # frame has left the buffer: burn in and write
            f, bx, b = pending.popleft()
            stats["boxes"] += len(bx)
            img, why = burn(f, bx, b)
            stats["blacked"] += bool(why)
            write(img)
    for f, bx, b in pending:
        stats["boxes"] += len(bx)
        img, why = burn(f, bx, b)
        stats["blacked"] += bool(why)
        write(img)
    for p in ov.encode():                      # flush whatever NVENC is still holding
        dc.mux(p)
    sc.close()
    dc.close()
    el = time.monotonic() - t0
    print(f"{Path(src).name:34s} {stats['frames']:4d} frames  {stats['boxes']:5d} boxes  "
          f"{stats['blacked']:4d} blackout ({100 * stats['blacked'] / max(1, stats['frames']):3.0f}%)  "
          f"{stats['frames'] / el:5.1f} fps")
    return stats


def bench():
    p = MODELS / "_bench.png"
    frame = cv2.imread(str(p)) if p.exists() else np.random.randint(0, 255, (1280, 720, 3), np.uint8)
    d = Detectors(frame.shape[1], frame.shape[0])
    total = 0.0
    for name, fn in [("face", d.faces), ("text", d.text_regions), ("codes", d.codes),
                     ("quality", quality_bad), ("detect", d.detect)]:
        fn(frame)
        t = time.monotonic()
        for _ in range(20):
            fn(frame)
        ms = (time.monotonic() - t) * 50
        total = ms if name == "detect" else total
        print(f"{name:8s} {ms:6.1f} ms/frame")
    print(f"-> {1000 / total:.0f} fps of detection headroom (need 30)")


def selftest():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
    assert pad_clip((0, 0, 10, 10), 100, 100, 0.5) == (0, 0, 15, 15)          # clipped at the edge
    assert pad_clip((50, 50, 10, 10), 100, 100, 0.5) == (45, 45, 20, 20)
    assert poly_box([[10, 20], [30, 20], [30, 60], [10, 60]]) == (10, 20, 20, 40)
    assert abs(coverage([(0, 0, 80, 80)], 160, 160) - 0.25) < 0.02
    assert abs(coverage([(0, 0, 80, 80), (0, 0, 80, 80)], 160, 160) - 0.25) < 0.02, "no double count"
    # Two boxes in opposite corners cover an eighth, not the whole frame via a bounding rectangle.
    assert coverage([(0, 0, 40, 40), (120, 120, 40, 40)], 160, 160) < 0.2, "overlaps must not cascade"

    far, near = (350, 630, 20, 20), (210, 490, 300, 300)                      # in a 720x1280 frame
    assert near_pad(far, 720, 1280) < near_pad(near, 720, 1280), "closer target, wider blur"
    assert near_pad(near, 720, 1280) == PAD + NEAR_PAD, "a close one swallows what it is printed on"
    assert pad_clip(far, 720, 1280, near_pad(far, 720, 1280))[2] < 2 * far[2], "distant: label only"
    grown = pad_clip(near, 720, 1280, near_pad(near, 720, 1280))
    assert grown[2] > 1.5 * near[2], "close: the blur spreads onto the parcel behind the label"
    assert grown[2] <= near[2] + 2 * 720 * MAX_PAD, "but the grow is capped"
    # The live failure: one big detection expanding until it covered the frame, every frame.
    assert pad_clip((0, 0, 500, 900), 720, 1280, PAD + NEAR_PAD)[2] < 720, "already-large stays bounded"

    back, sinks = Backfill(frames=3), [[] for _ in range(10)]
    for i, s in enumerate(sinks):
        back.feed([(100, 100, 20, 20)] if i >= 6 else [], s)
    assert sinks[5] and sinks[2] == [], "Backfill reaches frames still buffered, not beyond the window"
    assert all(s for s in sinks[6:]), "every frame from first sighting on stays blurred"
    assert len(sinks[7]) == 1, "an already-tracked target is not re-backfilled every frame"

    frame = np.random.randint(0, 255, (200, 200, 3), np.uint8)
    assert not np.array_equal(blur_boxes(frame.copy(), [(50, 50, 60, 60)])[50:110, 50:110], frame[50:110, 50:110])
    assert quality_bad(np.zeros((200, 200, 3), np.uint8)), "black frame must trigger Blackout"
    assert not quality_bad(np.random.randint(0, 255, (400, 400, 3), np.uint8)), "noise is sharp and bright"
    wide, why = burn(frame.copy(), [(0, 0, 200, 200)], False)       # the whole frame -> Blackout
    assert why == "cover" and np.array_equal(wide, blackout(frame)), "mostly covered -> blacked out"
    many, why = burn(frame.copy(), [(i, i, 2, 2) for i in range(MAX_BOXES + 1)], False)
    assert why == "count" and np.array_equal(many, blackout(frame)), "too many targets -> Blackout"
    assert burn(frame.copy(), [], True)[1] == "trust", "an untrusted frame -> Blackout"
    few, why = burn(frame.copy(), [(10, 10, 20, 20)], False)
    assert why == "", "a couple of small targets must NOT black out the whole frame"
    # The case that greyed out a live stream: a dozen scattered targets, none of them large.
    scattered = [(20 + 30 * (i % 5), 20 + 30 * (i // 5), 18, 18) for i in range(12)]
    assert burn(frame.copy(), scattered, False)[1] == "", "scattered targets must not cascade"
    print("selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="?", choices=["bench", "clip"])
    p.add_argument("args", nargs="*")
    p.add_argument("--every", type=int, default=EVERY, help="run detection every Nth frame")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or not a.cmd:
        return selftest()
    if a.cmd == "bench":
        return bench()
    if len(a.args) != 2:
        sys.exit("usage: redact.py clip IN.mp4 OUT.mp4")
    run_clip(a.args[0], a.args[1], a.every)


if __name__ == "__main__":
    main()
