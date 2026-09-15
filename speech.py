"""Audio redaction: find spoken numbers and names inside the Privacy Buffer and silence them.

Runs on the buffered Raw Feed, so a word can be muted before it ever airs (ADR 0001). Nothing is
transcribed to disk -- the text exists only long enough to decide which milliseconds to silence
(ADR 0002). Fail-closed: anything we cannot transcribe confidently is silenced rather than aired.

    python speech.py --selftest
"""
import argparse
import collections
import importlib.util
import os
import pathlib
import queue
import re
import threading

import numpy as np

SR = 16000          # what Whisper wants
CHUNK = 2.0         # seconds of audio per transcription pass
OVERLAP = 0.6       # carried into the next chunk, so a word on the boundary is heard whole once
PAD_MS = 150        # widen every Bleep: word timestamps are approximate and clip the edges
MIN_PROB = 0.45     # a word the model is unsure of is silenced, not trusted

# Digits spelled out. A house number is as identifying spoken as printed.
NUMBERS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy
eighty ninety hundred thousand million first second third fourth fifth sixth seventh eighth
ninth tenth oh double triple""".split())

# Capitalised words that are ordinary sentence starts rather than names. Everything else that is
# capitalised gets silenced -- over-muting "Monday" costs a beep, under-muting a surname does not.
COMMON = set("""a an and are as at be because but by can could do does for from get go had has have
he her here his how i if in is it its just let me my no not now of on once or our out she so
that the then there these they this to too very was we well were what when where which who why
will with would you your yes ok okay hi hey hello thanks thank please sorry sure right good
leave put take give come got going see look keep need want hold drop sign done back over about
after before still again only also all any some more them their him us been being did doing say
said know think make made alright yeah yep nope cheers welcome great perfect morning afternoon
evening today tomorrow delivery package parcel box door front sorry excuse
don won isn aren wasn weren didn doesn couldn wouldn shouldn hasn haven hadn ain gonna gotta""".split())

# Words that announce an address component. Whatever follows one is silenced whether or not it was
# recognised as a number: a live test heard "flat four" as "flat forward", so nothing matched and
# the flat number aired. Mishearing is exactly when we can least afford to be trusting.
CUES = set("""flat apartment apt unit suite block floor house number no postcode zip zipcode
street road avenue lane drive court close way place terrace crescent building room entrance""".split())
CUE_SPAN = 2        # words silenced after a cue

# Spoken panic words. Deliberately distinctive: a false trigger only hides more of your own stream,
# but one that fires on ordinary chatter is a stream that keeps blanking for no reason.
KEYWORDS = {"privacy", "blackout"}
# Releasing needs a two-word phrase, not a single word. Engaging by accident costs you a grey
# stream; releasing by accident un-hides whatever you panicked about, so this side wants friction.
CLEAR_PHRASE = ("all", "clear")
# Turning redaction off wholesale is the one command where a mishearing exposes somebody else, so
# it takes a two-word phrase and the way back is a single word -- the friction runs the opposite
# way to a Panic, because here it is stopping that is dangerous and resuming that is safe.
RAW_PHRASE = ("redaction", "off")
REDACT_WORD = "redact"

DIGIT = re.compile(r"\d")

# Prints every transcribed word, muted ones in [brackets]. OFF by default and it must stay that
# way: switching it on writes transcribed speech to the log, which is exactly what ADR 0002 forbids.
# Use it only against a rehearsed test line with a fake name and address.
DEBUG = False


def _enable_cuda12():
    """CTranslate2 links against CUDA 12 (cublas64_12.dll), but onnxruntime-gpu pulls in CUDA 13.
    The CUDA 12 runtime is already on disk inside torch, so just make it findable. find_spec locates
    torch without importing it -- we want the path, not several seconds and a gigabyte of module."""
    spec = importlib.util.find_spec("torch")
    if not spec or not spec.origin:
        return
    lib = pathlib.Path(spec.origin).parent / "lib"
    if not (lib / "cublas64_12.dll").exists():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib))
    os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"


def risky(word, prob=1.0):
    """True when a spoken word must be silenced. Deliberately generous: a false positive is one
    muted syllable, a false negative is someone's address going out to an audience."""
    bare = word.strip(" .,!?;:'\"-").strip()
    if not bare:
        return False
    if prob < MIN_PROB:
        return True                       # the model is guessing; do not gamble on what it guessed
    if DIGIT.search(bare):
        return True
    low = bare.lower()
    if low in NUMBERS:
        return True
    # ponytail: capitalisation as a proper-noun test. No POS tagger, no extra dependency. It misses
    # an all-lowercase transcription of a name; the COMMON list keeps ordinary sentences audible.
    if not bare[0].isupper():
        return False
    # Contractions: "I'm" and "Don't" are capitalised sentence starts, not names. Test the stem too.
    return low not in COMMON and low.split("'")[0] not in COMMON


def spans_from(segments, base_ms):
    """Turn Whisper word timings into (start_ms, end_ms) spans to silence."""
    out, after_cue = [], 0
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            bare = w.word.strip(" .,!?;:'\"-").lower()
            if after_cue or risky(w.word, getattr(w, "probability", 1.0)):
                out.append((base_ms + w.start * 1000 - PAD_MS, base_ms + w.end * 1000 + PAD_MS))
            after_cue = CUE_SPAN if bare in CUES else max(0, after_cue - 1)
    return merge_spans(out)


def _bare_words(segments):
    return [w.word.strip(" .,!?;:'\"-").lower()
            for seg in segments for w in (getattr(seg, "words", None) or [])]


def heard_keyword(segments):
    """True when a panic word was spoken. Matched on the bare word so punctuation and capitalisation
    from the transcriber cannot stop a panic from landing."""
    return any(w in KEYWORDS for w in _bare_words(segments))


def _heard_phrase(words, phrase):
    return any(tuple(words[i:i + len(phrase)]) == phrase for i in range(len(words) - len(phrase) + 1))


def heard_clear(segments):
    """True when the release phrase was spoken, as consecutive words."""
    return _heard_phrase(_bare_words(segments), CLEAR_PHRASE)


def heard_raw(segments):
    """True when redaction was asked to stop entirely, as consecutive words."""
    return _heard_phrase(_bare_words(segments), RAW_PHRASE)


def heard_redact(segments):
    """True when redaction was asked back on. One word, because resuming is the safe direction."""
    return REDACT_WORD in _bare_words(segments)


def merge_spans(spans):
    """Overlapping or near-touching spans become one, so a muted phrase has no audible gaps."""
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1] + PAD_MS:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def overlaps(spans, a, b):
    return any(s < b and a < e for s, e in spans)


class Speech:
    """Accumulates buffered audio, transcribes it on a worker thread, and reports what to silence."""

    def __init__(self, model="small", language="en", on_keyword=None, on_clear=None,
                 on_raw=None, on_redact=None):
        # Pinned to one language on purpose: auto-detect runs per chunk, and a live test had it
        # flip mid-sentence and hallucinate Portuguese. Change this if you stream in another one.
        self.on_keyword, self.on_clear = on_keyword, on_clear
        self.on_raw, self.on_redact = on_raw, on_redact
        _enable_cuda12()
        from faster_whisper import WhisperModel      # imported late: it pulls in CUDA libraries
        self.model = WhisperModel(model, device="cuda", compute_type="int8_float16")
        self.language = language
        self.q = queue.Queue(maxsize=3)
        self.spans = collections.deque(maxlen=500)
        self.lock = threading.Lock()
        self.pcm, self.at_ms = np.zeros(0, np.float32), None
        self.stats = collections.Counter()
        threading.Thread(target=self._work, daemon=True).start()

    def feed(self, pcm, t_ms):
        """Add mono 16 kHz float samples captured at t_ms on the relay clock."""
        if self.at_ms is None:
            self.at_ms = t_ms
        self.pcm = np.concatenate((self.pcm, pcm))
        if len(self.pcm) < int(CHUNK * SR):
            return
        chunk, at = self.pcm, self.at_ms
        try:
            self.q.put_nowait((chunk, at))
        except queue.Full:
            # Transcription is behind. Silence the whole chunk rather than let it through unchecked.
            self._add([(at, at + len(chunk) / SR * 1000)])
            self.stats["overrun"] += 1
        # Carry the tail forward: a word straddling the cut was transcribed as two fragments, and
        # "Ashford" arriving as "A" then "for" matches nothing worth silencing.
        keep = int(OVERLAP * SR)
        self.pcm = chunk[-keep:]
        self.at_ms = at + (len(chunk) - keep) / SR * 1000

    def _work(self):
        while True:
            pcm, t_ms = self.q.get()
            try:
                segs, _ = self.model.transcribe(pcm, language=self.language, word_timestamps=True,
                                                vad_filter=True, beam_size=1)
                segs = list(segs)
                found = spans_from(segs, t_ms)
                self._add(found)
                self.stats["bleeps"] += len(found)
                # Clear first, engage second: if both were said in one breath, hiding wins.
                if self.on_clear and heard_clear(segs):
                    self.stats["cleared"] += 1
                    self.on_clear()
                if self.on_keyword and heard_keyword(segs):
                    self.stats["keyword"] += 1
                    self.on_keyword()
                # Same order for the same reason: said in one breath, redacting wins.
                if self.on_redact and heard_redact(segs):
                    self.stats["redact"] += 1
                    self.on_redact()
                if self.on_raw and heard_raw(segs):
                    self.stats["raw"] += 1
                    self.on_raw()
                if DEBUG:
                    self._show(segs, found, t_ms)
            except Exception as e:                    # fail closed: unreadable audio is silenced
                self._add([(t_ms, t_ms + len(pcm) / SR * 1000)])
                self.stats["failed"] += 1
                print(f"speech: {e!r}")

    def _show(self, segs, spans, t_ms):
        """Temporary: echo the transcript with muted words in [brackets], to check the right ones go."""
        out = [f"[{w.word.strip()}]" if overlaps(spans, t_ms + w.start * 1000, t_ms + w.end * 1000)
               else w.word.strip()
               for seg in segs for w in (getattr(seg, "words", None) or [])]
        if out:
            print("speech:", " ".join(out), flush=True)

    def _add(self, spans):
        with self.lock:
            self.spans.extend(spans)

    def muted(self, a_ms, b_ms):
        with self.lock:
            return overlaps(self.spans, a_ms, b_ms)


def selftest():
    assert risky("33")
    assert risky("Bedford") and risky("Brooklyn"), "place and street names are targets"
    assert risky("thirty") and risky("Seventh")
    assert risky("anything", prob=0.1), "an unsure word is silenced whatever it says"
    assert not risky("the") and not risky("your") and not risky("package")
    assert not risky("I") and not risky("Hello"), "ordinary sentence starts stay audible"
    # Live test caught these: an apostrophe made every contraction look like a surname.
    assert not risky("I'm") and not risky("It's") and not risky("Don't"), "contractions are not names"
    assert risky("O'Brien"), "but a real name with an apostrophe still goes"
    assert not risky(""), "punctuation-only tokens are not words"

    assert merge_spans([(0, 100), (120, 200)]) == [(0, 200)], "near-touching spans join"
    assert merge_spans([(0, 100), (900, 1000)]) == [(0, 100), (900, 1000)], "distant spans stay apart"
    assert overlaps([(100, 200)], 150, 250) and not overlaps([(100, 200)], 250, 300)
    assert overlaps([(100, 200)], 50, 120), "a frame ending inside a span is muted"

    word = lambda t, s, e, p=0.9: type("W", (), {"word": t, "start": s, "end": e, "probability": p})()
    seg = type("S", (), {"words": [word("number", 0.0, 0.4), word("33", 0.5, 0.8),
                                   word("Bedford", 0.9, 1.4)]})()
    got = spans_from([seg], 10_000)
    assert len(got) == 1, "adjacent risky words become one continuous Bleep"
    assert got[0][0] < 10_500 and got[0][1] > 11_400, "the Bleep covers both words, with padding"
    assert not overlaps(got, 10_000, 10_300), "the safe word before them is left audible"

    # The live leak: "flat four" heard as "flat forward" matched nothing, so the number aired.
    cue = type("S", (), {"words": [word("flat", 0.0, 0.3), word("forward", 0.4, 0.9),
                                   word("now", 1.0, 1.2)]})()
    got = spans_from([cue], 0.0)
    assert overlaps(got, 400, 900), "whatever follows an address cue is silenced, recognised or not"
    assert not overlaps(got, 0, 250), "the cue word itself gives nothing away and stays audible"
    dull = type("S", (), {"words": [word("leave", 0.0, 0.3), word("it", 0.4, 0.6)]})()
    assert not spans_from([dull], 0.0), "ordinary delivery speech is left alone"

    kw = type("S", (), {"words": [word("uh", 0.0, 0.2), word("Privacy!", 0.3, 0.8)]})()
    assert heard_keyword([kw]), "the panic word lands through capitals and punctuation"
    assert not heard_keyword([dull]), "and ordinary speech does not trip it"

    ok = type("S", (), {"words": [word("All", 0.0, 0.3), word("clear.", 0.4, 0.8)]})()
    assert heard_clear([ok]), "the release phrase lands as consecutive words"
    assert not heard_clear([kw]) and not heard_clear([dull]), "ordinary speech does not release"
    split = type("S", (), {"words": [word("all", 0.0, 0.3), word("of", 0.4, 0.6),
                                     word("clear", 0.7, 0.9)]})()
    assert not heard_clear([split]), "the two words must be adjacent, not merely both present"

    off = type("S", (), {"words": [word("Redaction", 0.0, 0.4), word("off.", 0.5, 0.8)]})()
    assert heard_raw([off]), "the two words stop redaction"
    assert not heard_raw([kw]) and not heard_raw([dull]), "ordinary speech does not stop redaction"
    assert not heard_redact([off]), '"redaction" must not also read as the single word "redact"'
    back = type("S", (), {"words": [word("Redact", 0.0, 0.4)]})()
    assert heard_redact([back]), "one word puts redaction back on"
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.parse_args()
    selftest()
