# IRL Privacy

Redacting third-party personal information from a first-person IRL livestream (smart glasses POV, delivery-style work) before any viewer can see or hear it.

## Feeds

**Raw Feed**:
Glasses video and audio before redaction. Exists only inside the Privacy Buffer.
_Avoid_: source, input, camera feed

**Clean Feed**:
The redacted output. The only thing the broadcast, recordings and viewers ever receive.
_Avoid_: output, processed feed, blurred feed

**Privacy Buffer**:
The fixed delay between the Raw Feed arriving and the Clean Feed leaving.
_Avoid_: stream delay, lag, latency

## Redaction

**Redaction Target**:
A category that is always obscured regardless of its content: face, plate, text region, code, parcel, screen.
_Avoid_: PII, sensitive object, detection

**Backfill**:
Extending a confirmed redaction back to the first buffered frame where the object appeared.
_Avoid_: look-back, retro-blur

**Proximity**:
How much of the frame a Redaction Target fills, standing in for how close it is. A close target is
blurred wide enough to cover whatever it is printed on; a distant one is blurred on its own.
_Avoid_: distance, depth, size

**Blackout**:
Whole-frame blur together with muted audio.
_Avoid_: full blur, privacy mode, censor

**Bleep**:
Muting a single spoken number or proper noun.
_Avoid_: censor, beep

## Triggers

**Auto Trigger**:
A system condition that forces a Blackout because redaction can't be trusted (low resolution, blurry or dark frame, missed deadline, stalled ingest).
_Avoid_: fallback, failsafe

**Panic**:
A manually triggered, latched Blackout applied to everything already in the Privacy Buffer.
_Avoid_: kill switch, emergency mode

**Release**:
The deliberate act of ending a Panic. Never automatic, and harder to trigger than a Panic is.
_Avoid_: resume, cancel, undo

**Cue**:
A spoken word that announces an address component, so whatever follows it is Bleeped whether or
not it was recognised. "flat", "number", "postcode".
_Avoid_: trigger, marker

## Assurance

**Leak**:
Any Redaction Target, or any word that should have been Bleeped, still visible or audible in the Clean Feed.
_Avoid_: miss, slip, exposure

**Leak Auditor**:
The offline check that scans a recorded Clean Feed for Leaks.
_Avoid_: validator, QA pass
