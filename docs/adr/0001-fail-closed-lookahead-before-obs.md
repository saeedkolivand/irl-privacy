# Fail-closed redaction with a look-ahead buffer, before OBS

Detectors miss things, and a single unredacted frame of a parcel label is readable from a paused VOD, so redaction runs fail-closed: the stream airs 3 s behind reality, every object is Backfilled to its first frame, and anything the system can't vouch for becomes a Blackout. This has to happen in a separate relay before OBS, because OBS filters process one frame at a time and can't see future frames. The phone never holds a platform stream key; if the relay or home link dies, OBS shows BRB rather than any path for raw video.

## Considered Options

- Live detect-and-blur (no delay): lowest latency, but one-frame leaks on every new object.
- Zone-based whole-frame blur only (hotkey/geofence): simple, but depends on the streamer never forgetting.
- OBS filter plugin (obs-detect): frame-by-frame, no look-ahead, stalled project.

## Consequences

Viewers are ~3 s + platform latency behind; Streamlabs Alert Delay must match the buffer so alerts line up with the delayed video.
