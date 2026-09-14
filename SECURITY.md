# Security

## Threat model

The panic endpoint (`:8765`, in `relay.py`) and the upload endpoint (`:8766`, in `upload.py`) are
plain HTTP with no authentication, no TLS, and no rate limiting. That is deliberate, not an
oversight: both bind to the machine's Tailscale IP, so reaching them at all already means being on
the same tailnet. Tailnet membership *is* the auth. Do not put either port on a public interface,
behind a reverse proxy, or on a LAN you don't trust — there is nothing else standing between an
attacker and blanking your stream or dropping files on your disk.

The RTMP ingest in `relay.py` has the same property: it accepts one publisher, unauthenticated,
on whatever address `--listen` binds to. Same rule applies.

## Reporting a problem

This is a hobby project. Open a GitHub issue.
