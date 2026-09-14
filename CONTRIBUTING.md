# Contributing

This is a small, single-purpose tool built for one setup (Ray-Ban Meta glasses, one PC, one GPU).
PRs are welcome, but keep that in mind before adding options or abstractions for setups nobody has.

Before opening a PR:

- Run the self-checks: `python relay.py --selftest`, `python redact.py --selftest`,
  `python speech.py --selftest`, `python upload.py --selftest` (the first cascades into the other
  two). CI runs the same commands.
- If you change a threshold or add a rule with a real-footage failure case behind it, add an
  assertion for it in the relevant `selftest()` — that's the whole test suite, and it's how past
  regressions stay fixed.
- Match the existing comment style: explain *why*, not *what*. A comment that just restates the
  code below it should be deleted, not added.
- Terminology (Raw Feed, Clean Feed, Privacy Buffer, Backfill, Panic, etc.) is defined in
  `CONTEXT.md` — use it instead of inventing new names for the same concepts.
