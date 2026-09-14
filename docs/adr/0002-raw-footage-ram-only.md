# Raw footage never touches disk

The Raw Feed exists only in the in-memory Privacy Buffer; recordings, VODs and highlight edits are always made from the Clean Feed. An archived raw copy would allow re-cutting or re-running better models later, but raw PII on disk can leak through backups, cloud sync or a stolen machine, so we give up that flexibility. 3K highlight clips from the glasses go through the same redaction offline and the originals are deleted afterwards.
