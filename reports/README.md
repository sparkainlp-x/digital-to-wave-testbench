# Reports

Evidence status: **REPORTED** (written by the author before packaging) — the numbers they quote were
re-run and reproduced for v0.1.0 (see the main README, *Reproduction*).

| file | content |
|---|---|
| [`v6-timing-recovery-protocol-lock.md`](v6-timing-recovery-protocol-lock.md) | v6 blind timing-recovery conditions, committed before the full v6 run (written for v0.2.0; protocol only) |
| [`v5-timing-protocol-lock.md`](v5-timing-protocol-lock.md) | v5 conditions fixed before the full run (verbatim; the runner hash matches `MODULES.sha256`) |
| [`benchmark-v5-timing-report.pdf`](benchmark-v5-timing-report.pdf) | v5 integer-sample timing-offset report |
| [`benchmark-v4-cfo-report.pdf`](benchmark-v4-cfo-report.pdf) | v4 continuous carrier-frequency-offset report |

Notes for readers:

- The protocol lock and PDFs refer to project files that are not part of this public repository
  (the v4 protocol lock, the pre-run protected-file checksum lists and earlier v2/v3 reports). The test
  counts they quote (39 and 47 tests) refer to the author's original project suite, not to this
  repository's `tests/`.
- The links at the end of each PDF pointed into the original project layout; the equivalent files
  here are the modules at the repository root and `artifacts/`.
