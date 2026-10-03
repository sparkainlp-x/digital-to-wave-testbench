# v6 blind timing-recovery protocol lock — fixed before the full run

**Evidence status:** protocol (no results). All v6 results will be **SYNTHETIC**. **No physical validity is claimed.**
This is an exploratory software protocol, not a formal preregistration.

**Protocol source:** [`robustness_benchmark_v6_timing_recovery.py`](../robustness_benchmark_v6_timing_recovery.py),
SHA-256 `c3f2da47e9da81072cbdc02153c28f6c9b6f5958251a4c91f8d64f9e5cfa112b` (also recorded in [`MODULES.sha256`](../MODULES.sha256) and `tests/test_hash_lock.py`).
This document, the module and its reduced-size tests are committed and pushed **before** the full v6 run. The full run is
executed afterwards from that commit with a clean tracked tree; the results JSON records the commit
(`git_commit`), whether the tracked tree was clean, and the SHA-256 of the five hash-locked modules and of the v6 module.

**Lock hygiene disclosure.** During development the code was exercised only with reduced sizes (at most 100 frames per
vector and condition and 600 random frames per offset) under a development seed (`20261008`), and the tests use their
own seed (`4242`). The locked master seed `20261009` has not been run before this lock commit.

## Question

v5 showed that the unchanged nominal decoder fails once its windows start two or more samples off, while an oracle told
the exact offset recovers. v6 asks whether a standard **blind** timing search, never told the offset, recovers
oracle-level performance in this synthetic model, and how much an integer-only search loses at sub-sample offsets.

## Model (unchanged codec)

- Hash-locked v1–v5 modules are imported and not edited. Every symbol estimate comes from the unchanged
  `digital_to_wave.decode_wave` projection; v6 only chooses which 64-sample windows it is handed.
- Receiver buffer of 2,176 samples: `y[m] = S(m − 64 + Δ) + noise[m]`, where `S` is the continuous p.12 frame signal
  (32 symbols of 64 samples, 4 carrier cycles, `A_min = 1`, zero outside the frame). Positive Δ = receiver late (v5
  convention). For integer Δ the buffer is built from the unchanged encoder and equals the v5 nominal stream.
- IID Gaussian noise, SD 0.45, on every buffer sample including the 64-sample guards; clean (σ = 0) controls.
- Pass rule: max |error| ≤ 0.25 over the 32 values (inclusive, as v2–v5). The integer-alphabet panel also reports a
  round-to-integer rule (`numpy.round` of every estimate equals the transmitted integer).

## Receivers

| method | told the offset? | rule |
|---|---|---|
| nominal decoder | no | windows at 64 + 64k (no correction) — identical to v5 |
| oracle (exact timing) | yes | windows at 64 + 64k − Δ; analysis-only upper bound |
| oracle (best integer), fractional panel | yes | −floor(Δ) or −ceil(Δ), whichever has the higher pooled analytic pass probability at σ = 0.45 |
| **blind matched-filter-energy search** | **no** | for each candidate c ∈ {−16..16}, decode the 32 windows at 64 + 64k + c and choose c maximising Σ_k estimate_k²; ties → smallest \|c\|, then smallest c |
| blind raw frame-energy search | no | choose c maximising the received energy in samples [64 + c, 64 + c + 2048); same tie rule |

The search uses the known frame length, symbol length and carrier phase reference. This is deliberately an easy
setting: noise-only guards, integer (or constant fractional) offsets, no clock drift, no CFO, carrier locked to the
sample clock.

## Conditions fixed in advance

- **Master seed `20261009`** (distinct from v2 `20261003`, vector seed `20261004`, v3 `20261005`, v4 `20261006`,
  v5 `20261007`). Streams: `SeedSequence([seed, 1, offset_index, vector_index])` (integer panel),
  `[seed, 2, delta_index, vector_index]` (fractional), `[seed, 3, distribution_index, offset_index]` (random vectors).
- **Panel 1 — integer offsets:** Δ ∈ {−8, −4, −2, −1, 0, 1, 2, 4, 8} (the v5 grid); six fixed v2 vectors; 2,000 noisy
  frames per vector and offset (12,000 pooled) plus one clean frame each. Report pass counts, 95% Wilson intervals,
  timing-acquisition rate (chosen correction = −Δ exactly) with Wilson intervals, analytic predictions (nominal, oracle),
  and paired comparisons on the same noisy buffers: discordant counts and exact two-sided McNemar p for
  search vs oracle, search vs nominal, frame-energy vs oracle, frame-energy vs matched-filter search.
- **Panel 2 — fractional offsets:** Δ ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8}, continuous waveform sampled at n + Δ;
  same vectors and frame counts; nominal, best-integer oracle and matched-filter search; dense analytic curves
  (step 1/16 sample) for the nominal decoder and the best integer correction.
- **Panel 3 — randomised per-frame vectors (sensitivity):** U[−2, 2] and the integer alphabet {−2..2}; 12,000 frames per
  offset and distribution on the integer grid; nominal, oracle and matched-filter search; 0.25 rule for both, plus the
  round-to-integer rule for the integer alphabet; oracle closed forms for both rules.
- **Panel 4 — clean sweep:** Δ = −32..32 (deterministic, six fixed vectors): component MAE for nominal, oracle and both
  searches, showing the 16-sample carrier periodicity and the ±16 search range.
- **Total:** 444,000 noisy input frames (108,000 integer + 120,000 fractional + 216,000 random-vector). Chunk size 128 frames.
- **Uncertainty:** Wilson intervals quantify Monte Carlo variation only, conditional on the vector panel and the
  stipulated IID model; not model or physical uncertainty.

## Outputs

`artifacts/robustness-benchmark-v6-timing-recovery.{json,csv,png}`, `…-paired.csv`, `…-clean-sweep.csv`, with SHA-256 in
`artifacts/SHA256SUMS`. CSV and PNG are deterministic with the pinned dependencies; the JSON embeds software versions and
the git commit, so its hash is environment-specific.
