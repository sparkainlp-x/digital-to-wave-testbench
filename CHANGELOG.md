# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-10-03

### Added
- `robustness_benchmark_v8_ml_sync.py` (v8, SYNTHETIC): improved blind joint timing/clock/CFO synchronizer that is never told the impairments. It scores candidates with a raw-sample Gaussian GLRT (maximum-likelihood) metric, searches the v7 grid, refines the top 3 coarse candidates coarse-to-fine in ridge coordinates (timing, ppm, carrier offset), and decodes with the raw-sample least-squares fit. It is compared on identical buffers with the nominal decoder, the unchanged v7 oracle, a raw-LS informed oracle and the unchanged v7 blind search (re-run). Grid: 105 impairment cells (the 27 v7 cells, both clock signs, plus a symmetric extension to ±1250/±5000 ppm and ±0.01 CFO) × σ {0, 0.25, 0.45}, 1,200 noisy frames per cell, fresh master seed 20261011. Reports Wilson intervals, per-parameter and joint acquisition (loose v7 rule and strict half-step rule), paired exact McNemar tests and mechanically evaluated pre-stated success criteria.
- `reports/v8-ml-sync-protocol-lock.md`: diagnosis of the v7 weak cell and the v8 protocol, committed and pushed with green CI before the full run (lock commit `ba0143e`). The v8 module hash is in `MODULES.sha256` and `tests/test_hash_lock.py`.
- `tools/v7_weak_cell_diagnostic.py` (development seed 20261111) and `tests/test_v8_ml_sync.py`: diagnosis checks, clean recovery, determinism, validator tampering, reduced runs, checks on the committed results, and full-size re-runs of two protocol cells compared line by line with the committed CSVs.
- Artifacts `artifacts/robustness-benchmark-v8-ml-sync{.json,.csv,-paired.csv,.png}` and their `SHA256SUMS` lines.

### Changed
- `tools/reproduce_all.sh` also regenerates v8 (`SKIP_V8=1` keeps the committed v8 artifacts). CI uses `SKIP_V8=1` because a full v8 re-run (about 27 CPU-minutes) would exceed its time budget; it checks v8 through the two full-size cell re-runs in the tests instead.
- README, CITATION.cff and .zenodo.json updated for v8. Version 0.4.0.

### Results (SYNTHETIC; no physical validity claimed)
- Diagnosis: −2500 ppm of clock error is carrier-equivalent to −0.01 cycles/symbol of CFO. v7's metric (sum of squared decoded values after linear-interpolation resampling) penalises the fractional resampling that the truth needs, so the ridge point (0 ppm, −0.01) out-scored the truth even on clean buffers for 4 of 6 vectors. The cause was not grid resolution, grid asymmetry or a tie-break.
- Weak cell (CFO 0, −2500 ppm, +2, σ = 0.45): v8 passes 1,150/1,200 vs 1,034 for the re-run v7 blind search (McNemar 132 vs 16, p = 7.0e-24). The v7 oracle passes 1,117 and the raw-LS oracle 1,150. Joint acquisition is 1,199/1,200 vs 528/1,200.
- Overall at σ = 0.45 (126,000 frames): v8 119,273, v7 blind 109,753, v7 oracle 116,102, raw-LS oracle 119,310. All four pre-stated criteria were met, and v8 was significantly worse than v7 blind in no cell.
- Disclosed shortfalls: v8's gain over the v7 oracle is a decoder effect. v8 is slightly below the raw-LS oracle when pooled (35 vs 72 discordant, p = 4.5e-4; no single cell significant). It shows no gain in 83 of 210 noisy cells (1–2 frames fewer than v7 in 5 of them, not significant), and its strict joint acquisition is lower than v7's in 36 cells, mainly where v7's grid contains the truth exactly.

### Notes
- Before the lock, three design changes were made on development-seed runs only and are disclosed in the protocol lock: ridge-coordinate refinement, the raw-LS decoder, and a fix for frames left at zero parameters. The v8 module was not changed after the lock. v1–v7 modules and artifacts are unchanged.

## [0.3.0] - 2026-10-03

### Added
- `robustness_benchmark_v7_clock_drift_cfo.py`: synthetic joint clock-drift/CFO sweep with nominal, oracle and blind joint-search receivers, paired comparisons, acquisition metrics, and reproducible JSON/CSV/PNG artifacts. The v7 protocol and module hash are fixed before the full run; v1-v6 remain unchanged.
- `reports/v7-clock-drift-cfo-protocol-lock.md` (v7 conditions and lock history), `tests/test_v7_clock_drift_cfo.py`, and the v7 module hash in `MODULES.sha256`.

### Notes
- All v7 results are SYNTHETIC; no physical validity is claimed.
- Disclosure: a first full-protocol v7 run (`8da0e5e`, generated from `5e1d272`) was superseded after the module was changed to report pooled blind-search acquisition and re-locked in `1625166`; the full protocol was re-run from `1625166`. All 1,701 result rows have identical pass counts and error statistics and the paired comparisons are byte-identical; only the 81 pooled blind-search rows gained acquisition fields. σ = 0.25 was added after the first protocol lock but before any results existed.

## [0.2.0] - 2026-10-03

### Added
- `robustness_benchmark_v6_timing_recovery.py` (v6, SYNTHETIC): blind timing recovery for the unchanged codec.
  It imports the hash-locked v1-v5 modules without editing them. Every estimate uses `decode_wave`.
  - Blind matched-filter-energy window search and raw frame-energy search over integer starts −16..+16, never
    told the offset; nominal decoder and exact-offset oracle on the same noisy buffers (v5 grid, six fixed
    vectors, 12,000 frames per offset, σ = 0.45, master seed 20261009). Reports timing-acquisition rates,
    95% Wilson intervals, analytic predictions, and discordant counts with exact McNemar p-values.
  - Fractional (sub-sample) offsets from the continuous p.12 waveform sampled at n + Δ
    (Δ = 0…8), with a best-integer oracle and dense analytic curves.
  - Randomised per-frame vectors (U[−2, 2] and the integer alphabet {−2..2}) as a sensitivity panel, with
    both the 0.25 rule and a round-to-integer rule.
  - Clean deterministic sweep of Δ = −32..32 showing the 16-sample carrier periodicity.
  - Artifacts: `artifacts/robustness-benchmark-v6-timing-recovery{.json,.csv,.png,-paired.csv,-clean-sweep.csv}`.
    The JSON records the git commit, whether the tracked tree was clean, and the module SHA-256s.
- `reports/v6-timing-recovery-protocol-lock.md`: the v6 conditions. They were committed and pushed
  (`f41da8e`) before the full run, and the v6 module hash was added to `MODULES.sha256`.
- `tests/test_v6_timing_recovery.py`: closed-form ±1-sample bias, 180° sign inversion at ±8, near-periodicity
  at ±16, fractional = integer sampling at integer offsets, encoder bit-equivalence, exact clean acquisition,
  nominal = oracle at 0, McNemar values, determinism, reduced-run golden counts and validator tampering
  checks, plus checks of the committed full-run artifact.

### Results (SYNTHETIC)
- The MF-energy search acquired the exact start in 108,000/108,000 noisy frames and matched the oracle's
  pass count at every integer offset with 0 discordant frames. The raw frame-energy search acquired only
  41–43% of frames. For fractional offsets the integer search matched the best-integer oracle; its loss
  relative to on-time sampling was 2.4–3.3 percentage points at Δ = 0.5, 0.75 and 1.5.

### Changed
- CI: `actions/checkout@v7` and `actions/setup-python@v7` (Node 24 runtime; replaces the deprecated
  Node 20 majors). `tools/reproduce_all.sh` also regenerates the v6 artifacts, and CI compares them byte for byte.
- README: v6 section and figure. Timing recovery moved from TARGET/UNRUN to done, with its easy-case caveats.
  Carrier/I/Q recovery, clock drift and interpolating timing recovery remain TARGET.

### Unchanged
- The five original modules and their `MODULES.sha256` entries are byte-identical to v0.1.0.

## [0.1.0] - 2026-10-03

### Added
- First public release of the synthetic Digital-to-Wave testbench.
- Original, hash-locked modules (unchanged; SHA-256 in `MODULES.sha256`):
  `digital_to_wave.py` (v1 encoder/decoder), `robustness_benchmark_v2.py` (Gaussian/Laplace noise and
  constant phase rotation), `robustness_benchmark_v3.py` (closed-form Gaussian calibration),
  `robustness_benchmark_v4.py` (continuous carrier-frequency offset) and
  `robustness_benchmark_v5_timing.py` (constant integer-sample timing offset).
- Generated artifacts for v2–v5 (JSON, CSV, PNG) with `artifacts/SHA256SUMS`; CI checks that every
  CSV and PNG is regenerated byte for byte.
- `tools/timing_phase_diagnostic.py`: clean-frame diagnostic showing that the v5 timing effect is
  dominated by carrier-phase rotation (22.5° per sample).
- Reports: v5 protocol lock, v4 and v5 report PDFs (`reports/`).
- pytest suite (hash lock, codec round trip, Wilson intervals, analytic 94.78% oracle probability,
  v5 validator, determinism, golden counts from the full v3/v4/v5 protocols, v3/v4 tie check).
- GitHub Actions CI on Python 3.11, 3.12 and 3.13; CITATION.cff and .zenodo.json metadata.

### Known limitations
- No timing-recovery or carrier-recovery baseline is included yet (TARGET for a later version).
