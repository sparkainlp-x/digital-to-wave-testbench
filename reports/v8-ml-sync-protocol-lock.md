# v8 blind synchronization (GLRT metric + coarse-to-fine refinement) protocol lock — fixed before the full run

**Evidence status:** protocol only (no full-run results exist). All v8 results will be **SYNTHETIC**. **No physical
validity is claimed.** This is an exploratory software protocol, not a formal preregistration.

**Protocol source:** [`robustness_benchmark_v8_ml_sync.py`](../robustness_benchmark_v8_ml_sync.py), SHA-256
`77422f5f5d430749d911d05ecd678c08b73e51c7b7fd38d10aec56958570b9cf` (also recorded in [`MODULES.sha256`](../MODULES.sha256) and
`tests/test_hash_lock.py`). This document, the module and its reduced-size tests are committed and pushed, and CI must
be green, **before** the full run. The full run is executed afterwards from that commit with a clean tracked tree; the
results JSON records the commit, whether the tracked tree was clean, and the SHA-256 of the v1–v8 modules. The v1–v7
modules and artifacts are not modified.

**Lock hygiene disclosure.** All development used a separate development seed, **`20261111`**, and only small runs
(diagnostics with up to 50 frames per vector on a handful of cells and 200 per vector on two clock-free cells; one
full-size weak-cell check of 1,200 frames; and four sanity sweeps of 30 frames per vector over the 105 σ = 0.45 cells). Tests use their own seed (`5151`). The locked master
seed **`20261011`** has never been used in this repository (checked against the full Git history) and is not run before
this lock commit. The development runs led to three design changes before this lock, disclosed here: (1) refinement in
ridge coordinates instead of (timing, ppm, CFO) coordinates; (2) decoding with the raw-sample least-squares fit instead of
the v7 resampling projection, because the clock estimate is not identifiable inside likelihood plateaus (see below) while
the raw fit is; (3) a fix for a bug that left frames at zero parameters. No full-protocol result informed any choice.

## 1. Diagnosis of the v7 weak cell

v7 (`artifacts/robustness-benchmark-v7-clock-drift-cfo.csv`, master seed 20261010) at CFO 0, clock −2500 ppm, timing +2,
σ = 0.45: blind timing acquisition 1,200/1,200, joint acquisition 528/1,200 (44%), blind passes 1,056 versus the oracle's
1,119 of 1,200. Neighbouring cells show the same signature at a smaller scale (CFO +0.005/−2500 ppm and CFO −0.005/+2500
ppm: joint acquisition 86–93%).

Small diagnostics with the development seed (`tools/v7_weak_cell_diagnostic.py`) show:

- **Cause: a clock/CFO ambiguity (ridge) combined with a biased search metric.** In the received-sample domain the carrier
  runs at `(4 + cfo)(1 + ppm·1e-6)` cycles per nominal symbol, so −2500 ppm of clock error is carrier-equivalent to
  −0.01 cycles/symbol of CFO. The only information separating the two is symbol-boundary drift (≈5 samples over the
  frame). In the weak cell the v7 blind search chose either the truth (−2500 ppm, 0) or the ridge point
  (0 ppm, −0.01 cycles/symbol) — in the development run 138 versus 162 of 300 frames, never anything else. The ridge point
  is within one clock step of the truth but two CFO steps away, so joint acquisition fails.
- **Why the metric is biased:** v7 scores `Σ estimate²` after *linear-interpolation resampling*. Interpolating a
  carrier at 4/64 cycles/sample at fractional positions attenuates it (mean energy factor `2/3 + cos(π/8)/3 ≈ 0.975`).
  The truth (−2500 ppm) needs fractional positions; the ridge point (0 ppm, integer timing) needs none. That ≈2.5% energy
  penalty is about the size of the ridge point's boundary-misalignment loss, so on **clean** buffers the ridge point
  scores higher than the truth for 4 of the 6 vectors. Noise then decides.
- **Why it is asymmetric:** at +2500 ppm the ridge points (0 ppm, +0.01) and (+5000 ppm, −0.01) lose more energy at the
  frame edge than the truth, so the truth wins; at −2500 ppm the corresponding loss is smaller. The impairment and search
  grids are symmetric; the asymmetry comes from the frame-edge geometry combined with the interpolation penalty.
- **Not the cause:** grid resolution (the truth is on the v7 grid), an asymmetric grid (it is symmetric), or a tie-break
  (chosen candidates differ from the truth by noise-sized score margins; there are no exact ties).
- The same interpolation attenuation also lowers the **v7 oracle** (it uses the same resampling decoder), which is why the
  oracle passes fewer frames in clock-error cells than in clock-free cells.

## 2. v8 blind receiver (never told the impairments)

1. **Metric:** Gaussian generalized likelihood ratio on the **raw** received samples. For a candidate (timing, ppm, CFO),
   each received sample maps to source time `t(m)`, belongs to symbol `floor(t/64)` and is modelled as a free signed
   amplitude times `sin(2π(4+cfo)t/64 + π/4)`. Score `Σ_k (y·s_k)²/(s_k·s_k)` = `‖y‖²` minus the least-squares
   residual, i.e. maximum likelihood under white Gaussian noise. On clean buffers the truth has zero residual, so it scores
   strictly highest on the weak cell for all six vectors.
2. **Coarse search:** the unchanged v7 grid (timing {−4,−2,0,2,4}, clock {−5000,…,5000} ppm in 2500-ppm steps, CFO
   {−0.01,…,0.01} in 0.005 steps); ties keep the smallest corrections.
3. **Coarse-to-fine refinement:** the three best coarse candidates of each frame are refined in **ridge coordinates**
   (timing, ppm, carrier offset `g = (4+cfo)(1+ppm·1e-6) − 4`), so a clock move at fixed `g` follows the ridge. Three
   levels of a 3×3×3 local search, first steps (1 sample, 1250 ppm, 0.0025 cycles/symbol), halved per level (finest
   0.125 sample, 156.25 ppm, 0.0003 cycles/symbol); candidates must stay inside the v7 envelope (|timing| ≤ 4,
   |ppm| ≤ 5000, |CFO| ≤ 0.01); a move must beat the current score by a relative margin of 1e-9 (so plateau ties are
   resolved identically on every platform). The highest-scoring refined candidate is kept.
4. **Decoder:** least-squares amplitudes of the same raw-sample fit at the chosen parameters (no interpolation). Inside a
   likelihood plateau (clock values that do not change any sample's symbol membership at fixed `g`) the received model
   is identical, so this decoder is invariant there, unlike the v7 resampling decoder.

## 3. Frozen conditions

- **Master seed `20261011`** (fresh); vector seed `20261004` (the six fixed v2 vectors). RNG stream per
  (master seed, full-grid cell index, vector index), as in v7. 200 noisy frames per vector per cell (1,200 per cell);
  one clean frame per vector at σ = 0. Chunk size 50 frames.
- **Impairment grid (full factorial, symmetric):** CFO {−0.01, −0.005, 0, +0.005, +0.01} cycles/symbol × clock error
  {−5000, −2500, −1250, 0, +1250, +2500, +5000} ppm × timing {−2, 0, +2} samples × σ {0, 0.25, 0.45}: 105 impairment
  cells × 3 noise levels. The 27 v7 cells (both clock-error signs) are included; the other 78 form the **extension**
  (off-grid ±1250 ppm, edge ±5000 ppm and ±0.01 CFO), all inside the blind search envelope.
- Unchanged v7 signal model (`received_buffer`), pass rule max |error| ≤ 0.25 over 32 values.

## 4. Receivers (identical noisy buffers)

| receiver | told impairments? | description |
|---|---|---|
| `nominal` | no | unchanged `decode_wave`, fixed windows |
| `oracle` | yes | unchanged v7 oracle: exact parameters, v7 linear-resampling projection |
| `oracle_raw_ls` | yes | exact parameters, v8 raw-sample least-squares decoder (informed reference for the v8 decoder) |
| `v7_blind_joint_search` | no | unchanged v7 `blind_joint_search`, re-run on the new buffers |
| `v8_ml_refined_search` | no | Section 2 |

## 5. Metrics

Per vector and pooled per cell: frames passing with Wilson 95% intervals, error summaries; for both blind receivers,
per-parameter and joint acquisition under the **loose** v7 rule (within 2 samples, 2500 ppm, 0.005 cycles/symbol) and a
**strict** rule (within 1 sample, 1250 ppm, 0.0025 cycles/symbol), with Wilson intervals for joint acquisition, and mean
absolute parameter errors. Paired exact two-sided McNemar tests: v8 vs v7 blind, v8 vs oracle, v8 vs `oracle_raw_ls`,
v7 blind vs oracle, `oracle_raw_ls` vs oracle, v8 vs nominal; also pooled over cells (summed discordant counts) for all,
v7-grid and extension cells per noise level.

## 6. Pre-stated success criteria (evaluated mechanically in the JSON `summary.success_criteria`)

- **S1 (weak cell, primary):** at CFO 0, −2500 ppm, +2 samples, σ = 0.45, v8 loose joint acquisition ≥ 95% **and** v8
  passes more frames than the re-run v7 blind search with exact McNemar p < 0.01.
- **S2 (no regression on the v7 grid):** no noisy v7-grid cell (27 × 2) where v8 is significantly worse than v7 blind
  (v8 fails more discordant frames, unadjusted p < 0.05).
- **S3 (oracle):** no noisy v7-grid cell where v8 is significantly worse than the v7 oracle (unadjusted p < 0.05).
- **S4 (extension):** pooled over extension cells, per noise level, v8 passes more frames than v7 blind with p < 0.05.
  Any extension cell where v8 is significantly worse is disclosed.
- **Verdict:** the improvement is called real only if S1 and S2 both hold. S3/S4 are reported either way. Cells where v8 is
  no better or worse (including strict acquisition) will be reported. Unadjusted per-cell tests are many (≈200); the
  report will also state which results survive Bonferroni correction.

Expected but not a criterion: because `oracle_raw_ls` avoids interpolation attenuation it may beat the v7 oracle, in
which case v8 may too; any such gain is a **decoder** effect and will be reported as such, not as beating an upper bound.

## 7. Outputs and reproducibility

`python robustness_benchmark_v8_ml_sync.py` writes `artifacts/robustness-benchmark-v8-ml-sync.json`, `.csv`,
`-paired.csv` and a two-panel `.png` (log-scale failures per condition at σ = 0.45; strict joint acquisition). Lines are
added to `artifacts/SHA256SUMS`. Expected runtime ≈6 min on 8 cores (cells run in parallel; results do not depend on the
worker count). Tests: reduced runs with the test seed, diagnosis checks, validator tampering, determinism; after the run,
checks on the committed results and a full-size re-run of the weak cell compared with the committed rows.

## 8. Interpretation limits

Synthetic, dimensionless model of a specified signal and receiver. Known frame/symbol lengths, waveform family and guards;
static impairments; integer timing offsets in the grid; no pulse shaping, filtering, jitter or time-varying impairments.
Blind searches are bounded by the declared envelope. No physical performance is established.
