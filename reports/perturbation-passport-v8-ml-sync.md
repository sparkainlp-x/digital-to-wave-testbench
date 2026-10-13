# Perturbation Passport — v8 blind synchronization

> **SYNTHETIC** — synthetic dimensionless benchmark. No physical or biological validity is claimed.

## Provenance

- Source artifact: `robustness-benchmark-v8-ml-sync.json`
- Source SHA-256: `b169a7641f285879d1440e52c0e8ff9185f8fc31a03e85b174ae4999b0e9f3c8`
- Benchmark commit: `ba0143e55f99b3a508082f81f9174cf9dec54f02`
- Master / vector seed: `20261011` / `20261004`
- Full declared protocol: `true`
- Protocol lock: `reports/v8-ml-sync-protocol-lock.md`
- Passport generator SHA-256: `9f5e0218c56dc78e1a80ce751ab6da950da9b7a58f84e085ae90dd481e0a5faa`

## Controlled perturbations

| Axis | Unit | Tested values |
|---|---|---|
| carrier-frequency offset | cycles/symbol | -0.01, -0.005, 0, 0.005, 0.01 |
| sample-clock error | ppm | -5000, -2500, -1250, 0, 1250, 2500, 5000 |
| receiver timing offset | samples | -2, 0, 2 |
| Additive sample-noise standard deviation | dimensionless | 0, 0.25, 0.45 |

- Grid: 105 declared impairment cells; 200 frames per vector at positive noise; 6 fixed input patterns (seed `20261004`).
- Pass rule: `max absolute decoded-value error across 32 values <= 0.25`.

## Aggregate outcomes

Pass shares and Wilson 95% intervals are conditional on the benchmark's fixed vector panel and noise draws.

| Noise std | Receiver | Passing / frames | Pass share | Wilson 95% |
|---:|---|---:|---:|---:|
| 0.25 | `v8_ml_refined_search` | 126,000 / 126,000 | 100.00% | 100.00%–100.00% |
| 0.25 | `v7_blind_joint_search` | 122,631 / 126,000 | 97.33% | 97.24%–97.41% |
| 0.25 | `oracle_raw_ls` | 126,000 / 126,000 | 100.00% | 100.00%–100.00% |
| 0.25 | paired v8 vs v7 | 3,369 v8-only passes / 0 v7-only passes | — | McNemar p = 0 (stored underflow) |
| 0.45 | `v8_ml_refined_search` | 119,273 / 126,000 | 94.66% | 94.54%–94.78% |
| 0.45 | `v7_blind_joint_search` | 109,753 / 126,000 | 87.11% | 86.92%–87.29% |
| 0.45 | `oracle_raw_ls` | 119,310 / 126,000 | 94.69% | 94.57%–94.81% |
| 0.45 | paired v8 vs v7 | 11,001 v8-only passes / 1,481 v7-only passes | — | McNemar p = 0 (stored underflow) |

## Observed failure boundary

At the highest tested noise (σ = 0.45), 105 of 105 cells contain at least one frame outside the pass rule. The weakest pooled v8 cells were:

| CFO (cycles/symbol) | Clock error (ppm) | Timing (samples) | Passing / frames | Pass share |
|---:|---:|---:|---:|---:|
| +0.005 | +2500 | +0 | 1,115 / 1,200 | 92.92% |
| -0.01 | +1250 | -2 | 1,119 / 1,200 | 93.25% |
| -0.005 | +0 | -2 | 1,120 / 1,200 | 93.33% |

**Boundary meaning:** The lowest empirical pooled pass-share cell among the discrete conditions tested at the highest declared noise_std; no interpolation or extrapolation is performed. Untested; no claim is made outside the finite grid above.

## Pre-identified weak cell

CFO `0`, clock error `-2500 ppm`, timing `2 samples`, σ `0.45`: v8 passes 1,150/1,200 frames (95.83%; loose joint acquisition 1,199/1,200), versus v7 1,034/1,200 (86.17%; loose joint acquisition 528/1,200). Paired McNemar p = 6.95e-24.

## Assumptions and limitations

- Signal model: `unchanged v7 received_buffer: t=(m-guard)*(1+clock_error_ppm*1e-6)+timing_offset`.
- Noise model: For positive noise_std, independent zero-mean Gaussian samples with the declared standard deviation are added to each generated buffer; every receiver is scored on the same noisy buffer.
- synthetic model only; no physical claim.
- known frame and symbol lengths, known waveform family and noise-only guards.
- static CFO and clock error, linear interpolation, no pulse shaping, jitter, filtering or time-varying impairments.
- blind searches are bounded by the declared envelope; the oracle is informed of impairments.
- integer timing offsets only in the impairment grid.

Pre-stated v8 improvement verdict: `True` (full-grid evaluation: `True`).

## Reproduce this passport

```bash
python perturbation_passport.py
```

The default command reads the source JSON, verifies its declared v8 schema and aggregate counts, then writes the JSON passport and this report. The source and generator SHA-256 values identify the exact inputs used.
