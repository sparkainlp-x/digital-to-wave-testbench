# Digital-to-Wave v9 — exploratory generalization study
> **Status:** public repository snapshot of an exploratory, synthetic and dimensionless software study; results were generated locally before publication. No quantum, diamond, biological, hardware, or physical-channel claim is made.

## Frozen design and provenance

- Frozen protocol lock SHA-256: `bdd9210af42a8cccb4f4d89d6cabd0b05e1c7d6a8565f71da28dc4562f4910db`
- Source commit snapshot: `17a848c05c2e4a168ac12525f5a8ca4000dff098`
- Runtime: Python 3.12.3, NumPy 2.5.3, Matplotlib 3.11.2 (recorded rather than silently substituting the repository pins).
- Payload families: 12; independent family × channel-seed clusters: 144; frames per cluster: 2
- 288 paired frames per condition; 8 conditions; pass iff max absolute error across 32 values ≤ 0.25
- Confidence intervals are 95% percentile cluster-bootstrap intervals over independent payload-family × channel-seed units; replicate frames remain together. They are not frame-only intervals.
- Every receiver arm saw the same buffer within each condition/cluster/frame; SHA-256 hashes are in `buffer-hashes.csv`.
- Full v8 versus v7 is a bundled comparison. Score-only, refinement-only and decoder-only arms change one component at a time relative to v7; do not ascribe the bundled v8 difference to one component.

## Exact-frame pass results

Pass share with cluster-bootstrap 95% CI; counts use the exact all-32-symbol frame rule.

| Condition | Receiver | Exact-frame passes | Pass share (cluster-bootstrap 95% CI) |
|---|---|---:|---:|
| clean_control | `v7_blind` | 288/288 | 1.000 (1.000–1.000) |
| clean_control | `score_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| clean_control | `refinement_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| clean_control | `decoder_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| clean_control | `v8_full` | 288/288 | 1.000 (1.000–1.000) |
| clean_control | `raw_ls_informed_oracle` | 288/288 | 1.000 (1.000–1.000) |
| static_sync_control | `v7_blind` | 288/288 | 1.000 (1.000–1.000) |
| static_sync_control | `score_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| static_sync_control | `refinement_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| static_sync_control | `decoder_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| static_sync_control | `v8_full` | 288/288 | 1.000 (1.000–1.000) |
| static_sync_control | `raw_ls_informed_oracle` | 288/288 | 1.000 (1.000–1.000) |
| colored_noise_only | `v7_blind` | 288/288 | 1.000 (1.000–1.000) |
| colored_noise_only | `score_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| colored_noise_only | `refinement_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| colored_noise_only | `decoder_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| colored_noise_only | `v8_full` | 288/288 | 1.000 (1.000–1.000) |
| colored_noise_only | `raw_ls_informed_oracle` | 288/288 | 1.000 (1.000–1.000) |
| impulsive_noise_only | `v7_blind` | 288/288 | 1.000 (1.000–1.000) |
| impulsive_noise_only | `score_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| impulsive_noise_only | `refinement_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| impulsive_noise_only | `decoder_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| impulsive_noise_only | `v8_full` | 288/288 | 1.000 (1.000–1.000) |
| impulsive_noise_only | `raw_ls_informed_oracle` | 288/288 | 1.000 (1.000–1.000) |
| filter_frequency_selective_only | `v7_blind` | 281/288 | 0.976 (0.955–0.993) |
| filter_frequency_selective_only | `score_only_ablation` | 285/288 | 0.990 (0.976–1.000) |
| filter_frequency_selective_only | `refinement_only_ablation` | 271/288 | 0.941 (0.910–0.969) |
| filter_frequency_selective_only | `decoder_only_ablation` | 278/288 | 0.965 (0.941–0.986) |
| filter_frequency_selective_only | `v8_full` | 241/288 | 0.837 (0.785–0.885) |
| filter_frequency_selective_only | `raw_ls_informed_oracle` | 288/288 | 1.000 (1.000–1.000) |
| fractional_timing_only | `v7_blind` | 288/288 | 1.000 (1.000–1.000) |
| fractional_timing_only | `score_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| fractional_timing_only | `refinement_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| fractional_timing_only | `decoder_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| fractional_timing_only | `v8_full` | 288/288 | 1.000 (1.000–1.000) |
| fractional_timing_only | `raw_ls_informed_oracle` | 288/288 | 1.000 (1.000–1.000) |
| time_varying_cfo_clock_only | `v7_blind` | 288/288 | 1.000 (1.000–1.000) |
| time_varying_cfo_clock_only | `score_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| time_varying_cfo_clock_only | `refinement_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| time_varying_cfo_clock_only | `decoder_only_ablation` | 288/288 | 1.000 (1.000–1.000) |
| time_varying_cfo_clock_only | `v8_full` | 288/288 | 1.000 (1.000–1.000) |
| time_varying_cfo_clock_only | `raw_ls_informed_oracle` | 288/288 | 1.000 (1.000–1.000) |
| composite_unseen_channel | `v7_blind` | 169/288 | 0.587 (0.531–0.642) |
| composite_unseen_channel | `score_only_ablation` | 169/288 | 0.587 (0.531–0.642) |
| composite_unseen_channel | `refinement_only_ablation` | 67/288 | 0.233 (0.174–0.292) |
| composite_unseen_channel | `decoder_only_ablation` | 168/288 | 0.583 (0.528–0.636) |
| composite_unseen_channel | `v8_full` | 54/288 | 0.188 (0.135–0.243) |
| composite_unseen_channel | `raw_ls_informed_oracle` | 286/288 | 0.993 (0.983–1.000) |

## Synchronization/acquisition and error

Blind receiver errors are relative to the frozen static or mean trajectory coordinates in the condition definition. For time-varying conditions, center-coordinate acquisition is not a measure of tracking error across the entire trajectory. The informed raw-LS arm has no acquisition estimate; it is supplied the declared static/mean reference parameters.

| Condition | Blind receiver | Loose joint acquisition | Strict joint acquisition | Mean abs timing error (samples) | Mean abs clock error (ppm) | Mean abs CFO error (cycles/symbol) |
|---|---|---:|---:|---:|---:|---:|
| clean_control | `v7_blind` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| clean_control | `score_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| clean_control | `refinement_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| clean_control | `decoder_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| clean_control | `v8_full` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| static_sync_control | `v7_blind` | 288/288 (1.000) | 288/288 (1.000) | 0.75 | 1250 | 0.0025 |
| static_sync_control | `score_only_ablation` | 286/288 (0.993) | 286/288 (0.993) | 0.75 | 1250 | 0.002535 |
| static_sync_control | `refinement_only_ablation` | 210/288 (0.729) | 205/288 (0.712) | 0.3056 | 549 | 0.002503 |
| static_sync_control | `decoder_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0.75 | 1250 | 0.0025 |
| static_sync_control | `v8_full` | 288/288 (1.000) | 288/288 (1.000) | 0.0816 | 52.08 | 0.0006307 |
| colored_noise_only | `v7_blind` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| colored_noise_only | `score_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| colored_noise_only | `refinement_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 2.387e-05 |
| colored_noise_only | `decoder_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| colored_noise_only | `v8_full` | 285/288 (0.990) | 267/288 (0.927) | 0.006944 | 211.6 | 0.0008777 |
| impulsive_noise_only | `v7_blind` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| impulsive_noise_only | `score_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| impulsive_noise_only | `refinement_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| impulsive_noise_only | `decoder_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| impulsive_noise_only | `v8_full` | 288/288 (1.000) | 285/288 (0.990) | 0.002604 | 113.9 | 0.0004663 |
| filter_frequency_selective_only | `v7_blind` | 288/288 (1.000) | 164/288 (0.569) | 0.4861 | 208.3 | 0.002153 |
| filter_frequency_selective_only | `score_only_ablation` | 288/288 (1.000) | 61/288 (0.212) | 1.403 | 1753 | 0.003941 |
| filter_frequency_selective_only | `refinement_only_ablation` | 287/288 (0.997) | 285/288 (0.990) | 0.7778 | 8.681 | 0.001055 |
| filter_frequency_selective_only | `decoder_only_ablation` | 288/288 (1.000) | 164/288 (0.569) | 0.4861 | 208.3 | 0.002153 |
| filter_frequency_selective_only | `v8_full` | 36/288 (0.125) | 0/288 (0.000) | 0.8299 | 1519 | 0.005956 |
| fractional_timing_only | `v7_blind` | 288/288 (1.000) | 288/288 (1.000) | 0.375 | 0 | 0 |
| fractional_timing_only | `score_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0.375 | 0 | 0 |
| fractional_timing_only | `refinement_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0.3915 | 0 | 0.001287 |
| fractional_timing_only | `decoder_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0.375 | 0 | 0 |
| fractional_timing_only | `v8_full` | 288/288 (1.000) | 288/288 (1.000) | 0.125 | 299.5 | 0.0007109 |
| time_varying_cfo_clock_only | `v7_blind` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| time_varying_cfo_clock_only | `score_only_ablation` | 287/288 (0.997) | 286/288 (0.993) | 0.006944 | 17.36 | 5.208e-05 |
| time_varying_cfo_clock_only | `refinement_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0.7847 | 0 | 0.00171 |
| time_varying_cfo_clock_only | `decoder_only_ablation` | 288/288 (1.000) | 288/288 (1.000) | 0 | 0 | 0 |
| time_varying_cfo_clock_only | `v8_full` | 288/288 (1.000) | 288/288 (1.000) | 0.3594 | 128 | 0.0006156 |
| composite_unseen_channel | `v7_blind` | 250/288 (0.868) | 206/288 (0.715) | 0.6389 | 138.9 | 0.001424 |
| composite_unseen_channel | `score_only_ablation` | 189/288 (0.656) | 152/288 (0.528) | 1.056 | 859.4 | 0.002378 |
| composite_unseen_channel | `refinement_only_ablation` | 284/288 (0.986) | 160/288 (0.556) | 0.7969 | 26.04 | 0.001582 |
| composite_unseen_channel | `decoder_only_ablation` | 250/288 (0.868) | 206/288 (0.715) | 0.6389 | 138.9 | 0.001424 |
| composite_unseen_channel | `v8_full` | 33/288 (0.115) | 0/288 (0.000) | 1.092 | 1515 | 0.006131 |

Acquisition CIs are cluster-bootstrap intervals and are available in `condition-summary.csv`/`results.json` for each blind arm.

## One-change ablations and paired contrasts

Each contrast below is a paired pass-share difference over the same buffers, with a cluster-bootstrap 95% CI. A CI including zero does not establish a difference.

| Condition | A − B | Paired pass-share difference | Cluster-bootstrap 95% CI | A-only / B-only frame passes |
|---|---|---:|---:|---:|
| clean_control | `v8_full` − `raw_ls_informed_oracle` | +0.000 | +0.000–+0.000 | 0 / 0 |
| static_sync_control | `v8_full` − `raw_ls_informed_oracle` | +0.000 | +0.000–+0.000 | 0 / 0 |
| colored_noise_only | `v8_full` − `raw_ls_informed_oracle` | +0.000 | +0.000–+0.000 | 0 / 0 |
| impulsive_noise_only | `v8_full` − `raw_ls_informed_oracle` | +0.000 | +0.000–+0.000 | 0 / 0 |
| filter_frequency_selective_only | `v8_full` − `raw_ls_informed_oracle` | -0.163 | -0.215–-0.115 | 0 / 47 |
| fractional_timing_only | `v8_full` − `raw_ls_informed_oracle` | +0.000 | +0.000–+0.000 | 0 / 0 |
| time_varying_cfo_clock_only | `v8_full` − `raw_ls_informed_oracle` | +0.000 | +0.000–+0.000 | 0 / 0 |
| composite_unseen_channel | `v8_full` − `raw_ls_informed_oracle` | -0.806 | -0.858–-0.753 | 0 / 232 |

## Runtime and memory

Peak resident set size for the complete study process: **93.5 MB** (Linux `ru_maxrss`; process-wide peak, not method-attributed). Method runtime is cumulative wall-clock time; per-frame runtime divides by scored frames. Per-call p50/p95 and condition-by-method values are in `condition-summary.csv`.

| Condition | Receiver | Total runtime (s) | Mean runtime (ms/frame) | Call p50 / p95 (ms) |
|---|---|---:|---:|---:|
| clean_control | `v7_blind` | 6.2222 | 21.6047 | 41.718 / 54.564 |
| clean_control | `score_only_ablation` | 9.2219 | 32.0205 | 62.262 / 78.739 |
| clean_control | `refinement_only_ablation` | 28.9835 | 100.6373 | 194.498 / 248.315 |
| clean_control | `decoder_only_ablation` | 6.3101 | 21.9101 | 42.511 / 51.535 |
| clean_control | `v8_full` | 23.0414 | 80.0050 | 153.054 / 206.370 |
| clean_control | `raw_ls_informed_oracle` | 0.0836 | 0.2904 | 0.554 / 0.728 |
| static_sync_control | `v7_blind` | 7.4350 | 25.8161 | 43.782 / 78.032 |
| static_sync_control | `score_only_ablation` | 11.4320 | 39.6945 | 66.158 / 133.598 |
| static_sync_control | `refinement_only_ablation` | 34.7884 | 120.7929 | 211.702 / 377.699 |
| static_sync_control | `decoder_only_ablation` | 7.4607 | 25.9051 | 44.615 / 76.814 |
| static_sync_control | `v8_full` | 29.7481 | 103.2920 | 174.908 / 365.789 |
| static_sync_control | `raw_ls_informed_oracle` | 0.1031 | 0.3582 | 0.610 / 1.157 |
| colored_noise_only | `v7_blind` | 7.1439 | 24.8052 | 43.227 / 76.846 |
| colored_noise_only | `score_only_ablation` | 10.7257 | 37.2422 | 63.665 / 135.581 |
| colored_noise_only | `refinement_only_ablation` | 32.8135 | 113.9356 | 201.084 / 371.018 |
| colored_noise_only | `decoder_only_ablation` | 7.1495 | 24.8248 | 43.970 / 79.071 |
| colored_noise_only | `v8_full` | 27.1102 | 94.1327 | 165.879 / 314.953 |
| colored_noise_only | `raw_ls_informed_oracle` | 0.0990 | 0.3437 | 0.572 / 1.120 |
| impulsive_noise_only | `v7_blind` | 6.5024 | 22.5779 | 42.164 / 66.719 |
| impulsive_noise_only | `score_only_ablation` | 9.8640 | 34.2501 | 63.504 / 98.926 |
| impulsive_noise_only | `refinement_only_ablation` | 30.0230 | 104.2464 | 196.482 / 281.047 |
| impulsive_noise_only | `decoder_only_ablation` | 6.7041 | 23.2782 | 43.381 / 68.166 |
| impulsive_noise_only | `v8_full` | 24.4441 | 84.8752 | 159.134 / 242.445 |
| impulsive_noise_only | `raw_ls_informed_oracle` | 0.0894 | 0.3104 | 0.574 / 0.927 |
| filter_frequency_selective_only | `v7_blind` | 6.7367 | 23.3912 | 42.428 / 74.815 |
| filter_frequency_selective_only | `score_only_ablation` | 10.3496 | 35.9360 | 63.234 / 115.120 |
| filter_frequency_selective_only | `refinement_only_ablation` | 32.1136 | 111.5057 | 200.415 / 341.308 |
| filter_frequency_selective_only | `decoder_only_ablation` | 6.8997 | 23.9574 | 43.381 / 75.568 |
| filter_frequency_selective_only | `v8_full` | 29.7060 | 103.1457 | 189.839 / 324.540 |
| filter_frequency_selective_only | `raw_ls_informed_oracle` | 0.0944 | 0.3278 | 0.592 / 1.105 |
| fractional_timing_only | `v7_blind` | 7.0140 | 24.3541 | 43.093 / 70.690 |
| fractional_timing_only | `score_only_ablation` | 10.4853 | 36.4072 | 63.844 / 105.286 |
| fractional_timing_only | `refinement_only_ablation` | 31.7073 | 110.0950 | 203.367 / 297.076 |
| fractional_timing_only | `decoder_only_ablation` | 7.0344 | 24.4251 | 44.023 / 69.278 |
| fractional_timing_only | `v8_full` | 26.6574 | 92.5604 | 170.314 / 279.444 |
| fractional_timing_only | `raw_ls_informed_oracle` | 0.0932 | 0.3236 | 0.581 / 0.977 |
| time_varying_cfo_clock_only | `v7_blind` | 6.7440 | 23.4166 | 42.092 / 74.626 |
| time_varying_cfo_clock_only | `score_only_ablation` | 10.1190 | 35.1353 | 62.541 / 124.835 |
| time_varying_cfo_clock_only | `refinement_only_ablation` | 30.5639 | 106.1248 | 191.766 / 359.120 |
| time_varying_cfo_clock_only | `decoder_only_ablation` | 6.7390 | 23.3995 | 42.597 / 75.065 |
| time_varying_cfo_clock_only | `v8_full` | 26.3616 | 91.5332 | 164.149 / 341.036 |
| time_varying_cfo_clock_only | `raw_ls_informed_oracle` | 0.0901 | 0.3129 | 0.552 / 1.067 |
| composite_unseen_channel | `v7_blind` | 6.7048 | 23.2805 | 43.216 / 71.057 |
| composite_unseen_channel | `score_only_ablation` | 10.1251 | 35.1565 | 64.870 / 103.452 |
| composite_unseen_channel | `refinement_only_ablation` | 31.4935 | 109.3523 | 203.484 / 304.886 |
| composite_unseen_channel | `decoder_only_ablation` | 6.7952 | 23.5943 | 43.983 / 72.299 |
| composite_unseen_channel | `v8_full` | 29.6685 | 103.0155 | 193.945 / 298.071 |
| composite_unseen_channel | `raw_ls_informed_oracle` | 0.0942 | 0.3272 | 0.594 / 1.030 |

## Interpretation and scope limits

- This is a generated-data software stress test of the specified signed-amplitude sine signal and fixed receiver assumptions. It is not measurement, hardware testing, or validation of any physical system or preprint.
- Filtering, colored noise, impulses, fractional timing, time-varying CFO/clock and compound channel behavior are deliberately synthetic and dimensionless; the design does not imply realistic channel distributions.
- The raw-LS informed reference knows the static or mean timing/clock/CFO coordinates but uses the existing static raw-fit decoder. It is not a true upper bound for filtering or time-varying channels and has no equalizer/tracker for those effects.
- Results generalize only over the prespecified synthetic payload-family distributions, random seeds and finite conditions here. They do not establish performance outside the study grid.
- Nothing was committed, pushed, published, or uploaded. The bundle is for local review only.

## Reproduction

Run from this directory with `python study.py --run`. The script verifies `lock.json` against the frozen source/configuration SHA-256 values before executing. `python -m pytest -q` runs reduced deterministic tests; the reduced study test reuses a small set of payload-seed coordinates but, because it re-enumerates the selected conditions, does not reuse the noisy full-run channel buffers. See the deviation note below.


## Reporting correction

The initially rendered paired-contrast table matched one stored pair orientation and omitted v7-directed ablation contrasts. The full normalized contrasts are in [`report-correction.md`](report-correction.md) and [`paired-ablation-summary.csv`](paired-ablation-summary.csv). They are derived only from the already saved paired rows; no experiment rerun or data alteration occurred.


## Protocol deviation: reduced-test seed overlap

A final audit found a narrower but real deviation from the protocol’s distinct-test-seed statement. The reduced test reuses exact payload RNG coordinates for the first two payload families, first two channel-seed indices, and two frames; payload RNG does not include the condition index. It selects full-design conditions 0, 4, 6, and 7, but the reduced runner re-enumerates them as condition indices 0–3 for channel RNG. Thus only the clean condition has an identical full-run channel seed coordinate; the filtered/time-varying controls are noise-free, and the composite condition’s reduced channel stream is not the full-run composite stream. The test checks determinism and structure, not performance. This weakens independent-test-seed separation, but does not mean the selected noisy full-run channel buffers were reused wholesale. Numerical results and run artifacts remain unchanged. See [`protocol-deviation.md`](protocol-deviation.md); characterize this as a locally frozen exploratory study with disclosed payload-seed overlap and condition-index remapping, not a pristine preregistration.
