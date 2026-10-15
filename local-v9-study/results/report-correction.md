# Post-run reporting correction — v7-directed ablation contrasts

> This supplement corrects a presentation-filter issue in `report.md`: the initial table matched only the stored pair orientation, while `paired-comparisons.csv` stores method pairs in receiver-list order. It therefore omitted v7-versus-later-arm contrasts. Values below are normalized directly from the already saved paired rows; no experiment was rerun, no outcome data were modified, and no analysis-plan choice was changed. The full normalized table is `paired-ablation-summary.csv`.

## Outcome-changing contrasts

The following five preregistered contrasts had at least one discordant exact-frame pass in the two conditions with any pass discordance. Difference is A minus B, with the paired cluster-bootstrap 95% interval; A-only/B-only are frame discordances.

| Condition | A − B | Pass-share difference | Cluster-bootstrap 95% CI | A-only / B-only |
|---|---|---:|---:|---:|
| filter_frequency_selective_only | `score_only_ablation - v7_blind` | +0.0139 | +0.0035–+0.0278 | 4 / 0 |
| filter_frequency_selective_only | `refinement_only_ablation - v7_blind` | -0.0347 | -0.0625–-0.0069 | 3 / 13 |
| filter_frequency_selective_only | `decoder_only_ablation - v7_blind` | -0.0104 | -0.0243–-0.0000 | 0 / 3 |
| filter_frequency_selective_only | `v8_full - v7_blind` | -0.1389 | -0.1875–-0.0938 | 1 / 41 |
| filter_frequency_selective_only | `v8_full - raw_ls_informed_oracle` | -0.1632 | -0.2153–-0.1146 | 0 / 47 |
| composite_unseen_channel | `score_only_ablation - v7_blind` | -0.0000 | -0.0312–+0.0312 | 10 / 10 |
| composite_unseen_channel | `refinement_only_ablation - v7_blind` | -0.3542 | -0.4097–-0.2986 | 3 / 105 |
| composite_unseen_channel | `decoder_only_ablation - v7_blind` | -0.0035 | -0.0104–-0.0000 | 0 / 1 |
| composite_unseen_channel | `v8_full - v7_blind` | -0.3993 | -0.4549–-0.3472 | 3 / 118 |
| composite_unseen_channel | `v8_full - raw_ls_informed_oracle` | -0.8056 | -0.8576–-0.7535 | 0 / 232 |

Across the remaining six controls, all five listed contrasts had zero discordant exact-frame pass outcomes. In the filter-only condition, score-only gained 4 frames over v7, refinement-only lost 10, decoder-only lost 3, and full v8 lost 40. In the composite condition, score-only had equal marginal pass counts (10 discordances each way), refinement-only lost 102 frames, decoder-only lost 1, and full v8 lost 115. These isolated contrasts are not additive: the full v8 result bundles changes and interactions, so its loss is not attributed to one change alone.

The informed raw-LS comparison remains an informed static/mean-parameter reference, not a true upper bound for the filtered or time-varying channel models. A source-level audit also refined the seed-overlap disclosure: reduced-test payload coordinates overlap the full run, while condition-index remapping means the selected noisy channel buffers were not reused wholesale. See `protocol-deviation.md` and `report.md` for the exact accounting, scope, and runtime/memory disclosures.
