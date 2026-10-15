# Digital-to-Wave v9 — local preregistration and protocol lock

**Status:** locally frozen before the full run; unpublished and not registered in a public registry. All observations will be **SYNTHETIC** and **dimensionless**. The original repository and its v1–v8 history are not modified. This study is software-only: **no quantum, diamond, biological, hardware, measured-channel, or physical-validity claim** is made.

## Research question

Do v7 and v8 blind synchronization results generalize beyond the six reused fixed payload vectors and the original static/IID-Gaussian channel model, when evaluated on unseen payload families and independent channel-seed clusters? Which isolated v8 change—candidate score, refinement search, or raw-sample decoder—accounts for any observed difference from v7, and how do the receivers behave under newly specified synthetic channel perturbations?

## Frozen sample and seed design

- Master seed: `20261015`; payload seed: `20261016`; channel seed: `20261017`; cluster-bootstrap seed: `20261018`. These are separate from the prior v7/v8 full-run, development and test seeds recorded in the repository.
- Twelve new payload-family generators are specified in `study.py`: continuous uniform; left/right beta skew; sparse-zero mixture; Markov run lengths; clipped Student-t; near-zero uniform; extreme-magnitude mixture; nonuniform five-level; three-mode mixture; balanced signs with continuous magnitudes; and sinusoidal envelope with jitter. Every payload is a newly generated 32-vector in `[-2,2]`. No six-vector panel from v2–v8 is an input.
- Each of the 12 families is crossed with 12 independently derived channel-seed streams: **144 independent payload-family × channel-seed clusters**. Each cluster contributes 2 replicate 32-symbol frames, kept together for inference. Thus each condition has 288 paired frames. Payload and channel random streams are deterministically derived from `(seed, family index, channel-seed index, frame index[, condition index])`.
- The independent unit for uncertainty is the payload-family × channel-seed cluster, not the frame. Report 95% percentile bootstrap intervals by resampling the 144 cluster means; the 2 frames within each selected cluster travel together. Paired receiver differences use per-cluster differences and the same cluster bootstrap. No frame-only Wilson intervals are used for v9 inference.

## Frozen conditions and controls

1. `clean_control`: ideal synthetic buffer, no noise or channel distortion.
2. `static_sync_control`: static off-grid timing `+1.25` samples, clock `+1250 ppm`, CFO `+0.0025 cycles/symbol`, no noise/filter.
3. `colored_noise_only`: stationary AR(1), `rho=0.80`, marginal standard deviation `0.25`.
4. `impulsive_noise_only`: Bernoulli impulse probability `0.01`, Gaussian impulse amplitude SD `1.25`, Gaussian background SD `0.05`.
5. `filter_frequency_selective_only`: causal two-path FIR taps `[0.82, 0, 0, 0.34]`, normalized to unit energy; no additional impairment/noise.
6. `fractional_timing_only`: `+0.375` samples, no additional impairment/noise.
7. `time_varying_cfo_clock_only`: CFO varies linearly from `-0.003` to `+0.003 cycles/symbol`; instantaneous clock rate error varies from `-100` to `+100 ppm`; trajectory mean is zero; no noise/filter.
8. `composite_unseen_channel`: fractional timing `+0.375`; the same time-varying CFO/clock trajectories; the same unit-energy two-path FIR; AR(1) noise with marginal SD `0.20`; sparse impulses with probability `0.005`, impulse SD `1.0`, background SD `0.04`.

The two noise-only controls and four isolated channel/impairment controls are reported separately from the composite condition. Parameters are deterministic design choices, not fitted to observed outcomes and not asserted to represent a real communications channel.

## Receiver arms and ablations

All receiver methods process the **same exact generated buffer** for a condition/cluster/frame. The frozen v7/v8 code is vendored byte-for-byte under `compat/` and SHA-256 locked with the new study code before the full run.

- `v7_blind`: unchanged v7 blind joint search, its candidate-energy score, and its interpolation decoder.
- `score_only_ablation`: v7 candidate grid and interpolation decoder; replace only the coarse candidate score with v8's raw-sample GLRT.
- `refinement_only_ablation`: v7 candidate-energy score and interpolation decoder; replace only the single-best coarse choice with v8-style top-three ridge-coordinate coarse-to-fine refinement.
- `decoder_only_ablation`: unchanged v7 selected parameters/search; replace only decoding with the v8 raw-sample least-squares fit.
- `v8_full`: unchanged v8 full blind receiver, which bundles score, refinement and decoder changes.
- `raw_ls_informed_oracle`: exact static or mean timing/clock/CFO reference coordinates supplied to the existing v8 raw-LS decoder. All methods see the same received buffers.

The v8-v7 aggregate contrast is **not** attributed to a single component. Component interpretations rely on the three one-change ablations, and interactions may remain. In time-varying conditions, the informed raw-LS reference receives the declared mean/static coordinates but uses the existing static decoder; in filtered or time-varying conditions it is **not a true upper bound**, equalizer, or tracker.

## Frozen outcomes

- Exact-frame pass: maximum absolute reconstruction error across the full 32-value frame `<= 0.25` (inclusive), unchanged from the repository.
- Acquisition: blind estimates compared separately against the known static reference coordinates or the zero-mean/central coordinates of the time-varying trajectories. Report loose joint acquisition (all coordinates within `2 samples`, `2500 ppm`, `0.005 cycles/symbol`) and strict joint acquisition (`1 sample`, `1250 ppm`, `0.0025 cycles/symbol`), along with mean absolute timing, clock and CFO errors. Center-coordinate acquisition in a varying trajectory does not measure tracking error across the whole frame.
- Runtime: wall-clock duration by condition and receiver; total seconds, average milliseconds per scored frame, and per-cluster-call p50/p95.
- Memory: process-wide peak RSS in MB using Linux `ru_maxrss`; this is not method-attributed.
- Paired comparisons: pass discordance counts and paired pass-share differences with cluster-bootstrap 95% intervals, including each ablation versus v7 and full v8 versus v7 and the informed raw-LS reference.
- Reproducibility: `results.json`, condition/frame/cluster/paired CSVs, buffer SHA-256 catalog, report, plot, and artifact checksums.

## Frozen implementation and execution controls

- Full run uses the constants in `study_config.json`; only this config and `study.py` define the study.
- `lock.json` records SHA-256 of `study.py`, `study_config.json`, `tests/test_study.py`, and each vendored v1/v2/v7/v8 source module. `python study.py --run` fails closed if any locked file differs.
- Reduced deterministic tests use a smaller configuration and distinct seeds. They do not run the full-study master seed.
- Run only after this protocol, config, source, vendor snapshots, tests and lock manifest exist. Do not revise them after viewing full-run outcomes. A rerun is allowed only as exact reproduction and should not change the analysis plan.
- The source repository is public and read-only during the task. Do not commit, push, upload, publish, release, or create a public registration.

## Scope boundary

The testbench is a synthetic, dimensionless signal-processing software study. It does not simulate or test quantum systems, diamond, NV centers, biology, laboratory hardware, a measured propagation channel, or the validity of a preprint. The small finite generated design supports claims only about the conditions and payload-family generators explicitly frozen above.
