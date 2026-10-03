# v7 clock-drift and carrier-frequency-offset protocol lock

**Evidence status:** protocol only; results will be **SYNTHETIC**. No physical validity is claimed.
This exploratory software protocol is not a formal preregistration.

## Question

Can a blind joint timing-and-carrier search recover the fixed decoder's performance when constant carrier-frequency offset (CFO) and receiver sample-clock error occur together in the synthetic dimensionless frame model?

## Signal and conventions

The unchanged v1 encoder defines a frame of 32 symbols, 64 nominal samples per symbol, four carrier cycles per symbol, and dimensionless input values in `[-2, 2]`. In a 64-sample-guarded buffer, received index `m` represents source-time coordinate

`t(m) = (m - 64) * (1 + clock_error_ppm * 1e-6) + timing_offset_samples`.

The emitted signal is zero outside source time `[0, 2048)`. Inside the frame, symbol `k = floor(t/64)` has signed phase `π/4` for a positive value and `5π/4` for a negative value. Its carrier phase is `2π * (4 + cfo_cycles_per_symbol) * t / 64`. Positive timing offset means the receiver is late; positive clock error means its source-time coordinate advances faster per received sample; positive CFO increases carrier cycles per symbol. Noise is IID Gaussian in received samples, including guards.

The model is deliberately direct: no pulse shaping, channel filtering, clock jitter, phase noise, time-varying CFO, or hardware model.

## Frozen conditions

- Master seed: `20261010`; vector seed: `20261004`. RNG streams are independently derived from master seed, impairment-cell index, vector index, and noise level.
- Six fixed v2 vectors; 32 values per frame; 200 noisy frames per vector per impairment cell; clean controls; chunking does not affect output.
- CFO values: `{-0.005, 0, +0.005}` extra cycles per nominal symbol.
- Receiver clock errors: `{-2500, 0, +2500}` ppm. Over 32 symbols these produce approximately `±5.12` nominal samples of accumulated timing drift.
- Constant timing offsets: `{-2, 0, +2}` samples.
- Full factorial across CFO, clock error and timing offset, so the matrix includes impairment-free, single-impairment and combined cases. Noise standard deviations: `0`, `0.25` and `0.45`; the two nonzero levels provide more than one stochastic-noise condition.
- Locked implementation: `robustness_benchmark_v7_clock_drift_cfo.py`, SHA-256 `693ac18c9202f36a004610147f76642ae3f2859b64dd39e1d5c4a7f0cc6f7e84`, also recorded in `MODULES.sha256` and `tests/test_hash_lock.py`. This protocol and the module/tests are committed before the full run.
- Inclusive frame pass rule: maximum absolute reconstruction error `≤ 0.25` over 32 values.

## Receivers and metrics

- **Nominal:** unchanged v1 `decode_wave` with fixed 64-sample windows and no correction.
- **Oracle:** receiver informed of exact timing offset, clock error and CFO; resamples at exact symbol sample times and projects onto the corresponding carrier template.
- **Blind joint search:** evaluates candidate timing offsets `{-4, -2, 0, 2, 4}` samples, clock errors `{-5000, -2500, 0, 2500, 5000}` ppm, and CFO `{-0.01, -0.005, 0, 0.005, 0.01}` cycles/symbol. It chooses the candidate maximizing summed squared decoded values; exact ties prefer the smallest absolute timing, clock and CFO corrections, then their signed values. Every decoded value is produced by the v7 receiver's least-squares projection. The search is not told the impairment cell.

Methods use the same generated noisy buffers. Report acquisition separately (all three estimated parameters within one candidate-grid step of truth), as well as frame pass counts, Wilson intervals, error summaries, and paired discordant counts/exact McNemar p-values. Preserve per-vector and pooled rows.

## Outputs and reproducibility

The full run writes `artifacts/robustness-benchmark-v7-clock-drift-cfo.json`, `.csv`, paired-comparison CSV and `.png`. JSON records protocol values, environment, Git provenance and SHA-256 hashes for v1-v7. The v7 implementation, tests and this lock must be committed before running the full protocol. The committed result files are then added to `artifacts/SHA256SUMS`.

The tests use an independent seed and reduced trial counts; they verify waveform construction, exact oracle recovery in a clean impairment-free case, blind-search API independence from the truth, determinism, validator rejection of tampering, and module hash consistency.

## Interpretation limits

This is a synthetic dimensionless test of a specified signal and receiver model. Its oracle is an informed upper bound, and the blind search assumes known frame/symbol lengths, known carrier family, known guard size and bounded static impairment grids. It does not establish physical performance or address unmodeled impairments.
