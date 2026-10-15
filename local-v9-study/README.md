# Digital-to-Wave v9 local study bundle

This folder contains the v9 generalization study as a public repository snapshot, including its preregistration, deterministic implementation, reduced-size tests, exact v1/v2/v7/v8 source dependencies, machine-readable results and a human-readable report.

The underlying project and this exploratory study remain **synthetic, dimensionless software testbench** materials. No quantum, diamond, biological, hardware, measured-channel or physical claim is made. The results materials disclose payload-seed overlap.
The protocol lock and generated run metadata retain the study's pre-publication status; this repository snapshot was published afterward, without changing the experiment or results.

## Reproduce

Requirements: Python 3.11+ with NumPy, Matplotlib and pytest (the parent repository pins its compatible dependencies). Run from this directory:

```bash
python -m pytest -q
python study.py --verify-lock
python study.py --run
```

The full run has 12 payload families × 12 independent channel seeds = 144 independent family × channel-seed clusters, two frames per cluster, eight conditions and six receiver arms. It may take several minutes because each v7/v8 blind search is intentionally rerun on paired buffers. The lock verification refuses to execute if the study code/config/vendor snapshots differ from the locally frozen SHA-256 manifest.

## Contents

- `protocol-lock.md`, `study_config.json`, `lock.json`: frozen design and exact lock.
- `study.py`: signal/channel generators, v7/v8/ablation/oracle execution, cluster-bootstrap inference, runtime/memory recording and output generation.
- `compat/`: unmodified snapshots of `digital_to_wave.py`, v2, v7 and v8, preserving exact baseline code.
- `tests/`: reduced tests only; they do not use the full-run master seed.
- `results/`: full paired outputs, summaries, confidence intervals, checksums, plot and report.
