#!/usr/bin/env bash
# Regenerate every committed artifact (v2-v7 JSON/CSV/PNG, timing diagnostic) and SHA256SUMS.
# Runtime: about 4.5 min on a laptop (v7 alone is about 3 min, v6 about 70 s). Outputs are deterministic for a given environment;
# CSV and PNG files are byte-identical with the pinned dependencies (see pyproject.toml).
# JSON files embed the Python/NumPy/Matplotlib versions, so their hashes are environment-specific.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" robustness_benchmark_v2.py
"$PY" robustness_benchmark_v3.py
"$PY" robustness_benchmark_v4.py
"$PY" robustness_benchmark_v5_timing.py
"$PY" robustness_benchmark_v6_timing_recovery.py
"$PY" robustness_benchmark_v7_clock_drift_cfo.py
"$PY" tools/timing_phase_diagnostic.py
(cd artifacts && sha256sum -- *.csv *.json *.png > SHA256SUMS)
echo "wrote artifacts/SHA256SUMS"
