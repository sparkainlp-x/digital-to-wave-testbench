#!/usr/bin/env bash
# Regenerate every committed artifact (v2-v5 JSON/CSV/PNG, timing diagnostic) and SHA256SUMS.
# Runtime: about 20 s on a laptop. Outputs are deterministic for a given environment;
# CSV and PNG files are byte-identical with the pinned dependencies (see pyproject.toml).
# JSON files embed the Python/NumPy/Matplotlib versions, so their hashes are environment-specific.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" robustness_benchmark_v2.py
"$PY" robustness_benchmark_v3.py
"$PY" robustness_benchmark_v4.py
"$PY" robustness_benchmark_v5_timing.py
"$PY" tools/timing_phase_diagnostic.py
(cd artifacts && sha256sum -- *.csv *.json *.png > SHA256SUMS)
echo "wrote artifacts/SHA256SUMS"
