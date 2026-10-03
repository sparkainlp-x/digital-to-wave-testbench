#!/usr/bin/env bash
# Regenerate every committed artifact (v2-v8 JSON/CSV/PNG, timing diagnostic) and SHA256SUMS.
# Runtime: about 4.5 min on a laptop without v8 (v7 alone is about 3 min, v6 about 70 s); v8 adds about 6 min on
# 8 cores (about 27 CPU-minutes; cells run in parallel and results do not depend on the worker count).
# Set SKIP_V8=1 to keep the committed v8 artifacts (CI does this: a full v8 re-run would exceed its time budget, so
# CI instead re-runs two full-size v8 cells in tests/test_v8_ml_sync.py and compares them with the committed rows).
# Outputs are deterministic for a given environment; CSV and PNG files are byte-identical with the pinned
# dependencies (see pyproject.toml). JSON files embed the Python/NumPy/Matplotlib versions and the Git commit,
# so their hashes are environment-specific.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" robustness_benchmark_v2.py
"$PY" robustness_benchmark_v3.py
"$PY" robustness_benchmark_v4.py
"$PY" robustness_benchmark_v5_timing.py
"$PY" robustness_benchmark_v6_timing_recovery.py
"$PY" robustness_benchmark_v7_clock_drift_cfo.py
if [ "${SKIP_V8:-0}" != "1" ]; then
  "$PY" robustness_benchmark_v8_ml_sync.py
else
  echo "SKIP_V8=1: keeping the committed v8 artifacts"
fi
"$PY" tools/timing_phase_diagnostic.py
(cd artifacts && sha256sum -- *.csv *.json *.png > SHA256SUMS)
echo "wrote artifacts/SHA256SUMS"
