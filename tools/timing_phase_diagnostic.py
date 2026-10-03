"""Clean-frame diagnostic: how much of the v5 timing effect is carrier-phase rotation?

SYNTHETIC diagnostic added at repository packaging time (v0.1.0). It does not
modify or re-run the hash-locked v5 protocol; it only calls the unchanged
public functions of ``robustness_benchmark_v5_timing`` on clean (noise-free)
frames of the six fixed v2 vectors.

With 64 samples and 4 carrier cycles per symbol, the carrier advances
omega = 2*pi*4/64 rad = 22.5 degrees per sample. A window that starts Delta
samples early/late therefore sees its own symbol rotated by 22.5*Delta degrees
(and loses |Delta| of its 64 samples to the neighbouring symbol). The simple
phase-only prediction for the own-symbol gain of the unchanged in-phase
projection is ``(64 - |Delta|)/64 * cos(22.5 deg * Delta)``; this script
compares it with the gain measured by least squares on the clean decodes.

Run:  python tools/timing_phase_diagnostic.py
Writes artifacts/timing-phase-diagnostic.json and .csv (deterministic).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import robustness_benchmark_v5_timing as v5  # noqa: E402
from digital_to_wave import WaveConfig, decode_wave  # noqa: E402
from robustness_benchmark_v2 import build_input_patterns  # noqa: E402

OFFSETS = tuple(range(-32, 33))
FIELDS = (
    "timing_offset_samples",
    "carrier_rotation_deg",
    "measured_own_symbol_gain",
    "phase_only_predicted_gain",
    "abs_gain_gap",
    "clean_mae_nominal",
    "clean_frames_passing_nominal",
)


def diagnostic_rows(offsets=OFFSETS) -> list[dict[str, float | int]]:
    config = WaveConfig()
    omega_deg = 360.0 * config.cycles_per_symbol / config.samples_per_symbol
    patterns = build_input_patterns()
    rows = []
    for delta in offsets:
        gains, maes, passes = [], [], 0
        for name in v5.EXPECTED_PATTERN_NAMES:
            x = np.asarray(patterns[name], dtype=float)
            stream = v5._clean_stream(x, config, v5.GUARD_SAMPLES)
            y = decode_wave(stream[v5.window_indices(delta, method=v5.METHODS[0])], config)
            gains.append(float(x @ y / (x @ x)))
            maes.append(float(np.mean(np.abs(y - x))))
            passes += int(np.max(np.abs(y - x)) <= v5.TOLERANCE)
        predicted = (config.samples_per_symbol - abs(delta)) / config.samples_per_symbol * math.cos(
            math.radians(omega_deg * delta)
        )
        measured = float(np.mean(gains))
        rows.append(
            {
                "timing_offset_samples": int(delta),
                "carrier_rotation_deg": round(omega_deg * delta, 6),
                "measured_own_symbol_gain": round(measured, 10),
                "phase_only_predicted_gain": round(predicted, 10),
                "abs_gain_gap": round(abs(measured - predicted), 10),
                "clean_mae_nominal": round(float(np.mean(maes)), 10),
                "clean_frames_passing_nominal": passes,
            }
        )
    return rows


def main() -> int:
    rows = diagnostic_rows()
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    payload = {
        "diagnostic": "v5_clean_timing_carrier_phase_rotation",
        "evidence_status": "SYNTHETIC; clean frames only; diagnostic, not part of the hash-locked v5 protocol",
        "carrier_rotation_deg_per_sample": 22.5,
        "fixed_vectors": list(v5.EXPECTED_PATTERN_NAMES),
        "rows": rows,
    }
    (out / "timing-phase-diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out / "timing-phase-diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["timing_offset_samples"] in (-8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32):
            print(
                f"offset={row['timing_offset_samples']:+3d} rot={row['carrier_rotation_deg']:+7.1f} deg "
                f"gain={row['measured_own_symbol_gain']:+.4f} phase-only={row['phase_only_predicted_gain']:+.4f} "
                f"clean MAE={row['clean_mae_nominal']:.4f} pass={row['clean_frames_passing_nominal']}/6"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
