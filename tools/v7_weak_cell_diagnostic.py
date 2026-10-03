"""Diagnose the v7 weak cell (CFO 0, -2500 ppm, +2 samples, sigma 0.45) with the v8 development seed.

Small, stdout-only diagnostic used before the v8 protocol lock. It never touches the v7 or v8 master seeds.
SYNTHETIC: dimensionless model, no physical validity claimed.

    python tools/v7_weak_cell_diagnostic.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robustness_benchmark_v7_clock_drift_cfo as v7  # noqa: E402
import robustness_benchmark_v8_ml_sync as v8  # noqa: E402
from robustness_benchmark_v2 import build_input_patterns  # noqa: E402

FRAMES_PER_VECTOR = 50


def main() -> int:
    patterns = build_input_patterns()
    cfo, ppm, delay, sigma = v8.WEAK_CELL
    truth, ridge = (delay, ppm, cfo), (delay, 0.0, -0.01)
    print(f"development seed {v8.DEV_SEED}; weak cell CFO {cfo}, clock {ppm} ppm, timing {delay}, sigma {sigma}")
    print(f"carrier offset of the truth: {v8.carrier_offset(ppm, cfo):+.6f} cycles/symbol; "
          f"of the ridge point (0 ppm, -0.01): {v8.carrier_offset(0.0, -0.01):+.6f}")
    print("\nclean scores per vector (v7 metric = sum of squared decoded values; GLRT = raw-sample LS energy)")
    for name, vector in patterns.items():
        buffer = v7.received_buffer(vector, cfo_cycles_per_symbol=cfo, clock_error_ppm=ppm,
                                    timing_offset_samples=delay)[None, :]
        v7_t, v7_r = (float(np.sum(v7._candidate_decode(buffer, *c) ** 2)) for c in (truth, ridge))
        ml_t, ml_r = (float(v8.ml_score(buffer, *c)[0]) for c in (truth, ridge))
        print(f"  {name:18s} v7 truth {v7_t:7.3f} ridge {v7_r:7.3f} | GLRT truth {ml_t:8.3f} ridge {ml_r:8.3f}")
    choices: Counter = Counter()
    passes = {"oracle": 0, "v7": 0, "v8": 0}
    joint = {"v7": 0, "v8": 0}
    for index, vector in enumerate(patterns.values()):
        rng = np.random.default_rng(np.random.SeedSequence([v8.DEV_SEED, 7, index]))
        clean = v7.received_buffer(vector, cfo_cycles_per_symbol=cfo, clock_error_ppm=ppm, timing_offset_samples=delay)
        buffers = clean[None, :] + rng.normal(0.0, sigma, (FRAMES_PER_VECTOR, clean.size))
        d7, p7, f7, e7 = v7.blind_joint_search(buffers)
        d8, p8, f8, e8 = v8.v8_blind_search(buffers)
        choices.update(zip(d7.tolist(), p7.tolist(), f7.tolist(), strict=True))
        for key, est in (("oracle", v7._candidate_decode(buffers, *truth)), ("v7", e7), ("v8", e8)):
            passes[key] += int(np.count_nonzero(np.max(np.abs(est - vector), axis=1) <= v7.TOLERANCE))
        for key, (d, p, f) in (("v7", (d7, p7, f7)), ("v8", (d8, p8, f8))):
            joint[key] += int(np.count_nonzero((np.abs(d - delay) <= 2) & (np.abs(p - ppm) <= 2500)
                                               & (np.abs(f - cfo) <= 0.005)))
    total = FRAMES_PER_VECTOR * len(patterns)
    print(f"\nnoisy frames: {total}; v7 blind choices (timing, ppm, CFO): {choices.most_common(4)}")
    print(f"passes: {passes}; joint acquisition (v7 rule): {joint}")
    print("exact score ties among v7 choices are not the cause: chosen candidates differ in score by noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
