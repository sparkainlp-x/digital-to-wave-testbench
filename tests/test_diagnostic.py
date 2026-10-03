import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import timing_phase_diagnostic as diag  # noqa: E402


def test_phase_rotation_explains_most_of_the_clean_gain():
    rows = {r["timing_offset_samples"]: r for r in diag.diagnostic_rows((-8, -4, -1, 0, 1, 4, 8, 16))}
    assert rows[0]["measured_own_symbol_gain"] == 1.0
    # +-4 samples = 90 degree rotation: own-symbol gain ~ 0, clean MAE ~ zero-output floor
    assert abs(rows[4]["measured_own_symbol_gain"]) < 0.02
    # +-8 samples = 180 degrees: sign inversion
    assert rows[8]["measured_own_symbol_gain"] < -0.85
    # a full carrier cycle (16 samples) restores most of the gain despite a larger window error
    assert rows[16]["measured_own_symbol_gain"] > 0.75
    for d in (-8, -4, -1, 1, 4, 8):
        predicted = (64 - abs(d)) / 64 * math.cos(math.radians(22.5 * d))
        assert abs(rows[d]["measured_own_symbol_gain"] - predicted) < 0.03
