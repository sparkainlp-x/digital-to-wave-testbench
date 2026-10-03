"""v5 integer-sample timing benchmark: statistics, validator, determinism, golden counts."""

import math

import numpy as np
import pytest

import robustness_benchmark_v5_timing as v5
from robustness_benchmark_v2 import wilson_interval

NOMINAL, ORACLE = v5.METHODS
POOLED = v5.POOLED_PATTERN


def _pooled(result, offset, sigma, method):
    rows = [
        r for r in result["results"]
        if r["timing_offset_samples"] == offset and r["noise_std"] == sigma
        and r["method"] == method and r["vector_pattern"] == POOLED
    ]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize(
    "successes,trials,lower,upper",
    [
        (3256, 12000, 0.26345183734006344, 0.2793611847390233),
        (3231, 12000, 0.26138844047815246, 0.27725924834728166),
        (11382, 12000, 0.9444001076145897, 0.9523128352317497),
        (0, 12000, 0.0, 0.0003200191233674895),
    ],
)
def test_wilson_interval_values(successes, trials, lower, upper):
    lo, hi = wilson_interval(successes, trials)
    assert lo == pytest.approx(lower, abs=1e-12)
    assert hi == pytest.approx(upper, abs=1e-12)
    # independent textbook formula
    z = 1.959963984540054
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    assert lo == pytest.approx(max(0.0, centre - half), abs=1e-12)
    assert hi == pytest.approx(min(1.0, centre + half), abs=1e-12)


def test_analytic_oracle_pass_probability_is_94_78_percent():
    p = v5.theoretical_frame_pass_probability(np.zeros(32), 0.45)
    assert p == pytest.approx(0.9477992034686994, abs=1e-12)
    assert round(100 * p, 2) == 94.78
    # closed form: each of 32 components ~ N(0, 0.45/sqrt(32)) must lie within +-0.25
    sd = 0.45 / math.sqrt(32)
    component = math.erf(0.25 / (sd * math.sqrt(2)))
    assert p == pytest.approx(component**32, rel=1e-12)


def test_window_indices_shift_only_for_nominal():
    nominal = v5.window_indices(3, method=NOMINAL)
    oracle = v5.window_indices(3, method=ORACLE)
    assert nominal[0, 0] == 64 + 3 and oracle[0, 0] == 64
    with pytest.raises(ValueError):
        v5.window_indices(65, method=NOMINAL)


def test_small_run_validates_and_is_deterministic():
    kwargs = dict(trials_per_vector_per_condition=20, timing_offsets=(-1, 0, 1), chunk_frames=7)
    first = v5.run_timing_benchmark(**kwargs)
    second = v5.run_timing_benchmark(**kwargs)
    v5.validate_timing_result(first)
    assert first["contract"]["full_protocol_run"] is False
    assert first["results"] == second["results"]


def test_validator_rejects_tampering():
    result = v5.run_timing_benchmark(trials_per_vector_per_condition=5, timing_offsets=(0,))
    result["results"][0]["frames_passing"] += 1
    with pytest.raises(ValueError):
        v5.validate_timing_result(result)


def test_seed_must_differ_from_earlier_versions():
    from robustness_benchmark_v4 import MASTER_SEED as V4

    with pytest.raises(ValueError):
        v5.run_timing_benchmark(trials_per_vector_per_condition=2, master_seed=V4)


@pytest.fixture(scope="module")
def full_v5():
    result = v5.run_timing_benchmark()
    v5.validate_timing_result(result)
    return result


# Pooled noisy (sigma=0.45) counts out of 12,000 frames, as REPORTED in the v5 report and reproduced here.
GOLDEN_NOMINAL = {-8: 0, -4: 0, -2: 0, -1: 3256, 0: 11382, 1: 3231, 2: 0, 4: 0, 8: 0}
GOLDEN_ORACLE = {-8: 11406, -4: 11389, -2: 11389, -1: 11395, 0: 11382, 1: 11386, 2: 11357, 4: 11397, 8: 11346}


def test_full_protocol_golden_counts(full_v5):
    assert full_v5["contract"]["full_protocol_run"] is True
    for offset in v5.TIMING_OFFSETS_SAMPLES:
        nominal = _pooled(full_v5, offset, 0.45, NOMINAL)
        oracle = _pooled(full_v5, offset, 0.45, ORACLE)
        assert nominal["frames"] == oracle["frames"] == 12000
        assert nominal["frames_passing"] == GOLDEN_NOMINAL[offset]
        assert oracle["frames_passing"] == GOLDEN_ORACLE[offset]
        assert oracle["theory_frame_pass_probability"] == pytest.approx(0.9477992034686994, abs=1e-12)


def test_full_protocol_clean_rows(full_v5):
    expected_mae = {-8: 1.8684, -4: 0.9661, -2: 0.2947, -1: 0.0811, 0: 0.0, 1: 0.0815, 2: 0.2953, 4: 0.9653, 8: 1.8620}
    for offset, mae in expected_mae.items():
        clean = _pooled(full_v5, offset, 0.0, NOMINAL)
        assert clean["empirical_mae"] == pytest.approx(mae, abs=5e-5)
        assert clean["frames_passing"] == (6 if abs(offset) <= 1 else 0)
        assert _pooled(full_v5, offset, 0.0, ORACLE)["frames_passing"] == 6


def test_zero_offset_methods_are_identical(full_v5):
    a = _pooled(full_v5, 0, 0.45, NOMINAL)
    b = _pooled(full_v5, 0, 0.45, ORACLE)
    assert a["frames_passing"] == b["frames_passing"]
    assert a["empirical_mae"] == b["empirical_mae"]
