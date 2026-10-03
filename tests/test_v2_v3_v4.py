"""Earlier benchmarks: validators, reported counts and the v3/v4 'tie' check."""

import pytest

import robustness_benchmark_v2 as v2
import robustness_benchmark_v3 as v3
import robustness_benchmark_v4 as v4


def test_seeds_are_distinct():
    assert len({v2.MASTER_SEED, v3.MASTER_SEED, v4.MASTER_SEED}) == 3


def test_v2_full_run_validates():
    v2.validate_result_integrity(v2.run_benchmark_v2())


@pytest.fixture(scope="module")
def v3_full():
    result = v3.run_gaussian_calibration()
    v3.validate_calibration_result(result)
    return result


@pytest.fixture(scope="module")
def v4_full():
    result = v4.run_cfo_benchmark()
    v4.validate_cfo_result(result)
    return result


def _rows(result, **match):
    return [r for r in result["results"] if all(r.get(k) == v for k, v in match.items())]


def test_v4_reported_counts(v4_full):
    def pooled(cfo):
        (row,) = _rows(v4_full, cfo_cycles_per_symbol=cfo, noise_std=0.45,
                       method="nominal_existing_decoder", vector_pattern="POOLED_ALL_VECTORS")
        return row["frames_passing"]

    assert pooled(0.0) == 11365
    assert pooled(-0.0025) == 6380  # -0.0625 % of the carrier
    assert pooled(0.0025) == 6278   # +0.0625 %
    assert pooled(0.005) == 0 and pooled(-0.005) == 0


# Per-vector passes at sigma=0.45 (2,000 frames each). The pooled totals tie at 11,365/12,000
# but the per-vector counts differ, because v3 and v4 draw from different seeds.
V3_PER_VECTOR = {"mixed_signs": 1907, "sparse_and_zeros": 1889, "boundary_values": 1875,
                 "randomized_1": 1894, "randomized_2": 1895, "randomized_3": 1905}
V4_PER_VECTOR = {"mixed_signs": 1909, "sparse_and_zeros": 1877, "boundary_values": 1893,
                 "randomized_1": 1880, "randomized_2": 1895, "randomized_3": 1911}


def test_v3_v4_pooled_tie_is_coincidental(v3_full, v4_full):
    v3_rows = {r["vector_pattern"]: r for r in _rows(v3_full, noise_std=0.45)}
    v4_rows = {r["vector_pattern"]: r for r in _rows(v4_full, noise_std=0.45, cfo_cycles_per_symbol=0.0,
                                                      method="nominal_existing_decoder")}
    assert v3_rows["POOLED_ALL_VECTORS"]["frames_passing"] == 11365
    assert v4_rows["POOLED_ALL_VECTORS"]["frames_passing"] == 11365
    assert {k: v3_rows[k]["frames_passing"] for k in V3_PER_VECTOR} == V3_PER_VECTOR
    assert {k: v4_rows[k]["frames_passing"] for k in V4_PER_VECTOR} == V4_PER_VECTOR
    assert abs(v3_rows["POOLED_ALL_VECTORS"]["empirical_mae"] - v4_rows["POOLED_ALL_VECTORS"]["empirical_mae"]) > 1e-5
