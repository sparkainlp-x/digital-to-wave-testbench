"""v6 blind timing recovery: closed forms, waveform equivalence, search, statistics, validator, determinism."""

import copy
import math

import numpy as np
import pytest

import robustness_benchmark_v6_timing_recovery as v6
from digital_to_wave import decode_wave, encode_wave
from robustness_benchmark_v2 import build_input_patterns
from robustness_benchmark_v5_timing import MASTER_SEED as V5_SEED
from robustness_benchmark_v5_timing import window_indices

PATTERNS = {k: np.asarray(v, dtype=float) for k, v in build_input_patterns().items()}
OMEGA = 2 * math.pi * 4 / 64  # 22.5 degrees per sample
H = math.sin(math.pi / 4) * math.sin(math.pi / 4 - OMEGA) / 32  # one-sample edge term

# Tests use their own seed so that no test ever draws from the locked v6 master seed stream.
TEST_SEED = 4242
SMALL = dict(master_seed=TEST_SEED, trials_per_vector_per_condition=12, random_frames_per_offset=40,
             integer_offsets=(-8, -1, 0, 1, 8), fractional_offsets=(0.0, 0.5, 1.0, 1.5),
             clean_sweep_offsets=(-16, -8, 0, 8, 16), chunk_frames=7)


@pytest.fixture(scope="module")
def small():
    return v6.run_timing_recovery_benchmark(**SMALL)


def _clean_nominal(vector, offset):
    return v6.decode_at(v6.integer_offset_stream(vector[None, :], offset), 0)[0]


def _neighbour(vector, direction):
    """v_{k+1} (direction=+1) or v_{k-1} (direction=-1), zero beyond the frame (zero guards)."""
    out = np.zeros_like(vector)
    if direction > 0:
        out[:-1] = vector[1:]
    else:
        out[1:] = vector[:-1]
    return out


# --- waveform construction -------------------------------------------------------------------------------

def test_encode_frames_is_bitwise_identical_to_locked_encoder():
    rng = np.random.default_rng(1)
    values = np.vstack([np.stack(list(PATTERNS.values())), rng.uniform(-2, 2, (5, 32)),
                        rng.integers(-2, 3, (5, 32)).astype(float)])
    for row, frame in zip(values, v6.encode_frames(values), strict=True):
        assert np.array_equal(frame, encode_wave(row).reshape(-1))


def test_integer_stream_matches_v5_nominal_windows():
    for name, vector in PATTERNS.items():
        stream = np.pad(encode_wave(vector).reshape(-1), (64, 64))
        for offset in (-8, -4, -1, 0, 1, 4, 8):
            v5_windows = stream[window_indices(offset, method="existing_nominal_decoder")]
            v6_estimate = _clean_nominal(vector, offset)
            assert np.array_equal(decode_wave(v5_windows), v6_estimate), (name, offset)


@pytest.mark.parametrize("offset", range(-16, 17))
def test_fractional_sampling_equals_integer_sampling_at_integer_offsets(offset):
    for vector in PATTERNS.values():
        frac = v6.fractional_offset_stream(vector, float(offset))
        integer = v6.integer_offset_stream(vector[None, :], offset)
        np.testing.assert_allclose(frac, integer, rtol=0, atol=1e-12)


def test_fractional_half_sample_late_has_no_intersymbol_leakage():
    # sampling at n + 0.5 keeps all 64 samples inside their own symbol: pure 11.25 degree rotation
    vector = PATTERNS["mixed_signs"]
    estimate = v6.decode_at(v6.fractional_offset_stream(vector, 0.5), 0)[0]
    np.testing.assert_allclose(estimate, vector * math.cos(OMEGA / 2), atol=1e-12)


# --- closed-form clean biases ----------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [1, -1])
def test_closed_form_one_sample_bias(offset):
    # one sample late/early: own gain cos(22.5 deg) - h, leakage h from the next/previous symbol
    for vector in PATTERNS.values():
        expected = vector * (math.cos(OMEGA) - H) + _neighbour(vector, offset) * H
        np.testing.assert_allclose(_clean_nominal(vector, offset), expected, atol=1e-12)
    assert math.cos(OMEGA) - 1 - H == pytest.approx(-0.084577, abs=1e-6)  # own-symbol bias per unit value


@pytest.mark.parametrize("offset", [8, -8])
def test_half_carrier_cycle_inverts_sign(offset):
    # 8 samples = 180 degrees: decode = -0.875 v_k - 0.125 v_neighbour
    for vector in PATTERNS.values():
        estimate = _clean_nominal(vector, offset)
        np.testing.assert_allclose(estimate, -0.875 * vector - 0.125 * _neighbour(vector, np.sign(offset)), atol=1e-12)
    mixed = PATTERNS["mixed_signs"]
    est = _clean_nominal(mixed, offset)
    nonzero = mixed != 0
    assert np.all(np.sign(est[nonzero]) == -np.sign(mixed[nonzero]))


@pytest.mark.parametrize("offset", [16, -16])
def test_full_carrier_cycle_is_nearly_periodic(offset):
    # 16 samples = 360 degrees: phase restored; decode = 0.75 v_k + 0.25 v_neighbour (window loss only)
    maes = {}
    for vector in PATTERNS.values():
        estimate = _clean_nominal(vector, offset)
        np.testing.assert_allclose(estimate, 0.75 * vector + 0.25 * _neighbour(vector, np.sign(offset)), atol=1e-12)
    for d in (4, 8, 16):
        maes[d] = np.mean([np.mean(np.abs(_clean_nominal(v, int(np.sign(offset)) * d) - v)) for v in PATTERNS.values()])
    assert maes[16] < 0.3 < maes[4] < maes[8]


# --- blind search ----------------------------------------------------------------------------------------

def test_matched_filter_search_finds_exact_offset_on_clean_frames():
    rng = np.random.default_rng(7)
    values = np.vstack([np.stack(list(PATTERNS.values())), rng.uniform(-2, 2, (20, 32)),
                        rng.integers(-2, 3, (20, 32)).astype(float)])
    for offset in v6.SEARCH_CANDIDATES:
        chosen, estimates = v6.matched_filter_energy_search(v6.integer_offset_stream(values, offset))
        assert np.all(chosen == -offset), offset
        np.testing.assert_allclose(estimates, values, atol=1e-12)


def test_search_is_never_told_the_offset():
    # identical buffers give identical choices whatever offset produced them; the API takes only the buffer
    vector = PATTERNS["randomized_1"][None, :]
    buf = v6.integer_offset_stream(vector, 3)
    assert v6.matched_filter_energy_search(buf)[0][0] == -3
    assert v6.raw_energy_search(buf)[0] == -3


def test_tie_break_prefers_nominal_start():
    assert list(v6._tie_break_order(v6.SEARCH_CANDIDATES)[:5]) == [0, -1, 1, -2, 2]
    silent = np.zeros((1, v6.STREAM_SAMPLES))
    assert v6.raw_energy_search(silent)[0] == 0
    assert v6.matched_filter_energy_search(silent)[0][0] == 0


# --- statistics ------------------------------------------------------------------------------------------

def test_mcnemar_exact_values():
    assert v6.mcnemar_exact_p(0, 0) == 1.0
    assert v6.mcnemar_exact_p(5, 0) == pytest.approx(2 / 32)
    assert v6.mcnemar_exact_p(0, 5) == pytest.approx(2 / 32)
    # 10 vs 2 discordant: 2 * P(X <= 2), X ~ Bin(12, 0.5) = 2 * 79 / 4096
    assert v6.mcnemar_exact_p(10, 2) == pytest.approx(2 * 79 / 4096)
    assert v6.mcnemar_exact_p(3, 3) == 1.0


def test_round_rule_oracle_closed_form():
    sd = 0.45 / math.sqrt(32)
    assert v6.oracle_round_rule_probability() == pytest.approx(math.erf(0.5 / (sd * math.sqrt(2))) ** 32, rel=1e-12)
    assert v6.oracle_round_rule_probability() > 0.99999


# --- runner, validator, determinism ----------------------------------------------------------------------

def _pooled(result, panel, offset, method, sigma=0.45):
    rows = [r for r in v6.pooled_rows(result, panel, method, sigma=sigma) if r["offset_samples"] == offset]
    assert len(rows) == 1
    return rows[0]


def test_small_run_validates_and_is_not_full(small):
    v6.validate_timing_recovery_result(small)
    c = small["contract"]
    assert c["full_protocol_run"] is False
    assert c["evidence_label"] == "SYNTHETIC" and c["physical_validity"] == "no physical validity claimed"
    assert set(v6.LOCKED_MODULES) < set(c["module_sha256"])
    assert c["master_seed"] == TEST_SEED
    assert v6.MASTER_SEED not in (20261003, 20261004, 20261005, 20261006, 20261007)


def test_nominal_equals_oracle_at_zero_offset(small):
    for sigma in (0.0, 0.45):
        a = _pooled(small, v6.PANEL_INTEGER, 0.0, v6.NOMINAL, sigma)
        b = _pooled(small, v6.PANEL_INTEGER, 0.0, v6.ORACLE, sigma)
        assert a["frames_passing"] == b["frames_passing"] and a["empirical_mae"] == b["empirical_mae"]


def test_clean_rows_search_matches_oracle(small):
    for offset in SMALL["integer_offsets"]:
        mf = _pooled(small, v6.PANEL_INTEGER, float(offset), v6.MF_SEARCH, 0.0)
        assert mf["frames_passing"] == 6 and mf["timing_acquired_frames"] == 6 and mf["empirical_mae"] < 1e-12


def test_determinism(small):
    again = v6.run_timing_recovery_benchmark(**SMALL)
    for key in ("results", "paired_comparisons", "clean_sweep", "fractional_analytic_curve",
                "search_offset_error_histograms"):
        assert again[key] == small[key]


# Golden pooled counts for the reduced configuration above (test seed, sigma = 0.45, 72 frames per offset).
GOLDEN_SMALL = {
    v6.NOMINAL: [0, 19, 66, 22, 0],
    v6.ORACLE: [70, 65, 66, 72, 62],
    v6.MF_SEARCH: [70, 65, 66, 72, 62],
    v6.ENERGY_SEARCH: [33, 27, 29, 35, 26],
    "mf_acquired": [72, 72, 72, 72, 72],
}


def test_reduced_run_golden_counts(small):
    observed = {
        m: [_pooled(small, v6.PANEL_INTEGER, float(o), m)["frames_passing"] for o in SMALL["integer_offsets"]]
        for m in v6.INTEGER_METHODS
    }
    observed["mf_acquired"] = [_pooled(small, v6.PANEL_INTEGER, float(o), v6.MF_SEARCH)["timing_acquired_frames"]
                               for o in SMALL["integer_offsets"]]
    assert observed == GOLDEN_SMALL


@pytest.mark.parametrize("tamper", [
    "pass_count", "discordant", "label", "physical", "seed", "drop_row", "wilson", "acquisition", "full_flag",
    "vector",
])
def test_validator_rejects_tampering(small, tamper):
    bad = copy.deepcopy(small)
    noisy = next(r for r in bad["results"] if r["noise_std"] == 0.45 and r["method"] == v6.MF_SEARCH)
    if tamper == "pass_count":
        noisy["frames_passing"] += 1
    elif tamper == "discordant":
        bad["paired_comparisons"][0]["a_pass_b_fail"] += 1
    elif tamper == "label":
        noisy["evidence_label"] = "VALIDATED"
    elif tamper == "physical":
        bad["contract"]["physical_validity"] = "physically validated"
    elif tamper == "seed":
        bad["contract"]["master_seed"] = V5_SEED
    elif tamper == "drop_row":
        bad["results"].pop()
    elif tamper == "wilson":
        noisy["wilson95_lower"] = (noisy["wilson95_lower"] or 0.0) + 0.05
    elif tamper == "acquisition":
        noisy["timing_acquired_frames"] = noisy["frames"] + 1
    elif tamper == "full_flag":
        bad["contract"]["full_protocol_run"] = True
    elif tamper == "vector":
        bad["contract"]["vector_patterns"]["mixed_signs"][0] = 0.123
    with pytest.raises(ValueError):
        v6.validate_timing_recovery_result(bad)


def test_seed_must_differ_from_earlier_versions():
    for seed in (V5_SEED, 20261004):
        with pytest.raises(ValueError):
            v6.run_timing_recovery_benchmark(trials_per_vector_per_condition=1, random_frames_per_offset=1,
                                             master_seed=seed, integer_offsets=(0,), fractional_offsets=(0.0,),
                                             clean_sweep_offsets=(0,))
