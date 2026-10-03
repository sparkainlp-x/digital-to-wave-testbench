"""v7 combined clock-drift and CFO waveform, recovery and integrity tests."""

import copy

import numpy as np
import pytest

import robustness_benchmark_v7_clock_drift_cfo as v7
from digital_to_wave import encode_wave
from robustness_benchmark_v2 import build_input_patterns

PATTERN = np.asarray(build_input_patterns()["mixed_signs"], dtype=float)
SMALL = {
    "trials_per_vector": 2,
    "master_seed": 31337,
    "cfo_levels": (0.0,),
    "clock_levels_ppm": (0.0,),
    "timing_offsets": (0.0,),
    "noise_levels": (0.0, 0.45),
    "chunk_frames": 1,
}


def test_clean_unimpaired_waveform_and_oracle_are_exact():
    signal = v7.received_buffer(PATTERN)
    np.testing.assert_allclose(signal[64:64 + 2048].reshape(32, 64), encode_wave(PATTERN), atol=3e-13, rtol=0)
    decoded = v7._candidate_decode(signal[None, :], 0.0, 0.0, 0.0)[0]
    np.testing.assert_allclose(decoded, PATTERN, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cfo_cycles_per_symbol": 0.005},
        {"clock_error_ppm": 2500},
        {"timing_offset_samples": 2},
    ],
)
def test_impairment_changes_received_samples(kwargs):
    assert not np.array_equal(v7.received_buffer(PATTERN), v7.received_buffer(PATTERN, **kwargs))


def test_blind_search_has_no_truth_parameter_and_is_deterministic():
    buffer = v7.received_buffer(PATTERN, cfo_cycles_per_symbol=0.005, clock_error_ppm=2500, timing_offset_samples=2)
    first = v7.blind_joint_search(buffer[None, :])
    second = v7.blind_joint_search(buffer[None, :])
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_mcnemar_handles_large_paired_counts_without_overflow():
    assert v7._mcnemar(600, 600) == pytest.approx(1.0)
    assert 0.0 <= v7._mcnemar(1199, 1) < 1.0


def test_reduced_run_is_deterministic_and_validates():
    first = v7.run_benchmark(**SMALL)
    second = v7.run_benchmark(**SMALL)
    v7.validate_result(first)
    assert first["contract"]["full_protocol_run"] is False
    assert first["results"] == second["results"]
    assert first["paired_comparisons"] == second["paired_comparisons"]
    assert len(first["results"]) == 42


def test_validator_rejects_tampering():
    result = copy.deepcopy(v7.run_benchmark(**SMALL))
    result["results"][0]["frames_passing"] += 1
    with pytest.raises(ValueError):
        v7.validate_result(result)
