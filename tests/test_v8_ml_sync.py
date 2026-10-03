"""v8 GLRT + coarse-to-fine blind synchronization: diagnosis, receiver, statistics, validator, determinism."""

import copy
import inspect

import numpy as np
import pytest

import robustness_benchmark_v7_clock_drift_cfo as v7
import robustness_benchmark_v8_ml_sync as v8
from robustness_benchmark_v2 import build_input_patterns

PATTERNS = {k: np.asarray(v, dtype=float) for k, v in build_input_patterns().items()}
MIXED = PATTERNS["mixed_signs"]
# Tests use their own seed: never the locked v8 master seed nor the development seed.
SMALL = {
    "trials_per_vector": 4,
    "master_seed": v8.TEST_SEED,
    "cfo_levels": (0.0, 0.005),
    "clock_levels_ppm": (-2500.0, 1250.0),
    "timing_offsets": (2.0,),
    "noise_levels": (0.0, 0.45),
    "chunk_frames": 3,
    "workers": 1,
}


@pytest.fixture(scope="module")
def small():
    return v8.run_benchmark(**SMALL)


def test_seeds_are_distinct_and_fresh():
    assert v8.MASTER_SEED not in v8.PRIOR_SEEDS and v8.DEV_SEED not in v8.PRIOR_SEEDS
    assert len({v8.MASTER_SEED, v8.DEV_SEED, v8.TEST_SEED, v7.MASTER_SEED}) == 4


def test_ls_decode_and_glrt_are_exact_on_clean_truth():
    for kwargs in ({}, {"cfo_cycles_per_symbol": 0.005, "clock_error_ppm": 1250, "timing_offset_samples": 2}):
        signal = v7.received_buffer(MIXED, **kwargs)[None, :]
        truth = (kwargs.get("timing_offset_samples", 0.0), kwargs.get("clock_error_ppm", 0.0),
                 kwargs.get("cfo_cycles_per_symbol", 0.0))
        np.testing.assert_allclose(v8.ls_decode(signal, *truth)[0], MIXED, atol=1e-12)
        # GLRT score at truth equals the full signal energy (zero least-squares residual).
        assert v8.ml_score(signal, *truth)[0] == pytest.approx(float(np.sum(signal * signal)), rel=1e-12)


def test_carrier_offset_round_trip():
    ppm = np.array([-5000.0, -1250.0, 0.0, 2500.0])
    cfo = np.array([0.01, -0.005, 0.0, 0.0025])
    np.testing.assert_allclose(v8.cfo_from_carrier(ppm, v8.carrier_offset(ppm, cfo)), cfo, atol=1e-15)
    # 2500 ppm of clock error is carrier-equivalent to about 0.01 cycles/symbol of CFO: the v7 ridge.
    assert v8.carrier_offset(-2500.0, 0.0) == pytest.approx(-0.01, abs=1e-12)


def test_diagnosis_v7_metric_prefers_ridge_alternative_but_glrt_prefers_truth():
    """Clean weak cell (CFO 0, -2500 ppm, +2 samples): v7's metric does not rank the truth first."""
    truth, ridge = (2.0, -2500.0, 0.0), (2.0, 0.0, -0.01)
    v7_wins = glrt_wins = 0
    for vector in PATTERNS.values():
        buffer = v7.received_buffer(vector, clock_error_ppm=-2500, timing_offset_samples=2)[None, :]
        v7_truth = float(np.sum(v7._candidate_decode(buffer, *truth) ** 2))
        v7_ridge = float(np.sum(v7._candidate_decode(buffer, *ridge) ** 2))
        v7_wins += v7_ridge > v7_truth
        assert v8.ml_score(buffer, *truth)[0] > v8.ml_score(buffer, *ridge)[0]
        glrt_wins += 1
    assert v7_wins >= 3 and glrt_wins == len(PATTERNS)


@pytest.mark.parametrize("cfo,ppm,delay", [(0.0, -2500.0, 2.0), (0.005, 1250.0, 2.0), (-0.01, 5000.0, -2.0),
                                           (0.0, 0.0, 0.0), (0.01, -1250.0, 0.0)])
def test_clean_blind_search_recovers_values(cfo, ppm, delay):
    buffers = np.stack([v7.received_buffer(v, cfo_cycles_per_symbol=cfo, clock_error_ppm=ppm,
                                           timing_offset_samples=delay) for v in PATTERNS.values()])
    d, p, f, estimates = v8.v8_blind_search(buffers)
    # Finite refinement resolution (156.25 ppm, 0.0003 cycles/symbol) leaves a small clean error.
    assert np.max(np.abs(estimates - np.stack(list(PATTERNS.values())))) < 0.05
    assert np.all(np.abs(d - delay) <= 1) and np.all(np.abs(p - ppm) <= 1250)
    assert np.all(np.abs(v8.carrier_offset(p, f) - v8.carrier_offset(ppm, cfo)) < 1e-3)


def test_blind_search_takes_only_the_buffer_and_is_deterministic():
    assert list(inspect.signature(v8.v8_blind_search).parameters) == ["buffer"]
    rng = np.random.default_rng(v8.TEST_SEED)
    buffer = v7.received_buffer(MIXED, cfo_cycles_per_symbol=0.005, clock_error_ppm=1250, timing_offset_samples=2)
    noisy = buffer[None, :] + rng.normal(0, 0.45, (6, buffer.size))
    first, second = v8.v8_blind_search(noisy), v8.v8_blind_search(noisy)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    # Frames are processed independently: searching one frame alone gives the same answer.
    alone = v8.v8_blind_search(noisy[2:3])
    for batch, single in zip(first, alone, strict=True):
        np.testing.assert_array_equal(batch[2:3], single)


def test_blind_search_stays_inside_envelope():
    rng = np.random.default_rng(v8.TEST_SEED + 1)
    d, p, f, _ = v8.v8_blind_search(rng.normal(0, 1.0, (8, v8.BUFFER_SAMPLES)))
    assert np.all(np.abs(d) <= v8.TIMING_BOUND) and np.all(np.abs(p) <= v8.CLOCK_BOUND_PPM + 1e-6)
    assert np.all(np.abs(f) <= v8.CFO_BOUND + 1e-12)


def test_reduced_run_is_deterministic_chunk_independent_and_validates(small):
    v8.validate_result(small)
    assert small["contract"]["full_protocol_run"] is False
    assert small["contract"]["master_seed"] == v8.TEST_SEED
    same = v8.run_benchmark(**SMALL)
    assert same["results"] == small["results"] and same["paired_comparisons"] == small["paired_comparisons"]
    # Other chunk sizes and worker counts: identical counts and p-values; error statistics agree to rounding
    # (the locked v7 decoders' floating-point sums depend on the batch shape at the 1e-16 level).
    again = v8.run_benchmark(**{**SMALL, "chunk_frames": 4, "workers": 2})
    assert again["paired_comparisons"] == small["paired_comparisons"]
    for left, right in zip(again["results"], small["results"], strict=True):
        for key, value in left.items():
            if isinstance(value, float) and key.startswith(("empirical", "mean_", "max_abs")):
                assert value == pytest.approx(right[key], rel=1e-12, abs=1e-15)
            else:
                assert value == right[key], key
    cells, noise = 2 * 2 * 1, 2
    assert len(small["results"]) == cells * noise * (len(PATTERNS) + 1) * len(v8.METHODS)
    assert len(small["paired_comparisons"]) == cells * noise * (len(PATTERNS) + 1) * len(v8.PAIRS)


def test_reduced_run_clean_controls_are_exact_for_informed_ls_and_v8(small):
    clean = [r for r in small["results"] if r["noise_std"] == 0.0 and r["vector_pattern"] == v8.POOLED]
    for row in clean:
        if row["method"] in (v8.ORACLE_LS, v8.V8_BLIND):
            assert row["frames_passing"] == row["frames"] == len(PATTERNS)
            assert row["max_abs_error"] < 0.05


def test_reduced_run_acquisition_fields(small):
    for row in small["results"]:
        if row["method"] in v8.BLIND_METHODS:
            assert row["joint_acq_strict_frames"] <= row["joint_acq_loose_frames"] <= row["frames"]
        else:
            assert row["joint_acq_loose_frames"] is None


@pytest.mark.parametrize("seed", sorted(v8.PRIOR_SEEDS | {v8.DEV_SEED}))
def test_prior_and_development_seeds_are_rejected(seed):
    with pytest.raises(ValueError):
        v8.run_benchmark(**{**SMALL, "master_seed": seed})


def test_impairments_outside_envelope_are_rejected():
    with pytest.raises(ValueError):
        v8.run_benchmark(**{**SMALL, "clock_levels_ppm": (7500.0,)})
    with pytest.raises(ValueError):
        v8.run_benchmark(**{**SMALL, "cfo_levels": (0.02,)})


def test_protocol_cells_cover_v7_grid_and_symmetric_extension():
    cells = v8.protocol_cells()
    assert len(cells) == 5 * 7 * 3 * 3 and len(set(cells)) == len(cells)
    for f in v7.CFO_LEVELS:
        for p in v7.CLOCK_LEVELS_PPM:
            for d in v7.TIMING_OFFSETS:
                for s in v7.NOISE_LEVELS:
                    assert (f, p, d, s) in cells
    assert v8.WEAK_CELL in cells
    for grid in (v8.CFO_LEVELS, v8.CLOCK_LEVELS_PPM, v8.TIMING_OFFSETS):
        assert sorted(grid) == sorted(-x for x in grid)


@pytest.mark.parametrize("tamper", ["pass_count", "mcnemar", "discordant", "label", "hash", "seed", "acq", "wilson"])
def test_validator_rejects_tampering(small, tamper):
    bad = copy.deepcopy(small)
    noisy_row = next(r for r in bad["results"] if r["noise_std"] > 0 and r["method"] == v8.V8_BLIND)
    if tamper == "pass_count":
        noisy_row["frames_passing"] = (noisy_row["frames_passing"] + 1) % (noisy_row["frames"] + 1)
    elif tamper == "mcnemar":
        bad["paired_comparisons"][0]["mcnemar_exact_two_sided_p"] = 0.0123
    elif tamper == "discordant":
        bad["paired_comparisons"][-1]["a_pass_b_fail"] += 1
    elif tamper == "label":
        bad["contract"]["classification"] = "physical validation"
    elif tamper == "hash":
        bad["contract"]["module_sha256"]["robustness_benchmark_v8_ml_sync.py"] = "0" * 64
    elif tamper == "seed":
        bad["contract"]["master_seed"] = v7.MASTER_SEED
    elif tamper == "acq":
        noisy_row["joint_acq_strict_frames"] = noisy_row["frames"] + 1
    elif tamper == "wilson":
        noisy_row["wilson95_lower"] = 0.0 if noisy_row["wilson95_lower"] else 0.5
    with pytest.raises(ValueError):
        v8.validate_result(bad)


def test_write_outputs_round_trip(small, tmp_path):
    paths = v8.write_outputs(small, tmp_path)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    header = paths["csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == v8.RESULT_FIELDS


def test_summary_criteria_are_reported(small):
    criteria = small["summary"]["success_criteria"]
    assert criteria["evaluated_on_full_protocol_grid"] is False
    for key in ("S2_no_v7_grid_regression", "S3_not_worse_than_v7_oracle_on_v7_grid", "S4_extension_gain"):
        assert isinstance(criteria[key]["pass"], bool)


# --- committed full-protocol results (artifacts/robustness-benchmark-v8-ml-sync.*) -------------------------

ARTIFACTS = v8.ARTIFACTS
COMMITTED_JSON = ARTIFACTS / f"{v8.STEM}.json"


@pytest.fixture(scope="module")
def committed():
    import json

    return json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))


def test_committed_results_validate_and_come_from_the_lock_commit(committed):
    v8.validate_result(committed)
    contract = committed["contract"]
    assert contract["full_protocol_run"] is True and contract["master_seed"] == v8.MASTER_SEED == 20261011
    assert contract["git_tracked_tree_clean_at_run_start"] is True
    assert contract["git_commit"] == "ba0143e55f99b3a508082f81f9174cf9dec54f02"
    assert "no physical validity claimed" in contract["classification"]
    assert len(committed["results"]) == len(v8.protocol_cells()) * (len(PATTERNS) + 1) * len(v8.METHODS)


def test_committed_summary_is_recomputed_from_rows(committed):
    recomputed = v8.summarize(committed["results"], committed["paired_comparisons"], True)
    assert recomputed == committed["summary"]


def test_committed_csv_files_match_json(committed, tmp_path):
    paths = v8.write_outputs(committed, tmp_path)
    for key in ("csv", "paired_csv"):
        assert paths[key].read_bytes() == (ARTIFACTS / paths[key].name).read_bytes()


def test_committed_headline_numbers(committed):
    weak = committed["summary"]["weak_cell"]
    assert weak["frames"] == 1200
    assert weak["frames_passing"] == {v8.NOMINAL: 0, v8.ORACLE: 1117, v8.ORACLE_LS: 1150, v8.V7_BLIND: 1034,
                                      v8.V8_BLIND: 1150}
    assert weak["joint_acq_loose_frames"] == {v8.V7_BLIND: 528, v8.V8_BLIND: 1199}
    pair = weak["paired"][f"{v8.V8_BLIND}_vs_{v8.V7_BLIND}"]
    assert (pair["a_pass_b_fail"], pair["a_fail_b_pass"]) == (132, 16)
    assert pair["mcnemar_exact_two_sided_p"] < 1e-20
    criteria = committed["summary"]["success_criteria"]
    assert criteria["evaluated_on_full_protocol_grid"] is True
    for key in ("S1_weak_cell", "S2_no_v7_grid_regression", "S3_not_worse_than_v7_oracle_on_v7_grid",
                "S4_extension_gain"):
        assert criteria[key]["pass"] is True, key
    assert criteria["improvement_real"] is True
    assert criteria["all_cells_where_v8_significantly_worse_than_v7"] == []
    totals = committed["summary"]["by_noise"]["0.45"]["all_cells"]["totals"]
    assert {m: totals[m]["frames_passing"] for m in v8.METHODS} == {
        v8.NOMINAL: 4046, v8.ORACLE: 116102, v8.ORACLE_LS: 119310, v8.V7_BLIND: 109753, v8.V8_BLIND: 119273}


def test_committed_disclosed_shortfall_versus_raw_ls_oracle(committed):
    """v8 is slightly below the informed raw-LS oracle when pooled (small search loss); disclosed in the README."""
    pooled = committed["summary"]["by_noise"]["0.45"]["all_cells"]["paired"][f"{v8.V8_BLIND}_vs_{v8.ORACLE_LS}"]
    assert (pooled["a_pass_b_fail"], pooled["a_fail_b_pass"]) == (35, 72)
    assert committed["summary"]["success_criteria"]["reported_v8_vs_raw_ls_oracle_significant_cells"] == []


@pytest.mark.parametrize("cell", [v8.WEAK_CELL, (0.01, 1250.0, 2.0, 0.45)])
def test_full_size_protocol_cell_rerun_matches_committed_rows(committed, cell):
    """Byte-level spot check of the full run: one v7 cell and one extension cell, 1,200 frames each."""
    import csv
    import io

    rerun = v8.reproduce_protocol_cell(*cell)

    def lines(rows, fields):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writerows(rows)
        return buffer.getvalue().splitlines()

    committed_csv = set((ARTIFACTS / f"{v8.STEM}.csv").read_text(encoding="utf-8").splitlines())
    committed_paired = set((ARTIFACTS / f"{v8.STEM}-paired.csv").read_text(encoding="utf-8").splitlines())
    result_lines = lines(rerun["rows"], v8.RESULT_FIELDS)
    pair_lines = lines(rerun["paired"], v8.PAIR_FIELDS)
    assert len(result_lines) == (len(PATTERNS) + 1) * len(v8.METHODS)
    assert all(line in committed_csv for line in result_lines)
    assert all(line in committed_paired for line in pair_lines)
