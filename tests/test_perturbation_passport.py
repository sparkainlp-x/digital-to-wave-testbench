"""Reproducible v8 Perturbation Passport extraction and rendering."""

import copy
import hashlib
import json

import pytest

import perturbation_passport as passport_module

ROOT = passport_module.ROOT
SOURCE_PATH = ROOT / "artifacts" / "robustness-benchmark-v8-ml-sync.json"
JSON_OUTPUT = ROOT / "artifacts" / "perturbation-passport-v8-ml-sync.json"
MARKDOWN_OUTPUT = ROOT / "reports" / "perturbation-passport-v8-ml-sync.md"


@pytest.fixture(scope="module")
def source():
    payload = SOURCE_PATH.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def build(source):
    result, digest = source
    return passport_module.build_passport(
        result,
        source_name=SOURCE_PATH.name,
        source_sha256=digest,
    )


def test_passport_preserves_provenance_and_synthetic_scope(source):
    passport = build(source)
    assert passport["schema"] == "digital-to-wave-perturbation-passport/1.0"
    assert passport["evidence_label"] == "SYNTHETIC"
    assert passport["physical_validity_claimed"] is False
    assert passport["source"]["artifact_sha256"] == source[1]
    assert passport["source"]["master_seed"] == 20261011
    assert passport["source"]["full_protocol_run"] is True
    assert len(passport["source"]["passport_generator_sha256"]) == 64
    assert passport["experiment"]["declared_condition_cells"] == 105
    assert passport["experiment"]["vector_count"] == 6
    assert "Gaussian" in passport["experiment"]["noise_model"]
    assert passport["assumptions_and_limitations"]


def test_aggregate_outcomes_and_weak_cell_match_source(source):
    result, _ = source
    passport = build(source)
    outcome = next(row for row in passport["outcomes_by_noise"] if row["noise_std"] == 0.45)
    totals = result["summary"]["by_noise"]["0.45"]["all_cells"]["totals"]
    for method in (passport_module.V8, passport_module.V7, passport_module.RAW_LS_ORACLE):
        actual = outcome["receivers"][method]
        expected = totals[method]
        assert actual["frames_passing"] == expected["frames_passing"]
        assert actual["frames"] == expected["frames"]
        assert actual["pass_share"] == pytest.approx(expected["frames_passing"] / expected["frames"])
        assert actual["wilson95"] == expected["wilson95"]

    weak = passport["weak_cell"]
    assert weak["condition"] == {
        "cfo_cycles_per_symbol": 0.0,
        "clock_error_ppm": -2500.0,
        "timing_offset_samples": 2.0,
        "noise_std": 0.45,
    }
    assert weak["v8_frames_passing"] == 1150
    assert weak["v7_frames_passing"] == 1034
    assert weak["v8_joint_acq_loose_frames"] == 1199
    assert weak["v7_joint_acq_loose_frames"] == 528


def test_failure_boundary_is_discrete_and_derived_from_pooled_rows(source):
    result, _ = source
    passport = build(source)
    boundary = passport["observed_failure_boundary"]
    v8_rows = [
        row for row in result["results"]
        if row["vector_pattern"] == passport_module.POOLED
        and row["method"] == passport_module.V8
        and row["noise_std"] == 0.45
    ]
    expected = min(v8_rows, key=lambda row: row["frame_pass_share"])
    weakest = boundary["weakest_observed_cells"][0]
    assert boundary["cells_tested"] == 105
    assert boundary["cells_with_at_least_one_failed_frame"] == 105
    assert boundary["minimum_observed_pass_share"] == expected["frame_pass_share"]
    assert weakest["frames_passing"] == expected["frames_passing"] == 1115
    assert weakest["cfo_cycles_per_symbol"] == expected["cfo_cycles_per_symbol"] == 0.005
    assert weakest["clock_error_ppm"] == expected["clock_error_ppm"] == 2500.0
    assert "no interpolation or extrapolation" in boundary["definition"]
    assert "Untested" in boundary["outside_tested_grid"]


def test_passport_generation_is_deterministic_and_matches_committed_files(source, tmp_path):
    first = build(source)
    second = build(source)
    assert first == second
    assert passport_module.render_markdown(first) == passport_module.render_markdown(second)

    json_path, markdown_path = passport_module.write_passport(
        first,
        json_output=tmp_path / "passport.json",
        markdown_output=tmp_path / "passport.md",
    )
    assert json.loads(json_path.read_text(encoding="utf-8")) == first
    assert markdown_path.read_text(encoding="utf-8") == passport_module.render_markdown(first)
    assert JSON_OUTPUT.read_bytes() == json_path.read_bytes()
    assert MARKDOWN_OUTPUT.read_bytes() == markdown_path.read_bytes()


def test_cli_writes_both_formats(source, tmp_path):
    json_path = tmp_path / "cli" / "passport.json"
    markdown_path = tmp_path / "cli" / "passport.md"
    assert passport_module.main([
        "--input", str(SOURCE_PATH),
        "--json-output", str(json_path),
        "--markdown-output", str(markdown_path),
    ]) == 0
    assert json_path.is_file() and markdown_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["source"]["artifact_sha256"] == source[1]


def test_source_schema_and_aggregate_tampering_are_rejected(source):
    result, digest = source
    bad_benchmark = copy.deepcopy(result)
    bad_benchmark["benchmark"] = "physical_channel_validation"
    with pytest.raises(ValueError, match="expected benchmark"):
        passport_module.build_passport(bad_benchmark, source_name="bad.json", source_sha256=digest)

    bad_scope = copy.deepcopy(result)
    bad_scope["contract"]["classification"] = "physical validation"
    with pytest.raises(ValueError, match="synthetic/no-physical-validity"):
        passport_module.build_passport(bad_scope, source_name="bad.json", source_sha256=digest)

    bad_summary = copy.deepcopy(result)
    bad_summary["summary"]["by_noise"]["0.45"]["all_cells"]["totals"][passport_module.V8][
        "frames_passing"
    ] += 1
    with pytest.raises(ValueError, match="does not match result rows"):
        passport_module.build_passport(bad_summary, source_name="bad.json", source_sha256=digest)

    with pytest.raises(ValueError, match="64-character SHA-256"):
        passport_module.build_passport(result, source_name="bad.json", source_sha256="bad")
