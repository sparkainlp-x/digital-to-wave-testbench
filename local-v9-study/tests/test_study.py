import copy
import importlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
study = importlib.import_module("study")


def test_all_payload_families_are_finite_bounded_and_distinct_from_v2_panel():
    import robustness_benchmark_v2 as v2

    previous = [np.asarray(v, dtype=float) for v in v2.build_input_patterns().values()]
    for i, family in enumerate(study.load_config()["payload_families"]):
        payload = study.make_payload(family, np.random.default_rng(1000 + i))
        assert payload.shape == (32,)
        assert np.isfinite(payload).all()
        assert np.max(np.abs(payload)) <= 2.0
        assert not any(np.array_equal(payload, old) for old in previous)


def test_variable_channel_and_impairment_only_controls_are_finite():
    config = study.load_config()
    payload = study.make_payload("uniform_continuous", np.random.default_rng(71))
    for index, condition in enumerate(config["condition_design"]):
        buffer = study.make_received_buffer(payload, condition, np.random.default_rng(200 + index))
        assert buffer.shape == (study.v7.BUFFER_SAMPLES,)
        assert np.isfinite(buffer).all()
    clean = config["condition_design"][0]
    time_varying = next(c for c in config["condition_design"] if c["name"] == "time_varying_cfo_clock_only")
    clean_buffer = study.make_received_buffer(payload, clean, np.random.default_rng(5))
    varying_buffer = study.make_received_buffer(payload, time_varying, np.random.default_rng(5))
    assert not np.array_equal(clean_buffer, varying_buffer)


def test_decoder_only_changes_decoding_not_v7_parameter_search():
    payload = study.make_payload("balanced_sign_magnitudes", np.random.default_rng(8))
    condition = study.load_config()["condition_design"][1]
    buffer = study.make_received_buffer(payload, condition, np.random.default_rng(9))[None, :]
    truth = (condition["timing_samples"], condition["clock_ppm"], condition["cfo_cycles_per_symbol"])
    baseline_params, baseline_decoded = study.run_receiver("v7_blind", buffer, truth)
    decoder_params, decoder_decoded = study.run_receiver("decoder_only_ablation", buffer, truth)
    np.testing.assert_array_equal(decoder_params, baseline_params)
    assert decoder_decoded.shape == baseline_decoded.shape == (1, 32)
    score_params, score_decoded = study.run_receiver("score_only_ablation", buffer, truth)
    refine_params, refine_decoded = study.run_receiver("refinement_only_ablation", buffer, truth)
    for params, decoded in ((score_params, score_decoded), (refine_params, refine_decoded)):
        assert params.shape == (1, 3)
        assert decoded.shape == (1, 32)
        assert np.isfinite(decoded).all()


def test_cluster_bootstrap_is_deterministic_and_bounded():
    cluster_values = np.array([0.0, 0.25, 1.0, 0.5])
    one = study._cluster_bootstrap(cluster_values, 4321, 300)
    two = study._cluster_bootstrap(cluster_values, 4321, 300)
    assert one == two
    assert 0.0 <= one[0] <= one[1] <= 1.0
    assert study._cluster_bootstrap(np.array([0.25]), 7, 300) == (0.25, 0.25)


def test_reduced_study_is_paired_cluster_aware_and_deterministic():
    cfg = copy.deepcopy(study.load_config())
    cfg["payload_families"] = cfg["payload_families"][:2]
    cfg["independent_channel_seed_replicates_per_family"] = 2
    cfg["frames_per_family_channel_cluster"] = 2
    cfg["bootstrap_replicates"] = 120
    cfg["condition_design"] = [cfg["condition_design"][i] for i in (0, 4, 6, 7)]
    first = study.run_study(cfg)
    second = study.run_study(cfg)
    assert first["design"]["independent_family_x_channel_seed_clusters"] == 4
    assert first["design"]["frames_per_condition"] == 8
    assert len(first["condition_summaries"]) == 4 * len(study.METHODS)
    assert len(first["paired_comparisons"]) == 4 * (len(study.METHODS) * (len(study.METHODS) - 1) // 2)
    assert first["frame_results"] == second["frame_results"]
    def without_runtime(rows):
        return [{k: v for k, v in row.items() if not k.startswith("runtime_")} for row in rows]
    assert without_runtime(first["condition_summaries"]) == without_runtime(second["condition_summaries"])
    assert first["paired_comparisons"] == second["paired_comparisons"]
    for condition in cfg["condition_design"]:
        frame_rows = [r for r in first["frame_results"] if r["condition"] == condition["name"]]
        hashes = {(r["cluster_id"], r["buffer_sha256"]) for r in frame_rows}
        assert len(hashes) == first["design"]["independent_family_x_channel_seed_clusters"]
        for cluster_id, buffer_hash in hashes:
            paired = [r for r in frame_rows if r["cluster_id"] == cluster_id]
            assert len({r["buffer_sha256"] for r in paired}) == 1
            assert {r["method"] for r in paired} == set(study.METHODS)
            assert all(r["buffer_sha256"] == buffer_hash for r in paired)
        rows = [r for r in first["condition_summaries"] if r["condition"] == condition["name"]]
        assert all(r["clusters"] == 4 and r["frames"] == 8 for r in rows)
        assert all(0 <= r["cluster_bootstrap95_low"] <= r["cluster_bootstrap95_high"] <= 1 for r in rows)


def test_frozen_lock_matches_files():
    lock = study.verify_lock()
    assert lock["lock_status"] == "frozen_before_full_run"
    for relpath, expected in lock["source_sha256"].items():
        assert study.sha256_file(HERE / relpath) == expected
