"""Analytical Gaussian calibration for the synthetic digital-to-wave codec.

This v3 run calibrates empirical matched-filter error metrics against closed-form
predictions under known-template IID Gaussian sample noise. It is not a physical
validation study, a generalized channel test, or a decoder comparison. The v2
benchmark and source encoder are imported but never modified.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from digital_to_wave import WaveConfig, decode_wave, encode_wave
from robustness_benchmark_v2 import (
    GAUSSIAN_STD_LEVELS as V2_GAUSSIAN_STD_LEVELS,
    MASTER_SEED as V2_MASTER_SEED,
    VECTOR_SEED,
    build_input_patterns,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TRIALS_PER_VECTOR_PER_LEVEL = 2_000
MASTER_SEED = 20261005
TOLERANCE = 0.25
VECTOR_LENGTH = 32
SAMPLES_PER_SYMBOL = 64
CYCLES_PER_SYMBOL = 4
AMPLITUDE_MIN = 1.0
CHUNK_FRAMES = 128
POOLED_PATTERN = "POOLED_ALL_VECTORS"
GAUSSIAN_STD_LEVELS = tuple(V2_GAUSSIAN_STD_LEVELS)
EXPECTED_PATTERN_NAMES = (
    "mixed_signs",
    "sparse_and_zeros",
    "boundary_values",
    "randomized_1",
    "randomized_2",
    "randomized_3",
)
CSV_FIELDS = (
    "noise_std", "vector_pattern", "frames", "components", "frames_passing",
    "frame_pass_share", "wilson95_lower", "wilson95_upper",
    "empirical_mae", "theory_mae", "abs_mae_gap",
    "empirical_rmse", "theory_rmse", "abs_rmse_gap",
    "empirical_mean_bias", "theory_mean_bias", "abs_mean_bias_gap",
    "empirical_variance", "theory_variance", "abs_variance_gap",
    "theory_frame_pass_probability", "abs_pass_probability_gap",
)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def q_template(config: WaveConfig = WaveConfig()) -> np.ndarray:
    """Return the decoder's exact known template, q[n]=sin(omega*n+pi/4)."""
    n = np.arange(config.samples_per_symbol, dtype=float)
    omega = 2.0 * np.pi * config.cycles_per_symbol / config.samples_per_symbol
    return np.sin(omega * n + np.pi / 4.0)


def theoretical_component_metrics(noise_std: float, q_energy: float = 32.0) -> dict[str, float]:
    """Closed-form signed-error moments and MAE for one projected component."""
    sigma = _finite_nonnegative(noise_std, "noise_std")
    energy = _finite_nonnegative(q_energy, "q_energy")
    if energy == 0.0:
        raise ValueError("q_energy must be positive")
    component_sd = sigma / math.sqrt(energy)
    return {
        "mae": math.sqrt(2.0 / math.pi) * component_sd,
        "rmse": component_sd,
        "mean_bias": 0.0,
        "variance": sigma * sigma / energy,
    }


def theoretical_frame_pass_probability(
    noise_std: float,
    tolerance: float = TOLERANCE,
    vector_length: int = VECTOR_LENGTH,
    q_energy: float = 32.0,
) -> float:
    """Probability that every independent component error is within tolerance."""
    sigma = _finite_nonnegative(noise_std, "noise_std")
    tau = _finite_nonnegative(tolerance, "tolerance")
    count = _positive_integer(vector_length, "vector_length")
    energy = _finite_nonnegative(q_energy, "q_energy")
    if energy == 0.0:
        raise ValueError("q_energy must be positive")
    if sigma == 0.0:
        return 1.0
    # For E~N(0,sigma^2/q_energy), P(|E|<=tau)=erf(tau/(sqrt(2)*SD)).
    per_component = math.erf(tau * math.sqrt(energy) / (math.sqrt(2.0) * sigma))
    return min(1.0, max(0.0, per_component)) ** count


def summarize_component_errors(errors: Any, tolerance: float = TOLERANCE) -> dict[str, float | int]:
    """Calculate component-error moments and inclusive frame-pass counts."""
    array = np.asarray(errors, dtype=float)
    tau = _finite_nonnegative(tolerance, "tolerance")
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1 or not np.all(np.isfinite(array)):
        raise ValueError("errors must be a non-empty finite frame-by-component array")
    absolute = np.abs(array)
    passing = np.all(absolute <= tau, axis=1)
    return {
        "frames": int(array.shape[0]),
        "components": int(array.size),
        "frames_passing": int(np.count_nonzero(passing)),
        "frame_pass_share": float(np.mean(passing)),
        "empirical_mae": float(np.mean(absolute)),
        "empirical_rmse": float(np.sqrt(np.mean(np.square(array)))),
        "empirical_mean_bias": float(np.mean(array)),
        "empirical_variance": float(np.var(array, ddof=0)),
    }


def _validate_noise_levels(values: Iterable[float]) -> tuple[float, ...]:
    try:
        levels = tuple(_finite_nonnegative(value, "noise_std") for value in values)
    except TypeError as exc:
        raise ValueError("noise levels must be a non-empty iterable") from exc
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("noise levels must be non-empty and contain no duplicates")
    return levels


def _make_row(
    noise_std: float,
    pattern: str,
    errors: np.ndarray,
    tolerance: float,
    q_energy: float,
) -> dict[str, Any]:
    empirical = summarize_component_errors(errors, tolerance)
    theory = theoretical_component_metrics(noise_std, q_energy)
    theory_pass = theoretical_frame_pass_probability(noise_std, tolerance, errors.shape[1], q_energy)
    if noise_std > 0.0:
        lower, upper = wilson_interval(empirical["frames_passing"], empirical["frames"])
    else:
        # This is an exact deterministic condition; a sampling CI is not applicable.
        lower, upper = None, None
    return {
        "noise_std": float(noise_std),
        "vector_pattern": pattern,
        **empirical,
        "wilson95_lower": lower,
        "wilson95_upper": upper,
        "theory_mae": theory["mae"],
        "abs_mae_gap": abs(empirical["empirical_mae"] - theory["mae"]),
        "theory_rmse": theory["rmse"],
        "abs_rmse_gap": abs(empirical["empirical_rmse"] - theory["rmse"]),
        "theory_mean_bias": theory["mean_bias"],
        "abs_mean_bias_gap": abs(empirical["empirical_mean_bias"] - theory["mean_bias"]),
        "theory_variance": theory["variance"],
        "abs_variance_gap": abs(empirical["empirical_variance"] - theory["variance"]),
        "theory_frame_pass_probability": theory_pass,
        "abs_pass_probability_gap": abs(empirical["frame_pass_share"] - theory_pass),
    }


def run_gaussian_calibration(
    trials_per_vector_per_level: int = TRIALS_PER_VECTOR_PER_LEVEL,
    master_seed: int = MASTER_SEED,
    gaussian_levels: Iterable[float] = GAUSSIAN_STD_LEVELS,
    chunk_frames: int = CHUNK_FRAMES,
) -> dict[str, Any]:
    """Run seeded v3 Gaussian calibration, streaming frames in bounded chunks."""
    trials = _positive_integer(trials_per_vector_per_level, "trials_per_vector_per_level")
    chunk = _positive_integer(chunk_frames, "chunk_frames")
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(master_seed, (int, np.integer)):
        raise ValueError("master_seed must be an integer")
    seed = int(master_seed)
    if seed == int(V2_MASTER_SEED):
        raise ValueError("v3 master_seed must differ from the v2 master seed")
    levels = _validate_noise_levels(gaussian_levels)

    config = WaveConfig(
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        cycles_per_symbol=CYCLES_PER_SYMBOL,
        amplitude_min=AMPLITUDE_MIN,
    )
    template = q_template(config)
    energy = float(template @ template)
    if not math.isclose(energy, 32.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"unexpected template energy: {energy}")
    patterns = build_input_patterns()
    if tuple(patterns) != EXPECTED_PATTERN_NAMES:
        raise RuntimeError("the v2 fixed-vector panel no longer matches the v3 calibration contract")

    rows: list[dict[str, Any]] = []
    for level_index, sigma in enumerate(levels):
        pattern_errors: list[np.ndarray] = []
        for pattern_index, (pattern_name, reference) in enumerate(patterns.items()):
            clean = encode_wave(reference, config)
            rng = np.random.default_rng(np.random.SeedSequence([seed, level_index, pattern_index]))
            error_chunks: list[np.ndarray] = []
            completed = 0
            while completed < trials:
                batch_size = min(chunk, trials - completed)
                noise = rng.normal(0.0, sigma, size=(batch_size, VECTOR_LENGTH, SAMPLES_PER_SYMBOL))
                received = clean[None, :, :] + noise
                decoded = decode_wave(received.reshape(-1, SAMPLES_PER_SYMBOL), config)
                estimates = decoded.reshape(batch_size, VECTOR_LENGTH)
                error_chunks.append(estimates - reference[None, :])
                completed += batch_size
            errors = np.concatenate(error_chunks, axis=0)
            pattern_errors.append(errors)
            rows.append(_make_row(sigma, pattern_name, errors, TOLERANCE, energy))

        pooled_errors = np.concatenate(pattern_errors, axis=0)
        rows.append(_make_row(sigma, POOLED_PATTERN, pooled_errors, TOLERANCE, energy))

    total_frames = len(patterns) * len(levels) * trials
    full_protocol = (
        trials == TRIALS_PER_VECTOR_PER_LEVEL
        and levels == GAUSSIAN_STD_LEVELS
        and seed == MASTER_SEED
    )
    contract = {
        "classification": "synthetic dimensionless software calibration under a stipulated Gaussian-noise model; no physical validity claimed",
        "protocol_status": "v3 analytical Gaussian calibration; fixed six-vector/ten-level protocol; not a physical validation, generalized channel-robustness study, or decoder comparison",
        "preservation": "digital_to_wave.py, robustness_benchmark_v2.py, and tests/test_robustness_benchmark_v2.py are used without modification by this v3 addition",
        "full_protocol_run": full_protocol,
        "master_seed": seed,
        "v2_master_seed": int(V2_MASTER_SEED),
        "randomized_vector_seed": int(VECTOR_SEED),
        "trials_per_vector_per_noise_level": trials,
        "chunk_frames": chunk,
        "vector_length": VECTOR_LENGTH,
        "vector_patterns": {name: np.asarray(vector, dtype=float).tolist() for name, vector in patterns.items()},
        "samples_per_symbol": config.samples_per_symbol,
        "cycles_per_symbol": config.cycles_per_symbol,
        "angular_frequency_radians_per_sample": 2.0 * math.pi * config.cycles_per_symbol / config.samples_per_symbol,
        "phase_radians": math.pi / 4.0,
        "negative_encoder_phase_radians": 5.0 * math.pi / 4.0,
        "amplitude_min": config.amplitude_min,
        "template_energy_q_dot_q": energy,
        "input_range_dimensionless": [-2.0, 2.0],
        "decoder": "existing digital_to_wave.decode_wave least-squares projection onto q[n]=sin(omega*n+pi/4), divided by A_min*(q dot q); A_min=1",
        "noise_model": {
            "name": "iid_gaussian_sample_noise",
            "sample_standard_deviation_levels_dimensionless": list(levels),
            "independent_across_samples": True,
            "independent_across_segments": True,
            "independent_across_frames": True,
            "construction": "one NumPy IID normal batch of shape (frames, 32 segments, 64 samples) per seeded vector/level stream; disjoint segments therefore contain independent noise samples",
        },
        "tolerance_dimensionless": TOLERANCE,
        "frame_pass_rule": "inclusive: a frame passes iff all 32 absolute component errors are <= 0.25",
        "total_frames": total_frames,
        "frames_per_noise_level": len(patterns) * trials,
        "expected_component_count_per_noise_level": len(patterns) * trials * VECTOR_LENGTH,
        "seed_streams": "NumPy SeedSequence([master_seed, noise_level_index, vector_index]); each vector/level uses a separate stream, and successive frames consume disjoint IID draws",
        "closed_form_model": {
            "signed_estimator": "xhat=(r dot q)/(q dot q), because A_min=1",
            "component_error_distribution": "N(0, sigma^2/32), conditional on known template and IID N(0,sigma^2) sample noise",
            "component_mae": "sqrt(2/pi)*sigma/sqrt(32)",
            "component_rmse": "sigma/sqrt(32)",
            "component_mean_bias": "0",
            "component_variance": "sigma^2/32",
            "frame_pass_probability_sigma_positive": "[2*Phi(0.25*sqrt(32)/sigma)-1]^32; equivalently [erf(0.25*sqrt(32)/(sqrt(2)*sigma))]^32",
            "frame_pass_probability_sigma_zero": 1.0,
            "independence_derivation": "Each segment uses a disjoint 64-sample subset of mutually independent noise samples; linear projections of disjoint subsets are independent. The 32 components of a frame therefore have independent Gaussian projection errors.",
            "convention_check": "Positive encoding uses q=sin(omega*n+pi/4); negative phase 5pi/4 equals -q, so clean segments are N*q. The existing decoder divides by A_min*(q dot q); with A_min=1 and q dot q=32 this equals the stipulated estimator.",
        },
        "metric_definitions": {
            "empirical_mae": "mean absolute signed component error over all frame components",
            "empirical_rmse": "sqrt(mean squared signed component error over all frame components)",
            "empirical_mean_bias": "mean signed component error",
            "empirical_variance": "population variance (ddof=0) of signed component errors",
            "pass_rate": "passing frames divided by frames, with inclusive max-absolute residual threshold",
            "confidence_interval": "two-sided 95% Wilson score interval for stochastic pass rates; omitted for deterministic sigma=0 rows",
            "calibration_gaps": "absolute empirical-minus-theoretical difference for each metric; pass-probability gap is absolute rate difference",
            "pooled_results": "component metrics pool equal-size frame panels from all six vectors; pooled pass interval is Wilson on the summed frame-level successes/trials",
        },
        "expected_total_frames": len(patterns) * len(levels) * trials,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    result = {"benchmark": "synthetic_gaussian_matched_filter_calibration_v3", "contract": contract, "results": rows}
    validate_calibration_result(result)
    return result


def _close(actual: float, expected: float, *, atol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=atol)


def validate_calibration_result(result: dict[str, Any]) -> None:
    """Check protocol/result consistency and derived values before serialization."""
    if not isinstance(result, dict) or result.get("benchmark") != "synthetic_gaussian_matched_filter_calibration_v3":
        raise ValueError("unexpected v3 benchmark identifier")
    contract = result.get("contract")
    rows = result.get("results")
    if not isinstance(contract, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("result must contain a contract and non-empty results")
    if contract.get("v2_master_seed") != int(V2_MASTER_SEED) or contract.get("master_seed") == contract.get("v2_master_seed"):
        raise ValueError("v3 master seed must be distinct from v2")
    if isinstance(contract.get("master_seed"), bool) or not isinstance(contract.get("master_seed"), int):
        raise ValueError("manifest master seed must be an integer")
    if "not a physical validation" not in contract.get("protocol_status", ""):
        raise ValueError("protocol status must retain its scoped synthetic-calibration label")
    if contract.get("noise_model", {}).get("name") != "iid_gaussian_sample_noise":
        raise ValueError("v3 must be identified as Gaussian calibration, not another v2 condition")
    if contract.get("noise_model", {}).get("independent_across_segments") is not True:
        raise ValueError("independence across segments must be explicitly recorded")

    trials = _positive_integer(contract.get("trials_per_vector_per_noise_level"), "trials_per_vector_per_noise_level")
    vector_length = _positive_integer(contract.get("vector_length"), "vector_length")
    patterns = contract.get("vector_patterns")
    if not isinstance(patterns, dict) or tuple(patterns) != EXPECTED_PATTERN_NAMES:
        raise ValueError("manifest must retain the six named v2 vectors in their established order")
    if vector_length != VECTOR_LENGTH or any(
        not isinstance(vector, list) or len(vector) != vector_length
        or any(not math.isfinite(float(value)) or abs(float(value)) > 2.0 for value in vector)
        for vector in patterns.values()
    ):
        raise ValueError("manifest vector definitions do not match the 32-value input contract")
    if contract.get("samples_per_symbol") != SAMPLES_PER_SYMBOL or contract.get("cycles_per_symbol") != CYCLES_PER_SYMBOL:
        raise ValueError("sample geometry does not match the v2 receiver")
    if contract.get("amplitude_min") != AMPLITUDE_MIN or not _close(contract.get("template_energy_q_dot_q"), 32.0):
        raise ValueError("amplitude normalization or template energy is inconsistent")
    expected_omega = 2.0 * math.pi * CYCLES_PER_SYMBOL / SAMPLES_PER_SYMBOL
    if not _close(contract.get("phase_radians"), math.pi / 4.0) or not _close(
        contract.get("negative_encoder_phase_radians"), 5.0 * math.pi / 4.0
    ) or not _close(contract.get("angular_frequency_radians_per_sample"), expected_omega):
        raise ValueError("manifest phase or carrier frequency does not match the encoder convention")
    config = WaveConfig(SAMPLES_PER_SYMBOL, CYCLES_PER_SYMBOL, AMPLITUDE_MIN)
    energy = float(q_template(config) @ q_template(config))
    if not _close(energy, contract["template_energy_q_dot_q"]):
        raise ValueError("manifest q energy does not match the actual decoder template")

    noise_levels = _validate_noise_levels(contract.get("noise_model", {}).get("sample_standard_deviation_levels_dimensionless", ()))
    expected_frames = len(patterns) * len(noise_levels) * trials
    if contract.get("total_frames") != expected_frames or contract.get("expected_total_frames") != expected_frames:
        raise ValueError("manifest total frame count is inconsistent with vectors, levels, and trials")
    if contract.get("frames_per_noise_level") != len(patterns) * trials:
        raise ValueError("manifest per-level frame count is inconsistent")
    if contract.get("expected_component_count_per_noise_level") != len(patterns) * trials * vector_length:
        raise ValueError("manifest per-level component count is inconsistent")
    expected_full_protocol = (
        trials == TRIALS_PER_VECTOR_PER_LEVEL
        and noise_levels == GAUSSIAN_STD_LEVELS
        and contract.get("master_seed") == MASTER_SEED
    )
    if contract.get("full_protocol_run") is not expected_full_protocol:
        raise ValueError("full-protocol flag does not match the declared run settings")
    if contract.get("tolerance_dimensionless") != TOLERANCE:
        raise ValueError("tolerance differs from the inclusive v2 convention")
    if len(rows) != len(noise_levels) * (len(patterns) + 1):
        raise ValueError("result row count does not cover every vector and pooled level")

    row_map: dict[tuple[float, str], dict[str, Any]] = {}
    allowed_names = set(patterns) | {POOLED_PATTERN}
    for row in rows:
        if not isinstance(row, dict) or any(field not in row for field in CSV_FIELDS):
            raise ValueError("result row is missing required fields")
        key = (float(row["noise_std"]), row["vector_pattern"])
        if key in row_map:
            raise ValueError("duplicate vector/noise result row")
        if key[0] not in noise_levels or key[1] not in allowed_names:
            raise ValueError("result row is outside the declared manifest")
        frame_count = len(patterns) * trials if key[1] == POOLED_PATTERN else trials
        if row["frames"] != frame_count or row["components"] != frame_count * vector_length:
            raise ValueError("result frame/component count disagrees with the manifest")
        successes = row["frames_passing"]
        if isinstance(successes, bool) or not isinstance(successes, int) or not 0 <= successes <= frame_count:
            raise ValueError("invalid frame pass count")
        if not _close(row["frame_pass_share"], successes / frame_count):
            raise ValueError("pass share does not match frame counts")
        sigma = key[0]
        theory = theoretical_component_metrics(sigma, energy)
        expected_probability = theoretical_frame_pass_probability(sigma, TOLERANCE, vector_length, energy)
        for field, expected in (
            ("theory_mae", theory["mae"]),
            ("theory_rmse", theory["rmse"]),
            ("theory_mean_bias", theory["mean_bias"]),
            ("theory_variance", theory["variance"]),
            ("theory_frame_pass_probability", expected_probability),
        ):
            if not _close(row[field], expected):
                raise ValueError(f"theoretical value {field} is inconsistent")
        for field in ("empirical_mae", "empirical_rmse", "empirical_variance"):
            if not math.isfinite(float(row[field])) or float(row[field]) < 0.0:
                raise ValueError(f"invalid empirical metric {field}")
        if not math.isfinite(float(row["empirical_mean_bias"])):
            raise ValueError("invalid empirical mean bias")
        for empirical, theoretical, gap in (
            (row["empirical_mae"], row["theory_mae"], row["abs_mae_gap"]),
            (row["empirical_rmse"], row["theory_rmse"], row["abs_rmse_gap"]),
            (row["empirical_mean_bias"], row["theory_mean_bias"], row["abs_mean_bias_gap"]),
            (row["empirical_variance"], row["theory_variance"], row["abs_variance_gap"]),
            (row["frame_pass_share"], row["theory_frame_pass_probability"], row["abs_pass_probability_gap"]),
        ):
            if not _close(gap, abs(float(empirical) - float(theoretical))):
                raise ValueError("absolute calibration gap does not match its operands")
        lower, upper = row["wilson95_lower"], row["wilson95_upper"]
        if sigma == 0.0:
            if lower is not None or upper is not None or successes != frame_count:
                raise ValueError("zero-noise rows must be exact-pass deterministic rows without a sampling CI")
        else:
            expected_ci = wilson_interval(successes, frame_count)
            if lower is None or upper is None or not _close(lower, expected_ci[0]) or not _close(upper, expected_ci[1]):
                raise ValueError("Wilson interval does not match the observed binomial pass count")
        row_map[key] = row

    for sigma in noise_levels:
        vector_rows = [row_map[(sigma, name)] for name in patterns]
        pooled = row_map[(sigma, POOLED_PATTERN)]
        if pooled["frames_passing"] != sum(row["frames_passing"] for row in vector_rows):
            raise ValueError("pooled pass count does not equal the sum of vector counts")
        if not _close(pooled["frame_pass_share"], sum(row["frames_passing"] for row in vector_rows) / pooled["frames"]):
            raise ValueError("pooled pass share is inconsistent")
        for field in ("empirical_mae", "empirical_mean_bias"):
            pooled_mean = sum(float(row[field]) for row in vector_rows) / len(vector_rows)
            if not _close(pooled[field], pooled_mean):
                raise ValueError(f"pooled {field} is inconsistent with its vector rows")
        pooled_rmse = math.sqrt(sum(float(row["empirical_rmse"]) ** 2 for row in vector_rows) / len(vector_rows))
        if not _close(pooled["empirical_rmse"], pooled_rmse):
            raise ValueError("pooled RMSE is inconsistent with its vector rows")
        pooled_bias = float(pooled["empirical_mean_bias"])
        pooled_variance = sum(
            float(row["empirical_variance"]) + float(row["empirical_mean_bias"]) ** 2
            for row in vector_rows
        ) / len(vector_rows) - pooled_bias ** 2
        if not _close(pooled["empirical_variance"], pooled_variance):
            raise ValueError("pooled variance is inconsistent with its vector rows")


def write_calibration_outputs(
    result: dict[str, Any], output_dir: Path = ARTIFACTS
) -> tuple[Path, Path, Path]:
    """Write validated v3 JSON, CSV, and an empirical-versus-theory plot."""
    validate_calibration_result(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "robustness-benchmark-v3-gaussian-calibration.json"
    csv_path = output_dir / "robustness-benchmark-v3-gaussian-calibration.csv"
    plot_path = output_dir / "robustness-benchmark-v3-gaussian-calibration.png"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["results"])

    pooled_rows = sorted(
        (row for row in result["results"] if row["vector_pattern"] == POOLED_PATTERN),
        key=lambda row: row["noise_std"],
    )
    x = np.asarray([row["noise_std"] for row in pooled_rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    axes[0].plot(x, [row["empirical_mae"] for row in pooled_rows], "o", color="#277da1", label="empirical MAE")
    axes[0].plot(x, [row["theory_mae"] for row in pooled_rows], "-", color="#277da1", label="predicted MAE")
    axes[0].plot(x, [row["empirical_rmse"] for row in pooled_rows], "s", color="#f3722c", label="empirical RMSE")
    axes[0].plot(x, [row["theory_rmse"] for row in pooled_rows], "--", color="#f3722c", label="predicted RMSE")
    axes[0].set(title="Component error calibration", xlabel="sample-noise SD σ (dimensionless)", ylabel="component error (dimensionless)")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    empirical_rate = np.asarray([row["frame_pass_share"] for row in pooled_rows])
    lower = np.asarray([
        row["wilson95_lower"] if row["wilson95_lower"] is not None else row["frame_pass_share"]
        for row in pooled_rows
    ])
    upper = np.asarray([
        row["wilson95_upper"] if row["wilson95_upper"] is not None else row["frame_pass_share"]
        for row in pooled_rows
    ])
    axes[1].errorbar(
        x, empirical_rate,
        yerr=np.vstack((np.maximum(0.0, empirical_rate - lower), np.maximum(0.0, upper - empirical_rate))),
        fmt="o", capsize=2, color="#277da1", label="empirical pass rate (95% Wilson CI)",
    )
    axes[1].plot(x, [row["theory_frame_pass_probability"] for row in pooled_rows], "-", color="#d1495b", label="closed-form probability")
    axes[1].set(title="32-component frame pass probability", xlabel="sample-noise SD σ (dimensionless)", ylabel="frames passing", ylim=(-0.03, 1.03))
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("v3 Gaussian calibration: synthetic IID-noise model only")
    fig.savefig(plot_path, dpi=180, facecolor="white")
    plt.close(fig)
    return json_path, csv_path, plot_path


def main() -> int:
    result = run_gaussian_calibration()
    json_path, csv_path, plot_path = write_calibration_outputs(result)
    pooled = [row for row in result["results"] if row["vector_pattern"] == POOLED_PATTERN]
    print(
        f"v3 Gaussian calibration: {len(EXPECTED_PATTERN_NAMES)} vectors x "
        f"{TRIALS_PER_VECTOR_PER_LEVEL:,} frames x {len(GAUSSIAN_STD_LEVELS)} levels "
        f"= {result['contract']['total_frames']:,} frames; seed={MASTER_SEED}"
    )
    for row in pooled:
        ci = "deterministic" if row["wilson95_lower"] is None else (
            f"Wilson95=[{row['wilson95_lower']:.4f}, {row['wilson95_upper']:.4f}]"
        )
        print(
            f"sigma={row['noise_std']:.2f}: pass={row['frames_passing']}/{row['frames']} "
            f"({row['frame_pass_share']:.4f}; theory={row['theory_frame_pass_probability']:.4f}; "
            f"abs gap={row['abs_pass_probability_gap']:.4f}; {ci}); "
            f"MAE={row['empirical_mae']:.6f}/{row['theory_mae']:.6f}; "
            f"RMSE={row['empirical_rmse']:.6f}/{row['theory_rmse']:.6f}"
        )
    print(f"JSON: {json_path}\nCSV: {csv_path}\nplot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
