"""Exploratory v4 carrier-frequency-offset benchmark for the synthetic codec.

The v4 impairment is a synthetic continuous phase ramp across the complete
32-segment frame. It is deliberately separate from v2's constant segment-phase
rotation and v3's nominal-template Gaussian calibration. The original encoder,
nominal decoder, and v1-v3 sources/results are not modified.
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

from digital_to_wave import WaveConfig, decode_wave
from robustness_benchmark_v2 import (
    MASTER_SEED as V2_MASTER_SEED,
    VECTOR_SEED,
    build_input_patterns,
    wilson_interval,
)
from robustness_benchmark_v3 import MASTER_SEED as V3_MASTER_SEED


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TRIALS_PER_VECTOR_PER_CONDITION = 2_000
MASTER_SEED = 20261006
TOLERANCE = 0.25
VECTOR_LENGTH = 32
SAMPLES_PER_SYMBOL = 64
CYCLES_PER_SYMBOL = 4
AMPLITUDE_MIN = 1.0
NOISE_STD_LEVELS = (0.0, 0.45)
# Extra carrier cycles accrued during one nominal 64-sample symbol interval.
# These are synthetic stress values, not estimates of a physical oscillator.
CFO_CYCLES_PER_SYMBOL = (-0.05, -0.025, -0.01, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.01, 0.025, 0.05)
CHUNK_FRAMES = 128
POOLED_PATTERN = "POOLED_ALL_VECTORS"
METHODS = ("nominal_existing_decoder", "oracle_known_frequency")
EXPECTED_PATTERN_NAMES = (
    "mixed_signs",
    "sparse_and_zeros",
    "boundary_values",
    "randomized_1",
    "randomized_2",
    "randomized_3",
)
CSV_FIELDS = (
    "cfo_cycles_per_symbol", "relative_cfo_to_carrier", "phase_drift_deg_per_symbol",
    "phase_drift_deg_over_32_symbols", "noise_std", "vector_pattern", "method",
    "frames", "components", "frames_passing", "frame_pass_share",
    "wilson95_lower", "wilson95_upper", "uncertainty_kind", "empirical_mae",
    "empirical_rmse", "empirical_mean_bias", "empirical_variance",
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


def _finite_signed(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_levels(values: Iterable[float], label: str) -> tuple[float, ...]:
    try:
        levels = tuple(_finite_nonnegative(value, label) for value in values)
    except TypeError as exc:
        raise ValueError(f"{label} levels must be a non-empty iterable") from exc
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"{label} levels must be non-empty and unique")
    return levels


def _validate_signed_levels(values: Iterable[float], label: str) -> tuple[float, ...]:
    try:
        levels = tuple(_finite_signed(value, label) for value in values)
    except TypeError as exc:
        raise ValueError(f"{label} levels must be a non-empty iterable") from exc
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"{label} levels must be non-empty and unique")
    return levels


def nominal_template(config: WaveConfig = WaveConfig()) -> np.ndarray:
    """Return the unchanged receiver's local q[n]=sin(omega*n+pi/4) template."""
    n = np.arange(config.samples_per_symbol, dtype=float)
    omega = 2.0 * math.pi * config.cycles_per_symbol / config.samples_per_symbol
    return np.sin(omega * n + math.pi / 4.0)


def frequency_shifted_templates(
    cfo_cycles_per_symbol: float,
    config: WaveConfig = WaveConfig(),
    vector_length: int = VECTOR_LENGTH,
) -> np.ndarray:
    """Return one exact-offset sine template per symbol with continuous phase.

    The phase ramp uses the global sample index, so CFO phase does not reset at
    segment boundaries. Symbol timing and the frame boundary remain exact.
    """
    offset = _finite_signed(cfo_cycles_per_symbol, "cfo_cycles_per_symbol")
    count = _positive_integer(vector_length, "vector_length")
    length = config.samples_per_symbol
    symbol = np.arange(count, dtype=float)[:, None]
    local = np.arange(length, dtype=float)[None, :]
    global_sample = symbol * length + local
    nominal_omega = 2.0 * math.pi * config.cycles_per_symbol / length
    offset_omega = 2.0 * math.pi * offset / length
    return np.sin((nominal_omega + offset_omega) * global_sample + math.pi / 4.0)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def theoretical_frame_pass_probability(
    reference: Any,
    cfo_cycles_per_symbol: float,
    noise_std: float,
    method: str,
    tolerance: float = TOLERANCE,
    config: WaveConfig | None = None,
) -> float:
    """Closed-form pass probability for the stipulated linear-projection model."""
    values = np.asarray(reference, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("reference must be a non-empty finite vector")
    offset = _finite_signed(cfo_cycles_per_symbol, "cfo_cycles_per_symbol")
    sigma = _finite_nonnegative(noise_std, "noise_std")
    tau = _finite_nonnegative(tolerance, "tolerance")
    if method not in METHODS:
        raise ValueError("method must be one of the modeled receiver methods")
    cfg = config or WaveConfig(
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        cycles_per_symbol=CYCLES_PER_SYMBOL,
        amplitude_min=AMPLITUDE_MIN,
    )
    q0 = nominal_template(cfg)
    qf = frequency_shifted_templates(offset, cfg, values.size)
    energy0 = float(q0 @ q0)
    energyf = np.sum(qf * qf, axis=1)
    if method == "nominal_existing_decoder":
        gain = (qf @ q0) / energy0
        bias = (gain - 1.0) * values
        component_sd = np.full(values.size, sigma / math.sqrt(energy0))
    else:
        bias = np.zeros(values.size, dtype=float)
        component_sd = sigma / np.sqrt(energyf)
    if sigma == 0.0:
        return float(np.all(np.abs(bias) <= tau))
    probabilities = []
    for mean, sd in zip(bias, component_sd):
        upper = _normal_cdf((tau - float(mean)) / float(sd))
        lower = _normal_cdf((-tau - float(mean)) / float(sd))
        probabilities.append(max(0.0, min(1.0, upper - lower)))
    return float(np.prod(probabilities))


def _error_summary(errors: np.ndarray) -> dict[str, float | int]:
    if errors.ndim != 2 or errors.shape[0] < 1 or errors.shape[1] < 1 or not np.all(np.isfinite(errors)):
        raise ValueError("errors must be a non-empty finite frame-by-component array")
    absolute = np.abs(errors)
    passing = np.all(absolute <= TOLERANCE, axis=1)
    return {
        "frames": int(errors.shape[0]),
        "components": int(errors.size),
        "frames_passing": int(np.count_nonzero(passing)),
        "frame_pass_share": float(np.mean(passing)),
        "empirical_mae": float(np.mean(absolute)),
        "empirical_rmse": float(np.sqrt(np.mean(errors * errors))),
        "empirical_mean_bias": float(np.mean(errors)),
        "empirical_variance": float(np.var(errors, ddof=0)),
    }


def _make_row(
    offset: float,
    sigma: float,
    pattern: str,
    method: str,
    errors: np.ndarray,
    theory_probability: float,
    *,
    stochastic: bool,
) -> dict[str, Any]:
    summary = _error_summary(errors)
    if stochastic:
        lower, upper = wilson_interval(summary["frames_passing"], summary["frames"])
        uncertainty = "two-sided 95% Wilson interval; conditional on fixed vector panel and synthetic IID draws"
    else:
        lower, upper = None, None
        uncertainty = "deterministic; no sampling interval"
    return {
        "cfo_cycles_per_symbol": float(offset),
        "relative_cfo_to_carrier": float(offset / CYCLES_PER_SYMBOL),
        "phase_drift_deg_per_symbol": float(360.0 * offset),
        "phase_drift_deg_over_32_symbols": float(360.0 * offset * VECTOR_LENGTH),
        "noise_std": float(sigma),
        "vector_pattern": pattern,
        "method": method,
        **summary,
        "wilson95_lower": lower,
        "wilson95_upper": upper,
        "uncertainty_kind": uncertainty,
        "theory_frame_pass_probability": float(theory_probability),
        "abs_pass_probability_gap": abs(float(summary["frame_pass_share"]) - float(theory_probability)),
    }


def run_cfo_benchmark(
    trials_per_vector_per_condition: int = TRIALS_PER_VECTOR_PER_CONDITION,
    master_seed: int = MASTER_SEED,
    cfo_levels: Iterable[float] = CFO_CYCLES_PER_SYMBOL,
    noise_levels: Iterable[float] = NOISE_STD_LEVELS,
    chunk_frames: int = CHUNK_FRAMES,
) -> dict[str, Any]:
    """Run v4 with fixed vectors, paired receiver inputs, and bounded memory."""
    trials = _positive_integer(trials_per_vector_per_condition, "trials_per_vector_per_condition")
    chunk = _positive_integer(chunk_frames, "chunk_frames")
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(master_seed, (int, np.integer)):
        raise ValueError("master_seed must be an integer")
    seed = int(master_seed)
    if seed in (int(V2_MASTER_SEED), int(V3_MASTER_SEED)):
        raise ValueError("v4 master seed must differ from v2 and v3")
    offsets = _validate_signed_levels(cfo_levels, "cfo_cycles_per_symbol")
    noise_stds = _validate_levels(noise_levels, "noise_std")
    if 0.0 not in noise_stds or NOISE_STD_LEVELS[-1] not in noise_stds:
        raise ValueError("the v4 run must include its frozen clean and 0.45 AWGN controls")

    config = WaveConfig(SAMPLES_PER_SYMBOL, CYCLES_PER_SYMBOL, AMPLITUDE_MIN)
    q0 = nominal_template(config)
    energy0 = float(q0 @ q0)
    if not math.isclose(energy0, 32.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("unexpected nominal template energy")
    patterns = build_input_patterns()
    if tuple(patterns) != EXPECTED_PATTERN_NAMES or any(len(v) != VECTOR_LENGTH for v in patterns.values()):
        raise RuntimeError("the fixed six-vector panel no longer matches the v4 protocol")

    rows: list[dict[str, Any]] = []
    for offset_index, offset in enumerate(offsets):
        qf = frequency_shifted_templates(offset, config, VECTOR_LENGTH)
        energyf = np.sum(qf * qf, axis=1)
        for sigma in noise_stds:
            pattern_errors: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHODS}
            for pattern_index, (pattern_name, reference) in enumerate(patterns.items()):
                clean = reference[:, None] * qf
                actual_trials = 1 if sigma == 0.0 else trials
                rng = np.random.default_rng(np.random.SeedSequence([seed, offset_index, pattern_index]))
                chunks: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
                completed = 0
                while completed < actual_trials:
                    batch_size = min(chunk, actual_trials - completed)
                    if sigma == 0.0:
                        received = clean[None, :, :]
                    else:
                        noise = rng.normal(0.0, sigma, size=(batch_size, VECTOR_LENGTH, SAMPLES_PER_SYMBOL))
                        received = clean[None, :, :] + noise
                    nominal_estimates = decode_wave(
                        received.reshape(-1, SAMPLES_PER_SYMBOL), config
                    ).reshape(batch_size, VECTOR_LENGTH)
                    oracle_estimates = np.sum(received * qf[None, :, :], axis=2) / energyf[None, :]
                    chunks["nominal_existing_decoder"].append(nominal_estimates - reference[None, :])
                    chunks["oracle_known_frequency"].append(oracle_estimates - reference[None, :])
                    completed += batch_size

                for method in METHODS:
                    errors = np.concatenate(chunks[method], axis=0)
                    pattern_errors[method][pattern_name] = errors
                    probability = theoretical_frame_pass_probability(reference, offset, sigma, method, config=config)
                    rows.append(_make_row(
                        offset, sigma, pattern_name, method, errors, probability,
                        stochastic=(sigma > 0.0),
                    ))

            for method in METHODS:
                pooled_errors = np.concatenate(list(pattern_errors[method].values()), axis=0)
                pattern_probabilities = [
                    theoretical_frame_pass_probability(patterns[name], offset, sigma, method, config=config)
                    for name in patterns
                ]
                rows.append(_make_row(
                    offset, sigma, POOLED_PATTERN, method, pooled_errors,
                    float(np.mean(pattern_probabilities)), stochastic=(sigma > 0.0),
                ))

    # Retain the zero-output implementation only as the old weak sanity floor.
    # It is a deterministic comparison against the fixed inputs, not a decoder.
    floor_offset = 0.0
    floor_sigma_context = NOISE_STD_LEVELS[-1]
    for pattern_name, reference in patterns.items():
        errors = -np.asarray(reference, dtype=float)[None, :]
        rows.append(_make_row(
            floor_offset, floor_sigma_context, pattern_name, "weak_zero_output_floor",
            errors, float(np.all(np.abs(errors[0]) <= TOLERANCE)), stochastic=False,
        ))
    floor_rows = [row for row in rows if row["method"] == "weak_zero_output_floor"]
    floor_errors = np.concatenate([-np.asarray(v, dtype=float)[None, :] for v in patterns.values()], axis=0)
    rows.append(_make_row(
        floor_offset, floor_sigma_context, POOLED_PATTERN, "weak_zero_output_floor",
        floor_errors, float(np.mean([row["theory_frame_pass_probability"] for row in floor_rows])),
        stochastic=False,
    ))

    full_protocol = (
        trials == TRIALS_PER_VECTOR_PER_CONDITION
        and seed == MASTER_SEED
        and offsets == CFO_CYCLES_PER_SYMBOL
        and noise_stds == NOISE_STD_LEVELS
        and chunk == CHUNK_FRAMES
    )
    total_stochastic_frames = len(offsets) * len(patterns) * trials
    contract = {
        "classification": "synthetic dimensionless software experiment; CFO stress values are not measured or physically calibrated",
        "protocol_status": "v4 exploratory continuous carrier-frequency-offset sweep; conditions frozen in source before the full run; not a formal preregistration or physical validation",
        "full_protocol_run": full_protocol,
        "conditions_frozen_before_run": True,
        "preservation": "digital_to_wave.py and all v1-v3 source, tests, reports, and result artifacts are used without modification",
        "master_seed": seed,
        "v2_master_seed": int(V2_MASTER_SEED),
        "v3_master_seed": int(V3_MASTER_SEED),
        "randomized_vector_seed": int(VECTOR_SEED),
        "trials_per_vector_per_noisy_condition": trials,
        "chunk_frames": chunk,
        "vector_length": VECTOR_LENGTH,
        "vector_patterns": {name: np.asarray(vector, dtype=float).tolist() for name, vector in patterns.items()},
        "samples_per_symbol": config.samples_per_symbol,
        "nominal_cycles_per_symbol": config.cycles_per_symbol,
        "nominal_angular_frequency_radians_per_sample": 2.0 * math.pi * config.cycles_per_symbol / config.samples_per_symbol,
        "phase_radians": math.pi / 4.0,
        "negative_encoder_phase_radians": 5.0 * math.pi / 4.0,
        "amplitude_min": config.amplitude_min,
        "nominal_template_energy": energy0,
        "input_range_dimensionless": [-2.0, 2.0],
        "noise_model": {
            "name": "iid_gaussian_sample_noise",
            "sample_standard_deviation_levels_dimensionless": list(noise_stds),
            "independent_across_samples_segments_and_frames": True,
            "construction": "seeded NumPy normal draws on waveform samples; both receiver methods decode the same received realization",
        },
        "carrier_frequency_offset_model": {
            "name": "synthetic_continuous_linear_phase_ramp",
            "extra_cycles_per_64_sample_symbol_levels": list(offsets),
            "relative_offsets_to_nominal_carrier": [float(value / config.cycles_per_symbol) for value in offsets],
            "phase_drift_degrees_per_symbol_period": [float(360.0 * value) for value in offsets],
            "phase_drift_degrees_over_32_symbol_periods": [float(360.0 * value * VECTOR_LENGTH) for value in offsets],
            "equation": "q[k,n]=sin((omega0+2*pi*delta_cycles/64)*(64*k+n)+pi/4), for global sample index 64*k+n; signed input multiplies q",
            "phase_continuity": "CFO phase is continuous across the 32 segments; the known sign phase is retained; segment timing and frame boundary are exact",
            "sample_timing_offset": "none",
            "level_basis": "explicit synthetic stress grid only; no device oscillator, channel trace, or physical operating range was supplied",
        },
        "decoder_methods": {
            "nominal_existing_decoder": "unchanged digital_to_wave.decode_wave using the nominal 4-cycle/64-sample local template and exact segment framing; it is not CFO-compensated",
            "oracle_known_frequency": "analysis-only upper-bound receiver with exact CFO and frame timing supplied; projects each segment onto its exact frequency-shifted template and divides by that template's energy; not an implementation candidate or included in existing decoder",
            "weak_zero_output_floor": "always returns zeros; recorded once against each fixed vector at zero CFO; ignores the waveform and is not a competing decoder",
        },
        "tolerance_dimensionless": TOLERANCE,
        "frame_pass_rule": "inclusive: all 32 absolute component errors must be <= 0.25",
        "frame_counts": {
            "noisy_monte_carlo_input_frames_per_cfo": len(patterns) * trials,
            "total_noisy_monte_carlo_input_frames": total_stochastic_frames,
            "clean_deterministic_input_frames_per_cfo": len(patterns),
            "weak_floor_fixed_vectors": len(patterns),
        },
        "seed_streams": "NumPy SeedSequence([master_seed, cfo_level_index, vector_index]); paired receiver methods share each noise realization",
        "analytic_calibration": {
            "nominal_estimator": "xhat_k=(r_k dot q0)/32; under CFO its mean gain is (qk dot q0)/32 and sample-noise SD is sigma/sqrt(32)",
            "oracle_estimator": "xhat_k=(r_k dot qk)/(qk dot qk); under IID Gaussian sample noise its error is N(0,sigma^2/(qk dot qk))",
            "frame_probability": "product over 32 independent components of P(-0.25 <= Normal(bias_k,sd_k) <= 0.25); at sigma=0 use the exact deterministic pass rule",
            "pooled_probability": "equal-weight mean of the six fixed-pattern theoretical frame probabilities; observed pooled Wilson interval is conditional on this vector panel",
        },
        "source_basis": [
            {"url": "https://uk.mathworks.com/help/satcom/ug/ccsds-hdr-optical-link-simulation-for-1550nm.html", "relevance": "documents CFO as a separate impairment whose phase accumulates linearly over time, and distinguishes sample-clock offset"},
            {"url": "https://cioffi-group.stanford.edu/doc/book/chap6.pdf", "relevance": "communications textbook treatment of carrier recovery and receiver local-oscillator frequency offset"},
        ],
        "metric_definitions": {
            "frame_pass_share": "passing frames divided by frames; pass iff every absolute component error is <= 0.25",
            "confidence_interval": "two-sided 95% Wilson score interval for Monte Carlo pass shares; deterministic clean/floor rows have no interval",
            "uncertainty_scope": "intervals quantify only IID simulation variation conditional on the six fixed vectors and stated synthetic model; they do not quantify model or physical uncertainty",
            "component_errors": "empirical MAE, RMSE, mean signed error, and population variance across all decoded vector components",
        },
        "expected_total_stochastic_input_frames": total_stochastic_frames,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    result = {"benchmark": "synthetic_continuous_cfo_stress_v4", "contract": contract, "results": rows}
    validate_cfo_result(result)
    return result


def _close(a: float, b: float, atol: float = 1e-11) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=atol)


def validate_cfo_result(result: dict[str, Any]) -> None:
    """Reject inconsistent manifests, rows, pools, or uncertainty annotations."""
    if not isinstance(result, dict) or result.get("benchmark") != "synthetic_continuous_cfo_stress_v4":
        raise ValueError("unexpected v4 benchmark identifier")
    contract, rows = result.get("contract"), result.get("results")
    if not isinstance(contract, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("result must contain a contract and non-empty results")
    if contract.get("conditions_frozen_before_run") is not True:
        raise ValueError("the v4 protocol must be marked frozen before its full run")
    if "not a formal preregistration" not in contract.get("protocol_status", ""):
        raise ValueError("protocol status must not overstate preregistration")
    seed = contract.get("master_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed in (V2_MASTER_SEED, V3_MASTER_SEED):
        raise ValueError("v4 must use an integer seed distinct from v2 and v3")
    patterns = contract.get("vector_patterns")
    if not isinstance(patterns, dict) or tuple(patterns) != EXPECTED_PATTERN_NAMES:
        raise ValueError("manifest must retain all six v2 vectors in order")
    if any(not isinstance(v, list) or len(v) != VECTOR_LENGTH or any(not math.isfinite(float(x)) or abs(float(x)) > 2 for x in v) for v in patterns.values()):
        raise ValueError("manifest vectors do not match the input contract")
    cfo = _validate_signed_levels(contract.get("carrier_frequency_offset_model", {}).get("extra_cycles_per_64_sample_symbol_levels", ()), "cfo_cycles_per_symbol")
    noise = _validate_levels(contract.get("noise_model", {}).get("sample_standard_deviation_levels_dimensionless", ()), "noise_std")
    if 0.0 not in noise or NOISE_STD_LEVELS[-1] not in noise:
        raise ValueError("manifest must include clean and 0.45 AWGN controls")
    trials = _positive_integer(contract.get("trials_per_vector_per_noisy_condition"), "trials_per_vector_per_noisy_condition")
    expected_stochastic = len(cfo) * len(patterns) * trials
    if contract.get("expected_total_stochastic_input_frames") != expected_stochastic:
        raise ValueError("stochastic frame count does not agree with the frozen dimensions")
    expected_full = (
        trials == TRIALS_PER_VECTOR_PER_CONDITION
        and seed == MASTER_SEED
        and cfo == CFO_CYCLES_PER_SYMBOL
        and noise == NOISE_STD_LEVELS
        and contract.get("chunk_frames") == CHUNK_FRAMES
    )
    if contract.get("full_protocol_run") is not expected_full:
        raise ValueError("full-protocol marker disagrees with run dimensions")
    if contract.get("tolerance_dimensionless") != TOLERANCE or contract.get("vector_length") != VECTOR_LENGTH:
        raise ValueError("tolerance or vector length differs from frozen protocol")

    keys: set[tuple[float, float, str, str]] = set()
    row_map: dict[tuple[float, float, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or any(field not in row for field in CSV_FIELDS):
            raise ValueError("a result row is missing required fields")
        key = (float(row["cfo_cycles_per_symbol"]), float(row["noise_std"]), row["method"], row["vector_pattern"])
        if key in keys:
            raise ValueError("duplicate v4 result row")
        keys.add(key)
        offset, sigma, method, pattern = key
        if offset not in cfo or pattern not in set(patterns) | {POOLED_PATTERN}:
            raise ValueError("result row is outside the declared grid")
        if method not in set(METHODS) | {"weak_zero_output_floor"}:
            raise ValueError("unknown receiver method")
        floor = method == "weak_zero_output_floor"
        if floor and (offset != 0.0 or sigma != NOISE_STD_LEVELS[-1]):
            raise ValueError("weak floor must be recorded once at zero CFO and the 0.45-noise context")
        expected_frames = 1 if (sigma == 0.0 or floor) else trials
        if pattern == POOLED_PATTERN:
            expected_frames *= len(patterns)
        if row["frames"] != expected_frames or row["components"] != expected_frames * VECTOR_LENGTH:
            raise ValueError("row frame/component counts disagree with the manifest")
        successes = row["frames_passing"]
        if isinstance(successes, bool) or not isinstance(successes, int) or not 0 <= successes <= expected_frames:
            raise ValueError("invalid frame pass count")
        if not _close(row["frame_pass_share"], successes / expected_frames):
            raise ValueError("frame pass share disagrees with success count")
        if not 0.0 <= float(row["theory_frame_pass_probability"]) <= 1.0:
            raise ValueError("theoretical pass probability is outside [0, 1]")
        if not _close(row["abs_pass_probability_gap"], abs(row["frame_pass_share"] - row["theory_frame_pass_probability"])):
            raise ValueError("pass-probability gap is inconsistent")
        lower, upper = row["wilson95_lower"], row["wilson95_upper"]
        stochastic = sigma > 0.0 and not floor
        if stochastic:
            expected_ci = wilson_interval(successes, expected_frames)
            if lower is None or upper is None or not _close(lower, expected_ci[0]) or not _close(upper, expected_ci[1]):
                raise ValueError("Wilson interval disagrees with observed binomial counts")
            if "Wilson" not in row["uncertainty_kind"]:
                raise ValueError("stochastic rows must label their Wilson interval")
        elif lower is not None or upper is not None or "deterministic" not in row["uncertainty_kind"]:
            raise ValueError("deterministic rows must not report a sampling interval")
        row_map[key] = row

    expected_keys = set()
    for offset in cfo:
        for sigma in noise:
            for method in METHODS:
                for pattern in list(patterns) + [POOLED_PATTERN]:
                    expected_keys.add((offset, sigma, method, pattern))
    for pattern in list(patterns) + [POOLED_PATTERN]:
        expected_keys.add((0.0, NOISE_STD_LEVELS[-1], "weak_zero_output_floor", pattern))
    if keys != expected_keys:
        raise ValueError("v4 rows do not cover exactly the declared experiment and floor")

    for offset in cfo:
        for sigma in noise:
            for method in METHODS:
                component_rows = [row_map[(offset, sigma, method, name)] for name in patterns]
                pooled = row_map[(offset, sigma, method, POOLED_PATTERN)]
                if pooled["frames_passing"] != sum(row["frames_passing"] for row in component_rows):
                    raise ValueError("pooled pass count does not equal the six pattern counts")
                if not _close(pooled["theory_frame_pass_probability"], sum(row["theory_frame_pass_probability"] for row in component_rows) / len(component_rows)):
                    raise ValueError("pooled theoretical probability is not the equal-pattern mean")
                for field in ("empirical_mae", "empirical_mean_bias"):
                    expected = sum(float(row[field]) for row in component_rows) / len(component_rows)
                    if not _close(pooled[field], expected):
                        raise ValueError(f"pooled {field} disagrees with its vector rows")
                expected_rmse = math.sqrt(sum(float(row["empirical_rmse"]) ** 2 for row in component_rows) / len(component_rows))
                if not _close(pooled["empirical_rmse"], expected_rmse):
                    raise ValueError("pooled RMSE disagrees with its vector rows")
    floor_components = [row_map[(0.0, NOISE_STD_LEVELS[-1], "weak_zero_output_floor", name)] for name in patterns]
    floor_pooled = row_map[(0.0, NOISE_STD_LEVELS[-1], "weak_zero_output_floor", POOLED_PATTERN)]
    if floor_pooled["frames_passing"] != sum(row["frames_passing"] for row in floor_components):
        raise ValueError("weak-floor pooled pass count is inconsistent")


def write_cfo_outputs(result: dict[str, Any], output_dir: Path = ARTIFACTS) -> tuple[Path, Path, Path]:
    """Write the v4 protocol/results JSON, CSV, and focused comparison plot."""
    validate_cfo_result(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "robustness-benchmark-v4-cfo.json"
    csv_path = output_dir / "robustness-benchmark-v4-cfo.csv"
    plot_path = output_dir / "robustness-benchmark-v4-cfo.png"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["results"])

    pooled = [row for row in result["results"] if row["vector_pattern"] == POOLED_PATTERN]
    noisy = sorted((row for row in pooled if row["noise_std"] == NOISE_STD_LEVELS[-1]), key=lambda row: row["cfo_cycles_per_symbol"])
    clean = sorted((row for row in pooled if row["noise_std"] == 0.0), key=lambda row: row["cfo_cycles_per_symbol"])
    x = np.asarray([row["relative_cfo_to_carrier"] * 100.0 for row in noisy if row["method"] == METHODS[0]])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.27, wspace=0.23)
    colors = {METHODS[0]: "#277da1", METHODS[1]: "#f3722c"}
    labels = {METHODS[0]: "existing nominal decoder", METHODS[1]: "oracle: exact CFO known"}
    for method in METHODS:
        mr = [row for row in noisy if row["method"] == method]
        xx = np.asarray([row["relative_cfo_to_carrier"] * 100.0 for row in mr])
        rates = np.asarray([row["frame_pass_share"] for row in mr])
        lo = np.asarray([row["wilson95_lower"] for row in mr])
        hi = np.asarray([row["wilson95_upper"] for row in mr])
        axes[0].errorbar(xx, rates, yerr=np.vstack((np.maximum(0.0, rates - lo), np.maximum(0.0, hi - rates))), fmt="o", capsize=3,
                         color=colors[method], label=f"{labels[method]}: observed 95% Wilson CI")
        axes[0].plot(xx, [row["theory_frame_pass_probability"] for row in mr], "-", color=colors[method], alpha=0.7,
                     label=f"{labels[method]}: analytic prediction")
    axes[0].set(xlabel="CFO / nominal carrier (%)", ylabel="32-value frames passing", ylim=(-0.03, 1.03),
                title="AWGN SD 0.45; six fixed vectors")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for method in METHODS:
        mr = [row for row in clean if row["method"] == method]
        xx = [row["relative_cfo_to_carrier"] * 100.0 for row in mr]
        axes[1].plot(xx, [row["empirical_mae"] for row in mr], marker="o", color=colors[method], label=labels[method])
    axes[1].set(xlabel="CFO / nominal carrier (%)", ylabel="component MAE (dimensionless)", title="Clean deterministic frames")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.suptitle("v4 synthetic continuous-CFO stress test — no physical validity claimed")
    fig.savefig(plot_path, dpi=180, facecolor="white")
    plt.close(fig)
    return json_path, csv_path, plot_path


def main() -> int:
    result = run_cfo_benchmark()
    json_path, csv_path, plot_path = write_cfo_outputs(result)
    print(
        f"v4 CFO sweep: {len(EXPECTED_PATTERN_NAMES)} fixed vectors x "
        f"{TRIALS_PER_VECTOR_PER_CONDITION:,} frames x {len(CFO_CYCLES_PER_SYMBOL)} CFO levels "
        f"at SD=0.45 = {result['contract']['expected_total_stochastic_input_frames']:,} noisy input frames; "
        f"seed={MASTER_SEED}"
    )
    for row in result["results"]:
        if row["method"] == "nominal_existing_decoder" and row["noise_std"] == 0.45 and row["vector_pattern"] == POOLED_PATTERN:
            ci = f"Wilson95=[{row['wilson95_lower']:.4f}, {row['wilson95_upper']:.4f}]"
            print(
                f"CFO={row['cfo_cycles_per_symbol']:.4f} cycles/symbol "
                f"({100*row['relative_cfo_to_carrier']:.4f}%): "
                f"nominal pass={row['frames_passing']}/{row['frames']} ({row['frame_pass_share']:.4f}; {ci}); "
                f"theory={row['theory_frame_pass_probability']:.4f}; MAE={row['empirical_mae']:.5f}"
            )
    print(f"JSON: {json_path}\nCSV: {csv_path}\nplot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
