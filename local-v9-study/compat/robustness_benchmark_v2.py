"""Exploratory v2 robustness benchmark for the synthetic dimensionless codec.

The signed-amplitude encoder in digital_to_wave.py is not modified. Every
channel/noise/mismatch condition here is an explicit software stress test, not
an experimentally grounded physical model or a formal preregistration.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from digital_to_wave import WaveConfig, decode_wave, encode_wave


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TRIALS_PER_CONDITION = 200
MASTER_SEED = 20261003
VECTOR_SEED = 20261004
TOLERANCE = 0.25
GAUSSIAN_STD_LEVELS = (0.0, 0.03, 0.06, 0.12, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60)
LAPLACE_STD_LEVELS = (0.12, 0.30, 0.45)
PHASE_OFFSETS_RAD = (0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0)
WILSON_Z_95 = 1.959963984540054
AGGREGATE_PATTERN = "ALL_PATTERNS_EQUAL_WEIGHT"
CSV_FIELDS = (
    "scenario", "noise_model", "noise_std", "phase_offset_rad", "phase_offset_deg",
    "vector_pattern", "method", "trials", "vector_length", "mae", "rmse",
    "median_abs_error", "p95_abs_error", "max_abs_error", "frames_passing",
    "frame_pass_share", "wilson95_lower", "wilson95_upper",
)


def build_input_patterns(vector_seed: int = VECTOR_SEED) -> dict[str, np.ndarray]:
    """Return six fixed 32-value test vectors spanning distinct value patterns."""
    mixed_half = np.asarray(
        [-2.0, -1.75, -1.5, -1.0, -0.75, -0.25, -0.125, -1.25,
          2.0, 1.75, 1.5, 1.0, 0.75, 0.25, 0.125, 1.25], dtype=float
    )
    sparse = np.zeros(32, dtype=float)
    sparse[[0, 4, 15, 22, 31]] = [-2.0, 1.0, -0.25, 2.0, -1.5]
    boundary_half = np.asarray(
        [-2.0, -1.999, -1.5, -1.0, -0.25, -0.001, 0.0, 0.001,
          0.25, 1.0, 1.5, 1.999, 2.0, 0.0, -2.0, 2.0], dtype=float
    )
    rng = np.random.default_rng(vector_seed)
    patterns = {
        "mixed_signs": np.tile(mixed_half, 2),
        "sparse_and_zeros": sparse,
        "boundary_values": np.tile(boundary_half, 2),
    }
    for index in range(1, 4):
        patterns[f"randomized_{index}"] = rng.uniform(-2.0, 2.0, size=32)
    return patterns


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if isinstance(successes, bool) or not isinstance(successes, (int, np.integer)):
        raise ValueError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
        raise ValueError("trials must be an integer")
    if trials < 1 or successes < 0 or successes > trials or not math.isfinite(z) or z <= 0:
        raise ValueError("invalid Wilson interval arguments")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def make_noise(shape: tuple[int, ...], noise_model: str, noise_std: float, seed: int) -> np.ndarray:
    """Generate IID sample noise; Laplace is scaled to the requested marginal SD."""
    if not shape or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape):
        raise ValueError("shape must contain positive integer dimensions")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    if noise_model == "gaussian":
        return rng.normal(0.0, noise_std, size=shape)
    if noise_model == "laplace":
        # Var(Laplace(0,b)) = 2b²; this matches Gaussian marginal variance.
        return rng.laplace(0.0, noise_std / math.sqrt(2.0), size=shape)
    raise ValueError("noise_model must be 'gaussian' or 'laplace'")


def apply_carrier_phase_offset(signal: Any, phase_offset_rad: float, config: WaveConfig) -> np.ndarray:
    """Rotate each segment within the nominal single-carrier sine/cosine subspace.

    This is an arbitrary, idealized software phase-mismatch stress operator. It
    projects each row onto the known discrete sine/cosine basis and rotates the
    two coefficients. It is not a model of a physical channel or timing system.
    """
    array = np.asarray(signal, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != config.samples_per_symbol:
        raise ValueError("signal must have shape (n_values, samples_per_symbol)")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal must contain only finite values")
    if not math.isfinite(phase_offset_rad):
        raise ValueError("phase_offset_rad must be finite")
    n = np.arange(config.samples_per_symbol, dtype=float)
    theta = 2.0 * np.pi * config.cycles_per_symbol * n / config.samples_per_symbol
    sine = np.sin(theta)
    cosine = np.cos(theta)
    denominator = float(sine @ sine)
    sine_coefficient = (array @ sine) / denominator
    cosine_coefficient = (array @ cosine) / denominator
    delta = float(phase_offset_rad)
    shifted_sine_coefficient = sine_coefficient * math.cos(delta) - cosine_coefficient * math.sin(delta)
    shifted_cosine_coefficient = sine_coefficient * math.sin(delta) + cosine_coefficient * math.cos(delta)
    return shifted_sine_coefficient[:, None] * sine + shifted_cosine_coefficient[:, None] * cosine


def _summarize_frames(
    reference: np.ndarray,
    estimates: np.ndarray,
    tolerance: float,
    include_wilson: bool,
) -> dict[str, float | int | None]:
    expected = np.asarray(reference, dtype=float)
    actual = np.asarray(estimates, dtype=float)
    if expected.ndim not in (1, 2) or expected.size == 0 or not np.all(np.isfinite(expected)):
        raise ValueError("reference must be a non-empty finite vector or frame-by-vector array")
    if actual.ndim != 2 or actual.shape[0] == 0 or actual.shape[1] != expected.shape[-1]:
        raise ValueError("estimates must have shape (n_frames, len(reference))")
    if expected.ndim == 2 and expected.shape != actual.shape:
        raise ValueError("2D reference must have the same shape as estimates")
    if not np.all(np.isfinite(actual)):
        raise ValueError("estimates must contain only finite values")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    expected_frames = expected[None, :] if expected.ndim == 1 else expected
    absolute = np.abs(actual - expected_frames)
    frame_passes = np.max(absolute, axis=1) <= tolerance
    passed = int(np.count_nonzero(frame_passes))
    trials = int(actual.shape[0])
    lower, upper = wilson_interval(passed, trials) if include_wilson else (None, None)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
        "median_abs_error": float(np.median(absolute)),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "max_abs_error": float(np.max(absolute)),
        "frames_passing": passed,
        "frame_pass_share": passed / trials,
        "wilson95_lower": lower,
        "wilson95_upper": upper,
    }


def _validate_levels(values: Iterable[float], label: str, allow_zero: bool = True) -> tuple[float, ...]:
    try:
        levels = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-empty sequence of finite values") from exc
    if not levels or any(not math.isfinite(value) or value < 0 or (not allow_zero and value == 0) for value in levels):
        raise ValueError(f"{label} must contain finite non-negative values")
    if len(set(levels)) != len(levels):
        raise ValueError(f"{label} must not contain duplicates")
    return levels


def _validate_protocol_inputs(
    trials: int,
    master_seed: int,
    tolerance: float,
    gaussian_levels: Iterable[float],
    laplace_levels: Iterable[float],
    phase_offsets_rad: Iterable[float],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)):
        raise ValueError("master_seed must be an integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    gaussian = _validate_levels(gaussian_levels, "gaussian_levels")
    laplace = _validate_levels(laplace_levels, "laplace_levels", allow_zero=False)
    phases = _validate_levels(phase_offsets_rad, "phase_offsets_rad")
    return gaussian, laplace, phases


def _row(
    scenario: str,
    noise_model: str,
    noise_std: float | None,
    phase_offset_rad: float | None,
    pattern: str,
    method: str,
    reference: np.ndarray,
    estimates: np.ndarray,
    tolerance: float,
    include_wilson: bool,
) -> dict[str, Any]:
    phase_degrees = None if phase_offset_rad is None else math.degrees(phase_offset_rad)
    return {
        "scenario": scenario,
        "noise_model": noise_model,
        "noise_std": noise_std,
        "phase_offset_rad": phase_offset_rad,
        "phase_offset_deg": phase_degrees,
        "vector_pattern": pattern,
        "method": method,
        "trials": int(estimates.shape[0]),
        "vector_length": int(reference.shape[-1]),
        **_summarize_frames(reference, estimates, tolerance, include_wilson),
    }


def run_benchmark_v2(
    trials: int = TRIALS_PER_CONDITION,
    master_seed: int = MASTER_SEED,
    tolerance: float = TOLERANCE,
    gaussian_levels: Iterable[float] = GAUSSIAN_STD_LEVELS,
    laplace_levels: Iterable[float] = LAPLACE_STD_LEVELS,
    phase_offsets_rad: Iterable[float] = PHASE_OFFSETS_RAD,
) -> dict[str, Any]:
    """Run the fixed v2 exploratory protocol and return per-pattern/aggregate metrics."""
    gaussian, laplace, phases = _validate_protocol_inputs(
        trials, master_seed, tolerance, gaussian_levels, laplace_levels, phase_offsets_rad
    )
    config = WaveConfig(samples_per_symbol=64, cycles_per_symbol=4, amplitude_min=1.0)
    patterns = build_input_patterns()
    spawned = np.random.SeedSequence(int(master_seed)).spawn(
        len(patterns) * (len(gaussian) + len(laplace)) * trials
    )
    stream_index = 0
    rows: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str, float | None, float | None, str], list[np.ndarray]] = defaultdict(list)

    # IID sample-noise scenarios: independent child RNG streams per pattern,
    # distribution, level and trial. The Gaussian and Laplace scales denote
    # sample-level standard deviation; Laplace is variance-matched.
    for model, levels in (("gaussian", gaussian), ("laplace", laplace)):
        for level in levels:
            for pattern_name, reference in patterns.items():
                clean = encode_wave(reference, config)
                decoded: list[np.ndarray] = []
                for _ in range(trials):
                    rng = np.random.default_rng(spawned[stream_index])
                    stream_index += 1
                    if model == "gaussian":
                        noise = rng.normal(0.0, level, size=clean.shape)
                    else:
                        noise = rng.laplace(0.0, level / math.sqrt(2.0), size=clean.shape)
                    decoded.append(decode_wave(clean + noise, config))
                estimates = np.stack(decoded)
                include_ci = level > 0.0
                rows.append(_row("sample_noise", model, level, None, pattern_name,
                                 "nominal_matched_filter", reference, estimates, tolerance, include_ci))
                pooled[("sample_noise", model, level, None, "nominal_matched_filter")].append((reference, estimates))

                zero_frames = np.zeros_like(estimates)
                rows.append(_row("sample_noise", model, level, None, pattern_name,
                                 "zero_output_sanity_floor", reference, zero_frames, tolerance, False))
                pooled[("sample_noise", model, level, None, "zero_output_sanity_floor")].append((reference, zero_frames))

    # Deterministic phase rotations are single-frame software stress checks per
    # pattern; there is no sampling uncertainty interval for these rows.
    for phase in phases:
        for pattern_name, reference in patterns.items():
            clean = encode_wave(reference, config)
            mismatched = apply_carrier_phase_offset(clean, phase, config)
            estimate = decode_wave(mismatched, config)[None, :]
            rows.append(_row("carrier_phase_mismatch", "none", None, phase, pattern_name,
                             "nominal_matched_filter", reference, estimate, tolerance, False))
            pooled[("carrier_phase_mismatch", "none", None, phase, "nominal_matched_filter")].append((reference, estimate))

            zeros = np.zeros_like(estimate)
            rows.append(_row("carrier_phase_mismatch", "none", None, phase, pattern_name,
                             "zero_output_sanity_floor", reference, zeros, tolerance, False))
            pooled[("carrier_phase_mismatch", "none", None, phase, "zero_output_sanity_floor")].append((reference, zeros))

    for (scenario, model, level, phase, method), reference_groups in pooled.items():
        all_estimates = np.concatenate([estimates for _, estimates in reference_groups], axis=0)
        all_references = np.concatenate([
            np.broadcast_to(reference, estimates.shape) if reference.ndim == 1 else reference
            for reference, estimates in reference_groups
        ], axis=0)
        if scenario == "carrier_phase_mismatch":
            include_ci = False
        else:
            include_ci = bool(level is not None and level > 0 and method == "nominal_matched_filter")
        rows.append(_row(scenario, model, level, phase, AGGREGATE_PATTERN, method,
                         all_references, all_estimates, tolerance, include_ci))

    contract = {
        "classification": "exploratory synthetic dimensionless software benchmark; no physical validity claimed",
        "protocol_status": "v2 exploratory protocol; conditions fixed in the script before this final run; not formal preregistration",
        "equation_preservation": "digital_to_wave.encode_wave is unchanged; signed-amplitude equation and sign-phase convention are identical to the existing implementation",
        "paper_equation": "S_ijk(t) = (|N_ijk| A_min) sin(omega t + phi(N_ijk)); phi=pi/4 for N>0 and 5pi/4 for N<0; zero maps to zero amplitude",
        "paper_source_location": "DigitaltoWave-1, printed page 12 (as recorded in the project README)",
        "testbench_choices": {
            "input_range_dimensionless": [-2.0, 2.0],
            "vector_length": 32,
            "vector_patterns": {name: vector.tolist() for name, vector in patterns.items()},
            "randomized_vector_seed": VECTOR_SEED,
            "samples_per_symbol": config.samples_per_symbol,
            "cycles_per_symbol": config.cycles_per_symbol,
            "amplitude_min": config.amplitude_min,
            "sample_index_is_physical_time": False,
            "decoder": "least-squares projection onto the nominal sin(omega*n + pi/4) reference",
            "acceptance_tolerance_dimensionless": float(tolerance),
            "frame_pass_rule": "inclusive: a 32-value frame passes iff max absolute input/reconstruction residual <= tolerance",
            "trials_per_noise_cell": trials,
            "master_seed": int(master_seed),
            "random_streams": "NumPy SeedSequence(master_seed).spawn; one child generator per noise pattern/model/level/trial",
            "gaussian_sample_std_levels_dimensionless": list(gaussian),
            "laplace_sample_std_levels_dimensionless": list(laplace),
            "laplace_scale": "b = requested sample standard deviation / sqrt(2), giving the same marginal variance as Gaussian at equal listed std",
            "phase_offsets_radians": list(phases),
            "phase_mismatch_definition": "deterministic projection/rotation in the nominal single-carrier sine/cosine subspace; one frame per pattern and offset; arbitrary idealized software stress test, not a physical channel, timing model, or measured receiver mismatch",
            "zero_output_comparison": "weak sanity floor only: returns 32 zeros without using the received waveform; not a competing decoder and not evidence of superiority over a fair alternative",
            "alternative_nonzero_baseline": "none; no non-zero baseline is included because this protocol does not define an independently justified receiver with comparable information/assumptions",
            "confidence_intervals": "two-sided 95% Wilson score intervals for pass rates in stochastic nonzero sample-noise cells; omitted for zero-noise/deterministic rows and the deterministic zero-output baseline; pooled aggregate intervals use a binomial approximation over the balanced fixed vector panel",
            "uncertainty_scope": "Intervals quantify finite Monte Carlo uncertainty conditional on this synthetic protocol, not physical validity or model selection; pooled Wilson intervals are approximate when pass probabilities differ by fixed input pattern, so per-pattern intervals are more direct",
        },
        "metric_definitions": {
            "mae_rmse_median_p95_max": "component absolute-error summaries pooled over trials and vector entries; p95 uses NumPy linear quantile convention",
            "frames_passing": "number of frames satisfying the inclusive maximum-absolute residual threshold",
            "frame_pass_share": "frames_passing divided by trials; aggregate rows pool the six equally sized vector patterns",
            "wilson95_lower_upper": "95% Wilson score interval for stochastic matched-filter frame pass shares; null where not applicable",
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    result = {"benchmark": "synthetic_robustness_exploration_v2", "contract": contract, "results": rows}
    validate_result_integrity(result)
    return result


def validate_result_integrity(result: dict[str, Any]) -> None:
    """Reject malformed summaries before serialization or downstream reporting."""
    if result.get("benchmark") != "synthetic_robustness_exploration_v2":
        raise ValueError("unexpected benchmark identifier")
    contract = result.get("contract")
    rows = result.get("results")
    if not isinstance(contract, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("result must contain a contract and non-empty results list")
    if "not formal preregistration" not in contract.get("protocol_status", ""):
        raise ValueError("protocol status must preserve the exploratory/not-preregistered label")
    patterns = contract.get("testbench_choices", {}).get("vector_patterns", {})
    if len(patterns) != 6 or any(len(vector) != 32 for vector in patterns.values()):
        raise ValueError("protocol must include six fixed 32-value vectors")
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if any(field not in row for field in CSV_FIELDS):
            raise ValueError("result row is missing required fields")
        key = tuple(row[field] for field in (
            "scenario", "noise_model", "noise_std", "phase_offset_rad", "vector_pattern", "method"
        ))
        if key in seen:
            raise ValueError("duplicate result row key")
        seen.add(key)
        n = row["trials"]
        passed = row["frames_passing"]
        share = row["frame_pass_share"]
        if not isinstance(n, int) or n < 1 or not isinstance(passed, int) or not 0 <= passed <= n:
            raise ValueError("invalid trial or pass count")
        if not math.isclose(share, passed / n, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("frame pass share does not match counts")
        for field in ("mae", "rmse", "median_abs_error", "p95_abs_error", "max_abs_error"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"invalid metric: {field}")
        lower, upper = row["wilson95_lower"], row["wilson95_upper"]
        if (lower is None) != (upper is None):
            raise ValueError("Wilson interval endpoints must both be null or both be present")
        if lower is not None and not (
            -1e-12 <= lower <= share + 1e-12 and share <= upper + 1e-12 and upper <= 1.0 + 1e-12
        ):
            raise ValueError("invalid Wilson interval")


def write_benchmark_outputs(
    result: dict[str, Any], output_dir: Path = ARTIFACTS
) -> tuple[Path, Path, Path]:
    """Write validated v2 JSON/CSV summaries and an aggregate diagnostic plot."""
    validate_result_integrity(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "robustness-benchmark-v2.json"
    csv_path = output_dir / "robustness-benchmark-v2.csv"
    plot_path = output_dir / "robustness-benchmark-v2.png"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["results"])

    rows = result["results"]
    aggregate_nominal = [
        row for row in rows
        if row["vector_pattern"] == AGGREGATE_PATTERN and row["method"] == "nominal_matched_filter"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    colors = {"gaussian": "#277da1", "laplace": "#f3722c"}
    for model in ("gaussian", "laplace"):
        selected = sorted(
            (row for row in aggregate_nominal if row["scenario"] == "sample_noise" and row["noise_model"] == model),
            key=lambda row: row["noise_std"],
        )
        if not selected:
            continue
        x = np.asarray([row["noise_std"] for row in selected])
        y = np.asarray([100.0 * row["frame_pass_share"] for row in selected])
        lower = np.asarray([100.0 * row["wilson95_lower"] if row["wilson95_lower"] is not None else y[i]
                            for i, row in enumerate(selected)])
        upper = np.asarray([100.0 * row["wilson95_upper"] if row["wilson95_upper"] is not None else y[i]
                            for i, row in enumerate(selected)])
        yerr = np.vstack((np.maximum(0.0, y - lower), np.maximum(0.0, upper - y)))
        axes[0].errorbar(x, y, yerr=yerr, marker="o", capsize=2,
                         color=colors[model], label=f"{model.capitalize()} sample noise")
    axes[0].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0,
                    label="zero-output sanity floor (weak baseline)")
    axes[0].set(title="Stochastic sample-noise stress", xlabel="sample noise standard deviation (dimensionless)",
                ylabel="32-value frames passing (%)", ylim=(-3, 103))
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    phase_rows = sorted(
        (row for row in aggregate_nominal if row["scenario"] == "carrier_phase_mismatch"),
        key=lambda row: row["phase_offset_rad"],
    )
    if phase_rows:
        axes[1].plot([row["phase_offset_deg"] for row in phase_rows],
                     [100.0 * row["frame_pass_share"] for row in phase_rows], "o-",
                     color="#7b2cbf", label="nominal matched filter")
        axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0,
                        label="zero-output sanity floor (weak baseline)")
    axes[1].set(title="Deterministic phase-offset stress (software-only)",
                xlabel="applied carrier phase offset (degrees)", ylabel="six-pattern frames passing (%)",
                ylim=(-3, 103))
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Exploratory v2 synthetic benchmark; 95% Wilson intervals on noisy pass rates")
    fig.savefig(plot_path, dpi=170, facecolor="white")
    plt.close(fig)
    return json_path, csv_path, plot_path


def main() -> int:
    result = run_benchmark_v2()
    json_path, csv_path, plot_path = write_benchmark_outputs(result)
    aggregate = [
        row for row in result["results"]
        if row["vector_pattern"] == AGGREGATE_PATTERN and row["method"] == "nominal_matched_filter"
    ]
    print(f"v2 exploratory protocol: {len(result['contract']['testbench_choices']['vector_patterns'])} vectors; "
          f"{TRIALS_PER_CONDITION} trials per noise cell; seed={MASTER_SEED}")
    for row in aggregate:
        if row["scenario"] == "sample_noise":
            ci = ""
            if row["wilson95_lower"] is not None:
                ci = f", Wilson95=[{row['wilson95_lower']:.3f}, {row['wilson95_upper']:.3f}]"
            print(f"{row['noise_model']} std={row['noise_std']:.2f}: pass={row['frames_passing']}/{row['trials']} "
                  f"({100*row['frame_pass_share']:.1f}%), MAE={row['mae']:.5f}{ci}")
        else:
            print(f"phase={row['phase_offset_deg']:.1f} deg: pass={row['frames_passing']}/{row['trials']} "
                  f"({100*row['frame_pass_share']:.1f}%), MAE={row['mae']:.5f}")
    print(f"JSON: {json_path}\nCSV: {csv_path}\nplot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
