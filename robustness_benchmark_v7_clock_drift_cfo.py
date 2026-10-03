"""Synthetic v7 joint clock-drift and carrier-frequency-offset benchmark.

The v1-v6 modules are imported unchanged. This model is dimensionless and makes
no claim about a physical channel or receiver.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from digital_to_wave import WaveConfig, decode_wave
from robustness_benchmark_v2 import build_input_patterns, wilson_interval

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
BENCHMARK = "synthetic_clock_drift_cfo_v7"
STEM = "robustness-benchmark-v7-clock-drift-cfo"
MASTER_SEED = 20261010
VECTOR_SEED = 20261004
TRIALS_PER_VECTOR = 200
CHUNK_FRAMES = 32
NOISE_LEVELS = (0.0, 0.25, 0.45)
CFO_LEVELS = (-0.005, 0.0, 0.005)
CLOCK_LEVELS_PPM = (-2500.0, 0.0, 2500.0)
TIMING_OFFSETS = (-2.0, 0.0, 2.0)
SEARCH_CFO_LEVELS = (-0.01, -0.005, 0.0, 0.005, 0.01)
SEARCH_CLOCK_LEVELS_PPM = (-5000.0, -2500.0, 0.0, 2500.0, 5000.0)
SEARCH_TIMING_OFFSETS = (-4.0, -2.0, 0.0, 2.0, 4.0)
SAMPLES_PER_SYMBOL = 64
VECTOR_LENGTH = 32
GUARD_SAMPLES = 64
FRAME_SAMPLES = SAMPLES_PER_SYMBOL * VECTOR_LENGTH
BUFFER_SAMPLES = FRAME_SAMPLES + 2 * GUARD_SAMPLES
TOLERANCE = 0.25
NOMINAL = "nominal"
ORACLE = "oracle"
BLIND = "blind_joint_search"
METHODS = (NOMINAL, ORACLE, BLIND)
RESULT_FIELDS = (
    "cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples", "noise_std", "vector_pattern",
    "method", "frames", "frames_passing", "frame_pass_share", "wilson95_lower", "wilson95_upper",
    "empirical_mae", "empirical_rmse", "mean_bias", "max_abs_error", "timing_acquired_frames",
    "timing_acquisition_share", "acquisition_wilson95_lower", "acquisition_wilson95_upper",
    "joint_acquired_frames", "joint_acquisition_share", "joint_acquisition_wilson95_lower",
    "joint_acquisition_wilson95_upper",
)
PAIR_FIELDS = (
    "cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples", "noise_std", "vector_pattern",
    "method_a", "method_b", "frames", "a_pass_b_fail", "a_fail_b_pass", "mcnemar_exact_two_sided_p",
)
CONFIG = WaveConfig(SAMPLES_PER_SYMBOL, 4, 1.0)
OMEGA = 2.0 * math.pi * CONFIG.cycles_per_symbol / SAMPLES_PER_SYMBOL


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def received_buffer(
    values: Any,
    *,
    cfo_cycles_per_symbol: float = 0.0,
    clock_error_ppm: float = 0.0,
    timing_offset_samples: float = 0.0,
) -> np.ndarray:
    """Sample the continuous frame; + timing is late and + clock ppm advances source time faster."""
    vector = np.asarray(values, dtype=float)
    cfo = _finite(cfo_cycles_per_symbol, "cfo_cycles_per_symbol")
    ppm = _finite(clock_error_ppm, "clock_error_ppm")
    delay = _finite(timing_offset_samples, "timing_offset_samples")
    if vector.shape != (VECTOR_LENGTH,) or not np.all(np.isfinite(vector)) or np.any(np.abs(vector) > 2):
        raise ValueError("values must be a finite 32-vector inside [-2, 2]")
    if abs(ppm) > 100_000 or abs(delay) > GUARD_SAMPLES or abs(cfo) > 0.1:
        raise ValueError("impairment is outside the supported synthetic range")
    m = np.arange(BUFFER_SAMPLES, dtype=float) - GUARD_SAMPLES
    t = m * (1.0 + ppm * 1e-6) + delay
    symbol = np.floor(t / SAMPLES_PER_SYMBOL).astype(int)
    inside = (symbol >= 0) & (symbol < VECTOR_LENGTH)
    safe_symbol = np.clip(symbol, 0, VECTOR_LENGTH - 1)
    value = vector[safe_symbol]
    phase = np.where(value > 0, np.pi / 4, np.where(value < 0, 5 * np.pi / 4, 0.0))
    signal = np.abs(value) * np.sin((OMEGA + 2 * np.pi * cfo / SAMPLES_PER_SYMBOL) * t + phase)
    return np.where(inside, signal, 0.0)


def _candidate_decode(
    buffer: np.ndarray, timing: float, ppm: float, cfo: float
) -> np.ndarray:
    """Resample at candidate symbol times and project onto the candidate carrier."""
    b = np.asarray(buffer, dtype=float)
    if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] != BUFFER_SAMPLES or not np.all(np.isfinite(b)):
        raise ValueError("buffer must be a finite (frames, 2176) array")
    source = np.arange(FRAME_SAMPLES, dtype=float)
    sample_positions = GUARD_SAMPLES + (source - timing) / (1.0 + ppm * 1e-6)
    if sample_positions.min() < 0 or sample_positions.max() >= BUFFER_SAMPLES - 1:
        raise ValueError("candidate resampling positions exceed the guarded buffer")
    lo = np.floor(sample_positions).astype(int)
    fraction = sample_positions - lo
    samples = b[:, lo] * (1.0 - fraction)[None, :] + b[:, lo + 1] * fraction[None, :]
    phase = (OMEGA + 2.0 * math.pi * cfo / SAMPLES_PER_SYMBOL) * source + np.pi / 4.0
    reference = np.sin(phase).reshape(VECTOR_LENGTH, SAMPLES_PER_SYMBOL)
    windows = samples.reshape(b.shape[0], VECTOR_LENGTH, SAMPLES_PER_SYMBOL)
    denominator = np.sum(reference * reference, axis=1)
    return np.sum(windows * reference[None, :, :], axis=2) / denominator[None, :]


def blind_joint_search(buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose timing, clock ppm and CFO using only the received buffer."""
    b = np.asarray(buffer, dtype=float)
    if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] != BUFFER_SAMPLES or not np.all(np.isfinite(b)):
        raise ValueError("buffer must be a finite (frames, 2176) array")
    candidates = sorted(
        ((d, p, f) for d in SEARCH_TIMING_OFFSETS for p in SEARCH_CLOCK_LEVELS_PPM for f in SEARCH_CFO_LEVELS),
        key=lambda item: (abs(item[0]), abs(item[1]), abs(item[2]), item),
    )
    best_score = np.full(b.shape[0], -np.inf)
    best_parameters = np.zeros((b.shape[0], 3), dtype=float)
    best_estimates = np.zeros((b.shape[0], VECTOR_LENGTH), dtype=float)
    for timing, ppm, cfo in candidates:
        estimates = _candidate_decode(b, timing, ppm, cfo)
        score = np.sum(estimates * estimates, axis=1)
        better = score > best_score
        best_score[better] = score[better]
        best_parameters[better] = (timing, ppm, cfo)
        best_estimates[better] = estimates[better]
    return best_parameters[:, 0], best_parameters[:, 1], best_parameters[:, 2], best_estimates


def _mcnemar(a_pass_b_fail: int, a_fail_b_pass: int) -> float:
    total = a_pass_b_fail + a_fail_b_pass
    if total == 0:
        return 1.0
    minimum = min(a_pass_b_fail, a_fail_b_pass)
    if total < 1024:
        tail = sum(math.comb(total, i) for i in range(minimum + 1))
        return min(1.0, 2.0 * tail / (2**total))
    log_terms = [
        math.lgamma(total + 1) - math.lgamma(i + 1) - math.lgamma(total - i + 1) - total * math.log(2.0)
        for i in range(minimum + 1)
    ]
    peak = max(log_terms)
    log_probability = math.log(2.0) + peak + math.log(math.fsum(math.exp(term - peak) for term in log_terms))
    return min(1.0, math.exp(log_probability))


def _summary(
    errors: np.ndarray, timing_acquired: np.ndarray | None, joint_acquired: np.ndarray | None,
    *, include_intervals: bool,
) -> dict[str, Any]:
    absolute = np.abs(errors)
    passed = np.max(absolute, axis=1) <= TOLERANCE
    frames, count = int(errors.shape[0]), int(np.count_nonzero(passed))
    if include_intervals and frames > 1:
        low, high = wilson_interval(count, frames)
    else:
        low = high = None
    if timing_acquired is None:
        acq_count = acq_share = acq_low = acq_high = None
    else:
        acq_count = int(np.count_nonzero(timing_acquired))
        acq_share = acq_count / frames
        acq_low, acq_high = wilson_interval(acq_count, frames) if include_intervals and frames > 1 else (None, None)
    if joint_acquired is None:
        joint_count = joint_share = joint_low = joint_high = None
    else:
        joint_count = int(np.count_nonzero(joint_acquired))
        joint_share = joint_count / frames
        joint_low, joint_high = (
            wilson_interval(joint_count, frames) if include_intervals and frames > 1 else (None, None)
        )
    return {
        "frames": frames,
        "frames_passing": count,
        "frame_pass_share": count / frames,
        "wilson95_lower": low,
        "wilson95_upper": high,
        "empirical_mae": float(np.mean(absolute)),
        "empirical_rmse": float(np.sqrt(np.mean(errors * errors))),
        "mean_bias": float(np.mean(errors)),
        "max_abs_error": float(np.max(absolute)),
        "timing_acquired_frames": acq_count,
        "timing_acquisition_share": acq_share,
        "acquisition_wilson95_lower": acq_low,
        "acquisition_wilson95_upper": acq_high,
        "joint_acquired_frames": joint_count,
        "joint_acquisition_share": joint_share,
        "joint_acquisition_wilson95_lower": joint_low,
        "joint_acquisition_wilson95_upper": joint_high,
    }


def module_hashes() -> dict[str, str]:
    names = (
        "digital_to_wave.py", "robustness_benchmark_v2.py", "robustness_benchmark_v3.py",
        "robustness_benchmark_v4.py", "robustness_benchmark_v5_timing.py",
        "robustness_benchmark_v6_timing_recovery.py", Path(__file__).name,
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_tracked_tree_clean_at_run_start": None}
    return {"git_commit": commit, "git_tracked_tree_clean_at_run_start": status == ""}


def run_benchmark(
    *,
    trials_per_vector: int = TRIALS_PER_VECTOR,
    master_seed: int = MASTER_SEED,
    cfo_levels: Iterable[float] = CFO_LEVELS,
    clock_levels_ppm: Iterable[float] = CLOCK_LEVELS_PPM,
    timing_offsets: Iterable[float] = TIMING_OFFSETS,
    noise_levels: Iterable[float] = NOISE_LEVELS,
    chunk_frames: int = CHUNK_FRAMES,
) -> dict[str, Any]:
    trials = _positive_int(trials_per_vector, "trials_per_vector")
    chunk = _positive_int(chunk_frames, "chunk_frames")
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(master_seed, (int, np.integer)):
        raise ValueError("master_seed must be an integer")
    cfo_values = tuple(_finite(x, "CFO") for x in cfo_levels)
    ppm_values = tuple(_finite(x, "clock error") for x in clock_levels_ppm)
    timing_values = tuple(_finite(x, "timing offset") for x in timing_offsets)
    sigmas = tuple(_finite(x, "noise level") for x in noise_levels)
    if any(not x for x in (cfo_values, ppm_values, timing_values, sigmas)):
        raise ValueError("all condition grids must be non-empty")
    if any(x < 0 for x in sigmas):
        raise ValueError("noise levels must be non-negative")
    if len(set(cfo_values)) != len(cfo_values) or len(set(ppm_values)) != len(ppm_values):
        raise ValueError("CFO and clock grids must be unique")
    if len(set(timing_values)) != len(timing_values) or len(set(sigmas)) != len(sigmas):
        raise ValueError("timing and noise grids must be unique")
    if any(abs(x) > 0.1 for x in cfo_values) or any(abs(x) > 100_000 for x in ppm_values):
        raise ValueError("CFO or clock grid is outside the supported synthetic range")
    if any(abs(x) > GUARD_SAMPLES for x in timing_values):
        raise ValueError("timing grid exceeds the guard length")
    if int(master_seed) in {MASTER_SEED, VECTOR_SEED, 20261003, 20261005, 20261006, 20261007, 20261009}:
        if int(master_seed) != MASTER_SEED:
            raise ValueError("master seed must differ from prior protocol seeds")
    patterns = build_input_patterns()
    cell_specs = [(f, p, d, s) for f in cfo_values for p in ppm_values for d in timing_values for s in sigmas]
    result_rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for cell_index, (cfo, ppm, delay, sigma) in enumerate(cell_specs):
        vectors_for_pool: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
        passes_for_pool: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
        timing_for_pool: list[np.ndarray] = []
        joint_for_pool: list[np.ndarray] = []
        for vector_index, (name, reference) in enumerate(patterns.items()):
            n_trials = 1 if sigma == 0 else trials
            rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), cell_index, vector_index]))
            error_chunks: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
            pass_chunks: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
            timing_acquisition_chunks: list[np.ndarray] = []
            joint_acquisition_chunks: list[np.ndarray] = []
            for start in range(0, n_trials, chunk):
                batch = min(chunk, n_trials - start)
                clean = received_buffer(
                    reference, cfo_cycles_per_symbol=cfo, clock_error_ppm=ppm, timing_offset_samples=delay
                )
                buffers = np.broadcast_to(clean, (batch, BUFFER_SAMPLES)).copy()
                if sigma > 0:
                    buffers += rng.normal(0.0, sigma, size=buffers.shape)
                nominal_windows = buffers[:, GUARD_SAMPLES:GUARD_SAMPLES + FRAME_SAMPLES]
                nominal = decode_wave(nominal_windows.reshape(-1, SAMPLES_PER_SYMBOL), CONFIG).reshape(
                    batch, VECTOR_LENGTH
                )
                oracle = _candidate_decode(buffers, delay, ppm, cfo)
                chosen_d, chosen_p, chosen_f, blind = blind_joint_search(buffers)
                estimates = {NOMINAL: nominal, ORACLE: oracle, BLIND: blind}
                for method, decoded in estimates.items():
                    errors = decoded - reference[None, :]
                    error_chunks[method].append(errors)
                    pass_chunks[method].append(np.max(np.abs(errors), axis=1) <= TOLERANCE)
                timing_acquisition_chunks.append(np.abs(chosen_d - delay) <= 2.0)
                joint_acquisition_chunks.append(
                    (np.abs(chosen_d - delay) <= 2.0)
                    & (np.abs(chosen_p - ppm) <= 2500.0)
                    & (np.abs(chosen_f - cfo) <= 0.005)
                )
            errors_by_method = {m: np.concatenate(v) for m, v in error_chunks.items()}
            pass_by_method = {m: np.concatenate(v) for m, v in pass_chunks.items()}
            timing_acquired = np.concatenate(timing_acquisition_chunks)
            joint_acquired = np.concatenate(joint_acquisition_chunks)
            timing_for_pool.append(timing_acquired)
            joint_for_pool.append(joint_acquired)
            for method in METHODS:
                stats = _summary(
                    errors_by_method[method],
                    timing_acquired if method == BLIND else None,
                    joint_acquired if method == BLIND else None,
                    include_intervals=sigma > 0,
                )
                result_rows.append({
                    "cfo_cycles_per_symbol": cfo, "clock_error_ppm": ppm, "timing_offset_samples": delay,
                    "noise_std": sigma, "vector_pattern": name, "method": method, **stats,
                })
                vectors_for_pool[method].append(errors_by_method[method])
                passes_for_pool[method].append(pass_by_method[method])
            for method_a, method_b in ((BLIND, ORACLE), (BLIND, NOMINAL), (ORACLE, NOMINAL)):
                a, b = pass_by_method[method_a], pass_by_method[method_b]
                paired.append({
                    "cfo_cycles_per_symbol": cfo, "clock_error_ppm": ppm, "timing_offset_samples": delay,
                    "noise_std": sigma, "vector_pattern": name, "method_a": method_a, "method_b": method_b,
                    "frames": int(a.size), "a_pass_b_fail": int(np.count_nonzero(a & ~b)),
                    "a_fail_b_pass": int(np.count_nonzero(~a & b)),
                    "mcnemar_exact_two_sided_p": _mcnemar(int(np.count_nonzero(a & ~b)), int(np.count_nonzero(~a & b))),
                })
        pooled_passes = {m: np.concatenate(v) for m, v in passes_for_pool.items()}
        pooled_errors = {m: np.concatenate(v) for m, v in vectors_for_pool.items()}
        for method in METHODS:
            stats = _summary(
                pooled_errors[method],
                np.concatenate(timing_for_pool) if method == BLIND else None,
                np.concatenate(joint_for_pool) if method == BLIND else None,
                include_intervals=sigma > 0,
            )
            stats["frames_passing"] = int(np.count_nonzero(pooled_passes[method]))
            stats["frame_pass_share"] = stats["frames_passing"] / stats["frames"]
            if sigma > 0 and stats["frames"] > 1:
                stats["wilson95_lower"], stats["wilson95_upper"] = wilson_interval(
                    stats["frames_passing"], stats["frames"]
                )
            result_rows.append({
                "cfo_cycles_per_symbol": cfo, "clock_error_ppm": ppm, "timing_offset_samples": delay,
                "noise_std": sigma, "vector_pattern": "POOLED_ALL_VECTORS", "method": method, **stats,
            })
        for method_a, method_b in ((BLIND, ORACLE), (BLIND, NOMINAL), (ORACLE, NOMINAL)):
            a, b = pooled_passes[method_a], pooled_passes[method_b]
            paired.append({
                "cfo_cycles_per_symbol": cfo, "clock_error_ppm": ppm, "timing_offset_samples": delay,
                "noise_std": sigma, "vector_pattern": "POOLED_ALL_VECTORS", "method_a": method_a,
                "method_b": method_b, "frames": int(a.size), "a_pass_b_fail": int(np.count_nonzero(a & ~b)),
                "a_fail_b_pass": int(np.count_nonzero(~a & b)),
                "mcnemar_exact_two_sided_p": _mcnemar(int(np.count_nonzero(a & ~b)), int(np.count_nonzero(~a & b))),
            })
    hashes = module_hashes()
    contract = {
        "classification": "synthetic dimensionless benchmark; no physical validity claimed",
        "protocol_status": "exploratory v7 protocol fixed before full run; not formal preregistration",
        "protocol_lock": "reports/v7-clock-drift-cfo-protocol-lock.md",
        "master_seed": int(master_seed), "vector_seed": VECTOR_SEED, "trials_per_vector": trials,
        "full_protocol_run": (
            trials == TRIALS_PER_VECTOR and int(master_seed) == MASTER_SEED
            and cfo_values == CFO_LEVELS and ppm_values == CLOCK_LEVELS_PPM
            and timing_values == TIMING_OFFSETS and sigmas == NOISE_LEVELS
        ),
        "cfo_cycles_per_symbol": list(cfo_values), "clock_error_ppm": list(ppm_values),
        "timing_offset_samples": list(timing_values), "noise_std": list(sigmas),
        "blind_search_cfo": list(SEARCH_CFO_LEVELS), "blind_search_clock_ppm": list(SEARCH_CLOCK_LEVELS_PPM),
        "blind_search_timing_samples": list(SEARCH_TIMING_OFFSETS), "tolerance": TOLERANCE,
        "symbol_samples": SAMPLES_PER_SYMBOL, "frame_symbols": VECTOR_LENGTH,
        "clock_drift_at_2500ppm_samples_per_frame": FRAME_SAMPLES * 2500e-6,
        "model": "t=(m-guard)*(1+clock_error_ppm*1e-6)+timing_offset; carrier adds CFO cycles/symbol",
        "receivers": {
            NOMINAL: "unchanged decode_wave with fixed windows and no correction",
            ORACLE: "exact impairment parameters; linear resampling and least-squares carrier projection",
            BLIND: "joint bounded grid search maximizing sum of squared decoded values",
        },
        "acquisition_rule": (
            "all estimates within one grid step: timing 2 samples, clock 2500 ppm, "
            "CFO 0.005 cycles/symbol"
        ),
        "paired_methods": "all receivers use identical noisy buffers; exact two-sided McNemar tests",
        "vector_patterns": {k: np.asarray(v).tolist() for k, v in patterns.items()},
        "module_sha256": hashes, **git_provenance(),
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        },
        "limitations": [
            "synthetic model only; no physical claim",
            "known frame and symbol lengths, known waveform family and noise-only guards",
            "static CFO and clock error, linear interpolation, no pulse shaping, jitter, filtering "
            "or time-varying impairments",
            "oracle is informed of impairments; blind search is limited to the declared grid",
        ],
    }
    result = {"benchmark": BENCHMARK, "contract": contract, "results": result_rows, "paired_comparisons": paired}
    validate_result(result)
    return result


def validate_result(result: dict[str, Any]) -> None:
    if result.get("benchmark") != BENCHMARK:
        raise ValueError("unexpected benchmark identifier")
    contract, rows, paired = result.get("contract"), result.get("results"), result.get("paired_comparisons")
    if not isinstance(contract, dict) or not isinstance(rows, list) or not rows or not isinstance(paired, list):
        raise ValueError("result requires a contract, result rows and paired comparisons")
    if contract.get("protocol_lock") != "reports/v7-clock-drift-cfo-protocol-lock.md":
        raise ValueError("protocol lock reference is missing")
    if contract.get("classification", "").find("no physical validity claimed") < 0:
        raise ValueError("synthetic/no-physical-validity label is required")
    expected_hashes = module_hashes()
    if contract.get("module_sha256") != expected_hashes:
        raise ValueError("module hashes do not match the current source files")
    keys = set()
    for row in rows:
        if any(field not in row for field in RESULT_FIELDS):
            raise ValueError("result row is missing required fields")
        key = tuple(row[k] for k in ("cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples",
                                    "noise_std", "vector_pattern", "method"))
        if key in keys:
            raise ValueError("duplicate result row")
        keys.add(key)
        if row["method"] not in METHODS or row["frames"] < 1:
            raise ValueError("invalid method or frame count")
        if not 0 <= row["frames_passing"] <= row["frames"]:
            raise ValueError("invalid pass count")
        if not math.isclose(row["frame_pass_share"], row["frames_passing"] / row["frames"], abs_tol=1e-12):
            raise ValueError("frame pass share does not agree with counts")
        if not all(math.isfinite(float(row[field])) for field in
                   ("empirical_mae", "empirical_rmse", "mean_bias", "max_abs_error")):
            raise ValueError("metrics must be finite")
    for row in paired:
        if row["a_pass_b_fail"] + row["a_fail_b_pass"] > row["frames"]:
            raise ValueError("paired discordant count exceeds frame count")


def write_outputs(result: dict[str, Any], output_dir: Path = ARTIFACTS) -> dict[str, Path]:
    validate_result(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": output_dir / f"{STEM}.json", "csv": output_dir / f"{STEM}.csv",
             "paired_csv": output_dir / f"{STEM}-paired.csv", "png": output_dir / f"{STEM}.png"}
    paths["json"].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["results"])
    with paths["paired_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["paired_comparisons"])
    pooled = [r for r in result["results"] if r["vector_pattern"] == "POOLED_ALL_VECTORS" and r["noise_std"] == 0.45]
    conditions = sorted(set((r["cfo_cycles_per_symbol"], r["clock_error_ppm"], r["timing_offset_samples"])
                            for r in pooled))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method, marker in ((NOMINAL, "o"), (ORACLE, "s"), (BLIND, "D")):
        selected = [next(r for r in pooled if (r["cfo_cycles_per_symbol"], r["clock_error_ppm"],
                                                r["timing_offset_samples"]) == condition and r["method"] == method)
                    for condition in conditions]
        axes[0].plot(range(len(conditions)), [100 * r["frame_pass_share"] for r in selected],
                     marker=marker, ms=3, lw=0.8, label=method)
    blind_rows = [next(r for r in pooled if (r["cfo_cycles_per_symbol"], r["clock_error_ppm"],
                                              r["timing_offset_samples"]) == condition and r["method"] == BLIND)
                  for condition in conditions]
    axes[1].plot(range(len(conditions)), [100 * r["timing_acquisition_share"] for r in blind_rows],
                 marker="o", ms=3, lw=0.8, label="timing acquired")
    axes[1].plot(range(len(conditions)), [100 * r["joint_acquisition_share"] for r in blind_rows],
                 marker="D", ms=3, lw=0.8, label="joint timing + CFO + clock acquired")
    axes[0].set(ylabel="frames passing (%)", xlabel="CFO / clock / timing condition",
                title="Decode success (σ = 0.45)", ylim=(-2, 102))
    axes[1].set(ylabel="blind acquisition (%)", xlabel="CFO / clock / timing condition",
                title="Synchronization acquisition (σ = 0.45)", ylim=(-2, 102))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("v7 joint clock-drift + CFO — SYNTHETIC; no physical validity claimed")
    fig.savefig(paths["png"], dpi=150, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return paths


def main() -> int:
    result = run_benchmark()
    paths = write_outputs(result)
    pooled = [r for r in result["results"] if r["vector_pattern"] == "POOLED_ALL_VECTORS" and r["noise_std"] == 0.45]
    for row in pooled:
        if row["cfo_cycles_per_symbol"] == 0 and row["clock_error_ppm"] == 0 and row["timing_offset_samples"] == 0:
            print(f"no-impairment {row['method']}: {row['frames_passing']}/{row['frames']} passing")
    print(f"wrote {paths['json']}, {paths['csv']}, {paths['paired_csv']} and {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
