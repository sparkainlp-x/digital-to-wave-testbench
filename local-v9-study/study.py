#!/usr/bin/env python3
"""Local-only v9 generalization study for the Digital-to-Wave synthetic testbench.

The v1/v7/v8 implementation snapshots in compat/ are unmodified and hash-locked.
This study evaluates generated dimensionless signals only; it makes no physical,
biological, quantum, diamond, hardware, or real-channel claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
COMPAT = HERE / "compat"
sys.path.insert(0, str(COMPAT))
import robustness_benchmark_v7_clock_drift_cfo as v7  # noqa: E402
import robustness_benchmark_v8_ml_sync as v8  # noqa: E402

LOCK_PATH = HERE / "lock.json"
CONFIG_PATH = HERE / "study_config.json"
RESULTS_DIR = HERE / "results"
METHODS = (
    "v7_blind",
    "score_only_ablation",
    "refinement_only_ablation",
    "decoder_only_ablation",
    "v8_full",
    "raw_ls_informed_oracle",
)
BLIND_METHODS = METHODS[:-1]
FRAME_FIELDS = (
    "condition", "cluster_id", "family", "channel_seed_index", "frame_index", "method",
    "buffer_sha256", "frame_pass", "max_abs_error", "mae", "rmse",
    "estimated_timing_samples", "estimated_clock_ppm", "estimated_cfo_cycles_per_symbol",
    "abs_timing_error_samples", "abs_clock_error_ppm", "abs_cfo_error_cycles_per_symbol",
    "sync_acquired_loose", "sync_acquired_strict",
)
SUMMARY_FIELDS = (
    "condition", "method", "clusters", "frames", "frames_passing", "frame_pass_share",
    "cluster_bootstrap95_low", "cluster_bootstrap95_high", "mean_abs_error",
    "mean_abs_timing_error_samples", "mean_abs_clock_error_ppm", "mean_abs_cfo_error_cycles_per_symbol",
    "sync_acquired_loose_frames", "sync_acquired_loose_share", "sync_loose_cluster_bootstrap95_low",
    "sync_loose_cluster_bootstrap95_high", "sync_acquired_strict_frames", "sync_acquired_strict_share",
    "sync_strict_cluster_bootstrap95_low", "sync_strict_cluster_bootstrap95_high",
    "runtime_total_seconds", "runtime_mean_ms_per_frame", "runtime_call_p50_ms", "runtime_call_p95_ms",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def verify_lock() -> dict[str, Any]:
    """Fail closed if the preregistered source snapshot or config was edited."""
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for relpath, expected in lock["source_sha256"].items():
        actual = sha256_file(HERE / relpath)
        if actual != expected:
            raise RuntimeError(f"Frozen source mismatch for {relpath}: expected {expected}, got {actual}")
    if lock.get("lock_status") != "frozen_before_full_run":
        raise RuntimeError("Protocol lock is not marked frozen_before_full_run")
    return lock


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rng(seed_parts: list[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed_parts))


def make_payload(family: str, rng: np.random.Generator) -> np.ndarray:
    """Draw one unseen 32-value payload from one of twelve prespecified families."""
    n = v7.VECTOR_LENGTH
    if family == "uniform_continuous":
        x = rng.uniform(-2.0, 2.0, n)
    elif family == "left_skew_beta":
        x = 4.0 * rng.beta(2.0, 5.0, n) - 2.0
    elif family == "sparse_zero_mixture":
        x = rng.uniform(-2.0, 2.0, n)
        x[rng.random(n) < 0.55] = 0.0
    elif family == "markov_run_lengths":
        x = np.empty(n, dtype=float)
        x[0] = rng.uniform(-2.0, 2.0)
        for i in range(1, n):
            if rng.random() < 0.22:
                x[i] = rng.uniform(-2.0, 2.0)
            else:
                x[i] = np.clip(x[i - 1] + rng.normal(0.0, 0.30), -2.0, 2.0)
    elif family == "clipped_student_t":
        x = np.clip(rng.standard_t(df=3, size=n) * 0.85, -2.0, 2.0)
    elif family == "near_zero_uniform":
        x = rng.uniform(-0.55, 0.55, n)
    elif family == "extreme_magnitude_mixture":
        large = rng.random(n) < 0.70
        mag = np.where(large, rng.uniform(1.15, 2.0, n), rng.uniform(0.0, 0.50, n))
        x = rng.choice(np.array([-1.0, 1.0]), n) * mag
    elif family == "nonuniform_five_level":
        x = rng.choice(np.array([-2.0, -0.65, 0.0, 0.65, 2.0]), n,
                       p=np.array([0.10, 0.20, 0.40, 0.20, 0.10]))
    elif family == "three_mode_mixture":
        centers = rng.choice(np.array([-1.45, 0.0, 1.45]), n, p=np.array([0.35, 0.30, 0.35]))
        x = np.clip(centers + rng.normal(0.0, 0.20, n), -2.0, 2.0)
    elif family == "balanced_sign_magnitudes":
        signs = np.array([-1.0] * (n // 2) + [1.0] * (n - n // 2))
        rng.shuffle(signs)
        x = signs * rng.uniform(0.08, 2.0, n)
    elif family == "sinusoidal_envelope_jitter":
        phase = rng.uniform(-math.pi, math.pi)
        x = 1.55 * np.sin(2.0 * math.pi * np.arange(n) / n + phase) + rng.normal(0.0, 0.24, n)
        x = np.clip(x, -2.0, 2.0)
    elif family == "right_skew_beta":
        x = 4.0 * rng.beta(5.0, 2.0, n) - 2.0
    else:
        raise ValueError(f"Unknown payload family: {family}")
    result = np.asarray(x, dtype=float)
    if result.shape != (n,) or not np.all(np.isfinite(result)) or np.any(np.abs(result) > 2.0):
        raise AssertionError("Payload generator violated the frozen 32-value [-2,2] contract")
    return result


def _variable_signal(values: np.ndarray, condition: dict[str, Any]) -> np.ndarray:
    """Generate a frame with a linear-in-frame CFO and clock-rate trajectory."""
    m = np.arange(v7.BUFFER_SAMPLES, dtype=float) - v7.GUARD_SAMPLES
    tau = float(condition["timing_samples"])
    clock_start = float(condition.get("tv_clock_start_ppm", 0.0))
    clock_end = float(condition.get("tv_clock_end_ppm", 0.0))
    cfo_start = float(condition.get("tv_cfo_start", 0.0))
    cfo_end = float(condition.get("tv_cfo_end", 0.0))
    clock_delta = clock_end - clock_start
    # Derivative d(source_time)/d(received_sample) moves linearly from start to end.
    t = m * (1.0 + clock_start * 1e-6) + 0.5 * clock_delta * 1e-6 * (m * m / v7.FRAME_SAMPLES) + tau
    q = t / v7.SAMPLES_PER_SYMBOL
    symbol = np.floor(q).astype(int)
    inside = (symbol >= 0) & (symbol < v7.VECTOR_LENGTH)
    safe_symbol = np.clip(symbol, 0, v7.VECTOR_LENGTH - 1)
    value = values[safe_symbol]
    symbol_phase = np.where(value > 0, math.pi / 4.0, np.where(value < 0, 5.0 * math.pi / 4.0, 0.0))
    # Integrate a linearly changing extra CFO (cycles/symbol) over symbol-time q.
    cfo_delta = cfo_end - cfo_start
    extra_cycles = cfo_start * q + cfo_delta * q * q / (2.0 * v7.VECTOR_LENGTH)
    phase = 2.0 * math.pi * (v7.CONFIG.cycles_per_symbol * q + extra_cycles) + symbol_phase
    signal = np.abs(value) * np.sin(phase)
    return np.where(inside, signal, 0.0)


def _colored_noise(rng: np.random.Generator, size: int, marginal_std: float, rho: float = 0.80) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size)
    out = np.empty(size, dtype=float)
    out[0] = marginal_std * white[0]
    innovation_scale = marginal_std * math.sqrt(1.0 - rho * rho)
    for i in range(1, size):
        out[i] = rho * out[i - 1] + innovation_scale * white[i]
    return out


def make_received_buffer(values: np.ndarray, condition: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """One reproducible synthetic channel realization, with no method-specific randomness."""
    varying = (float(condition.get("tv_cfo_start", 0.0)) != float(condition.get("tv_cfo_end", 0.0)) or
               float(condition.get("tv_clock_start_ppm", 0.0)) != float(condition.get("tv_clock_end_ppm", 0.0)))
    if varying:
        signal = _variable_signal(values, condition)
    else:
        signal = v7.received_buffer(
            values,
            timing_offset_samples=float(condition["timing_samples"]),
            clock_error_ppm=float(condition["clock_ppm"]),
            cfo_cycles_per_symbol=float(condition["cfo_cycles_per_symbol"]),
        )
    signal = np.asarray(signal, dtype=float)
    if bool(condition.get("filter", False)):
        taps = np.array([0.82, 0.0, 0.0, 0.34], dtype=float)
        taps /= np.linalg.norm(taps)
        signal = np.convolve(signal, taps, mode="full")[:v7.BUFFER_SAMPLES]
    colored_std = float(condition.get("colored_noise_std", 0.0))
    if colored_std > 0.0:
        signal = signal + _colored_noise(rng, signal.size, colored_std)
    impulse_p = float(condition.get("impulsive_noise_probability", 0.0))
    if impulse_p > 0.0:
        background = float(condition.get("impulse_background_std", 0.0))
        impulse_std = float(condition["impulse_amplitude_std"])
        signal = signal + rng.normal(0.0, background, signal.size)
        mask = rng.random(signal.size) < impulse_p
        signal = signal + mask * rng.normal(0.0, impulse_std, signal.size)
    if signal.shape != (v7.BUFFER_SAMPLES,) or not np.all(np.isfinite(signal)):
        raise AssertionError("Channel generator returned an invalid buffer")
    return signal


def _candidate_grid() -> np.ndarray:
    return np.asarray(sorted(
        ((d, p, f) for d in v7.SEARCH_TIMING_OFFSETS
         for p in v7.SEARCH_CLOCK_LEVELS_PPM for f in v7.SEARCH_CFO_LEVELS),
        key=lambda item: (abs(item[0]), abs(item[1]), abs(item[2]), item),
    ), dtype=float)


def _score_per_frame(buffers: np.ndarray, params: np.ndarray, metric: str) -> np.ndarray:
    scores = np.full(buffers.shape[0], -np.inf, dtype=float)
    inside = ((np.abs(params[:, 0]) <= v8.TIMING_BOUND + 1e-9) &
              (np.abs(params[:, 1]) <= v8.CLOCK_BOUND_PPM + 1e-6) &
              (np.abs(params[:, 2]) <= v8.CFO_BOUND + 1e-12))
    if not np.any(inside):
        return scores
    rows = np.flatnonzero(inside)
    unique, inverse = np.unique(params[inside], axis=0, return_inverse=True)
    for group, candidate in enumerate(unique):
        members = rows[inverse == group]
        if metric == "glrt":
            scores[members] = v8.ml_score(buffers[members], *candidate)
        elif metric == "v7_energy":
            decoded = v7._candidate_decode(buffers[members], *candidate)
            scores[members] = np.sum(decoded * decoded, axis=1)
        else:
            raise ValueError(f"Unknown candidate metric: {metric}")
    return scores


def _decode_v7(buffers: np.ndarray, params: np.ndarray) -> np.ndarray:
    decoded = np.empty((buffers.shape[0], v7.VECTOR_LENGTH), dtype=float)
    unique, inverse = np.unique(params, axis=0, return_inverse=True)
    for group, candidate in enumerate(unique):
        members = np.flatnonzero(inverse == group)
        decoded[members] = v7._candidate_decode(buffers[members], *candidate)
    return decoded


def _ablation_search(buffers: np.ndarray, metric: str, refine: bool) -> tuple[np.ndarray, np.ndarray]:
    """Switch one search component while holding the other receiver components fixed."""
    grid = _candidate_grid()
    coarse = np.stack([_score_per_frame(buffers, np.broadcast_to(c, (buffers.shape[0], 3)), metric)
                       for c in grid], axis=1)
    order = np.argsort(-coarse, axis=1, kind="stable")
    chosen = grid[order[:, 0]].copy()
    if refine:
        best_score = np.full(buffers.shape[0], -np.inf)
        best_params = np.zeros((buffers.shape[0], 3), dtype=float)
        row_index = np.arange(buffers.shape[0])
        offsets = v8._ordered_offsets()
        for rank in range(min(v8.TOP_K, order.shape[1])):
            start = grid[order[:, rank]]
            center = np.column_stack((start[:, 0], start[:, 1], v8.carrier_offset(start[:, 1], start[:, 2])))
            params = start.copy()
            score = coarse[row_index, order[:, rank]]
            step = np.asarray(v8.REFINE_STEPS, dtype=float)
            for _ in range(v8.REFINE_LEVELS):
                new_center, new_params, new_score = center.copy(), params.copy(), score.copy()
                for offset in offsets:
                    if not np.any(offset):
                        continue
                    ridge = center + offset * step
                    candidate = np.column_stack((ridge[:, 0], ridge[:, 1],
                                                 v8.cfo_from_carrier(ridge[:, 1], ridge[:, 2])))
                    candidate_score = _score_per_frame(buffers, candidate, metric)
                    better = candidate_score > new_score + v8.SCORE_MARGIN * np.maximum(1.0, np.abs(new_score))
                    new_center[better] = ridge[better]
                    new_params[better] = candidate[better]
                    new_score[better] = candidate_score[better]
                center, params, score = new_center, new_params, new_score
                step /= 2.0
            if rank == 0:
                better_rank = np.ones(buffers.shape[0], dtype=bool)
            else:
                better_rank = score > best_score + v8.SCORE_MARGIN * np.maximum(1.0, np.abs(best_score))
            best_params[better_rank] = params[better_rank]
            best_score[better_rank] = score[better_rank]
        chosen = best_params
    decoded = _decode_v7(buffers, chosen)
    return chosen, decoded


def run_receiver(method: str, buffers: np.ndarray, truth: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    if method == "v7_blind":
        d, p, f, decoded = v7.blind_joint_search(buffers)
        return np.column_stack((d, p, f)), decoded
    if method == "score_only_ablation":
        return _ablation_search(buffers, metric="glrt", refine=False)
    if method == "refinement_only_ablation":
        return _ablation_search(buffers, metric="v7_energy", refine=True)
    if method == "decoder_only_ablation":
        d, p, f, _ = v7.blind_joint_search(buffers)
        params = np.column_stack((d, p, f))
        decoded = v8._decode_per_frame(buffers, params)
        return params, decoded
    if method == "v8_full":
        d, p, f, decoded = v8.v8_blind_search(buffers)
        return np.column_stack((d, p, f)), decoded
    if method == "raw_ls_informed_oracle":
        params = np.broadcast_to(np.asarray(truth, dtype=float), (buffers.shape[0], 3)).copy()
        return params, v8._decode_per_frame(buffers, params)
    raise ValueError(f"Unknown receiver method: {method}")


def _cluster_bootstrap(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    """Percentile interval resampling independent family x channel-seed clusters."""
    cluster_values = np.asarray(values, dtype=float).reshape(-1)
    if cluster_values.size == 0:
        return float("nan"), float("nan")
    if cluster_values.size == 1:
        value = float(cluster_values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, cluster_values.size, size=(replicates, cluster_values.size))
    means = np.mean(cluster_values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. This study runs on Linux, but keep conversion explicit.
    return float(peak / 1024.0 if sys.platform.startswith("linux") else peak / (1024.0 * 1024.0))


def _runtime_summary(samples: list[float], frames: int) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    total = float(np.sum(arr))
    return {
        "runtime_total_seconds": total,
        "runtime_mean_ms_per_frame": total * 1000.0 / max(1, frames),
        "runtime_call_p50_ms": float(np.quantile(arr, 0.50) * 1000.0),
        "runtime_call_p95_ms": float(np.quantile(arr, 0.95) * 1000.0),
    }


def _sync_mask(params: np.ndarray, truth: tuple[float, float, float], tolerance: dict[str, float]) -> np.ndarray:
    error = np.abs(params - np.asarray(truth, dtype=float)[None, :])
    limits = np.array([tolerance["timing_samples"], tolerance["clock_ppm"], tolerance["cfo_cycles_per_symbol"]])
    return np.all(error <= limits[None, :] * (1.0 + 1e-9), axis=1)


def run_study(config: dict[str, Any] | None = None, *, write_outputs: bool = False) -> dict[str, Any]:
    """Execute all paired methods on same generated buffers and summarize by cluster."""
    cfg = load_config() if config is None else config
    families = list(cfg["payload_families"])
    seeds_per_family = int(cfg["independent_channel_seed_replicates_per_family"])
    frames_per_cluster = int(cfg["frames_per_family_channel_cluster"])
    bootstrap_replicates = int(cfg["bootstrap_replicates"])
    if not families or seeds_per_family < 1 or frames_per_cluster < 1 or bootstrap_replicates < 100:
        raise ValueError("Study configuration has invalid sample counts")
    conditions = list(cfg["condition_design"])
    methods = tuple(cfg["receiver_methods"])
    if methods != METHODS:
        raise ValueError("Receiver method order does not match frozen implementation")
    n_clusters = len(families) * seeds_per_family
    frame_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    runtime_samples: dict[tuple[str, str], list[float]] = {(c["name"], m): [] for c in conditions for m in methods}
    runtime_frames: dict[tuple[str, str], int] = {(c["name"], m): 0 for c in conditions for m in methods}
    buffer_hash_by_condition_cluster: dict[tuple[str, str], str] = {}
    seed_catalog: list[dict[str, Any]] = []

    for condition_index, condition in enumerate(conditions):
        name = str(condition["name"])
        truth = (float(condition["timing_samples"]), float(condition["clock_ppm"]),
                 float(condition["cfo_cycles_per_symbol"]))
        outcomes: dict[str, np.ndarray] = {
            m: np.zeros((n_clusters, frames_per_cluster), dtype=bool) for m in methods
        }
        loose: dict[str, np.ndarray] = {
            m: np.zeros((n_clusters, frames_per_cluster), dtype=bool) for m in BLIND_METHODS
        }
        strict: dict[str, np.ndarray] = {
            m: np.zeros((n_clusters, frames_per_cluster), dtype=bool) for m in BLIND_METHODS
        }
        abs_sync: dict[str, np.ndarray] = {
            m: np.zeros((n_clusters, frames_per_cluster, 3), dtype=float) for m in BLIND_METHODS
        }
        abs_error_mean: dict[str, np.ndarray] = {
            m: np.zeros((n_clusters, frames_per_cluster), dtype=float) for m in methods
        }
        cluster_index = 0
        for family_index, family in enumerate(families):
            for channel_index in range(seeds_per_family):
                cluster_id = f"F{family_index + 1:02d}-C{channel_index + 1:02d}"
                payload_frames: list[np.ndarray] = []
                received_frames: list[np.ndarray] = []
                if condition_index == 0:
                    seed_catalog.append({
                        "cluster_id": cluster_id, "family": family,
                        "payload_seed_parts": [int(cfg["payload_seed"]), family_index, channel_index],
                        "channel_seed_parts": [int(cfg["channel_seed"]), family_index, channel_index],
                    })
                for frame_index in range(frames_per_cluster):
                    payload_rng = _rng([int(cfg["payload_seed"]), family_index, channel_index, frame_index])
                    values = make_payload(family, payload_rng)
                    channel_rng = _rng([int(cfg["channel_seed"]), family_index, channel_index,
                                        condition_index, frame_index])
                    received = make_received_buffer(values, condition, channel_rng)
                    payload_frames.append(values)
                    received_frames.append(received)
                buffers = np.stack(received_frames, axis=0)
                payloads = np.stack(payload_frames, axis=0)
                buffer_hash = sha256_bytes(np.ascontiguousarray(buffers, dtype="<f8").tobytes())
                buffer_hash_by_condition_cluster[(name, cluster_id)] = buffer_hash
                for method in methods:
                    started = time.perf_counter()
                    params, decoded = run_receiver(method, buffers, truth)
                    elapsed = time.perf_counter() - started
                    params = np.asarray(params, dtype=float)
                    decoded = np.asarray(decoded, dtype=float)
                    if params.shape != (frames_per_cluster, 3) or decoded.shape != payloads.shape:
                        raise AssertionError(f"{method} returned an invalid shape")
                    runtime_samples[(name, method)].append(elapsed)
                    runtime_frames[(name, method)] += frames_per_cluster
                    errors = decoded - payloads
                    passed = np.max(np.abs(errors), axis=1) <= float(cfg["pass_tolerance_max_abs"])
                    outcomes[method][cluster_index] = passed
                    abs_error_mean[method][cluster_index] = np.mean(np.abs(errors), axis=1)
                    if method in BLIND_METHODS:
                        abs_sync[method][cluster_index] = np.abs(params - np.asarray(truth)[None, :])
                        loose[method][cluster_index] = _sync_mask(params, truth, cfg["loose_sync_tolerance"])
                        strict[method][cluster_index] = _sync_mask(params, truth, cfg["strict_sync_tolerance"])
                    for frame_index in range(frames_per_cluster):
                        error = errors[frame_index]
                        row = {
                            "condition": name, "cluster_id": cluster_id, "family": family,
                            "channel_seed_index": channel_index + 1, "frame_index": frame_index + 1,
                            "method": method, "buffer_sha256": buffer_hash,
                            "frame_pass": bool(passed[frame_index]),
                            "max_abs_error": float(np.max(np.abs(error))),
                            "mae": float(np.mean(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(error * error))),
                        }
                        if method == "raw_ls_informed_oracle":
                            row.update({"estimated_timing_samples": truth[0], "estimated_clock_ppm": truth[1],
                                        "estimated_cfo_cycles_per_symbol": truth[2],
                                        "abs_timing_error_samples": None, "abs_clock_error_ppm": None,
                                        "abs_cfo_error_cycles_per_symbol": None,
                                        "sync_acquired_loose": None, "sync_acquired_strict": None})
                        else:
                            row.update({
                                "estimated_timing_samples": float(params[frame_index, 0]),
                                "estimated_clock_ppm": float(params[frame_index, 1]),
                                "estimated_cfo_cycles_per_symbol": float(params[frame_index, 2]),
                                "abs_timing_error_samples": float(abs_sync[method][cluster_index, frame_index, 0]),
                                "abs_clock_error_ppm": float(abs_sync[method][cluster_index, frame_index, 1]),
                                "abs_cfo_error_cycles_per_symbol": float(
                                    abs_sync[method][cluster_index, frame_index, 2]
                                ),
                                "sync_acquired_loose": bool(loose[method][cluster_index, frame_index]),
                                "sync_acquired_strict": bool(strict[method][cluster_index, frame_index]),
                            })
                        frame_rows.append(row)
                    cluster_rows.append({
                        "condition": name, "cluster_id": cluster_id, "family": family, "method": method,
                        "buffer_sha256": buffer_hash, "frames": frames_per_cluster,
                        "frames_passing": int(np.count_nonzero(passed)),
                        "cluster_pass_share": float(np.mean(passed)),
                        "runtime_seconds": float(elapsed),
                    })
                cluster_index += 1
        if cluster_index != n_clusters:
            raise AssertionError("Cluster count differs from frozen design")

        for method_index, method in enumerate(methods):
            frame_outcomes = outcomes[method]
            cluster_pass = np.mean(frame_outcomes, axis=1)
            pass_low, pass_high = _cluster_bootstrap(
                cluster_pass,
                int(cfg["bootstrap_seed"]) + condition_index * 100 + method_index,
                bootstrap_replicates,
            )
            frame_count = int(frame_outcomes.size)
            pass_count = int(np.count_nonzero(frame_outcomes))
            runtime = _runtime_summary(runtime_samples[(name, method)], runtime_frames[(name, method)])
            result: dict[str, Any] = {
                "condition": name, "method": method, "clusters": n_clusters, "frames": frame_count,
                "frames_passing": pass_count, "frame_pass_share": pass_count / frame_count,
                "cluster_bootstrap95_low": pass_low, "cluster_bootstrap95_high": pass_high,
                "mean_abs_error": float(np.mean(abs_error_mean[method])), **runtime,
            }
            if method in BLIND_METHODS:
                err = abs_sync[method]
                loose_cluster = np.mean(loose[method], axis=1)
                strict_cluster = np.mean(strict[method], axis=1)
                loose_low, loose_high = _cluster_bootstrap(loose_cluster,
                    int(cfg["bootstrap_seed"]) + condition_index * 100 + method_index + 17, bootstrap_replicates)
                strict_low, strict_high = _cluster_bootstrap(strict_cluster,
                    int(cfg["bootstrap_seed"]) + condition_index * 100 + method_index + 29, bootstrap_replicates)
                result.update({
                    "mean_abs_timing_error_samples": float(np.mean(err[:, :, 0])),
                    "mean_abs_clock_error_ppm": float(np.mean(err[:, :, 1])),
                    "mean_abs_cfo_error_cycles_per_symbol": float(np.mean(err[:, :, 2])),
                    "sync_acquired_loose_frames": int(np.count_nonzero(loose[method])),
                    "sync_acquired_loose_share": float(np.mean(loose[method])),
                    "sync_loose_cluster_bootstrap95_low": loose_low,
                    "sync_loose_cluster_bootstrap95_high": loose_high,
                    "sync_acquired_strict_frames": int(np.count_nonzero(strict[method])),
                    "sync_acquired_strict_share": float(np.mean(strict[method])),
                    "sync_strict_cluster_bootstrap95_low": strict_low,
                    "sync_strict_cluster_bootstrap95_high": strict_high,
                })
            else:
                result.update({
                    "mean_abs_timing_error_samples": None, "mean_abs_clock_error_ppm": None,
                    "mean_abs_cfo_error_cycles_per_symbol": None,
                    "sync_acquired_loose_frames": None, "sync_acquired_loose_share": None,
                    "sync_loose_cluster_bootstrap95_low": None, "sync_loose_cluster_bootstrap95_high": None,
                    "sync_acquired_strict_frames": None, "sync_acquired_strict_share": None,
                    "sync_strict_cluster_bootstrap95_low": None, "sync_strict_cluster_bootstrap95_high": None,
                })
            condition_summaries.append(result)

        # Paired comparisons retain within-buffer pairing and bootstrap paired differences at cluster level.
        pair_index = 0
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                a = outcomes[method_a]
                b = outcomes[method_b]
                cluster_diff = np.mean(a.astype(float) - b.astype(float), axis=1)
                lo, hi = _cluster_bootstrap(cluster_diff,
                    int(cfg["bootstrap_seed"]) + condition_index * 1000 + pair_index + 101,
                    bootstrap_replicates)
                paired_rows.append({
                    "condition": name, "method_a": method_a, "method_b": method_b,
                    "frames": int(a.size), "a_pass_b_fail_frames": int(np.count_nonzero(a & ~b)),
                    "a_fail_b_pass_frames": int(np.count_nonzero(~a & b)),
                    "paired_pass_share_difference_a_minus_b": float(np.mean(a) - np.mean(b)),
                    "cluster_bootstrap95_low": lo, "cluster_bootstrap95_high": hi,
                    "clusters": n_clusters,
                })
                pair_index += 1

    runtime_totals = {}
    for condition in conditions:
        for method in methods:
            key = (condition["name"], method)
            runtime_totals[f"{key[0]}::{key[1]}"] = _runtime_summary(runtime_samples[key], runtime_frames[key])
    result = {
        "study_id": cfg["study_id"],
        "classification": "SYNTHETIC, DIMENSIONLESS, SOFTWARE-ONLY; LOCAL AND UNPUBLISHED",
        "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_sha256": json.loads(LOCK_PATH.read_text(encoding="utf-8"))["source_sha256"],
        "provenance": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(HERE.parent),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "platform": platform.platform(), "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(), "peak_process_rss_mb": _rss_mb(),
            "note": (
                "v1/v7/v8 source snapshots are vendored unchanged under compat/; "
                "no network write or publication was performed."
            ),
        },
        "protocol": cfg,
        "design": {
            "payload_families": len(families), "independent_channel_seed_replicates_per_family": seeds_per_family,
            "independent_family_x_channel_seed_clusters": n_clusters,
            "frames_per_cluster": frames_per_cluster, "frames_per_condition": n_clusters * frames_per_cluster,
            "conditions": len(conditions), "receiver_methods": list(methods),
            "pairing": (
                "Each method in a condition received the same generated buffers; "
                "buffers are hashed in frame/cluster outputs."
            ),
            "uncertainty": (
                "Percentile bootstrap over independent payload-family x channel-seed clusters; "
                "all replicate frames in a cluster are resampled together."
            ),
            "peak_process_rss_mb": _rss_mb(),
        },
        "seed_catalog": seed_catalog,
        "condition_summaries": condition_summaries,
        "paired_comparisons": paired_rows,
        "runtime_by_condition_method": runtime_totals,
        "frame_results": frame_rows,
        "cluster_results": cluster_rows,
        "buffer_hashes": [{"condition": k[0], "cluster_id": k[1], "buffer_sha256": val}
                          for k, val in sorted(buffer_hash_by_condition_cluster.items())],
    }
    if write_outputs:
        write_result_bundle(result)
    return result


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(result: dict[str, Any]) -> str:
    summaries = result["condition_summaries"]
    lines = [
        "# Digital-to-Wave v9 — local generalization study",
        "",
        (
            "> **Status:** local, unpublished, synthetic and dimensionless software study. "
            "No quantum, diamond, biological, hardware, or physical-channel claim is made."
        ),
        "",
        "## Frozen design and provenance",
        "",
        f"- Frozen protocol lock SHA-256: `{result['protocol_lock_sha256']}`",
        f"- Source commit snapshot: `{result['provenance']['git_commit']}`",
        (
            f"- Runtime: Python {result['provenance']['python']}, NumPy {result['provenance']['numpy']}, "
            f"Matplotlib {result['provenance']['matplotlib']} "
            "(recorded rather than silently substituting the repository pins)."
        ),
        (
            f"- Payload families: {result['design']['payload_families']}; independent family × channel-seed clusters: "
            f"{result['design']['independent_family_x_channel_seed_clusters']}; "
            f"frames per cluster: {result['design']['frames_per_cluster']}"
        ),
        (
            f"- {result['design']['frames_per_condition']} paired frames per condition; "
            f"{result['design']['conditions']} conditions; pass iff max absolute error across 32 values "
            f"≤ {result['protocol']['pass_tolerance_max_abs']}"
        ),
        (
            "- Confidence intervals are 95% percentile cluster-bootstrap intervals over independent "
            "payload-family × channel-seed units; replicate frames remain together. They are not frame-only intervals."
        ),
        (
            "- Every receiver arm saw the same buffer within each condition/cluster/frame; "
            "SHA-256 hashes are in `buffer-hashes.csv`."
        ),
        (
            "- Full v8 versus v7 is a bundled comparison. Score-only, refinement-only and decoder-only arms "
            "change one component at a time relative to v7; do not ascribe the bundled v8 difference to one component."
        ),
        "",
        "## Exact-frame pass results",
        "",
        "Pass share with cluster-bootstrap 95% CI; counts use the exact all-32-symbol frame rule.",
        "",
        "| Condition | Receiver | Exact-frame passes | Pass share (cluster-bootstrap 95% CI) |",
        "|---|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | `{row['method']}` | {row['frames_passing']}/{row['frames']} | "
            f"{row['frame_pass_share']:.3f} ("
            f"{row['cluster_bootstrap95_low']:.3f}–{row['cluster_bootstrap95_high']:.3f}) |"
        )
    lines += [
        "",
        "## Synchronization/acquisition and error",
        "",
        (
            "Blind receiver errors are relative to the frozen static or mean trajectory "
            "coordinates in the condition definition. For time-varying conditions, "
            "center-coordinate acquisition is not a measure of tracking error across "
            "the entire trajectory. The informed raw-LS arm has no acquisition estimate; "
            "it is supplied the declared static/mean reference parameters."
        ),
        "",
        (
            "| Condition | Blind receiver | Loose joint acquisition | Strict joint acquisition | "
            "Mean abs timing error (samples) | Mean abs clock error (ppm) | "
            "Mean abs CFO error (cycles/symbol) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["method"] in BLIND_METHODS:
            lines.append(
                f"| {row['condition']} | `{row['method']}` | "
                f"{row['sync_acquired_loose_frames']}/{row['frames']} ({row['sync_acquired_loose_share']:.3f}) | "
                f"{row['sync_acquired_strict_frames']}/{row['frames']} ({row['sync_acquired_strict_share']:.3f}) | "
                f"{row['mean_abs_timing_error_samples']:.4g} | {row['mean_abs_clock_error_ppm']:.4g} | "
                f"{row['mean_abs_cfo_error_cycles_per_symbol']:.4g} |"
            )
    lines += [
        "",
        (
            "Acquisition CIs are cluster-bootstrap intervals and are available in "
            "`condition-summary.csv`/`results.json` for each blind arm."
        ),
        "",
        "## One-change ablations and paired contrasts",
        "",
        (
            "Each contrast below is a paired pass-share difference over the same buffers, with a cluster-bootstrap "
            "95% CI. A CI including zero does not establish a difference."
        ),
        "",
        (
            "| Condition | A − B | Paired pass-share difference | Cluster-bootstrap 95% CI | "
            "A-only / B-only frame passes |"
        ),
        "|---|---|---:|---:|---:|",
    ]
    key_pairs = {("score_only_ablation", "v7_blind"), ("refinement_only_ablation", "v7_blind"),
                 ("decoder_only_ablation", "v7_blind"), ("v8_full", "v7_blind"),
                 ("v8_full", "raw_ls_informed_oracle")}
    for row in result["paired_comparisons"]:
        if (row["method_a"], row["method_b"]) in key_pairs:
            lines.append(
                f"| {row['condition']} | `{row['method_a']}` − `{row['method_b']}` | "
                f"{row['paired_pass_share_difference_a_minus_b']:+.3f} | "
                f"{row['cluster_bootstrap95_low']:+.3f}–{row['cluster_bootstrap95_high']:+.3f} | "
                f"{row['a_pass_b_fail_frames']} / {row['a_fail_b_pass_frames']} |"
            )
    lines += [
        "",
        "## Runtime and memory",
        "",
        (
            f"Peak resident set size for the complete study process: "
            f"**{result['provenance']['peak_process_rss_mb']:.1f} MB** "
            "(Linux `ru_maxrss`; process-wide peak, not method-attributed). "
            "Method runtime is cumulative wall-clock time; "
            "per-frame runtime divides by scored frames. Per-call p50/p95 and "
            "condition-by-method values are in `condition-summary.csv`."
        ),
        "",
        "| Condition | Receiver | Total runtime (s) | Mean runtime (ms/frame) | Call p50 / p95 (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | `{row['method']}` | {row['runtime_total_seconds']:.4f} | "
            f"{row['runtime_mean_ms_per_frame']:.4f} | "
            f"{row['runtime_call_p50_ms']:.3f} / {row['runtime_call_p95_ms']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation and scope limits",
        "",
        (
            "- This is a generated-data software stress test of the specified signed-amplitude sine "
            "signal and fixed receiver assumptions. It is not measurement, hardware testing, "
            "or validation of any physical system or preprint."
        ),
        (
            "- Filtering, colored noise, impulses, fractional timing, time-varying CFO/clock "
            "and compound channel behavior are deliberately synthetic and dimensionless; "
            "the design does not imply realistic channel distributions."
        ),
        (
            "- The raw-LS informed reference knows the static or mean timing/clock/CFO coordinates "
            "but uses the existing static raw-fit decoder. It is not a true upper bound for "
            "filtering or time-varying channels and has no equalizer/tracker for those effects."
        ),
        (
            "- Results generalize only over the prespecified synthetic payload-family distributions, "
            "random seeds and finite conditions here. They do not establish performance "
            "outside the study grid."
        ),
        "- Nothing was committed, pushed, published, or uploaded. The bundle is for local review only.",
        "",
        "## Reproduction",
        "",
        (
            "Run from this directory with `python study.py --run`. The script verifies `lock.json` "
            "against the frozen source/configuration SHA-256 values before executing. "
            "`python -m pytest -q` runs reduced deterministic tests and does not consume the full-run seeds."
        ),
        "",
    ]
    return "\n".join(lines)


def _make_plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [c["name"] for c in result["protocol"]["condition_design"]]
    methods = list(METHODS)
    rows = {(r["condition"], r["method"]): r for r in result["condition_summaries"]}
    x = np.arange(len(conditions), dtype=float)
    width = 0.12
    fig, ax = plt.subplots(figsize=(15, 6.2))
    palette = ["#536d8a", "#65a89a", "#e1ad5a", "#c87562", "#695b95", "#282c34"]
    for i, method in enumerate(methods):
        vals = [rows[(condition, method)]["frame_pass_share"] for condition in conditions]
        lo = [vals[j] - rows[(condition, method)]["cluster_bootstrap95_low"] for j, condition in enumerate(conditions)]
        hi = [rows[(condition, method)]["cluster_bootstrap95_high"] - vals[j] for j, condition in enumerate(conditions)]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=method,
               color=palette[i], yerr=np.array([lo, hi]), capsize=2, linewidth=0,
               error_kw={"elinewidth": 0.8})
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Exact-frame pass share")
    ax.set_title("v9 synthetic generalization study — cluster-bootstrap 95% intervals")
    ax.set_xticks(x, conditions, rotation=26, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_result_bundle(result: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(RESULTS_DIR / "condition-summary.csv", result["condition_summaries"], SUMMARY_FIELDS)
    _write_csv(RESULTS_DIR / "frame-results.csv", result["frame_results"], FRAME_FIELDS)
    _write_csv(RESULTS_DIR / "cluster-results.csv", result["cluster_results"])
    _write_csv(RESULTS_DIR / "paired-comparisons.csv", result["paired_comparisons"])
    _write_csv(RESULTS_DIR / "buffer-hashes.csv", result["buffer_hashes"])
    _make_plot(result, RESULTS_DIR / "pass-share.png")
    (RESULTS_DIR / "report.md").write_text(_render_report(result), encoding="utf-8")
    file_hashes = {}
    for path in sorted(RESULTS_DIR.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            file_hashes[path.name] = sha256_file(path)
    (RESULTS_DIR / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in file_hashes.items()),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the frozen full protocol and write local outputs")
    parser.add_argument(
        "--verify-lock",
        action="store_true",
        help="verify the source/config SHA-256 lock without running",
    )
    args = parser.parse_args()
    lock = verify_lock()
    if args.verify_lock and not args.run:
        print(json.dumps({"lock_status": lock["lock_status"], "protocol_lock_sha256": sha256_file(LOCK_PATH),
                          "source_sha256": lock["source_sha256"]}, indent=2))
        return 0
    if not args.run:
        parser.error("choose --run or --verify-lock")
    print(f"Frozen v9 protocol verified; starting {load_config()['study_id']} locally.", flush=True)
    result = run_study(write_outputs=True)
    print(f"Completed {result['design']['independent_family_x_channel_seed_clusters']} clusters x "
          f"{result['design']['frames_per_cluster']} frames x {result['design']['conditions']} conditions; "
          f"peak RSS {result['provenance']['peak_process_rss_mb']:.1f} MB.", flush=True)
    print(f"Outputs: {RESULTS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
