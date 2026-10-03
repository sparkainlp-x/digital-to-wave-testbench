"""Synthetic v8 blind synchronization benchmark: maximum-likelihood metric with coarse-to-fine refinement.

v7 diagnosis (development seed 20261111, see reports/v8-ml-sync-protocol-lock.md): a clock error of
2500 ppm changes the carrier by the same amount as a CFO of 0.01 cycles/symbol, so (timing, clock, CFO)
candidates lie on a ridge. The v7 metric (sum of squared values decoded after linear resampling) is
biased against the true candidate, because linear interpolation attenuates the carrier, and at
-2500 ppm the wrong ridge point (0 ppm, -0.01 cycles/symbol) scores as high as the truth.

The v8 blind receiver scores every candidate with the Gaussian generalized likelihood ratio computed
on the raw received samples (no resampling), searches the v7 grid, refines the three best coarse
candidates by coarse-to-fine halving in ridge coordinates (timing, clock, carrier offset), and decodes
with the least-squares amplitudes of the same raw-sample fit (no interpolation). It is never told the true
impairments. Two informed references are reported: the unchanged v7 oracle (exact impairments, v7 linear
resampling decoder) and a raw-sample least-squares oracle (exact impairments, v8 decoder), so that search
loss and decoder gain can be separated. The v1-v7 modules are
imported unchanged. This model is dimensionless and makes no claim about a physical channel or receiver.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import robustness_benchmark_v7_clock_drift_cfo as v7
from digital_to_wave import decode_wave
from robustness_benchmark_v2 import build_input_patterns, wilson_interval

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
BENCHMARK = "synthetic_ml_sync_refinement_v8"
STEM = "robustness-benchmark-v8-ml-sync"
PROTOCOL_LOCK = "reports/v8-ml-sync-protocol-lock.md"
MASTER_SEED = 20261011
DEV_SEED = 20261111
TEST_SEED = 5151
PRIOR_SEEDS = frozenset({20261003, 20261004, 20261005, 20261006, 20261007, 20261008, 20261009, 20261010})
VECTOR_SEED = v7.VECTOR_SEED
TRIALS_PER_VECTOR = 200
CHUNK_FRAMES = 50
NOISE_LEVELS = (0.0, 0.25, 0.45)
CFO_LEVELS = (-0.01, -0.005, 0.0, 0.005, 0.01)
CLOCK_LEVELS_PPM = (-5000.0, -2500.0, -1250.0, 0.0, 1250.0, 2500.0, 5000.0)
TIMING_OFFSETS = (-2.0, 0.0, 2.0)
WEAK_CELL = (0.0, -2500.0, 2.0, 0.45)
SAMPLES_PER_SYMBOL = v7.SAMPLES_PER_SYMBOL
VECTOR_LENGTH = v7.VECTOR_LENGTH
GUARD_SAMPLES = v7.GUARD_SAMPLES
FRAME_SAMPLES = v7.FRAME_SAMPLES
BUFFER_SAMPLES = v7.BUFFER_SAMPLES
TOLERANCE = v7.TOLERANCE
OMEGA = v7.OMEGA
CONFIG = v7.CONFIG
# Search envelope and coarse grid are identical to v7; refinement stays inside the envelope.
COARSE_TIMING = v7.SEARCH_TIMING_OFFSETS
COARSE_CLOCK_PPM = v7.SEARCH_CLOCK_LEVELS_PPM
COARSE_CFO = v7.SEARCH_CFO_LEVELS
TIMING_BOUND, CLOCK_BOUND_PPM, CFO_BOUND = 4.0, 5000.0, 0.01
TOP_K = 3
REFINE_LEVELS = 3
REFINE_STEPS = (1.0, 1250.0, 0.0025)  # half of the coarse step; halved at each further level
LOOSE_TOLERANCE = (2.0, 2500.0, 0.005)  # v7 rule: within one coarse grid step
STRICT_TOLERANCE = (1.0, 1250.0, 0.0025)  # within half a coarse grid step
NOMINAL = "nominal"
ORACLE = "oracle"
ORACLE_LS = "oracle_raw_ls"
V7_BLIND = "v7_blind_joint_search"
V8_BLIND = "v8_ml_refined_search"
METHODS = (NOMINAL, ORACLE, ORACLE_LS, V7_BLIND, V8_BLIND)
BLIND_METHODS = (V7_BLIND, V8_BLIND)
PAIRS = (
    (V8_BLIND, V7_BLIND), (V8_BLIND, ORACLE), (V8_BLIND, ORACLE_LS), (V7_BLIND, ORACLE), (ORACLE_LS, ORACLE),
    (V8_BLIND, NOMINAL),
)
# Relative score margin a refinement move must exceed; makes plateau ties platform-independent.
SCORE_MARGIN = 1e-9
ACQ_FIELDS = (
    "timing_acq_loose_frames", "clock_acq_loose_frames", "cfo_acq_loose_frames", "joint_acq_loose_frames",
    "joint_acq_loose_share", "joint_acq_loose_wilson95_lower", "joint_acq_loose_wilson95_upper",
    "timing_acq_strict_frames", "clock_acq_strict_frames", "cfo_acq_strict_frames", "joint_acq_strict_frames",
    "joint_acq_strict_share", "joint_acq_strict_wilson95_lower", "joint_acq_strict_wilson95_upper",
    "mean_abs_timing_error", "mean_abs_clock_error_ppm", "mean_abs_cfo_error",
)
RESULT_FIELDS = (
    "cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples", "noise_std", "vector_pattern",
    "in_v7_grid", "method", "frames", "frames_passing", "frame_pass_share", "wilson95_lower", "wilson95_upper",
    "empirical_mae", "empirical_rmse", "mean_bias", "max_abs_error", *ACQ_FIELDS,
)
PAIR_FIELDS = (
    "cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples", "noise_std", "vector_pattern",
    "in_v7_grid", "method_a", "method_b", "frames", "a_pass_b_fail", "a_fail_b_pass", "mcnemar_exact_two_sided_p",
)
POOLED = "POOLED_ALL_VECTORS"
_SOURCE_INDEX = np.arange(BUFFER_SAMPLES, dtype=float) - GUARD_SAMPLES


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


def _check_buffer(buffer: Any) -> np.ndarray:
    b = np.asarray(buffer, dtype=float)
    if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] != BUFFER_SAMPLES or not np.all(np.isfinite(b)):
        raise ValueError("buffer must be a finite (frames, 2176) array")
    return b


def _raw_fit(b: np.ndarray, timing: float, ppm: float, cfo: float) -> tuple[np.ndarray, np.ndarray]:
    t = _SOURCE_INDEX * (1.0 + float(ppm) * 1e-6) + float(timing)
    symbol = np.floor(t / SAMPLES_PER_SYMBOL).astype(int)
    index = np.nonzero((symbol >= 0) & (symbol < VECTOR_LENGTH))[0]
    k = symbol[index]
    if k.size == 0 or not np.array_equal(np.unique(k), np.arange(VECTOR_LENGTH)):
        raise ValueError("candidate must place every symbol inside the buffer")
    template = np.sin((OMEGA + 2.0 * math.pi * float(cfo) / SAMPLES_PER_SYMBOL) * t[index] + math.pi / 4.0)
    starts = np.concatenate(([0], np.nonzero(np.diff(k))[0] + 1))
    numerator = np.add.reduceat(b[:, index] * template[None, :], starts, axis=1)
    denominator = np.add.reduceat(template * template, starts)
    return numerator, denominator


def ml_score(buffer: Any, timing: float, ppm: float, cfo: float) -> np.ndarray:
    """Gaussian GLRT score of one candidate: energy of the least-squares fit on raw received samples.

    For candidate parameters each received sample maps to source time t(m); samples inside the frame
    belong to symbol floor(t/64) with template sin(2*pi*(4+cfo)*t/64 + pi/4) and a free signed amplitude.
    The score sum_k (y . s_k)^2 / (s_k . s_k) equals ||y||^2 minus the least-squares residual, so
    maximizing it is maximum likelihood over (timing, clock, CFO) under white Gaussian noise.
    """
    numerator, denominator = _raw_fit(_check_buffer(buffer), timing, ppm, cfo)
    return np.sum(numerator * numerator / denominator[None, :], axis=1)


def ls_decode(buffer: Any, timing: float, ppm: float, cfo: float) -> np.ndarray:
    """Least-squares signed amplitudes of the raw-sample fit at the given parameters (no interpolation)."""
    numerator, denominator = _raw_fit(_check_buffer(buffer), timing, ppm, cfo)
    return numerator / denominator[None, :]


def _scores_per_frame(b: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Score each frame at its own candidate; frames sharing a candidate are scored together."""
    scores = np.full(b.shape[0], -np.inf)
    inside = (
        (np.abs(params[:, 0]) <= TIMING_BOUND + 1e-9) & (np.abs(params[:, 1]) <= CLOCK_BOUND_PPM + 1e-6)
        & (np.abs(params[:, 2]) <= CFO_BOUND + 1e-12)
    )
    if not np.any(inside):
        return scores
    unique, inverse = np.unique(params[inside], axis=0, return_inverse=True)
    rows = np.nonzero(inside)[0]
    inverse = np.asarray(inverse).reshape(-1)
    for group, candidate in enumerate(unique):
        members = rows[inverse == group]
        scores[members] = ml_score(b[members], *candidate)
    return scores


def _decode_per_frame(b: np.ndarray, params: np.ndarray) -> np.ndarray:
    estimates = np.zeros((b.shape[0], VECTOR_LENGTH))
    unique, inverse = np.unique(params, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    for group, candidate in enumerate(unique):
        members = np.nonzero(inverse == group)[0]
        estimates[members] = ls_decode(b[members], *candidate)
    return estimates


def _ordered_offsets() -> np.ndarray:
    offsets = [(a, c, f) for a in (-1, 0, 1) for c in (-1, 0, 1) for f in (-1, 0, 1)]
    return np.asarray(sorted(offsets, key=lambda o: (abs(o[0]) + abs(o[1]) + abs(o[2]), o)), dtype=float)


def carrier_offset(ppm: Any, cfo: Any) -> Any:
    """Carrier offset seen in received samples, in cycles per nominal symbol: (4 + cfo)(1 + ppm 1e-6) - 4."""
    return (CONFIG.cycles_per_symbol + np.asarray(cfo, dtype=float)) * (1.0 + np.asarray(ppm, dtype=float) * 1e-6) \
        - CONFIG.cycles_per_symbol


def cfo_from_carrier(ppm: Any, carrier: Any) -> Any:
    return (CONFIG.cycles_per_symbol + np.asarray(carrier, dtype=float)) / (1.0 + np.asarray(ppm, dtype=float) * 1e-6) \
        - CONFIG.cycles_per_symbol


def v8_blind_search(buffer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose timing, clock ppm and CFO using only the received buffer; return them and the decoded values.

    1. Score the 125-point v7 coarse grid with the GLRT metric (ties keep the smallest corrections).
    2. Refine each frame's three best coarse candidates in ridge coordinates (timing, clock ppm, carrier
       offset g = (4 + cfo)(1 + ppm 1e-6) - 4), so that a clock move at fixed g follows the clock/CFO ridge.
       Three levels of a 3x3x3 local search start at half the coarse step (1 sample, 1250 ppm,
       0.0025 cycles/symbol) and halve per level; candidates must stay inside the v7 search envelope and a
       move is accepted only if it increases the score by more than a relative margin of 1e-9 (inside a
       plateau, where sample-to-symbol membership does not change, the earlier candidate is kept).
    3. Keep the refined candidate with the highest score and decode with the raw-sample least-squares fit.
    """
    b = _check_buffer(buffer)
    frames = b.shape[0]
    coarse = sorted(
        ((d, p, f) for d in COARSE_TIMING for p in COARSE_CLOCK_PPM for f in COARSE_CFO),
        key=lambda item: (abs(item[0]), abs(item[1]), abs(item[2]), item),
    )
    grid = np.asarray(coarse, dtype=float)
    coarse_scores = np.stack([ml_score(b, *candidate) for candidate in coarse], axis=1)
    order = np.argsort(-coarse_scores, axis=1, kind="stable")[:, :TOP_K]
    offsets = _ordered_offsets()
    best_score = np.full(frames, -np.inf)
    best = np.zeros((frames, 3))
    rows = np.arange(frames)
    for rank in range(order.shape[1]):
        start = grid[order[:, rank]]
        center = np.stack([start[:, 0], start[:, 1], carrier_offset(start[:, 1], start[:, 2])], axis=1)
        params = start.copy()
        score = coarse_scores[rows, order[:, rank]]
        step = np.asarray(REFINE_STEPS, dtype=float)
        for _ in range(REFINE_LEVELS):
            new_center, new_params, new_score = center.copy(), params.copy(), score.copy()
            for offset in offsets:
                if not np.any(offset):
                    continue
                ridge = center + offset * step
                candidate = np.stack([ridge[:, 0], ridge[:, 1], cfo_from_carrier(ridge[:, 1], ridge[:, 2])], axis=1)
                candidate_score = _scores_per_frame(b, candidate)
                better = candidate_score > new_score + SCORE_MARGIN * np.maximum(1.0, np.abs(new_score))
                new_center[better], new_params[better] = ridge[better], candidate[better]
                new_score[better] = candidate_score[better]
            center, params, score = new_center, new_params, new_score
            step = step / 2.0
        if rank == 0:
            improved = np.ones(frames, dtype=bool)
        else:
            improved = score > best_score + SCORE_MARGIN * np.maximum(1.0, np.abs(best_score))
        best[improved] = params[improved]
        best_score[improved] = score[improved]
    return best[:, 0], best[:, 1], best[:, 2], _decode_per_frame(b, best)


def _acquisition(chosen: np.ndarray, truth: tuple[float, float, float]) -> dict[str, np.ndarray]:
    error = np.abs(chosen - np.asarray(truth)[None, :])
    out: dict[str, np.ndarray] = {"abs_error": error}
    for label, tol in (("loose", LOOSE_TOLERANCE), ("strict", STRICT_TOLERANCE)):
        within = error <= np.asarray(tol)[None, :] * (1 + 1e-9)
        out[f"timing_{label}"], out[f"clock_{label}"], out[f"cfo_{label}"] = within[:, 0], within[:, 1], within[:, 2]
        out[f"joint_{label}"] = np.all(within, axis=1)
    return out


def _wilson_or_none(count: int, frames: int, include: bool) -> tuple[float | None, float | None]:
    return wilson_interval(count, frames) if include and frames > 1 else (None, None)


def _summary(errors: np.ndarray, acq: dict[str, np.ndarray] | None, include_intervals: bool) -> dict[str, Any]:
    absolute = np.abs(errors)
    passed = np.max(absolute, axis=1) <= TOLERANCE
    frames, count = int(errors.shape[0]), int(np.count_nonzero(passed))
    low, high = _wilson_or_none(count, frames, include_intervals)
    row: dict[str, Any] = {
        "frames": frames, "frames_passing": count, "frame_pass_share": count / frames,
        "wilson95_lower": low, "wilson95_upper": high,
        "empirical_mae": float(np.mean(absolute)), "empirical_rmse": float(np.sqrt(np.mean(errors * errors))),
        "mean_bias": float(np.mean(errors)), "max_abs_error": float(np.max(absolute)),
    }
    for field in ACQ_FIELDS:
        row[field] = None
    if acq is not None:
        for label in ("loose", "strict"):
            for part in ("timing", "clock", "cfo", "joint"):
                row[f"{part}_acq_{label}_frames"] = int(np.count_nonzero(acq[f"{part}_{label}"]))
            joint = row[f"joint_acq_{label}_frames"]
            row[f"joint_acq_{label}_share"] = joint / frames
            lo, hi = _wilson_or_none(joint, frames, include_intervals)
            row[f"joint_acq_{label}_wilson95_lower"], row[f"joint_acq_{label}_wilson95_upper"] = lo, hi
        mean_error = np.mean(acq["abs_error"], axis=0)
        row["mean_abs_timing_error"] = float(mean_error[0])
        row["mean_abs_clock_error_ppm"] = float(mean_error[1])
        row["mean_abs_cfo_error"] = float(mean_error[2])
    return row


def _pair_row(key: dict[str, Any], a_name: str, b_name: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    apbf, afbp = int(np.count_nonzero(a & ~b)), int(np.count_nonzero(~a & b))
    return {**key, "method_a": a_name, "method_b": b_name, "frames": int(a.size), "a_pass_b_fail": apbf,
            "a_fail_b_pass": afbp, "mcnemar_exact_two_sided_p": v7._mcnemar(apbf, afbp)}


def _in_v7_grid(cfo: float, ppm: float, delay: float) -> bool:
    return cfo in v7.CFO_LEVELS and ppm in v7.CLOCK_LEVELS_PPM and delay in v7.TIMING_OFFSETS


def _run_cell(args: tuple[int, int, tuple[float, float, float, float], int, int]) -> dict[str, Any]:
    master_seed, cell_index, (cfo, ppm, delay, sigma), trials, chunk = args
    patterns = build_input_patterns()
    key_base = {"cfo_cycles_per_symbol": cfo, "clock_error_ppm": ppm, "timing_offset_samples": delay,
                "noise_std": sigma}
    in_grid = _in_v7_grid(cfo, ppm, delay)
    truth = (delay, ppm, cfo)
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    pool_err: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    pool_chosen: dict[str, list[np.ndarray]] = {m: [] for m in BLIND_METHODS}
    for vector_index, (name, reference) in enumerate(patterns.items()):
        n_trials = 1 if sigma == 0 else trials
        rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), cell_index, vector_index]))
        clean = v7.received_buffer(reference, cfo_cycles_per_symbol=cfo, clock_error_ppm=ppm,
                                   timing_offset_samples=delay)
        err: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
        chosen: dict[str, list[np.ndarray]] = {m: [] for m in BLIND_METHODS}
        for start in range(0, n_trials, chunk):
            batch = min(chunk, n_trials - start)
            buffers = np.broadcast_to(clean, (batch, BUFFER_SAMPLES)).copy()
            if sigma > 0:
                buffers += rng.normal(0.0, sigma, size=buffers.shape)
            windows = buffers[:, GUARD_SAMPLES:GUARD_SAMPLES + FRAME_SAMPLES]
            nominal = decode_wave(windows.reshape(-1, SAMPLES_PER_SYMBOL), CONFIG).reshape(batch, VECTOR_LENGTH)
            oracle = v7._candidate_decode(buffers, delay, ppm, cfo)
            oracle_ls = ls_decode(buffers, delay, ppm, cfo)
            d7, p7, f7, blind7 = v7.blind_joint_search(buffers)
            d8, p8, f8, blind8 = v8_blind_search(buffers)
            decoded_by_method = {NOMINAL: nominal, ORACLE: oracle, ORACLE_LS: oracle_ls, V7_BLIND: blind7,
                                 V8_BLIND: blind8}
            for method, decoded in decoded_by_method.items():
                err[method].append(decoded - reference[None, :])
            chosen[V7_BLIND].append(np.stack([d7, p7, f7], axis=1))
            chosen[V8_BLIND].append(np.stack([d8, p8, f8], axis=1))
        errors = {m: np.concatenate(v) for m, v in err.items()}
        picks = {m: np.concatenate(v) for m, v in chosen.items()}
        key = {**key_base, "vector_pattern": name, "in_v7_grid": in_grid}
        for method in METHODS:
            acq = _acquisition(picks[method], truth) if method in BLIND_METHODS else None
            rows.append({**key, "method": method, **_summary(errors[method], acq, sigma > 0)})
            pool_err[method].append(errors[method])
            if method in BLIND_METHODS:
                pool_chosen[method].append(picks[method])
        passes = {m: np.max(np.abs(e), axis=1) <= TOLERANCE for m, e in errors.items()}
        for a_name, b_name in PAIRS:
            paired.append(_pair_row(key, a_name, b_name, passes[a_name], passes[b_name]))
    key = {**key_base, "vector_pattern": POOLED, "in_v7_grid": in_grid}
    pooled_errors = {m: np.concatenate(v) for m, v in pool_err.items()}
    for method in METHODS:
        acq = _acquisition(np.concatenate(pool_chosen[method]), truth) if method in BLIND_METHODS else None
        rows.append({**key, "method": method, **_summary(pooled_errors[method], acq, sigma > 0)})
    passes = {m: np.max(np.abs(e), axis=1) <= TOLERANCE for m, e in pooled_errors.items()}
    for a_name, b_name in PAIRS:
        paired.append(_pair_row(key, a_name, b_name, passes[a_name], passes[b_name]))
    return {"rows": rows, "paired": paired}


def protocol_cells() -> list[tuple[float, float, float, float]]:
    """Full-protocol impairment cells in run order; the index of a cell seeds its noise streams."""
    return [(f, p, d, s) for f in CFO_LEVELS for p in CLOCK_LEVELS_PPM for d in TIMING_OFFSETS for s in NOISE_LEVELS]


def reproduce_protocol_cell(
    cfo: float, ppm: float, delay: float, sigma: float, *, chunk_frames: int = CHUNK_FRAMES
) -> dict[str, Any]:
    """Re-run one cell of the full protocol (master seed, full trials, full-grid cell index) on its own."""
    spec = (float(cfo), float(ppm), float(delay), float(sigma))
    cells = protocol_cells()
    if spec not in cells:
        raise ValueError("cell is not part of the full protocol grid")
    return _run_cell((MASTER_SEED, cells.index(spec), spec, TRIALS_PER_VECTOR, _positive_int(chunk_frames, "chunk")))


def module_hashes() -> dict[str, str]:
    names = (
        "digital_to_wave.py", "robustness_benchmark_v2.py", "robustness_benchmark_v3.py",
        "robustness_benchmark_v4.py", "robustness_benchmark_v5_timing.py",
        "robustness_benchmark_v6_timing_recovery.py", "robustness_benchmark_v7_clock_drift_cfo.py", Path(__file__).name,
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


def _pooled_pairs(paired: list[dict[str, Any]], a: str, b: str, sigma: float, in_grid: bool | None) -> dict[str, Any]:
    selected = [r for r in paired if r["vector_pattern"] == POOLED and r["method_a"] == a and r["method_b"] == b
                and r["noise_std"] == sigma and (in_grid is None or r["in_v7_grid"] == in_grid)]
    apbf = sum(r["a_pass_b_fail"] for r in selected)
    afbp = sum(r["a_fail_b_pass"] for r in selected)
    return {"cells": len(selected), "frames": sum(r["frames"] for r in selected), "a_pass_b_fail": apbf,
            "a_fail_b_pass": afbp, "mcnemar_exact_two_sided_p": v7._mcnemar(apbf, afbp)}


def _totals(rows: list[dict[str, Any]], method: str, sigma: float, in_grid: bool | None) -> dict[str, Any]:
    selected = [r for r in rows if r["vector_pattern"] == POOLED and r["method"] == method
                and r["noise_std"] == sigma and (in_grid is None or r["in_v7_grid"] == in_grid)]
    frames = sum(r["frames"] for r in selected)
    passing = sum(r["frames_passing"] for r in selected)
    out: dict[str, Any] = {"cells": len(selected), "frames": frames, "frames_passing": passing}
    if frames:
        out["wilson95"] = list(wilson_interval(passing, frames))
    if method in BLIND_METHODS and frames:
        for label in ("loose", "strict"):
            out[f"joint_acq_{label}_frames"] = sum(r[f"joint_acq_{label}_frames"] for r in selected)
    return out


def summarize(rows: list[dict[str, Any]], paired: list[dict[str, Any]], full_grid: bool) -> dict[str, Any]:
    """Headline numbers and the mechanical evaluation of the pre-stated success criteria."""
    summary: dict[str, Any] = {"by_noise": {}}
    sigmas = sorted({r["noise_std"] for r in rows if r["noise_std"] > 0})
    for sigma in sigmas:
        block: dict[str, Any] = {}
        for scope, flag in (("all_cells", None), ("v7_grid_cells", True), ("extension_cells", False)):
            block[scope] = {
                "totals": {m: _totals(rows, m, sigma, flag) for m in METHODS},
                "paired": {f"{a}_vs_{b}": _pooled_pairs(paired, a, b, sigma, flag) for a, b in PAIRS},
            }
        summary["by_noise"][str(sigma)] = block

    def cell_pair(cell: tuple[float, float, float, float], a: str, b: str) -> dict[str, Any] | None:
        return next((r for r in paired if r["vector_pattern"] == POOLED and r["method_a"] == a and r["method_b"] == b
                     and (r["cfo_cycles_per_symbol"], r["clock_error_ppm"], r["timing_offset_samples"],
                          r["noise_std"]) == cell), None)

    weak = {m: next((r for r in rows if r["vector_pattern"] == POOLED and r["method"] == m
                     and (r["cfo_cycles_per_symbol"], r["clock_error_ppm"], r["timing_offset_samples"],
                          r["noise_std"]) == WEAK_CELL), None) for m in METHODS}
    if all(weak.values()):
        summary["weak_cell"] = {
            "cell": {"cfo_cycles_per_symbol": WEAK_CELL[0], "clock_error_ppm": WEAK_CELL[1],
                     "timing_offset_samples": WEAK_CELL[2], "noise_std": WEAK_CELL[3]},
            "frames_passing": {m: weak[m]["frames_passing"] for m in METHODS},
            "frames": weak[ORACLE]["frames"],
            "joint_acq_loose_frames": {m: weak[m]["joint_acq_loose_frames"] for m in BLIND_METHODS},
            "joint_acq_strict_frames": {m: weak[m]["joint_acq_strict_frames"] for m in BLIND_METHODS},
            "paired": {f"{a}_vs_{b}": cell_pair(WEAK_CELL, a, b) for a, b in PAIRS},
        }
    noisy_pooled = [r for r in paired if r["vector_pattern"] == POOLED and r["noise_std"] > 0]

    def worse(a: str, b: str, in_grid: bool | None) -> list[dict[str, Any]]:
        return [r for r in noisy_pooled if r["method_a"] == a and r["method_b"] == b
                and (in_grid is None or r["in_v7_grid"] == in_grid)
                and r["a_fail_b_pass"] > r["a_pass_b_fail"] and r["mcnemar_exact_two_sided_p"] < 0.05]

    def diff(a: str, b: str, in_grid: bool | None) -> list[dict[str, Any]]:
        return [r for r in noisy_pooled if r["method_a"] == a and r["method_b"] == b
                and (in_grid is None or r["in_v7_grid"] == in_grid) and r["mcnemar_exact_two_sided_p"] < 0.05]

    def cells(selected: list[dict[str, Any]]) -> list[list[float]]:
        return [[r["cfo_cycles_per_symbol"], r["clock_error_ppm"], r["timing_offset_samples"], r["noise_std"]]
                for r in selected]

    criteria: dict[str, Any] = {"evaluated_on_full_protocol_grid": full_grid}
    if "weak_cell" in summary:
        w = summary["weak_cell"]
        p = w["paired"][f"{V8_BLIND}_vs_{V7_BLIND}"]
        criteria["S1_weak_cell"] = {
            "rule": "v8 loose joint acquisition >= 95% and v8 passes more frames than v7 blind, McNemar p < 0.01",
            "pass": bool(w["joint_acq_loose_frames"][V8_BLIND] >= 0.95 * w["frames"]
                         and w["frames_passing"][V8_BLIND] > w["frames_passing"][V7_BLIND]
                         and p["mcnemar_exact_two_sided_p"] < 0.01),
        }
    worse_grid = worse(V8_BLIND, V7_BLIND, True)
    criteria["S2_no_v7_grid_regression"] = {
        "rule": "no noisy v7-grid cell where v8 is significantly worse than v7 blind (unadjusted p < 0.05)",
        "cells_where_v8_worse": cells(worse_grid), "pass": not worse_grid,
    }
    worse_oracle = worse(V8_BLIND, ORACLE, True)
    criteria["S3_not_worse_than_v7_oracle_on_v7_grid"] = {
        "rule": "no noisy v7-grid cell where v8 is significantly worse than the v7 oracle (unadjusted p < 0.05)",
        "cells_where_v8_worse_than_oracle": cells(worse_oracle), "pass": not worse_oracle,
    }
    criteria["reported_v8_vs_raw_ls_oracle_significant_cells"] = cells(diff(V8_BLIND, ORACLE_LS, None))
    criteria["reported_v8_better_than_v7_oracle_cells"] = [
        c for c in cells(diff(V8_BLIND, ORACLE, None)) if c not in cells(worse(V8_BLIND, ORACLE, None))
    ]
    ext = {str(s): summary["by_noise"][str(s)]["extension_cells"] for s in sigmas}
    ext_ok = all(
        b["totals"][V8_BLIND]["frames_passing"] > b["totals"][V7_BLIND]["frames_passing"]
        and b["paired"][f"{V8_BLIND}_vs_{V7_BLIND}"]["mcnemar_exact_two_sided_p"] < 0.05
        for b in ext.values() if b["totals"][V8_BLIND]["cells"]
    ) and any(b["totals"][V8_BLIND]["cells"] for b in ext.values())
    worse_ext = worse(V8_BLIND, V7_BLIND, False)
    criteria["S4_extension_gain"] = {
        "rule": "on extension cells, pooled per noise level, v8 passes more frames than v7 blind with McNemar p < 0.05",
        "pass": bool(ext_ok), "extension_cells_where_v8_worse_disclosed": cells(worse_ext),
    }
    criteria["all_cells_where_v8_significantly_worse_than_v7"] = cells(worse(V8_BLIND, V7_BLIND, None))
    criteria["improvement_real"] = bool(criteria.get("S1_weak_cell", {}).get("pass") and
                                        criteria["S2_no_v7_grid_regression"]["pass"])
    summary["success_criteria"] = criteria
    return summary


def run_benchmark(
    *,
    trials_per_vector: int = TRIALS_PER_VECTOR,
    master_seed: int = MASTER_SEED,
    cfo_levels: Iterable[float] = CFO_LEVELS,
    clock_levels_ppm: Iterable[float] = CLOCK_LEVELS_PPM,
    timing_offsets: Iterable[float] = TIMING_OFFSETS,
    noise_levels: Iterable[float] = NOISE_LEVELS,
    chunk_frames: int = CHUNK_FRAMES,
    workers: int | None = None,
) -> dict[str, Any]:
    trials = _positive_int(trials_per_vector, "trials_per_vector")
    chunk = _positive_int(chunk_frames, "chunk_frames")
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(master_seed, (int, np.integer)):
        raise ValueError("master_seed must be an integer")
    if int(master_seed) in PRIOR_SEEDS or int(master_seed) == DEV_SEED:
        raise ValueError("master seed must differ from prior protocol and development seeds")
    cfo_values = tuple(_finite(x, "CFO") for x in cfo_levels)
    ppm_values = tuple(_finite(x, "clock error") for x in clock_levels_ppm)
    timing_values = tuple(_finite(x, "timing offset") for x in timing_offsets)
    sigmas = tuple(_finite(x, "noise level") for x in noise_levels)
    for grid in (cfo_values, ppm_values, timing_values, sigmas):
        if not grid or len(set(grid)) != len(grid):
            raise ValueError("condition grids must be non-empty and unique")
    if any(x < 0 for x in sigmas):
        raise ValueError("noise levels must be non-negative")
    if any(abs(x) > CFO_BOUND for x in cfo_values) or any(abs(x) > CLOCK_BOUND_PPM for x in ppm_values):
        raise ValueError("impairment grid must stay inside the blind search envelope")
    if any(abs(x) > TIMING_BOUND for x in timing_values):
        raise ValueError("timing grid must stay inside the blind search envelope")
    cell_specs = [(f, p, d, s) for f in cfo_values for p in ppm_values for d in timing_values for s in sigmas]
    jobs = [(int(master_seed), i, spec, trials, chunk) for i, spec in enumerate(cell_specs)]
    n_workers = int(workers) if workers is not None else min(len(jobs), os.cpu_count() or 1)
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            outputs = list(pool.map(_run_cell, jobs))
    else:
        outputs = [_run_cell(job) for job in jobs]
    rows = [r for out in outputs for r in out["rows"]]
    paired = [r for out in outputs for r in out["paired"]]
    full = (trials == TRIALS_PER_VECTOR and int(master_seed) == MASTER_SEED and cfo_values == CFO_LEVELS
            and ppm_values == CLOCK_LEVELS_PPM and timing_values == TIMING_OFFSETS and sigmas == NOISE_LEVELS)
    contract = {
        "classification": "synthetic dimensionless benchmark; no physical validity claimed",
        "protocol_status": "exploratory v8 protocol locked before the full run; not formal preregistration",
        "protocol_lock": PROTOCOL_LOCK,
        "master_seed": int(master_seed), "development_seed_not_used_here": DEV_SEED, "vector_seed": VECTOR_SEED,
        "trials_per_vector": trials, "full_protocol_run": full,
        "cfo_cycles_per_symbol": list(cfo_values), "clock_error_ppm": list(ppm_values),
        "timing_offset_samples": list(timing_values), "noise_std": list(sigmas),
        "weak_cell": list(WEAK_CELL), "tolerance": TOLERANCE,
        "v8_search": {
            "metric": "Gaussian GLRT on raw received samples: sum_k (y.s_k)^2/(s_k.s_k)",
            "coarse_timing": list(COARSE_TIMING), "coarse_clock_ppm": list(COARSE_CLOCK_PPM),
            "coarse_cfo": list(COARSE_CFO), "top_k": TOP_K, "refine_levels": REFINE_LEVELS,
            "first_refine_steps": list(REFINE_STEPS),
            "refine_coordinates": "timing, clock ppm, carrier offset g=(4+cfo)(1+ppm*1e-6)-4",
            "envelope": {"timing": TIMING_BOUND, "clock_ppm": CLOCK_BOUND_PPM, "cfo": CFO_BOUND},
            "decoder": "least-squares signed amplitudes of the raw-sample fit at the chosen parameters",
            "score_margin_relative": SCORE_MARGIN,
        },
        "acquisition_rules": {
            "loose": {"timing": LOOSE_TOLERANCE[0], "clock_ppm": LOOSE_TOLERANCE[1], "cfo": LOOSE_TOLERANCE[2],
                      "note": "v7 rule, within one coarse grid step"},
            "strict": {"timing": STRICT_TOLERANCE[0], "clock_ppm": STRICT_TOLERANCE[1], "cfo": STRICT_TOLERANCE[2],
                       "note": "within half a coarse grid step"},
        },
        "model": "unchanged v7 received_buffer: t=(m-guard)*(1+clock_error_ppm*1e-6)+timing_offset",
        "receivers": {
            NOMINAL: "unchanged decode_wave with fixed windows and no correction",
            ORACLE: "v7 oracle: exact impairments, linear resampling and least-squares carrier projection",
            ORACLE_LS: "informed reference for the v8 decoder: exact impairments, raw-sample least-squares fit",
            V7_BLIND: "unchanged v7 blind_joint_search (125-point grid, sum of squared decoded values)",
            V8_BLIND: "GLRT metric, v7 coarse grid, top-3 coarse-to-fine ridge-coordinate refinement, raw LS decoder",
        },
        "paired_methods": "all receivers use identical noisy buffers; exact two-sided McNemar tests",
        "module_sha256": module_hashes(), **git_provenance(),
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        },
        "limitations": [
            "synthetic model only; no physical claim",
            "known frame and symbol lengths, known waveform family and noise-only guards",
            "static CFO and clock error, linear interpolation, no pulse shaping, jitter, filtering "
            "or time-varying impairments",
            "blind searches are bounded by the declared envelope; the oracle is informed of impairments",
            "integer timing offsets only in the impairment grid",
        ],
    }
    result = {"benchmark": BENCHMARK, "contract": contract, "summary": summarize(rows, paired, full),
              "results": rows, "paired_comparisons": paired}
    validate_result(result)
    return result


def validate_result(result: dict[str, Any]) -> None:
    if result.get("benchmark") != BENCHMARK:
        raise ValueError("unexpected benchmark identifier")
    contract, rows, paired = result.get("contract"), result.get("results"), result.get("paired_comparisons")
    if not isinstance(contract, dict) or not isinstance(rows, list) or not rows or not isinstance(paired, list):
        raise ValueError("result requires a contract, result rows and paired comparisons")
    if contract.get("protocol_lock") != PROTOCOL_LOCK:
        raise ValueError("protocol lock reference is missing")
    if "no physical validity claimed" not in str(contract.get("classification", "")):
        raise ValueError("synthetic/no-physical-validity label is required")
    if contract.get("module_sha256") != module_hashes():
        raise ValueError("module hashes do not match the current source files")
    seed = contract.get("master_seed")
    if seed in PRIOR_SEEDS or seed == DEV_SEED:
        raise ValueError("master seed reuses a prior or development seed")
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
        if row["wilson95_lower"] is not None:
            low, high = wilson_interval(row["frames_passing"], row["frames"])
            if not (math.isclose(row["wilson95_lower"], low, abs_tol=1e-12)
                    and math.isclose(row["wilson95_upper"], high, abs_tol=1e-12)):
                raise ValueError("Wilson interval does not agree with counts")
        if (row["method"] in BLIND_METHODS) != (row["joint_acq_loose_frames"] is not None):
            raise ValueError("acquisition fields are reported for blind receivers only")
        if row["method"] in BLIND_METHODS:
            for label in ("loose", "strict"):
                joint = row[f"joint_acq_{label}_frames"]
                parts = [row[f"{p}_acq_{label}_frames"] for p in ("timing", "clock", "cfo")]
                if not 0 <= joint <= min(parts) or max(parts) > row["frames"]:
                    raise ValueError("acquisition counts are inconsistent")
            if row["joint_acq_strict_frames"] > row["joint_acq_loose_frames"]:
                raise ValueError("strict acquisition cannot exceed loose acquisition")
        if not all(math.isfinite(float(row[f])) for f in ("empirical_mae", "empirical_rmse", "mean_bias",
                                                          "max_abs_error")):
            raise ValueError("metrics must be finite")
    pass_lookup = {k: r["frames_passing"] for k, r in zip(
        (tuple(r[k] for k in ("cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples", "noise_std",
                              "vector_pattern", "method")) for r in rows), rows, strict=True)}
    for row in paired:
        if row["a_pass_b_fail"] + row["a_fail_b_pass"] > row["frames"]:
            raise ValueError("paired discordant count exceeds frame count")
        base = tuple(row[k] for k in ("cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples",
                                      "noise_std", "vector_pattern"))
        a, b = pass_lookup.get((*base, row["method_a"])), pass_lookup.get((*base, row["method_b"]))
        if a is None or b is None or a - b != row["a_pass_b_fail"] - row["a_fail_b_pass"]:
            raise ValueError("paired counts do not agree with pass counts")
        if not math.isclose(row["mcnemar_exact_two_sided_p"],
                            v7._mcnemar(row["a_pass_b_fail"], row["a_fail_b_pass"]), rel_tol=1e-12):
            raise ValueError("McNemar p-value does not agree with discordant counts")


def _condition_label(row: dict[str, Any]) -> str:
    return f"{row['cfo_cycles_per_symbol']:+g}/{row['clock_error_ppm']:+g}/{row['timing_offset_samples']:+g}"


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
    sigma = max(r["noise_std"] for r in result["results"])
    pooled = [r for r in result["results"] if r["vector_pattern"] == POOLED and r["noise_std"] == sigma]
    conditions = sorted({(r["cfo_cycles_per_symbol"], r["clock_error_ppm"], r["timing_offset_samples"])
                         for r in pooled}, key=lambda c: (c[1], c[0], c[2]))

    def pick(method: str) -> list[dict[str, Any]]:
        lookup = {(r["cfo_cycles_per_symbol"], r["clock_error_ppm"], r["timing_offset_samples"]): r
                  for r in pooled if r["method"] == method}
        return [lookup[c] for c in conditions]

    x = np.arange(len(conditions))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, sharex=True)
    for method, marker in ((ORACLE, "s"), (V7_BLIND, "D"), (V8_BLIND, "o")):
        selected = pick(method)
        failing = [r["frames"] - r["frames_passing"] for r in selected]
        axes[0].plot(x, [max(f, 0.5) for f in failing], marker=marker, ms=3, lw=0.8, label=method)
    for method, marker in ((V7_BLIND, "D"), (V8_BLIND, "o")):
        selected = pick(method)
        axes[1].plot(x, [100 * r["joint_acq_strict_share"] for r in selected], marker=marker, ms=3, lw=0.8,
                     label=f"{method} (strict joint)")
    weak = [i for i, c in enumerate(conditions) if (*c, sigma) == WEAK_CELL]
    for ax in axes:
        for i in weak:
            ax.axvline(i, color="0.6", ls=":", lw=1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for boundary in [i for i in range(1, len(conditions)) if conditions[i][1] != conditions[i - 1][1]]:
        for ax in axes:
            ax.axvline(boundary - 0.5, color="0.85", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set(ylabel="frames failing of 1,200 (log; 0 drawn at 0.5)",
                title=f"Decode failures (σ = {sigma:g}); dotted line = v7 weak cell")
    axes[1].set(ylabel="joint acquisition (%)", ylim=(-2, 102),
                title="Blind joint acquisition, strict rule (1 sample, 1250 ppm, 0.0025 cycles/symbol)")
    axes[1].set_xticks(x, [_condition_label(r) for r in pick(ORACLE)], rotation=90, fontsize=5)
    axes[1].set_xlabel("CFO (cycles/symbol) / clock error (ppm) / timing offset (samples), grouped by clock error")
    fig.suptitle("v8 GLRT + coarse-to-fine blind sync vs v7 — SYNTHETIC; no physical validity claimed")
    fig.savefig(paths["png"], dpi=150, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return paths


def main() -> int:
    result = run_benchmark()
    paths = write_outputs(result)
    print(json.dumps(result["summary"].get("weak_cell"), indent=1))
    print(json.dumps(result["summary"]["success_criteria"], indent=1))
    print(f"wrote {paths['json']}, {paths['csv']}, {paths['paired_csv']} and {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
