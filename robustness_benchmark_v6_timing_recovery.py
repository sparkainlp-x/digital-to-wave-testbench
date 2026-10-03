"""Exploratory v6 blind timing-recovery benchmark for the synthetic codec (SYNTHETIC).

v5 showed that the unchanged nominal matched-filter decoder collapses once its
64-sample windows start two or more samples away from the true symbol
boundaries, and that an oracle told the exact offset recovers. v6 asks the
obvious follow-up question: does a standard *blind* timing search, one that is
never told the offset, recover oracle-level performance in this model?

The v1-v5 modules are imported and used unchanged (they are hash-locked in
MODULES.sha256). Every symbol estimate is produced by the unchanged
``digital_to_wave.decode_wave`` projection; v6 only chooses *which* 64-sample
windows it is handed.

Panels (all dimensionless; no physical validity claimed):

1. Integer offsets, six fixed v2 vectors, clean and AWGN SD 0.45: nominal decoder,
   exact-offset oracle, a blind matched-filter-energy window search and a blind
   raw frame-energy window search over candidate integer starts -16..+16.
2. Fractional (sub-sample) offsets: the continuous p.12 waveform is sampled at
   n + Delta; the integer search can at best reach the nearest integer start.
3. Randomised per-frame vectors (U[-2, 2] and the integer alphabet {-2..2}) as a
   sensitivity panel, with the 0.25 pass rule and a round-to-integer pass rule.
4. A clean, deterministic sweep of integer offsets -32..+32.
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from digital_to_wave import WaveConfig, decode_wave, encode_wave  # noqa: E402
from robustness_benchmark_v2 import MASTER_SEED as V2_MASTER_SEED  # noqa: E402
from robustness_benchmark_v2 import VECTOR_SEED, build_input_patterns, wilson_interval  # noqa: E402
from robustness_benchmark_v3 import MASTER_SEED as V3_MASTER_SEED  # noqa: E402
from robustness_benchmark_v4 import MASTER_SEED as V4_MASTER_SEED  # noqa: E402
from robustness_benchmark_v5_timing import MASTER_SEED as V5_MASTER_SEED  # noqa: E402
from robustness_benchmark_v5_timing import theoretical_frame_pass_probability  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
BENCHMARK_ID = "synthetic_blind_timing_recovery_v6"
ARTIFACT_STEM = "robustness-benchmark-v6-timing-recovery"
LOCKED_MODULES = (
    "digital_to_wave.py",
    "robustness_benchmark_v2.py",
    "robustness_benchmark_v3.py",
    "robustness_benchmark_v4.py",
    "robustness_benchmark_v5_timing.py",
)

# ---------------------------------------------------------------------------
# Frozen v6 protocol (see reports/v6-timing-recovery-protocol-lock.md).
# ---------------------------------------------------------------------------
MASTER_SEED = 20261009
TRIALS_PER_VECTOR_PER_CONDITION = 2_000
RANDOM_FRAMES_PER_OFFSET = 12_000
CHUNK_FRAMES = 128
NOISE_STD = 0.45
NOISE_STD_LEVELS = (0.0, NOISE_STD)
TOLERANCE = 0.25
VECTOR_LENGTH = 32
SAMPLES_PER_SYMBOL = 64
CYCLES_PER_SYMBOL = 4
AMPLITUDE_MIN = 1.0
GUARD_SAMPLES = 64
FRAME_SAMPLES = VECTOR_LENGTH * SAMPLES_PER_SYMBOL
STREAM_SAMPLES = 2 * GUARD_SAMPLES + FRAME_SAMPLES
SEARCH_CANDIDATES = tuple(range(-16, 17))
INTEGER_OFFSETS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)  # identical to the v5 grid
FRACTIONAL_OFFSETS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
CLEAN_SWEEP_OFFSETS = tuple(range(-32, 33))
RANDOM_DISTRIBUTIONS = ("random_uniform_-2_2", "random_integer_alphabet_-2..2")

FIXED_SET = "fixed_six_v2_vectors"
POOLED_PATTERN = "POOLED_ALL_VECTORS"
PER_FRAME_PATTERN = "PER_FRAME_RANDOM"
EXPECTED_PATTERN_NAMES = (
    "mixed_signs", "sparse_and_zeros", "boundary_values", "randomized_1", "randomized_2", "randomized_3",
)
NOMINAL = "existing_nominal_decoder"
ORACLE = "oracle_exact_timing"
ORACLE_BEST_INT = "oracle_best_integer_timing"
MF_SEARCH = "blind_matched_filter_energy_search"
ENERGY_SEARCH = "blind_raw_frame_energy_search"
SEARCH_METHODS = (MF_SEARCH, ENERGY_SEARCH)
INTEGER_METHODS = (NOMINAL, ORACLE, MF_SEARCH, ENERGY_SEARCH)
FRACTIONAL_METHODS = (NOMINAL, ORACLE_BEST_INT, MF_SEARCH)
RANDOM_METHODS = (NOMINAL, ORACLE, MF_SEARCH)
CLEAN_SWEEP_METHODS = (NOMINAL, ORACLE, MF_SEARCH, ENERGY_SEARCH)
RULE_TOL = "abs_error_le_0.25"
RULE_ROUND = "round_to_integer_exact"
PANEL_INTEGER = "integer_offsets"
PANEL_FRACTIONAL = "fractional_offsets"
PANEL_RANDOM = "random_vectors"
PAIRS_INTEGER = ((MF_SEARCH, ORACLE), (MF_SEARCH, NOMINAL), (ENERGY_SEARCH, ORACLE), (ENERGY_SEARCH, MF_SEARCH))
PAIRS_FRACTIONAL = ((MF_SEARCH, ORACLE_BEST_INT), (MF_SEARCH, NOMINAL))
PAIRS_RANDOM = ((MF_SEARCH, ORACLE), (MF_SEARCH, NOMINAL))

ROW_FIELDS = (
    "panel", "vector_set", "vector_pattern", "offset_samples", "offset_percent_of_symbol", "noise_std", "method",
    "pass_rule", "frames", "frames_passing", "frame_pass_share", "wilson95_lower", "wilson95_upper",
    "failure_share", "failure_wilson95_lower", "failure_wilson95_upper", "timing_acquired_frames",
    "timing_acquisition_share", "acquisition_wilson95_lower", "acquisition_wilson95_upper",
    "theory_frame_pass_probability", "empirical_mae", "empirical_rmse", "empirical_mean_bias",
    "uncertainty_kind", "evidence_label",
)
PAIR_FIELDS = (
    "panel", "vector_set", "offset_samples", "noise_std", "pass_rule", "method_a", "method_b", "frames",
    "a_passing", "b_passing", "both_pass", "both_fail", "a_pass_b_fail", "a_fail_b_pass",
    "mcnemar_exact_two_sided_p", "evidence_label",
)
SWEEP_FIELDS = (
    "offset_samples", "offset_percent_of_symbol", "carrier_rotation_degrees", "within_search_range", "method",
    "frames", "frames_passing", "empirical_mae", "max_abs_error", "chosen_start_errors", "evidence_label",
)

CONFIG = WaveConfig(SAMPLES_PER_SYMBOL, CYCLES_PER_SYMBOL, AMPLITUDE_MIN)
OMEGA = 2.0 * math.pi * CYCLES_PER_SYMBOL / SAMPLES_PER_SYMBOL
NO_PHYSICAL_CLAIM = "no physical validity claimed"


# ---------------------------------------------------------------------------
# Small validators and statistics.
# ---------------------------------------------------------------------------
def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _integer_tuple(values: Iterable[int], label: str, *, require_zero: bool, bound: int) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty iterable of integers") from exc
    if not raw or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in raw):
        raise ValueError(f"{label} must be a non-empty iterable of integers")
    result = tuple(int(v) for v in raw)
    if len(set(result)) != len(result) or (require_zero and 0 not in result):
        raise ValueError(f"{label} must be unique" + (" and include zero" if require_zero else ""))
    if max(abs(v) for v in result) > bound:
        raise ValueError(f"{label} exceed the allowed range of +-{bound} samples")
    return result


def _fractional_tuple(values: Iterable[float]) -> tuple[float, ...]:
    try:
        result = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("fractional offsets must be a non-empty iterable of finite numbers") from exc
    if not result or len(set(result)) != len(result) or 0.0 not in result:
        raise ValueError("fractional offsets must be non-empty, unique and include zero")
    if any(not math.isfinite(v) or abs(v) > max(SEARCH_CANDIDATES) for v in result):
        raise ValueError("fractional offsets must be finite and inside the search range")
    return result


def mcnemar_exact_p(a_pass_b_fail: int, a_fail_b_pass: int) -> float:
    """Exact two-sided McNemar (binomial) p-value from the two discordant counts."""
    b, c = int(a_pass_b_fail), int(a_fail_b_pass)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def best_integer_correction(delta: float, patterns: dict[str, np.ndarray], noise_std: float = NOISE_STD) -> int:
    """Analysis-only: the integer start correction (-floor or -ceil of delta) with the higher pooled analytic
    pass probability on the fixed vectors at the given noise level; ties go to -floor(delta).

    Halves are not symmetric here: sampling at n + 0.5 keeps every sample inside its own symbol interval
    [64k, 64k + 64), whereas n - 0.5 puts sample 0 into the previous symbol.
    """
    options = sorted({-math.floor(delta), -math.ceil(delta)}, key=lambda c: c != -math.floor(delta))
    if len(options) == 1:
        return int(options[0])
    scores = []
    for c in options:
        probabilities = [
            theoretical_frame_pass_probability(decode_at(fractional_offset_stream(ref, delta), c)[0] - ref, noise_std)
            for ref in patterns.values()
        ]
        scores.append(float(np.mean(probabilities)))
    return int(options[int(np.argmax(scores))])


def oracle_round_rule_probability(noise_std: float = NOISE_STD) -> float:
    """Closed-form oracle frame probability for the round-to-integer rule (|error| < 0.5)."""
    sd = noise_std / math.sqrt(32.0)
    return math.erf(0.5 / (sd * math.sqrt(2.0))) ** VECTOR_LENGTH


# ---------------------------------------------------------------------------
# Waveform construction (bitwise identical to the hash-locked encoder).
# ---------------------------------------------------------------------------
def _signed_templates() -> tuple[np.ndarray, np.ndarray]:
    positive = encode_wave([1.0], CONFIG)[0]
    negative = encode_wave([-1.0], CONFIG)[0]
    return positive, negative


_POSITIVE, _NEGATIVE = _signed_templates()


def encode_frames(values: Any) -> np.ndarray:
    """Vectorised serialisation of the p.12 encoder; equals encode_wave(row).reshape(-1) bit for bit."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[1] != VECTOR_LENGTH or not np.all(np.isfinite(v)) or np.any(np.abs(v) > 2.0):
        raise ValueError("values must be a finite (frames, 32) array inside [-2, 2]")
    magnitude = np.abs(v)[:, :, None]
    sign = v[:, :, None]
    segments = np.where(sign > 0.0, magnitude * _POSITIVE, np.where(sign < 0.0, magnitude * _NEGATIVE, 0.0))
    return segments.reshape(v.shape[0], FRAME_SAMPLES)


def integer_offset_stream(values: Any, offset: int) -> np.ndarray:
    """Receiver buffer y[m] = x[m + offset], x = zero-guarded serialised frame (v5 convention: + = late)."""
    if isinstance(offset, (bool, np.bool_)) or not isinstance(offset, (int, np.integer)):
        raise ValueError("offset must be an integer")
    d = int(offset)
    if abs(d) > GUARD_SAMPLES:
        raise ValueError("absolute offset must not exceed the guard length")
    frames = encode_frames(values)
    stream = np.zeros((frames.shape[0], STREAM_SAMPLES), dtype=float)
    stream[:, GUARD_SAMPLES - d:GUARD_SAMPLES - d + FRAME_SAMPLES] = frames
    return stream


def continuous_waveform(values: Any, t: Any) -> np.ndarray:
    """Evaluate the p.12 continuous signal S(t) of one frame; symbol k occupies t in [64k, 64k+64)."""
    v = np.asarray(values, dtype=float)
    if v.shape != (VECTOR_LENGTH,) or not np.all(np.isfinite(v)) or np.any(np.abs(v) > 2.0):
        raise ValueError("values must be a finite 32-vector inside [-2, 2]")
    time = np.asarray(t, dtype=float)
    k = np.floor(time / SAMPLES_PER_SYMBOL).astype(int)
    inside = (k >= 0) & (k < VECTOR_LENGTH)
    kk = np.clip(k, 0, VECTOR_LENGTH - 1)
    local = time - SAMPLES_PER_SYMBOL * kk
    value = v[kk]
    phi = np.where(value > 0.0, np.pi / 4.0, np.where(value < 0.0, 5.0 * np.pi / 4.0, 0.0))
    signal = np.abs(value) * AMPLITUDE_MIN * np.sin(OMEGA * local + phi)
    return np.where(inside, signal, 0.0)


def fractional_offset_stream(values: Any, delta: float) -> np.ndarray:
    """Receiver buffer y[m] = S(m - guard + delta) for each frame row (sub-sample timing phase)."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[None, :]
    d = float(delta)
    if not math.isfinite(d) or abs(d) > GUARD_SAMPLES:
        raise ValueError("delta must be finite and within the guard length")
    t = np.arange(STREAM_SAMPLES, dtype=float) - GUARD_SAMPLES + d
    return np.stack([continuous_waveform(row, t) for row in v])


# ---------------------------------------------------------------------------
# Receivers. All symbol estimates come from the unchanged decode_wave.
# ---------------------------------------------------------------------------
def decode_at(buffer: np.ndarray, start_correction: int) -> np.ndarray:
    """Decode all 32 symbols with windows starting at guard + 64k + start_correction."""
    b = np.asarray(buffer, dtype=float)
    start = GUARD_SAMPLES + int(start_correction)
    if b.ndim != 2 or b.shape[1] != STREAM_SAMPLES or start < 0 or start + FRAME_SAMPLES > STREAM_SAMPLES:
        raise ValueError("buffer shape or start correction is invalid")
    windows = b[:, start:start + FRAME_SAMPLES].reshape(-1, SAMPLES_PER_SYMBOL)
    return decode_wave(windows, CONFIG).reshape(b.shape[0], VECTOR_LENGTH)


def _tie_break_order(candidates: tuple[int, ...]) -> np.ndarray:
    """Candidates sorted by (|c|, c): exact metric ties resolve toward the nominal start."""
    return np.asarray(sorted(candidates, key=lambda c: (abs(c), c)), dtype=int)


def all_candidate_decodes(buffer: np.ndarray,
                          candidates: tuple[int, ...] = SEARCH_CANDIDATES) -> tuple[np.ndarray, np.ndarray]:
    """Return (ordered candidates, decodes with shape (frames, n_candidates, 32))."""
    order = _tie_break_order(tuple(candidates))
    decodes = np.stack([decode_at(buffer, int(c)) for c in order], axis=1)
    return order, decodes


def matched_filter_energy_search(
    buffer: np.ndarray,
    candidates: tuple[int, ...] = SEARCH_CANDIDATES,
    _decodes: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blind search: pick the start correction maximising sum_k decode_k^2; never told the offset.

    Returns (chosen start corrections, decoded estimates at the chosen start).
    """
    order, decodes = _decodes if _decodes is not None else all_candidate_decodes(buffer, candidates)
    metric = np.sum(decodes * decodes, axis=2)
    index = np.argmax(metric, axis=1)
    rows = np.arange(decodes.shape[0])
    return order[index], decodes[rows, index]


def raw_energy_search(buffer: np.ndarray, candidates: tuple[int, ...] = SEARCH_CANDIDATES) -> np.ndarray:
    """Blind search: pick the start correction maximising received energy in the 2048-sample frame span.

    The energy change relative to the nominal span is summed over the |c| samples that enter and leave,
    so exact ties (e.g. a clean frame whose edge symbol is zero) stay exact rather than roundoff-dependent.
    """
    b = np.asarray(buffer, dtype=float)
    order = _tie_break_order(tuple(candidates))
    squared = b * b
    lo, hi = GUARD_SAMPLES, GUARD_SAMPLES + FRAME_SAMPLES
    gains = []
    for c in order:
        c = int(c)
        if c > 0:
            gains.append(squared[:, hi:hi + c].sum(axis=1) - squared[:, lo:lo + c].sum(axis=1))
        elif c < 0:
            gains.append(squared[:, lo + c:lo].sum(axis=1) - squared[:, hi + c:hi].sum(axis=1))
        else:
            gains.append(np.zeros(b.shape[0]))
    return order[np.argmax(np.stack(gains, axis=1), axis=1)]


# ---------------------------------------------------------------------------
# Row helpers.
# ---------------------------------------------------------------------------
def _passes(errors: np.ndarray, estimates: np.ndarray, reference: np.ndarray, rule: str) -> np.ndarray:
    if rule == RULE_TOL:
        return np.all(np.abs(errors) <= TOLERANCE, axis=1)
    if rule == RULE_ROUND:
        return np.all(np.round(estimates) == reference, axis=1)
    raise ValueError("unknown pass rule")


def _make_row(panel: str, vector_set: str, pattern: str, offset: float, sigma: float, method: str, rule: str,
              passing: np.ndarray, errors: np.ndarray, acquired: np.ndarray | None,
              theory: float | None) -> dict[str, Any]:
    frames = int(passing.size)
    successes = int(np.count_nonzero(passing))
    stochastic = sigma > 0.0
    if stochastic:
        lo, hi = wilson_interval(successes, frames)
        flo, fhi = 1.0 - hi, 1.0 - lo
        kind = "two-sided 95% Wilson interval; conditional on the vector panel and synthetic IID draws"
    else:
        lo = hi = flo = fhi = None
        kind = "deterministic; no sampling interval"
    acq_frames = acq_share = alo = ahi = None
    if acquired is not None:
        acq_frames = int(np.count_nonzero(acquired))
        acq_share = acq_frames / frames
        if stochastic:
            alo, ahi = wilson_interval(acq_frames, frames)
    return {
        "panel": panel, "vector_set": vector_set, "vector_pattern": pattern,
        "offset_samples": float(offset), "offset_percent_of_symbol": 100.0 * float(offset) / SAMPLES_PER_SYMBOL,
        "noise_std": float(sigma), "method": method, "pass_rule": rule,
        "frames": frames, "frames_passing": successes, "frame_pass_share": successes / frames,
        "wilson95_lower": lo, "wilson95_upper": hi,
        "failure_share": (frames - successes) / frames, "failure_wilson95_lower": flo, "failure_wilson95_upper": fhi,
        "timing_acquired_frames": acq_frames, "timing_acquisition_share": acq_share,
        "acquisition_wilson95_lower": alo, "acquisition_wilson95_upper": ahi,
        "theory_frame_pass_probability": None if theory is None else float(theory),
        "empirical_mae": float(np.mean(np.abs(errors))), "empirical_rmse": float(np.sqrt(np.mean(errors * errors))),
        "empirical_mean_bias": float(np.mean(errors)),
        "uncertainty_kind": kind, "evidence_label": "SYNTHETIC",
    }


def _pair_row(panel: str, vector_set: str, offset: float, sigma: float, rule: str, method_a: str, method_b: str,
              pass_a: np.ndarray, pass_b: np.ndarray) -> dict[str, Any]:
    b = int(np.count_nonzero(pass_a & ~pass_b))
    c = int(np.count_nonzero(~pass_a & pass_b))
    return {
        "panel": panel, "vector_set": vector_set, "offset_samples": float(offset), "noise_std": float(sigma),
        "pass_rule": rule, "method_a": method_a, "method_b": method_b, "frames": int(pass_a.size),
        "a_passing": int(np.count_nonzero(pass_a)), "b_passing": int(np.count_nonzero(pass_b)),
        "both_pass": int(np.count_nonzero(pass_a & pass_b)), "both_fail": int(np.count_nonzero(~pass_a & ~pass_b)),
        "a_pass_b_fail": b, "a_fail_b_pass": c, "mcnemar_exact_two_sided_p": mcnemar_exact_p(b, c),
        "evidence_label": "SYNTHETIC",
    }


def _batches(total: int, chunk: int) -> Iterable[int]:
    done = 0
    while done < total:
        size = min(chunk, total - done)
        yield size
        done += size


def _fixed_panel(panel: str, seed_tag: int, seed: int, trials: int, chunk: int, levels: tuple[float, ...],
                 patterns: dict[str, np.ndarray], candidates: tuple[int, ...]) -> tuple[list[dict], list[dict], dict]:
    """Shared engine for the integer and fractional fixed-vector panels."""
    integer = panel == PANEL_INTEGER
    methods = INTEGER_METHODS if integer else FRACTIONAL_METHODS
    reference_method = ORACLE if integer else ORACLE_BEST_INT
    search_methods = SEARCH_METHODS if integer else (MF_SEARCH,)
    pair_list = PAIRS_INTEGER if integer else PAIRS_FRACTIONAL
    rows: list[dict] = []
    pairs: list[dict] = []
    histograms: dict[str, dict[str, dict[str, int]]] = {m: {} for m in search_methods}
    for li, level in enumerate(levels):
        correction = -int(level) if integer else best_integer_correction(level, patterns)
        acquired_set = np.asarray([correction])
        for sigma in NOISE_STD_LEVELS:
            pooled_pass: dict[str, list[np.ndarray]] = {m: [] for m in methods}
            pooled_err: dict[str, list[np.ndarray]] = {m: [] for m in methods}
            pooled_chosen: dict[str, list[np.ndarray]] = {m: [] for m in search_methods}
            pooled_theory: dict[str, list[float]] = {NOMINAL: [], reference_method: []}
            for pi, (name, reference) in enumerate(patterns.items()):
                clean = (integer_offset_stream(reference[None, :], int(level)) if integer
                         else fractional_offset_stream(reference, level))
                theory = {
                    NOMINAL: theoretical_frame_pass_probability(decode_at(clean, 0)[0] - reference, sigma),
                    reference_method: theoretical_frame_pass_probability(
                        decode_at(clean, correction)[0] - reference, sigma),
                }
                rng = np.random.default_rng(np.random.SeedSequence([seed, seed_tag, li, pi]))
                total = 1 if sigma == 0.0 else trials
                passes: dict[str, list[np.ndarray]] = {m: [] for m in methods}
                errs: dict[str, list[np.ndarray]] = {m: [] for m in methods}
                chosen: dict[str, list[np.ndarray]] = {m: [] for m in search_methods}
                for size in _batches(total, chunk):
                    received = np.repeat(clean, size, axis=0)
                    if sigma > 0.0:
                        received = received + rng.normal(0.0, sigma, size=received.shape)
                    order, decodes = all_candidate_decodes(received, candidates)
                    position = {int(c): i for i, c in enumerate(order)}
                    c_mf, est_mf = matched_filter_energy_search(received, candidates, _decodes=(order, decodes))
                    estimates = {NOMINAL: decodes[:, position[0]], reference_method: decodes[:, position[correction]],
                                 MF_SEARCH: est_mf}
                    chosen[MF_SEARCH].append(c_mf)
                    if integer:
                        c_en = raw_energy_search(received, candidates)
                        estimates[ENERGY_SEARCH] = decodes[np.arange(size), [position[int(c)] for c in c_en]]
                        chosen[ENERGY_SEARCH].append(c_en)
                    for m in methods:
                        e = estimates[m] - reference[None, :]
                        errs[m].append(e)
                        passes[m].append(_passes(e, estimates[m], reference[None, :], RULE_TOL))
                for m in methods:
                    p, e = np.concatenate(passes[m]), np.concatenate(errs[m])
                    acq = None
                    if m in search_methods:
                        cm = np.concatenate(chosen[m])
                        acq = np.isin(cm, acquired_set)
                        pooled_chosen[m].append(cm)
                    rows.append(_make_row(panel, FIXED_SET, name, level, sigma, m, RULE_TOL, p, e, acq, theory.get(m)))
                    pooled_pass[m].append(p)
                    pooled_err[m].append(e)
                for m in pooled_theory:
                    pooled_theory[m].append(theory[m])
            pooled = {m: np.concatenate(pooled_pass[m]) for m in methods}
            for m in methods:
                acq = np.isin(np.concatenate(pooled_chosen[m]), acquired_set) if m in search_methods else None
                th = float(np.mean(pooled_theory[m])) if m in pooled_theory else None
                rows.append(_make_row(panel, FIXED_SET, POOLED_PATTERN, level, sigma, m, RULE_TOL, pooled[m],
                                      np.concatenate(pooled_err[m]), acq, th))
            if sigma > 0.0:
                for a, b in pair_list:
                    pairs.append(_pair_row(panel, FIXED_SET, level, sigma, RULE_TOL, a, b, pooled[a], pooled[b]))
            for m in search_methods:
                # chosen correction + level = residual start error in samples (0 = exact for integer offsets)
                residual = np.round(np.concatenate(pooled_chosen[m]) + float(level), 6)
                values, counts = np.unique(residual, return_counts=True)
                histograms[m][f"offset={float(level):+g};sigma={sigma:g}"] = {
                    f"{float(v):+g}": int(n) for v, n in zip(values, counts, strict=True)
                }
    return rows, pairs, histograms


def _fractional_analytic_curve(patterns: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Dense analytic curves (no Monte Carlo): nominal decoder and best integer correction (floor/ceil)."""
    curve = []
    for delta in np.round(np.arange(0.0, 8.0 + 1e-9, 0.0625), 6):
        nominal_p: list[float] = []
        nearest_p: dict[int, list[float]] = {}
        for reference in patterns.values():
            clean = fractional_offset_stream(reference, float(delta))
            nominal_p.append(theoretical_frame_pass_probability(decode_at(clean, 0)[0] - reference, NOISE_STD))
            for c in {-math.floor(float(delta)), -math.ceil(float(delta))}:
                nearest_p.setdefault(c, []).append(theoretical_frame_pass_probability(
                    decode_at(clean, c)[0] - reference, NOISE_STD))
        curve.append({"offset_samples": float(delta), "nominal_theory": float(np.mean(nominal_p)),
                      "best_integer_correction_theory": max(float(np.mean(v)) for v in nearest_p.values())})
    return curve


def _draw_vectors(rng: np.random.Generator, distribution: str, size: int) -> np.ndarray:
    if distribution == RANDOM_DISTRIBUTIONS[0]:
        return rng.uniform(-2.0, 2.0, size=(size, VECTOR_LENGTH))
    if distribution == RANDOM_DISTRIBUTIONS[1]:
        return rng.integers(-2, 3, size=(size, VECTOR_LENGTH)).astype(float)
    raise ValueError("unknown random vector distribution")


def _rules_for(distribution: str) -> tuple[str, ...]:
    return (RULE_TOL, RULE_ROUND) if distribution == RANDOM_DISTRIBUTIONS[1] else (RULE_TOL,)


def _run_random_panel(seed: int, frames_per_offset: int, chunk: int, offsets: tuple[int, ...],
                      candidates: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    pairs: list[dict] = []
    oracle_theory = {RULE_TOL: theoretical_frame_pass_probability(np.zeros(VECTOR_LENGTH), NOISE_STD),
                     RULE_ROUND: oracle_round_rule_probability(NOISE_STD)}
    for vi, distribution in enumerate(RANDOM_DISTRIBUTIONS):
        rules = _rules_for(distribution)
        for oi, offset in enumerate(offsets):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3, vi, oi]))
            passes: dict[tuple[str, str], list[np.ndarray]] = {(m, r): [] for m in RANDOM_METHODS for r in rules}
            errs: dict[str, list[np.ndarray]] = {m: [] for m in RANDOM_METHODS}
            acqs = []
            for size in _batches(frames_per_offset, chunk):
                reference = _draw_vectors(rng, distribution, size)
                received = integer_offset_stream(reference, offset)
                received = received + rng.normal(0.0, NOISE_STD, size=received.shape)
                order, decodes = all_candidate_decodes(received, candidates)
                position = {int(c): i for i, c in enumerate(order)}
                c_mf, est_mf = matched_filter_energy_search(received, candidates, _decodes=(order, decodes))
                estimates = {NOMINAL: decodes[:, position[0]], ORACLE: decodes[:, position[-offset]], MF_SEARCH: est_mf}
                for m in RANDOM_METHODS:
                    e = estimates[m] - reference
                    errs[m].append(e)
                    for r in rules:
                        passes[(m, r)].append(_passes(e, estimates[m], reference, r))
                acqs.append(c_mf == -offset)
            acq_all = np.concatenate(acqs)
            for r in rules:
                pooled = {m: np.concatenate(passes[(m, r)]) for m in RANDOM_METHODS}
                for m in RANDOM_METHODS:
                    rows.append(_make_row(PANEL_RANDOM, distribution, PER_FRAME_PATTERN, offset, NOISE_STD, m, r,
                                          pooled[m], np.concatenate(errs[m]), acq_all if m == MF_SEARCH else None,
                                          oracle_theory[r] if m == ORACLE else None))
                for a, b in PAIRS_RANDOM:
                    pairs.append(_pair_row(PANEL_RANDOM, distribution, offset, NOISE_STD, r, a, b, pooled[a],
                                           pooled[b]))
    return rows, pairs


def _run_clean_sweep(offsets: tuple[int, ...], patterns: dict[str, np.ndarray],
                     candidates: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    reference = np.stack(list(patterns.values()))
    for offset in offsets:
        received = integer_offset_stream(reference, offset)
        c_mf, est_mf = matched_filter_energy_search(received, candidates)
        c_en = raw_energy_search(received, candidates)
        estimates = {
            NOMINAL: decode_at(received, 0),
            ORACLE: decode_at(received, -offset),
            MF_SEARCH: est_mf,
            ENERGY_SEARCH: np.stack([decode_at(received[i:i + 1], int(c))[0] for i, c in enumerate(c_en)]),
        }
        chosen = {MF_SEARCH: c_mf, ENERGY_SEARCH: c_en}
        for m in CLEAN_SWEEP_METHODS:
            e = estimates[m] - reference
            passing = np.all(np.abs(e) <= TOLERANCE, axis=1)
            rows.append({
                "offset_samples": int(offset), "offset_percent_of_symbol": 100.0 * offset / SAMPLES_PER_SYMBOL,
                "carrier_rotation_degrees": 22.5 * offset, "within_search_range": bool(-offset in candidates),
                "method": m, "frames": int(reference.shape[0]), "frames_passing": int(np.count_nonzero(passing)),
                "empirical_mae": float(np.mean(np.abs(e))), "max_abs_error": float(np.max(np.abs(e))),
                "chosen_start_errors": (";".join(f"{int(c) + offset:+d}" for c in chosen[m]) if m in chosen else ""),
                "evidence_label": "SYNTHETIC",
            })
    return rows


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------
def module_hashes() -> dict[str, str]:
    """SHA-256 of the five hash-locked modules and of this v6 module."""
    names = (*LOCKED_MODULES, Path(__file__).name)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_tracked_tree_clean_at_run_start": None if status is None else status == ""}


def _forbidden_seeds() -> set[int]:
    return {int(V2_MASTER_SEED), int(V3_MASTER_SEED), int(V4_MASTER_SEED), int(V5_MASTER_SEED), int(VECTOR_SEED)}


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
def run_timing_recovery_benchmark(
    trials_per_vector_per_condition: int = TRIALS_PER_VECTOR_PER_CONDITION,
    random_frames_per_offset: int = RANDOM_FRAMES_PER_OFFSET,
    master_seed: int = MASTER_SEED,
    integer_offsets: Iterable[int] = INTEGER_OFFSETS,
    fractional_offsets: Iterable[float] = FRACTIONAL_OFFSETS,
    clean_sweep_offsets: Iterable[int] = CLEAN_SWEEP_OFFSETS,
    chunk_frames: int = CHUNK_FRAMES,
) -> dict[str, Any]:
    """Run all four v6 panels; paired methods share every noisy buffer."""
    trials = _positive_integer(trials_per_vector_per_condition, "trials_per_vector_per_condition")
    random_frames = _positive_integer(random_frames_per_offset, "random_frames_per_offset")
    chunk = _positive_integer(chunk_frames, "chunk_frames")
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(master_seed, (int, np.integer)):
        raise ValueError("master_seed must be an integer")
    seed = int(master_seed)
    if seed in _forbidden_seeds():
        raise ValueError("v6 master seed must differ from the v2-v5 master seeds and the vector seed")
    candidates = SEARCH_CANDIDATES
    offsets = _integer_tuple(integer_offsets, "integer offsets", require_zero=True, bound=max(candidates))
    deltas = _fractional_tuple(fractional_offsets)
    sweep = _integer_tuple(clean_sweep_offsets, "clean sweep offsets", require_zero=True,
                           bound=GUARD_SAMPLES - max(candidates))
    raw_patterns = build_input_patterns()
    if tuple(raw_patterns) != EXPECTED_PATTERN_NAMES or any(len(v) != VECTOR_LENGTH for v in raw_patterns.values()):
        raise RuntimeError("the fixed six-vector panel no longer matches the v2 protocol")
    patterns = {k: np.asarray(v, dtype=float) for k, v in raw_patterns.items()}
    q = np.sin(OMEGA * np.arange(SAMPLES_PER_SYMBOL) + math.pi / 4.0)
    if not math.isclose(float(q @ q), 32.0, abs_tol=1e-12):
        raise RuntimeError("unexpected template energy")

    provenance = git_provenance()
    int_rows, int_pairs, int_hist = _fixed_panel(PANEL_INTEGER, 1, seed, trials, chunk, offsets, patterns, candidates)
    frac_rows, frac_pairs, frac_hist = _fixed_panel(PANEL_FRACTIONAL, 2, seed, trials, chunk, deltas, patterns,
                                                    candidates)
    rand_rows, rand_pairs = _run_random_panel(seed, random_frames, chunk, offsets, candidates)
    sweep_rows = _run_clean_sweep(sweep, patterns, candidates)
    curve = _fractional_analytic_curve(patterns)

    full = (
        trials == TRIALS_PER_VECTOR_PER_CONDITION and random_frames == RANDOM_FRAMES_PER_OFFSET
        and seed == MASTER_SEED and offsets == INTEGER_OFFSETS and deltas == FRACTIONAL_OFFSETS
        and sweep == CLEAN_SWEEP_OFFSETS and chunk == CHUNK_FRAMES
    )
    noisy_frames = (len(offsets) + len(deltas)) * len(patterns) * trials + len(RANDOM_DISTRIBUTIONS) * len(
        offsets) * random_frames
    contract = {
        "evidence_label": "SYNTHETIC",
        "classification": ("synthetic dimensionless software test of blind integer timing recovery; "
                           f"{NO_PHYSICAL_CLAIM}"),
        "protocol_status": ("v6 exploratory protocol; conditions frozen in source and committed before the full run; "
                            "not a formal preregistration or physical validation"),
        "protocol_lock_document": "reports/v6-timing-recovery-protocol-lock.md",
        "conditions_frozen_before_run": True,
        "full_protocol_run": full,
        "preservation": "hash-locked v1-v5 modules are imported and used unchanged; every estimate uses decode_wave",
        "master_seed": seed,
        "earlier_master_seeds": {"v2": int(V2_MASTER_SEED), "v3": int(V3_MASTER_SEED), "v4": int(V4_MASTER_SEED),
                                 "v5": int(V5_MASTER_SEED)},
        "randomized_vector_seed": int(VECTOR_SEED),
        "seed_streams": {
            PANEL_INTEGER: "SeedSequence([master_seed, 1, offset_index, vector_index])",
            PANEL_FRACTIONAL: "SeedSequence([master_seed, 2, delta_index, vector_index])",
            PANEL_RANDOM: ("SeedSequence([master_seed, 3, distribution_index, offset_index]); per batch the frame "
                           "vectors are drawn first, then the noise"),
        },
        "trials_per_vector_per_condition": trials,
        "random_frames_per_offset_per_distribution": random_frames,
        "chunk_frames": chunk,
        "vector_length": VECTOR_LENGTH,
        "vector_patterns": {k: v.tolist() for k, v in patterns.items()},
        "samples_per_symbol": SAMPLES_PER_SYMBOL,
        "cycles_per_symbol": CYCLES_PER_SYMBOL,
        "carrier_rotation_degrees_per_sample": 360.0 * CYCLES_PER_SYMBOL / SAMPLES_PER_SYMBOL,
        "amplitude_min": AMPLITUDE_MIN,
        "guard_samples_each_side": GUARD_SAMPLES,
        "noise_std_levels": list(NOISE_STD_LEVELS),
        "tolerance_dimensionless": TOLERANCE,
        "pass_rules": {
            RULE_TOL: "inclusive: all 32 absolute component errors <= 0.25 (the v2-v5 rule)",
            RULE_ROUND: ("integer-alphabet panel only: numpy.round (half-to-even) of every estimate equals the "
                         "transmitted integer"),
        },
        "integer_offsets": list(offsets),
        "fractional_offsets": list(deltas),
        "clean_sweep_offsets": [min(sweep), max(sweep)],
        "search_candidates": list(candidates),
        "random_distributions": list(RANDOM_DISTRIBUTIONS),
        "receiver_buffer_model": ("y[m] = S(m - 64 + Delta) + noise[m] for m = 0..2175, S the continuous p.12 frame "
                                  "signal (zero outside the 32 symbols); positive Delta = receiver late (v5 "
                                  "convention). For integer Delta, y is built from the unchanged encoder and equals "
                                  "the v5 nominal stream; IID Gaussian noise on every buffer sample, guards included"),
        "methods": {
            NOMINAL: ("unchanged decode_wave on windows at 64 + 64k (no correction); identical to the v5 nominal "
                      "decoder"),
            ORACLE: "analysis-only upper bound told the exact integer offset; windows at 64 + 64k - Delta",
            ORACLE_BEST_INT: ("fractional panel: analysis-only reference told Delta; uses whichever of -floor(Delta) "
                              "and -ceil(Delta) has the higher pooled analytic pass probability at SD 0.45 (the best "
                              "a perfect integer-only search can do on average)"),
            MF_SEARCH: ("blind: for each candidate c in -16..16 decode all 32 windows at 64 + 64k + c with decode_wave "
                        "and choose c maximising sum_k estimate_k^2; ties resolve to smallest |c| then smallest c; "
                        "never told the offset; uses the known frame length, symbol length and carrier phase "
                        "reference"),
            ENERGY_SEARCH: ("blind: choose c maximising the received energy in samples [64 + c, 64 + c + 2048) "
                            "(frame-energy detection against noise-only guards); same tie rule; then decode_wave"),
        },
        "timing_acquisition_definition": ("integer panels: chosen correction == -Delta exactly; fractional panel: "
                                          "chosen correction == the best-integer oracle's correction"),
        "paired_comparisons": ("methods share every noisy buffer; discordant counts and exact two-sided McNemar "
                               "(binomial) p-values on pooled frames"),
        "analytic_calibration": ("theory = product over 32 windows of P(|Normal(bias_k, sigma/sqrt(32))| <= 0.25) "
                                 "with bias_k from the clean buffer (unchanged v5 function); pooled = equal-vector "
                                 "mean; random panel theory reported for the oracle only"),
        "why_easy_here": ("noise-only zero guards, known frame and symbol length, integer offsets, constant offset, no "
                          "clock drift, no CFO, carrier locked to the sample clock, IID Gaussian noise"),
        "expected_total_stochastic_input_frames": noisy_frames,
        "module_sha256": module_hashes(),
        **provenance,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "matplotlib": matplotlib.__version__},
        "physical_validity": NO_PHYSICAL_CLAIM,
    }
    result = {
        "benchmark": BENCHMARK_ID, "contract": contract, "results": int_rows + frac_rows + rand_rows,
        "paired_comparisons": int_pairs + frac_pairs + rand_pairs, "clean_sweep": sweep_rows,
        "fractional_analytic_curve": curve, "search_offset_error_histograms": {PANEL_INTEGER: int_hist,
                                                                               PANEL_FRACTIONAL: frac_hist},
    }
    validate_timing_recovery_result(result)
    return result


# ---------------------------------------------------------------------------
# Validator.
# ---------------------------------------------------------------------------
def _close(a: float, b: float, atol: float = 1e-11) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=atol)


def _expected_row_keys(contract: dict[str, Any]) -> set[tuple]:
    patterns = list(contract["vector_patterns"]) + [POOLED_PATTERN]
    keys = set()
    for panel, levels, methods in ((PANEL_INTEGER, contract["integer_offsets"], INTEGER_METHODS),
                                   (PANEL_FRACTIONAL, contract["fractional_offsets"], FRACTIONAL_METHODS)):
        for level in levels:
            for sigma in NOISE_STD_LEVELS:
                for m in methods:
                    for p in patterns:
                        keys.add((panel, FIXED_SET, p, float(level), sigma, m, RULE_TOL))
    for distribution in RANDOM_DISTRIBUTIONS:
        for offset in contract["integer_offsets"]:
            for m in RANDOM_METHODS:
                for r in _rules_for(distribution):
                    keys.add((PANEL_RANDOM, distribution, PER_FRAME_PATTERN, float(offset), NOISE_STD, m, r))
    return keys


def _expected_pair_keys(contract: dict[str, Any]) -> set[tuple]:
    keys = set()
    for offset in contract["integer_offsets"]:
        for a, b in PAIRS_INTEGER:
            keys.add((PANEL_INTEGER, FIXED_SET, float(offset), NOISE_STD, RULE_TOL, a, b))
    for delta in contract["fractional_offsets"]:
        for a, b in PAIRS_FRACTIONAL:
            keys.add((PANEL_FRACTIONAL, FIXED_SET, float(delta), NOISE_STD, RULE_TOL, a, b))
    for distribution in RANDOM_DISTRIBUTIONS:
        for offset in contract["integer_offsets"]:
            for r in _rules_for(distribution):
                for a, b in PAIRS_RANDOM:
                    keys.add((PANEL_RANDOM, distribution, float(offset), NOISE_STD, r, a, b))
    return keys


def _check_contract(contract: dict[str, Any]) -> tuple[int, int]:
    if contract.get("evidence_label") != "SYNTHETIC" or contract.get("physical_validity") != NO_PHYSICAL_CLAIM:
        raise ValueError("v6 results must stay labelled SYNTHETIC with no physical validity claimed")
    if NO_PHYSICAL_CLAIM not in contract.get("classification", ""):
        raise ValueError("classification must state that no physical validity is claimed")
    if contract.get("conditions_frozen_before_run") is not True or "not a formal preregistration" not in contract.get(
            "protocol_status", ""):
        raise ValueError("v6 conditions must be frozen while remaining exploratory")
    seed = contract.get("master_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed in _forbidden_seeds():
        raise ValueError("v6 seed must be an integer distinct from the v2-v5 seeds and the vector seed")
    patterns = contract.get("vector_patterns")
    if not isinstance(patterns, dict) or tuple(patterns) != EXPECTED_PATTERN_NAMES:
        raise ValueError("manifest must retain the six fixed vectors in order")
    expected_patterns = build_input_patterns()
    for name, vector in patterns.items():
        if not np.array_equal(np.asarray(vector, dtype=float), np.asarray(expected_patterns[name], dtype=float)):
            raise ValueError("manifest vectors differ from the v2 fixed panel")
    offsets = _integer_tuple(contract.get("integer_offsets", ()), "integer offsets", require_zero=True,
                             bound=max(SEARCH_CANDIDATES))
    deltas = _fractional_tuple(contract.get("fractional_offsets", ()))
    if tuple(contract.get("search_candidates", ())) != SEARCH_CANDIDATES:
        raise ValueError("search candidates differ from the locked -16..16 grid")
    if contract.get("tolerance_dimensionless") != TOLERANCE or contract.get("vector_length") != VECTOR_LENGTH:
        raise ValueError("tolerance or vector length differs from the locked conditions")
    trials = _positive_integer(contract.get("trials_per_vector_per_condition"), "trials")
    random_frames = _positive_integer(contract.get("random_frames_per_offset_per_distribution"), "random frames")
    expected_noisy = (len(offsets) + len(deltas)) * len(patterns) * trials + len(RANDOM_DISTRIBUTIONS) * len(
        offsets) * random_frames
    if contract.get("expected_total_stochastic_input_frames") != expected_noisy:
        raise ValueError("total stochastic frame count is inconsistent")
    full = (
        trials == TRIALS_PER_VECTOR_PER_CONDITION and random_frames == RANDOM_FRAMES_PER_OFFSET
        and seed == MASTER_SEED and offsets == INTEGER_OFFSETS and deltas == FRACTIONAL_OFFSETS
        and contract.get("chunk_frames") == CHUNK_FRAMES
        and contract.get("clean_sweep_offsets") == [min(CLEAN_SWEEP_OFFSETS), max(CLEAN_SWEEP_OFFSETS)]
    )
    if contract.get("full_protocol_run") is not full:
        raise ValueError("full-protocol flag disagrees with the run conditions")
    hashes = contract.get("module_sha256")
    if not isinstance(hashes, dict) or any(name not in hashes for name in LOCKED_MODULES):
        raise ValueError("manifest must record the hash-locked module digests")
    return trials, random_frames


def _check_row(row: dict[str, Any], key: tuple, trials: int, random_frames: int, n_patterns: int) -> None:
    if row["evidence_label"] != "SYNTHETIC":
        raise ValueError("every v6 row must be labelled SYNTHETIC")
    if key[0] == PANEL_RANDOM:
        frames = random_frames
    else:
        frames = (1 if key[4] == 0.0 else trials) * (n_patterns if key[2] == POOLED_PATTERN else 1)
    n, s = row["frames"], row["frames_passing"]
    if n != frames or isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= n:
        raise ValueError("frame or pass count disagrees with the manifest")
    if not _close(row["frame_pass_share"], s / n) or not _close(row["failure_share"], (n - s) / n):
        raise ValueError("pass or failure share disagrees with the pass count")
    if not _close(row["offset_percent_of_symbol"], 100.0 * key[3] / SAMPLES_PER_SYMBOL):
        raise ValueError("offset normalisation is inconsistent")
    stochastic = key[4] > 0.0
    if stochastic:
        lo, hi = wilson_interval(s, n)
        if (row["wilson95_lower"] is None or not _close(row["wilson95_lower"], lo)
                or not _close(row["wilson95_upper"], hi) or not _close(row["failure_wilson95_lower"], 1 - hi)
                or not _close(row["failure_wilson95_upper"], 1 - lo) or "Wilson" not in row["uncertainty_kind"]):
            raise ValueError("stochastic row has an incorrect or missing Wilson interval")
    elif (row["wilson95_lower"] is not None or row["wilson95_upper"] is not None
          or "deterministic" not in row["uncertainty_kind"]):
        raise ValueError("deterministic row must not report a sampling interval")
    acq = row["timing_acquired_frames"]
    if key[5] in SEARCH_METHODS:
        if isinstance(acq, bool) or not isinstance(acq, int) or not 0 <= acq <= n or not _close(
                row["timing_acquisition_share"], acq / n):
            raise ValueError("search row has an invalid timing-acquisition count")
        if stochastic:
            alo, ahi = wilson_interval(acq, n)
            if not _close(row["acquisition_wilson95_lower"], alo) or not _close(row["acquisition_wilson95_upper"], ahi):
                raise ValueError("acquisition Wilson interval is inconsistent")
    elif acq is not None:
        raise ValueError("non-search methods must not report timing acquisition")
    theory = row["theory_frame_pass_probability"]
    if theory is not None and not 0.0 <= float(theory) <= 1.0:
        raise ValueError("theoretical probability outside [0, 1]")
    if key[5] == ORACLE and stochastic and key[6] == RULE_TOL and not _close(theory, 0.9477992034686994, atol=1e-9):
        raise ValueError("oracle theory must equal the closed-form 94.78%")


def validate_timing_recovery_result(result: dict[str, Any]) -> None:
    """Check manifest, row grid, counts, intervals, pooling and paired-comparison consistency."""
    if not isinstance(result, dict) or result.get("benchmark") != BENCHMARK_ID:
        raise ValueError("unexpected v6 benchmark identifier")
    contract, rows = result.get("contract"), result.get("results")
    pairs, sweep = result.get("paired_comparisons"), result.get("clean_sweep")
    if not isinstance(contract, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("result must contain a contract and non-empty results")
    if not isinstance(pairs, list) or not isinstance(sweep, list) or not sweep:
        raise ValueError("result must contain paired comparisons and a clean sweep")
    trials, random_frames = _check_contract(contract)
    patterns = contract["vector_patterns"]

    expected = _expected_row_keys(contract)
    row_map: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or any(f not in row for f in ROW_FIELDS):
            raise ValueError("a v6 row is missing required fields")
        key = (row["panel"], row["vector_set"], row["vector_pattern"], float(row["offset_samples"]),
               float(row["noise_std"]), row["method"], row["pass_rule"])
        if key in row_map:
            raise ValueError("duplicate v6 result row")
        if key not in expected:
            raise ValueError("result row is outside the declared v6 grid")
        _check_row(row, key, trials, random_frames, len(patterns))
        row_map[key] = row
    if set(row_map) != expected:
        raise ValueError("v6 rows do not cover exactly the frozen conditions")

    for (panel, vset, pattern, offset, sigma, method, rule), pooled in row_map.items():
        if pattern != POOLED_PATTERN:
            continue
        members = [row_map[(panel, vset, name, offset, sigma, method, rule)] for name in patterns]
        if pooled["frames_passing"] != sum(r["frames_passing"] for r in members):
            raise ValueError("pooled pass count disagrees with vector rows")
        if pooled["timing_acquired_frames"] is not None and pooled["timing_acquired_frames"] != sum(
                r["timing_acquired_frames"] for r in members):
            raise ValueError("pooled acquisition count disagrees with vector rows")
        if not _close(pooled["empirical_mae"], sum(r["empirical_mae"] for r in members) / len(members)):
            raise ValueError("pooled MAE disagrees with vector rows")
        if pooled["theory_frame_pass_probability"] is not None and not _close(
                pooled["theory_frame_pass_probability"],
                sum(r["theory_frame_pass_probability"] for r in members) / len(members)):
            raise ValueError("pooled theory is not the equal-vector mean")

    expected_pairs = _expected_pair_keys(contract)
    seen_pairs = set()
    for pair in pairs:
        if not isinstance(pair, dict) or any(f not in pair for f in PAIR_FIELDS):
            raise ValueError("a paired comparison is missing fields")
        pkey = (pair["panel"], pair["vector_set"], float(pair["offset_samples"]), float(pair["noise_std"]),
                pair["pass_rule"], pair["method_a"], pair["method_b"])
        if pkey not in expected_pairs or pkey in seen_pairs:
            raise ValueError("paired comparison is duplicated or outside the declared grid")
        seen_pairs.add(pkey)
        pattern = PER_FRAME_PATTERN if pair["panel"] == PANEL_RANDOM else POOLED_PATTERN
        ka = (pkey[0], pkey[1], pattern, pkey[2], pkey[3], pkey[5], pkey[4])
        kb = (pkey[0], pkey[1], pattern, pkey[2], pkey[3], pkey[6], pkey[4])
        n = pair["frames"]
        if (n != row_map[ka]["frames"] or pair["a_passing"] != row_map[ka]["frames_passing"]
                or pair["b_passing"] != row_map[kb]["frames_passing"]):
            raise ValueError("paired comparison disagrees with the method rows")
        b, c = pair["a_pass_b_fail"], pair["a_fail_b_pass"]
        if pair["both_pass"] + pair["both_fail"] + b + c != n or b - c != pair["a_passing"] - pair["b_passing"]:
            raise ValueError("discordant counts are inconsistent with the pass counts")
        if pair["both_pass"] + b != pair["a_passing"]:
            raise ValueError("concordant counts are inconsistent with the pass counts")
        if not _close(pair["mcnemar_exact_two_sided_p"], mcnemar_exact_p(b, c)):
            raise ValueError("McNemar p-value is inconsistent with the discordant counts")
    if seen_pairs != expected_pairs:
        raise ValueError("paired comparisons do not cover the declared grid")

    for row in sweep:
        if not isinstance(row, dict) or any(f not in row for f in SWEEP_FIELDS) or row["evidence_label"] != "SYNTHETIC":
            raise ValueError("clean sweep row is malformed")
        if row["frames"] != len(patterns) or not 0 <= row["frames_passing"] <= row["frames"]:
            raise ValueError("clean sweep counts are inconsistent")


# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pooled_rows(result: dict[str, Any], panel: str, method: str, *, vector_set: str = FIXED_SET,
                rule: str = RULE_TOL, sigma: float = NOISE_STD) -> list[dict[str, Any]]:
    """Pooled (or per-frame random) rows for one panel/method, sorted by offset."""
    pattern = PER_FRAME_PATTERN if panel == PANEL_RANDOM else POOLED_PATTERN
    rows = [r for r in result["results"] if r["panel"] == panel and r["vector_set"] == vector_set
            and r["vector_pattern"] == pattern and r["method"] == method and r["pass_rule"] == rule
            and r["noise_std"] == sigma]
    return sorted(rows, key=lambda r: r["offset_samples"])


def _failure_series(ax: Any, rows: list[dict[str, Any]], color: str, marker: str, label: str, *,
                    linestyle: str = "-", label_dx: float = 0.3, label_dy: float = 1.0) -> None:
    x = np.asarray([r["offset_samples"] for r in rows], dtype=float)
    fail = np.asarray([r["failure_share"] for r in rows], dtype=float)
    lo = np.asarray([r["failure_wilson95_lower"] for r in rows], dtype=float)
    hi = np.asarray([r["failure_wilson95_upper"] for r in rows], dtype=float)
    zero = fail == 0.0
    shown = np.where(zero, hi, fail)  # zero observed failures: plot the 95% Wilson upper bound (open triangle)
    ax.plot(x, shown, linestyle=linestyle, color=color, lw=1.3, alpha=0.85)
    if np.any(~zero):
        err = np.vstack((fail - lo, hi - fail))[:, ~zero]
        ax.errorbar(x[~zero], fail[~zero], yerr=err, fmt=marker, color=color, ms=5, capsize=2.5, lw=1.0)
    if np.any(zero):
        ax.plot(x[zero], shown[zero], marker="v", linestyle="none", mfc="white", color=color, ms=6)
    ax.annotate(label, xy=(x[-1], shown[-1]), xytext=(x[-1] + label_dx, shown[-1] * label_dy), color=color,
                fontsize=7.5, va="center", ha="left", annotation_clip=False)


def write_timing_recovery_outputs(result: dict[str, Any], output_dir: Path = ARTIFACTS) -> dict[str, Path]:
    """Write the v6 JSON, three CSVs and the summary PNG."""
    validate_timing_recovery_result(result)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out / f"{ARTIFACT_STEM}.json",
        "csv": out / f"{ARTIFACT_STEM}.csv",
        "paired_csv": out / f"{ARTIFACT_STEM}-paired.csv",
        "sweep_csv": out / f"{ARTIFACT_STEM}-clean-sweep.csv",
        "png": out / f"{ARTIFACT_STEM}.png",
    }
    paths["json"].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(paths["csv"], ROW_FIELDS, result["results"])
    _write_csv(paths["paired_csv"], PAIR_FIELDS, result["paired_comparisons"])
    _write_csv(paths["sweep_csv"], SWEEP_FIELDS, result["clean_sweep"])
    plot_timing_recovery(result, paths["png"])
    return paths


def plot_timing_recovery(result: dict[str, Any], path: Path) -> None:
    colors = {NOMINAL: "#277da1", ORACLE: "#f3722c", ORACLE_BEST_INT: "#f3722c", MF_SEARCH: "#2a9d8f",
              ENERGY_SEARCH: "#9b5de5"}
    short = {NOMINAL: "nominal (no correction)", ORACLE: "oracle (told Δ)", MF_SEARCH: "blind MF-energy search",
             ENERGY_SEARCH: "blind frame-energy search", ORACLE_BEST_INT: "oracle best integer"}
    markers = {NOMINAL: "o", ORACLE: "s", ORACLE_BEST_INT: "s", MF_SEARCH: "D", ENERGY_SEARCH: "^"}
    analytic_fail = 1.0 - theoretical_frame_pass_probability(np.zeros(VECTOR_LENGTH), NOISE_STD)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    fig.subplots_adjust(left=0.065, right=0.86, top=0.91, bottom=0.08, wspace=0.45, hspace=0.32)

    ax = axes[0, 0]
    label_shift = {NOMINAL: 1.0, ORACLE: 1.45, MF_SEARCH: 0.72, ENERGY_SEARCH: 1.0}
    for m in INTEGER_METHODS:
        _failure_series(ax, pooled_rows(result, PANEL_INTEGER, m), colors[m], markers[m], short[m],
                        label_dy=label_shift[m])
    ax.axhline(analytic_fail, color="0.45", ls=":", lw=1.0)
    ax.text(2.3, analytic_fail * 0.55, f"analytic oracle failure {100 * analytic_fail:.2f}%", fontsize=7,
            color="0.35")
    ax.set(yscale="log", ylim=(1e-3, 1.8), xlabel="integer timing offset Δ (samples; + = late)",
           ylabel="frame failure share (log scale; 95% Wilson)",
           title="(a) integer offsets, six fixed vectors, σ = 0.45", xticks=INTEGER_OFFSETS)
    ax.grid(alpha=0.22, which="both")

    ax = axes[0, 1]
    curve = result["fractional_analytic_curve"]
    cx = [c["offset_samples"] for c in curve]
    ax.plot(cx, [max(1.0 - c["nominal_theory"], 1e-6) for c in curve], color=colors[NOMINAL], lw=0.9, ls="--",
            alpha=0.75)
    ax.plot(cx, [max(1.0 - c["best_integer_correction_theory"], 1e-6) for c in curve],
            color=colors[ORACLE_BEST_INT], lw=0.9, ls="--", alpha=0.75)
    label_shift = {NOMINAL: 1.0, ORACLE_BEST_INT: 1.45, MF_SEARCH: 0.72}
    for m in FRACTIONAL_METHODS:
        _failure_series(ax, pooled_rows(result, PANEL_FRACTIONAL, m), colors[m], markers[m], short[m],
                        linestyle="none", label_dx=0.25, label_dy=label_shift[m])
    ax.set(yscale="log", ylim=(1e-3, 1.8), xlabel="sub-sample timing offset Δ (samples; waveform sampled at n + Δ)",
           ylabel="frame failure share (log scale; 95% Wilson)",
           title="(b) fractional offsets; dashed = analytic (nominal, best integer)", xticks=(0, 1, 2, 3, 4, 8))
    ax.grid(alpha=0.22, which="both")

    ax = axes[1, 0]
    label_offset = {NOMINAL: 0.14, ENERGY_SEARCH: 0.0, MF_SEARCH: -0.14, ORACLE: 0.0}
    for m in (NOMINAL, ENERGY_SEARCH, MF_SEARCH, ORACLE):
        srows = sorted((r for r in result["clean_sweep"] if r["method"] == m), key=lambda r: r["offset_samples"])
        x = [r["offset_samples"] for r in srows]
        y = [r["empirical_mae"] for r in srows]
        ax.plot(x, y, marker=markers[m], ms=3, lw=1.1, color=colors[m])
        ax.annotate(short[m], xy=(x[-1], y[-1]), xytext=(x[-1] + 1.5, y[-1] + label_offset[m]),
                    color=colors[m], fontsize=7.5, va="center", annotation_clip=False)
    for edge in (min(SEARCH_CANDIDATES), max(SEARCH_CANDIDATES)):
        ax.axvline(edge, color="0.6", ls=":", lw=0.9)
    ax.set(xlabel="integer timing offset Δ (samples)", ylabel="clean component MAE (dimensionless)",
           title="(c) clean frames, Δ = −32..32 (dotted: search range ±16)", xticks=range(-32, 33, 8))
    ax.grid(alpha=0.22)

    ax = axes[1, 1]
    datasets = (
        (RANDOM_DISTRIBUTIONS[0], RULE_TOL, "#577590", "U[−2,2], |err| ≤ 0.25", 1.4),
        (RANDOM_DISTRIBUTIONS[1], RULE_TOL, "#6a994e", "integers, |err| ≤ 0.25", 0.7),
        (RANDOM_DISTRIBUTIONS[1], RULE_ROUND, "#d62828", "integers, round rule", 1.0),
    )
    for dist, rule, color, name, dy in datasets:
        _failure_series(ax, pooled_rows(result, PANEL_RANDOM, MF_SEARCH, vector_set=dist, rule=rule), color, "D",
                        f"search: {name}", label_dy=dy)
        nominal_rows = pooled_rows(result, PANEL_RANDOM, NOMINAL, vector_set=dist, rule=rule)
        shown = [r["failure_share"] if r["failure_share"] > 0 else r["failure_wilson95_upper"] for r in nominal_rows]
        ax.plot([r["offset_samples"] for r in nominal_rows], shown, color=color, ls=":", marker="o", mfc="white",
                ms=3.5, lw=1.0, alpha=0.85)
    ax.text(0, 1.3, "dotted, open circles = nominal decoder (zero failures at Wilson upper bound)", ha="center",
            fontsize=7, color="0.35")
    ax.set(yscale="log", ylim=(1e-4, 2.2), xlabel="integer timing offset Δ (samples)",
           ylabel="frame failure share (log scale; 95% Wilson)",
           title="(d) randomised per-frame vectors, σ = 0.45", xticks=INTEGER_OFFSETS)
    ax.grid(alpha=0.22, which="both")

    fig.suptitle(f"v6 blind timing recovery — SYNTHETIC (seed {result['contract']['master_seed']}); "
                 f"{NO_PHYSICAL_CLAIM}", fontsize=12)
    fig.text(0.065, 0.02, "Open triangles: zero observed failures, plotted at the 95% Wilson upper bound. "
             "Easy case by construction: noise-only guards, known frame length, constant offset, no drift or CFO.",
             fontsize=7.5, color="0.35")
    fig.savefig(path, dpi=150, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def main() -> int:
    result = run_timing_recovery_benchmark()
    paths = write_timing_recovery_outputs(result)
    c = result["contract"]
    print(f"v6 timing recovery: seed={c['master_seed']}; {c['expected_total_stochastic_input_frames']:,} noisy "
          f"frames; full_protocol_run={c['full_protocol_run']}; git={c['git_commit']}")
    for panel, methods in ((PANEL_INTEGER, INTEGER_METHODS), (PANEL_FRACTIONAL, FRACTIONAL_METHODS)):
        for m in methods:
            for r in pooled_rows(result, panel, m):
                acq = "" if r["timing_acquired_frames"] is None else f"  acquired={r['timing_acquired_frames']}"
                th = r["theory_frame_pass_probability"]
                theory = "" if th is None else f"  theory={th:.4f}"
                print(f"  {panel:18s} Δ={r['offset_samples']:+6.2f} {m:36s} pass={r['frames_passing']:>5}/"
                      f"{r['frames']}{acq}{theory}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
