"""Synthetic, dimensionless Digital-to-Wave-inspired software testbench.

The encoder follows the signed-amplitude sine expression printed in
DigitaltoWave-1.pdf, p. 12. This module does not model LGD, spin waves,
phonons, GHz hardware, or any biological signal or mechanism. Sampling,
noise, vector framing, and matched-filter decoding are testbench choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WaveConfig:
    """Dimensionless discrete-time carrier setup for one synthetic symbol."""

    samples_per_symbol: int = 64
    cycles_per_symbol: int = 4
    amplitude_min: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.samples_per_symbol, bool) or not isinstance(self.samples_per_symbol, int):
            raise ValueError("samples_per_symbol must be an integer")
        if isinstance(self.cycles_per_symbol, bool) or not isinstance(self.cycles_per_symbol, int):
            raise ValueError("cycles_per_symbol must be an integer")
        if self.samples_per_symbol < 4:
            raise ValueError("samples_per_symbol must be at least 4")
        if self.cycles_per_symbol < 1 or 2 * self.cycles_per_symbol >= self.samples_per_symbol:
            raise ValueError("cycles_per_symbol must be positive and below Nyquist")
        if not math.isfinite(self.amplitude_min) or self.amplitude_min <= 0:
            raise ValueError("amplitude_min must be finite and positive")


def _finite_vector(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional finite numeric vector") from exc
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty one-dimensional finite numeric vector")
    return array


def encode_wave(values: Any, config: WaveConfig = WaveConfig()) -> np.ndarray:
    """Encode a real vector into one synthetic carrier segment per value.

    Contract: inputs are finite real scalars in [-2, 2]; each has dimensionless
    units. The magnitude scales linearly by ``amplitude_min``. Positive and
    negative values use the paper's stated phases (pi/4 and 5*pi/4); zero is
    represented by the zero-amplitude baseline. Returns a dimensionless array
    with shape (number_of_values, samples_per_symbol).
    """
    vector = _finite_vector(values, "values")
    if np.any(np.abs(vector) > 2.0):
        raise ValueError("each input value must lie in the testbench range [-2, 2]")

    sample_index = np.arange(config.samples_per_symbol, dtype=float)
    omega = 2.0 * np.pi * config.cycles_per_symbol / config.samples_per_symbol
    carrier_phase = omega * sample_index
    output = np.zeros((vector.size, config.samples_per_symbol), dtype=float)
    for row, value in enumerate(vector):
        if value > 0.0:
            phi = np.pi / 4.0
        elif value < 0.0:
            phi = 5.0 * np.pi / 4.0
        else:
            phi = 0.0  # phase is immaterial at zero amplitude
        output[row] = abs(value) * config.amplitude_min * np.sin(carrier_phase + phi)
    return output


def add_gaussian_noise(signal: Any, noise_std: float, seed: int) -> np.ndarray:
    """Add seeded, dimensionless IID Gaussian noise to a finite signal array."""
    array = np.asarray(signal, dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("signal must be a non-empty, finite 2D array")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    return array + rng.normal(0.0, noise_std, size=array.shape)


def decode_wave(received: Any, config: WaveConfig = WaveConfig()) -> np.ndarray:
    """Recover one real scalar per segment with a synthetic matched filter.

    The least-squares projection is an added software receiver design, not a
    demodulator specified by either supplied PDF. The local carrier reference
    is sin(omega*n + pi/4), so a 5*pi/4 paper-encoded negative value projects
    to a negative scalar. No quantization is applied.
    """
    array = np.asarray(received, dtype=float)
    expected_samples = config.samples_per_symbol
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != expected_samples:
        raise ValueError(
            f"received must have shape (n_values, {expected_samples}) with n_values > 0"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("received must contain only finite values")

    sample_index = np.arange(expected_samples, dtype=float)
    omega = 2.0 * np.pi * config.cycles_per_symbol / expected_samples
    reference = np.sin(omega * sample_index + np.pi / 4.0)
    denominator = config.amplitude_min * float(reference @ reference)
    return (array @ reference) / denominator


def reconstruction_metrics(reference: Any, candidate: Any, tolerance: float) -> dict[str, float | int | bool]:
    """Compute generic vector reconstruction errors and an inclusive threshold."""
    expected = _finite_vector(reference, "reference")
    actual = _finite_vector(candidate, "candidate")
    if expected.shape != actual.shape:
        raise ValueError("reference and candidate must have the same vector length")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    residual = actual - expected
    absolute = np.abs(residual)
    return {
        "max_abs_error": float(np.max(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "components_over_tolerance": int(np.count_nonzero(absolute > tolerance)),
        "tolerance": float(tolerance),
        "pass": bool(np.max(absolute) <= tolerance),
    }


def oes32_max_abs_residual(reference: Any, candidate: Any) -> float:
    """Return max(|reference-candidate|) for exactly two 32-value vectors.

    This is only the separately documented OES32 residual comparison rule;
    it is not the OES-Resilience detector and it is not used by the codec.
    """
    expected = _finite_vector(reference, "reference")
    actual = _finite_vector(candidate, "candidate")
    if expected.size != 32 or actual.size != 32:
        raise ValueError("the OES32 residual rule requires two vectors of exactly 32 values")
    return float(np.max(np.abs(expected - actual)))


def oes32_passes_max_abs_rule(reference: Any, candidate: Any, tolerance: float) -> bool:
    """Apply the OES32 max-absolute residual rule: pass iff residual <= tolerance."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    return oes32_max_abs_residual(reference, candidate) <= tolerance
