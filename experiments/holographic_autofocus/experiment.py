"""Classical SYNTHETIC holographic autofocus; AGPL-3.0-only, see ../../LICENSE."""

import argparse
import csv
import json
import platform
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PEAK_LEVELS = (200, 2000)
READ_NOISE_ELECTRONS = 3.0
TRUE_DISTANCE_M = -0.004
TOLERANCE_MM = 0.25
COARSE_DISTANCES_M = np.linspace(-0.008, 0.0, 65)
REFINED_SAMPLES = 21
RESULT_NAMES = ("trials.csv", "config.json", "summary.json")
TRIAL_COLUMNS = (
    "peak_expected_electrons", "trial", "estimated_distance_mm", "error_mm", "at_scan_edge",
)
LIMITATIONS = [
    "One fixed synthetic two-Gaussian scene, not measured membrane data.",
    "Periodic FFT propagation without padding permits wraparound.",
    "Finite sampling, evanescent-component removal and chosen filter bandwidth.",
    "An in-Nyquist mask does not guarantee complete spectral separation or eliminate leakage.",
    "Hypothetical camera parameters, not calibrated measurements.",
    "No clipping, saturation, quantization or fixed-pattern noise.",
    "The intensity-gradient focus metric is not universal across object classes.",
    "Local refinement follows only one winning coarse peak, not all competing peaks.",
    "Numerical search spacing is not physical depth accuracy.",
    "Empirical percentiles describe trial spread, not a confidence interval for true distance.",
    "40/40 successes do not prove future 100% reliability.",
    "Propagation round trips do not validate the entire reconstruction pipeline.",
    "No physical, quantum, display, calibrated membrane or Digital to Wave preprint validation.",
]


@dataclass(frozen=True)
class Model:
    n: int
    pixel_pitch_m: float
    wavelength_m: float
    recording_distance_m: float
    carrier_bins_x: int
    carrier_bins_y: int
    reference_amplitude: float
    outer_radius_bins: float
    inner_radius_bins: float
    object_field: np.ndarray
    reference: np.ndarray
    propagating: np.ndarray
    kz: np.ndarray
    mask: np.ndarray


def make_model():
    """Construct the fixed float64/complex128 object and Fourier grids."""
    n = 96
    pixel_pitch_m, wavelength_m = 6.5e-6, 532e-9
    y, x = np.indices((n, n), dtype=np.float64)
    X, Y = x - (n - 1) / 2, y - (n - 1) / 2
    object_field = (
        np.exp(-((X + 12) ** 2 + (Y + 8) ** 2) / 32)
        + 0.7 * np.exp(-((X - 11) ** 2 + (Y - 10) ** 2) / 50)
    ).astype(np.complex128)
    reference = np.exp(2j * np.pi * 24 * X / n)
    f = np.fft.fftfreq(n, d=pixel_pitch_m)
    FX, FY = np.meshgrid(f, f)
    q = 1 / wavelength_m**2 - FX**2 - FY**2
    propagating = q >= 0
    kz = 2 * np.pi * np.sqrt(np.maximum(q, 0))
    bins = np.fft.fftshift(np.fft.fftfreq(n)) * n
    BX, BY = np.meshgrid(bins, bins)
    r = np.hypot(BX + 24, BY)
    outer, inner = 9.0, 0.75 * 9.0
    mask = np.zeros((n, n), dtype=np.float64)
    mask[r <= inner] = 1.0
    transition = (r > inner) & (r < outer)
    mask[transition] = 0.5 * (1 + np.cos(np.pi * (r[transition] - inner) / (outer - inner)))
    return Model(
        n, pixel_pitch_m, wavelength_m, 0.004, 24, 0, 1.0, outer, inner,
        object_field, reference, propagating, kz, mask,
    )


def _array(value, *, shape=None, real=False):
    array = np.asarray(value)
    if real and np.iscomplexobj(array):
        raise ValueError("Intensity must be real.")
    array = np.asarray(array, dtype=np.float64 if real else np.complex128)
    if array.ndim != 2 or (shape is not None and array.shape != shape):
        raise ValueError(f"Expected a two-dimensional array with shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Array must contain only finite values.")
    return array


def transfer_function(model, distance_m):
    """exp(+i z kz) for exp(-i omega t); remove evanescent components."""
    if not np.isscalar(distance_m) or not np.isreal(distance_m) or not np.isfinite(distance_m):
        raise ValueError("Distance must be a finite real scalar.")
    transfer = np.zeros((model.n, model.n), dtype=np.complex128)
    transfer[model.propagating] = np.exp(1j * distance_m * model.kz[model.propagating])
    return transfer


def propagate(field, distance_m, model):
    """Periodic, unpadded angular-spectrum propagation at unchanged pixel pitch."""
    field = _array(field, shape=(model.n, model.n))
    transfer = transfer_function(model, distance_m)
    if distance_m == 0:
        return field.copy()
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def clean_hologram(model):
    """Unnormalized off-axis interference intensity, reference amplitude one."""
    sensor = propagate(model.object_field, model.recording_distance_m, model)
    return np.abs(sensor + model.reference) ** 2


def recover_field(hologram, model):
    """Select S conjugate(R) at (-24, 0), then remove its carrier with R."""
    hologram = _array(hologram, shape=(model.n, model.n), real=True)
    spectrum = np.fft.fftshift(np.fft.fft2(hologram))
    selected = np.fft.ifft2(np.fft.ifftshift(spectrum * model.mask))
    return selected * model.reference


def focus_score(field):
    """Normalized intensity-gradient energy; larger is sharper for this scene."""
    field = _array(field)
    if min(field.shape) < 2:
        raise ValueError("Focus scoring needs at least two samples along each axis.")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        intensity = np.abs(field) ** 2
        gy, gx = np.gradient(intensity)
        denominator = np.mean(intensity**2)
        numerator = np.mean(gx**2 + gy**2)
        if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
            raise ValueError("Focus-score denominator must be positive and all terms finite.")
        score = numerator / denominator
    if not np.isfinite(score):
        raise ValueError("Focus score must be finite.")
    return float(score)


def estimate_focus(hologram, model):
    """Maximize one coarse peak and refine it, without using ground truth."""
    spectrum = np.fft.fft2(recover_field(hologram, model))

    def scan(distances):
        scores = [
            focus_score(np.fft.ifft2(spectrum * transfer_function(model, distance)))
            for distance in distances
        ]
        winner = int(np.argmax(scores))
        return winner, scores[winner]

    winner, score = scan(COARSE_DISTANCES_M)
    if winner in (0, len(COARSE_DISTANCES_M) - 1):
        return float(COARSE_DISTANCES_M[winner]), score, True
    refined = np.linspace(
        COARSE_DISTANCES_M[winner - 1], COARSE_DISTANCES_M[winner + 1], REFINED_SAMPLES,
    )
    winner, score = scan(refined)
    return float(refined[winner]), score, winner in (0, len(refined) - 1)


def add_camera_noise(hologram, rng, peak_electrons, read_noise_electrons=3.0):
    """Use the clean frame's fixed gain; draw Poisson before Gaussian noise."""
    hologram = _array(hologram, real=True)
    for value, positive in ((peak_electrons, True), (read_noise_electrons, False)):
        if (
            not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value)
            or value < 0 or (positive and value == 0)
        ):
            raise ValueError("Peak electrons must be positive and read noise nonnegative, both finite.")
    if np.any(hologram < 0) or hologram.size == 0 or np.max(hologram) <= 0:
        raise ValueError("Expected intensity must be nonnegative with a positive maximum.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = peak_electrons / np.max(hologram)
        expected_counts = gain * hologram
    if gain <= 0 or not np.isfinite(gain) or not np.all(np.isfinite(expected_counts)):
        raise ValueError("Gain and expected counts must be finite, with positive gain.")
    measured_counts = rng.poisson(expected_counts).astype(np.float64)
    measured_counts += rng.normal(0, read_noise_electrons, hologram.shape)
    noisy_hologram = measured_counts / gain
    if not np.all(np.isfinite(noisy_hologram)):
        raise ValueError("Noisy hologram must remain finite.")
    return noisy_hologram


def _configuration(model, trials, seed, clean):
    return {
        "scope": "SYNTHETIC classical scalar holographic autofocus, independent of encoding benchmarks",
        "seed": seed,
        "rng": "numpy.random.default_rng; one stream; 200 then 2000; Poisson then Gaussian",
        "trials_per_level": trials,
        "total_trials": trials * len(PEAK_LEVELS),
        "tolerance_mm": TOLERANCE_MM,
        "grid": {"n": model.n, "shape": [model.n, model.n], "real_dtype": "float64", "complex_dtype": "complex128"},
        "object": {
            "coordinates": "X=x-(n-1)/2; Y=y-(n-1)/2, in pixels",
            "amplitude": "exp(-((X+12)^2+(Y+8)^2)/32)+0.7*exp(-((X-11)^2+(Y-10)^2)/50)",
            "gaussians": [
                {"center_xy_pixels": [-12, -8], "amplitude": 1.0, "exponent_denominator_pixels2": 32},
                {"center_xy_pixels": [11, 10], "amplitude": 0.7, "exponent_denominator_pixels2": 50},
            ],
            "initial_phase_rad": 0.0,
        },
        "wavelength_m": model.wavelength_m,
        "pixel_pitch_m": model.pixel_pitch_m,
        "recording_distance_m": model.recording_distance_m,
        "known_reconstruction_distance_m": TRUE_DISTANCE_M,
        "reference": {
            "carrier_bins_xy": [model.carrier_bins_x, model.carrier_bins_y],
            "carrier_cycles_per_pixel_xy": [0.25, 0.0], "amplitude": model.reference_amplitude,
            "formula": "exp(2j*pi*24*X/n)",
        },
        "filter": {
            "sideband_center_bins_xy": [-24, 0], "outer_radius_bins": model.outer_radius_bins,
            "inner_radius_bins": model.inner_radius_bins, "transition": "raised cosine",
            "carrier_removal": "multiply selected field by R",
        },
        "noise": {
            "peak_expected_electrons": list(PEAK_LEVELS), "read_noise_electrons": READ_NOISE_ELECTRONS,
            "gain": "peak_expected_electrons/max(clean_hologram), fixed within each level",
            "gain_electrons_per_intensity": {str(level): float(level / np.max(clean)) for level in PEAK_LEVELS},
            "negative_offset_corrected_samples_allowed": True,
            "per_trial_normalization": False, "camera_parameters": "hypothetical, uncalibrated",
        },
        "focus": {
            "metric": "mean(gx^2+gy^2)/mean(I^2); I=abs(field)^2; gy,gx=numpy.gradient(I)",
            "objective": "maximize", "coarse_range_m": [-0.008, 0.0], "coarse_samples": 65,
            "coarse_step_m": 0.000125, "refined_samples": REFINED_SAMPLES, "refined_step_m": 0.0000125,
            "refinement": "21 points between immediate neighbors of one coarse winner",
            "endpoint_handling": "coarse endpoint: return unrefined; refined endpoint: flag; retain all trials",
            "tie_break": "first maximum in ascending-distance order",
            "ground_truth_used_for_selection": False,
        },
        "propagation": {
            "time_dependence": "exp(-i*omega*t)", "transfer": "exp(+i*distance_m*kz)",
            "kz": "2*pi*sqrt(1/wavelength_m^2-FX^2-FY^2)",
            "positive_distance": "forward", "negative_distance": "back-propagation",
            "boundaries": "periodic FFT, no padding", "pixel_pitch": "unchanged",
            "evanescent_components": "removed (transfer=0)", "zero_distance": "copy input",
        },
        "statistics": {"standard_deviation_ddof": 1, "percentiles": [2.5, 97.5], "percentile_method": "linear"},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "limitations": LIMITATIONS,
    }


def run_experiment(trials_per_level=40, seed=42):
    """Return config, summary and all trial rows; no files are written here."""
    if isinstance(trials_per_level, bool) or not isinstance(trials_per_level, (int, np.integer)):
        raise ValueError("Trials per level must be an integer >= 2.")
    if trials_per_level < 2:
        raise ValueError("Trials per level must be >= 2.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    trials_per_level, seed = int(trials_per_level), int(seed)
    model = make_model()
    clean = clean_hologram(model)
    distance, score, edge = estimate_focus(clean, model)
    summary = {
        "noise_free": {"estimated_distance_mm": distance * 1000, "score": score, "at_scan_edge": edge},
        "levels": [],
    }
    rng = np.random.default_rng(seed)
    rows = []
    for level in PEAK_LEVELS:
        level_rows = []
        for trial in range(1, trials_per_level + 1):
            noisy = add_camera_noise(clean, rng, level, READ_NOISE_ELECTRONS)
            distance, _, edge = estimate_focus(noisy, model)
            row = {
                "peak_expected_electrons": level, "trial": trial,
                "estimated_distance_mm": distance * 1000,
                "error_mm": distance * 1000 - TRUE_DISTANCE_M * 1000, "at_scan_edge": edge,
            }
            rows.append(row)
            level_rows.append(row)
        estimates = np.array([row["estimated_distance_mm"] for row in level_rows], dtype=np.float64)
        errors = np.array([row["error_mm"] for row in level_rows], dtype=np.float64)
        summary["levels"].append({
            "peak_expected_electrons": level, "trial_count": trials_per_level,
            "mean_estimated_distance_mm": float(np.mean(estimates)),
            "sample_standard_deviation_mm": float(np.std(estimates, ddof=1)),
            "bias_mm": float(np.mean(errors)), "rmse_mm": float(np.sqrt(np.mean(errors**2))),
            "within_tolerance_fraction": float(np.mean(np.abs(errors) <= TOLERANCE_MM)),
            "search_edge_fraction": float(np.mean([row["at_scan_edge"] for row in level_rows])),
            "estimated_distance_percentile_2_5_mm": float(np.percentile(estimates, 2.5, method="linear")),
            "estimated_distance_percentile_97_5_mm": float(np.percentile(estimates, 97.5, method="linear")),
        })
    return {"config": _configuration(model, trials_per_level, seed, clean), "summary": summary, "trials": rows}


def check_output_directory(output):
    """Refuse existing intended files (including dangling symlinks) before a run."""
    output = Path(output)
    for name in RESULT_NAMES:
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a fresh output directory with --output.")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"{output} is not a directory; choose a fresh output directory.")
    return output


def write_results(result, output):
    output = check_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        files = {
            name: stack.enter_context((output / name).open("x", encoding="utf-8", newline=""))
            for name in RESULT_NAMES
        }
        writer = csv.DictWriter(files["trials.csv"], fieldnames=TRIAL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["trials"])
        for name, key in (("config.json", "config"), ("summary.json", "summary")):
            json.dump(result[key], files[name], indent=2, allow_nan=False)
            files[name].write("\n")
    for name in RESULT_NAMES:
        if not (output / name).is_file() or (output / name).stat().st_size == 0:
            raise OSError(f"Missing or empty result file: {output / name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40, help="Trials per level, minimum 2 (default: 40).")
    parser.add_argument("--seed", type=int, default=42, help="Nonnegative RNG seed (default: 42).")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    try:
        output = check_output_directory(args.output)
        result = run_experiment(args.trials, args.seed)
        write_results(result, output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
