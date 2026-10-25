# Synthetic holographic-autofocus Monte Carlo experiment

**SYNTHETIC — classical, fixed-scene numerical optics only.**
This standalone experiment tests scalar angular-spectrum propagation, off-axis
interference, Poisson photoelectron plus Gaussian read noise, Fourier sideband
recovery, and the noise sensitivity of one intensity-gradient autofocus estimator.
It is independent of the repository's dimensionless signed-amplitude encoding
benchmarks and imports none of their modules.

It demonstrates neither quantum hardware, quantum advantage, physical quantum
waves, a physical holographic display, nor a calibrated physical membrane.
It does not validate the Digital to Wave preprints, general autofocus across object
classes, or physical depth accuracy at the numerical search spacing.
All camera parameters are hypothetical, not calibrated measurements.

## Installation and reproduction

Python 3.11 or newer is recommended, consistent with the parent project.
The experiment requires only `numpy>=1.24`; it uses standard-library `unittest`.
The committed results were executed with **Python 3.12.3 and NumPy 2.4.6**.
For this checkout, from the repository root:

```bash
REPO=/home/runner/work/digital-to-wave-testbench/digital-to-wave-testbench
EXP="$REPO/experiments/holographic_autofocus"
python -m venv /tmp/holographic-autofocus-venv
. /tmp/holographic-autofocus-venv/bin/activate
python -m pip install numpy==2.4.6  # match the recorded runtime
python -m pip install -r "$EXP/requirements.txt"
python -m unittest discover -s "$EXP" -p "test_*.py" -v

SMOKE=$(mktemp -d /tmp/holographic-autofocus-smoke.XXXXXX)
python "$EXP/experiment.py" --trials 2 --seed 42 --output "$SMOKE"
```

The initial full run, before `results/` existed, was:

```bash
python "$EXP/experiment.py" --trials 40 --seed 42
```

**Fresh-directory rerun:** the committed results already exist, so the preceding
default-output command now deliberately fails before the simulation. Instead:

```bash
RERUN=$(mktemp -d /tmp/holographic-autofocus-rerun.XXXXXX)
python "$EXP/experiment.py" --trials 40 --seed 42 --output "$RERUN"
cmp "$EXP/results/trials.csv" "$RERUN/trials.csv"
cmp "$EXP/results/summary.json" "$RERUN/summary.json"
cmp "$EXP/results/config.json" "$RERUN/config.json"
```

`--trials` is per level (default 40, minimum 2); `--seed` defaults to 42.
`--output` defaults to `results/` beside the script, independent of the working
directory. Existing intended result files, including symlinks, are never
overwritten; use a fresh directory. Exclusive creation also protects against
overwriting a file created after the initial check. All three outputs are checked
for existence and nonzero size. The actual JSON summary is printed to stdout.
Other Python/NumPy versions can change floating-point details, random draws or
near-tied winners; exact byte reproduction is scoped to the recorded environment.

## Fixed model and propagation

All real arrays use float64 and complex arrays complex128.
The square grid is 96 × 96, with pixel pitch 6.5e-6 m and wavelength 532e-9 m.
For `y,x = numpy.indices((96,96))`, use `X=x-47.5`, `Y=y-47.5` in pixels.
The zero-phase object is a chosen two-Gaussian amplitude, cast to complex128:

```text
O = exp(-((X+12)^2 + (Y+8)^2)/32)
    + 0.7 exp(-((X-11)^2 + (Y-10)^2)/50).
```

This is not measured membrane data. The time convention is `exp(-i omega t)`;
positive distance is forward propagation and negative distance is back-propagation.
With unshifted `f=fftfreq(96,d=pixel_pitch_m)` and `FX,FY=meshgrid(f,f)`:

```text
q = 1/wavelength_m^2 - FX^2 - FY^2
T(z) = exp(+i z 2 pi sqrt(q)) for q >= 0; zero otherwise
U(z) = ifft2(fft2(U(0)) T(z)).
```

At zero distance `propagate` returns a copy of the input. Input/output pitch is
unchanged. Boundaries are periodic, with no padding: light wrapping across an edge
is a numerical boundary effect. Finite sampling and evanescent-component removal
limit this model. All sampled components propagate for the chosen grid; the
forward/backward test asserts that condition before checking a round trip.
Shapes, finite fields, and finite real distances are validated.

## Hologram and negative-sideband recovery

The recording distance is **+0.004 m**; the known reconstruction distance is
**−0.004 m**, used only for evaluation, never to select a focus peak.
Reference amplitude is one, with carrier bins `(24,0)`:

```text
R = exp(2j pi 24 X/96)                 # 0.25 cycles/pixel along x
S = propagate(O, +0.004, model)
H = abs(S + R)^2.
```

There is no clipping or per-frame normalization. In the shifted hologram FFT,
`bins=fftshift(fftfreq(96))*96`, `BX,BY=meshgrid(bins,bins)`, and
`r=hypot(BX+24,BY)`. The mask centered at **(−24,0)** is one through radius
6.75 bins, zero at and beyond radius 9 bins, and between them:

```text
mask = 0.5 (1 + cos(pi (r-6.75)/(9-6.75)))
selected = ifft2(ifftshift(fftshift(fft2(H)) mask))
recovered = selected R.
```

The negative sideband contains `S conjugate(R)`; multiplying by `R` removes the
carrier because reference amplitude is one. `24/96+9/96 < 0.5` and the mask excludes
DC. **These geometric checks do not guarantee complete separation from the central
band or eliminate spectral leakage.** The finite filter bandwidth also changes
the recovered field.

## Autofocus and noise

For each propagated recovered field, `I=abs(field)^2`,
`gy,gx=numpy.gradient(I)` (pixel-unit gradients, NumPy's default edge handling).
Maximize `mean(gx^2+gy^2)/mean(I^2)`. Zero/nonfinite denominators and nonfinite scores
are rejected. The score is invariant to nonzero overall amplitude scaling when
the intermediate arithmetic is representable.

The recovered sensor-plane field is Fourier transformed once; **every candidate
uses that same spectrum**, not sequential propagation. Coarse search is
`linspace(-0.008,0,65)` (0.125 mm spacing). An endpoint winner is flagged and returned
without refinement. Otherwise scan 21 evenly spaced distances between its
immediate coarse neighbors (0.0125 mm spacing). A refined endpoint winner is also
flagged. Ties choose the first maximum in ascending-distance order. This refines
only one coarse peak, not all competing peaks; no ground truth enters selection.

At each illumination level (200 then 2000 peak expected electrons), the same clean
hologram fixes `gain=peak_electrons/max(H)` and `expected_counts=gain H`. Per trial:

```text
measured_counts = rng.poisson(expected_counts), cast to float64
measured_counts += rng.normal(0, 3.0, H.shape)
noisy_hologram = measured_counts/gain.
```

One `numpy.random.default_rng(42)` serves the entire run, in level and trial order,
with Poisson drawn before Gaussian noise. The gain is fixed within each level;
recomputing it from the unchanged clean frame does not renormalize noisy trials.
Negative offset-corrected samples are allowed. No clipping, saturation,
quantization, fixed-pattern noise, or camera calibration is modeled. Expected
intensities must be finite and nonnegative with a positive maximum; peak
electrons must be finite and positive, and read-noise standard deviation finite
and nonnegative.

## Executed results (SYNTHETIC)

Command: `python "$EXP/experiment.py" --trials 40 --seed 42`.
Noise-free estimate: **−4.000000 mm**, score **0.08771226776995489**, not an endpoint.
The following table comes from the newly executed `results/summary.json`:

| Peak expected electrons | Trials | Mean distance (mm) | Sample SD (mm) | Bias (mm) | RMSE (mm) | Within ±0.25 mm | Search edges | 2.5th–97.5th percentiles (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 200 | 40 | −3.986250 | 0.260943 | +0.013750 | 0.258027 | 28/40 (70%) | 0/40 | −4.3790625 to −3.6146875 |
| 2000 | 40 | −3.984375 | 0.082855 | +0.015625 | 0.083292 | 40/40 (100%) | 0/40 | −4.1378125 to −3.8371875 |

Errors are `estimated_distance_mm - (-4.0)`: positive means a less negative
back-propagation distance. SD uses `ddof=1`; bias is mean error and RMSE is
`sqrt(mean(error^2))`. Success is inclusive `abs(error)<=0.25 mm`. All 80 trials,
including any endpoint estimates, enter statistics. Percentiles use NumPy's linear
method and describe **trial spread, not a confidence interval for true distance**.
The supplied prior observations agree at their reported precision; no material
differences were observed, no assumptions were changed, and no parameter was
tuned to match those observations.

`trials.csv` contains 80 data rows, 40 per level, with one-based trial numbers.
`config.json` records parameters, conventions, limits and runtime versions.
`summary.json` records clean and per-level results. All three were produced by
executing the script, not hand-edited.

## Validation and isolation

Executed successfully:

- The unittest command above: **20 tests passed**, covering all required cases,
  fixed-gain draw ordering, endpoint behavior, one recovered spectrum, input
  validation, dangling symlinks and output round trips.
- Two-trial smoke run into a fresh `/tmp` directory.
- Full 40-trial-per-level run and same-environment fresh-directory reproduction.
- All per-level statistics independently recomputed from the committed CSV.
- Existing checks: `python -m pytest "$REPO/tests"` (**153 passed**),
  `ruff check "$REPO"` (clean), and
  `sha256sum -c "$REPO/MODULES.sha256"` from the repository root (all eight pass).
- Historical artifact checksums verified; existing tracked files are unchanged.

No root dependency, CI, protocol lock, module hash, historical artifact, existing
scientific claim or result is changed. Environments, caches and smoke outputs are
not committed. To honor the requested directory isolation, this local README
documents the addition instead of modifying the root README/CHANGELOG normally
requested by CONTRIBUTING. No existing artifact regeneration was necessary.

## Interpretation limits

- Only one fixed synthetic scene; no general object-class autofocus validation.
- Periodic FFT propagation without padding; finite sampling, removed evanescent
  components, and a chosen filter bandwidth.
- No guaranteed complete spectral separation or elimination of leakage.
- Hypothetical, uncalibrated camera; no saturation, quantization or fixed-pattern
  noise.
- Non-universal focus metric; local refinement follows only one coarse peak.
- Numerical search-grid spacing is **not physical depth accuracy**.
- Percentiles describe trial spread, not a confidence interval for true distance.
- **40/40 successes do not prove future 100% reliability.**
- Propagation round-trip tests do not validate the entire hologram, filtering,
  noise or autofocus pipeline.
- No physical, quantum, display, calibrated-membrane or preprint validation.

## License

AGPL-3.0-only, consistent with the parent repository; see
[LICENSE](../../LICENSE) and [NOTICE](../../NOTICE).
