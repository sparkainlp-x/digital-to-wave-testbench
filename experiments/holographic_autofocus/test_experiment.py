"""Tests for the isolated SYNTHETIC experiment; AGPL-3.0-only."""

import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("holographic_experiment", Path(__file__).with_name("experiment.py"))
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = experiment.make_model()
        cls.clean = experiment.clean_hologram(cls.model)
        cls.small = experiment.run_experiment(trials_per_level=2, seed=42)

    def test_zero_distance_identity_and_copy(self):
        field = self.model.object_field
        result = experiment.propagate(field, 0.0, self.model)
        np.testing.assert_array_equal(result, field)
        self.assertFalse(np.shares_memory(result, field))
        self.assertEqual(result.dtype, np.complex128)

    def test_periodic_forward_backward_roundtrip(self):
        self.assertTrue(np.all(self.model.propagating))
        rng = np.random.default_rng(73)
        field = rng.normal(size=(96, 96)) + 1j * rng.normal(size=(96, 96))
        forward = experiment.propagate(field, 0.004, self.model)
        backward = experiment.propagate(forward, -0.004, self.model)
        np.testing.assert_allclose(backward, field, rtol=1e-12, atol=1e-12)

    def test_hologram_finite_nonnegative(self):
        self.assertEqual(self.clean.shape, (96, 96))
        self.assertEqual(self.clean.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(self.clean)))
        self.assertTrue(np.all(self.clean >= 0))

    def test_mask_nyquist_and_dc_exclusion(self):
        # This checks geometry, not complete spectral separation.
        self.assertLess(24 / 96 + 9 / 96, 0.5)
        self.assertEqual(self.model.mask[48, 48], 0)
        self.assertEqual(self.model.mask[48, 24], 1)
        self.assertEqual(self.model.mask[48, 15], 0)
        bins = np.fft.fftshift(np.fft.fftfreq(96)) * 96
        bx, by = np.meshgrid(bins, bins)
        r = np.hypot(bx + 24, by)
        np.testing.assert_array_equal(self.model.mask[r <= 6.75], 1)
        np.testing.assert_array_equal(self.model.mask[r >= 9], 0)
        transition = (r > 6.75) & (r < 9)
        np.testing.assert_allclose(
            self.model.mask[transition], 0.5 * (1 + np.cos(np.pi * (r[transition] - 6.75) / 2.25)),
        )

    def test_same_seed_identical_noisy_arrays(self):
        a = experiment.add_camera_noise(self.clean, np.random.default_rng(42), 200)
        b = experiment.add_camera_noise(self.clean, np.random.default_rng(42), 200)
        np.testing.assert_array_equal(a, b)

    def test_noise_fixed_gain_and_draw_order(self):
        actual_rng, expected_rng = np.random.default_rng(81), np.random.default_rng(81)
        gain = 200 / self.clean.max()
        for _ in range(2):
            expected = expected_rng.poisson(gain * self.clean).astype(np.float64)
            expected += expected_rng.normal(0, 3.0, self.clean.shape)
            actual = experiment.add_camera_noise(self.clean, actual_rng, 200)
            np.testing.assert_array_equal(actual, expected / gain)
        rng = mock.Mock()
        rng.poisson.return_value = np.zeros(self.clean.shape, dtype=np.int64)
        rng.normal.return_value = -np.ones(self.clean.shape)
        noisy = experiment.add_camera_noise(self.clean, rng, 200)
        self.assertTrue(np.all(noisy < 0))

    def test_invalid_camera_noise_parameters(self):
        rng = np.random.default_rng(0)
        for peak in (0, -1, np.nan, np.inf, -np.inf, 1j):
            with self.subTest(peak=peak), self.assertRaises(ValueError):
                experiment.add_camera_noise(self.clean, rng, peak)
        for read_noise in (-1, np.nan, np.inf, -np.inf, 1j):
            with self.subTest(read_noise=read_noise), self.assertRaises(ValueError):
                experiment.add_camera_noise(self.clean, rng, 200, read_noise)
        for field in (-self.clean, np.zeros((2, 2)), np.full((2, 2), np.nan), self.clean.astype(complex)):
            with self.subTest(field=field.shape), self.assertRaises(ValueError):
                experiment.add_camera_noise(field, rng, 200)
        self.assertTrue(np.all(np.isfinite(experiment.add_camera_noise(self.clean, rng, 200, 0))))

    def test_focus_amplitude_scale_invariance(self):
        field = self.model.object_field
        baseline = experiment.focus_score(field)
        for scale in (0.01, -3.0, 2 + 5j, 100):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(experiment.focus_score(scale * field), baseline, places=13)

    def test_clean_focus_within_one_coarse_step(self):
        estimate, score, edge = experiment.estimate_focus(self.clean, self.model)
        self.assertLessEqual(abs(estimate + 0.004), 0.000125)
        self.assertGreater(score, 0)
        self.assertFalse(edge)

    def test_two_trials_per_level(self):
        rows = self.small["trials"]
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["peak_expected_electrons"] for r in rows], [200, 200, 2000, 2000])
        self.assertEqual([r["trial"] for r in rows], [1, 2, 1, 2])

    def test_summary_recomputed_from_rows(self):
        for stats in self.small["summary"]["levels"]:
            rows = [r for r in self.small["trials"] if r["peak_expected_electrons"] == stats["peak_expected_electrons"]]
            estimates = np.array([r["estimated_distance_mm"] for r in rows])
            errors = estimates - (-4.0)
            np.testing.assert_array_equal(errors, [r["error_mm"] for r in rows])
            self.assertEqual(stats["trial_count"], len(rows))
            expected = {
                "mean_estimated_distance_mm": np.mean(estimates),
                "sample_standard_deviation_mm": np.std(estimates, ddof=1),
                "bias_mm": np.mean(errors), "rmse_mm": np.sqrt(np.mean(errors**2)),
                "within_tolerance_fraction": np.mean(np.abs(errors) <= 0.25),
                "search_edge_fraction": np.mean([r["at_scan_edge"] for r in rows]),
                "estimated_distance_percentile_2_5_mm": np.percentile(estimates, 2.5),
                "estimated_distance_percentile_97_5_mm": np.percentile(estimates, 97.5),
            }
            for key, value in expected.items():
                self.assertAlmostEqual(stats[key], value, places=14)

    def test_existing_files_not_overwritten_before_computation(self):
        for name in experiment.RESULT_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                path = output / name
                path.write_text("sentinel", encoding="utf-8")
                with mock.patch.object(experiment, "run_experiment") as run:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                        experiment.main(["--trials", "2", "--output", directory])
                    self.assertEqual(exc.exception.code, 2)
                    run.assert_not_called()
                with self.assertRaises(FileExistsError):
                    experiment.write_results(self.small, output)
                self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")
                self.assertEqual(list(output.iterdir()), [path])

    def test_dangling_output_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            path.symlink_to(Path(directory) / "missing")
            with self.assertRaises(FileExistsError):
                experiment.check_output_directory(directory)

    def test_zero_nonfinite_focus_fields_rejected(self):
        for field in (
            np.zeros((96, 96)), np.full((96, 96), np.nan), np.full((96, 96), np.inf),
            np.full((2, 2), 1e300), np.ones((1, 2)), np.ones(96),
        ):
            with self.subTest(shape=field.shape), self.assertRaises(ValueError):
                experiment.focus_score(field)

    def test_propagation_and_recovery_input_validation(self):
        for field in (np.ones((95, 96)), np.full((96, 96), np.nan)):
            with self.assertRaises(ValueError):
                experiment.propagate(field, 0.004, self.model)
            with self.assertRaises(ValueError):
                experiment.recover_field(field, self.model)
        for distance in (np.nan, np.inf, -np.inf, 1j):
            with self.assertRaises(ValueError):
                experiment.propagate(self.model.object_field, distance, self.model)

    def test_trial_and_summary_values_finite(self):
        def check(value):
            if isinstance(value, dict):
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value))
        check(self.small["trials"])
        check(self.small["summary"])

    def test_written_files_nonempty_and_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment.write_results(self.small, directory)
            for name in experiment.RESULT_NAMES:
                self.assertGreater((Path(directory) / name).stat().st_size, 0)
            self.assertNotIn(b"\r", (Path(directory) / "trials.csv").read_bytes())
            with (Path(directory) / "trials.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]["error_mm"]), float(rows[0]["estimated_distance_mm"]) + 4.0)
            for name, key in (("config.json", "config"), ("summary.json", "summary")):
                self.assertEqual(json.loads((Path(directory) / name).read_text()), self.small[key])

    def test_endpoint_handling(self):
        with mock.patch.object(experiment, "focus_score", side_effect=range(65)) as score:
            distance, _, edge = experiment.estimate_focus(self.clean, self.model)
            self.assertEqual(distance, 0)
            self.assertTrue(edge)
            self.assertEqual(score.call_count, 65)
        coarse = [0.0] * 65
        coarse[32] = 1.0
        with mock.patch.object(experiment, "focus_score", side_effect=coarse + list(range(21))) as score:
            distance, _, edge = experiment.estimate_focus(self.clean, self.model)
            self.assertAlmostEqual(distance, experiment.COARSE_DISTANCES_M[33])
            self.assertTrue(edge)
            self.assertEqual(score.call_count, 86)

    def test_one_recovered_spectrum_per_search(self):
        with mock.patch.object(experiment, "recover_field", wraps=experiment.recover_field) as recover:
            with mock.patch.object(np.fft, "fft2", wraps=np.fft.fft2) as fft:
                experiment.estimate_focus(self.clean, self.model)
        recover.assert_called_once()
        # One FFT for the hologram, one for the recovered field; none per candidate.
        self.assertEqual(fft.call_count, 2)

    def test_invalid_trial_counts_and_seed(self):
        for trials in (0, 1, -2, 2.5, True):
            with self.subTest(trials=trials), self.assertRaises(ValueError):
                experiment.run_experiment(trials)
        for seed in (-1, 0.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                experiment.run_experiment(2, seed)


if __name__ == "__main__":
    unittest.main()
