import math

import numpy as np
import pytest

from digital_to_wave import WaveConfig, decode_wave, encode_wave, reconstruction_metrics


def test_round_trip_is_exact_without_noise():
    values = np.array([-2.0, -0.5, 0.0, 0.25, 1.0, 2.0])
    decoded = decode_wave(encode_wave(values))
    assert np.allclose(decoded, values, atol=1e-12)


def test_carrier_advances_22_5_degrees_per_sample():
    config = WaveConfig()
    omega_deg = 360.0 * config.cycles_per_symbol / config.samples_per_symbol
    assert omega_deg == pytest.approx(22.5)
    segment = encode_wave([1.0], config)[0]
    assert segment[0] == pytest.approx(math.sin(math.pi / 4))


def test_range_is_enforced():
    with pytest.raises(ValueError):
        encode_wave([2.5])


def test_metrics_inclusive_tolerance():
    m = reconstruction_metrics([0.0, 0.0], [0.25, -0.1], tolerance=0.25)
    assert m["pass"] is True and m["components_over_tolerance"] == 0
