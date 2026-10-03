"""The original v1-v6 modules and the v7 and v8 protocol modules must stay byte-identical."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LOCKED = {
    "digital_to_wave.py": "866cab09ae3b64387de9fc9909288dac37f9f81e7e2f6603b2e786674d46ac18",
    "robustness_benchmark_v2.py": "dc3a1166c873892c9b861248b53037217c361e8939b48b96d5de3fc7c00e5c0f",
    "robustness_benchmark_v3.py": "ec8c9d7a9e1aac1b807a3d577ba7254781e32fc6f6895cea6555ac14270d6222",
    "robustness_benchmark_v4.py": "248c7672d310df2c06cf2fbbdb6d4de11090d849b5f05b7c4510f5a60da181e6",
    "robustness_benchmark_v5_timing.py": "1bd9500ce3ceadadae25ce6731bc8ea9058adefa2781adf4d16d403b6c383419",
}

# v6 protocol module: committed and hash-locked before the full v6 run (reports/v6-timing-recovery-protocol-lock.md).
LOCKED_V6 = {
    "robustness_benchmark_v6_timing_recovery.py": "c3f2da47e9da81072cbdc02153c28f6c9b6f5958251a4c91f8d64f9e5cfa112b",
}
LOCKED_V7 = {
    "robustness_benchmark_v7_clock_drift_cfo.py": "693ac18c9202f36a004610147f76642ae3f2859b64dd39e1d5c4a7f0cc6f7e84",
}
# v8 protocol module: committed and hash-locked before the full v8 run (reports/v8-ml-sync-protocol-lock.md).
LOCKED_V8 = {
    "robustness_benchmark_v8_ml_sync.py": "77422f5f5d430749d911d05ecd678c08b73e51c7b7fd38d10aec56958570b9cf",
}


@pytest.mark.parametrize("name,digest", sorted({**LOCKED, **LOCKED_V6, **LOCKED_V7, **LOCKED_V8}.items()))
def test_module_sha256_matches_lock(name, digest):
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_lock_file_agrees_with_test_table():
    lines = (ROOT / "MODULES.sha256").read_text(encoding="utf-8").split("\n")
    table = {}
    for line in lines:
        if line.strip():
            digest, name = line.split()
            table[name] = digest
    assert table == {**LOCKED, **LOCKED_V6, **LOCKED_V7, **LOCKED_V8}


def test_artifact_checksums_file_is_consistent():
    for line in (ROOT / "artifacts" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        path = ROOT / "artifacts" / name
        assert path.is_file(), name
        if path.suffix in (".csv", ".png"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
