"""Build a compact, reproducible passport from the synthetic v8 benchmark artifact.

The passport preserves the benchmark's declared dimensionless scope, records the
source artifact and generator hashes, and summarizes only observed grid results.
It does not infer behavior between tested conditions or validate a physical channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "artifacts" / "robustness-benchmark-v8-ml-sync.json"
DEFAULT_JSON_OUTPUT = ROOT / "artifacts" / "perturbation-passport-v8-ml-sync.json"
DEFAULT_MARKDOWN_OUTPUT = ROOT / "reports" / "perturbation-passport-v8-ml-sync.md"
BENCHMARK = "synthetic_ml_sync_refinement_v8"
PASSPORT_SCHEMA = "digital-to-wave-perturbation-passport/1.0"
POOLED = "POOLED_ALL_VECTORS"
V8 = "v8_ml_refined_search"
V7 = "v7_blind_joint_search"
RAW_LS_ORACLE = "oracle_raw_ls"
PAIR_KEY = f"{V8}_vs_{V7}"

AXES = (
    ("cfo_cycles_per_symbol", "carrier-frequency offset", "cycles/symbol"),
    ("clock_error_ppm", "sample-clock error", "ppm"),
    ("timing_offset_samples", "receiver timing offset", "samples"),
)


def _format_p_value(value: float) -> str:
    return "0 (stored underflow)" if value == 0.0 else f"{value:.3g}"


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _grid(contract: dict[str, Any], field: str) -> list[float]:
    raw = contract.get(field)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"contract.{field} must be a non-empty list")
    values = [_finite_number(value, f"contract.{field}") for value in raw]
    if len(set(values)) != len(values):
        raise ValueError(f"contract.{field} must contain unique values")
    return values


def _pooled_row(
    rows: list[dict[str, Any]],
    *,
    method: str,
    cfo: float,
    ppm: float,
    timing: float,
    sigma: float,
) -> dict[str, Any] | None:
    matches = [
        row for row in rows
        if row.get("vector_pattern") == POOLED
        and row.get("method") == method
        and float(row.get("cfo_cycles_per_symbol")) == cfo
        and float(row.get("clock_error_ppm")) == ppm
        and float(row.get("timing_offset_samples")) == timing
        and float(row.get("noise_std")) == sigma
    ]
    if len(matches) > 1:
        raise ValueError("duplicate pooled result row for a condition and receiver")
    return matches[0] if matches else None


def _summary_noise_key(by_noise: dict[str, Any], sigma: float) -> str:
    for key in by_noise:
        if math.isclose(_finite_number(key, "summary noise key"), sigma, rel_tol=0.0, abs_tol=1e-12):
            return key
    raise ValueError(f"summary is missing the positive noise level {sigma}")


def _checked_aggregate(
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    method: str,
    sigma: float,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row.get("vector_pattern") == POOLED
        and row.get("method") == method
        and float(row.get("noise_std")) == sigma
    ]
    frames = sum(_positive_integer(row.get("frames"), "result.frames") for row in selected)
    passing = sum(int(row.get("frames_passing")) for row in selected)
    if not selected or aggregate.get("cells") != len(selected):
        raise ValueError(f"summary aggregate has an inconsistent cell count for {method} at noise {sigma}")
    if aggregate.get("frames") != frames or aggregate.get("frames_passing") != passing:
        raise ValueError(f"summary aggregate does not match result rows for {method} at noise {sigma}")
    interval = aggregate.get("wilson95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("summary aggregate must include a 95% Wilson interval")
    bounds = [_finite_number(value, "Wilson interval") for value in interval]
    if not 0.0 <= bounds[0] <= bounds[1] <= 1.0:
        raise ValueError("invalid Wilson interval")
    return {
        "cells": len(selected),
        "frames": frames,
        "frames_passing": passing,
        "pass_share": passing / frames,
        "wilson95": bounds,
    }


def build_passport(
    result: dict[str, Any],
    *,
    source_name: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Validate and summarize one v8 JSON result without changing the benchmark."""
    if not isinstance(result, dict) or result.get("benchmark") != BENCHMARK:
        raise ValueError(f"expected benchmark {BENCHMARK!r}")
    contract = result.get("contract")
    rows = result.get("results")
    if not isinstance(contract, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("source must contain a contract and non-empty results list")
    classification = str(contract.get("classification", "")).lower()
    if "synthetic" not in classification or "no physical validity claimed" not in classification:
        raise ValueError("source must retain the synthetic/no-physical-validity classification")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character SHA-256 digest")
    try:
        int(source_sha256, 16)
    except ValueError as exc:
        raise ValueError("source_sha256 must be hexadecimal") from exc

    grids = {
        field: _grid(contract, field)
        for field in ("cfo_cycles_per_symbol", "clock_error_ppm", "timing_offset_samples", "noise_std")
    }
    if any(value < 0.0 for value in grids["noise_std"]):
        raise ValueError("noise levels must be non-negative")
    tolerance = _finite_number(contract.get("tolerance"), "contract.tolerance")
    if tolerance <= 0:
        raise ValueError("contract.tolerance must be positive")
    trials_per_vector = _positive_integer(contract.get("trials_per_vector"), "contract.trials_per_vector")

    vector_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each result row must be an object")
        for field, _, _ in AXES:
            if field not in row or not math.isfinite(_finite_number(row[field], f"result.{field}")):
                raise ValueError(f"result row is missing a valid {field}")
        sigma = _finite_number(row.get("noise_std"), "result.noise_std")
        if sigma not in grids["noise_std"]:
            raise ValueError("result row contains a noise level outside the declared grid")
        if row.get("vector_pattern") != POOLED:
            if not isinstance(row.get("vector_pattern"), str):
                raise ValueError("result.vector_pattern must be a string")
            vector_names.add(row["vector_pattern"])
        if row.get("method") == V8 and row.get("vector_pattern") == POOLED:
            frames = _positive_integer(row.get("frames"), "result.frames")
            passing = row.get("frames_passing")
            if isinstance(passing, bool) or not isinstance(passing, int) or not 0 <= passing <= frames:
                raise ValueError("result.frames_passing must be between zero and frames")
            share = _finite_number(row.get("frame_pass_share"), "result.frame_pass_share")
            if not math.isclose(share, passing / frames, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("result.frame_pass_share does not agree with counts")

    if not vector_names:
        raise ValueError("source must contain per-vector result rows")

    positive_sigmas = sorted(value for value in grids["noise_std"] if value > 0.0)
    if not positive_sigmas:
        raise ValueError("source must include at least one positive noise level")
    summary = result.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("by_noise"), dict):
        raise ValueError("source must contain summary.by_noise")

    outcomes: list[dict[str, Any]] = []
    for sigma in positive_sigmas:
        key = _summary_noise_key(summary["by_noise"], sigma)
        noise_summary = summary["by_noise"][key]
        try:
            all_cells = noise_summary["all_cells"]
            totals = all_cells["totals"]
            paired = all_cells["paired"][PAIR_KEY]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"summary for noise {sigma} is missing required pooled comparisons") from exc
        receiver_totals: dict[str, Any] = {}
        for method in (V8, V7, RAW_LS_ORACLE):
            if method not in totals:
                raise ValueError(f"summary for noise {sigma} is missing receiver {method}")
            receiver_totals[method] = _checked_aggregate(
                rows, totals[method], method=method, sigma=sigma
            )
        paired_counts = {
            field: _positive_integer(paired.get(field), f"paired.{field}")
            for field in ("cells", "frames")
        }
        discordant_a = paired.get("a_pass_b_fail")
        discordant_b = paired.get("a_fail_b_pass")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (discordant_a, discordant_b)):
            raise ValueError("paired discordant counts must be non-negative integers")
        if paired_counts["frames"] != receiver_totals[V8]["frames"]:
            raise ValueError("paired comparison frame count does not match pooled outcomes")
        p_value = _finite_number(paired.get("mcnemar_exact_two_sided_p"), "paired McNemar p-value")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("paired McNemar p-value must be between zero and one")
        outcomes.append({
            "noise_std": sigma,
            "cells": receiver_totals[V8]["cells"],
            "frames": receiver_totals[V8]["frames"],
            "receivers": receiver_totals,
            "paired_v8_vs_v7": {
                "frames_v8_pass_v7_fail": discordant_a,
                "frames_v8_fail_v7_pass": discordant_b,
                "mcnemar_exact_two_sided_p": p_value,
            },
        })

    max_sigma = max(grids["noise_std"])
    boundary_rows = [
        row for row in rows
        if row.get("vector_pattern") == POOLED
        and row.get("method") == V8
        and float(row.get("noise_std")) == max_sigma
    ]
    boundary_rows.sort(key=lambda row: (
        _finite_number(row.get("frame_pass_share"), "result.frame_pass_share"),
        row["cfo_cycles_per_symbol"], row["clock_error_ppm"], row["timing_offset_samples"],
    ))
    expected_cells = math.prod(len(grids[field]) for field, _, _ in AXES)
    is_full = contract.get("full_protocol_run") is True
    if not boundary_rows:
        raise ValueError("source has no pooled v8 rows at the maximum tested noise level")
    if is_full and len(boundary_rows) != expected_cells:
        raise ValueError("full-protocol source does not cover every declared maximum-noise cell")

    worst_rows = boundary_rows[:3]
    worst_cells = [
        {
            "cfo_cycles_per_symbol": row["cfo_cycles_per_symbol"],
            "clock_error_ppm": row["clock_error_ppm"],
            "timing_offset_samples": row["timing_offset_samples"],
            "frames_passing": row["frames_passing"],
            "frames": row["frames"],
            "pass_share": row["frame_pass_share"],
        }
        for row in worst_rows
    ]
    weak_cell_summary = summary.get("weak_cell")
    weak_cell: dict[str, Any] | None = None
    if isinstance(weak_cell_summary, dict) and isinstance(weak_cell_summary.get("cell"), dict):
        condition = weak_cell_summary["cell"]
        weak_sigma = _finite_number(condition.get("noise_std"), "summary.weak_cell.noise_std")
        cfo = _finite_number(condition.get("cfo_cycles_per_symbol"), "summary.weak_cell.cfo")
        ppm = _finite_number(condition.get("clock_error_ppm"), "summary.weak_cell.clock_error_ppm")
        timing = _finite_number(condition.get("timing_offset_samples"), "summary.weak_cell.timing_offset_samples")
        v8_row = _pooled_row(rows, method=V8, cfo=cfo, ppm=ppm, timing=timing, sigma=weak_sigma)
        v7_row = _pooled_row(rows, method=V7, cfo=cfo, ppm=ppm, timing=timing, sigma=weak_sigma)
        if v8_row is not None and v7_row is not None:
            weak_cell = {
                "condition": {
                    "cfo_cycles_per_symbol": cfo,
                    "clock_error_ppm": ppm,
                    "timing_offset_samples": timing,
                    "noise_std": weak_sigma,
                },
                "frames": weak_cell_summary.get("frames"),
                "v8_frames_passing": v8_row["frames_passing"],
                "v8_pass_share": v8_row["frame_pass_share"],
                "v7_frames_passing": v7_row["frames_passing"],
                "v7_pass_share": v7_row["frame_pass_share"],
                "v8_joint_acq_loose_frames": v8_row.get("joint_acq_loose_frames"),
                "v7_joint_acq_loose_frames": v7_row.get("joint_acq_loose_frames"),
                "paired": weak_cell_summary.get("paired", {}).get(PAIR_KEY),
            }

    criteria_summary = summary.get("success_criteria", {})
    criteria = {
        "evaluated_on_full_protocol_grid": criteria_summary.get("evaluated_on_full_protocol_grid"),
        "improvement_real": criteria_summary.get("improvement_real"),
    }
    for key in ("S1_weak_cell", "S2_no_v7_grid_regression", "S3_not_worse_than_v7_oracle_on_v7_grid",
                "S4_extension_gain"):
        if isinstance(criteria_summary.get(key), dict):
            criteria[key] = {
                "rule": criteria_summary[key].get("rule"),
                "pass": criteria_summary[key].get("pass"),
            }

    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol_lock = contract.get("protocol_lock")
    return {
        "schema": PASSPORT_SCHEMA,
        "title": "Perturbation Passport — v8 blind synchronization",
        "evidence_label": "SYNTHETIC",
        "classification": classification,
        "physical_validity_claimed": False,
        "source": {
            "benchmark": BENCHMARK,
            "artifact": Path(source_name).name,
            "artifact_sha256": source_sha256.lower(),
            "protocol_status": contract.get("protocol_status"),
            "protocol_lock": protocol_lock,
            "git_commit": contract.get("git_commit"),
            "full_protocol_run": is_full,
            "master_seed": contract.get("master_seed"),
            "vector_seed": contract.get("vector_seed"),
            "module_sha256": contract.get("module_sha256"),
            "passport_generator": Path(__file__).name,
            "passport_generator_sha256": generator_digest,
        },
        "experiment": {
            "perturbation_axes": [
                {"name": name, "field": field, "unit": unit, "tested_values": grids[field]}
                for field, name, unit in AXES
            ],
            "sample_noise_std": grids["noise_std"],
            "declared_condition_cells": expected_cells,
            "observed_max_noise_cells": len(boundary_rows),
            "trials_per_vector_at_positive_noise": trials_per_vector,
            "vector_patterns": sorted(vector_names),
            "vector_count": len(vector_names),
            "vector_seed": contract.get("vector_seed"),
            "frame_pass_rule": f"max absolute decoded-value error across 32 values <= {tolerance:g}",
            "signal_model": contract.get("model"),
            "noise_model": (
                "For positive noise_std, independent zero-mean Gaussian samples with the declared standard deviation "
                "are added to each generated buffer; every receiver is scored on the same noisy buffer."
            ),
            "receiver_definitions": contract.get("receivers"),
        },
        "assumptions_and_limitations": list(contract.get("limitations", [])),
        "outcomes_by_noise": outcomes,
        "weak_cell": weak_cell,
        "observed_failure_boundary": {
            "definition": (
                "The lowest empirical pooled pass-share cell among the discrete conditions tested at the highest "
                "declared noise_std; no interpolation or extrapolation is performed."
            ),
            "maximum_tested_noise_std": max_sigma,
            "cells_tested": len(boundary_rows),
            "cells_with_at_least_one_failed_frame": sum(
                int(row["frames_passing"]) < int(row["frames"]) for row in boundary_rows
            ),
            "minimum_observed_pass_share": worst_cells[0]["pass_share"],
            "weakest_observed_cells": worst_cells,
            "outside_tested_grid": "Untested; no claim is made outside the finite grid above.",
        },
        "pre_stated_criteria": criteria,
    }


def render_markdown(passport: dict[str, Any]) -> str:
    """Render a readable report from a passport object."""
    source = passport["source"]
    experiment = passport["experiment"]
    lines = [
        "# Perturbation Passport — v8 blind synchronization",
        "",
        f"> **{passport['evidence_label']}** — synthetic dimensionless benchmark. "
        "No physical or biological validity is claimed.",
        "",
        "## Provenance",
        "",
        f"- Source artifact: `{source['artifact']}`",
        f"- Source SHA-256: `{source['artifact_sha256']}`",
        f"- Benchmark commit: `{source['git_commit']}`",
        f"- Master / vector seed: `{source['master_seed']}` / `{source['vector_seed']}`",
        f"- Full declared protocol: `{str(source['full_protocol_run']).lower()}`",
        f"- Protocol lock: `{source['protocol_lock']}`",
        f"- Passport generator SHA-256: `{source['passport_generator_sha256']}`",
        "",
        "## Controlled perturbations",
        "",
        "| Axis | Unit | Tested values |",
        "|---|---|---|",
    ]
    for axis in experiment["perturbation_axes"]:
        values = ", ".join(f"{value:g}" for value in axis["tested_values"])
        lines.append(f"| {axis['name']} | {axis['unit']} | {values} |")
    noise = ", ".join(f"{value:g}" for value in experiment["sample_noise_std"])
    lines.extend([
        f"| Additive sample-noise standard deviation | dimensionless | {noise} |",
        "",
        f"- Grid: {experiment['declared_condition_cells']} declared impairment cells; "
        f"{experiment['trials_per_vector_at_positive_noise']} frames per vector at positive noise; "
        f"{experiment['vector_count']} fixed input patterns (seed `{experiment['vector_seed']}`).",
        f"- Pass rule: `{experiment['frame_pass_rule']}`.",
        "",
        "## Aggregate outcomes",
        "",
        "Pass shares and Wilson 95% intervals are conditional on the benchmark's fixed vector panel and noise draws.",
        "",
        "| Noise std | Receiver | Passing / frames | Pass share | Wilson 95% |",
        "|---:|---|---:|---:|---:|",
    ])
    for outcome in passport["outcomes_by_noise"]:
        for method in (V8, V7, RAW_LS_ORACLE):
            values = outcome["receivers"][method]
            lo, hi = values["wilson95"]
            lines.append(
                f"| {outcome['noise_std']:g} | `{method}` | {values['frames_passing']:,} / {values['frames']:,} "
                f"| {values['pass_share']:.2%} | {lo:.2%}–{hi:.2%} |"
            )
        paired = outcome["paired_v8_vs_v7"]
        lines.append(
            f"| {outcome['noise_std']:g} | paired v8 vs v7 | "
            f"{paired['frames_v8_pass_v7_fail']:,} v8-only passes / "
            f"{paired['frames_v8_fail_v7_pass']:,} v7-only passes | — | "
            f"McNemar p = {_format_p_value(paired['mcnemar_exact_two_sided_p'])} |"
        )
    lines.extend(["", "## Observed failure boundary", ""])
    boundary = passport["observed_failure_boundary"]
    lines.append(
        f"At the highest tested noise (σ = {boundary['maximum_tested_noise_std']:g}), "
        f"{boundary['cells_with_at_least_one_failed_frame']} of {boundary['cells_tested']} cells contain at least "
        "one frame outside the pass rule. The weakest pooled v8 cells were:"
    )
    lines.extend([
        "",
        "| CFO (cycles/symbol) | Clock error (ppm) | Timing (samples) | Passing / frames | Pass share |",
        "|---:|---:|---:|---:|---:|",
    ])
    for cell in boundary["weakest_observed_cells"]:
        lines.append(
            f"| {cell['cfo_cycles_per_symbol']:+g} | {cell['clock_error_ppm']:+g} "
            f"| {cell['timing_offset_samples']:+g} | {cell['frames_passing']:,} / {cell['frames']:,} "
            f"| {cell['pass_share']:.2%} |"
        )
    lines.extend([
        "",
        f"**Boundary meaning:** {boundary['definition']} {boundary['outside_tested_grid']}",
        "",
    ])
    if passport["weak_cell"]:
        weak = passport["weak_cell"]
        cond = weak["condition"]
        paired = weak.get("paired") or {}
        lines.extend([
            "## Pre-identified weak cell",
            "",
            f"CFO `{cond['cfo_cycles_per_symbol']:g}`, clock error `{cond['clock_error_ppm']:g} ppm`, "
            f"timing `{cond['timing_offset_samples']:g} samples`, σ `{cond['noise_std']:g}`: "
            f"v8 passes {weak['v8_frames_passing']:,}/{weak['frames']:,} frames "
            f"({weak['v8_pass_share']:.2%}; loose joint acquisition "
            f"{weak['v8_joint_acq_loose_frames']:,}/{weak['frames']:,}), versus v7 "
            f"{weak['v7_frames_passing']:,}/{weak['frames']:,} ({weak['v7_pass_share']:.2%}; loose joint acquisition "
            f"{weak['v7_joint_acq_loose_frames']:,}/{weak['frames']:,}). "
            f"Paired McNemar p = {_format_p_value(paired.get('mcnemar_exact_two_sided_p', float('nan')))}.",
            "",
        ])
    lines.extend(["## Assumptions and limitations", ""])
    lines.append(f"- Signal model: `{experiment['signal_model']}`.")
    lines.append(f"- Noise model: {experiment['noise_model']}")
    for limitation in passport["assumptions_and_limitations"]:
        lines.append(f"- {limitation}.")
    criteria = passport["pre_stated_criteria"]
    if criteria.get("improvement_real") is not None:
        lines.extend([
            "",
            f"Pre-stated v8 improvement verdict: `{criteria['improvement_real']}` "
            f"(full-grid evaluation: `{criteria.get('evaluated_on_full_protocol_grid')}`).",
        ])
    lines.extend([
        "",
        "## Reproduce this passport",
        "",
        "```bash",
        "python perturbation_passport.py",
        "```",
        "",
        "The default command reads the source JSON, verifies its declared v8 schema and aggregate counts, then writes "
        "the JSON passport and this report. The source and generator SHA-256 values identify the exact inputs used.",
        "",
    ])
    return "\n".join(lines)


def write_passport(
    passport: dict[str, Any],
    *,
    json_output: Path,
    markdown_output: Path,
) -> tuple[Path, Path]:
    json_path, markdown_path = Path(json_output), Path(markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(passport, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(passport), encoding="utf-8")
    return json_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="v8 benchmark result JSON")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT, help="output passport JSON")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT,
                        help="output human-readable Markdown report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = args.input.read_bytes()
    result = json.loads(payload)
    passport = build_passport(
        result,
        source_name=args.input.name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )
    json_path, markdown_path = write_passport(
        passport,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Source SHA-256: {passport['source']['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
