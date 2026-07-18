"""Markdown report and comparison generation for persisted runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ornith_mlx_eval.results import (
    ResultArtifactError,
    load_run_artifacts,
    truncate_text,
)


FIXED_COMPARE_INVARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("schema_version", ("schema_version",)),
    ("suite_hash", ("suite", "suite_hash")),
    ("prompt_template_hash", ("suite", "prompt_template_hash")),
    ("tokenizer_identity", ("model", "tokenizer_identity")),
    ("chat_template_identity", ("model", "chat_template_identity")),
    ("decoding", ("settings", "decoding")),
    ("runtime", ("runtime", "kind")),
    ("grader_version", ("harness", "grader_version")),
    ("prompt_order", ("settings", "prompt_order")),
    ("seed", ("settings", "seed")),
    ("concurrency", ("settings", "concurrency")),
)

DECLARED_MODEL_VARIABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("model_repo", ("model", "repo_id")),
    ("model_revision", ("model", "revision")),
    ("quantization", ("model", "quantization")),
    ("model_variant", ("model", "variant")),
)


def regenerate_report(run_dir: Path) -> Path:
    """Regenerate ``report.md`` from persisted run files only."""
    manifest, rows, summary = load_run_artifacts(run_dir)
    report = render_report(manifest, rows, summary)
    output = run_dir / "report.md"
    output.write_text(report, encoding="utf-8")
    return output


def render_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Render deterministic Markdown for a run."""
    totals = summary["totals"]
    settings = manifest["settings"]
    suite = manifest["suite"]
    model = manifest["model"]
    runtime = manifest["runtime"]

    decoding_json = json.dumps(settings["decoding"], sort_keys=True, separators=(",", ":"))

    lines: list[str] = [
        "# Ornith MLX Eval Report",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Classification: `{summary['classification']}`",
        "",
        "## Environment",
        "",
        f"- Harness: `{manifest['harness']['name']} {manifest['harness']['version']}`",
        f"- Runtime: `{runtime['kind']}`",
        f"- Model: `{model['repo_id']}`",
        f"- Revision: `{model['revision']}`",
        f"- Python: `{manifest['environment']['python_version']}`",
        f"- Platform: `{manifest['environment']['platform']}`",
        "",
        "## Run Settings",
        "",
        f"- Suite: `{suite['suite_id']}`",
        f"- Suite hash: `{suite['suite_hash']}`",
        f"- Prompt-template hash: `{suite['prompt_template_hash']}`",
        f"- Seed: `{settings['seed']}`",
        f"- Decoding: `{decoding_json}`",
        f"- Limit: `{settings.get('limit')}`",
        f"- Prompt order: `{', '.join(settings['prompt_order'])}`",
        "",
        "## Score Summary",
        "",
        f"- Scored cases: `{totals['scored']}`",
        f"- Passed: `{totals['passed']}`",
        f"- Failed: `{totals['failed']}`",
        f"- Pass rate: `{summary['pass_rate']:.3f}`",
        f"- Average score: `{summary['average_score']:.3f}`",
        f"- Parse failures: `{totals['parse_failures']}`",
        f"- Truncated responses: `{totals['truncated']}`",
        f"- Resource stops: `{totals['resource_stops']}`",
        "",
        "## Failures, Skips, And Errors",
        "",
    ]

    failures = [row for row in rows if row.get("scored") and not row["grade"]["passed"]]
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['case_id']}`: {row['grade']['reason']} "
                f"(parse: `{row['parse']['status']}`)"
            )
            final = truncate_text(str(row["parse"].get("final_text", "")), 220)
            if final:
                lines.append(f"  Final text: `{final}`")
    else:
        lines.append("- None")

    performance = summary.get("performance", {})
    resources = summary.get("resources", {})
    lines.extend([
        "",
        "## Performance",
        "",
        f"- Total wall time seconds: `{sum(float(row['timing']['wall_seconds']) for row in rows):.6f}`",
        f"- Total generated tokens: `{sum(int(row['tokens']['generated']) for row in rows)}`",
    ])
    if runtime["kind"] == "mlx":
        lines.extend([
            f"- Cold-load time seconds: `{float(performance.get('cold_load_seconds', 0)):.6f}`",
            f"- First-token time seconds: `{max((float(row['timing'].get('first_token_seconds', 0)) for row in rows), default=0):.6f}`",
            f"- Decode tokens per second: `{float(performance.get('decode_tokens_per_second', 0)):.6f}`",
        ])
    lines.extend([
        "",
        "## Resources",
        "",
        f"- Peak MLX memory bytes: `{max(int(row['resources'].get('peak_mlx_memory_bytes', 0)) for row in rows) if rows else 0}`",
        f"- Runtime resource status: `{runtime.get('resource_status', 'not-measured')}`",
    ])
    if runtime["kind"] == "mlx":
        lines.extend([
            f"- Disk free before bytes: `{int(resources.get('disk_free_before_bytes', 0) or 0)}`",
            f"- Disk free after bytes: `{int(resources.get('disk_free_after_bytes', 0) or 0)}`",
            f"- Memory pressure after: `{resources.get('memory_pressure_after', 'unavailable')}`",
            f"- Swap delta bytes: `{resources.get('swap_delta_bytes', 'unavailable')}`",
        ])
    lines.extend([
        "",
        "## Caveats",
        "",
    ])
    if summary.get("smoke_only"):
        lines.append("- This is smoke-only output and is not benchmark-quality evidence.")
    if runtime["kind"] == "mock":
        lines.append("- Mock runtime uses deterministic synthetic responses and downloads no model weights.")
    lines.append("- Hidden expected-answer metadata is not serialized in this report.")
    lines.append("")
    return "\n".join(lines)


def compare_runs(run_a: Path, run_b: Path, *, output: Path | None = None, allow_mismatch: bool = False) -> Path:
    """Compare two completed persisted runs and write Markdown output."""
    manifest_a, rows_a, summary_a = load_run_artifacts(run_a)
    manifest_b, rows_b, summary_b = load_run_artifacts(run_b)
    if output is None and run_a.resolve().parent != run_b.resolve().parent:
        raise ResultArtifactError(
            "Runs with different parent directories require an explicit --output path"
        )

    mismatches = _fixed_mismatches(manifest_a, manifest_b)
    smoke_only = bool(summary_a.get("smoke_only") or summary_b.get("smoke_only"))
    if smoke_only:
        mismatches.append({
            "key": "smoke_only",
            "a": summary_a.get("smoke_only"),
            "b": summary_b.get("smoke_only"),
        })

    if mismatches and not allow_mismatch:
        details = ", ".join(m["key"] for m in mismatches)
        raise ResultArtifactError(f"Fixed-invariant mismatch: {details}")

    model_variables = _model_variable_differences(manifest_a, manifest_b)
    markdown = render_compare(
        manifest_a,
        summary_a,
        manifest_b,
        summary_b,
        mismatches=mismatches,
        model_variables=model_variables,
        qualitative=bool(mismatches),
    )

    output_path = output or (
        run_a.parent / f"compare_{run_a.name}_vs_{run_b.name}.md"
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def render_compare(
    manifest_a: dict[str, Any],
    summary_a: dict[str, Any],
    manifest_b: dict[str, Any],
    summary_b: dict[str, Any],
    *,
    mismatches: list[dict[str, Any]],
    model_variables: list[dict[str, Any]],
    qualitative: bool,
) -> str:
    """Render deterministic comparison Markdown."""
    title = "Qualitative comparison only" if qualitative else "Comparable run comparison"
    lines: list[str] = [
        "# Ornith MLX Eval Comparison",
        "",
        title,
        "",
        f"Run A: `{manifest_a['run_id']}`",
        f"Run B: `{manifest_b['run_id']}`",
        "",
        "## Declared Model Variables",
        "",
    ]
    if model_variables:
        for item in model_variables:
            lines.append(f"- `{item['key']}`: `{item['a']}` -> `{item['b']}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Fixed-Invariant Mismatches", ""])
    if mismatches:
        for item in mismatches:
            lines.append(f"- `{item['key']}`: `{item['a']}` vs `{item['b']}`")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Score Delta",
        "",
        f"- Pass rate: `{summary_a['pass_rate']:.3f}` -> `{summary_b['pass_rate']:.3f}`",
        f"- Average score: `{summary_a['average_score']:.3f}` -> `{summary_b['average_score']:.3f}`",
        f"- Passed: `{summary_a['totals']['passed']}` -> `{summary_b['totals']['passed']}`",
        f"- Failed: `{summary_a['totals']['failed']}` -> `{summary_b['totals']['failed']}`",
        "",
        "## Performance Delta",
        "",
        f"- Wall seconds: `{_summary_float(summary_a, 'performance', 'wall_seconds'):.6f}` -> "
        f"`{_summary_float(summary_b, 'performance', 'wall_seconds'):.6f}`",
        f"- Generated tokens: `{_summary_int(summary_a, 'performance', 'generated_tokens')}` -> "
        f"`{_summary_int(summary_b, 'performance', 'generated_tokens')}`",
        f"- Decode tokens per second: "
        f"`{_summary_float(summary_a, 'performance', 'decode_tokens_per_second'):.6f}` -> "
        f"`{_summary_float(summary_b, 'performance', 'decode_tokens_per_second'):.6f}`",
        "",
        "## Resource Delta",
        "",
        f"- Peak MLX memory bytes: `{_summary_int(summary_a, 'resources', 'peak_mlx_memory_bytes')}` -> "
        f"`{_summary_int(summary_b, 'resources', 'peak_mlx_memory_bytes')}`",
        f"- Disk free after bytes: `{_summary_int(summary_a, 'resources', 'disk_free_after_bytes')}` -> "
        f"`{_summary_int(summary_b, 'resources', 'disk_free_after_bytes')}`",
        f"- Memory pressure after: "
        f"`{_summary_value(summary_a, 'resources', 'memory_pressure_after')}` -> "
        f"`{_summary_value(summary_b, 'resources', 'memory_pressure_after')}`",
        f"- Swap delta bytes: `{_summary_int(summary_a, 'resources', 'swap_delta_bytes')}` -> "
        f"`{_summary_int(summary_b, 'resources', 'swap_delta_bytes')}`",
        "",
        "## Caveats",
        "",
    ])
    if qualitative:
        lines.append("- Qualitative comparison only; do not use this as benchmark ranking evidence.")
    if summary_a.get("smoke_only") or summary_b.get("smoke_only"):
        lines.append("- At least one run is smoke-only; benchmark-quality claims are disabled.")
    lines.append("")
    return "\n".join(lines)


def _fixed_mismatches(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key, path in FIXED_COMPARE_INVARIANTS:
        av = _get_path(a, path)
        bv = _get_path(b, path)
        if av != bv:
            mismatches.append({"key": key, "a": av, "b": bv})
    return mismatches


def _model_variable_differences(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for key, path in DECLARED_MODEL_VARIABLES:
        av = _get_path(a, path)
        bv = _get_path(b, path)
        if av != bv:
            diffs.append({"key": key, "a": av, "b": bv})
    return diffs


def _get_path(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _summary_value(summary: dict[str, Any], section: str, key: str) -> Any:
    value = summary.get(section, {}).get(key)
    return "not-measured" if value is None else value


def _summary_float(summary: dict[str, Any], section: str, key: str) -> float:
    value = summary.get(section, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _summary_int(summary: dict[str, Any], section: str, key: str) -> int:
    value = summary.get(section, {}).get(key)
    return int(value) if isinstance(value, (int, float)) else 0
