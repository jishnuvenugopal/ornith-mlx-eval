"""Run artifact schema helpers for Ornith MLX Eval.

This module owns the persisted artifact contract used by ``run``, ``report``,
and ``compare``.  The schemas are intentionally small and explicit so unit
tests and CLI validators can audit generated files without loading models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ornith-result-v1"
GRADER_VERSION = "ornith-graders-v1"
MOCK_REVISION = "mock-revision-v1"


class ResultArtifactError(ValueError):
    """Raised when persisted run artifacts are missing or invalid."""


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write deterministic pretty JSON to *path*."""
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows deterministically."""
    lines = [
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_json(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON object from *path* with clear user-facing errors."""
    if not path.exists():
        raise ResultArtifactError(f"Missing required {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultArtifactError(f"Corrupt {label}: {path}: {exc}")
    if not isinstance(data, dict):
        raise ResultArtifactError(f"Invalid {label}: {path} must contain a JSON object")
    return data


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    """Load JSONL object rows from *path* with clear errors."""
    if not path.exists():
        raise ResultArtifactError(f"Missing required {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResultArtifactError(f"Corrupt {label}: {path}:{line_no}: {exc}")
        if not isinstance(row, dict):
            raise ResultArtifactError(f"Invalid {label}: {path}:{line_no} must be an object")
        rows.append(row)
    return rows


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the minimum manifest schema used by this harness."""
    required = [
        "schema_version",
        "run_id",
        "status",
        "timestamp",
        "harness",
        "runtime",
        "model",
        "suite",
        "settings",
        "output",
        "classification",
    ]
    _require_keys(manifest, required, "manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ResultArtifactError(
            f"Unsupported manifest schema_version: {manifest.get('schema_version')}"
        )
    if manifest["status"] not in {"completed", "stopped"}:
        raise ResultArtifactError("Run is incomplete: manifest status is neither 'completed' nor 'stopped'")
    if manifest["status"] == "stopped" and manifest.get("runtime", {}).get("kind") != "mlx":
        raise ResultArtifactError("Only MLX runs may use manifest status 'stopped'")
    _require_keys(manifest["suite"], ["suite_id", "suite_hash", "prompt_template_hash"], "manifest.suite")
    _require_keys(manifest["settings"], ["seed", "decoding", "prompt_order", "concurrency"], "manifest.settings")
    _require_keys(manifest["runtime"], ["kind"], "manifest.runtime")
    _require_keys(manifest["model"], ["repo_id", "revision"], "manifest.model")


def validate_result_row(row: dict[str, Any], index: int) -> None:
    """Validate one results.jsonl row."""
    required = [
        "schema_version",
        "run_id",
        "case_id",
        "case_index",
        "scored",
        "prompt_hash",
        "raw_response",
        "parse",
        "grade",
        "timing",
        "tokens",
        "resources",
        "errors",
    ]
    _require_keys(row, required, f"results.jsonl row {index}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ResultArtifactError(f"Unsupported row schema_version at row {index}")
    _require_keys(row["parse"], ["status", "is_truncated", "reasoning_text", "final_text"], f"row {index}.parse")
    _require_keys(row["grade"], ["passed", "score", "reason", "grader_type"], f"row {index}.grade")
    _require_keys(row["timing"], ["wall_seconds", "first_token_seconds", "decode_tokens_per_second"], f"row {index}.timing")
    _require_keys(row["tokens"], ["prompt", "generated"], f"row {index}.tokens")
    _require_keys(row["resources"], ["peak_mlx_memory_bytes", "swap_delta_bytes"], f"row {index}.resources")


def validate_summary(summary: dict[str, Any]) -> None:
    """Validate the minimum summary schema."""
    required = [
        "schema_version",
        "run_id",
        "status",
        "classification",
        "smoke_only",
        "totals",
        "pass_rate",
        "average_score",
    ]
    _require_keys(summary, required, "summary.json")
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ResultArtifactError(
            f"Unsupported summary schema_version: {summary.get('schema_version')}"
        )
    if summary["status"] not in {"completed", "stopped"}:
        raise ResultArtifactError("Run is incomplete: summary status is neither 'completed' nor 'stopped'")


def validate_artifact_set(manifest: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Validate manifest, result rows, summary, and internal consistency."""
    validate_manifest(manifest)
    for idx, row in enumerate(rows):
        validate_result_row(row, idx)
        if row["run_id"] != manifest["run_id"]:
            raise ResultArtifactError(f"results.jsonl row {idx} run_id mismatch")
    validate_summary(summary)
    if summary["run_id"] != manifest["run_id"]:
        raise ResultArtifactError("summary.json run_id mismatch")
    if summary["status"] != manifest["status"]:
        raise ResultArtifactError("summary.json status does not match manifest.json")
    scored_rows = [row for row in rows if row.get("scored")]
    totals = summary["totals"]
    passed = sum(1 for row in scored_rows if row["grade"]["passed"])
    expected_totals = {
        "rows": len(rows),
        "scored": len(scored_rows),
        "passed": passed,
        "failed": len(scored_rows) - passed,
        "skipped": 0,
        "parse_failures": sum(
            1 for row in scored_rows if row["parse"]["status"] != "success"
        ),
        "truncated": sum(
            1 for row in scored_rows if row["parse"]["is_truncated"]
        ),
        "resource_stops": sum(
            1
            for row in rows
            if any(
                error.get("type") == "resource_stop"
                for error in row.get("errors", [])
            )
        ),
        "errors": sum(1 for row in scored_rows if row.get("errors")),
    }
    for key, expected in expected_totals.items():
        if totals.get(key) != expected:
            raise ResultArtifactError(
                f"summary totals.{key} does not match results.jsonl"
            )
    expected_pass_rate = passed / len(scored_rows) if scored_rows else 0.0
    expected_average = (
        sum(float(row["grade"]["score"]) for row in scored_rows)
        / len(scored_rows)
        if scored_rows
        else 0.0
    )
    if abs(float(summary["pass_rate"]) - expected_pass_rate) > 1e-12:
        raise ResultArtifactError("summary pass_rate does not match results.jsonl")
    if abs(float(summary["average_score"]) - expected_average) > 1e-12:
        raise ResultArtifactError("summary average_score does not match results.jsonl")
    if summary["classification"] != manifest["classification"]:
        raise ResultArtifactError(
            "summary classification does not match manifest.json"
        )
    if bool(summary["smoke_only"]) != (manifest["classification"] == "smoke-only"):
        raise ResultArtifactError(
            "summary smoke_only does not match manifest classification"
        )
    if manifest.get("runtime", {}).get("kind") == "mlx":
        _validate_mlx_evidence(manifest, rows, summary)


def _validate_mlx_evidence(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    runtime = manifest["runtime"]
    if runtime.get("preflight_status") != "pass":
        raise ResultArtifactError("Real MLX artifacts require a passing recorded preflight")
    cache_status = runtime.get("cache_status", {}).get("status")
    if cache_status not in {"hit-complete", "miss"}:
        raise ResultArtifactError("Real MLX artifacts require a valid recorded cache status")
    model = manifest["model"]
    if len(str(model.get("revision", ""))) != 40 or int(model.get("size_bytes", 0) or 0) <= 0:
        raise ResultArtifactError("Real MLX artifacts require exact model revision and size")

    timing_keys = {
        "cold_load_seconds",
        "wall_seconds",
        "first_token_seconds",
        "decode_seconds",
        "decode_tokens_per_second",
        "prompt_tokens_per_second",
    }
    resource_keys = {
        "peak_mlx_memory_bytes",
        "disk_free_before_bytes",
        "disk_free_after_bytes",
        "memory_pressure_before",
        "memory_pressure_after",
        "swap_used_before_bytes",
        "swap_used_after_bytes",
        "swap_delta_bytes",
        "resource_stop_reason",
    }
    for index, row in enumerate(rows):
        missing_timing = sorted(timing_keys - set(row["timing"]))
        missing_resources = sorted(resource_keys - set(row["resources"]))
        if missing_timing:
            raise ResultArtifactError(
                f"Real MLX row {index} missing timing fields: {', '.join(missing_timing)}"
            )
        if missing_resources:
            raise ResultArtifactError(
                f"Real MLX row {index} missing resource fields: {', '.join(missing_resources)}"
            )
        has_evidence_error = any(
            error.get("type") in {"runtime_error", "missing_metrics"}
            for error in row.get("errors", [])
        )
        resources = row["resources"]
        if not has_evidence_error:
            if float(row["timing"]["wall_seconds"] or 0) <= 0:
                raise ResultArtifactError(f"Real MLX row {index} has invalid wall timing")
            if int(row["tokens"]["prompt"] or 0) <= 0 or int(row["tokens"]["generated"] or 0) <= 0:
                raise ResultArtifactError(f"Real MLX row {index} has invalid token counts")
            if float(row["timing"]["decode_tokens_per_second"] or 0) < 0:
                raise ResultArtifactError(f"Real MLX row {index} has invalid decode throughput")
            if int(resources["peak_mlx_memory_bytes"] or 0) <= 0:
                raise ResultArtifactError(f"Real MLX row {index} is missing peak MLX memory")
            if int(resources["disk_free_before_bytes"] or 0) <= 0 or int(resources["disk_free_after_bytes"] or 0) <= 0:
                raise ResultArtifactError(f"Real MLX row {index} is missing disk measurements")
            for key in (
                "memory_pressure_before",
                "memory_pressure_after",
                "swap_used_before_bytes",
                "swap_used_after_bytes",
                "swap_delta_bytes",
            ):
                if resources.get(key) is None:
                    raise ResultArtifactError(f"Real MLX row {index} is missing {key}")

        if "repeats" in manifest.get("settings", {}) and not has_evidence_error:
            _require_keys(
                row,
                ["response_sha256", "token_ids_sha256", "determinism"],
                f"Real MLX row {index}",
            )
            if len(str(row["response_sha256"])) != 64:
                raise ResultArtifactError(
                    f"Real MLX row {index} has invalid response_sha256"
                )
            if len(str(row["token_ids_sha256"])) != 64:
                raise ResultArtifactError(
                    f"Real MLX row {index} has invalid token_ids_sha256"
                )
            _require_keys(
                row["determinism"],
                [
                    "repeats",
                    "identical_token_ids",
                    "identical_text",
                    "per_repeat_hashes",
                ],
                f"Real MLX row {index}.determinism",
            )

    _require_keys(summary, ["performance", "resources"], "summary.json")
    _require_keys(
        summary["performance"],
        ["cold_load_seconds", "wall_seconds", "generated_tokens", "decode_tokens_per_second"],
        "summary.performance",
    )
    _require_keys(
        summary["resources"],
        [
            "peak_mlx_memory_bytes",
            "disk_free_before_bytes",
            "disk_free_after_bytes",
            "swap_delta_bytes",
            "memory_pressure_after",
        ],
        "summary.resources",
    )
    if "repeats" in manifest.get("settings", {}):
        _require_keys(
            summary,
            ["determinism"],
            "summary.json",
        )
        _require_keys(
            summary["determinism"],
            ["repeats", "identical_token_ids", "identical_text", "status", "cases"],
            "summary.determinism",
        )
        if summary["determinism"]["status"] not in {
            "identical",
            "divergent",
            "unavailable",
        }:
            raise ResultArtifactError("summary.determinism status is invalid")


def load_run_artifacts(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load and validate a completed run directory."""
    if not run_dir.exists() or not run_dir.is_dir():
        raise ResultArtifactError(f"Invalid run directory: {run_dir}")
    manifest = load_json(run_dir / "manifest.json", "manifest.json")
    rows = load_jsonl(run_dir / "results.jsonl", "results.jsonl")
    summary = load_json(run_dir / "summary.json", "summary.json")
    validate_artifact_set(manifest, rows, summary)
    return manifest, rows, summary


def summarize_rows(run_id: str, rows: list[dict[str, Any]], *, classification: str, limit: int | None) -> dict[str, Any]:
    """Build summary.json content from result rows."""
    scored_rows = [row for row in rows if row.get("scored")]
    passed = sum(1 for row in scored_rows if row["grade"]["passed"])
    failed = len(scored_rows) - passed
    parse_failures = sum(1 for row in scored_rows if row["parse"]["status"] != "success")
    truncated = sum(1 for row in scored_rows if row["parse"]["is_truncated"])
    errors = sum(1 for row in scored_rows if row.get("errors"))
    total_score = sum(float(row["grade"]["score"]) for row in scored_rows)
    scored = len(scored_rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "classification": classification,
        "smoke_only": classification == "smoke-only",
        "requested_limit": limit,
        "totals": {
            "rows": len(rows),
            "scored": scored,
            "passed": passed,
            "failed": failed,
            "skipped": 0,
            "parse_failures": parse_failures,
            "truncated": truncated,
            "resource_stops": 0,
            "errors": errors,
        },
        "pass_rate": (passed / scored) if scored else 0.0,
        "average_score": (total_score / scored) if scored else 0.0,
    }


def truncate_text(text: str, limit: int = 500) -> str:
    """Deterministically truncate user-facing text."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[TRUNCATED]"


def _require_keys(obj: Any, keys: list[str], label: str) -> None:
    if not isinstance(obj, dict):
        raise ResultArtifactError(f"{label} must be a JSON object")
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ResultArtifactError(f"{label} missing required keys: {', '.join(missing)}")
