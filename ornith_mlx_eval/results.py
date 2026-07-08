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
    if manifest["status"] != "completed":
        raise ResultArtifactError("Run is incomplete: manifest status is not 'completed'")
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
    if summary["status"] != "completed":
        raise ResultArtifactError("Run is incomplete: summary status is not 'completed'")


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
    scored_rows = [row for row in rows if row.get("scored")]
    totals = summary["totals"]
    if totals.get("scored") != len(scored_rows):
        raise ResultArtifactError("summary totals.scored does not match results.jsonl")
    if totals.get("passed") != sum(1 for row in scored_rows if row["grade"]["passed"]):
        raise ResultArtifactError("summary totals.passed does not match results.jsonl")


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
