"""Evaluation run orchestration.

The default runtime is a deterministic no-download mock path.  Real MLX runs
are wired through ``mlx_session`` in the runtime milestone, but this module
keeps runner/artifact behavior runtime-agnostic.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ornith_mlx_eval import __version__
from ornith_mlx_eval.graders import grade
from ornith_mlx_eval.parsing import parse_response
from ornith_mlx_eval.reporting import render_report
from ornith_mlx_eval.results import (
    GRADER_VERSION,
    MOCK_REVISION,
    SCHEMA_VERSION,
    ResultArtifactError,
    summarize_rows,
    validate_artifact_set,
    write_json,
    write_jsonl,
)
from ornith_mlx_eval.suites import (
    DEFAULT_SUITES_DIR,
    SuiteValidationError,
    compute_case_prompt_hash,
    compute_prompt_template_hash,
    compute_suite_hash,
    discover_suites,
    load_suite,
    render_prompt,
    validate_suite,
)


@dataclass(frozen=True)
class RunOptions:
    """Options needed to execute an evaluation run."""

    runtime: str = "mock"
    suite: str | None = None
    model: str | None = None
    output_root: str = "benchmark_results"
    limit: int | None = None
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_prompt_tokens: int | None = None
    max_kv_size: int | None = None


def run_evaluation(options: RunOptions) -> Path:
    """Run an evaluation and return the completed run directory."""
    if options.limit is not None and options.limit <= 0:
        raise ResultArtifactError("--limit must be a positive integer")
    if options.runtime != "mock":
        raise ResultArtifactError("Only --runtime mock is available until the MLX runtime milestone")

    suite, suite_path = _load_selected_suite(options.suite)
    errors = validate_suite(suite, suite_path=str(suite_path) if suite_path else "")
    if errors:
        raise ResultArtifactError("Suite validation failed: " + "; ".join(errors))

    selected_cases = list(suite.get("cases", []))
    if options.limit is not None:
        selected_cases = selected_cases[: options.limit]
    if not selected_cases:
        raise ResultArtifactError("Selected suite has no scored cases")

    output_root = _prepare_output_root(options.output_root)
    run_id = _new_run_id()
    run_dir = _create_run_dir(output_root, run_id)
    incomplete = run_dir / ".incomplete"
    incomplete.write_text("run in progress\n", encoding="utf-8")

    try:
        manifest = _build_manifest(options, suite, selected_cases, run_dir, run_id)
        rows = _run_mock_cases(manifest, selected_cases)
        classification = manifest["classification"]
        summary = summarize_rows(run_id, rows, classification=classification, limit=options.limit)

        validate_artifact_set(manifest, rows, summary)
        write_json(run_dir / "manifest.json", manifest)
        write_jsonl(run_dir / "results.jsonl", rows)
        write_json(run_dir / "summary.json", summary)
        (run_dir / "report.md").write_text(render_report(manifest, rows, summary), encoding="utf-8")
        incomplete.unlink(missing_ok=True)
        return run_dir
    except Exception:
        incomplete.write_text("run failed before completion\n", encoding="utf-8")
        raise


def _load_selected_suite(selection: str | None) -> tuple[dict[str, Any], Path | None]:
    """Load suite by name, path, or ``all``."""
    if selection is None:
        selection = "smoke"

    candidate = Path(selection)
    if candidate.exists():
        return load_suite(candidate), candidate

    if selection == "all":
        suites = []
        paths = discover_suites()
        for path in paths:
            suite = load_suite(path)
            suites.append((suite, path))
        if not suites:
            raise ResultArtifactError("No suites are available for --suite all")
        merged_cases: list[dict[str, Any]] = []
        versions: list[str] = []
        for suite, _path in suites:
            errors = validate_suite(suite, suite_path=str(_path))
            if errors:
                raise ResultArtifactError("Suite validation failed: " + "; ".join(errors))
            versions.append(str(suite.get("suite_version", "unknown")))
            merged_cases.extend(list(suite.get("cases", [])))
        return {
            "suite_id": "all",
            "suite_version": "+".join(versions),
            "description": "All discoverable suites",
            "tags": ["all"],
            "cases": merged_cases,
        }, None

    path = DEFAULT_SUITES_DIR / f"{selection}.json"
    if path.exists():
        return load_suite(path), path

    raise ResultArtifactError(f"Unknown suite selection: {selection}")


def _prepare_output_root(output_root: str) -> Path:
    root = Path(output_root)
    if root.exists() and root.is_symlink():
        raise ResultArtifactError(f"Output root must not be a symlink: {root}")
    if root.exists() and not root.is_dir():
        raise ResultArtifactError(f"Output root exists but is not a directory: {root}")
    parent = root.parent if root.parent != Path("") else Path(".")
    if not parent.exists():
        raise ResultArtifactError(f"Output root parent does not exist: {parent}")
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve()
    if resolved.is_symlink():
        raise ResultArtifactError(f"Output root must not resolve to a symlink: {root}")
    test_file = resolved / ".ornith_write_test"
    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        raise ResultArtifactError(f"Output root is not writable: {resolved}: {exc}")
    return resolved


def _create_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    try:
        run_dir.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError:
        raise ResultArtifactError(f"Run directory collision: {run_dir}")
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError:
        raise ResultArtifactError(f"Run directory escaped output root: {run_dir}")
    return resolved


def _new_run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{now}_{uuid.uuid4().hex[:8]}"


def _build_manifest(
    options: RunOptions,
    suite: dict[str, Any],
    selected_cases: list[dict[str, Any]],
    run_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    seed = 42 if options.seed is None else options.seed
    decoding = {
        "temperature": 0 if options.temperature is None else options.temperature,
        "top_p": 1 if options.top_p is None else options.top_p,
        "top_k": 0 if options.top_k is None else options.top_k,
        "max_tokens": 512 if options.max_tokens is None else options.max_tokens,
        "max_prompt_tokens": 8192 if options.max_prompt_tokens is None else options.max_prompt_tokens,
        "max_kv_size": 4096 if options.max_kv_size is None else options.max_kv_size,
    }
    model_id = options.model or "mock://ornith-mlx-eval"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": {
            "runtime": options.runtime,
            "suite": options.suite or "smoke",
            "model": options.model,
            "output_root": options.output_root,
        },
        "cwd": os.getcwd(),
        "harness": {
            "name": "ornith-mlx-eval",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "grader_version": GRADER_VERSION,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        },
        "runtime": {
            "kind": "mock",
            "resource_status": "not-measured",
        },
        "model": {
            "repo_id": model_id,
            "revision": MOCK_REVISION,
            "quantization": "mock",
            "variant": "mock",
            "tokenizer_identity": "mock-tokenizer-v1",
            "chat_template_identity": "mock-chat-template-v1",
        },
        "suite": {
            "suite_id": suite.get("suite_id", "unknown"),
            "suite_version": suite.get("suite_version", "unknown"),
            "suite_hash": compute_suite_hash(suite),
            "prompt_template_hash": compute_prompt_template_hash(suite),
            "case_count": len(selected_cases),
        },
        "settings": {
            "seed": seed,
            "decoding": decoding,
            "limit": options.limit,
            "prompt_order": [str(case.get("case_id", "")) for case in selected_cases],
            "concurrency": 1,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "output": {
            "run_dir": str(run_dir),
            "artifact_files": ["manifest.json", "results.jsonl", "summary.json", "report.md"],
        },
        "classification": "smoke-only" if options.limit is not None else "benchmark",
    }


def _run_mock_cases(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        raw_response = _mock_response(case)
        parsed = parse_response(raw_response)
        grader = case.get("grader", {})
        expected = _expected_value(case)
        options = _grader_options(grader)
        grade_result = grade(parsed, expected, grader.get("type"), options)
        final_text = parsed.final_text
        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "case_id": str(case.get("case_id", f"case-{index}")),
            "case_index": index,
            "scored": True,
            "category": "smoke",
            "prompt_hash": compute_case_prompt_hash(case),
            "prompt_chars": len(render_prompt(case)),
            "raw_response": raw_response,
            "parse": {
                "status": parsed.parse_status,
                "is_truncated": parsed.is_truncated,
                "reasoning_text": parsed.reasoning_text,
                "final_text": final_text,
            },
            "grade": {
                "passed": bool(grade_result.passed),
                "score": float(grade_result.score),
                "reason": grade_result.reason,
                "grader_type": grade_result.grader_type,
                "evidence": grade_result.evidence,
            },
            "timing": {
                "wall_seconds": 0.0,
                "first_token_seconds": 0.0,
                "decode_tokens_per_second": 0.0,
            },
            "tokens": {
                "prompt": len(render_prompt(case).split()),
                "generated": len(raw_response.split()),
            },
            "resources": {
                "peak_mlx_memory_bytes": 0,
                "memory_pressure": "not-measured",
                "swap_delta_bytes": 0,
            },
            "errors": [] if grade_result.passed else [{"type": "case_failure", "message": grade_result.reason}],
        }
        rows.append(row)
    return rows


def _mock_response(case: dict[str, Any]) -> str:
    grader = case.get("grader", {})
    gtype = grader.get("type")
    hidden = str(case.get("expected_answer", {}).get("hidden_answer", ""))
    if gtype == "code":
        expected = _normalise_option(grader.get("options", {}).get("expected_output", ""))
        return f"```python\nprint({expected!r})\n```"
    if gtype == "json_match":
        return hidden
    return hidden


def _expected_value(case: dict[str, Any]) -> Any:
    hidden = case.get("expected_answer", {}).get("hidden_answer", "")
    gtype = case.get("grader", {}).get("type")
    if gtype == "json_match":
        import json

        try:
            return json.loads(hidden)
        except (TypeError, ValueError):
            return hidden
    return hidden


def _grader_options(grader: dict[str, Any]) -> dict[str, Any]:
    options = dict(grader.get("options", {}) or {})
    if grader.get("type") == "code":
        for key in ("test_input", "expected_output"):
            if key in options:
                options[key] = _normalise_option(options[key])
    return options


def _normalise_option(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False
