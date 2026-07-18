"""Evaluation run orchestration.

The default runtime is a deterministic no-download mock path. Real MLX runs
use ``mlx_session`` while this module keeps runner and artifact behavior
runtime-agnostic.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ornith_mlx_eval import __version__
from ornith_mlx_eval.graders import grade
from ornith_mlx_eval.mlx_session import (
    MlxGenerationOptions,
    MlxSessionError,
    SIX_BIT_MODEL,
    SUPPORTED_MODEL_SHAS,
    generate_with_mlx,
    validate_6bit_promotion_source,
)
from ornith_mlx_eval.parsing import parse_response
from ornith_mlx_eval.profile import DISK_HEADROOM_BYTES, run_profile
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
    allow_download: bool = False
    promotion_source: str | None = None


def run_evaluation(options: RunOptions) -> Path:
    """Run an evaluation and return the completed run directory."""
    _validate_run_options(options)
    if options.runtime not in {"mock", "mlx"}:
        raise ResultArtifactError(f"Unsupported runtime: {options.runtime}")
    mlx_profile: dict[str, Any] | None = None
    if options.runtime == "mlx":
        _gate_mlx_run(options)

    suite, suite_path = _load_selected_suite(options.suite)
    errors = validate_suite(suite, suite_path=str(suite_path) if suite_path else "")
    if errors:
        raise ResultArtifactError("Suite validation failed: " + "; ".join(errors))

    selected_cases = list(suite.get("cases", []))
    if options.limit is not None:
        selected_cases = selected_cases[: options.limit]
    if not selected_cases:
        raise ResultArtifactError("Selected suite has no scored cases")

    if options.runtime == "mlx":
        mlx_profile = _preflight_mlx_run(options)

    output_root = _prepare_output_root(options.output_root)
    run_id = _new_run_id()
    run_dir = _create_run_dir(output_root, run_id)
    incomplete = run_dir / ".incomplete"
    incomplete.write_text("run in progress\n", encoding="utf-8")

    try:
        manifest = _build_manifest(
            options,
            suite,
            selected_cases,
            run_dir,
            run_id,
            mlx_profile=mlx_profile,
        )
        if options.runtime == "mock":
            rows = _run_mock_cases(manifest, selected_cases)
        else:
            rows = _run_mlx_cases(manifest, selected_cases, options)
            _refresh_mlx_artifact_identity(manifest)
        classification = manifest["classification"]
        summary = summarize_rows(run_id, rows, classification=classification, limit=options.limit)
        if options.runtime == "mlx":
            _add_mlx_summary_metrics(summary, rows)
            if summary["totals"]["resource_stops"]:
                manifest["status"] = "stopped"
                manifest["runtime"]["resource_status"] = "stopped"
                summary["status"] = "stopped"

        validate_artifact_set(manifest, rows, summary)
        write_json(run_dir / "manifest.json", manifest)
        write_jsonl(run_dir / "results.jsonl", rows)
        write_json(run_dir / "summary.json", summary)
        (run_dir / "report.md").write_text(render_report(manifest, rows, summary), encoding="utf-8")
        incomplete.unlink(missing_ok=True)
        if summary["status"] == "stopped":
            reasons = [
                error["message"]
                for row in rows
                for error in row.get("errors", [])
                if error.get("type") == "resource_stop"
            ]
            raise ResultArtifactError(
                f"MLX resource stop ({'; '.join(reasons)}); artifacts: {run_dir}"
            )
        return run_dir
    except Exception as exc:
        if not (run_dir / "manifest.json").exists():
            write_json(
                run_dir / "failure.json",
                {
                    "run_id": run_id,
                    "status": "failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
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
    *,
    mlx_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = 42 if options.seed is None else options.seed
    decoding = {
        "temperature": 0 if options.temperature is None else options.temperature,
        "top_p": 1 if options.top_p is None else options.top_p,
        "top_k": 0 if options.top_k is None else options.top_k,
        "max_tokens": 512 if options.max_tokens is None else options.max_tokens,
        "max_prompt_tokens": 8192 if options.max_prompt_tokens is None else options.max_prompt_tokens,
        "max_kv_size": 4096 if options.max_kv_size is None else options.max_kv_size,
        "enable_thinking": False,
    }
    model_id = options.model or "mock://ornith-mlx-eval"
    model_details = (mlx_profile or {}).get("model", {}).get("details", {})
    runtime: dict[str, Any] = {
        "kind": options.runtime,
        "resource_status": "not-measured" if options.runtime == "mock" else "measured",
    }
    if options.runtime == "mlx":
        runtime.update(
            {
                "preflight_status": (mlx_profile or {}).get("status", "missing"),
                "profile_checks": _compact_profile_checks(mlx_profile or {}),
                "cache_status": _cached_snapshot_status(model_id, SUPPORTED_MODEL_SHAS[model_id]),
            }
        )
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
        "runtime": runtime,
        "model": {
            "repo_id": model_id,
            "revision": _model_revision(options.runtime, model_id),
            "quantization": _model_quantization(options.runtime, model_id),
            "variant": _model_variant(options.runtime, model_id),
            "tokenizer_identity": "mock-tokenizer-v1" if options.runtime == "mock" else f"{model_id}:tokenizer",
            "chat_template_identity": "mock-chat-template-v1" if options.runtime == "mock" else f"{model_id}:chat-template",
            "size_bytes": int(model_details.get("size_bytes", 0) or 0),
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
            "host": platform.node(),
            "packages": _runtime_package_versions(),
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


def _run_mlx_cases(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    options: RunOptions,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    revision = manifest["model"]["revision"]
    model_id = manifest["model"]["repo_id"]
    gen_options = MlxGenerationOptions(
        max_tokens=manifest["settings"]["decoding"]["max_tokens"],
        temperature=manifest["settings"]["decoding"]["temperature"],
        top_p=manifest["settings"]["decoding"]["top_p"],
        top_k=manifest["settings"]["decoding"]["top_k"],
        seed=manifest["settings"]["seed"],
        max_prompt_tokens=manifest["settings"]["decoding"]["max_prompt_tokens"],
        max_kv_size=manifest["settings"]["decoding"]["max_kv_size"],
        enable_thinking=manifest["settings"]["decoding"]["enable_thinking"],
    )
    for index, case in enumerate(cases):
        chat_template_applied = False
        timing = {
            "cold_load_seconds": 0.0,
            "wall_seconds": 0.0,
            "first_token_seconds": 0.0,
            "decode_seconds": 0.0,
            "decode_tokens_per_second": 0.0,
            "prompt_tokens_per_second": 0.0,
        }
        resources: dict[str, Any] = {
            "peak_mlx_memory_bytes": 0,
            "disk_free_before_bytes": 0,
            "disk_free_after_bytes": 0,
            "memory_pressure_before": None,
            "memory_pressure_after": None,
            "swap_used_before_bytes": None,
            "swap_used_after_bytes": None,
            "swap_delta_bytes": None,
            "resource_stop_reason": None,
        }
        try:
            generation = generate_with_mlx(
                model_id,
                revision,
                render_prompt(case),
                gen_options,
            )
            raw_response = generation.raw_text
            prompt_tokens = generation.prompt_tokens
            generated_tokens = generation.generated_tokens
            peak_memory = generation.peak_mlx_memory_bytes
            chat_template_applied = generation.chat_template_applied
            case_errors: list[dict[str, str]] = []
            timing = {
                "cold_load_seconds": generation.cold_load_seconds,
                "wall_seconds": generation.wall_seconds,
                "first_token_seconds": generation.first_token_seconds,
                "decode_seconds": generation.decode_seconds,
                "decode_tokens_per_second": generation.decode_tokens_per_second,
                "prompt_tokens_per_second": generation.prompt_tokens_per_second,
            }
            swap_delta = None
            if (
                generation.swap_used_before_bytes is not None
                and generation.swap_used_after_bytes is not None
            ):
                swap_delta = (
                    generation.swap_used_after_bytes - generation.swap_used_before_bytes
                )
            resources = {
                "peak_mlx_memory_bytes": peak_memory,
                "disk_free_before_bytes": generation.disk_free_before_bytes,
                "disk_free_after_bytes": generation.disk_free_after_bytes,
                "memory_pressure_before": generation.memory_pressure_before,
                "memory_pressure_after": generation.memory_pressure_after,
                "swap_used_before_bytes": generation.swap_used_before_bytes,
                "swap_used_after_bytes": generation.swap_used_after_bytes,
                "swap_delta_bytes": swap_delta,
                "resource_stop_reason": None,
            }
            stop_reason = _resource_stop_reason(manifest, timing, resources)
            if stop_reason:
                resources["resource_stop_reason"] = stop_reason
                case_errors.append({"type": "resource_stop", "message": stop_reason})
        except MlxSessionError as exc:
            raw_response = ""
            prompt_tokens = len(render_prompt(case).split())
            generated_tokens = 0
            peak_memory = 0
            case_errors = [{"type": "runtime_error", "message": str(exc)}]

        parsed = parse_response(raw_response)
        grader = case.get("grader", {})
        expected = _expected_value(case)
        grade_result = grade(parsed, expected, grader.get("type"), _grader_options(grader))
        if not grade_result.passed:
            case_errors.append({"type": "case_failure", "message": grade_result.reason})

        rows.append({
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
                "final_text": parsed.final_text,
            },
            "grade": {
                "passed": bool(grade_result.passed),
                "score": float(grade_result.score),
                "reason": grade_result.reason,
                "grader_type": grade_result.grader_type,
                "evidence": grade_result.evidence,
            },
            "timing": timing,
            "tokens": {
                "prompt": prompt_tokens,
                "generated": generated_tokens,
                "chat_template_applied": chat_template_applied,
            },
            "resources": resources,
            "errors": case_errors,
        })
    if rows and all(
        any(error.get("type") == "runtime_error" for error in row.get("errors", []))
        for row in rows
    ):
        messages = [
            error["message"]
            for row in rows
            for error in row.get("errors", [])
            if error.get("type") == "runtime_error"
        ]
        raise ResultArtifactError(
            "MLX runtime failed for every selected case: " + "; ".join(messages)
        )
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


def _gate_mlx_run(options: RunOptions) -> None:
    if not options.model:
        raise ResultArtifactError("--runtime mlx requires --model")
    if "8bit" in options.model.lower() or "35b" in options.model.lower():
        raise ResultArtifactError(f"Unsupported MLX model: {options.model}")
    if options.model not in SUPPORTED_MODEL_SHAS:
        raise ResultArtifactError(f"Unsupported MLX model: {options.model}")
    if not options.allow_download or os.environ.get("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD") != "1":
        raise ResultArtifactError(
            "MLX runtime requires explicit opt-in: pass --allow-download and set "
            "ORNITH_MLX_ALLOW_MODEL_DOWNLOAD=1"
        )
    if options.model == SIX_BIT_MODEL and not options.promotion_source:
        raise ResultArtifactError(
            "6bit MLX runs require --promotion-source pointing to a fresh completed 4bit smoke manifest"
        )


def _preflight_mlx_run(options: RunOptions) -> dict[str, Any]:
    assert options.model is not None
    profile = run_profile(model_id=options.model, output_root=options.output_root)
    if profile.get("status") != "pass":
        failures = [
            f"{check.get('name', 'unknown')}: {check.get('reason', 'gate failed')}"
            for check in profile.get("checks", [])
            if check.get("status") == "fail"
        ]
        model = profile.get("model")
        if isinstance(model, dict) and model.get("status") == "fail":
            failures.append(f"model: {model.get('reason', 'gate failed')}")
        raise ResultArtifactError("MLX profile failed: " + "; ".join(failures))

    expected_sha = SUPPORTED_MODEL_SHAS[options.model]
    resolved_sha = profile.get("model", {}).get("details", {}).get("sha")
    if resolved_sha != expected_sha:
        raise ResultArtifactError(
            f"Resolved model SHA does not match pinned revision: {resolved_sha} != {expected_sha}"
        )

    memory_details = next(
        (
            check.get("details", {})
            for check in profile.get("checks", [])
            if check.get("name") == "memory"
        ),
        {},
    )
    pressure_match = re.search(
        r"(\d+)\s*$", str(memory_details.get("memory_pressure", ""))
    )
    if pressure_match is None:
        raise ResultArtifactError("MLX profile did not provide a memory-pressure measurement")
    pressure = int(pressure_match.group(1))
    if pressure <= 20:
        raise ResultArtifactError(
            f"MLX profile memory pressure is warning/critical before model load: {pressure}"
        )
    if not re.search(r"used\s*=\s*[0-9.]+[KMGTP]", str(memory_details.get("swap", ""))):
        raise ResultArtifactError("MLX profile did not provide a swap-usage measurement")

    cache = _cached_snapshot_status(options.model, expected_sha)
    if cache["status"] == "incomplete":
        raise ResultArtifactError(
            f"Cached model snapshot is incomplete at {cache['path']}: {cache['reason']}"
        )

    if options.model == SIX_BIT_MODEL:
        try:
            validate_6bit_promotion_source(options.promotion_source)
        except MlxSessionError as exc:
            raise ResultArtifactError(str(exc)) from exc
    return profile


def _compact_profile_checks(profile: dict[str, Any]) -> dict[str, Any]:
    keep: dict[str, set[str]] = {
        "python": {"version", "executable", "machine"},
        "mlx_packages": {"mlx_version", "mlx_lm_version"},
        "metal": {"metal_available", "gpu_name", "memory_size", "max_recommended_working_set_size"},
        "hf_cache": {"cache_path", "writable"},
        "output": {"output_root", "writable"},
        "disk": {"free_bytes", "model_size_bytes", "headroom_required_bytes", "required_bytes", "target"},
        "memory": {"physical_memory_bytes", "memory_pressure", "swap"},
    }
    compact: dict[str, Any] = {}
    for check in profile.get("checks", []):
        name = str(check.get("name", "unknown"))
        details = check.get("details", {})
        compact[name] = {
            "status": check.get("status"),
            "details": {
                key: value
                for key, value in details.items()
                if key in keep.get(name, set())
            },
        }
        if check.get("reason"):
            compact[name]["reason"] = check["reason"]
    return compact


def _cached_snapshot_status(model_id: str, revision: str) -> dict[str, Any]:
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    snapshot = (
        hf_home
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not snapshot.exists():
        return {"status": "miss", "path": str(snapshot)}
    required = [snapshot / "config.json", snapshot / "tokenizer.json"]
    weights = sorted(snapshot.glob("*.safetensors")) + sorted(snapshot.glob("*.npz"))
    missing = [str(path.name) for path in required if not path.exists()]
    broken = [str(path.name) for path in [*required, *weights] if path.is_symlink() and not path.resolve().exists()]
    empty = [str(path.name) for path in weights if path.exists() and path.stat().st_size <= 0]
    if not weights:
        missing.append("model weight files")
    problems = [*missing, *broken, *empty]
    if problems:
        return {
            "status": "incomplete",
            "path": str(snapshot),
            "reason": ", ".join(problems),
        }
    return {
        "status": "hit-complete",
        "path": str(snapshot),
        "weight_files": len(weights),
    }


def _refresh_mlx_artifact_identity(manifest: dict[str, Any]) -> None:
    model_id = str(manifest["model"]["repo_id"])
    revision = str(manifest["model"]["revision"])
    cache = _cached_snapshot_status(model_id, revision)
    if cache["status"] != "hit-complete":
        raise ResultArtifactError(
            f"Completed MLX generation did not leave a complete pinned snapshot: {cache}"
        )
    manifest["runtime"]["cache_status"] = cache
    snapshot = Path(cache["path"])
    tokenizer_path = snapshot / "tokenizer.json"
    chat_template_path = snapshot / "chat_template.jinja"
    manifest["model"]["tokenizer_identity"] = (
        f"tokenizer.json:sha256:{_sha256_file(tokenizer_path)}"
    )
    if chat_template_path.exists():
        manifest["model"]["chat_template_identity"] = (
            f"chat_template.jinja:sha256:{_sha256_file(chat_template_path)}"
        )
    else:
        manifest["model"]["chat_template_identity"] = "tokenizer-config-embedded"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resource_stop_reason(
    manifest: dict[str, Any],
    timing: dict[str, Any],
    resources: dict[str, Any],
) -> str | None:
    if int(resources.get("peak_mlx_memory_bytes") or 0) <= 0:
        return "Peak MLX memory was unavailable"
    if float(timing.get("wall_seconds") or 0) <= 0:
        return "Real MLX wall timing was unavailable"
    if int(resources.get("disk_free_after_bytes") or 0) <= 0:
        return "Post-run disk measurement was unavailable"
    required_disk = int(manifest["model"].get("size_bytes", 0)) + DISK_HEADROOM_BYTES
    if resources["disk_free_after_bytes"] < required_disk:
        return (
            "Post-run disk headroom fell below the model plus 12 GiB policy: "
            f"{resources['disk_free_after_bytes']} < {required_disk} bytes"
        )
    pressure = resources.get("memory_pressure_after")
    if pressure is None:
        return "Post-run memory pressure measurement was unavailable"
    if pressure <= 20:
        return f"Memory pressure reached warning/critical level: {pressure}"
    swap_delta = resources.get("swap_delta_bytes")
    if swap_delta is None:
        return "Swap growth measurement was unavailable"
    if swap_delta > 1024**3:
        return f"Swap growth exceeded 1 GiB: {swap_delta} bytes"
    return None


def _add_mlx_summary_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary["performance"] = {
        "cold_load_seconds": sum(float(row["timing"]["cold_load_seconds"]) for row in rows),
        "wall_seconds": sum(float(row["timing"]["wall_seconds"]) for row in rows),
        "generated_tokens": sum(int(row["tokens"]["generated"]) for row in rows),
        "decode_tokens_per_second": (
            sum(int(row["tokens"]["generated"]) for row in rows)
            / sum(float(row["timing"]["decode_seconds"]) for row in rows)
            if sum(float(row["timing"]["decode_seconds"]) for row in rows) > 0
            else 0.0
        ),
    }
    swap_deltas = [
        int(row["resources"]["swap_delta_bytes"])
        for row in rows
        if row["resources"].get("swap_delta_bytes") is not None
    ]
    memory_pressure_after = [
        int(row["resources"]["memory_pressure_after"])
        for row in rows
        if row["resources"].get("memory_pressure_after") is not None
    ]
    summary["resources"] = {
        "peak_mlx_memory_bytes": max(
            (int(row["resources"]["peak_mlx_memory_bytes"]) for row in rows),
            default=0,
        ),
        "disk_free_before_bytes": min(
            (int(row["resources"]["disk_free_before_bytes"]) for row in rows),
            default=0,
        ),
        "disk_free_after_bytes": min(
            (int(row["resources"]["disk_free_after_bytes"]) for row in rows),
            default=0,
        ),
        "swap_delta_bytes": max(swap_deltas, default=0),
        "memory_pressure_after": min(memory_pressure_after, default=None),
    }
    summary["totals"]["resource_stops"] = sum(
        1
        for row in rows
        if any(error.get("type") == "resource_stop" for error in row.get("errors", []))
    )


def _runtime_package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for package in ("mlx", "mlx-lm", "transformers", "huggingface-hub", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "missing"
    return packages


def _validate_run_options(options: RunOptions) -> None:
    if options.limit is not None and options.limit <= 0:
        raise ResultArtifactError("--limit must be a positive integer")
    if options.max_tokens is not None and options.max_tokens <= 0:
        raise ResultArtifactError("--max-tokens must be greater than zero")
    if options.max_prompt_tokens is not None and options.max_prompt_tokens <= 0:
        raise ResultArtifactError("--max-prompt-tokens must be greater than zero")
    if options.max_kv_size is not None and options.max_kv_size <= 0:
        raise ResultArtifactError("--max-kv-size must be greater than zero")
    if options.temperature is not None and options.temperature < 0:
        raise ResultArtifactError("--temperature must be zero or greater")
    if options.top_p is not None and not 0 < options.top_p <= 1:
        raise ResultArtifactError("--top-p must be greater than zero and at most one")
    if options.top_k is not None and options.top_k < 0:
        raise ResultArtifactError("--top-k must be zero or greater")


def _model_revision(runtime: str, model_id: str) -> str:
    if runtime == "mock":
        return MOCK_REVISION
    return SUPPORTED_MODEL_SHAS[model_id]


def _model_quantization(runtime: str, model_id: str) -> str:
    if runtime == "mock":
        return "mock"
    lower = model_id.lower()
    if "6bit" in lower:
        return "6bit"
    if "4bit" in lower:
        return "4bit"
    return "unknown"


def _model_variant(runtime: str, model_id: str) -> str:
    if runtime == "mock":
        return "mock"
    if "9b" in model_id.lower():
        return "9B"
    return "unknown"


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
