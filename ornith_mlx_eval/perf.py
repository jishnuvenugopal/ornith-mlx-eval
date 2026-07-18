"""Warmed, fixed-length MLX throughput probe.

The probe shares smoke's model/download/profile gates, loads one pinned model,
discards one warmup generation, and reports direct first-to-last-token decode
timing across repeated trials. Tests inject the MLX API and never download
weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ornith_mlx_eval import __version__
from ornith_mlx_eval.mlx_session import (
    MlxGenerationOptions,
    MlxSessionError,
    SUPPORTED_MODEL_SHAS,
    open_mlx_session,
)
from ornith_mlx_eval.results import ResultArtifactError, write_json
from ornith_mlx_eval.runner import (
    RunOptions,
    _cached_snapshot_status,
    _compact_profile_checks,
    _create_run_dir,
    _gate_mlx_run,
    _git_commit,
    _git_dirty,
    _model_quantization,
    _model_variant,
    _new_run_id,
    _preflight_mlx_run,
    _prepare_output_root,
    _profile_check_details,
    _resource_stop_reason,
    _runtime_package_versions,
    _sha256_file,
)


PERF_SCHEMA_VERSION = "ornith-perf-v1"
PERF_PROMPT = (
    "Continue producing plain explanatory prose about local model evaluation "
    "until the generation limit is reached. Do not stop early."
)


@dataclass(frozen=True)
class PerfOptions:
    """CLI and runner settings for one throughput-probe directory."""

    model: str
    output_root: str = "benchmark_results"
    trials: int = 3
    decode_tokens: int = 128
    warmup_tokens: int = 16
    seed: int = 42
    max_prompt_tokens: int = 8192
    max_kv_size: int = 4096
    allow_download: bool = False
    promotion_source: str | None = None


def run_perf_probe(
    options: PerfOptions,
    *,
    api: Any | None = None,
    clock: Any = time.perf_counter,
    resource_probe: Any | None = None,
) -> Path:
    """Run a warmed multi-trial probe and return its persisted directory."""
    _validate_perf_options(options)
    run_options = RunOptions(
        runtime="mlx",
        suite="perf",
        model=options.model,
        output_root=options.output_root,
        seed=options.seed,
        temperature=0,
        top_p=1,
        top_k=0,
        max_tokens=options.decode_tokens,
        max_prompt_tokens=options.max_prompt_tokens,
        max_kv_size=options.max_kv_size,
        allow_download=options.allow_download,
        promotion_source=options.promotion_source,
    )
    _gate_mlx_run(run_options)
    profile = _preflight_mlx_run(run_options)
    output_root = _prepare_output_root(options.output_root)
    run_id = _new_run_id().replace("run_", "perf_", 1)
    run_dir = _create_run_dir(output_root, run_id)
    incomplete = run_dir / ".incomplete"
    incomplete.write_text("perf probe in progress\n", encoding="utf-8")

    manifest = _build_perf_manifest(options, profile, run_id, run_dir)
    try:
        max_working_set = int(
            _profile_check_details(profile, "metal").get(
                "max_recommended_working_set_size", 0
            )
            or 0
        )
        if max_working_set <= 0:
            raise ResultArtifactError(
                "MLX profile did not provide a working-set limit"
            )
        memory_limit_bytes = int(max_working_set * 0.85)
        manifest["runtime"]["memory_limit_bytes"] = memory_limit_bytes

        generation_options = MlxGenerationOptions(
            max_tokens=options.decode_tokens,
            temperature=0,
            top_p=1,
            top_k=0,
            seed=options.seed,
            max_prompt_tokens=options.max_prompt_tokens,
            max_kv_size=options.max_kv_size,
            enable_thinking=False,
        )
        open_kwargs: dict[str, Any] = {
            "api": api,
            "clock": clock,
            "memory_limit_bytes": memory_limit_bytes,
        }
        if resource_probe is not None:
            open_kwargs["resource_probe"] = resource_probe

        with open_mlx_session(
            options.model,
            SUPPORTED_MODEL_SHAS[options.model],
            generation_options,
            **open_kwargs,
        ) as session:
            eos_suppressor = _make_eos_suppressor(session.tokenizer)
            warmup = session.generate(
                PERF_PROMPT,
                replace(generation_options, max_tokens=options.warmup_tokens),
                logits_processors=[eos_suppressor],
            )
            if warmup.generated_tokens != options.warmup_tokens:
                raise MlxSessionError(
                    "Perf warmup did not reach its fixed token budget: "
                    f"{warmup.generated_tokens} != {options.warmup_tokens}"
                )

            trial_results = []
            for _ in range(options.trials):
                result = session.generate(
                    PERF_PROMPT,
                    generation_options,
                    logits_processors=[eos_suppressor],
                )
                if result.generated_tokens != options.decode_tokens:
                    raise MlxSessionError(
                        "Perf trial did not reach its fixed token budget: "
                        f"{result.generated_tokens} != {options.decode_tokens}"
                    )
                if result.decode_tokens_per_second <= 0:
                    raise MlxSessionError(
                        "Perf trial did not produce a positive measured decode rate"
                    )
                trial_results.append(result)

        _refresh_perf_identity(manifest)
        trial_tps = [result.decode_tokens_per_second for result in trial_results]
        aggregate = aggregate_perf_trials(
            trial_tps,
            trials=options.trials,
            decode_tokens=options.decode_tokens,
        )
        resources = _aggregate_resources(warmup, trial_results)
        perf_data = {
            "schema_version": PERF_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "completed",
            "warmup": {
                "generated_tokens": warmup.generated_tokens,
                "measured_tps": warmup.decode_tokens_per_second,
                "runtime_reported_tps": warmup.runtime_reported_tps,
            },
            "trials": [
                {
                    "trial": index + 1,
                    "generated_tokens": result.generated_tokens,
                    "decode_seconds": result.decode_seconds,
                    "measured_tps": result.decode_tokens_per_second,
                    "runtime_reported_tps": result.runtime_reported_tps,
                    "prompt_tokens_per_second": result.prompt_tokens_per_second,
                    "peak_mlx_memory_bytes": result.peak_mlx_memory_bytes,
                }
                for index, result in enumerate(trial_results)
            ],
            "aggregate": aggregate["aggregate"],
            "headline": aggregate["headline"],
            "resources": resources,
        }
        stop_reason = _resource_stop_reason(
            manifest,
            {
                "wall_seconds": sum(
                    result.wall_seconds for result in [warmup, *trial_results]
                )
            },
            resources,
        )
        if stop_reason:
            manifest["status"] = "stopped"
            manifest["runtime"]["resource_status"] = "stopped"
            perf_data["status"] = "stopped"
            perf_data["resources"]["resource_stop_reason"] = stop_reason

        _validate_perf_artifact(manifest, perf_data)
        write_json(run_dir / "manifest.json", manifest)
        write_json(run_dir / "perf.json", perf_data)
        (run_dir / "report.md").write_text(
            render_perf_report(manifest, perf_data), encoding="utf-8"
        )
        incomplete.unlink(missing_ok=True)
        if stop_reason:
            raise ResultArtifactError(
                f"MLX resource stop ({stop_reason}); artifacts: {run_dir}"
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
            incomplete.write_text("perf probe failed before completion\n", encoding="utf-8")
        raise


def aggregate_perf_trials(
    trial_tps: list[float], *, trials: int, decode_tokens: int
) -> dict[str, Any]:
    """Aggregate trial rates and decide whether a headline is publishable."""
    if not trial_tps or any(not math.isfinite(value) or value <= 0 for value in trial_tps):
        raise ResultArtifactError("Perf trials require finite positive decode rates")
    median_tps = statistics.median(trial_tps)
    mean_tps = statistics.fmean(trial_tps)
    cv = statistics.pstdev(trial_tps) / mean_tps if mean_tps > 0 else math.inf
    reasons: list[str] = []
    if trials < 3:
        reasons.append("at least 3 trials")
    if decode_tokens < 64:
        reasons.append("at least 64 decode tokens")
    if cv > 0.10:
        reasons.append("CV exceeds 10%")
    publishable = not reasons
    return {
        "aggregate": {
            "median_tps": median_tps,
            "min_tps": min(trial_tps),
            "max_tps": max(trial_tps),
            "cv": cv,
        },
        "headline": {
            "status": "publishable" if publishable else "unstable",
            "decode_tokens_per_second": median_tps if publishable else None,
            "reasons": reasons,
        },
    }


def load_perf_artifact(run_dir: str | Path) -> dict[str, Any]:
    """Load the persisted perf payload used by the CLI and later publication."""
    path = Path(run_dir) / "perf.json"
    if not path.exists():
        raise ResultArtifactError(f"Missing required perf.json: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultArtifactError(f"Corrupt perf.json: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ResultArtifactError(f"Invalid perf.json: {path} must contain an object")
    if data.get("schema_version") != PERF_SCHEMA_VERSION:
        raise ResultArtifactError("Unsupported perf.json schema_version")
    return data


def render_perf_report(manifest: dict[str, Any], perf: dict[str, Any]) -> str:
    """Render the human-readable throughput section from persisted values."""
    aggregate = perf["aggregate"]
    headline = perf["headline"]
    lines = [
        "# Ornith MLX Eval Performance Probe",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Model: `{manifest['model']['repo_id']}`",
        f"Revision: `{manifest['model']['revision']}`",
        "",
        "## Throughput Probe",
        "",
        f"- Warmup tokens (discarded): `{manifest['settings']['warmup_tokens']}`",
        f"- Trials: `{manifest['settings']['trials']}`",
        f"- Decode tokens per trial: `{manifest['settings']['decode_tokens']}`",
        "",
        "| Trial | Measured TPS | Runtime-reported TPS |",
        "|---:|---:|---:|",
    ]
    for trial in perf["trials"]:
        lines.append(
            f"| {trial['trial']} | {trial['measured_tps']:.6f} | "
            f"{trial['runtime_reported_tps']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Median measured TPS: `{aggregate['median_tps']:.6f}`",
            f"- Minimum measured TPS: `{aggregate['min_tps']:.6f}`",
            f"- Maximum measured TPS: `{aggregate['max_tps']:.6f}`",
            f"- CV: `{aggregate['cv']:.6f}`",
        ]
    )
    if headline["status"] == "publishable":
        lines.append(
            "- Headline decode tokens/second: "
            f"`{headline['decode_tokens_per_second']:.6f}`"
        )
    else:
        lines.append(
            "- Headline: `unstable` (" + "; ".join(headline["reasons"]) + ")"
        )
    resources = perf["resources"]
    lines.extend(
        [
            "",
            "## Resources",
            "",
            f"- Peak MLX memory bytes: `{resources['peak_mlx_memory_bytes']}`",
            f"- Memory percent-free after: `{resources['memory_pressure_after']}`",
            f"- Swap delta bytes: `{resources['swap_delta_bytes']}`",
            "",
            "## Caveats",
            "",
            "- Throughput is local to the pinned model, runtime stack, and host.",
            "- Runtime-reported TPS is persisted only as a cross-check; the headline uses direct chunk timing.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_perf_options(options: PerfOptions) -> None:
    if options.trials <= 0:
        raise ResultArtifactError("--trials must be greater than zero")
    if options.decode_tokens <= 1:
        raise ResultArtifactError("--decode-tokens must be greater than one")
    if options.warmup_tokens <= 1:
        raise ResultArtifactError("--warmup-tokens must be greater than one")
    if options.max_prompt_tokens <= 0:
        raise ResultArtifactError("--max-prompt-tokens must be greater than zero")
    if options.max_kv_size <= 0:
        raise ResultArtifactError("--max-kv-size must be greater than zero")


def _make_eos_suppressor(tokenizer: Any):
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        eos_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    if not isinstance(eos_ids, (list, tuple)) or not eos_ids:
        raise MlxSessionError("Perf requires tokenizer EOS token ids for suppression")
    normalized = [int(token) for token in eos_ids]

    def suppress_eos(tokens, logits):
        return logits.at[:, normalized].set(-float("inf"))

    return suppress_eos


def _aggregate_resources(warmup, trials) -> dict[str, Any]:
    all_results = [warmup, *trials]
    first = all_results[0]
    last = all_results[-1]
    swap_delta = None
    if first.swap_used_before_bytes is not None and last.swap_used_after_bytes is not None:
        swap_delta = last.swap_used_after_bytes - first.swap_used_before_bytes
    return {
        "peak_mlx_memory_bytes": max(
            result.peak_mlx_memory_bytes for result in all_results
        ),
        "disk_free_before_bytes": first.disk_free_before_bytes,
        "disk_free_after_bytes": last.disk_free_after_bytes,
        "memory_pressure_before": first.memory_pressure_before,
        "memory_pressure_after": last.memory_pressure_after,
        "swap_used_before_bytes": first.swap_used_before_bytes,
        "swap_used_after_bytes": last.swap_used_after_bytes,
        "swap_delta_bytes": swap_delta,
        "resource_stop_reason": None,
    }


def _build_perf_manifest(
    options: PerfOptions,
    profile: dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    model_details = profile.get("model", {}).get("details", {})
    prompt_hash = hashlib.sha256(PERF_PROMPT.encode("utf-8")).hexdigest()
    return {
        "schema_version": PERF_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": {"name": "perf", "model": options.model},
        "cwd": os.getcwd(),
        "harness": {
            "name": "ornith-mlx-eval",
            "version": __version__,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        },
        "runtime": {
            "kind": "mlx",
            "preflight_status": profile.get("status"),
            "resource_status": "measured",
            "profile_checks": _compact_profile_checks(profile),
            "cache_status": _cached_snapshot_status(
                options.model, SUPPORTED_MODEL_SHAS[options.model]
            ),
        },
        "model": {
            "repo_id": options.model,
            "revision": SUPPORTED_MODEL_SHAS[options.model],
            "quantization": _model_quantization("mlx", options.model),
            "variant": _model_variant("mlx", options.model),
            "size_bytes": int(model_details.get("size_bytes", 0) or 0),
            "tokenizer_identity": f"{options.model}:tokenizer",
            "chat_template_identity": f"{options.model}:chat-template",
        },
        "settings": {
            "seed": options.seed,
            "trials": options.trials,
            "decode_tokens": options.decode_tokens,
            "warmup_tokens": options.warmup_tokens,
            "max_prompt_tokens": options.max_prompt_tokens,
            "max_kv_size": options.max_kv_size,
            "temperature": 0,
            "eos_suppressed": True,
            "prompt_sha256": prompt_hash,
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
            "artifact_files": ["manifest.json", "perf.json", "report.md"],
        },
    }


def _refresh_perf_identity(manifest: dict[str, Any]) -> None:
    model_id = manifest["model"]["repo_id"]
    revision = manifest["model"]["revision"]
    cache = _cached_snapshot_status(model_id, revision)
    if cache.get("status") != "hit-complete":
        raise ResultArtifactError(
            f"Completed perf generation did not leave a complete pinned snapshot: {cache}"
        )
    manifest["runtime"]["cache_status"] = cache
    snapshot = Path(cache["path"])
    manifest["model"]["tokenizer_identity"] = (
        f"tokenizer.json:sha256:{_sha256_file(snapshot / 'tokenizer.json')}"
    )
    template = snapshot / "chat_template.jinja"
    manifest["model"]["chat_template_identity"] = (
        f"chat_template.jinja:sha256:{_sha256_file(template)}"
        if template.exists()
        else "tokenizer-config-embedded"
    )


def _validate_perf_artifact(
    manifest: dict[str, Any], perf: dict[str, Any]
) -> None:
    if manifest.get("schema_version") != PERF_SCHEMA_VERSION:
        raise ResultArtifactError("Invalid perf manifest schema_version")
    if manifest.get("status") not in {"completed", "stopped"}:
        raise ResultArtifactError("Invalid perf manifest status")
    if perf.get("schema_version") != PERF_SCHEMA_VERSION:
        raise ResultArtifactError("Invalid perf.json schema_version")
    if perf.get("run_id") != manifest.get("run_id"):
        raise ResultArtifactError("perf.json run_id mismatch")
    if perf.get("status") != manifest.get("status"):
        raise ResultArtifactError("perf.json status mismatch")
    if len(perf.get("trials", [])) != int(manifest["settings"]["trials"]):
        raise ResultArtifactError("perf.json trial count mismatch")
    if any(
        int(trial.get("generated_tokens", 0))
        != int(manifest["settings"]["decode_tokens"])
        for trial in perf["trials"]
    ):
        raise ResultArtifactError("perf.json fixed decode-token count mismatch")
