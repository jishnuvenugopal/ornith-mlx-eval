"""Direct MLX/MLX-LM runtime integration.

The runtime path uses verified MLX APIs only and never falls back to Ollama or
HTTP-compatible backends.  Tests can inject fake dependencies through the
``api`` parameter to avoid model downloads.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ornith_mlx_eval import __version__
from ornith_mlx_eval.results import ResultArtifactError, load_run_artifacts


FOUR_BIT_MODEL = "mlx-community/Ornith-1.0-9B-4bit"
SIX_BIT_MODEL = "mlx-community/Ornith-1.0-9B-6bit"
FOUR_BIT_SHA = "1e980b9742a9e554a4d57e90b4c597811fb2fc4e"
SIX_BIT_SHA = "a2800933352a607ffbb1f814295fc3ff8e10ad69"

SUPPORTED_MODEL_SHAS: dict[str, str] = {
    FOUR_BIT_MODEL: FOUR_BIT_SHA,
    SIX_BIT_MODEL: SIX_BIT_SHA,
}


class MlxSessionError(RuntimeError):
    """Raised when MLX runtime setup or generation fails."""


@dataclass(frozen=True)
class MlxGenerationOptions:
    """Generation settings translated to MLX-LM APIs."""

    max_tokens: int = 512
    temperature: float = 0
    top_p: float = 1
    top_k: int = 0
    seed: int = 42
    max_prompt_tokens: int = 8192
    max_kv_size: int = 4096
    enable_thinking: bool = False


@dataclass(frozen=True)
class MlxGenerationResult:
    """Result of one MLX generation call."""

    raw_text: str
    prompt_tokens: int
    generated_tokens: int
    peak_mlx_memory_bytes: int
    model_id: str
    revision: str
    max_kv_size: int
    cold_load_seconds: float = 0.0
    first_token_seconds: float = 0.0
    decode_seconds: float = 0.0
    decode_tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    wall_seconds: float = 0.0
    disk_free_before_bytes: int = 0
    disk_free_after_bytes: int = 0
    memory_pressure_before: int | None = None
    memory_pressure_after: int | None = None
    swap_used_before_bytes: int | None = None
    swap_used_after_bytes: int | None = None
    chat_template_applied: bool = False


def generate_with_mlx(
    model_id: str,
    revision_sha: str,
    prompt: str,
    options: MlxGenerationOptions,
    *,
    api: Any | None = None,
    clock: Any = time.perf_counter,
    resource_probe: Any | None = None,
) -> MlxGenerationResult:
    """Generate text with MLX-LM using exact revision and verified APIs.

    The function intentionally accepts an injected ``api`` object for tests.
    Real execution imports MLX/MLX-LM lazily only after prompt gates pass.
    """
    estimated_prompt_tokens = _count_prompt_tokens(prompt)
    if estimated_prompt_tokens > options.max_prompt_tokens:
        raise MlxSessionError(
            "Rendered prompt exceeds prompt token limit: "
            f"{estimated_prompt_tokens} > {options.max_prompt_tokens}"
        )
    if not revision_sha or len(revision_sha) != 40:
        raise MlxSessionError("MLX runtime requires an exact 40-character revision SHA")
    if options.max_tokens <= 0:
        raise MlxSessionError("MLX runtime requires max_tokens greater than zero")
    if options.max_kv_size <= 0:
        raise MlxSessionError("MLX runtime requires max_kv_size greater than zero")

    api = api or _load_real_api()
    probe = resource_probe or capture_resource_snapshot
    before = probe()
    wall_started = clock()
    stage = "MLX setup"
    chunks: list[str] = []
    prompt_tokens = estimated_prompt_tokens
    generated_tokens = 0
    prompt_tps = 0.0
    generation_tps = 0.0
    response_peak_memory_bytes = 0
    cold_load_seconds = 0.0
    first_token_seconds = 0.0
    generation_started = 0.0
    generation_elapsed = 0.0
    try:
        stage = "memory measurement reset"
        api.mx.reset_peak_memory()
        stage = "random seed setup"
        api.mx.random.seed(options.seed)

        stage = "model load"
        load_started = clock()
        model, tokenizer = api.load(model_id, revision=revision_sha)
        cold_load_seconds = max(0.0, clock() - load_started)

        generation_prompt, chat_template_applied = _prepare_generation_prompt(
            tokenizer,
            prompt,
            enable_thinking=options.enable_thinking,
        )
        actual_prompt_tokens = (
            len(generation_prompt)
            if isinstance(generation_prompt, (list, tuple))
            else _tokenizer_prompt_tokens(tokenizer, str(generation_prompt))
        )
        if actual_prompt_tokens is not None:
            prompt_tokens = actual_prompt_tokens
        if prompt_tokens > options.max_prompt_tokens:
            raise MlxSessionError(
                "Rendered prompt exceeds prompt token limit after tokenization: "
                f"{prompt_tokens} > {options.max_prompt_tokens}"
            )

        stage = "sampler setup"
        sampler = api.make_sampler(
            temp=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
        )
        stage = "prompt cache setup"
        prompt_cache = api.make_prompt_cache(model, max_kv_size=options.max_kv_size)

        stage = "generation"
        generation_started = clock()
        for chunk in api.stream_generate(
            model,
            tokenizer,
            generation_prompt,
            max_tokens=options.max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            if not chunks:
                first_token_seconds = max(0.0, clock() - generation_started)
            chunks.append(_chunk_text(chunk))
            prompt_tokens = _positive_int_metric(chunk, "prompt_tokens", prompt_tokens)
            generated_tokens = _positive_int_metric(
                chunk, "generation_tokens", generated_tokens
            )
            prompt_tps = _positive_float_metric(chunk, "prompt_tps", prompt_tps)
            generation_tps = _positive_float_metric(
                chunk, "generation_tps", generation_tps
            )
            chunk_peak_gb = _positive_float_metric(chunk, "peak_memory", 0.0)
            response_peak_memory_bytes = max(
                response_peak_memory_bytes, int(chunk_peak_gb * 1_000_000_000)
            )
        generation_elapsed = max(0.0, clock() - generation_started)
        stage = "peak memory read"
        peak_memory = max(int(api.mx.get_peak_memory()), response_peak_memory_bytes)
    except MlxSessionError:
        raise
    except Exception as exc:
        raise MlxSessionError(f"{stage} failed: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            api.mx.clear_cache()
        except Exception:
            pass

    after = probe()
    raw_text = "".join(chunks)
    if generated_tokens <= 0 and raw_text:
        generated_tokens = len(raw_text.split())
    if generation_tps <= 0 and generated_tokens > 0 and generation_elapsed > 0:
        generation_tps = generated_tokens / generation_elapsed
    decode_seconds = (
        generated_tokens / generation_tps
        if generated_tokens > 0 and generation_tps > 0
        else generation_elapsed
    )
    wall_seconds = max(0.0, clock() - wall_started)
    return MlxGenerationResult(
        raw_text=raw_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        peak_mlx_memory_bytes=peak_memory,
        model_id=model_id,
        revision=revision_sha,
        max_kv_size=options.max_kv_size,
        cold_load_seconds=cold_load_seconds,
        first_token_seconds=first_token_seconds,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=generation_tps,
        prompt_tokens_per_second=prompt_tps,
        wall_seconds=wall_seconds,
        disk_free_before_bytes=int(before.get("disk_free_bytes", 0) or 0),
        disk_free_after_bytes=int(after.get("disk_free_bytes", 0) or 0),
        memory_pressure_before=before.get("memory_pressure"),
        memory_pressure_after=after.get("memory_pressure"),
        swap_used_before_bytes=before.get("swap_used_bytes"),
        swap_used_after_bytes=after.get("swap_used_bytes"),
        chat_template_applied=chat_template_applied,
    )


def validate_6bit_promotion_source(promotion_source: str | None) -> None:
    """Require a fresh, same-host, fully measured successful 4-bit smoke."""
    if not promotion_source:
        raise MlxSessionError("6bit promotion requires --promotion-source pointing to a fresh 4bit smoke manifest")
    manifest_path = Path(promotion_source)
    try:
        manifest, rows, summary = load_run_artifacts(manifest_path.parent)
    except ResultArtifactError as exc:
        raise MlxSessionError(f"Invalid 4bit promotion source: {exc}") from exc
    model = manifest.get("model", {})
    if model.get("repo_id") != FOUR_BIT_MODEL or model.get("revision") != FOUR_BIT_SHA:
        raise MlxSessionError("6bit promotion source must be a completed 4bit smoke artifact")
    if manifest.get("status") != "completed" or summary.get("status") != "completed":
        raise MlxSessionError("6bit promotion source must be completed, not stopped or failed")
    if manifest.get("runtime", {}).get("kind") != "mlx":
        raise MlxSessionError("6bit promotion source must use the MLX runtime")
    if not manifest.get("classification") == "smoke-only" or not summary.get("smoke_only"):
        raise MlxSessionError("6bit promotion source must be classified smoke-only")
    totals = summary.get("totals", {})
    if totals.get("failed") or totals.get("resource_stops") or not totals.get("passed"):
        raise MlxSessionError("6bit promotion source must contain a successful 4bit smoke with no resource stop")
    if not rows or any(not str(row.get("parse", {}).get("final_text", "")).strip() for row in rows):
        raise MlxSessionError("6bit promotion source must contain nonempty parsed final text")
    timestamp = manifest.get("timestamp")
    if not isinstance(timestamp, str):
        raise MlxSessionError("6bit promotion source is missing timestamp")
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise MlxSessionError(f"6bit promotion source timestamp is invalid: {exc}")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - when
    if age > timedelta(hours=24):
        raise MlxSessionError("6bit promotion source is older than 24 hours")
    if age < -timedelta(minutes=5):
        raise MlxSessionError("6bit promotion source timestamp is in the future")

    environment = manifest.get("environment", {})
    if environment.get("host") != platform.node() or environment.get("machine") != platform.machine():
        raise MlxSessionError("6bit promotion source was produced on a different host")
    if manifest.get("harness", {}).get("version") != __version__:
        raise MlxSessionError("6bit promotion source uses a different harness version")

    from importlib.metadata import PackageNotFoundError, version

    recorded_packages = environment.get("packages", {})
    for package in ("mlx", "mlx-lm", "transformers", "huggingface-hub", "numpy"):
        try:
            current = version(package)
        except PackageNotFoundError:
            current = "missing"
        if recorded_packages.get(package) != current:
            raise MlxSessionError(
                f"6bit promotion source package mismatch for {package}: "
                f"{recorded_packages.get(package)} != {current}"
            )

    performance = summary.get("performance", {})
    resources = summary.get("resources", {})
    if float(performance.get("wall_seconds", 0) or 0) <= 0:
        raise MlxSessionError("6bit promotion source is missing wall timing")
    peak_memory = int(resources.get("peak_mlx_memory_bytes", 0) or 0)
    if peak_memory <= 0:
        raise MlxSessionError("6bit promotion source is missing peak MLX memory")
    max_working_set = int(
        manifest.get("runtime", {})
        .get("profile_checks", {})
        .get("metal", {})
        .get("details", {})
        .get("max_recommended_working_set_size", 0)
        or 0
    )
    if max_working_set <= 0:
        raise MlxSessionError("6bit promotion source is missing the MLX working-set limit")
    if peak_memory >= int(max_working_set * 0.85):
        raise MlxSessionError("6bit promotion denied: 4bit peak memory reached 85% of the MLX working-set limit")
    pressure = resources.get("memory_pressure_after")
    if pressure is None or int(pressure) <= 20:
        raise MlxSessionError("6bit promotion denied: 4bit memory pressure was unsafe or unavailable")
    swap_delta = resources.get("swap_delta_bytes")
    if swap_delta is None or int(swap_delta) > 1024**3:
        raise MlxSessionError("6bit promotion denied: 4bit swap growth was unsafe or unavailable")


def _load_real_api() -> SimpleNamespace:
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise MlxSessionError(f"Required MLX package import failed: {exc}")
    return SimpleNamespace(
        mx=mx,
        load=load,
        stream_generate=stream_generate,
        make_prompt_cache=make_prompt_cache,
        make_sampler=make_sampler,
    )


def _count_prompt_tokens(prompt: str) -> int:
    return len(prompt.split())


def _tokenizer_prompt_tokens(tokenizer: Any, prompt: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None
    try:
        encoded = encode(prompt)
        return len(encoded)
    except Exception:
        return None


def _prepare_generation_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool,
) -> tuple[Any, bool]:
    if bool(getattr(tokenizer, "has_chat_template", False)):
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                return (
                    apply_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    ),
                    True,
                )
            except Exception as exc:
                raise MlxSessionError(
                    f"chat template application failed: {type(exc).__name__}: {exc}"
                ) from exc
    return prompt, False


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return str(chunk)


def _positive_int_metric(chunk: Any, name: str, fallback: int) -> int:
    value = getattr(chunk, name, None)
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    return fallback


def _positive_float_metric(chunk: Any, name: str, fallback: float) -> float:
    value = getattr(chunk, name, None)
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return fallback


def capture_resource_snapshot() -> dict[str, int | None]:
    """Capture the small host-resource set required for real-run evidence."""
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    disk_target = hf_home if hf_home.exists() else hf_home.parent
    try:
        disk_free_bytes = int(shutil.disk_usage(disk_target).free)
    except OSError:
        disk_free_bytes = 0

    pressure: int | None = None
    try:
        completed = subprocess.run(
            ["sysctl", "kern.memorystatus_level"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            match = re.search(r"(\d+)\s*$", completed.stdout)
            if match:
                pressure = int(match.group(1))
    except Exception:
        pass

    swap_used_bytes: int | None = None
    try:
        completed = subprocess.run(
            ["sysctl", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", completed.stdout)
            if match:
                units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
                swap_used_bytes = int(float(match.group(1)) * units[match.group(2)])
    except Exception:
        pass

    return {
        "disk_free_bytes": disk_free_bytes,
        "memory_pressure": pressure,
        "swap_used_bytes": swap_used_bytes,
    }
