"""Direct MLX/MLX-LM runtime integration.

The runtime path uses verified MLX APIs only and never falls back to Ollama or
HTTP-compatible backends.  Tests can inject fake dependencies through the
``api`` parameter to avoid model downloads.
"""

from __future__ import annotations

import hashlib
import json
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
    token_ids: tuple[int, ...] = ()
    response_sha256: str = ""
    token_ids_sha256: str = ""
    cold_load_seconds: float = 0.0
    first_token_seconds: float = 0.0
    decode_seconds: float = 0.0
    decode_tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    runtime_reported_tps: float = 0.0
    wall_seconds: float = 0.0
    disk_free_before_bytes: int = 0
    disk_free_after_bytes: int = 0
    memory_pressure_before: int | None = None
    memory_pressure_after: int | None = None
    swap_used_before_bytes: int | None = None
    swap_used_after_bytes: int | None = None
    chat_template_applied: bool = False


class MlxLoadedSession:
    """One pinned loaded model reused across fresh-cache generations."""

    def __init__(
        self,
        model_id: str,
        revision_sha: str,
        options: MlxGenerationOptions,
        *,
        api: Any,
        clock: Any,
        resource_probe: Any,
        memory_limit_bytes: int | None,
    ) -> None:
        _validate_generation_request(revision_sha, options)
        self.model_id = model_id
        self.revision_sha = revision_sha
        self.api = api
        self.clock = clock
        self.resource_probe = resource_probe
        self.memory_limit_bytes = memory_limit_bytes
        self._closed = False
        self._generation_count = 0
        self._before_load = resource_probe()
        self._session_wall_started = clock()

        stage = "memory limit setup"
        try:
            if memory_limit_bytes is not None:
                if memory_limit_bytes <= 0:
                    raise MlxSessionError("MLX memory limit must be greater than zero")
                self.api.mx.set_memory_limit(int(memory_limit_bytes))
            stage = "model load"
            load_started = clock()
            self.model, self.tokenizer = self.api.load(
                model_id, revision=revision_sha
            )
            self.cold_load_seconds = max(0.0, clock() - load_started)
        except MlxSessionError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise MlxSessionError(
                f"{stage} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def __enter__(self) -> "MlxLoadedSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.api.mx.clear_cache()
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        options: MlxGenerationOptions,
        *,
        logits_processors: list[Any] | None = None,
    ) -> MlxGenerationResult:
        """Generate once with a fresh prompt cache and a reset seed/peak."""
        if self._closed:
            raise MlxSessionError("MLX loaded session is closed")
        _validate_generation_request(self.revision_sha, options, prompt=prompt)

        before = self._before_load if self._generation_count == 0 else self.resource_probe()
        generation_wall_started = self.clock()
        stage = "memory measurement reset"
        chunks: list[str] = []
        token_ids: list[int] = []
        chunk_times: list[float] = []
        prompt_tokens = _count_prompt_tokens(prompt)
        prompt_tps = 0.0
        runtime_reported_tps = 0.0
        response_peak_memory_bytes = 0
        chat_template_applied = False
        try:
            self.api.mx.reset_peak_memory()
            stage = "random seed setup"
            self.api.mx.random.seed(options.seed)

            generation_prompt, chat_template_applied = _prepare_generation_prompt(
                self.tokenizer,
                prompt,
                enable_thinking=options.enable_thinking,
            )
            actual_prompt_tokens = (
                len(generation_prompt)
                if isinstance(generation_prompt, (list, tuple))
                else _tokenizer_prompt_tokens(self.tokenizer, str(generation_prompt))
            )
            if actual_prompt_tokens is not None:
                prompt_tokens = actual_prompt_tokens
            if prompt_tokens > options.max_prompt_tokens:
                raise MlxSessionError(
                    "Rendered prompt exceeds prompt token limit after tokenization: "
                    f"{prompt_tokens} > {options.max_prompt_tokens}"
                )

            stage = "sampler setup"
            sampler = self.api.make_sampler(
                temp=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
            )
            stage = "prompt cache setup"
            prompt_cache = self.api.make_prompt_cache(
                self.model, max_kv_size=options.max_kv_size
            )

            stage = "generation"
            generation_started = self.clock()
            stream_kwargs: dict[str, Any] = {
                "max_tokens": options.max_tokens,
                "sampler": sampler,
                "prompt_cache": prompt_cache,
            }
            if logits_processors is not None:
                stream_kwargs["logits_processors"] = logits_processors
            final_generation_tokens = 0
            for chunk in self.api.stream_generate(
                self.model,
                self.tokenizer,
                generation_prompt,
                **stream_kwargs,
            ):
                observed_at = self.clock()
                chunks.append(_chunk_text(chunk))
                chunk_times.append(observed_at)
                token_ids.append(_required_token_id(chunk))
                prompt_tokens = _required_nonnegative_int_metric(
                    chunk, "prompt_tokens"
                )
                final_generation_tokens = _required_positive_int_metric(
                    chunk, "generation_tokens"
                )
                prompt_tps = _positive_float_metric(chunk, "prompt_tps", prompt_tps)
                runtime_reported_tps = _positive_float_metric(
                    chunk, "generation_tps", runtime_reported_tps
                )
                chunk_peak_gb = _positive_float_metric(chunk, "peak_memory", 0.0)
                response_peak_memory_bytes = max(
                    response_peak_memory_bytes,
                    int(chunk_peak_gb * 1_000_000_000),
                )
            if not token_ids or final_generation_tokens != len(token_ids):
                raise MlxSessionError(
                    "generation token metrics were missing or inconsistent"
                )
            stage = "peak memory read"
            peak_memory = max(
                int(self.api.mx.get_peak_memory()), response_peak_memory_bytes
            )
        except MlxSessionError:
            raise
        except Exception as exc:
            raise MlxSessionError(
                f"{stage} failed: {type(exc).__name__}: {exc}"
            ) from exc

        after = self.resource_probe()
        raw_text = "".join(chunks)
        first_token_seconds = max(0.0, chunk_times[0] - generation_started)
        decode_seconds = (
            max(0.0, chunk_times[-1] - chunk_times[0])
            if len(chunk_times) > 1
            else 0.0
        )
        measured_tps = (
            (len(token_ids) - 1) / decode_seconds
            if len(token_ids) > 1 and decode_seconds > 0
            else 0.0
        )
        is_first_generation = self._generation_count == 0
        self._generation_count += 1
        wall_seconds = max(
            0.0,
            self.clock()
            - (self._session_wall_started if is_first_generation else generation_wall_started),
        )
        response_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        token_ids_sha256 = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return MlxGenerationResult(
            raw_text=raw_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(token_ids),
            peak_mlx_memory_bytes=peak_memory,
            model_id=self.model_id,
            revision=self.revision_sha,
            max_kv_size=options.max_kv_size,
            token_ids=tuple(token_ids),
            response_sha256=response_sha256,
            token_ids_sha256=token_ids_sha256,
            cold_load_seconds=self.cold_load_seconds if is_first_generation else 0.0,
            first_token_seconds=first_token_seconds,
            decode_seconds=decode_seconds,
            decode_tokens_per_second=measured_tps,
            prompt_tokens_per_second=prompt_tps,
            runtime_reported_tps=runtime_reported_tps,
            wall_seconds=wall_seconds,
            disk_free_before_bytes=int(before.get("disk_free_bytes", 0) or 0),
            disk_free_after_bytes=int(after.get("disk_free_bytes", 0) or 0),
            memory_pressure_before=before.get("memory_pressure"),
            memory_pressure_after=after.get("memory_pressure"),
            swap_used_before_bytes=before.get("swap_used_bytes"),
            swap_used_after_bytes=after.get("swap_used_bytes"),
            chat_template_applied=chat_template_applied,
        )


def open_mlx_session(
    model_id: str,
    revision_sha: str,
    options: MlxGenerationOptions,
    *,
    api: Any | None = None,
    clock: Any = time.perf_counter,
    resource_probe: Any | None = None,
    memory_limit_bytes: int | None = None,
) -> MlxLoadedSession:
    """Load a pinned model once; tests may inject the complete MLX API surface."""
    return MlxLoadedSession(
        model_id,
        revision_sha,
        options,
        api=api or _load_real_api(),
        clock=clock,
        resource_probe=resource_probe or capture_resource_snapshot,
        memory_limit_bytes=memory_limit_bytes,
    )


def generate_with_mlx(
    model_id: str,
    revision_sha: str,
    prompt: str,
    options: MlxGenerationOptions,
    *,
    api: Any | None = None,
    clock: Any = time.perf_counter,
    resource_probe: Any | None = None,
    memory_limit_bytes: int | None = None,
) -> MlxGenerationResult:
    """Compatibility wrapper for one generation through a loaded session."""
    _validate_generation_request(revision_sha, options, prompt=prompt)
    with open_mlx_session(
        model_id,
        revision_sha,
        options,
        api=api,
        clock=clock,
        resource_probe=resource_probe,
        memory_limit_bytes=memory_limit_bytes,
    ) as session:
        return session.generate(prompt, options)


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
    projected_six_bit_peak = peak_memory * 3 // 2
    working_set_budget = int(max_working_set * 0.85)
    if projected_six_bit_peak > working_set_budget:
        raise MlxSessionError(
            "6bit promotion denied: projected 6bit peak exceeds 85% of the "
            f"MLX working-set limit ({projected_six_bit_peak} > {working_set_budget})"
        )
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


def _validate_generation_request(
    revision_sha: str,
    options: MlxGenerationOptions,
    *,
    prompt: str | None = None,
) -> None:
    if not revision_sha or len(revision_sha) != 40:
        raise MlxSessionError("MLX runtime requires an exact 40-character revision SHA")
    if options.max_tokens <= 0:
        raise MlxSessionError("MLX runtime requires max_tokens greater than zero")
    if options.max_kv_size <= 0:
        raise MlxSessionError("MLX runtime requires max_kv_size greater than zero")
    if options.max_prompt_tokens <= 0:
        raise MlxSessionError("MLX runtime requires max_prompt_tokens greater than zero")
    if prompt is not None:
        estimated_prompt_tokens = _count_prompt_tokens(prompt)
        if estimated_prompt_tokens > options.max_prompt_tokens:
            raise MlxSessionError(
                "Rendered prompt exceeds prompt token limit: "
                f"{estimated_prompt_tokens} > {options.max_prompt_tokens}"
            )


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


def _required_token_id(chunk: Any) -> int:
    value = getattr(chunk, "token", None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MlxSessionError("generation token metrics were missing: token id")
    return value


def _required_nonnegative_int_metric(chunk: Any, name: str) -> int:
    value = getattr(chunk, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise MlxSessionError(f"generation token metrics were missing: {name}")
    return int(value)


def _required_positive_int_metric(chunk: Any, name: str) -> int:
    value = _required_nonnegative_int_metric(chunk, name)
    if value <= 0:
        raise MlxSessionError(f"generation token metrics were missing: {name}")
    return value


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
