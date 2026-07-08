"""Direct MLX/MLX-LM runtime integration.

The runtime path uses verified MLX APIs only and never falls back to Ollama or
HTTP-compatible backends.  Tests can inject fake dependencies through the
``api`` parameter to avoid model downloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ornith_mlx_eval.profile import run_profile
from ornith_mlx_eval.results import ResultArtifactError, load_json


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


def generate_with_mlx(
    model_id: str,
    revision_sha: str,
    prompt: str,
    options: MlxGenerationOptions,
    *,
    api: Any | None = None,
) -> MlxGenerationResult:
    """Generate text with MLX-LM using exact revision and verified APIs.

    The function intentionally accepts an injected ``api`` object for tests.
    Real execution imports MLX/MLX-LM lazily only after prompt gates pass.
    """
    prompt_tokens = _count_prompt_tokens(prompt)
    if prompt_tokens > options.max_prompt_tokens:
        raise MlxSessionError(
            f"Rendered prompt exceeds prompt token limit: {prompt_tokens} > {options.max_prompt_tokens}"
        )
    if not revision_sha or len(revision_sha) != 40:
        raise MlxSessionError("MLX runtime requires an exact 40-character revision SHA")

    api = api or _load_real_api()

    api.mx.random.seed(options.seed)
    model, tokenizer = api.load(model_id, revision=revision_sha)
    sampler = api.make_sampler(
        temp=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
    )
    prompt_cache = api.make_prompt_cache(model, max_kv_size=options.max_kv_size)
    api.mx.reset_peak_memory()

    chunks: list[str] = []
    for chunk in api.stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=options.max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        chunks.append(_chunk_text(chunk))

    peak_memory = int(api.mx.get_peak_memory())
    api.mx.clear_cache()
    raw_text = "".join(chunks)
    return MlxGenerationResult(
        raw_text=raw_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(raw_text.split()),
        peak_mlx_memory_bytes=peak_memory,
        model_id=model_id,
        revision=revision_sha,
        max_kv_size=options.max_kv_size,
    )


def run_real_smoke(
    model_id: str,
    *,
    allow_download: bool,
    promotion_source: str | None = None,
    max_tokens: int = 32,
    seed: int = 42,
    temperature: float = 0,
    top_p: float = 1,
    top_k: int = 0,
    max_prompt_tokens: int = 8192,
    max_kv_size: int = 4096,
) -> dict[str, Any]:
    """Run a gated one-prompt real MLX smoke.

    This may download model weights only when ``allow_download`` is true and
    ``ORNITH_MLX_ALLOW_MODEL_DOWNLOAD=1`` is set.
    """
    revision = _gate_real_smoke(model_id, allow_download=allow_download)
    if model_id == SIX_BIT_MODEL:
        _validate_6bit_promotion_source(promotion_source)

    profile = run_profile(model_id=model_id)
    if profile["status"] != "pass":
        raise MlxSessionError("Profile gates failed before model load")
    model_result = profile.get("model", {})
    if model_result.get("details", {}).get("sha") != revision:
        raise MlxSessionError("Resolved model SHA does not match required pinned revision")

    result = generate_with_mlx(
        model_id,
        revision,
        "Answer with the single word Paris.",
        MlxGenerationOptions(
            max_tokens=min(max_tokens, 32),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_prompt_tokens=max_prompt_tokens,
            max_kv_size=max_kv_size,
        ),
    )
    if not result.raw_text.strip():
        raise MlxSessionError("Smoke generation produced empty output")
    return {
        "status": "pass",
        "smoke_only": True,
        "model_id": model_id,
        "revision": revision,
        "raw_text": result.raw_text,
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "peak_mlx_memory_bytes": result.peak_mlx_memory_bytes,
    }


def _gate_real_smoke(model_id: str, *, allow_download: bool) -> str:
    if "8bit" in model_id.lower() or "35b" in model_id.lower():
        raise MlxSessionError(f"Unsupported real-smoke model: {model_id}")
    revision = SUPPORTED_MODEL_SHAS.get(model_id)
    if revision is None:
        raise MlxSessionError(f"Unsupported real-smoke model: {model_id}")
    env_opt_in = os.environ.get("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD") == "1"
    if not allow_download or not env_opt_in:
        raise MlxSessionError(
            "Real MLX smoke requires explicit opt-in: pass --allow-download and set "
            "ORNITH_MLX_ALLOW_MODEL_DOWNLOAD=1"
        )
    return revision


def _validate_6bit_promotion_source(promotion_source: str | None) -> None:
    if not promotion_source:
        raise MlxSessionError("6bit promotion requires --promotion-source pointing to a fresh 4bit smoke manifest")
    manifest = load_json(Path(promotion_source), "promotion-source manifest.json")
    model = manifest.get("model", {})
    if model.get("repo_id") != FOUR_BIT_MODEL or model.get("revision") != FOUR_BIT_SHA:
        raise MlxSessionError("6bit promotion source must be a completed 4bit smoke artifact")
    timestamp = manifest.get("timestamp")
    if not isinstance(timestamp, str):
        raise MlxSessionError("6bit promotion source is missing timestamp")
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise MlxSessionError(f"6bit promotion source timestamp is invalid: {exc}")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - when > timedelta(hours=24):
        raise MlxSessionError("6bit promotion source is older than 24 hours")


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


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return str(chunk)
