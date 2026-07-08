"""Tests for MLX runtime wiring and smoke gates.

These tests never download model weights.  They use fake MLX/MLX-LM surfaces
to verify API wiring and gate behavior.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ornith_mlx_eval.mlx_session import (
    FOUR_BIT_MODEL,
    FOUR_BIT_SHA,
    MlxGenerationResult,
    MlxGenerationOptions,
    MlxSessionError,
    generate_with_mlx,
)
from ornith_mlx_eval.runner import RunOptions, run_evaluation


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLI = _REPO_ROOT / ".venv" / "bin" / "ornith-mlx-eval"


class FakeChunk:
    def __init__(self, text: str):
        self.text = text


class FakeMx:
    def __init__(self):
        self.seed_values = []
        self.reset_called = False
        self.clear_called = False

        class Random:
            def __init__(self, outer):
                self._outer = outer

            def seed(self, value):
                self._outer.seed_values.append(value)

        self.random = Random(self)

    def reset_peak_memory(self):
        self.reset_called = True

    def get_peak_memory(self):
        return 123456

    def clear_cache(self):
        self.clear_called = True


class FakeApi:
    def __init__(self):
        self.mx = FakeMx()
        self.load_calls = []
        self.sampler_calls = []
        self.cache_calls = []
        self.stream_kwargs = None

    def load(self, model_id, *, revision):
        self.load_calls.append((model_id, revision))
        return "model", "tokenizer"

    def make_sampler(self, *, temp, top_p, top_k):
        self.sampler_calls.append({"temp": temp, "top_p": top_p, "top_k": top_k})
        return "sampler"

    def make_prompt_cache(self, model, *, max_kv_size):
        self.cache_calls.append({"model": model, "max_kv_size": max_kv_size})
        return "prompt-cache"

    def stream_generate(self, model, tokenizer, prompt, **kwargs):
        self.stream_kwargs = kwargs
        yield FakeChunk("Par")
        yield FakeChunk("is")


def _cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_CLI), *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestMlxSessionCore:
    def test_load_is_pinned_to_exact_revision_and_sampler_options_translate(self):
        api = FakeApi()
        result = generate_with_mlx(
            FOUR_BIT_MODEL,
            "1e980b9742a9e554a4d57e90b4c597811fb2fc4e",
            "What is the capital of France?",
            MlxGenerationOptions(
                max_tokens=5,
                temperature=0.2,
                top_p=0.9,
                top_k=10,
                seed=123,
                max_prompt_tokens=100,
                max_kv_size=2048,
            ),
            api=api,
        )

        assert api.load_calls == [(FOUR_BIT_MODEL, "1e980b9742a9e554a4d57e90b4c597811fb2fc4e")]
        assert api.sampler_calls == [{"temp": 0.2, "top_p": 0.9, "top_k": 10}]
        assert api.mx.seed_values == [123]
        assert api.cache_calls == [{"model": "model", "max_kv_size": 2048}]
        assert api.mx.reset_called is True
        assert api.mx.clear_called is True
        assert result.raw_text == "Paris"
        assert result.peak_mlx_memory_bytes == 123456

    def test_stream_generate_receives_sampler_and_prompt_cache_not_unsupported_kwargs(self):
        api = FakeApi()
        generate_with_mlx(
            FOUR_BIT_MODEL,
            "1e980b9742a9e554a4d57e90b4c597811fb2fc4e",
            "Prompt",
            MlxGenerationOptions(temperature=0.4, top_p=0.8, top_k=4, seed=1),
            api=api,
        )

        kwargs = api.stream_kwargs
        assert kwargs["sampler"] == "sampler"
        assert kwargs["prompt_cache"] == "prompt-cache"
        assert kwargs["max_tokens"] == 512
        for unsupported in ["temperature", "top_p", "top_k", "seed"]:
            assert unsupported not in kwargs

    def test_prompt_token_limit_fails_before_model_load(self):
        api = FakeApi()
        with pytest.raises(MlxSessionError, match="prompt token limit"):
            generate_with_mlx(
                FOUR_BIT_MODEL,
                "1e980b9742a9e554a4d57e90b4c597811fb2fc4e",
                "one two three",
                MlxGenerationOptions(max_prompt_tokens=2),
                api=api,
            )
        assert api.load_calls == []


class TestSmokeCliGates:
    def test_smoke_requires_explicit_real_model_opt_in(self):
        result = _cli(["smoke", "--model", FOUR_BIT_MODEL])
        assert result.returncode != 0
        assert "opt-in" in result.stderr.lower()

    def test_smoke_rejects_unsupported_model_before_download(self):
        result = _cli(["smoke", "--model", "mlx-community/Ornith-1.0-9B-8bit"])
        assert result.returncode != 0
        assert "unsupported" in result.stderr.lower()

    def test_smoke_help_documents_download_opt_in(self):
        result = _cli(["smoke", "--help"])
        assert result.returncode == 0
        assert "--allow-download" in result.stdout


class TestMlxRunnerAdapter:
    def test_mlx_runtime_requires_explicit_download_opt_in_before_output(self, tmp_path):
        output_root = tmp_path / "runs"
        with pytest.raises(Exception, match="explicit opt-in"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(output_root),
                )
            )
        assert not output_root.exists()

    def test_mlx_runtime_adapter_writes_artifacts_with_fake_generation(self, tmp_path, monkeypatch):
        from ornith_mlx_eval import runner as runner_module

        calls = []

        def fake_generate(model_id, revision, prompt, options):
            calls.append((model_id, revision, prompt, options.max_kv_size))
            return MlxGenerationResult(
                raw_text="Paris",
                prompt_tokens=5,
                generated_tokens=1,
                peak_mlx_memory_bytes=99,
                model_id=model_id,
                revision=revision,
                max_kv_size=options.max_kv_size,
            )

        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "generate_with_mlx", fake_generate)
        run_dir = run_evaluation(
            RunOptions(
                runtime="mlx",
                suite="smoke",
                model=FOUR_BIT_MODEL,
                output_root=str(tmp_path / "runs"),
                limit=1,
                allow_download=True,
                max_kv_size=2048,
            )
        )

        manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
        results = (run_dir / "results.jsonl").read_text(encoding="utf-8")
        assert '"kind": "mlx"' in manifest
        assert FOUR_BIT_SHA in manifest
        assert '"peak_mlx_memory_bytes":99' in results
        assert calls[0][0] == FOUR_BIT_MODEL
        assert calls[0][1] == FOUR_BIT_SHA
        assert calls[0][3] == 2048
