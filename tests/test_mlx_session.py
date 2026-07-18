"""Tests for MLX runtime wiring and smoke gates.

These tests never download model weights.  They use fake MLX/MLX-LM surfaces
to verify API wiring and gate behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ornith_mlx_eval.mlx_session import (
    FOUR_BIT_MODEL,
    FOUR_BIT_SHA,
    MlxGenerationResult,
    MlxGenerationOptions,
    MlxSessionError,
    SIX_BIT_MODEL,
    SIX_BIT_SHA,
    generate_with_mlx,
    validate_6bit_promotion_source,
)
from ornith_mlx_eval.runner import RunOptions, run_evaluation


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLI = _REPO_ROOT / ".venv" / "bin" / "ornith-mlx-eval"


class FakeChunk:
    def __init__(
        self,
        text: str,
        *,
        prompt_tokens: int = 7,
        generation_tokens: int = 1,
        prompt_tps: float = 35.0,
        generation_tps: float = 12.5,
        peak_memory: float = 0.25,
        token: int = 1,
    ):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.prompt_tps = prompt_tps
        self.generation_tps = generation_tps
        self.peak_memory = peak_memory
        self.token = token


class FakeMx:
    def __init__(self):
        self.seed_values = []
        self.reset_called = False
        self.reset_count = 0
        self.clear_called = False
        self.events = []
        self.memory_limits = []

        class Random:
            def __init__(self, outer):
                self._outer = outer

            def seed(self, value):
                self._outer.seed_values.append(value)

        self.random = Random(self)

    def reset_peak_memory(self):
        self.reset_called = True
        self.reset_count += 1
        self.events.append("reset")

    def set_memory_limit(self, value):
        self.memory_limits.append(value)
        self.events.append("memory-limit")
        return 0

    def get_peak_memory(self):
        return 123456

    def clear_cache(self):
        self.clear_called = True
        self.events.append("clear")


class FakeApi:
    def __init__(self):
        self.mx = FakeMx()
        self.load_calls = []
        self.sampler_calls = []
        self.cache_calls = []
        self.stream_kwargs = None

    def load(self, model_id, *, revision):
        self.mx.events.append("load")
        self.load_calls.append((model_id, revision))
        return "model", "tokenizer"

    def make_sampler(self, *, temp, top_p, top_k):
        self.sampler_calls.append({"temp": temp, "top_p": top_p, "top_k": top_k})
        return "sampler"

    def make_prompt_cache(self, model, *, max_kv_size):
        self.cache_calls.append({"model": model, "max_kv_size": max_kv_size})
        return f"prompt-cache-{len(self.cache_calls)}"

    def stream_generate(self, model, tokenizer, prompt, **kwargs):
        self.stream_kwargs = kwargs
        yield FakeChunk("Par", generation_tokens=1, token=101)
        yield FakeChunk("is", generation_tokens=2, token=102)


class FakeChatTokenizer:
    has_chat_template = True

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [101, 102, 103, 104]


def _passing_profile(
    *,
    model_id: str = FOUR_BIT_MODEL,
    revision: str = FOUR_BIT_SHA,
    size_bytes: int = 5_950_219_560,
) -> dict:
    return {
        "status": "pass",
        "checks": [
            {
                "name": "metal",
                "status": "pass",
                "details": {
                    "metal_available": True,
                    "max_recommended_working_set_size": 8 * 1024**3,
                },
            },
            {
                "name": "disk",
                "status": "pass",
                "details": {
                    "free_bytes": 50 * 1024**3,
                    "required_bytes": 18 * 1024**3,
                },
            },
            {
                "name": "memory",
                "status": "pass",
                "details": {
                    "memory_pressure": "kern.memorystatus_level: 75",
                    "swap": "vm.swapusage: total = 1024.00M used = 128.00M free = 896.00M",
                },
            },
        ],
        "model": {
            "name": "model",
            "status": "pass",
            "details": {
                "model_id": model_id,
                "sha": revision,
                "size_bytes": size_bytes,
            },
        },
    }


def _measured_generation(
    raw_text: str = "Paris",
    *,
    model_id: str = FOUR_BIT_MODEL,
    revision: str = FOUR_BIT_SHA,
) -> MlxGenerationResult:
    token_ids = (101, 102)
    return MlxGenerationResult(
        raw_text=raw_text,
        prompt_tokens=7,
        generated_tokens=2,
        peak_mlx_memory_bytes=256_000_000,
        model_id=model_id,
        revision=revision,
        max_kv_size=2048,
        token_ids=token_ids,
        response_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        token_ids_sha256=hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        cold_load_seconds=0.5,
        first_token_seconds=0.2,
        decode_seconds=0.08,
        decode_tokens_per_second=12.5,
        prompt_tokens_per_second=35.0,
        runtime_reported_tps=12.5,
        wall_seconds=0.86,
        disk_free_before_bytes=50 * 1024**3,
        disk_free_after_bytes=50 * 1024**3,
        memory_pressure_before=75,
        memory_pressure_after=74,
        swap_used_before_bytes=128 * 1024**2,
        swap_used_after_bytes=128 * 1024**2,
    )


def _cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_CLI), *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _create_passing_4bit_source(tmp_path, monkeypatch) -> Path:
    from ornith_mlx_eval import runner as runner_module

    monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
    monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
    monkeypatch.setattr(
        runner_module,
        "generate_with_mlx",
        lambda *args, **kwargs: _measured_generation(),
    )
    return run_evaluation(
        RunOptions(
            runtime="mlx",
            suite="smoke",
            model=FOUR_BIT_MODEL,
            output_root=str(tmp_path / "four-bit-runs"),
            limit=1,
            allow_download=True,
        )
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
        assert api.mx.events.index("load") < api.mx.events.index("reset")
        assert api.mx.clear_called is True
        assert result.raw_text == "Paris"
        assert result.peak_mlx_memory_bytes == 250_000_000
        assert result.prompt_tokens == 7
        assert result.generated_tokens == 2
        assert result.token_ids == (101, 102)
        assert result.response_sha256 == hashlib.sha256(b"Paris").hexdigest()
        assert result.decode_tokens_per_second > 0
        assert result.runtime_reported_tps == 12.5
        assert result.wall_seconds > 0

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
        assert kwargs["prompt_cache"] == "prompt-cache-1"
        assert kwargs["max_tokens"] == 512
        for unsupported in ["temperature", "top_p", "top_k", "seed"]:
            assert unsupported not in kwargs

    def test_instruction_models_receive_tokenized_chat_template_with_thinking_disabled(self):
        api = FakeApi()
        tokenizer = FakeChatTokenizer()

        def load(model_id, *, revision):
            api.mx.events.append("load")
            api.load_calls.append((model_id, revision))
            return "model", tokenizer

        api.load = load
        streamed_prompts = []

        def stream(model, loaded_tokenizer, prompt, **kwargs):
            streamed_prompts.append(prompt)
            yield FakeChunk("Paris", prompt_tokens=4, generation_tokens=1)

        api.stream_generate = stream
        result = generate_with_mlx(
            FOUR_BIT_MODEL,
            FOUR_BIT_SHA,
            "What is the capital of France?",
            MlxGenerationOptions(max_tokens=8, enable_thinking=False),
            api=api,
        )

        assert streamed_prompts == [[101, 102, 103, 104]]
        messages, kwargs = tokenizer.calls[0]
        assert messages == [{"role": "user", "content": "What is the capital of France?"}]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        assert result.prompt_tokens == 4

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

    def test_model_load_errors_are_stage_specific_and_cache_is_cleared(self):
        api = FakeApi()

        def fail_load(model_id, *, revision):
            raise ValueError("unsupported architecture")

        api.load = fail_load
        with pytest.raises(MlxSessionError, match="model load failed.*unsupported architecture"):
            generate_with_mlx(
                FOUR_BIT_MODEL,
                FOUR_BIT_SHA,
                "Prompt",
                MlxGenerationOptions(),
                api=api,
            )
        assert api.mx.clear_called is True

    def test_loaded_session_reuses_one_load_with_fresh_cache_reseed_and_memory_limit(self):
        from ornith_mlx_eval.mlx_session import open_mlx_session

        api = FakeApi()
        options = MlxGenerationOptions(seed=17, max_kv_size=2048)
        with open_mlx_session(
            FOUR_BIT_MODEL,
            FOUR_BIT_SHA,
            options,
            api=api,
            memory_limit_bytes=7_000_000_000,
        ) as session:
            first = session.generate("Prompt", options)
            second = session.generate("Prompt", options)

        assert api.load_calls == [(FOUR_BIT_MODEL, FOUR_BIT_SHA)]
        assert api.mx.events.index("memory-limit") < api.mx.events.index("load")
        assert api.mx.memory_limits == [7_000_000_000]
        assert api.mx.reset_count == 2
        assert api.mx.seed_values == [17, 17]
        assert [call["max_kv_size"] for call in api.cache_calls] == [2048, 2048]
        assert first.token_ids_sha256 == second.token_ids_sha256
        assert first.response_sha256 == second.response_sha256
        assert first.cold_load_seconds >= 0
        assert second.cold_load_seconds == 0

    def test_real_generation_fails_closed_when_chunk_token_metrics_are_missing(self):
        from ornith_mlx_eval.mlx_session import open_mlx_session

        api = FakeApi()

        class TextOnlyChunk:
            text = "Paris"

        api.stream_generate = lambda *args, **kwargs: iter([TextOnlyChunk()])
        with open_mlx_session(
            FOUR_BIT_MODEL,
            FOUR_BIT_SHA,
            MlxGenerationOptions(),
            api=api,
        ) as session:
            with pytest.raises(MlxSessionError, match="token metrics"):
                session.generate("Prompt", MlxGenerationOptions())

    def test_same_seed_repeats_match_and_a_different_seed_may_diverge(self):
        from ornith_mlx_eval.mlx_session import open_mlx_session

        api = FakeApi()

        def stream(model, tokenizer, prompt, **kwargs):
            token = api.mx.seed_values[-1]
            yield FakeChunk(str(token), generation_tokens=1, token=token)

        api.stream_generate = stream
        with open_mlx_session(
            FOUR_BIT_MODEL,
            FOUR_BIT_SHA,
            MlxGenerationOptions(temperature=0.7),
            api=api,
        ) as session:
            first = session.generate("Prompt", MlxGenerationOptions(seed=11, temperature=0.7))
            second = session.generate("Prompt", MlxGenerationOptions(seed=11, temperature=0.7))
            different = session.generate("Prompt", MlxGenerationOptions(seed=12, temperature=0.7))

        assert first.token_ids == second.token_ids
        assert first.token_ids != different.token_ids

    def test_session_reuse_matches_fresh_session_with_fresh_prompt_caches(self):
        from ornith_mlx_eval.mlx_session import open_mlx_session

        options = MlxGenerationOptions(seed=42)
        reused_api = FakeApi()
        with open_mlx_session(
            FOUR_BIT_MODEL, FOUR_BIT_SHA, options, api=reused_api
        ) as reused:
            reused_first = reused.generate("Prompt", options)
            reused_second = reused.generate("Prompt", options)

        fresh_api = FakeApi()
        with open_mlx_session(
            FOUR_BIT_MODEL, FOUR_BIT_SHA, options, api=fresh_api
        ) as fresh:
            fresh_result = fresh.generate("Prompt", options)

        assert len(reused_api.cache_calls) == 2
        assert reused_first.token_ids_sha256 == reused_second.token_ids_sha256
        assert reused_second.token_ids_sha256 == fresh_result.token_ids_sha256

    def test_decode_window_uses_first_and_last_chunk_timestamps_not_runtime_tps(self):
        from ornith_mlx_eval.mlx_session import open_mlx_session

        api = FakeApi()

        def stream(model, tokenizer, prompt, **kwargs):
            yield FakeChunk("A", generation_tokens=1, generation_tps=999.0, token=1)
            yield FakeChunk("B", generation_tokens=2, generation_tps=999.0, token=2)
            yield FakeChunk("C", generation_tokens=3, generation_tps=999.0, token=3)

        api.stream_generate = stream
        clock_values = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 11.0])
        clock = lambda: next(clock_values)
        with open_mlx_session(
            FOUR_BIT_MODEL,
            FOUR_BIT_SHA,
            MlxGenerationOptions(),
            api=api,
            clock=clock,
        ) as session:
            result = session.generate("Prompt", MlxGenerationOptions())

        assert result.first_token_seconds == 1.0
        assert result.decode_seconds == 5.0
        assert result.decode_tokens_per_second == 0.4
        assert result.runtime_reported_tps == 999.0


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

    def test_smoke_persists_measured_artifacts_under_output_root(
        self, tmp_path, monkeypatch, capsys
    ):
        from ornith_mlx_eval import runner as runner_module
        from ornith_mlx_eval.cli import _cmd_smoke

        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
        monkeypatch.setattr(
            runner_module,
            "generate_with_mlx",
            lambda *args, **kwargs: _measured_generation(),
        )
        output_root = tmp_path / "smokes"
        result = _cmd_smoke(
            argparse.Namespace(
                model=FOUR_BIT_MODEL,
                max_tokens=32,
                temperature=None,
                top_p=None,
                top_k=None,
                seed=None,
                output_root=str(output_root),
                max_prompt_tokens=8192,
                max_kv_size=2048,
                allow_download=True,
                promotion_source=None,
            )
        )

        assert result == 0
        run_dirs = list(output_root.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "manifest.json").exists()
        assert (run_dirs[0] / "summary.json").exists()
        output = capsys.readouterr().out
        assert "Smoke status: PASS" in output
        assert "Decode tokens/second: 12.500" in output
        assert str(run_dirs[0]) in output


class TestMlxRunnerAdapter:
    def test_repeats_persist_identical_token_and_text_hash_evidence_from_one_load(
        self, tmp_path, monkeypatch
    ):
        from ornith_mlx_eval import runner as runner_module

        api = FakeApi()
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
        monkeypatch.setattr(
            runner_module,
            "_cached_snapshot_status",
            lambda *args: {"status": "hit-complete", "path": str(snapshot)},
        )

        run_dir = run_evaluation(
            RunOptions(
                runtime="mlx",
                suite="smoke",
                model=FOUR_BIT_MODEL,
                output_root=str(tmp_path / "runs"),
                limit=1,
                repeats=3,
                allow_download=True,
            ),
            mlx_api=api,
        )

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        assert api.load_calls == [(FOUR_BIT_MODEL, FOUR_BIT_SHA)]
        assert manifest["settings"]["repeats"] == 3
        assert row["response_sha256"] == hashlib.sha256(b"Paris").hexdigest()
        assert len(row["token_ids_sha256"]) == 64
        assert row["determinism"]["repeats"] == 3
        assert row["determinism"]["identical_token_ids"] is True
        assert row["determinism"]["identical_text"] is True
        assert len(row["determinism"]["per_repeat_hashes"]) == 3
        assert summary["determinism"]["status"] == "identical"
        assert "## Determinism" in report
        assert "Token IDs identical: `true`" in report

    def test_repeat_divergence_is_persisted_and_exits_nonzero(self, tmp_path, monkeypatch):
        from ornith_mlx_eval import runner as runner_module

        api = FakeApi()
        stream_calls = 0

        def divergent_stream(model, tokenizer, prompt, **kwargs):
            nonlocal stream_calls
            stream_calls += 1
            token = 100 + stream_calls
            yield FakeChunk(str(token), generation_tokens=1, token=token)

        api.stream_generate = divergent_stream
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
        monkeypatch.setattr(
            runner_module,
            "_cached_snapshot_status",
            lambda *args: {"status": "hit-complete", "path": str(snapshot)},
        )
        output_root = tmp_path / "runs"

        with pytest.raises(Exception, match="determinism divergence"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(output_root),
                    limit=1,
                    repeats=2,
                    allow_download=True,
                ),
                mlx_api=api,
            )

        run_dir = next(output_root.iterdir())
        row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert row["determinism"]["identical_token_ids"] is False
        assert summary["determinism"]["status"] == "divergent"
        assert any(error["type"] == "determinism_divergence" for error in row["errors"])

    def test_missing_real_token_metrics_persist_an_error_row_before_nonzero_exit(
        self, tmp_path, monkeypatch
    ):
        from ornith_mlx_eval import runner as runner_module

        api = FakeApi()

        class TextOnlyChunk:
            text = "Paris"

        api.stream_generate = lambda *args, **kwargs: iter([TextOnlyChunk()])
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
        monkeypatch.setattr(
            runner_module,
            "_cached_snapshot_status",
            lambda *args: {"status": "hit-complete", "path": str(snapshot)},
        )
        output_root = tmp_path / "runs"

        with pytest.raises(Exception, match="token metrics"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(output_root),
                    limit=1,
                    allow_download=True,
                ),
                mlx_api=api,
            )

        run_dir = next(output_root.iterdir())
        row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
        assert row["tokens"]["generated"] == 0
        assert any(error["type"] == "missing_metrics" for error in row["errors"])
        assert not (run_dir / ".incomplete").exists()

    @pytest.mark.parametrize("repeats", [0, 6])
    def test_repeats_are_bounded_before_profile_or_output(
        self, repeats, tmp_path, monkeypatch
    ):
        from ornith_mlx_eval import runner as runner_module

        profile_calls = []
        monkeypatch.setattr(
            runner_module,
            "run_profile",
            lambda **kwargs: profile_calls.append(kwargs) or _passing_profile(),
        )
        with pytest.raises(Exception, match="repeats"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(tmp_path / "runs"),
                    repeats=repeats,
                )
            )
        assert profile_calls == []
        assert not (tmp_path / "runs").exists()
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

    def test_invalid_generation_budget_fails_before_profile_or_output(
        self, tmp_path, monkeypatch
    ):
        from ornith_mlx_eval import runner as runner_module

        profile_calls = []
        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(
            runner_module,
            "run_profile",
            lambda **kwargs: profile_calls.append(kwargs) or _passing_profile(),
        )
        output_root = tmp_path / "runs"
        with pytest.raises(Exception, match="max-tokens must be greater than zero"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(output_root),
                    limit=1,
                    max_tokens=0,
                    allow_download=True,
                )
            )
        assert profile_calls == []
        assert not output_root.exists()

    def test_mlx_runtime_adapter_writes_artifacts_with_fake_generation(self, tmp_path, monkeypatch):
        from ornith_mlx_eval import runner as runner_module

        calls = []

        def fake_generate(model_id, revision, prompt, options):
            calls.append((model_id, revision, prompt, options.max_kv_size))
            return _measured_generation()

        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
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

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        assert manifest["runtime"]["kind"] == "mlx"
        assert manifest["model"]["revision"] == FOUR_BIT_SHA
        assert manifest["model"]["tokenizer_identity"].startswith("tokenizer.json:sha256:")
        assert manifest["model"]["chat_template_identity"].startswith("chat_template.jinja:sha256:")
        assert manifest["runtime"]["preflight_status"] == "pass"
        assert row["resources"]["peak_mlx_memory_bytes"] == 256_000_000
        assert row["timing"]["cold_load_seconds"] == 0.5
        assert row["timing"]["decode_tokens_per_second"] == 12.5
        assert row["resources"]["disk_free_before_bytes"] > 0
        assert summary["performance"]["generated_tokens"] == 2
        assert summary["resources"]["peak_mlx_memory_bytes"] == 256_000_000
        assert "Cold-load time seconds: `0.500000`" in report
        assert "Decode tokens per second: `12.500000`" in report
        assert "Disk free before bytes:" in report
        assert "Memory percent-free after:" in report
        assert calls[0][0] == FOUR_BIT_MODEL
        assert calls[0][1] == FOUR_BIT_SHA
        assert calls[0][3] == 2048

    def test_mlx_profile_failure_stops_before_run_directory(self, tmp_path, monkeypatch):
        from ornith_mlx_eval import runner as runner_module

        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(
            runner_module,
            "run_profile",
            lambda **kwargs: {
                "status": "fail",
                "checks": [
                    {
                        "name": "disk",
                        "status": "fail",
                        "details": {"free_bytes": 1, "required_bytes": 2},
                        "reason": "Insufficient free disk",
                    }
                ],
                "model": {
                    "status": "pass",
                    "details": {"sha": FOUR_BIT_SHA, "size_bytes": 1},
                },
            },
        )

        output_root = tmp_path / "runs"
        with pytest.raises(Exception, match="disk.*Insufficient free disk"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(output_root),
                    limit=1,
                    allow_download=True,
                )
            )
        assert not output_root.exists()

    def test_mlx_suite_validation_precedes_host_profile(self, tmp_path, monkeypatch):
        from ornith_mlx_eval import runner as runner_module

        bad_suite = tmp_path / "bad.json"
        bad_suite.write_text(
            '{"suite_id":"bad","suite_version":"1","cases":[]}',
            encoding="utf-8",
        )
        profile_calls = []
        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(
            runner_module,
            "run_profile",
            lambda **kwargs: profile_calls.append(kwargs) or _passing_profile(),
        )

        with pytest.raises(Exception, match="Suite validation failed"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite=str(bad_suite),
                    model=FOUR_BIT_MODEL,
                    output_root=str(tmp_path / "runs"),
                    limit=1,
                    allow_download=True,
                )
            )
        assert profile_calls == []

    def test_incorrect_model_answer_is_preserved_as_completed_case_failure(self, tmp_path, monkeypatch):
        from ornith_mlx_eval import runner as runner_module

        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
        monkeypatch.setattr(
            runner_module,
            "generate_with_mlx",
            lambda *args, **kwargs: _measured_generation("London"),
        )

        run_dir = run_evaluation(
            RunOptions(
                runtime="mlx",
                suite="smoke",
                model=FOUR_BIT_MODEL,
                output_root=str(tmp_path / "runs"),
                limit=1,
                allow_download=True,
            )
        )

        assert not (run_dir / ".incomplete").exists()
        row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert row["raw_response"] == "London"
        assert row["grade"]["passed"] is False
        assert summary["totals"]["failed"] == 1

    def test_resource_stop_writes_valid_stopped_artifacts_and_returns_nonzero(
        self, tmp_path, monkeypatch
    ):
        from ornith_mlx_eval import runner as runner_module

        stopped = _measured_generation()
        stopped = MlxGenerationResult(
            **{
                **stopped.__dict__,
                "swap_used_after_bytes": stopped.swap_used_before_bytes + 2 * 1024**3,
            }
        )
        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
        monkeypatch.setattr(runner_module, "generate_with_mlx", lambda *args, **kwargs: stopped)
        output_root = tmp_path / "runs"

        with pytest.raises(Exception, match="MLX resource stop"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(output_root),
                    limit=1,
                    allow_download=True,
                )
            )

        run_dir = next(output_root.iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "stopped"
        assert summary["status"] == "stopped"
        assert summary["totals"]["resource_stops"] == 1
        assert not (run_dir / ".incomplete").exists()

    def test_systemic_runtime_failure_preserves_specific_failure_json(
        self, tmp_path, monkeypatch
    ):
        from ornith_mlx_eval import runner as runner_module

        monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
        monkeypatch.setattr(runner_module, "run_profile", lambda **kwargs: _passing_profile())
        monkeypatch.setattr(
            runner_module,
            "generate_with_mlx",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                MlxSessionError("model load failed: unsupported architecture")
            ),
        )
        output_root = tmp_path / "runs"

        with pytest.raises(Exception, match="model load failed: unsupported architecture"):
            run_evaluation(
                RunOptions(
                    runtime="mlx",
                    suite="smoke",
                    model=FOUR_BIT_MODEL,
                    output_root=str(output_root),
                    limit=1,
                    allow_download=True,
                )
            )

        run_dir = next(output_root.iterdir())
        failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
        assert failure["status"] == "failed"
        assert "unsupported architecture" in failure["message"]
        assert (run_dir / ".incomplete").exists()

    def test_6bit_promotion_accepts_only_fresh_same_host_measured_4bit_smoke(
        self, tmp_path, monkeypatch
    ):
        run_dir = _create_passing_4bit_source(tmp_path, monkeypatch)
        manifest_path = run_dir / "manifest.json"
        validate_6bit_promotion_source(str(manifest_path))

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["environment"]["host"] = "different-host"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(MlxSessionError, match="different host"):
            validate_6bit_promotion_source(str(manifest_path))

    def test_6bit_promotion_rejects_stale_revision_and_package_mismatch(
        self, tmp_path, monkeypatch
    ):
        run_dir = _create_passing_4bit_source(tmp_path, monkeypatch)
        manifest_path = run_dir / "manifest.json"
        original = json.loads(manifest_path.read_text(encoding="utf-8"))

        stale = json.loads(json.dumps(original))
        stale["timestamp"] = (
            datetime.now(timezone.utc) - timedelta(hours=25)
        ).isoformat()
        manifest_path.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(MlxSessionError, match="older than 24 hours"):
            validate_6bit_promotion_source(str(manifest_path))

        wrong_revision = json.loads(json.dumps(original))
        wrong_revision["model"]["revision"] = "0" * 40
        manifest_path.write_text(json.dumps(wrong_revision), encoding="utf-8")
        with pytest.raises(MlxSessionError, match="completed 4bit smoke artifact"):
            validate_6bit_promotion_source(str(manifest_path))

        wrong_stack = json.loads(json.dumps(original))
        wrong_stack["environment"]["packages"]["mlx-lm"] = "0.0.0"
        manifest_path.write_text(json.dumps(wrong_stack), encoding="utf-8")
        with pytest.raises(MlxSessionError, match="package mismatch for mlx-lm"):
            validate_6bit_promotion_source(str(manifest_path))

    @pytest.mark.parametrize(
        ("resource_key", "unsafe_value", "message"),
        [
            (
                "peak_mlx_memory_bytes",
                int(8 * 1024**3 * 0.85),
                "85% of the MLX working-set limit",
            ),
            ("memory_pressure_after", 20, "memory pressure was unsafe"),
            ("swap_delta_bytes", 1024**3 + 1, "swap growth was unsafe"),
        ],
    )
    def test_6bit_promotion_enforces_resource_thresholds(
        self,
        tmp_path,
        monkeypatch,
        resource_key,
        unsafe_value,
        message,
    ):
        run_dir = _create_passing_4bit_source(tmp_path, monkeypatch)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["resources"][resource_key] = unsafe_value
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with pytest.raises(MlxSessionError, match=message):
            validate_6bit_promotion_source(str(run_dir / "manifest.json"))

    def test_6bit_promotion_rejects_projected_peak_above_working_set_budget(
        self, tmp_path, monkeypatch
    ):
        run_dir = _create_passing_4bit_source(tmp_path, monkeypatch)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["resources"]["peak_mlx_memory_bytes"] = 5 * 1024**3
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with pytest.raises(MlxSessionError, match="projected 6bit peak"):
            validate_6bit_promotion_source(str(run_dir / "manifest.json"))

    def test_6bit_fake_runtime_uses_exact_revision_after_promotion_gate(
        self, tmp_path, monkeypatch
    ):
        from ornith_mlx_eval import runner as runner_module

        source = _create_passing_4bit_source(tmp_path, monkeypatch)
        calls = []
        snapshot = tmp_path / "six-bit-cache" / SIX_BIT_SHA
        snapshot.mkdir(parents=True)
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "chat_template.jinja").write_text(
            "{{ messages }}", encoding="utf-8"
        )
        monkeypatch.setattr(
            runner_module,
            "_cached_snapshot_status",
            lambda model_id, revision: {
                "status": "hit-complete",
                "path": str(snapshot),
            },
        )
        monkeypatch.setattr(
            runner_module,
            "run_profile",
            lambda **kwargs: _passing_profile(
                model_id=SIX_BIT_MODEL,
                revision=SIX_BIT_SHA,
                size_bytes=8_188_404_909,
            ),
        )

        def fake_generate(model_id, revision, prompt, options):
            calls.append((model_id, revision))
            return _measured_generation(model_id=model_id, revision=revision)

        monkeypatch.setattr(runner_module, "generate_with_mlx", fake_generate)
        run_dir = run_evaluation(
            RunOptions(
                runtime="mlx",
                suite="smoke",
                model=SIX_BIT_MODEL,
                output_root=str(tmp_path / "six-bit-runs"),
                limit=1,
                allow_download=True,
                promotion_source=str(source / "manifest.json"),
            )
        )

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert calls == [(SIX_BIT_MODEL, SIX_BIT_SHA)]
        assert manifest["model"]["repo_id"] == SIX_BIT_MODEL
        assert manifest["model"]["revision"] == SIX_BIT_SHA
