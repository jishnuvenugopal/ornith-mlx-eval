"""Fake-API tests for the warmed multi-trial throughput probe."""

from __future__ import annotations

import argparse
import json

import pytest

from ornith_mlx_eval.mlx_session import (
    FOUR_BIT_MODEL,
    FOUR_BIT_SHA,
    SIX_BIT_MODEL,
)
from tests.test_mlx_session import FakeApi, FakeChunk, _passing_profile


class PerfTokenizer:
    has_chat_template = False
    eos_token_ids = [0, 2]

    def encode(self, prompt):
        return list(range(12))


class PerfFakeApi(FakeApi):
    def __init__(self):
        super().__init__()
        self.stream_history = []

    def load(self, model_id, *, revision):
        self.mx.events.append("load")
        self.load_calls.append((model_id, revision))
        return "model", PerfTokenizer()

    def stream_generate(self, model, tokenizer, prompt, **kwargs):
        self.stream_history.append(kwargs)
        for index in range(kwargs["max_tokens"]):
            yield FakeChunk(
                "x",
                prompt_tokens=12,
                generation_tokens=index + 1,
                prompt_tps=80.0,
                generation_tps=222.0,
                token=100 + index,
            )


def _safe_probe():
    return {
        "disk_free_bytes": 50 * 1024**3,
        "memory_pressure": 75,
        "swap_used_bytes": 128 * 1024**2,
    }


def _prepare_fake_perf(tmp_path, monkeypatch):
    from ornith_mlx_eval import perf as perf_module
    from ornith_mlx_eval import runner as runner_module

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
    monkeypatch.setattr(
        perf_module,
        "_cached_snapshot_status",
        lambda *args: {"status": "hit-complete", "path": str(snapshot)},
    )
    return snapshot


def test_perf_reuses_one_load_discards_warmup_and_wires_eos_suppression(
    tmp_path, monkeypatch
):
    from ornith_mlx_eval.perf import PerfOptions, run_perf_probe

    _prepare_fake_perf(tmp_path, monkeypatch)
    api = PerfFakeApi()
    tick = -0.01

    def clock():
        nonlocal tick
        tick += 0.01
        return tick

    run_dir = run_perf_probe(
        PerfOptions(
            model=FOUR_BIT_MODEL,
            output_root=str(tmp_path / "perf-runs"),
            trials=3,
            decode_tokens=64,
            warmup_tokens=2,
            allow_download=True,
        ),
        api=api,
        clock=clock,
        resource_probe=_safe_probe,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    perf = json.loads((run_dir / "perf.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert api.load_calls == [(FOUR_BIT_MODEL, FOUR_BIT_SHA)]
    assert [call["max_tokens"] for call in api.stream_history] == [2, 64, 64, 64]
    assert all(len(call["logits_processors"]) == 1 for call in api.stream_history)
    assert manifest["settings"]["warmup_tokens"] == 2
    assert len(perf["trials"]) == 3
    assert all(trial["generated_tokens"] == 64 for trial in perf["trials"])
    assert perf["aggregate"]["median_tps"] == pytest.approx(100.0)
    assert perf["aggregate"]["cv"] == pytest.approx(0.0)
    assert perf["headline"]["status"] == "publishable"
    assert perf["headline"]["decode_tokens_per_second"] == pytest.approx(100.0)
    assert "## Throughput Probe" in report
    assert "Runtime-reported TPS" in report


@pytest.mark.parametrize(
    ("trial_tps", "trials", "decode_tokens", "reason"),
    [
        ([100.0, 100.0], 2, 128, "at least 3 trials"),
        ([100.0, 100.0, 100.0], 3, 32, "at least 64 decode tokens"),
        ([80.0, 100.0, 120.0], 3, 128, "CV exceeds 10%"),
    ],
)
def test_perf_headline_is_refused_for_weak_or_unstable_samples(
    trial_tps, trials, decode_tokens, reason
):
    from ornith_mlx_eval.perf import aggregate_perf_trials

    result = aggregate_perf_trials(
        trial_tps, trials=trials, decode_tokens=decode_tokens
    )
    assert result["headline"]["status"] == "unstable"
    assert result["headline"]["decode_tokens_per_second"] is None
    assert reason in result["headline"]["reasons"]


def test_perf_gates_match_smoke_for_opt_in_and_rejected_models(tmp_path, monkeypatch):
    from ornith_mlx_eval.perf import PerfOptions, run_perf_probe

    monkeypatch.delenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", raising=False)
    with pytest.raises(Exception, match="explicit opt-in"):
        run_perf_probe(
            PerfOptions(model=FOUR_BIT_MODEL, output_root=str(tmp_path / "a"))
        )
    with pytest.raises(Exception, match="Unsupported MLX model"):
        run_perf_probe(
            PerfOptions(
                model="mlx-community/Ornith-1.0-9B-8bit",
                output_root=str(tmp_path / "b"),
            )
        )
    monkeypatch.setenv("ORNITH_MLX_ALLOW_MODEL_DOWNLOAD", "1")
    with pytest.raises(Exception, match="promotion-source"):
        run_perf_probe(
            PerfOptions(
                model=SIX_BIT_MODEL,
                output_root=str(tmp_path / "c"),
                allow_download=True,
            )
        )
    assert not any((tmp_path / name).exists() for name in ["a", "b", "c"])


def test_perf_cli_prints_unstable_instead_of_a_headline(monkeypatch, capsys):
    from ornith_mlx_eval import cli as cli_module

    monkeypatch.setattr(
        "ornith_mlx_eval.perf.run_perf_probe",
        lambda options: "/tmp/perf-run",
    )
    monkeypatch.setattr(
        "ornith_mlx_eval.perf.load_perf_artifact",
        lambda run_dir: {
            "headline": {
                "status": "unstable",
                "decode_tokens_per_second": None,
                "reasons": ["CV exceeds 10%"],
            },
            "aggregate": {"median_tps": 95.0, "cv": 0.12},
        },
    )
    result = cli_module._cmd_perf(
        argparse.Namespace(
            model=FOUR_BIT_MODEL,
            trials=3,
            decode_tokens=128,
            warmup_tokens=16,
            seed=42,
            output_root="benchmark_results",
            max_prompt_tokens=8192,
            max_kv_size=4096,
            allow_download=True,
            promotion_source=None,
        )
    )
    output = capsys.readouterr().out
    assert result == 0
    assert "unstable" in output
    assert "Headline decode tokens/second" not in output
