"""End-to-end tests for mock runs, persisted reports, and comparison.

Coverage:
  VAL-CLI-011/012/013/014/017/018/020
  VAL-EVAL-027/030
  VAL-RESULTS-001 through VAL-RESULTS-029
  VAL-CROSS-004 through VAL-CROSS-010
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLI = _REPO_ROOT / ".venv" / "bin" / "ornith-mlx-eval"


def _cli(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_CLI), *args],
        cwd=str(cwd or _REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_mock(output_root: Path, *extra: str) -> Path:
    result = _cli([
        "run",
        "--runtime",
        "mock",
        "--suite",
        "smoke",
        "--output-root",
        str(output_root),
        *extra,
    ])
    assert result.returncode == 0, result.stderr
    match = re.search(r"Run directory:\s*(.+)", result.stdout)
    assert match, result.stdout
    return Path(match.group(1).strip())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestMockRunArtifacts:
    def test_mock_run_creates_one_isolated_run_directory(self, tmp_path):
        output_root = tmp_path / "runs"
        result = _cli([
            "run",
            "--runtime",
            "mock",
            "--suite",
            "smoke",
            "--output-root",
            str(output_root),
        ])

        assert result.returncode == 0, result.stderr
        children = list(output_root.iterdir())
        assert len(children) == 1
        assert children[0].is_dir()
        assert "Run directory:" in result.stdout

    def test_successful_run_writes_required_parseable_artifacts(self, tmp_path):
        run_dir = _run_mock(tmp_path / "runs")

        for name in ["manifest.json", "results.jsonl", "summary.json", "report.md"]:
            assert (run_dir / name).exists(), name

        manifest = _load_json(run_dir / "manifest.json")
        rows = _load_jsonl(run_dir / "results.jsonl")
        summary = _load_json(run_dir / "summary.json")
        report = (run_dir / "report.md").read_text(encoding="utf-8")

        assert manifest["schema_version"] == "ornith-result-v1"
        assert manifest["status"] == "completed"
        assert manifest["runtime"]["kind"] == "mock"
        assert manifest["suite"]["suite_id"] == "smoke"
        assert manifest["suite"]["suite_hash"]
        assert manifest["suite"]["prompt_template_hash"]
        assert manifest["settings"]["seed"] == 42
        assert manifest["settings"]["concurrency"] == 1
        assert len(rows) == 5
        assert summary["totals"]["scored"] == 5
        assert summary["totals"]["passed"] >= 4
        assert "# Ornith MLX Eval Report" in report

    def test_results_rows_are_in_suite_order_and_record_prompt_hashes(self, tmp_path):
        run_dir = _run_mock(tmp_path / "runs")
        rows = _load_jsonl(run_dir / "results.jsonl")

        assert [row["case_id"] for row in rows] == [
            "smoke-fact-001",
            "smoke-fact-002",
            "smoke-numeric-001",
            "smoke-reasoning-001",
            "smoke-code-001",
        ]
        assert all(row["scored"] is True for row in rows)
        assert all(len(row["prompt_hash"]) == 64 for row in rows)
        assert all("raw_response" in row for row in rows)
        assert all("parse" in row for row in rows)
        assert all("grade" in row for row in rows)

    def test_hidden_expected_answer_metadata_is_not_serialized(self, tmp_path):
        run_dir = _run_mock(tmp_path / "runs")
        combined = "\n".join(
            [
                (run_dir / "manifest.json").read_text(encoding="utf-8"),
                (run_dir / "results.jsonl").read_text(encoding="utf-8"),
                (run_dir / "summary.json").read_text(encoding="utf-8"),
                (run_dir / "report.md").read_text(encoding="utf-8"),
            ]
        )

        assert '"hidden_answer"' not in combined
        assert '"expected_answer"' not in combined

    def test_limit_marks_run_smoke_only_and_limits_scored_cases(self, tmp_path):
        run_dir = _run_mock(tmp_path / "runs", "--limit", "2")

        manifest = _load_json(run_dir / "manifest.json")
        rows = _load_jsonl(run_dir / "results.jsonl")
        summary = _load_json(run_dir / "summary.json")

        assert len(rows) == 2
        assert manifest["settings"]["limit"] == 2
        assert manifest["classification"] == "smoke-only"
        assert summary["classification"] == "smoke-only"
        assert summary["smoke_only"] is True

    def test_non_positive_limit_fails_before_creating_run_dir(self, tmp_path):
        output_root = tmp_path / "runs"
        result = _cli([
            "run",
            "--runtime",
            "mock",
            "--suite",
            "smoke",
            "--limit",
            "0",
            "--output-root",
            str(output_root),
        ])

        assert result.returncode != 0
        assert "limit" in result.stderr.lower()
        assert not output_root.exists()

    def test_invalid_suite_fails_before_runtime_work(self, tmp_path):
        bad_suite = tmp_path / "bad.json"
        bad_suite.write_text('{"suite_id": "bad", "suite_version": "1", "cases": []}', encoding="utf-8")
        output_root = tmp_path / "runs"

        result = _cli([
            "run",
            "--runtime",
            "mock",
            "--suite",
            str(bad_suite),
            "--output-root",
            str(output_root),
        ])

        assert result.returncode != 0
        assert "suite validation failed" in result.stderr.lower()
        assert not output_root.exists()


class TestReportCommand:
    def test_report_regenerates_from_persisted_files_only(self, tmp_path):
        run_dir = _run_mock(tmp_path / "runs")
        report_path = run_dir / "report.md"
        report_path.unlink()

        result = _cli(["report", str(run_dir)])

        assert result.returncode == 0, result.stderr
        assert report_path.exists()
        report = report_path.read_text(encoding="utf-8")
        for section in [
            "## Environment",
            "## Run Settings",
            "## Score Summary",
            "## Failures, Skips, And Errors",
            "## Performance",
            "## Resources",
            "## Caveats",
        ]:
            assert section in report

    def test_report_refuses_missing_or_corrupt_inputs(self, tmp_path):
        missing = tmp_path / "missing-run"
        result = _cli(["report", str(missing)])
        assert result.returncode != 0
        assert "manifest.json" in result.stderr or "run directory" in result.stderr

        run_dir = _run_mock(tmp_path / "runs")
        (run_dir / "summary.json").write_text("{not-json", encoding="utf-8")
        result = _cli(["report", str(run_dir)])
        assert result.returncode != 0
        assert "summary.json" in result.stderr

    def test_repeated_report_rendering_is_deterministic(self, tmp_path):
        run_dir = _run_mock(tmp_path / "runs")
        first = (run_dir / "report.md").read_text(encoding="utf-8")
        result = _cli(["report", str(run_dir)])
        assert result.returncode == 0, result.stderr
        second = (run_dir / "report.md").read_text(encoding="utf-8")
        assert first == second


class TestCompareCommand:
    def test_comparable_runs_produce_compare_report(self, tmp_path):
        root = tmp_path / "runs"
        run_a = _run_mock(root)
        run_b = _run_mock(root)

        result = _cli(["compare", str(run_a), str(run_b)])

        assert result.returncode == 0, result.stderr
        output_path = root / f"compare_{run_a.name}_vs_{run_b.name}.md"
        assert output_path.exists()
        text = output_path.read_text(encoding="utf-8")
        assert "# Ornith MLX Eval Comparison" in text
        assert run_a.name in text
        assert run_b.name in text

    def test_compare_refuses_fixed_invariant_mismatch_by_default(self, tmp_path):
        root = tmp_path / "runs"
        run_a = _run_mock(root)
        run_b = _run_mock(root, "--seed", "7")

        result = _cli(["compare", str(run_a), str(run_b)])

        assert result.returncode != 0
        assert "seed" in result.stderr.lower()

    def test_compare_allow_mismatch_is_qualitative(self, tmp_path):
        root = tmp_path / "runs"
        run_a = _run_mock(root)
        run_b = _run_mock(root, "--seed", "7")

        result = _cli(["compare", str(run_a), str(run_b), "--allow-mismatch"])

        assert result.returncode == 0, result.stderr
        output_path = root / f"compare_{run_a.name}_vs_{run_b.name}.md"
        text = output_path.read_text(encoding="utf-8")
        assert "Qualitative comparison only" in text
        assert "seed" in text

    def test_compare_refuses_bad_inputs_before_mismatch_analysis(self, tmp_path):
        result = _cli(["compare", str(tmp_path / "a"), str(tmp_path / "b")])
        assert result.returncode != 0
        assert "manifest.json" in result.stderr or "run directory" in result.stderr

    def test_compare_explicit_output_path_is_respected(self, tmp_path):
        root = tmp_path / "runs"
        run_a = _run_mock(root)
        run_b = _run_mock(root)
        output = tmp_path / "custom_compare.md"

        result = _cli(["compare", str(run_a), str(run_b), "--output", str(output)])

        assert result.returncode == 0, result.stderr
        assert output.exists()
        assert "# Ornith MLX Eval Comparison" in output.read_text(encoding="utf-8")
