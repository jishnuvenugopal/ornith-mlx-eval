"""Public repository readiness checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ornith_mlx_eval import __version__


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_readme_documents_safe_setup_and_workflows():
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "/opt/homebrew/bin/python3.12 -m venv .venv",
        ".venv/bin/python -m pip install -e '.[dev]'",
        ".venv/bin/python -m pytest -q",
        "ornith-mlx-eval run --runtime mock --suite smoke",
        "ORNITH_MLX_ALLOW_MODEL_DOWNLOAD=1",
        "--allow-download",
        "benchmark_results/",
        "plan.md",
        "status.md",
    ]
    for text in required:
        assert text in readme


def test_local_only_and_generated_files_are_not_tracked():
    result = _git(["ls-files"])
    assert result.returncode == 0, result.stderr
    tracked = set(result.stdout.splitlines())

    forbidden_exact = {
        "plan.md",
        "status.md",
        "whatisdone.md",
        "whatisleft.md",
        ".DS_Store",
    }
    assert not forbidden_exact.intersection(tracked)
    assert not any(path.startswith("benchmark_results/") for path in tracked)
    assert not any(path.startswith(".venv/") for path in tracked)
    assert not any("huggingface" in path.lower() for path in tracked)


def test_dependency_pins_are_exact():
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for pin in [
        "mlx==0.31.2",
        "mlx-lm==0.31.3",
        "transformers==5.0.0",
        "huggingface_hub==1.22.0",
        "numpy==2.5.1",
        "pytest==9.1.1",
        "jsonschema==4.26.0",
    ]:
        assert pin in pyproject


def test_package_and_project_versions_match_v020():
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert __version__ == "0.2.0"
    assert 'version = "0.2.0"' in pyproject


def test_public_tree_contains_readme_and_no_upstream_copied_artifacts():
    result = _git(["ls-files", "--cached", "--others", "--exclude-standard"])
    assert result.returncode == 0, result.stderr
    tracked = result.stdout.splitlines()
    assert "README.md" in tracked
    assert not any("ornith-eval" in path.lower() for path in tracked)

    suite = (_REPO_ROOT / "suites" / "smoke.json").read_text(encoding="utf-8")
    assert "jikkuatwork" not in suite
    assert "upstream" not in suite.lower()
