"""Tests for the CLI module.

Coverage:
  VAL-CLI-001 – Top-level help lists the command surface
  VAL-CLI-002 – Command help is side-effect free
  VAL-CLI-003 – Invalid commands fail cleanly
  VAL-CLI-004 – Invalid arguments fail cleanly
  VAL-CLI-005 – Exit codes distinguish success from failure
  VAL-CLI-015 – Help documents safe default workflow
"""

import os
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

EXPECTED_COMMANDS = [
    "profile",
    "list-suites",
    "validate-suite",
    "smoke",
    "run",
    "report",
    "compare",
]

# Absolute path to the repo root so subprocess finds the venv regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli(args: list[str] | None = None, *, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run the ornith-mlx-eval console script with the given arguments."""
    cmd = [os.path.join(_REPO_ROOT, ".venv", "bin", "ornith-mlx-eval")]
    if args:
        cmd.extend(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd if cwd is not None else _REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# VAL-CLI-001 – Top-level help lists the command surface
# ---------------------------------------------------------------------------

class TestTopLevelHelp:
    """Top-level --help lists every expected command and exits cleanly."""

    def test_help_exits_zero(self):
        result = _cli(["--help"])
        assert result.returncode == 0, f"exit code {result.returncode}"

    def test_help_stderr_is_empty(self):
        result = _cli(["--help"])
        assert result.stderr == "", f"stderr not empty: {result.stderr!r}"

    @pytest.mark.parametrize("command", EXPECTED_COMMANDS)
    def test_help_lists_command(self, command):
        result = _cli(["--help"])
        assert command in result.stdout, f"missing command '{command}' in help"


# ---------------------------------------------------------------------------
# VAL-CLI-002 – Command help is side-effect free
# ---------------------------------------------------------------------------

class TestCommandHelp:
    """Every subcommand --help documents usage, exits 0, and performs no work."""

    @pytest.mark.parametrize("command", EXPECTED_COMMANDS)
    def test_command_help_exits_zero(self, command):
        result = _cli([command, "--help"])
        assert result.returncode == 0, f"{command} --help exit {result.returncode}"

    @pytest.mark.parametrize("command", EXPECTED_COMMANDS)
    def test_command_help_stderr_empty(self, command):
        result = _cli([command, "--help"])
        assert result.stderr == "", f"stderr not empty for {command}: {result.stderr!r}"

    @pytest.mark.parametrize("command", EXPECTED_COMMANDS)
    def test_command_help_has_usage_text(self, command):
        result = _cli([command, "--help"])
        assert result.stdout, f"no help output for {command}"
        assert "usage" in result.stdout.lower(), f"no usage text for {command}"

    def test_command_help_does_not_create_files(self, tmp_path):
        """Help invocations do not create output or model-cache files."""
        before = set(os.listdir(tmp_path))
        for cmd in EXPECTED_COMMANDS:
            result = _cli([cmd, "--help"], cwd=str(tmp_path))
            assert result.returncode == 0
        after = set(os.listdir(tmp_path))
        assert before == after, f"files changed after help: {after - before}"


# ---------------------------------------------------------------------------
# VAL-CLI-003 – Invalid commands fail cleanly
# ---------------------------------------------------------------------------

class TestInvalidCommands:
    """Unknown subcommands fail as user input errors without tracebacks."""

    def test_unknown_command_exits_nonzero(self):
        result = _cli(["notacommand"])
        assert result.returncode != 0

    def test_unknown_command_stderr_mentions_invalid_name(self):
        result = _cli(["notacommand"])
        assert result.stderr, "expected stderr output"
        lower = result.stderr.lower()
        assert "notacommand" in lower or "invalid" in lower or "usage" in lower, (
            f"stderr does not mention invalid command: {result.stderr!r}"
        )

    def test_unknown_command_no_traceback(self):
        result = _cli(["notacommand"])
        assert "Traceback (most recent call last)" not in result.stderr

    def test_unknown_command_no_files_created(self, tmp_path):
        before = set(os.listdir(tmp_path))
        _cli(["notacommand"], cwd=str(tmp_path))
        after = set(os.listdir(tmp_path))
        assert before == after, f"files changed after invalid command: {after - before}"


# ---------------------------------------------------------------------------
# VAL-CLI-004 – Invalid arguments fail cleanly
# ---------------------------------------------------------------------------

class TestInvalidArguments:
    """Missing required args, unknown flags, and invalid values fail cleanly."""

    def test_missing_required_arg_exits_nonzero(self):
        # validate-suite requires suite_path
        result = _cli(["validate-suite"])
        assert result.returncode != 0

    def test_missing_required_arg_stderr_explains(self):
        result = _cli(["validate-suite"])
        assert result.stderr, "expected stderr output"

    def test_unknown_flag_exits_nonzero(self):
        result = _cli(["--not-a-real-flag"])
        assert result.returncode != 0

    def test_unknown_flag_stderr(self):
        result = _cli(["--not-a-real-flag"])
        assert result.stderr, "expected stderr output"

    def test_invalid_choice_exits_nonzero(self):
        result = _cli(["run", "--runtime", "ollama"])
        assert result.returncode != 0

    def test_invalid_choice_stderr(self):
        result = _cli(["run", "--runtime", "ollama"])
        assert result.stderr, "expected stderr output"

    def test_argument_error_no_traceback(self):
        result = _cli(["validate-suite"])
        assert "Traceback (most recent call last)" not in result.stderr

    def test_invalid_args_no_benchmark_dir(self, tmp_path):
        """Invalid arguments do not create benchmark directory."""
        before = set(os.listdir(tmp_path))
        _cli(["validate-suite"], cwd=str(tmp_path))
        after = set(os.listdir(tmp_path))
        assert "benchmark_results" not in (after - before)


# ---------------------------------------------------------------------------
# VAL-CLI-005 – Exit codes distinguish success from failure
# ---------------------------------------------------------------------------

class TestExitCodes:
    """Success = 0; user/validation/resource/report/compare failures = nonzero."""

    def test_help_exits_zero(self):
        assert _cli(["--help"]).returncode == 0

    @pytest.mark.parametrize(
        "args",
        [
            ["profile"],
            ["list-suites"],
            # Commands that need positional args get a placeholder
            ["validate-suite", "dummy.json"],
            ["smoke", "--model", "dummy-model"],
            ["run"],
            ["report", "dummy_dir"],
            ["compare", "dummy_a", "dummy_b"],
        ],
    )
    def test_known_command_exits_zero(self, args):
        result = _cli(args)
        assert result.returncode == 0, f"'{' '.join(args)}' exit {result.returncode}"

    def test_invalid_command_exits_nonzero(self):
        assert _cli(["bad"]).returncode != 0

    def test_missing_required_args_exits_nonzero_report(self):
        assert _cli(["report"]).returncode != 0

    def test_missing_required_args_exits_nonzero_validate_suite(self):
        assert _cli(["validate-suite"]).returncode != 0


# ---------------------------------------------------------------------------
# VAL-CLI-015 – Help documents safe default workflow
# ---------------------------------------------------------------------------

class TestHelpDocumentsWorkflow:
    """Relevant commands expose --output-root, --suite, --limit, --runtime,
    --model, decoding/resource options, and --allow-mismatch in help."""

    def test_run_help_documents_output_root(self):
        result = _cli(["run", "--help"])
        assert "--output-root" in result.stdout

    def test_run_help_documents_suite(self):
        result = _cli(["run", "--help"])
        assert "--suite" in result.stdout

    def test_run_help_documents_limit(self):
        result = _cli(["run", "--help"])
        assert "--limit" in result.stdout

    def test_run_help_documents_runtime(self):
        result = _cli(["run", "--help"])
        assert "--runtime" in result.stdout

    def test_run_help_documents_model(self):
        result = _cli(["run", "--help"])
        assert "--model" in result.stdout

    def test_smoke_help_documents_model(self):
        result = _cli(["smoke", "--help"])
        assert "--model" in result.stdout

    def test_smoke_help_documents_output_root(self):
        result = _cli(["smoke", "--help"])
        assert "--output-root" in result.stdout

    def test_smoke_help_documents_decoding_options(self):
        result = _cli(["smoke", "--help"])
        for opt in ["--temperature", "--top-p", "--top-k", "--seed"]:
            assert opt in result.stdout, f"missing '{opt}' in smoke --help"

    def test_run_help_documents_decoding_options(self):
        result = _cli(["run", "--help"])
        for opt in ["--temperature", "--top-p", "--top-k", "--seed"]:
            assert opt in result.stdout, f"missing '{opt}' in run --help"

    def test_run_help_documents_resource_options(self):
        result = _cli(["run", "--help"])
        for opt in ["--max-tokens", "--max-prompt-tokens", "--max-kv-size"]:
            assert opt in result.stdout, f"missing '{opt}' in run --help"

    def test_compare_help_documents_allow_mismatch(self):
        result = _cli(["compare", "--help"])
        assert "--allow-mismatch" in result.stdout

    def test_compare_help_documents_output(self):
        result = _cli(["compare", "--help"])
        assert "--output" in result.stdout

    def test_profile_help_documents_model(self):
        result = _cli(["profile", "--help"])
        assert "--model" in result.stdout
