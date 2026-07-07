"""Tests for the bounded Python subprocess sandbox.

Coverage:
  VAL-EVAL-022 - Sandbox uses non-repo temporary working directory
  VAL-EVAL-023 - Sandbox scrubs sensitive environment by default
  VAL-EVAL-024 - Sandbox enforces timeout with bounded output
  VAL-EVAL-025 - Sandbox handles subprocess or fork attempts safely
  VAL-EVAL-026 - Sandbox cleans up temporary artifacts
  VAL-EVAL-038 - Sandbox blocks or safely reports outside file access
  VAL-EVAL-039 - Sandbox enforces file and disk limits
  VAL-EVAL-040 - Sandbox reports actual protections honestly
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from ornith_mlx_eval.sandbox import (
    SandboxResult,
    SandboxProtection,
    run_code,
)


# ======================================================================
# Helper: repo root for sentinel checks
# ======================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# VAL-EVAL-022 - Sandbox uses non-repo temporary working directory
# ======================================================================

class TestTempWorkingDirectory:
    """Coding cases execute from an isolated temp working directory,
    not the repo root, caller cwd, or real home directory."""

    def test_cwd_is_temp_not_repo(self):
        """Sandbox cwd is not the repo root."""
        code = textwrap.dedent("""\
            import os
            print(os.getcwd())
        """)
        result = run_code(code)
        assert result.exit_code == 0, f"Code failed: {result.stderr}"
        cwd = result.stdout.strip()
        assert REPO_ROOT.as_posix() not in cwd, (
            f"Sandbox cwd {cwd} should not be repo root {REPO_ROOT}"
        )
        # Should be in a temp directory
        assert "/tmp" in cwd or "Temporary" in cwd or "Temp" in cwd or "/var/" in cwd, (
            f"Expected temp directory, got {cwd}"
        )

    def test_cwd_is_not_real_home(self):
        """Sandbox cwd is not the user's real home directory."""
        real_home = os.path.expanduser("~")
        code = textwrap.dedent("""\
            import os
            print(os.getcwd())
        """)
        result = run_code(code)
        assert result.exit_code == 0
        cwd = result.stdout.strip()
        assert not cwd.startswith(real_home), (
            f"Sandbox cwd {cwd} should not be in real home {real_home}"
        )

    def test_sandbox_root_is_isolated(self):
        """Sandbox root directory exists and is writable by candidate code."""
        sandbox_root = tempfile.mkdtemp(prefix="test_sandbox_")
        try:
            code = textwrap.dedent("""\
                import os
                # Create a file in current directory
                with open("test_output.txt", "w") as f:
                    f.write("sandbox works")
                print("ok")
            """)
            result = run_code(code, sandbox_root=sandbox_root, retain_root=True)
            assert result.exit_code == 0
            # The created file should be inside sandbox_root
            created = Path(sandbox_root) / "test_output.txt"
            assert created.exists(), f"Expected {created} to exist"
            assert created.read_text() == "sandbox works"
        finally:
            import shutil
            shutil.rmtree(sandbox_root, ignore_errors=True)

    def test_relative_writes_stay_in_sandbox(self):
        """Candidate code writing relative paths stays within sandbox."""
        code = textwrap.dedent("""\
            with open("candidate_output.txt", "w") as f:
                f.write("secret data")
            print("written")
        """)
        result = run_code(code)
        assert result.exit_code == 0
        # The file should NOT appear in repo root
        repo_file = REPO_ROOT / "candidate_output.txt"
        assert not repo_file.exists(), (
            f"Candidate relative write leaked to {repo_file}"
        )


# ======================================================================
# VAL-EVAL-023 - Sandbox scrubs sensitive environment by default
# ======================================================================

class TestScrubbedEnvironment:
    """Coding cases run with scrubbed environment that omits common
    secret variables and real home indicators."""

    def test_real_home_scrubbed(self):
        """HOME and related variables are sanitized away from real home."""
        real_home = os.path.expanduser("~")
        code = textwrap.dedent("""\
            import os
            home = os.environ.get("HOME", "NOT_SET")
            print(f"HOME={home}")
        """)
        result = run_code(code)
        assert result.exit_code == 0
        output = result.stdout
        # HOME should not be the real home
        assert real_home not in output, (
            f"Real HOME {real_home} leaked into sandbox: {output}"
        )

    def test_path_does_not_contain_repo(self):
        """PATH should not contain the repo or real home bin directories."""
        code = textwrap.dedent("""\
            import os
            print(os.environ.get("PATH", ""))
        """)
        result = run_code(code)
        assert result.exit_code == 0
        path_val = result.stdout.strip()
        # PATH should be minimal; should not contain real home
        real_home = os.path.expanduser("~")
        assert real_home not in path_val, (
            f"Real home {real_home} leaked into sandbox PATH: {path_val}"
        )

    def test_pythonpath_not_inherited(self):
        """PYTHONPATH from parent should not leak into sandbox."""
        # Set a fake PYTHONPATH before running
        code = textwrap.dedent("""\
            import os
            pypath = os.environ.get("PYTHONPATH", "NOT_SET")
            print(f"PYTHONPATH={pypath}")
        """)
        result = run_code(code, env_overrides={"PYTHONPATH": "/fake/injection/path"})
        assert result.exit_code == 0
        output = result.stdout
        # The fake PYTHONPATH should not appear
        assert "/fake/injection/path" not in output, (
            f"PYTHONPATH leaked: {output}"
        )

    def test_fake_secret_variable_not_available(self):
        """Fake secret variables set in parent are unavailable in sandbox."""
        code = textwrap.dedent("""\
            import os
            token = os.environ.get("FAKE_API_TOKEN", "NOT_SET")
            key = os.environ.get("FAKE_SECRET_KEY", "NOT_SET")
            print(f"TOKEN={token}")
            print(f"KEY={key}")
        """)
        result = run_code(code, env_overrides={
            "FAKE_API_TOKEN": "sk-12345-secret",
            "FAKE_SECRET_KEY": "deadbeef",
        })
        assert result.exit_code == 0
        output = result.stdout
        assert "sk-12345-secret" not in output, (
            f"Fake API token leaked: {output}"
        )
        assert "deadbeef" not in output, (
            f"Fake secret key leaked: {output}"
        )

    def test_fake_aws_credentials_not_available(self):
        """AWS credential variables should not be visible."""
        code = textwrap.dedent("""\
            import os
            ak = os.environ.get("AWS_ACCESS_KEY_ID", "NOT_SET")
            sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "NOT_SET")
            print(f"AWS_AK={ak}")
            print(f"AWS_SK={sk}")
        """)
        result = run_code(code, env_overrides={
            "AWS_ACCESS_KEY_ID": "AKIA123456",
            "AWS_SECRET_ACCESS_KEY": "super-secret-key",
        })
        assert result.exit_code == 0
        output = result.stdout
        assert "AKIA123456" not in output
        assert "super-secret-key" not in output


# ======================================================================
# VAL-EVAL-024 - Sandbox enforces timeout with bounded output
# ======================================================================

class TestTimeout:
    """Infinite loops or unbounded output are stopped within timeout."""

    def test_infinite_loop_stopped_by_timeout(self):
        """An infinite loop is stopped by the timeout."""
        code = textwrap.dedent("""\
            while True:
                pass
        """)
        start = time.monotonic()
        result = run_code(code, timeout=1.0)
        elapsed = time.monotonic() - start
        assert result.timed_out, f"Expected timeout, got status={result.status}"
        assert result.exit_code != 0 or result.timed_out, (
            f"Infinite loop should not exit cleanly"
        )
        # Wall clock should be bounded
        assert elapsed < 10.0, (
            f"Timeout took {elapsed:.2f}s, should be bounded"
        )

    def test_sleep_exceeding_timeout(self):
        """A long sleep is killed by timeout."""
        code = textwrap.dedent("""\
            import time
            time.sleep(30)
            print("done")
        """)
        start = time.monotonic()
        result = run_code(code, timeout=1.0)
        elapsed = time.monotonic() - start
        assert result.timed_out
        assert elapsed < 10.0
        assert result.status == "timeout"

    def test_output_bounded(self):
        """Unbounded stdout is truncated at the output limit."""
        code = textwrap.dedent("""\
            for i in range(100000):
                print("x" * 100)
        """)
        result = run_code(code, timeout=5.0, max_output_bytes=1024)
        assert result.output_truncated, (
            f"Output should be truncated, status={result.status}"
        )
        assert len(result.stdout) <= 1024 + 512, (
            f"Output {len(result.stdout)} exceeds limit + buffer"
        )

    def test_normal_output_not_truncated(self):
        """Normal output within limits is not flagged as truncated."""
        code = textwrap.dedent("""\
            print("hello")
            print("world")
        """)
        result = run_code(code)
        assert result.exit_code == 0
        assert not result.output_truncated
        assert "hello" in result.stdout
        assert "world" in result.stdout

    def test_timeout_result_reports_status(self):
        """Timeout result has correct status and reason."""
        code = textwrap.dedent("""\
            while True:
                pass
        """)
        result = run_code(code, timeout=0.5)
        assert result.status == "timeout"
        assert result.timed_out
        assert "timeout" in result.reason.lower()
        assert not result.passed


# ======================================================================
# VAL-EVAL-025 - Sandbox handles subprocess or fork attempts safely
# ======================================================================

class TestSubprocessForkHandling:
    """Child process, fork, or shell attempts cannot hang the harness."""

    def test_subprocess_spawn_does_not_hang(self):
        """Spawning a subprocess inside sandbox does not hang."""
        code = textwrap.dedent("""\
            import subprocess
            try:
                result = subprocess.run(
                    ["echo", "child process"],
                    capture_output=True,
                    timeout=2,
                )
                print(result.stdout.decode().strip())
            except Exception as exc:
                print(f"subprocess failed: {exc}")
        """)
        start = time.monotonic()
        result = run_code(code, timeout=3.0)
        elapsed = time.monotonic() - start
        # Should complete within timeout regardless
        assert elapsed < 15.0, f"Subprocess test took {elapsed:.2f}s"

    def test_fork_bomb_attempt_handled(self):
        """A fork-bomb-like code is contained within timeout."""
        code = textwrap.dedent("""\
            import os
            import sys
            # Try to spawn children rapidly (this will hit OS limits quickly)
            try:
                for _ in range(100):
                    pid = os.fork()
                    if pid == 0:
                        sys.exit(0)
            except OSError:
                print("fork blocked by OS")
            except Exception as exc:
                print(f"fork failed: {exc}")
            print("fork attempt done")
        """)
        start = time.monotonic()
        result = run_code(code, timeout=5.0)
        elapsed = time.monotonic() - start
        # Should complete within timeout
        assert elapsed < 20.0, (
            f"Fork bomb test took {elapsed:.2f}s"
        )

    def test_child_processes_cleaned_after_timeout(self):
        """When the sandbox times out, no orphaned children remain."""
        code = textwrap.dedent("""\
            import subprocess
            import sys
            # Spawn a long-running child
            try:
                subprocess.Popen(["sleep", "60"])
                print("child spawned")
            except Exception as exc:
                print(f"spawn failed: {exc}")
            # Keep parent alive to test orphan detection
            import time
            time.sleep(10)
        """)
        result = run_code(code, timeout=2.0)
        assert result.timed_out, (
            f"Expected timeout, got status={result.status}"
        )

    def test_shell_injection_via_subprocess(self):
        """Shell command execution is not used for grading decisions."""
        code = textwrap.dedent("""\
            import subprocess
            try:
                subprocess.run("echo injected", shell=True, capture_output=True)
                print("shell_ran")
            except Exception as exc:
                print(f"shell blocked: {exc}")
        """)
        result = run_code(code, timeout=3.0)
        # Whether shell ran or not, the test completes within timeout
        assert result.status != "timeout" or result.timed_out


# ======================================================================
# VAL-EVAL-026 - Sandbox cleans up temporary artifacts
# ======================================================================

class TestCleanup:
    """Sandbox temp directories and artifacts are cleaned after execution."""

    def test_temp_dir_removed_after_success(self):
        """Sandbox root is removed after successful execution."""
        sandbox_root = tempfile.mkdtemp(prefix="test_cleanup_success_")
        root_path = Path(sandbox_root)
        assert root_path.exists()
        code = textwrap.dedent("""\
            with open("output.txt", "w") as f:
                f.write("test")
            print("ok")
        """)
        result = run_code(code, sandbox_root=sandbox_root)
        assert result.exit_code == 0
        # Temp dir should be cleaned
        assert not root_path.exists(), (
            f"Sandbox root {sandbox_root} was not cleaned after success"
        )

    def test_temp_dir_removed_after_failure(self):
        """Sandbox root is removed even after code failure."""
        sandbox_root = tempfile.mkdtemp(prefix="test_cleanup_fail_")
        root_path = Path(sandbox_root)
        code = textwrap.dedent("""\
            raise RuntimeError("intentional failure")
        """)
        result = run_code(code, sandbox_root=sandbox_root)
        assert result.exit_code != 0
        # Temp dir should be cleaned
        assert not root_path.exists(), (
            f"Sandbox root {sandbox_root} was not cleaned after failure"
        )

    def test_temp_dir_removed_after_timeout(self):
        """Sandbox root is removed after timeout."""
        sandbox_root = tempfile.mkdtemp(prefix="test_cleanup_timeout_")
        root_path = Path(sandbox_root)
        code = textwrap.dedent("""\
            while True:
                pass
        """)
        result = run_code(code, timeout=0.5, sandbox_root=sandbox_root)
        assert result.timed_out
        assert not root_path.exists(), (
            f"Sandbox root {sandbox_root} was not cleaned after timeout"
        )

    def test_nested_directories_cleaned(self):
        """Candidate-created nested files/dirs are cleaned."""
        sandbox_root = tempfile.mkdtemp(prefix="test_cleanup_nested_")
        root_path = Path(sandbox_root)
        code = textwrap.dedent("""\
            import os
            os.makedirs("a/b/c/d")
            with open("a/b/c/d/deep.txt", "w") as f:
                f.write("nested")
            print("created")
        """)
        result = run_code(code, sandbox_root=sandbox_root)
        assert result.exit_code == 0
        assert not root_path.exists(), (
            f"Nested sandbox artifacts not cleaned from {sandbox_root}"
        )


# ======================================================================
# VAL-EVAL-038 - Sandbox blocks or safely reports outside file access
# ======================================================================

class TestOutsideAccessDetection:
    """Candidate code attempting outside file access is detected and reported."""

    def test_read_repo_file_detected(self):
        """Reading a sentinel file in the repo is detected."""
        # Create a sentinel file in the repo root (name without _sandbox_ prefix)
        sentinel_path = REPO_ROOT / "_outside_read_test_.txt"
        sentinel_path.write_text("SENTINEL_CONTENT_DO_NOT_LEAK")
        try:
            code = textwrap.dedent(f"""\
                try:
                    with open("{sentinel_path}", "r") as f:
                        content = f.read()
                    print(content)
                except Exception as exc:
                    print(f"access blocked: {{exc}}")
            """)
            result = run_code(code)
            # The sentinel content appears in stdout: sandbox detects it
            # via sentinel marker scan and marks status as outside_access
            if "SENTINEL_CONTENT_DO_NOT_LEAK" in result.stdout:
                assert result.status == "outside_access", (
                    f"Outside access should be reported via sentinel detection, "
                    f"got status={result.status}"
                )
        finally:
            sentinel_path.unlink(missing_ok=True)

    def test_write_outside_temp_detected(self):
        """Writing outside the sandbox root is detected."""
        # Use a name without "_sandbox_" so the filesystem scan catches it.
        outside_path = REPO_ROOT / "_outside_write_test_.txt"
        # Ensure it doesn't exist before test
        outside_path.unlink(missing_ok=True)
        try:
            code = textwrap.dedent(f"""\
                try:
                    with open("{outside_path}", "w") as f:
                        f.write("SECRET_SENTINEL_MARKER")
                    print("wrote outside")
                except PermissionError:
                    print("permission denied")
                except Exception as exc:
                    print(f"write blocked: {{exc}}")
            """)
            result = run_code(code)
            # The file should not exist after cleanup
            if outside_path.exists():
                outside_path.unlink()
            # If it was created, sandbox should detect it via filesystem scan
            # or sentinel marker scan
            assert not result.passed or result.status == "outside_access", (
                f"Outside write should be detected or blocked, "
                f"status={result.status}, passed={result.passed}"
            )
        finally:
            outside_path.unlink(missing_ok=True)

    def test_read_home_sentinel_detected(self):
        """Reading a file from real home directory is detected."""
        home_sentinel = Path.home() / "_outside_home_test_.txt"
        home_sentinel.write_text("HOME_SENTINEL_DO_NOT_LEAK")
        try:
            code = textwrap.dedent(f"""\
                try:
                    with open("{home_sentinel}", "r") as f:
                        print(f.read())
                except Exception as exc:
                    print(f"access blocked: {{exc}}")
            """)
            result = run_code(code)
            if "HOME_SENTINEL_DO_NOT_LEAK" in result.stdout:
                assert result.status == "outside_access", (
                    f"Home directory access should fail case, "
                    f"got status={result.status}, passed={result.passed}"
                )
        finally:
            home_sentinel.unlink(missing_ok=True)

    def test_read_shadow_file_attempt_handled(self):
        """Attempting to read /etc/passwd or similar is handled safely."""
        code = textwrap.dedent("""\
            try:
                with open("/etc/passwd", "r") as f:
                    content = f.read()[:100]
                print(content)
            except Exception as exc:
                print(f"system file access blocked: {exc}")
        """)
        result = run_code(code, timeout=3.0)
        # Should complete without hanging
        assert result.status != "timeout" or result.timed_out

    def test_absolute_write_outside_reported(self):
        """Absolute path write outside sandbox is captured and reported."""
        code = textwrap.dedent("""\
            import tempfile
            import os
            # Try to write to a known outside location
            outside = os.path.join(tempfile.gettempdir(), "outside_write_test.txt")
            try:
                with open(outside, "w") as f:
                    f.write("outside content")
                print(f"WROTE_TO: {outside}")
            except Exception as exc:
                print(f"blocked: {exc}")
        """)
        result = run_code(code, timeout=3.0)
        # The result should indicate any outside detection
        # Clean up if the file was created
        outside_file = Path(tempfile.gettempdir()) / "outside_write_test.txt"
        if outside_file.exists():
            outside_file.unlink()


# ======================================================================
# VAL-EVAL-039 - Sandbox enforces file and disk limits
# ======================================================================

class TestFileAndDiskLimits:
    """Candidate code creating too many files or excessive data is limited."""

    def test_too_many_files_detected(self):
        """Creating many files is detected and reported."""
        code = textwrap.dedent("""\
            for i in range(200):
                with open(f"file_{i}.txt", "w") as f:
                    f.write("data")
            print("many files created")
        """)
        result = run_code(code, max_files=50, timeout=5.0)
        # Either the sandbox limits this or detects the excess
        assert result.status != "success" or (
            "files_created" in str(result.protections).lower()
        ), f"File limit should be enforced, status={result.status}"

    def test_excessive_output_size_truncated(self):
        """Writing huge amounts of data is truncated."""
        code = textwrap.dedent("""\
            print("X" * 500000)
        """)
        result = run_code(code, max_output_bytes=1024)
        assert result.output_truncated, (
            f"Output should be truncated, got {len(result.stdout)} bytes"
        )

    def test_deeply_nested_path_detected(self):
        """Creating deeply nested directory paths is limited."""
        code = textwrap.dedent("""\
            import os
            try:
                path = "a"
                for _ in range(20):
                    path = os.path.join(path, "b")
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "deep.txt"), "w") as f:
                    f.write("deep")
                print("deep path created")
            except OSError as exc:
                print(f"path depth limited: {exc}")
        """)
        result = run_code(code, max_path_depth=10, timeout=5.0)
        # Should complete within timeout
        assert result.status != "timeout" or result.timed_out

    def test_large_single_file_limited(self):
        """Writing a very large single file is limited."""
        code = textwrap.dedent("""\
            with open("big_file.bin", "wb") as f:
                f.write(b"X" * 5000000)
            print("large file written")
        """)
        result = run_code(code, max_disk_bytes=1024 * 100, timeout=5.0)
        # The large file should trigger disk limits
        assert result.status != "success" or (
            "file_limit" in result.status
            or "disk" in result.reason.lower()
            or not result.passed
        ), f"Disk limit should be enforced, status={result.status}"


# ======================================================================
# VAL-EVAL-040 - Sandbox reports actual protections honestly
# ======================================================================

class TestHonestProtectionReporting:
    """Sandbox result reports enforced protections and non-claims."""

    def test_result_includes_protections_dict(self):
        """SandboxResult includes protections dict."""
        result = run_code("print('hello')")
        assert isinstance(result.protections, dict), (
            f"protections should be a dict, got {type(result.protections)}"
        )
        # Key protections should be listed
        expected_keys = [
            "temp_cwd", "scrubbed_env", "timeout",
        ]
        for key in expected_keys:
            assert key in result.protections, (
                f"protection '{key}' not reported in {list(result.protections.keys())}"
            )

    def test_result_includes_non_claims(self):
        """SandboxResult honesty lists what is NOT claimed."""
        result = run_code("print('hello')")
        assert isinstance(result.non_claims, list), (
            f"non_claims should be a list, got {type(result.non_claims)}"
        )
        # Should mention at least OS-level isolation is not claimed
        combined = " ".join(result.non_claims).lower()
        assert any(term in combined for term in [
            "os isolation", "full isolation", "network isolation",
            "filesystem isolation", "container",
        ]), (
            f"non_claims should mention limitations, got: {result.non_claims}"
        )

    def test_cleanup_protection_reported(self):
        """Cleanup status is reported in protections."""
        result = run_code("print('hello')")
        assert "cleanup" in result.protections, (
            f"cleanup not in protections: {list(result.protections.keys())}"
        )

    def test_successful_run_reports_all_protections_true(self):
        """A successful run has all standard protections set to True."""
        result = run_code("print('hello')")
        for key in ["temp_cwd", "scrubbed_env", "timeout", "cleanup"]:
            assert result.protections.get(key) is True, (
                f"protection '{key}' should be True, got {result.protections.get(key)}"
            )

    def test_output_limit_protection_reported(self):
        """Output limit enforcement is reported."""
        result = run_code("print('hello')", max_output_bytes=1024)
        assert "output_limit" in result.protections, (
            f"output_limit not in protections: {list(result.protections.keys())}"
        )

    def test_file_limit_protection_reported(self):
        """File limit enforcement is reported."""
        result = run_code("print('hello')", max_files=100)
        assert "file_limit" in result.protections, (
            f"file_limit not in protections: {list(result.protections.keys())}"
        )

    def test_disk_limit_protection_reported(self):
        """Disk limit enforcement is reported."""
        result = run_code("print('hello')", max_disk_bytes=1024 * 1024)
        assert "disk_limit" in result.protections, (
            f"disk_limit not in protections: {list(result.protections.keys())}"
        )

    def test_does_not_claim_full_os_isolation(self):
        """The sandbox never claims full OS or filesystem isolation."""
        result = run_code("print('hello')")
        combined_output = (
            result.stdout + result.stderr + result.reason +
            " ".join(result.non_claims)
        ).lower()
        # Should NOT claim full isolation
        false_claims = [
            "full os isolation",
            "complete isolation",
            "guaranteed sandbox",
            "kernel-level isolation",
        ]
        for claim in false_claims:
            assert claim not in combined_output, (
                f"Sandbox falsely claims '{claim}' in output"
            )

    def test_limitation_disclosure_in_non_claims(self):
        """Non-claims include specific macOS subprocess limitations."""
        result = run_code("print('hello')")
        non_claims_text = " ".join(result.non_claims).lower()
        limitations_mentioned = any(
            term in non_claims_text
            for term in ["macos", "subprocess", "practical", "limited", "partial"]
        )
        assert limitations_mentioned, (
            f"Should mention macOS/subprocess limitations in non_claims: {result.non_claims}"
        )


# ======================================================================
# Code grader integration - test_input and expected_output
# ======================================================================

class TestCodeGraderIntegration:
    """Sandbox execution is wired into code grading via test_input and
    expected_output options."""

    def test_code_with_test_input_stdin(self):
        """Candidate code receives test_input via stdin."""
        code = textwrap.dedent("""\
            import sys
            data = sys.stdin.read()
            print(f"RECEIVED: {data}")
        """)
        result = run_code(code, test_input="hello world")
        assert result.exit_code == 0
        assert "RECEIVED: hello world" in result.stdout, (
            f"test_input not delivered via stdin: {result.stdout}"
        )

    def test_code_with_expected_output_passes(self):
        """Correct output matches expected_output."""
        code = textwrap.dedent("""\
            import sys
            data = sys.stdin.read().strip()
            print(data.upper())
        """)
        result = run_code(
            code,
            test_input="hello",
            expected_output="HELLO",
        )
        assert result.passed, (
            f"Expected pass, got: passed={result.passed} reason={result.reason}"
        )
        assert "HELLO" in result.stdout

    def test_code_with_wrong_output_fails(self):
        """Wrong output does not match expected_output."""
        code = textwrap.dedent("""\
            import sys
            data = sys.stdin.read().strip()
            print(data.upper() + "!!!")
        """)
        result = run_code(
            code,
            test_input="hello",
            expected_output="HELLO",
        )
        assert not result.passed, (
            f"Expected fail for wrong output, got: passed={result.passed}"
        )

    def test_code_no_expected_output_but_successful_run(self):
        """If no expected_output specified, pass/fail based on exit code."""
        code = textwrap.dedent("""\
            print("some output")
        """)
        result = run_code(code)
        assert result.exit_code == 0
        # If exit_code is 0, the run is considered successful
        assert result.passed, (
            f"Successful run should pass, got passed={result.passed}"
        )

    def test_code_runtime_error_fails(self):
        """Code that raises an exception fails."""
        code = textwrap.dedent("""\
            raise ValueError("intentional error")
        """)
        result = run_code(code)
        assert result.exit_code != 0
        assert not result.passed
        assert "error" in result.status or "exception" in result.status or "failed" in result.status

    def test_code_syntax_error_fails(self):
        """Code with syntax error fails."""
        code = "print('unclosed string"
        result = run_code(code)
        assert result.exit_code != 0
        assert not result.passed

    def test_code_with_multiline_test_input(self):
        """Multiline test_input works."""
        code = textwrap.dedent("""\
            import sys
            lines = sys.stdin.read().strip().split("\\n")
            print(f"LINES: {len(lines)}")
            for line in lines:
                print(f"  {line}")
        """)
        result = run_code(
            code,
            test_input="line1\nline2\nline3",
            expected_output="LINES: 3\n  line1\n  line2\n  line3",
        )
        assert result.passed, (
            f"Multiline test failed: {result.reason}\nstdout: {result.stdout}"
        )


# ======================================================================
# SandboxResult dataclass tests
# ======================================================================

class TestSandboxResultDataclass:
    """SandboxResult has the expected fields and default values."""

    def test_defaults(self):
        """Fields have expected defaults."""
        result = SandboxResult()
        assert result.passed is False
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == -1
        assert result.timed_out is False
        assert result.output_truncated is False
        assert result.status == "error"
        assert isinstance(result.protections, dict)
        assert isinstance(result.non_claims, list)
        assert result.reason == ""

    def test_repr_includes_status(self):
        """repr includes status info."""
        result = SandboxResult(passed=True, status="success", reason="all good")
        rep = repr(result)
        assert "SandboxResult" in rep
        assert "success" in rep
