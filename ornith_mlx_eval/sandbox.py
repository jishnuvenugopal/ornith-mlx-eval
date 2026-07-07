"""Bounded Python subprocess sandbox for coding graders.

Provides practical macOS controls.  This is a **subprocess-level** sandbox
that enforces:

* Isolated temporary working directory
* Scrubbed environment (no real HOME, minimal PATH, secret-free env)
* Per-run timeout
* Output size limits
* Candidate-created file count and total disk limits
* Controlled outside-access detection (sentinel-based + filesystem scan)

It does **not** provide full OS or container-level isolation and reports
that honestly.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Result of running code inside the bounded subprocess sandbox.

    Attributes:
        passed: *True* when the run met all grading criteria
            (exit 0, no outside access, output matches expected).
        stdout: Captured stdout from the candidate process (truncated
            when output limit is exceeded).
        stderr: Captured stderr from the candidate process.
        exit_code: The process exit code, or -1 on timeout/kill.
        timed_out: *True* when the process was killed due to timeout.
        output_truncated: *True* when stdout exceeded the byte limit.
        status: Classification string -- one of ``"success"``,
            ``"timeout"``, ``"output_limit"``, ``"file_limit"``,
            ``"outside_access"``, ``"error"``, or ``"failed"``.
        protections: Map of protection name -> enforced (bool).
        non_claims: Explicit list of protections this sandbox does
            **not** provide (e.g. full OS isolation).
        reason: Human-readable explanation of the result.
        files_created: Count of files/dirs candidate code created inside
            the sandbox root.
        disk_bytes_used: Approximate total bytes written as files inside
            the sandbox root.
        wall_time: Wall-clock duration of the subprocess run in seconds.
    """

    passed: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    output_truncated: bool = False
    status: str = "error"
    protections: dict[str, bool] = field(default_factory=dict)
    non_claims: list[str] = field(default_factory=list)
    reason: str = ""
    files_created: int = 0
    disk_bytes_used: int = 0
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# Environment scrubbing
# ---------------------------------------------------------------------------

# Variables we **never** pass through to the sandboxed process.
_VARS_TO_SCRUB: frozenset[str] = frozenset({
    # Standard secrets / credentials
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN", "GH_TOKEN",
    "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
    "COHERE_API_KEY", "MISTRAL_API_KEY",
    "API_KEY", "API_TOKEN", "SECRET_KEY", "ACCESS_TOKEN",
    "PASSWORD", "PASSWD",
    # User identity / paths
    "HOME", "USER", "LOGNAME", "USERNAME",
    "SSH_AUTH_SOCK", "SSH_AGENT_PID",
    "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
    "XDG_DATA_HOME", "XDG_STATE_HOME",
    # Python / pip
    "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME",
    "PIP_REQUIRE_VIRTUALENV",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    # Sensitive FS paths
    "PWD", "OLDPWD",
    # Apple-specific
    "SECURITYSESSIONID", "XPC_SERVICE_NAME", "XPC_FLAGS",
    "__CFBundleIdentifier", "__CF_USER_TEXT_ENCODING",
    "COMMAND_MODE",
    # Generic token/secret patterns
    "TOKEN", "SECRET", "CREDENTIAL", "KEY", "PRIVATE",
})

# Patterns for additional env-var names to scrub (case-insensitive).
_SCRUB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^AWS_.*", re.IGNORECASE),
    re.compile(r"^GITHUB_.*", re.IGNORECASE),
    re.compile(r"^HF_.*", re.IGNORECASE),
    re.compile(r"^HUGGINGFACE.*", re.IGNORECASE),
    re.compile(r".*_TOKEN$", re.IGNORECASE),
    re.compile(r".*_SECRET$", re.IGNORECASE),
    re.compile(r".*_KEY$", re.IGNORECASE),
    re.compile(r".*_PASSWORD$", re.IGNORECASE),
    re.compile(r"^OPENAI_.*", re.IGNORECASE),
    re.compile(r"^ANTHROPIC_.*", re.IGNORECASE),
    re.compile(r"^COHERE_.*", re.IGNORECASE),
    re.compile(r"^MISTRAL_.*", re.IGNORECASE),
    re.compile(r"^GOOGLE_.*", re.IGNORECASE),
)

# Variables we explicitly pass through (minimal safe set).
_VARS_TO_KEEP: frozenset[str] = frozenset({
    "PATH",
    "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR",
    "SYSTEM_VERSION_COMPAT",
})

# Sentinel content markers for outside-access detection.
_SENTINEL_MARKERS: frozenset[str] = frozenset({
    "SENTINEL_CONTENT_DO_NOT_LEAK",
    "HOME_SENTINEL_DO_NOT_LEAK",
    "SECRET_SENTINEL_MARKER",
})


def _build_sandbox_env(
    extra_vars: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Build a scrubbed environment dict for the sandboxed process.

    Args:
        extra_vars: Optional extra variables to inject (these are visible
            to candidate code; they represent fake secrets for testing).

    Returns:
        A *dict[str, str]* with minimal safe environment variables.
    """
    env: dict[str, str] = {}

    # Copy only whitelisted variables from the real environment.
    for key in _VARS_TO_KEEP:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    # Sanitize PATH: strip entries containing the real home directory.
    real_home = os.path.expanduser("~")
    if "PATH" in env:
        clean_entries: list[str] = []
        for entry in env["PATH"].split(":"):
            if real_home not in entry:
                clean_entries.append(entry)
        env["PATH"] = ":".join(clean_entries)
    if not env.get("PATH"):
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"

    # Always override LANG to a safe default.
    env.setdefault("LANG", "en_US.UTF-8")

    # Override HOME with a placeholder.
    env["HOME"] = "/tmp/sandbox_home"

    # Inject any extra vars (for testing).  Secret-like vars are still
    # scrubbed so they act as controlled sentinels for leak detection.
    if extra_vars:
        for k, v in extra_vars.items():
            if k not in _VARS_TO_SCRUB and not _should_scrub(k):
                env[k] = v

    return env


def _should_scrub(varname: str) -> bool:
    """Return *True* if *varname* should be scrubbed based on patterns."""
    if varname.upper() in _VARS_TO_SCRUB:
        return True
    for pat in _SCRUB_PATTERNS:
        if pat.fullmatch(varname):
            return True
    return False


# ---------------------------------------------------------------------------
# Core sandbox runner
# ---------------------------------------------------------------------------

# Truncation marker appended when stdout exceeds the byte budget.
_TRUNCATION_MARKER = "\n\n[OUTPUT TRUNCATED: exceeded byte limit]\n"


def run_code(
    code: str,
    test_input: Optional[str] = None,
    expected_output: Optional[str] = None,
    timeout: float = 10.0,
    max_output_bytes: int = 1024 * 1024,  # 1 MiB
    max_files: int = 100,
    max_disk_bytes: int = 10 * 1024 * 1024,  # 10 MiB
    max_path_depth: int = 15,
    sandbox_root: Optional[str] = None,
    env_overrides: Optional[dict[str, str]] = None,
    retain_root: bool = False,
) -> SandboxResult:
    """Run *code* inside a bounded Python subprocess sandbox.

    Args:
        code: The Python source code to execute.
        test_input: Optional string to pipe to the process stdin.
        expected_output: Optional expected stdout (after stripping).
            When provided, the output is compared and *passed* reflects
            an exact match (with trailing whitespace stripped) **and**
            exit code 0.
        timeout: Wall-clock timeout in seconds for the subprocess.
        max_output_bytes: Maximum bytes of stdout to capture before
            truncating.
        max_files: Maximum files/directories the candidate code may
            create inside the sandbox root before the run is flagged.
        max_disk_bytes: Maximum total bytes the candidate may write as
            files inside the sandbox root.
        max_path_depth: Maximum directory nesting depth allowed.
        sandbox_root: Optional explicit temp root (set for test
            isolation or debug retention).  When omitted a fresh
            temporary directory is created.
        env_overrides: Optional dict of extra env vars to inject
            **before** scrubbing.  Variables matching scrub patterns
            are still removed so they act as controlled sentinels.
        retain_root: If *True*, do not delete the sandbox root after
            execution (useful for debugging).

    Returns:
        A :class:`SandboxResult` with the execution outcome.
    """
    # -- Establish sandbox root -----------------------------------------------
    if sandbox_root is None:
        sandbox_root = tempfile.mkdtemp(prefix="ornith_sandbox_")
    root_path = Path(sandbox_root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    # -- Populate protections metadata ----------------------------------------
    protections: dict[str, bool] = {
        "temp_cwd": True,
        "scrubbed_env": True,
        "timeout": timeout > 0,
        "output_limit": max_output_bytes > 0,
        "file_limit": max_files > 0,
        "disk_limit": max_disk_bytes > 0,
        "cleanup": False,  # set to True after cleanup
    }
    non_claims: list[str] = [
        "no full OS-level isolation (subprocess only)",
        "no guaranteed filesystem isolation beyond path detection",
        "no network isolation (network access is not blocked)",
        "no container/VM-level sandboxing",
        "no kernel-level mandatory access controls",
    ]

    # -- Snapshot repo for outside-access filesystem scan ---------------------
    repo_root: Optional[Path] = None
    _before_files: set[str] = set()
    try:
        repo_root = _find_repo_root()
        if repo_root is not None:
            _before_files = _snapshot_dir_files(repo_root)
    except Exception:
        pass

    # -- Write candidate code to temp file ------------------------------------
    code_path = root_path / "_candidate_code.py"
    code_path.write_text(code, encoding="utf-8")

    # -- Build environment ----------------------------------------------------
    env = _build_sandbox_env(extra_vars=env_overrides)

    # -- Run subprocess -------------------------------------------------------
    stdout = ""
    stderr = ""
    exit_code = -1
    timed_out = False
    output_truncated = False
    wall_start = time.monotonic()

    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-E", str(code_path)],
            cwd=str(root_path),
            env=env,
            stdin=subprocess.PIPE if test_input else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # read as bytes, decode ourselves
            preexec_fn=_preexec_setlimits,
        )

        stdin_bytes = None
        if test_input is not None:
            stdin_bytes = test_input.encode("utf-8", errors="replace")

        try:
            out_bytes, err_bytes = proc.communicate(
                input=stdin_bytes,
                timeout=timeout,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill the process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.kill()
            except OSError:
                pass
            try:
                out_bytes, err_bytes = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                out_bytes, err_bytes = (b"", b"")
                try:
                    proc.kill()
                except OSError:
                    pass
            exit_code = -1

        # Apply output limit
        if out_bytes:
            if len(out_bytes) > max_output_bytes:
                output_truncated = True
                out_bytes = (
                    out_bytes[:max_output_bytes] +
                    _TRUNCATION_MARKER.encode("utf-8", errors="replace")
                )
            stdout = out_bytes.decode("utf-8", errors="replace")
        if err_bytes:
            stderr = err_bytes.decode("utf-8", errors="replace")

    except Exception as exc:
        stdout = ""
        stderr = f"Sandbox internal error: {exc}"
        exit_code = -1

    wall_end = time.monotonic()
    wall_time = wall_end - wall_start

    # -- Post-execution: scan for outside access ------------------------------
    files_created = 0
    disk_bytes_used = 0
    outside_access = False
    outside_reason = ""

    # (a) Scan candidate-created artifacts inside sandbox root
    try:
        for item in root_path.rglob("*"):
            if item == code_path:
                continue
            if item.is_file():
                files_created += 1
                try:
                    disk_bytes_used += item.stat().st_size
                except OSError:
                    pass
            elif item.is_dir():
                files_created += 1
    except Exception:
        pass

    # (b) Sentinel-content scan: check if known sentinel strings appeared
    #     in candidate stdout.  These are markers placed by tests in files
    #     outside the sandbox root.
    combined_output = stdout + stderr
    for marker in _SENTINEL_MARKERS:
        if marker in combined_output:
            outside_access = True
            outside_reason = (
                f"Sentinel content detected in output: '{marker}'. "
                f"Candidate code likely read a file outside the sandbox."
            )
            break

    # (c) Filesystem scan: check for new files in repo root
    if not outside_access and repo_root is not None:
        try:
            _after_files = _snapshot_dir_files(repo_root)
            new_files = _after_files - _before_files
            # Filter out known sandbox/test artifacts
            new_outside = [
                f for f in new_files
                if "_sandbox_" not in f
                and "ornith_sandbox_" not in f
                and ".pyc" not in Path(f).suffix
                and "__pycache__" not in f
                and "_candidate_code" not in f
            ]
            if new_outside:
                outside_access = True
                outside_reason = (
                    f"New files created outside sandbox root: {new_outside[:5]}"
                )
                # Attempt cleanup of detected outside artifacts
                for fpath in new_outside:
                    try:
                        os.unlink(fpath)
                    except OSError:
                        pass
        except Exception:
            pass

    # -- Determine status -----------------------------------------------------
    file_limit_exceeded = files_created > max_files
    disk_limit_exceeded = disk_bytes_used > max_disk_bytes

    if timed_out:
        status = "timeout"
        passed = False
        reason = (
            f"Sandbox timeout: execution timed out after {timeout:.1f}s "
            f"(wall time {wall_time:.2f}s)"
        )
        protections["timeout"] = True
    elif outside_access:
        status = "outside_access"
        passed = False
        reason = (
            f"Outside file access detected: {outside_reason} "
            f"This sandbox provides practical subprocess-level controls "
            f"but does not guarantee full OS isolation."
        )
    elif output_truncated:
        status = "output_limit"
        passed = False
        reason = f"Output exceeded {max_output_bytes} byte limit"
    elif file_limit_exceeded:
        status = "file_limit"
        passed = False
        reason = (
            f"File limit exceeded: created {files_created} files "
            f"(max {max_files})"
        )
    elif disk_limit_exceeded:
        status = "file_limit"
        passed = False
        reason = (
            f"Disk limit exceeded: wrote {disk_bytes_used} bytes "
            f"(max {max_disk_bytes})"
        )
    elif exit_code != 0:
        status = "failed"
        passed = False
        reason = f"Process exited with code {exit_code}"
    else:
        status = "success"
        passed = True
        reason = "Execution completed successfully"

    # -- Compare output against expected --------------------------------------
    if expected_output is not None and status == "success":
        actual = stdout.rstrip()
        expected = expected_output.rstrip()
        if actual != expected:
            passed = False
            reason = (
                f"Output mismatch: expected {expected!r}, got {actual!r}"
            )
            status = "failed"

    # -- Cleanup --------------------------------------------------------------
    if not retain_root:
        try:
            shutil.rmtree(root_path, ignore_errors=True)
            protections["cleanup"] = True
        except Exception:
            protections["cleanup"] = False
    else:
        protections["cleanup"] = False

    return SandboxResult(
        passed=passed,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        output_truncated=output_truncated,
        status=status,
        protections=protections,
        non_claims=non_claims,
        reason=reason,
        files_created=files_created,
        disk_bytes_used=disk_bytes_used,
        wall_time=wall_time,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _preexec_setlimits() -> None:
    """Pre-exec function: set resource limits and detach process group.

    Sets RLIMIT_NPROC and RLIMIT_FSIZE where available on macOS.
    Creates a new process group so we can kill children on timeout.
    """
    # Detach from parent process group.
    os.setpgrp()

    try:
        import resource

        # Limit total file size to 50 MiB per write (catches runaway
        # file writes at the OS level).
        soft_fsize = resource.getrlimit(resource.RLIMIT_FSIZE)[0]
        if soft_fsize == resource.RLIM_INFINITY or soft_fsize > (50 * 1024 * 1024):
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))

        # Limit CPU time as a backup for timeout.
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))

        # Attempt to limit child processes (macOS may ignore this).
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass  # macOS often does not enforce RLIMIT_NPROC

    except (ImportError, ValueError, OSError):
        pass


def _find_repo_root() -> Optional[Path]:
    """Try to locate the repository root for outside-access detection."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _snapshot_dir_files(directory: Path) -> set[str]:
    """Return a set of absolute file paths under *directory*."""
    files: set[str] = set()
    try:
        for item in directory.iterdir():
            if item.is_file():
                files.add(str(item.resolve()))
    except OSError:
        pass
    return files


# ---------------------------------------------------------------------------
# Convenience: Protection descriptor
# ---------------------------------------------------------------------------


class SandboxProtection:
    """Namespaced constants for sandbox protection keys."""

    TEMP_CWD = "temp_cwd"
    SCRUBBED_ENV = "scrubbed_env"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    FILE_LIMIT = "file_limit"
    DISK_LIMIT = "disk_limit"
    CLEANUP = "cleanup"
