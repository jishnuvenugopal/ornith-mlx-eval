"""Profile / preflight checks and machine metadata.

Owns all preflight checks including Python runtime, MLX packages, Metal
availability, Hugging Face cache writability, output directory writability,
disk headroom, memory/swap, and model metadata resolution via the Hugging
Face Hub API.  All checks are metadata-only and must never download model
weights.

Profile Results
---------------

Every check returns a dict with:
- ``name``: short check identifier (e.g. "python")
- ``status``: "pass" or "fail"
- ``details``: dict of check-specific facts
- ``reason``: human-readable failure reason (only when status is "fail")
"""

from __future__ import annotations

import os
import platform as _platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional imports (safe try/except for testability)
# ---------------------------------------------------------------------------

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError as HfRepoNotFoundError
except ImportError:  # pragma: no cover — not reached with dev dependencies installed
    HfApi = None  # type: ignore[assignment]
    HfRepoNotFoundError = Exception  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PYTHON_VERSION: tuple[int, int] = (3, 10)
PREFERRED_PYTHON_VERSION: tuple[int, int] = (3, 12)
DISK_HEADROOM_BYTES: int = 12 * 1024**3  # 12 GiB
MIN_MLX_VERSION = "0.31.2"
MIN_MLX_LM_VERSION = "0.31.3"

# Model scope: models whose ID includes these substrings are unsupported
# in normal work.  Checked case-insensitively.
UNSUPPORTED_MODEL_PATTERNS: list[str] = ["8bit", "35b"]

# Default output root for commands that create run artifacts.
DEFAULT_OUTPUT_ROOT: str = "benchmark_results"

# Repo root, for venv detection heuristics.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_check(name: str, status: str, details: dict[str, Any], reason: str = "") -> dict[str, Any]:
    """Build a standardised check result dict."""
    result: dict[str, Any] = {"name": name, "status": status, "details": details}
    if reason:
        result["reason"] = reason
    return result


def _is_project_venv() -> bool:
    """Heuristic: is the current interpreter inside the project's .venv?

    Uses ``sys.prefix`` (set by venv's site.py to point at the venv directory)
    and compares it against the expected project .venv path.
    """
    # sys.prefix points to the virtual environment's root when running in a venv
    prefix = Path(sys.prefix).resolve()
    venv_root = _REPO_ROOT.resolve() / ".venv"
    try:
        prefix.relative_to(venv_root)
        return True
    except ValueError:
        pass
    # Fallback: check if the (unresolved) executable starts with .venv/
    exe = Path(sys.executable)
    try:
        exe.relative_to(venv_root)
        return True
    except ValueError:
        pass
    return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python() -> dict[str, Any]:
    """Check Python runtime: arm64 architecture, version >= 3.10, project venv.

    Returns
    -------
    dict with keys ``name``, ``status``, ``details``, and ``reason`` on failure.
    """
    arch = _platform.machine()
    vi = sys.version_info
    version = f"{vi[0]}.{vi[1]}.{vi[2]}"
    executable = sys.executable
    in_venv = _is_project_venv()
    version_tuple = vi[:2]
    is_preferred = version_tuple == PREFERRED_PYTHON_VERSION

    details = {
        "architecture": arch,
        "version": version,
        "executable": executable,
        "in_venv": in_venv,
        "preferred": is_preferred,
    }

    failures: list[str] = []

    if arch != "arm64":
        failures.append(
            f"Architecture is '{arch}', expected native arm64. "
            "Running under Rosetta is not supported for MLX."
        )
    if version_tuple < MIN_PYTHON_VERSION:
        failures.append(
            f"Python {version} is below minimum {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}. "
            "Please use Python 3.10+."
        )

    if failures:
        return _make_check("python", "fail", details, "; ".join(failures))
    return _make_check("python", "pass", details)


def check_mlx_packages() -> dict[str, Any]:
    """Check that ``mlx`` and ``mlx-lm`` are importable and report versions.

    Returns
    -------
    dict with import status and version strings.
    """
    details: dict[str, Any] = {}
    failures: list[str] = []

    # mlx
    try:
        import mlx

        mlx_ver = getattr(mlx, "__version__", None)
        if mlx_ver is None:
            # ``mlx`` does not expose ``__version__``; resolve via package metadata.
            try:
                mlx_ver = version("mlx")
            except PackageNotFoundError:
                mlx_ver = "unknown"
        details["mlx_version"] = mlx_ver
    except ImportError:
        details["mlx_version"] = None
        failures.append(
            "Package 'mlx' is not installed. Install it with: "
            "pip install mlx==" + MIN_MLX_VERSION
        )

    # mlx-lm
    try:
        import mlx_lm

        mlx_lm_ver = getattr(mlx_lm, "__version__", None)
        if mlx_lm_ver is None:
            try:
                mlx_lm_ver = version("mlx-lm")
            except PackageNotFoundError:
                mlx_lm_ver = "unknown"
        details["mlx_lm_version"] = mlx_lm_ver
    except ImportError:
        details["mlx_lm_version"] = None
        failures.append(
            "Package 'mlx-lm' is not installed. Install it with: "
            "pip install mlx-lm==" + MIN_MLX_LM_VERSION
        )

    if failures:
        return _make_check("mlx_packages", "fail", details, "; ".join(failures))
    return _make_check("mlx_packages", "pass", details)


def check_metal() -> dict[str, Any]:
    """Check that MLX Metal is available and report device details.

    Must be called after ``check_mlx_packages`` succeeds or via import guard.
    """
    details: dict[str, Any] = {}
    failures: list[str] = []

    try:
        import mlx.core as mx
    except ImportError:
        return _make_check(
            "metal",
            "fail",
            {"metal_available": False},
            "Cannot import mlx.core. Install mlx to check Metal availability.",
        )

    # Check Metal availability via top-level is_available()
    try:
        metal_available = bool(mx.metal.is_available())
    except Exception:
        metal_available = False
        failures.append("MLX Metal is not available on this system.")

    details["metal_available"] = metal_available

    if metal_available:
        try:
            dev_info = mx.device_info()
            details["device_info"] = dev_info
            if isinstance(dev_info, dict):
                details["gpu_name"] = dev_info.get("gpu_name", dev_info.get("name", "unknown"))
                details["memory_size"] = dev_info.get("memory_size", 0)
                details["max_recommended_working_set_size"] = dev_info.get(
                    "max_recommended_working_set_size", 0
                )
        except Exception:
            details["gpu_name"] = "unknown"

    if not metal_available:
        failures.append(
            "MLX Metal is not available. MLX requires Apple Silicon with Metal support."
        )

    if failures:
        return _make_check("metal", "fail", details, "; ".join(failures))
    return _make_check("metal", "pass", details)


def check_hf_cache() -> dict[str, Any]:
    """Check HF cache path exists and is writable.

    Uses ``HF_HOME`` env var if set, otherwise defaults to
    ``~/.cache/huggingface``.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_path = Path(hf_home)

    details: dict[str, Any] = {
        "hf_home_env": os.environ.get("HF_HOME"),
        "cache_path": str(cache_path),
    }
    failures: list[str] = []

    # Try to create the cache directory if it doesn't exist
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        failures.append(f"Hugging Face cache path is not creatable: {cache_path} ({exc})")
        return _make_check("hf_cache", "fail", details, "; ".join(failures))

    # Check writability
    test_file = cache_path / ".ornith_write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        details["writable"] = True
    except (OSError, PermissionError) as exc:
        details["writable"] = False
        failures.append(f"Hugging Face cache path is not writable: {cache_path} ({exc})")

    if failures:
        return _make_check("hf_cache", "fail", details, "; ".join(failures))
    return _make_check("hf_cache", "pass", details)


def check_output_dir(output_root: str = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Check that the output root directory is creatable and writable.

    Parameters
    ----------
    output_root: Path to the intended output root directory.

    Returns
    -------
    Check result dict.
    """
    root = Path(output_root).resolve()
    details: dict[str, Any] = {"output_root": str(root)}
    failures: list[str] = []

    if root.exists():
        # Directory exists — check writability
        test_file = root / ".ornith_write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            details["writable"] = True
        except (OSError, PermissionError) as exc:
            details["writable"] = False
            failures.append(f"Output root is not writable: {root} ({exc})")
    else:
        # Directory doesn't exist — check if parent allows creation
        parent = root.parent
        details["exists"] = False
        try:
            # Try to determine if parent is writable
            if parent.exists():
                test_file = parent / ".ornith_write_test"
                test_file.write_text("test")
                test_file.unlink()
                details["writable"] = True
            else:
                details["writable"] = False
                failures.append(f"Output root parent does not exist: {parent}")
        except (OSError, PermissionError) as exc:
            details["writable"] = False
            failures.append(f"Output root parent is not writable: {parent} ({exc})")

    if failures:
        return _make_check("output", "fail", details, "; ".join(failures))
    return _make_check("output", "pass", details)


def check_disk(model_size_bytes: int = 0) -> dict[str, Any]:
    """Check free disk space against model size + headroom.

    Parameters
    ----------
    model_size_bytes: Expected model LFS size in bytes.  When 0, only
        a general check is performed.

    Returns
    -------
    Check result with free bytes, model size, headroom, and required bytes.
    """
    # Use the repo root as the target disk for measurement.
    target = str(_REPO_ROOT)

    try:
        usage = shutil.disk_usage(target)
        free_bytes = usage.free
    except Exception as exc:
        return _make_check(
            "disk", "fail", {"error": str(exc)}, f"Cannot determine free disk space: {exc}"
        )

    required_bytes = model_size_bytes + (DISK_HEADROOM_BYTES if model_size_bytes > 0 else 0)
    details: dict[str, Any] = {
        "free_bytes": free_bytes,
        "model_size_bytes": model_size_bytes,
        "headroom_required_bytes": DISK_HEADROOM_BYTES if model_size_bytes > 0 else 0,
        "required_bytes": required_bytes,
        "target": target,
    }

    if required_bytes > 0 and free_bytes < required_bytes:
        free_gib = free_bytes / (1024**3)
        required_gib = required_bytes / (1024**3)
        headroom_gib = DISK_HEADROOM_BYTES / (1024**3)
        return _make_check(
            "disk",
            "fail",
            details,
            f"Insufficient free disk: {free_gib:.1f} GiB available, "
            f"{required_gib:.1f} GiB required "
            f"(model: {model_size_bytes / (1024**3):.1f} GiB + "
            f"{headroom_gib:.0f} GiB headroom).",
        )

    return _make_check("disk", "pass", details)


def check_memory() -> dict[str, Any]:
    """Report memory pressure, swap, and physical memory where available.

    This is a best-effort check on macOS.  It reports current state but does
    not fail on numbers alone (the real gate is in run/smoke).
    """
    import subprocess

    details: dict[str, Any] = {}

    # Use sysctl for memory info
    try:
        memsize_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        details["physical_memory_bytes"] = memsize_bytes
    except (ValueError, OSError):
        pass

    # vm_stat for memory pressure
    try:
        vm_stat = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if vm_stat.returncode == 0:
            details["vm_stat"] = vm_stat.stdout.strip()
    except Exception:
        pass

    # Check swap usage via sysctl
    try:
        swap = subprocess.run(
            ["sysctl", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if swap.returncode == 0:
            details["swap"] = swap.stdout.strip()
    except Exception:
        pass

    # Check memory pressure level (macOS specific)
    try:
        pressure = subprocess.run(
            ["sysctl", "kern.memorystatus_level"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if pressure.returncode == 0:
            details["memory_pressure"] = pressure.stdout.strip()
    except Exception:
        pass

    # Load average
    try:
        loadavg = os.getloadavg()
        details["load_average"] = f"{loadavg[0]:.2f} {loadavg[1]:.2f} {loadavg[2]:.2f}"
    except (OSError, AttributeError):
        pass

    # Memory check never fails standalone; it's informational
    if not details:
        details["note"] = "No memory metrics could be collected."
    return _make_check("memory", "pass", details)


# ---------------------------------------------------------------------------
# Model metadata resolution (Hugging Face, no downloads)
# ---------------------------------------------------------------------------


def _is_model_supported(model_id: str) -> tuple[bool, str]:
    """Check whether a model identifier is in the supported scope.

    Returns
    -------
    (supported, reason) — ``supported`` is False when the model is out of
    scope; ``reason`` explains why.
    """
    lower = model_id.lower()
    for pattern in UNSUPPORTED_MODEL_PATTERNS:
        if pattern.lower() in lower:
            return False, f"Model '{model_id}' matches unsupported pattern '{pattern}'. Only 4bit and 6bit models are in scope."
    return True, ""


def resolve_model_metadata(model_id: str) -> dict[str, Any]:
    """Resolve Hugging Face model metadata without downloading weights.

    Uses ``huggingface_hub.HfApi.model_info`` (metadata-only API call) to
    look up the exact revision SHA, model size, and gating status.

    Parameters
    ----------
    model_id: A Hugging Face repo ID like ``mlx-community/Ornith-1.0-9B-4bit``.

    Returns
    -------
    Dict with keys ``name``, ``status``, ``model_id``, ``sha``,
    ``size_bytes``, ``gated``, ``private``, ``details``, and ``reason`` on
    failure.
    """
    details: dict[str, Any] = {"model_id": model_id}

    # 1. Check model is in supported scope
    supported, reason = _is_model_supported(model_id)
    if not supported:
        return _make_check("model", "fail", details, reason)

    # 2. Resolve via Hugging Face Hub (metadata only, no download)
    if HfApi is None:
        return _make_check(
            "model",
            "fail",
            details,
            "Package 'huggingface_hub' is not installed. Cannot resolve model metadata.",
        )

    api = HfApi()
    try:
        info = api.model_info(model_id, files_metadata=True)
    except HfRepoNotFoundError:
        return _make_check(
            "model",
            "fail",
            details,
            f"Repository '{model_id}' not found on Hugging Face.",
        )
    except Exception as exc:
        return _make_check(
            "model",
            "fail",
            details,
            f"Failed to fetch model info for '{model_id}': {exc}",
        )

    # 3. Check gated / private
    if getattr(info, "gated", False):
        return _make_check(
            "model",
            "fail",
            details,
            f"Model '{model_id}' is gated and requires access approval.",
        )
    if getattr(info, "private", False):
        return _make_check(
            "model",
            "fail",
            details,
            f"Model '{model_id}' is private and cannot be accessed.",
        )

    # 4. Extract SHA and compute LFS size
    sha: str = getattr(info, "sha", "") or ""
    if not sha or len(sha) != 40:
        return _make_check(
            "model",
            "fail",
            details,
            f"Could not resolve exact revision SHA for '{model_id}'.",
        )

    # Compute total size of model weight files (LFS blobs)
    # MLX repos may use .safetensors or .npz format
    WEIGHT_EXTENSIONS = {".safetensors", ".npz"}
    siblings = getattr(info, "siblings", []) or []
    total_size: int = 0
    for sibling in siblings:
        rfname = getattr(sibling, "rfilename", "")
        if isinstance(rfname, str):
            _, ext = os.path.splitext(rfname)
            if ext.lower() in WEIGHT_EXTENSIONS:
                sibling_size = getattr(sibling, "size", 0) or 0
                total_size += sibling_size

    details.update({
        "sha": sha,
        "size_bytes": total_size,
        "gated": False,
        "private": False,
        "tags": list(getattr(info, "tags", []) or []),
    })

    return _make_check("model", "pass", details)


# ---------------------------------------------------------------------------
# Profile orchestration
# ---------------------------------------------------------------------------


def run_profile(
    model_id: str | None = None,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Run all preflight checks and return structured results.

    Parameters
    ----------
    model_id: Optional Hugging Face model ID for metadata resolution.
        When provided, model scope, access, and SHA resolution are checked
        without downloading weights.
    output_root: Output root directory to check writability.

    Returns
    -------
    Dict with:
    - ``status``: overall "pass" or "fail"
    - ``checks``: list of individual check results
    - ``model``: model metadata check result (only when model_id was supplied)
    """
    checks: list[dict[str, Any]] = []

    # Run checks in deterministic order
    checks.append(check_python())
    checks.append(check_mlx_packages())
    checks.append(check_metal())
    checks.append(check_hf_cache())
    checks.append(check_output_dir(output_root=output_root))

    # Disk check: if model metadata resolved, use actual size
    model_result: dict[str, Any] | None = None
    if model_id is not None:
        model_result = resolve_model_metadata(model_id)
        if model_result["status"] == "pass":
            disk_check = check_disk(model_size_bytes=model_result["details"].get("size_bytes", 0))
        else:
            # Even if model resolution fails, still check disk
            disk_check = check_disk(model_size_bytes=0)
    else:
        disk_check = check_disk(model_size_bytes=0)

    checks.append(disk_check)
    checks.append(check_memory())

    # Determine overall status
    overall = "pass"
    for c in checks:
        if c["status"] == "fail":
            overall = "fail"
    if model_result is not None and model_result["status"] == "fail":
        overall = "fail"

    result: dict[str, Any] = {"status": overall, "checks": checks}
    if model_result is not None:
        result["model"] = model_result

    return result


def format_profile_output(profile_result: dict[str, Any]) -> str:
    """Format a profile result dict as human-readable text for stdout.

    Parameters
    ----------
    profile_result: The dict returned by ``run_profile``.

    Returns
    -------
    Formatted string suitable for printing.
    """
    lines: list[str] = []
    overall = profile_result["status"]
    status_label = "PASS" if overall == "pass" else "FAIL"

    lines.append("=" * 64)
    lines.append(f"  Ornith MLX Eval — Profile Check — {status_label}")
    lines.append("=" * 64)
    lines.append("")

    for check in profile_result["checks"]:
        name = check["name"]
        status = check["status"]
        marker = "✓" if status == "pass" else "✗"
        lines.append(f"  [{marker}] {name.replace('_', ' ').title()}: {status.upper()}")

        details = check.get("details", {})
        if details:
            for key, value in details.items():
                # Skip overly verbose or binary data
                if isinstance(value, str) and len(value) > 120:
                    value = value[:120] + "..."
                lines.append(f"        {key}: {value}")

        if status == "fail" and "reason" in check:
            lines.append(f"        Reason: {check['reason']}")
        lines.append("")

    # Model section
    model = profile_result.get("model")
    if model is not None:
        lines.append("-" * 64)
        lines.append(f"  Model: {model['details'].get('model_id', '?')}")
        if model["status"] == "pass":
            sha = model["details"].get("sha", "")
            size_bytes = model["details"].get("size_bytes", 0)
            size_gib = size_bytes / (1024**3) if size_bytes else 0
            lines.append(f"  SHA: {sha}")
            lines.append(f"  Size: {size_bytes:,} bytes ({size_gib:.2f} GiB)")
            lines.append(f"  Gate Status: PASS")
        else:
            lines.append(f"  Status: FAIL")
            lines.append(f"  Reason: {model.get('reason', 'unknown error')}")
        lines.append("")

    lines.append("=" * 64)
    return "\n".join(lines)
