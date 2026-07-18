"""License publication checks."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mit_license_text_is_present():
    license_text = (_REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
