"""Tests for command-line ergonomics on local developer machines."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pytest_uses_repo_local_temp_directory():
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "--basetemp=.pytest_tmp" in config


def test_data_download_script_bootstraps_repo_root_imports():
    script = (ROOT / "notebooks" / "01_data_download.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script
