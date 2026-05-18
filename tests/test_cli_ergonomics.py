"""Tests for command-line ergonomics on local developer machines."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pytest_avoids_shared_temp_and_cache_directories():
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    # Pytest's cache provider must stay disabled (avoids shared .pytest_cache).
    assert "-p no:cacheprovider" in config

    # If --basetemp is set, it must point at a project-local path, not the
    # system %TEMP% (which is locked down on some Windows accounts and shared
    # between users). Project-local basetemp directories are gitignored.
    if "--basetemp" in config:
        for forbidden in ("%TEMP%", "/tmp/", "C:\\Users\\", "AppData"):
            assert forbidden not in config, (
                f"--basetemp must point at a project-local path; "
                f"found forbidden segment {forbidden!r}"
            )


def test_data_download_script_bootstraps_repo_root_imports():
    script = (ROOT / "notebooks" / "01_data_download.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_data_download_script_exposes_provider_option():
    script = (ROOT / "notebooks" / "01_data_download.py").read_text(encoding="utf-8")

    assert "--provider" in script
    assert "MARKOV_DATA_PROVIDER" in script


def test_state_expectancy_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "build_state_expectancy_table.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_narrow_backtest_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "run_narrow_backtest.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_vol_conditioned_expectancy_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "build_vol_conditioned_expectancy.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_walkforward_backtest_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "run_walkforward_backtest.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_asset_clustering_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "build_asset_clusters.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_dashboard_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "build_dashboard.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_cluster_pooled_expectancy_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "build_cluster_pooled_expectancy.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_markov_weighted_ev_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "build_markov_weighted_ev.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_sensitivity_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "run_sensitivity_tests.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script


def test_source_comparison_script_bootstraps_repo_root_imports():
    script = (ROOT / "scripts" / "compare_data_sources.py").read_text(encoding="utf-8")

    assert "sys.path.insert(0, str(ROOT))" in script
    assert "--left-provider" in script
    assert "--right-provider" in script
