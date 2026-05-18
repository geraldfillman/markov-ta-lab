"""Project setup contract tests."""

import json

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_setup_plan_exists_and_names_bootstrap_scope():
    plan = ROOT / "docs" / "superpowers" / "plans" / "2026-05-16-markov-ta-lab-setup.md"

    assert plan.exists()
    text = plan.read_text(encoding="utf-8")
    assert "Bootstrap the Markov TA research lab" in text
    assert "Do not implement strategy logic" in text


def test_pytest_configuration_sets_pythonpath():
    config = ROOT / "pyproject.toml"

    assert config.exists()
    text = config.read_text(encoding="utf-8")
    assert 'pythonpath = ["." ]' in text or 'pythonpath = ["."]' in text
    assert 'testpaths = ["tests"]' in text


def test_readme_documents_supported_python_versions():
    readme = ROOT / "README.md"

    text = readme.read_text(encoding="utf-8")
    assert "Python 3.11 and 3.12" in text
    assert "Python 3.14" in text
    assert "GitHub Codespaces" in text
    assert 'pip install -e ".[dev,notebook]"' in text


def test_known_unimplemented_surfaces_are_documented():
    backtests = (ROOT / "src" / "backtests.py").read_text(encoding="utf-8")
    metrics = (ROOT / "src" / "metrics.py").read_text(encoding="utf-8")
    plotting = (ROOT / "src" / "plotting.py").read_text(encoding="utf-8")

    assert "Known unimplemented surface" in backtests
    assert "run_vectorbt_sweep" in backtests
    assert "Known unimplemented surface" in metrics
    assert "adverse_excursion" in metrics
    assert "favorable_excursion" in metrics
    assert "Known unimplemented surface" in plotting


def test_codespaces_devcontainer_contract():
    devcontainer = ROOT / ".devcontainer" / "devcontainer.json"

    assert devcontainer.exists()
    config = json.loads(devcontainer.read_text(encoding="utf-8"))
    assert "python:1-3.11" in config["image"]
    assert config["postCreateCommand"] == 'python -m pip install --upgrade pip && pip install -e ".[dev,notebook]"'
    assert "ms-python.python" in config["customizations"]["vscode"]["extensions"]
    assert "ms-toolsai.jupyter" in config["customizations"]["vscode"]["extensions"]
