"""Project setup contract tests."""

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
