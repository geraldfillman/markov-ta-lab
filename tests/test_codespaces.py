"""Codespaces development-container contract tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_codespaces_devcontainer_uses_python_311_editable_install() -> None:
    devcontainer = ROOT / ".devcontainer" / "devcontainer.json"

    assert devcontainer.exists()
    config = json.loads(devcontainer.read_text(encoding="utf-8"))
    assert "python:1-3.11" in config["image"]
    assert config["postCreateCommand"] == 'python -m pip install --upgrade pip && pip install -e ".[dev,notebook]"'
    assert "ms-python.python" in config["customizations"]["vscode"]["extensions"]
    assert "ms-toolsai.jupyter" in config["customizations"]["vscode"]["extensions"]
