"""Local development environment contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repo_conftest_prepares_writable_test_output() -> None:
    conftest = ROOT / "conftest.py"

    assert conftest.exists()
    spec = importlib.util.spec_from_file_location("repo_conftest", conftest)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    test_output = ROOT / ".test_output"
    assert test_output.exists()
    assert test_output.is_dir()

    probe = test_output / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "ok"
    probe.unlink()


def test_repo_conftest_uses_per_process_pytest_basetemp() -> None:
    conftest = ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location("repo_conftest", conftest)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Options:
        basetemp = str(ROOT / ".test_output" / "pytest-tmp")

    class Config:
        option = Options()

    config = Config()
    module.pytest_configure(config)

    basetemp = Path(config.option.basetemp)
    assert basetemp.parent == ROOT / ".test_output"
    assert basetemp.name.startswith("pytest-tmp-")
