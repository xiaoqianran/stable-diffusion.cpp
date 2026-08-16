from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolate_cost_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCPP_COST_LOG", str(tmp_path / "cost-ledger.jsonl"))


@pytest.fixture
def current_help_text() -> str:
    return (FIXTURES / "sd_cli_help.txt").read_text(encoding="utf-8")


@pytest.fixture
def renamed_help_text() -> str:
    return (FIXTURES / "sd_cli_help_renamed.txt").read_text(encoding="utf-8")
