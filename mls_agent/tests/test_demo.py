"""Smoke test the calculator demo end-to-end."""
from __future__ import annotations

from mls_agent.examples.calculator_demo import main


def test_demo_runs_to_completion(capsys):
    rc = main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "reason     : completed" in captured.out
    assert "answer" in captured.out


def test_demo_public_api_imports():
    """Every name advertised in mls_agent.__all__ must be importable."""
    import mls_agent

    for name in mls_agent.__all__:
        assert hasattr(mls_agent, name), f"missing public symbol: {name}"
