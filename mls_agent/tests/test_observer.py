"""Unit tests for StdoutObserver — line truncation semantics."""
from __future__ import annotations

import io

from mls_agent.runtime.observer import StdoutObserver
from mls_agent.tools.response import ToolResponse, ToolStatus


def _make_obs(*, truncate_response_at, truncate_arg_log_at=120):
    buf = io.StringIO()
    obs = StdoutObserver(
        prefix="",
        stream=buf,
        truncate_arg_log_at=truncate_arg_log_at,
        truncate_response_at=truncate_response_at,
    )
    return obs, buf


def test_clip_honors_per_call_limit_when_truncate_is_none():
    """truncate_response_at=None must still apply the per-call display
    limit (200 for success, 80 for error) — otherwise 32 KB read_skill
    responses flood the log."""
    obs, _ = _make_obs(truncate_response_at=None)
    long_text = "x" * 1000
    clipped = obs._clip(long_text, 200)
    assert clipped.startswith("x" * 200)
    assert "<+800 chars>" in clipped


def test_clip_widens_to_configured_response_cap():
    """When truncate_response_at is explicitly set, it overrides the
    per-call limit. Lets orchestrator give file traces more context."""
    obs, _ = _make_obs(truncate_response_at=500)
    long_text = "y" * 1000
    clipped = obs._clip(long_text, 200)
    assert clipped.startswith("y" * 500)
    assert "<+500 chars>" in clipped


def test_clip_shorter_than_limit_returns_unchanged():
    obs, _ = _make_obs(truncate_response_at=None)
    assert obs._clip("hello", 200) == "hello"


def test_clip_zero_or_negative_limit_returns_text():
    obs, _ = _make_obs(truncate_response_at=0)
    assert obs._clip("hello" * 100, 200) == "hello" * 100


def test_summarize_response_clips_skill_payload():
    """End-to-end via _summarize_response: a 32 KB success response
    must NOT appear in full when truncate_response_at is None."""
    obs, _ = _make_obs(truncate_response_at=None)
    big = "z" * 32_000
    resp = ToolResponse(status=ToolStatus.SUCCESS, text=big)
    line = obs._summarize_response(resp)
    # Display line must be much shorter than the underlying response.
    assert len(line) < 1000
    assert "<+" in line and "chars>" in line
