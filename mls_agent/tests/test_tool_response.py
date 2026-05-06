"""Unit tests for mls_agent.tools.response."""
from __future__ import annotations

import pytest

from mls_agent.tools.response import (
    Event,
    Measurement,
    ToolErrorCode,
    ToolResponse,
    ToolStatus,
)


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


class TestEvent:
    def test_basic(self):
        e = Event(type="clock_locked", severity="warn", detail="locked at 1.5GHz")
        assert e.type == "clock_locked"
        assert e.severity == "warn"

    def test_empty_type_rejected(self):
        with pytest.raises(ValueError, match="type"):
            Event(type="", severity="info", detail="x")

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValueError, match="severity"):
            Event(type="x", severity="critical", detail="x")  # type: ignore[arg-type]

    def test_frozen(self):
        e = Event(type="x", severity="info", detail="d")
        with pytest.raises(Exception):
            e.detail = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class TestMeasurement:
    def test_basic(self):
        m = Measurement(
            metric="dram_bw",
            value=400.0,
            unit="GB/s",
            confidence=0.9,
            method="bandwidthTest",
            evidence=("stdout: 400.2 GB/s",),
        )
        assert m.metric == "dram_bw"
        assert m.evidence == ("stdout: 400.2 GB/s",)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            Measurement(
                metric="m",
                value=1,
                unit=None,
                confidence=1.5,
                method="x",
                evidence=("e",),
            )
        with pytest.raises(ValueError, match="confidence"):
            Measurement(
                metric="m",
                value=1,
                unit=None,
                confidence=-0.1,
                method="x",
                evidence=("e",),
            )

    def test_confidence_at_boundaries(self):
        Measurement(metric="m", value=1, unit=None, confidence=0.0,
                    method="x", evidence=("e",))
        Measurement(metric="m", value=1, unit=None, confidence=1.0,
                    method="x", evidence=("e",))

    def test_evidence_must_be_non_empty(self):
        with pytest.raises(ValueError, match="evidence"):
            Measurement(metric="m", value=1, unit=None, confidence=0.5,
                        method="x", evidence=())

    def test_evidence_must_be_tuple(self):
        with pytest.raises(TypeError, match="tuple"):
            Measurement(metric="m", value=1, unit=None, confidence=0.5,
                        method="x", evidence=["e"])  # type: ignore[arg-type]

    def test_empty_metric_rejected(self):
        with pytest.raises(ValueError, match="metric"):
            Measurement(metric="", value=1, unit=None, confidence=0.5,
                        method="x", evidence=("e",))

    def test_empty_method_rejected(self):
        with pytest.raises(ValueError, match="method"):
            Measurement(metric="m", value=1, unit=None, confidence=0.5,
                        method="", evidence=("e",))


# ---------------------------------------------------------------------------
# ToolResponse — base behavior + factories
# ---------------------------------------------------------------------------


class TestToolResponse:
    def test_success_factory(self):
        r = ToolResponse.success("ok", data={"x": 1})
        assert r.status == ToolStatus.SUCCESS
        assert r.text == "ok"
        assert r.data == {"x": 1}
        assert r.error_info is None
        assert r.terminate is False

    def test_partial_factory(self):
        r = ToolResponse.partial("timed out", data={"elapsed": 30})
        assert r.status == ToolStatus.PARTIAL
        assert r.error_info is None

    def test_error_factory(self):
        r = ToolResponse.error(ToolErrorCode.INVALID_ARGS, "missing x")
        assert r.status == ToolStatus.ERROR
        assert r.error_info == {
            "code": ToolErrorCode.INVALID_ARGS,
            "message": "missing x",
        }
        assert r.text == "missing x"

    def test_terminate_with_factory(self):
        r = ToolResponse.terminate_with(
            summary="all done", payload={"score": 0.95}
        )
        assert r.terminate is True
        assert r.terminate_summary == "all done"
        assert r.terminate_payload == {"score": 0.95}
        assert r.status == ToolStatus.SUCCESS

    def test_to_dict_omits_empty_error_and_stats(self):
        d = ToolResponse.success("ok").to_dict()
        assert "error" not in d
        assert "stats" not in d
        assert d == {"status": "success", "text": "ok", "data": {}}

    def test_to_dict_includes_error(self):
        d = ToolResponse.error("invalid_args", "x").to_dict()
        assert d["error"] == {"code": "invalid_args", "message": "x"}


# ---------------------------------------------------------------------------
# ToolResponse — invariants
# ---------------------------------------------------------------------------


class TestToolResponseInvariants:
    def test_error_status_requires_error_info(self):
        with pytest.raises(ValueError, match="error_info"):
            ToolResponse(status=ToolStatus.ERROR, text="fail")

    def test_non_error_with_error_info_rejected(self):
        with pytest.raises(ValueError, match="error_info"):
            ToolResponse(
                status=ToolStatus.SUCCESS,
                text="ok",
                error_info={"code": "x", "message": "y"},
            )

    def test_terminate_with_error_status_rejected(self):
        with pytest.raises(ValueError, match="terminate"):
            ToolResponse(
                status=ToolStatus.ERROR,
                text="fail",
                error_info={"code": "x", "message": "y"},
                terminate=True,
            )

    def test_terminate_payload_requires_terminate_flag(self):
        with pytest.raises(ValueError, match="terminate"):
            ToolResponse(
                status=ToolStatus.SUCCESS,
                text="ok",
                terminate=False,
                terminate_payload={"x": 1},
            )

    def test_terminate_summary_requires_terminate_flag(self):
        with pytest.raises(ValueError, match="terminate"):
            ToolResponse(
                status=ToolStatus.SUCCESS,
                text="ok",
                terminate=False,
                terminate_summary="done",
            )

    def test_events_must_be_tuple(self):
        with pytest.raises(TypeError, match="events"):
            ToolResponse(
                status=ToolStatus.SUCCESS,
                text="ok",
                events=[Event("x", "info", "d")],  # type: ignore[arg-type]
            )

    def test_measurements_must_be_tuple(self):
        with pytest.raises(TypeError, match="measurements"):
            ToolResponse(
                status=ToolStatus.SUCCESS,
                text="ok",
                measurements=[],  # type: ignore[arg-type]
            )

    def test_terminate_payload_must_be_json_serializable(self):
        class NotSerializable:
            pass

        with pytest.raises(ValueError, match="JSON"):
            ToolResponse.terminate_with(
                summary="x", payload={"obj": NotSerializable()}
            )

    def test_status_must_be_enum(self):
        with pytest.raises(TypeError, match="ToolStatus"):
            ToolResponse(status="success", text="ok")  # type: ignore[arg-type]

    def test_can_carry_events_and_measurements(self):
        ev = Event(type="x", severity="info", detail="d")
        m = Measurement(
            metric="m", value=1, unit=None, confidence=0.5,
            method="x", evidence=("e",),
        )
        r = ToolResponse.success("ok", events=(ev,), measurements=(m,))
        assert r.events == (ev,)
        assert r.measurements == (m,)
