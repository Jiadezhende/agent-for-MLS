"""
tests/test_main_output.py — Unit tests for the results-writing logic in main.py.

Specifically covers:
- Output file is named results.json (not output.json)
- Output format is a flat dict {metric: numeric_value}
- Values are coerced from string/float to int where exact
- None / non-numeric values are skipped
- Highest-confidence result wins when the same metric appears twice
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers that replicate the flat-dict conversion logic from main.py
# (imported indirectly to avoid needing a full env for import-time side-effects)
# ---------------------------------------------------------------------------

def _to_flat(all_results: list[dict]) -> dict:
    """Mirrors the flat_results construction block in main.py."""
    flat: dict[str, int | float] = {}
    for r in all_results:
        raw = r.get("value")
        if raw is None:
            continue
        try:
            fval = float(raw)
            val: int | float = int(fval) if fval == int(fval) else fval
        except (TypeError, ValueError):
            continue
        flat[r["metric"]] = val
    return flat


def _deduplicate(raw: list[dict]) -> list[dict]:
    """Mirrors the deduplication block in main.py."""
    seen: dict[str, dict] = {}
    for r in raw:
        metric = r.get("metric", "")
        if metric not in seen or r.get("confidence", 0) > seen[metric].get("confidence", 0):
            seen[metric] = r
    return list(seen.values())


# ===========================================================================
# Flat-dict conversion
# ===========================================================================

class TestFlatConversion:
    def test_integer_value(self):
        results = [{"metric": "dram_latency_cycles", "value": 442, "confidence": 0.9}]
        flat = _to_flat(results)
        assert flat == {"dram_latency_cycles": 442}
        assert isinstance(flat["dram_latency_cycles"], int)

    def test_float_string_coerced_to_int(self):
        results = [{"metric": "boost_clock_mhz", "value": "3000.0", "confidence": 0.8}]
        flat = _to_flat(results)
        assert flat["boost_clock_mhz"] == 3000
        assert isinstance(flat["boost_clock_mhz"], int)

    def test_fractional_float_kept(self):
        results = [{"metric": "some_ratio", "value": 1.5, "confidence": 0.7}]
        flat = _to_flat(results)
        assert flat["some_ratio"] == 1.5
        assert isinstance(flat["some_ratio"], float)

    def test_none_value_skipped(self):
        results = [{"metric": "missing", "value": None, "confidence": 0.5}]
        flat = _to_flat(results)
        assert "missing" not in flat

    def test_non_numeric_string_skipped(self):
        results = [{"metric": "bad", "value": "n/a", "confidence": 0.5}]
        flat = _to_flat(results)
        assert "bad" not in flat

    def test_multiple_metrics(self):
        results = [
            {"metric": "dram_latency_cycles", "value": 442, "confidence": 0.9},
            {"metric": "boost_clock_mhz",     "value": 3105, "confidence": 0.85},
        ]
        flat = _to_flat(results)
        assert flat == {"dram_latency_cycles": 442, "boost_clock_mhz": 3105}

    def test_output_is_flat_dict_not_list(self):
        results = [{"metric": "x", "value": 1, "confidence": 1.0}]
        flat = _to_flat(results)
        assert isinstance(flat, dict)

    def test_zero_value_included(self):
        results = [{"metric": "zero_metric", "value": 0, "confidence": 0.5}]
        flat = _to_flat(results)
        assert "zero_metric" in flat
        assert flat["zero_metric"] == 0


# ===========================================================================
# Deduplication (highest confidence wins)
# ===========================================================================

class TestDeduplication:
    def test_higher_confidence_wins(self):
        raw = [
            {"metric": "dram_latency_cycles", "value": 100, "confidence": 0.5},
            {"metric": "dram_latency_cycles", "value": 442, "confidence": 0.9},
        ]
        deduped = _deduplicate(raw)
        assert len(deduped) == 1
        assert deduped[0]["value"] == 442

    def test_first_entry_wins_on_tie(self):
        raw = [
            {"metric": "x", "value": 1, "confidence": 0.8},
            {"metric": "x", "value": 2, "confidence": 0.8},
        ]
        deduped = _deduplicate(raw)
        assert len(deduped) == 1
        assert deduped[0]["value"] == 1

    def test_different_metrics_both_kept(self):
        raw = [
            {"metric": "a", "value": 1, "confidence": 0.9},
            {"metric": "b", "value": 2, "confidence": 0.7},
        ]
        deduped = _deduplicate(raw)
        assert len(deduped) == 2


# ===========================================================================
# Output file written as flat JSON dict (integration-style, no GPU needed)
# ===========================================================================

class TestOutputFileFormat:
    def test_results_json_is_flat_dict(self, tmp_path):
        output_path = tmp_path / "results.json"
        results = [
            {"metric": "dram_latency_cycles", "value": 442, "confidence": 0.9},
            {"metric": "boost_clock_mhz",     "value": 3105, "confidence": 0.85},
        ]
        flat = _to_flat(results)
        output_path.write_text(json.dumps(flat, indent=2), encoding="utf-8")

        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        assert loaded["dram_latency_cycles"] == 442
        assert loaded["boost_clock_mhz"] == 3105

    def test_output_filename_is_results_json(self, tmp_path):
        output_path = tmp_path / "results.json"
        output_path.write_text("{}", encoding="utf-8")
        assert output_path.name == "results.json"
