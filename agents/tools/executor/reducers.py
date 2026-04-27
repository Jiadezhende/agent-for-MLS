from __future__ import annotations

import csv
import io
from typing import Any


def _parse_metric_value(value: str) -> float | str:
    cleaned = value.strip().strip('"').replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return value.strip().strip('"')


def _find_ncu_csv_start(raw_text: str) -> str:
    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        if "Metric Name" in line:
            return "\n".join(lines[i:])
    return raw_text


def _reduce_ncu(raw_text: str, metrics_requested: list[str]) -> dict:
    """Parse ncu --csv --page raw output with exact Metric Name matching."""
    notes: list[str] = []
    samples: dict[str, list[float | str]] = {m: [] for m in metrics_requested}
    metric_units: dict[str, str] = {}
    kernel_names_seen: list[str] = []
    invocation_keys: set[tuple[str, str]] = set()

    if not raw_text.strip():
        return {
            "metrics": {},
            "metric_samples": samples,
            "metric_units": metric_units,
            "missing_metrics": list(metrics_requested),
            "kernels_profiled": 0,
            "kernel_names_seen": [],
            "notes": ["empty ncu output"],
        }

    try:
        reader = csv.DictReader(io.StringIO(_find_ncu_csv_start(raw_text)))
        rows = list(reader)
    except Exception as exc:
        return {
            "metrics": {},
            "metric_samples": samples,
            "metric_units": metric_units,
            "missing_metrics": list(metrics_requested),
            "kernels_profiled": 0,
            "kernel_names_seen": [],
            "notes": [f"CSV parse error: {exc}"],
        }

    requested = set(metrics_requested)
    for idx, row in enumerate(rows):
        metric_name = (row.get("Metric Name") or "").strip().strip('"')
        kernel_name = (row.get("Kernel Name") or "").strip().strip('"')
        if kernel_name and kernel_name not in kernel_names_seen:
            kernel_names_seen.append(kernel_name)
        if metric_name:
            invocation_id = (row.get("ID") or row.get("Instance") or str(idx)).strip()
            invocation_keys.add((invocation_id, kernel_name))
        if metric_name not in requested:
            continue

        raw_value = row.get("Metric Value")
        if raw_value is None:
            notes.append(f"Metric Value column missing for {metric_name}")
            continue
        value = _parse_metric_value(raw_value)
        samples[metric_name].append(value)
        unit = (row.get("Metric Unit") or "").strip().strip('"')
        if unit:
            metric_units[metric_name] = unit

    metrics: dict[str, float | str] = {}
    for metric, values in samples.items():
        numeric_values = [v for v in values if isinstance(v, float)]
        if not numeric_values:
            if values:
                metrics[metric] = values[-1]
            continue
        if metric.endswith(".sum"):
            metrics[metric] = sum(numeric_values)
        else:
            metrics[metric] = sum(numeric_values) / len(numeric_values)

    missing = [m for m in metrics_requested if m not in metrics]
    if missing:
        notes.append(f"Could not parse metrics exactly: {missing}")

    return {
        "metrics": metrics,
        "metric_samples": samples,
        "metric_units": metric_units,
        "missing_metrics": missing,
        "kernels_profiled": len(invocation_keys),
        "kernel_names_seen": kernel_names_seen,
        "notes": notes,
    }


def _reduce_nsys(raw_text: str) -> dict:
    """Extract top GPU kernels from nsys stats text output."""
    lines = raw_text.strip().splitlines()
    result: dict[str, Any] = {"timeline_summary": [], "notes": []}

    in_table = False
    header: list[str] = []
    rows: list[dict] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                in_table = False
            continue
        if "Time (%)" in stripped and not in_table:
            header = [h.strip() for h in stripped.split(",")]
            in_table = True
            continue
        if in_table:
            parts = stripped.split(",")
            if len(parts) == len(header):
                rows.append(dict(zip(header, [p.strip() for p in parts])))

    if rows:
        result["timeline_summary"] = rows[:10]
    else:
        result["timeline_summary"] = lines[:50]
        result["notes"].append("Could not parse nsys stats table; raw excerpt returned")

    return result


def _reduce_torch(raw_text: str) -> dict:
    """Parse torch.profiler text output and return top operators by self CPU time."""
    lines = raw_text.strip().splitlines()
    result: dict[str, Any] = {"op_stats": [], "notes": []}

    header_idx = -1
    for i, line in enumerate(lines):
        if "Self CPU" in line or "CPU total" in line:
            header_idx = i
            break

    if header_idx >= 0:
        data_lines = lines[header_idx + 1:]
        ops = []
        for line in data_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("-"):
                continue
            ops.append(stripped)
            if len(ops) >= 20:
                break
        result["op_stats"] = ops
    else:
        result["op_stats"] = lines[:30]
        result["notes"].append("Could not parse torch profiler table; raw excerpt returned")

    return result

