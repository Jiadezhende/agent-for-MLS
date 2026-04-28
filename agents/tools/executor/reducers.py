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


def _clean_csv_cell(value: str | None) -> str:
    return (value or "").strip().strip('"')


def _read_csv_rows(raw_text: str) -> list[list[str]]:
    return [
        [_clean_csv_cell(cell) for cell in row]
        for row in csv.reader(io.StringIO(raw_text))
        if any(_clean_csv_cell(cell) for cell in row)
    ]


def _aggregate_ncu_samples(
    samples: dict[str, list[float | str]],
    metrics_requested: list[str],
) -> dict[str, float | str]:
    # Empty metrics_requested means "aggregate all found metrics"
    targets = metrics_requested if metrics_requested else list(samples.keys())
    metrics: dict[str, float | str] = {}
    for metric in targets:
        values = samples.get(metric, [])
        numeric_values = [v for v in values if isinstance(v, float)]
        if not numeric_values:
            if values:
                metrics[metric] = values[-1]
            continue
        if metric.endswith(".sum"):
            metrics[metric] = sum(numeric_values)
        else:
            metrics[metric] = sum(numeric_values) / len(numeric_values)
    return metrics


def _reduce_ncu_long_rows(rows: list[list[str]], metrics_requested: list[str]) -> dict:
    header_idx = next(
        i for i, row in enumerate(rows)
        if "Metric Name" in row and "Metric Value" in row
    )
    header = rows[header_idx]
    indexes = {name: idx for idx, name in enumerate(header)}
    collect_all = not metrics_requested
    samples: dict[str, list[float | str]] = {m: [] for m in metrics_requested}
    metric_units: dict[str, str] = {}
    kernel_names_seen: list[str] = []
    invocation_keys: set[tuple[str, str]] = set()
    notes: list[str] = []
    requested = set(metrics_requested)

    for idx, row in enumerate(rows[header_idx + 1:]):
        metric_name = row[indexes["Metric Name"]] if indexes["Metric Name"] < len(row) else ""
        kernel_idx = indexes.get("Kernel Name")
        kernel_name = row[kernel_idx] if kernel_idx is not None and kernel_idx < len(row) else ""
        if kernel_name and kernel_name not in kernel_names_seen:
            kernel_names_seen.append(kernel_name)
        if metric_name:
            id_idx = indexes.get("ID")
            instance_idx = indexes.get("Instance")
            invocation_id = str(idx)
            if id_idx is not None and id_idx < len(row) and row[id_idx]:
                invocation_id = row[id_idx]
            elif instance_idx is not None and instance_idx < len(row) and row[instance_idx]:
                invocation_id = row[instance_idx]
            invocation_keys.add((invocation_id, kernel_name))
        if not collect_all and metric_name not in requested:
            continue

        value_idx = indexes["Metric Value"]
        if value_idx >= len(row):
            notes.append(f"Metric Value column missing for {metric_name}")
            continue
        if metric_name not in samples:
            samples[metric_name] = []
        samples[metric_name].append(_parse_metric_value(row[value_idx]))
        unit_idx = indexes.get("Metric Unit")
        if unit_idx is not None and unit_idx < len(row) and row[unit_idx]:
            metric_units[metric_name] = row[unit_idx]

    metrics = _aggregate_ncu_samples(samples, metrics_requested)
    missing = [m for m in metrics_requested if m not in metrics]
    if missing:
        notes.append(f"Could not parse metrics exactly: {missing}")

    return {
        "metrics": metrics,
        "metric_units": metric_units,
        "missing_metrics": missing,
        "kernels_profiled": len(invocation_keys),
        "kernel_names_seen": kernel_names_seen,
        "notes": notes,
    }


def _reduce_ncu_wide_rows(rows: list[list[str]], metrics_requested: list[str]) -> dict:
    """Parse newer ncu raw CSV where each requested metric is a column."""
    header_idx = next(i for i, row in enumerate(rows) if "Kernel Name" in row)
    header = rows[header_idx]
    collect_all = not metrics_requested
    samples: dict[str, list[float | str]] = {m: [] for m in metrics_requested}
    metric_units: dict[str, str] = {}
    kernel_names_seen: list[str] = []
    invocation_keys: set[tuple[str, str]] = set()
    notes: list[str] = []

    # Non-metric columns to skip when collecting all
    _NON_METRIC_COLS = {"ID", "Kernel Name", "Kernel Time (ns)", "Context", "Stream",
                        "Block", "Grid", "Device", "Process ID", "Process Name",
                        "Host Name", "Section Name"}

    column_indexes: dict[str, int] = {}
    if collect_all:
        column_indexes = {
            name: idx for idx, name in enumerate(header)
            if name and name not in _NON_METRIC_COLS
        }
    else:
        for metric in metrics_requested:
            matches = [idx for idx, name in enumerate(header) if name == metric]
            if matches:
                column_indexes[metric] = matches[0]

    unit_row: list[str] | None = None
    data_start = header_idx + 1
    if data_start < len(rows):
        candidate = rows[data_start]
        # A unit row has no numeric first cell and contains unit strings like "ns", "%", "cycle"
        is_unit_row = bool(candidate) and (
            not candidate[0]
            or not any(c.isdigit() for c in candidate[0])
        ) and any(
            cell.lower() in {"ns", "%", "cycle", "cycles", "byte", "bytes",
                              "gb/s", "tb/s", "mhz", "ghz", "warp", "warps", "inst"}
            for cell in candidate
            if cell
        )
        if is_unit_row:
            unit_row = candidate
            data_start += 1

    kernel_idx = header.index("Kernel Name")
    id_idx = header.index("ID") if "ID" in header else None
    for row_offset, row in enumerate(rows[data_start:]):
        if kernel_idx >= len(row):
            continue
        kernel_name = row[kernel_idx]
        if not kernel_name:
            continue
        if kernel_name not in kernel_names_seen:
            kernel_names_seen.append(kernel_name)

        invocation_id = str(row_offset)
        if id_idx is not None and id_idx < len(row) and row[id_idx]:
            invocation_id = row[id_idx]
        invocation_keys.add((invocation_id, kernel_name))

        for metric, col_idx in column_indexes.items():
            if col_idx >= len(row) or row[col_idx] == "":
                continue
            if metric not in samples:
                samples[metric] = []
            samples[metric].append(_parse_metric_value(row[col_idx]))
            if unit_row is not None and col_idx < len(unit_row) and unit_row[col_idx]:
                metric_units[metric] = unit_row[col_idx]

    metrics = _aggregate_ncu_samples(samples, metrics_requested)
    missing = [m for m in metrics_requested if m not in metrics]
    if missing:
        notes.append(f"Could not parse metrics exactly: {missing}")
    if not column_indexes and not collect_all:
        notes.append("ncu CSV uses wide raw format, but no requested metric columns were present")

    return {
        "metrics": metrics,
        "metric_units": metric_units,
        "missing_metrics": missing,
        "kernels_profiled": len(invocation_keys),
        "kernel_names_seen": kernel_names_seen,
        "notes": notes,
    }


def _reduce_ncu(raw_text: str, metrics_requested: list[str]) -> dict:
    """Parse ncu --csv --page raw output with exact metric matching."""
    if not raw_text.strip():
        return {
            "metrics": {},
            "metric_units": {},
            "missing_metrics": list(metrics_requested),
            "kernels_profiled": 0,
            "kernel_names_seen": [],
            "notes": ["empty ncu output"],
        }

    try:
        rows = _read_csv_rows(raw_text)
    except Exception as exc:
        return {
            "metrics": {},
            "metric_units": {},
            "missing_metrics": list(metrics_requested),
            "kernels_profiled": 0,
            "kernel_names_seen": [],
            "notes": [f"CSV parse error: {exc}"],
        }

    try:
        if any("Metric Name" in row and "Metric Value" in row for row in rows):
            return _reduce_ncu_long_rows(rows, metrics_requested)
        if any("Kernel Name" in row for row in rows):
            return _reduce_ncu_wide_rows(rows, metrics_requested)
    except Exception as exc:
        return {
            "metrics": {},
            "metric_units": {},
            "missing_metrics": list(metrics_requested),
            "kernels_profiled": 0,
            "kernel_names_seen": [],
            "notes": [f"CSV parse error: {exc}"],
        }

    return {
        "metrics": {},
        "metric_units": {},
        "missing_metrics": list(metrics_requested),
        "kernels_profiled": 0,
        "kernel_names_seen": [],
        "notes": ["Could not find ncu CSV header row"],
    }


def _reduce_nsys(csv_text: str) -> dict:
    """Parse nsys stats --format csv output (cuda_gpu_kern_sum report).

    Expected input: CSV from `nsys stats <report>.nsys-rep --format csv --report cuda_gpu_kern_sum`.
    Falls back to raw text excerpt if CSV parsing fails.
    """
    result: dict[str, Any] = {"timeline_summary": [], "notes": []}
    if not csv_text.strip():
        result["notes"].append("empty nsys stats output")
        return result
    try:
        rows = _read_csv_rows(csv_text)
        if not rows:
            raise ValueError("no rows parsed")
        # Find header: first row that has a time/name column
        header_idx = next(
            (i for i, row in enumerate(rows)
             if any("name" in col.lower() for col in row)
             and any("time" in col.lower() or col.strip() == "%" for col in row)),
            None,
        )
        if header_idx is not None:
            header = rows[header_idx]
            data = [
                dict(zip(header, row))
                for row in rows[header_idx + 1:]
                if len(row) == len(header) and any(cell for cell in row)
            ]
            result["timeline_summary"] = data[:15]
        else:
            result["timeline_summary"] = rows[:15]
            result["notes"].append("nsys CSV header not recognized; raw rows returned")
    except Exception as exc:
        lines = csv_text.strip().splitlines()
        result["timeline_summary"] = lines[:50]
        result["notes"].append(f"nsys stats parse error: {exc}; raw excerpt returned")
    return result


