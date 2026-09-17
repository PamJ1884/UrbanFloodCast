"""Read FEH design-rainfall point exports directly into the DDF layer.

Only the variable-duration, annual-maximum point format is supported. Depths
are already millimetres; no scaling, interpolation or spatial work is done.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .ddf import DDFRecord, DDFTable, _duration_seconds, _positive_number


_RETURN_PERIOD_COLUMN = re.compile(r"(.+) year rainfall \(mm\)")
_DURATION_COLUMNS = ("Duration hours", "Duration days")


@dataclass(frozen=True)
class FEHPointDDFMetadata:
    """Source provenance, with CSV field text and ordering preserved.

    ``preamble_text`` retains original quoting and line endings (UTF-8 BOM
    excluded). ``parameters`` contains all '=' metadata fields, including
    trailing empty fields. ``parsed`` contains named, stripped source values;
    its export timestamp is source text, never a processing timestamp.
    Column indices are zero-based. No CRS is inferred from location fields.
    """

    source_path: Path
    preamble_text: str
    preamble_rows: tuple[tuple[str, ...], ...]
    parameters: Mapping[str, tuple[str, ...]]
    parsed: Mapping[str, str]
    header_columns: tuple[str, ...]
    duration_hours_column: int
    duration_days_column: int
    return_period_columns: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class FEHPointDDFResult:
    """Validated DDF surface and the metadata from its original point export."""

    table: DDFTable
    metadata: FEHPointDDFMetadata


def _trimmed_fields(fields: tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(value.strip() for value in fields)
    while values and not values[-1]:
        values = values[:-1]
    return values


def _parse_metadata(preamble):
    parameters: dict[str, tuple[str, ...]] = {}
    parsed: dict[str, str] = {}
    for row in preamble:
        if not row:
            continue
        label = row[0].strip()
        if label.endswith("="):
            name = label[:-1].strip()
            if name in parameters:
                raise ValueError(f"ambiguous duplicate FEH metadata: {name}")
            parameters[name] = row[1:]
        elif label == "VERSION":
            if "source_version" in parsed:
                raise ValueError("ambiguous duplicate FEH VERSION metadata")
            values = _trimmed_fields(row)
            if (
                len(values) != 9
                or values[2] != "Version"
                or values[4] != "exported at"
                or any(not value for value in values)
            ):
                raise ValueError("unsupported FEH VERSION/export timestamp metadata")
            parsed.update(
                source_service=values[1],
                source_version=values[3],
                source_export_time=values[5],
                source_export_timezone=values[6],
                source_export_weekday=values[7],
                source_export_date=values[8],
                source_export_timestamp=" ".join(values[5:]),
            )

    for name, key, expected in (
        ("Rainfall model", "rainfall_model", None),
        ("Calculation type", "calculation_type", "Design rainfall"),
        ("Calculation mode", "calculation_mode", "For a point"),
        ("Fixed duration", "fixed_duration", "no"),
        ("Annual maximum", "annual_maximum", "yes"),
    ):
        values = _trimmed_fields(parameters.get(name, ()))
        if len(values) != 1 or not values[0]:
            raise ValueError(f"missing or ambiguous FEH metadata: {name}")
        if expected is not None and values[0] != expected:
            raise ValueError(
                f"unsupported FEH {name}: {values[0]!r}; expected {expected!r}"
            )
        parsed[key] = values[0]

    location = _trimmed_fields(parameters.get("Calculation location", ()))
    if len(location) != 5 or location[0] != "Point" or not all(location):
        raise ValueError("missing or unsupported FEH Calculation location fields")
    for value in location[2:4]:
        try:
            coordinate = float(value)
        except ValueError as exc:
            raise ValueError("FEH source coordinates must be finite numeric values") from exc
        if not math.isfinite(coordinate):
            raise ValueError("FEH source coordinates must be finite numeric values")
    parsed.update(zip(
        ("source_location_type", "source_location_region", "source_x",
         "source_y", "source_grid_reference"),
        location,
    ))
    return MappingProxyType(parameters), MappingProxyType(parsed)


def load_feh_point_ddf_csv(
    path: Path | str, *, location_id: str
) -> FEHPointDDFResult:
    """Load one annual-maximum FEH design-rainfall point CSV.

    Require an explicit non-empty caller location ID and the source settings
    ``Design rainfall``, ``For a point``, ``Fixed duration=no`` and
    ``Annual maximum=yes``. The rainfall model is retained as supplied.
    Locate exactly one table after a variable-length metadata preamble;
    support reordered columns with explicit ``<T> year rainfall (mm)`` labels.

    Hours determine minutes and whole seconds using the existing DDF helpers.
    Days must be finite and positive but may be rounded: retain their original
    text in every record without using them for conversion or requiring exact
    agreement with hours. Preserve original depth text as provenance alongside
    its unscaled float value. Return immutable metadata and record provenance.

    Raise ValueError for unreadable, malformed, ambiguous or unsupported input,
    invalid numbers, fractional seconds, duplicate columns or duplicate keys.
    No climate scenario, uplift, EPSG code or catchment role is inferred.
    """

    if not isinstance(location_id, str) or not location_id.strip():
        raise ValueError("location_id must be a non-empty string")
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open(encoding="utf-8-sig", newline="") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read FEH point CSV: {source_path}") from exc

    reader = csv.reader(lines, strict=True)
    rows: list[tuple[int, tuple[str, ...]]] = []
    try:
        while True:
            line_number = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break
            rows.append((line_number, tuple(row)))
    except csv.Error as exc:
        raise ValueError(f"malformed FEH CSV near line {reader.line_num}: {exc}") from exc

    headers = [
        index for index, (_, row) in enumerate(rows)
        if any(value.strip() in _DURATION_COLUMNS for value in row)
    ]
    if len(headers) != 1:
        raise ValueError("expected exactly one FEH rainfall-table header; missing or ambiguous")
    header_index = headers[0]
    header_line, original_columns = rows[header_index]
    columns = tuple(value.strip() for value in original_columns)
    for name in _DURATION_COLUMNS:
        if columns.count(name) != 1:
            raise ValueError(f"missing or duplicate FEH duration column: {name}")
    periods: list[tuple[int, float]] = []
    seen_periods: set[float] = set()
    for column, label in enumerate(columns):
        if label in _DURATION_COLUMNS:
            continue
        match = _RETURN_PERIOD_COLUMN.fullmatch(label)
        if match is None:
            raise ValueError(f"unsupported FEH table column: {label!r}")
        period = _positive_number(match[1], f"return period in column {label!r}")
        if period in seen_periods:
            raise ValueError(f"duplicate FEH return-period column: {label!r}")
        seen_periods.add(period)
        periods.append((column, period))
    if not periods:
        raise ValueError("FEH table requires return-period columns")

    preamble = tuple(row for _, row in rows[:header_index])
    parameters, parsed = _parse_metadata(preamble)
    hours_column = columns.index("Duration hours")
    days_column = columns.index("Duration days")
    metadata = FEHPointDDFMetadata(
        source_path=source_path,
        preamble_text="".join(lines[:header_line - 1]),
        preamble_rows=preamble,
        parameters=parameters,
        parsed=parsed,
        header_columns=original_columns,
        duration_hours_column=hours_column,
        duration_days_column=days_column,
        return_period_columns=tuple(periods),
    )
    common_provenance = {
        "source": "FEH point rainfall CSV",
        "source_file_path": str(source_path),
        **parsed,
        "source_duration_hours_column": original_columns[hours_column],
        "source_duration_days_column": original_columns[days_column],
    }
    records: list[DDFRecord] = []
    seen_durations: set[int] = set()
    for line_number, row in rows[header_index + 1:]:
        context = f"at FEH CSV line {line_number}"
        if len(row) != len(columns) or any(not value.strip() for value in row):
            raise ValueError(f"malformed or incomplete data row {context}")
        hours = _positive_number(row[hours_column], f"Duration hours {context}")
        _positive_number(row[days_column], f"Duration days {context}")
        minutes = _positive_number(hours * 60.0, f"duration_minutes {context}")
        _positive_number(minutes * 60.0, f"duration_seconds {context}")
        try:
            seconds = _duration_seconds(minutes)
        except ValueError as exc:
            raise ValueError(f"{exc} {context}") from exc
        if seconds <= 0:
            raise ValueError(f"duration must represent positive whole seconds {context}")
        if seconds in seen_durations:
            raise ValueError(f"duplicate DDF key (duplicate duration) {context}")
        seen_durations.add(seconds)
        for column, period in periods:
            depth = _positive_number(row[column], f"depth_mm in {columns[column]!r} {context}")
            provenance = MappingProxyType({
                **common_provenance,
                "source_csv_line": str(line_number),
                "source_duration_hours": row[hours_column],
                "source_duration_days": row[days_column],
                "source_return_period_column": original_columns[column],
                "source_return_period_column_index": str(column),
                "source_depth_mm": row[column],
            })
            records.append(DDFRecord(
                location_id, minutes, seconds, period, depth, provenance,
            ))
    if not records:
        raise ValueError("FEH CSV contains no data records")
    return FEHPointDDFResult(DDFTable(location_id, tuple(records)), metadata)
