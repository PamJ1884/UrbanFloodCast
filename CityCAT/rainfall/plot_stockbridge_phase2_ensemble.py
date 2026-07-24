#!/usr/bin/env python3
"""Plot and validate all 100 Stockbridge Phase 2 rainfall hyetographs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


EXPECTED_EVENTS = 100
EXPECTED_INDICES = tuple(range(1, EXPECTED_EVENTS + 1))
FILENAME_PATTERN = re.compile(r"^Rainfall_Data_(?P<index>\d+)\.csv$")
REQUIRED_TIMESERIES_COLUMNS = {
    "time_seconds",
    "time_minutes",
    "intensity_mm_per_hour",
    "intensity_m_per_second",
    "cumulative_depth_mm",
}
REQUIRED_SUMMARY_COLUMNS = {
    "rainfall_index",
    "storm_id",
    "split",
    "target_total_depth_mm",
    "active_duration_seconds",
    "peak_time_seconds",
    "peak_intensity_mm_per_hour",
}


@dataclass(frozen=True)
class RainfallSeries:
    """Validated rainfall time series for one sequential event."""

    rainfall_index: int
    path: Path
    time_seconds: np.ndarray
    time_minutes: np.ndarray
    intensity_mm_per_hour: np.ndarray


@dataclass(frozen=True)
class SummaryRecord:
    """Summary metadata needed for plotting and validation."""

    rainfall_index: int
    storm_id: str
    split: str
    target_total_depth_mm: float
    active_duration_seconds: float
    peak_time_seconds: float
    peak_intensity_mm_per_hour: float


def parse_args() -> argparse.Namespace:
    """Parse input and figure output paths."""

    parser = argparse.ArgumentParser(
        description="Plot all 100 Stockbridge Phase 2 rainfall hyetographs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-storm rainfall CSV files",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path to phase2_core_rainfall_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the three figures will be written",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve and validate input paths and prepare the output directory."""

    input_dir = args.input_dir.expanduser().resolve()
    summary_path = args.summary_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Rainfall CSV input directory not found: {input_dir}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Phase 2 summary CSV not found: {summary_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Figure output path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError(f"Figure output directory is invalid: {output_dir}")
    return input_dir, summary_path, output_dir


def discover_timeseries_files(input_dir: Path) -> dict[int, Path]:
    """Discover exactly one CSV file for every rainfall index from 1 to 100."""

    csv_paths = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if len(csv_paths) != EXPECTED_EVENTS:
        raise ValueError(
            f"Expected exactly {EXPECTED_EVENTS} rainfall CSV files in {input_dir}, "
            f"found {len(csv_paths)}"
        )

    indexed_paths: dict[int, Path] = {}
    for path in csv_paths:
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"Could not determine a rainfall index from CSV filename: {path.name}"
            )
        rainfall_index = int(match.group("index"))
        if rainfall_index in indexed_paths:
            raise ValueError(
                f"Duplicate rainfall index {rainfall_index} in {indexed_paths[rainfall_index]} "
                f"and {path}"
            )
        indexed_paths[rainfall_index] = path

    missing = sorted(set(EXPECTED_INDICES) - set(indexed_paths))
    extra = sorted(set(indexed_paths) - set(EXPECTED_INDICES))
    if missing or extra:
        raise ValueError(
            f"Rainfall indices must be continuous from 1 to {EXPECTED_EVENTS}: "
            f"missing={missing}, extra={extra}"
        )
    return indexed_paths


def require_columns(
    fieldnames: Sequence[str] | None,
    required_columns: set[str],
    path: Path,
) -> None:
    """Raise a clear error when a CSV omits required columns."""

    available = set(fieldnames or [])
    missing = sorted(required_columns - available)
    if missing:
        raise ValueError(f"Required columns missing from {path}: {missing}")


def parse_finite_float(value: str | None, column: str, path: Path, row_number: int) -> float:
    """Parse one finite CSV numeric value with source context."""

    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column} value at row {row_number} of {path}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite {column} value at row {row_number} of {path}: {value!r}"
        )
    return parsed


def read_summary(summary_path: Path) -> dict[int, SummaryRecord]:
    """Read and validate one summary row for each rainfall index."""

    records: dict[int, SummaryRecord] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, REQUIRED_SUMMARY_COLUMNS, summary_path)
        for row_number, row in enumerate(reader, start=2):
            index_value = parse_finite_float(
                row.get("rainfall_index"), "rainfall_index", summary_path, row_number
            )
            rainfall_index = int(index_value)
            if index_value != rainfall_index:
                raise ValueError(
                    f"Non-integer rainfall_index at row {row_number} of {summary_path}: "
                    f"{index_value}"
                )
            if rainfall_index in records:
                raise ValueError(
                    f"Duplicate rainfall_index {rainfall_index} in summary {summary_path}"
                )
            storm_id = (row.get("storm_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not storm_id or not split:
                raise ValueError(
                    f"Empty storm_id or split at row {row_number} of {summary_path}"
                )
            values = {
                column: parse_finite_float(row.get(column), column, summary_path, row_number)
                for column in (
                    "target_total_depth_mm",
                    "active_duration_seconds",
                    "peak_time_seconds",
                    "peak_intensity_mm_per_hour",
                )
            }
            if values["target_total_depth_mm"] < 0.0:
                raise ValueError(
                    f"Negative total depth at row {row_number} of {summary_path}"
                )
            records[rainfall_index] = SummaryRecord(
                rainfall_index=rainfall_index,
                storm_id=storm_id,
                split=split,
                target_total_depth_mm=values["target_total_depth_mm"],
                active_duration_seconds=values["active_duration_seconds"],
                peak_time_seconds=values["peak_time_seconds"],
                peak_intensity_mm_per_hour=values["peak_intensity_mm_per_hour"],
            )

    missing = sorted(set(EXPECTED_INDICES) - set(records))
    extra = sorted(set(records) - set(EXPECTED_INDICES))
    if missing or extra or len(records) != EXPECTED_EVENTS:
        raise ValueError(
            f"Summary must contain exactly rainfall indices 1 to {EXPECTED_EVENTS}: "
            f"missing={missing}, extra={extra}, rows={len(records)}"
        )
    return records


def read_timeseries(path: Path, rainfall_index: int) -> RainfallSeries:
    """Read and validate one rainfall time-series CSV."""

    time_seconds: list[float] = []
    time_minutes: list[float] = []
    intensity_mm_per_hour: list[float] = []
    intensity_m_per_second: list[float] = []
    cumulative_depth_mm: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, REQUIRED_TIMESERIES_COLUMNS, path)
        for row_number, row in enumerate(reader, start=2):
            time_seconds.append(
                parse_finite_float(row.get("time_seconds"), "time_seconds", path, row_number)
            )
            time_minutes.append(
                parse_finite_float(row.get("time_minutes"), "time_minutes", path, row_number)
            )
            intensity_mm_per_hour.append(
                parse_finite_float(
                    row.get("intensity_mm_per_hour"),
                    "intensity_mm_per_hour",
                    path,
                    row_number,
                )
            )
            intensity_m_per_second.append(
                parse_finite_float(
                    row.get("intensity_m_per_second"),
                    "intensity_m_per_second",
                    path,
                    row_number,
                )
            )
            cumulative_depth_mm.append(
                parse_finite_float(
                    row.get("cumulative_depth_mm"),
                    "cumulative_depth_mm",
                    path,
                    row_number,
                )
            )

    if not time_seconds:
        raise ValueError(f"Rainfall time-series CSV contains no records: {path}")
    seconds = np.asarray(time_seconds, dtype=np.float64)
    minutes = np.asarray(time_minutes, dtype=np.float64)
    intensity_mmhr = np.asarray(intensity_mm_per_hour, dtype=np.float64)
    intensity_mps = np.asarray(intensity_m_per_second, dtype=np.float64)
    cumulative = np.asarray(cumulative_depth_mm, dtype=np.float64)
    if np.any(intensity_mmhr < 0.0) or np.any(intensity_mps < 0.0):
        raise ValueError(f"Negative rainfall intensity found in {path}")
    if np.any(cumulative < 0.0):
        raise ValueError(f"Negative cumulative rainfall depth found in {path}")
    if not np.allclose(minutes, seconds / 60.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"time_minutes is inconsistent with time_seconds in {path}")
    if seconds.size < 2 or np.any(np.diff(seconds) <= 0.0):
        raise ValueError(f"Rainfall timestamps must be strictly increasing in {path}")
    return RainfallSeries(
        rainfall_index=rainfall_index,
        path=path,
        time_seconds=seconds,
        time_minutes=minutes,
        intensity_mm_per_hour=intensity_mmhr,
    )


def load_ensemble(indexed_paths: dict[int, Path]) -> list[RainfallSeries]:
    """Load all series in strict rainfall-index order and compare time vectors."""

    series = [read_timeseries(indexed_paths[index], index) for index in EXPECTED_INDICES]
    reference_seconds = series[0].time_seconds
    reference_minutes = series[0].time_minutes
    for event in series[1:]:
        if not np.array_equal(event.time_seconds, reference_seconds) or not np.array_equal(
            event.time_minutes, reference_minutes
        ):
            raise ValueError(
                f"Time vector in {event.path} differs from {series[0].path}"
            )
    expected_seconds = np.arange(0, 7201, 300, dtype=np.float64)
    if not np.array_equal(reference_seconds, expected_seconds):
        raise ValueError("Rainfall timestamps must be exactly 0, 300, ..., 7200 seconds")
    if reference_minutes[0] != 0.0 or reference_minutes[-1] != 120.0:
        raise ValueError(
            f"Rainfall time range must be exactly 0 to 120 minutes, got "
            f"{reference_minutes[0]:g} to {reference_minutes[-1]:g}"
        )
    timestep_seconds = np.diff(reference_seconds)
    if not np.allclose(timestep_seconds, timestep_seconds[0], rtol=0.0, atol=1e-9):
        raise ValueError("Rainfall timestep is not constant")
    return series


def plot_heatmap(
    intensity_matrix: np.ndarray,
    times_minutes: np.ndarray,
    rainfall_indices: Sequence[int],
    output_path: Path,
    title: str,
    y_axis_label: str,
) -> None:
    """Write one rainfall intensity heatmap."""

    import matplotlib.pyplot as plt

    row_count = intensity_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(
        intensity_matrix,
        aspect="auto",
        cmap="viridis",
        origin="upper",
        extent=(times_minutes[0], times_minutes[-1], row_count + 0.5, 0.5),
    )
    ax.set_xticks(np.arange(0, 121, 15))
    y_positions = np.asarray([1, *range(10, row_count + 1, 10)], dtype=np.int64)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"R{rainfall_indices[position - 1]}" for position in y_positions])
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    colour_bar = fig.colorbar(image, ax=ax)
    colour_bar.set_label("Rainfall intensity (mm/h)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_overlay(
    intensity_matrix: np.ndarray,
    times_minutes: np.ndarray,
    output_path: Path,
) -> None:
    """Plot all event hyetographs plus the ensemble mean."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    for intensity in intensity_matrix:
        ax.plot(times_minutes, intensity, linewidth=0.6, alpha=0.2)
    ax.plot(
        times_minutes,
        np.mean(intensity_matrix, axis=0),
        linewidth=2.5,
        label="Ensemble mean",
    )
    ax.set_xlim(0, 120)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Rainfall intensity (mm/h)")
    ax.set_title("Stockbridge Phase 2 rainfall ensemble — 100 events")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_figures(
    series: Sequence[RainfallSeries],
    summary: dict[int, SummaryRecord],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Create the indexed heatmap, overlay, and depth-ordered heatmap."""

    times_minutes = series[0].time_minutes
    intensity_matrix = np.stack([event.intensity_mm_per_hour for event in series])
    rainfall_indices = [event.rainfall_index for event in series]
    heatmap_path = output_dir / "phase2_all_100_rainfall_heatmap.png"
    overlay_path = output_dir / "phase2_all_100_rainfall_overlay.png"
    ordered_path = output_dir / "phase2_rainfall_heatmap_ordered_by_depth.png"

    plot_heatmap(
        intensity_matrix,
        times_minutes,
        rainfall_indices,
        heatmap_path,
        "Stockbridge Phase 2 rainfall ensemble — 100 events",
        "Rainfall index",
    )
    plot_overlay(intensity_matrix, times_minutes, overlay_path)

    order = sorted(
        range(len(series)),
        key=lambda row: (
            summary[series[row].rainfall_index].target_total_depth_mm,
            series[row].rainfall_index,
        ),
    )
    ordered_matrix = intensity_matrix[np.asarray(order, dtype=np.int64)]
    ordered_indices = [rainfall_indices[row] for row in order]
    plot_heatmap(
        ordered_matrix,
        times_minutes,
        ordered_indices,
        ordered_path,
        "Stockbridge Phase 2 rainfall ensemble ordered by depth",
        "Events ordered by total depth",
    )

    output_paths = (heatmap_path, overlay_path, ordered_path)
    missing = [str(path) for path in output_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Expected figure files were not created: {missing}")
    return output_paths


def print_validation_report(
    series: Sequence[RainfallSeries],
    summary: dict[int, SummaryRecord],
    figure_paths: Sequence[Path],
) -> None:
    """Print the requested concise ensemble and output report."""

    times_seconds = series[0].time_seconds
    times_minutes = series[0].time_minutes
    intensity_matrix = np.stack([event.intensity_mm_per_hour for event in series])
    depths = np.asarray(
        [summary[index].target_total_depth_mm for index in EXPECTED_INDICES],
        dtype=np.float64,
    )
    timestep_seconds = float(np.diff(times_seconds)[0])
    print(f"Events loaded: {len(series)}")
    print(
        f"Time range: {times_seconds[0]:g} to {times_seconds[-1]:g} seconds "
        f"({times_minutes[0]:g} to {times_minutes[-1]:g} minutes)"
    )
    print(
        f"Timestep: {timestep_seconds:g} seconds "
        f"({timestep_seconds / 60.0:g} minutes)"
    )
    print(
        f"Rainfall intensity range: {np.min(intensity_matrix):.10g} to "
        f"{np.max(intensity_matrix):.10g} mm/h"
    )
    print(f"Total depth range: {np.min(depths):.10g} to {np.max(depths):.10g} mm")
    for path in figure_paths:
        print(f"Figure: {path}")


def main() -> None:
    """Load, validate, plot, and report the complete Phase 2 ensemble."""

    args = parse_args()
    input_dir, summary_path, output_dir = resolve_paths(args)
    indexed_paths = discover_timeseries_files(input_dir)
    summary = read_summary(summary_path)
    series = load_ensemble(indexed_paths)
    figure_paths = create_figures(series, summary, output_dir)
    print_validation_report(series, summary, figure_paths)


if __name__ == "__main__":
    main()
