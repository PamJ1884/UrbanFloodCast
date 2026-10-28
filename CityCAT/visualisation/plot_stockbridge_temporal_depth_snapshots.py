#!/usr/bin/env python3
"""Create temporal water-depth maps from one Stockbridge CityCAT simulation."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


OUTPUT_NODATA = -9999.0
RAINFALL_RECORDS = 25
RAINFALL_TIMESTEP_S = 300
SIMULATION_HORIZON_S = 7200
AREA_THRESHOLDS_M = (0.01, 0.10, 0.30)
REQUIRED_RSL_COLUMNS = ("XCen", "YCen", "Depth", "Vx", "Vy")
RSL_TIME_METADATA_PATTERN = re.compile(
    r"^T_(?P<seconds>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)_sec$"
)
REQUIRED_SUMMARY_COLUMNS = {
    "rainfall_index",
    "split",
    "target_total_depth_mm",
    "active_duration_minutes",
    "peak_time_minutes",
}


@dataclass(frozen=True)
class DemGrid:
    """DEM values and target raster-grid metadata."""

    data: np.ndarray
    active_mask: np.ndarray
    width: int
    height: int
    transform: Any
    crs: Any
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    pixel_area_m2: float
    profile: dict[str, Any]


@dataclass(frozen=True)
class EventMetadata:
    """Rainfall summary metadata for the selected event."""

    rainfall_index: int
    split: str
    target_total_depth_mm: float
    active_duration_minutes: float
    peak_time_minutes: float


@dataclass(frozen=True)
class RainfallSeries:
    """Validated CityCAT rainfall forcing and cumulative depth."""

    times_seconds: np.ndarray
    intensity_m_per_second: np.ndarray
    intensity_mm_per_hour: np.ndarray
    cumulative_depth_mm: np.ndarray


@dataclass(frozen=True)
class SnapshotDepth:
    """One parsed and spatially validated RSL depth snapshot."""

    event: int
    timestep_index: int
    time_minutes: int
    rsl_path: Path
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    depths_m: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    coordinate_set: frozenset[tuple[float, float]]


@dataclass(frozen=True)
class SnapshotResult:
    """Temporal forcing, hydraulic metrics, and output paths."""

    event: int
    timestep_index: int
    time_minutes: int
    rainfall_intensity_mm_per_hour: float
    cumulative_rainfall_mm: float
    hydraulic_cell_count: int
    wet_cell_count_ge_0_01m: int
    maximum_depth_m: float
    p99_depth_m: float
    mean_depth_wet_cells_m: float
    area_ge_0_01_m2: float
    area_ge_0_10_m2: float
    area_ge_0_30_m2: float
    rsl_path: Path
    geotiff_path: Path
    png_path: Path
    depth_raster: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse input, event, snapshot, display, and output options."""

    parser = argparse.ArgumentParser(
        description="Plot temporal Stockbridge water-depth snapshots from CityCAT RSL files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing R<N>C1_SurfaceMaps event folders",
    )
    parser.add_argument(
        "--dem-path",
        type=Path,
        required=True,
        help="Path to the georeferenced Stockbridge DEM GeoTIFF",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path to phase2_core_rainfall_summary.csv",
    )
    parser.add_argument(
        "--rainfall-path",
        type=Path,
        required=True,
        help="Path to the corresponding CityCAT Rainfall_Data_<N>.txt file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for GeoTIFF, PNG, and diagnostic CSV outputs",
    )
    parser.add_argument(
        "--event",
        type=int,
        default=3,
        help="Rainfall index (default: %(default)s)",
    )
    parser.add_argument(
        "--times-minutes",
        type=int,
        nargs="+",
        default=[30, 60, 90, 120],
        help="Snapshot times in minutes (default: 30 60 90 120)",
    )
    parser.add_argument(
        "--display-min-depth",
        type=float,
        default=0.01,
        help="Minimum displayed water depth in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--display-max-depth",
        type=float,
        default=1.20,
        help="Maximum colour-scale depth in metres (default: %(default)s)",
    )
    return parser.parse_args()


def validate_cli_values(args: argparse.Namespace) -> None:
    """Validate event, requested times, and shared colour limits."""

    if args.event <= 0:
        raise ValueError("--event must be a positive rainfall index")
    if not args.times_minutes:
        raise ValueError("At least one snapshot time must be supplied")
    if len(set(args.times_minutes)) != len(args.times_minutes):
        raise ValueError(f"Duplicate snapshot times supplied: {args.times_minutes}")
    for time_minutes in args.times_minutes:
        if time_minutes < 0 or time_minutes > SIMULATION_HORIZON_S // 60:
            raise ValueError(
                f"Snapshot time must lie between 0 and 120 minutes: {time_minutes}"
            )
        if (time_minutes * 60) % RAINFALL_TIMESTEP_S != 0:
            raise ValueError(
                f"Snapshot time must align to a 5-minute interval: {time_minutes}"
            )
    if not math.isfinite(args.display_min_depth) or args.display_min_depth < 0.0:
        raise ValueError("--display-min-depth must be finite and non-negative")
    if not math.isfinite(args.display_max_depth) or (
        args.display_max_depth <= args.display_min_depth
    ):
        raise ValueError("--display-max-depth must be finite and exceed the display minimum")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve source paths and create the output directory."""

    results_dir = args.results_dir.expanduser().resolve()
    dem_path = args.dem_path.expanduser().resolve()
    summary_path = args.summary_path.expanduser().resolve()
    rainfall_path = args.rainfall_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"CityCAT results directory not found: {results_dir}")
    for label, path in (
        ("DEM", dem_path),
        ("rainfall summary", summary_path),
        ("CityCAT rainfall", rainfall_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError(f"Output directory is invalid: {output_dir}")
    return results_dir, dem_path, summary_path, rainfall_path, output_dir


def load_dem(dem_path: Path) -> DemGrid:
    """Read and validate the georeferenced target DEM."""

    import rasterio

    with rasterio.open(dem_path) as source:
        if source.count != 1:
            raise ValueError(f"DEM must contain exactly one band: {dem_path}")
        if source.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        if source.nodata is None:
            raise ValueError(f"DEM has no defined NoData value: {dem_path}")
        transform_values = tuple(float(value) for value in source.transform[:6])
        if not all(math.isfinite(value) for value in transform_values):
            raise ValueError(f"DEM transform contains non-finite values: {dem_path}")
        determinant = source.transform.a * source.transform.e - source.transform.b * source.transform.d
        pixel_area_m2 = abs(float(determinant))
        if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0.0:
            raise ValueError(f"DEM transform has a non-positive pixel area: {dem_path}")
        resolution = tuple(float(value) for value in source.res)
        if not all(math.isfinite(value) and value > 0.0 for value in resolution):
            raise ValueError(f"DEM resolution must be finite and positive: {resolution}")
        data = source.read(1).astype(np.float64)
        nodata = float(source.nodata)
        finite_mask = np.isfinite(data)
        active_mask = finite_mask if math.isnan(nodata) else finite_mask & (data != nodata)
        if not np.any(active_mask):
            raise ValueError(f"DEM contains no valid cells: {dem_path}")
        return DemGrid(
            data=data,
            active_mask=active_mask,
            width=int(source.width),
            height=int(source.height),
            transform=source.transform,
            crs=source.crs,
            bounds=tuple(float(value) for value in source.bounds),
            resolution=resolution,
            pixel_area_m2=pixel_area_m2,
            profile=source.profile.copy(),
        )


def require_columns(
    fieldnames: Sequence[str] | None,
    required: set[str],
    source_path: Path,
) -> None:
    """Raise a clear error for missing CSV columns."""

    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise ValueError(f"Required columns missing from {source_path}: {missing}")


def parse_finite_float(
    value: str | None,
    field_name: str,
    source_path: Path,
    line_number: int,
) -> float:
    """Parse one finite number with source-line context."""

    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} at line {line_number} of {source_path}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite {field_name} at line {line_number} of {source_path}: {value!r}"
        )
    return parsed


def read_event_metadata(summary_path: Path, event: int) -> EventMetadata:
    """Read and validate summary metadata for the selected rainfall index."""

    selected: EventMetadata | None = None
    seen_indices: set[int] = set()
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, REQUIRED_SUMMARY_COLUMNS, summary_path)
        for line_number, row in enumerate(reader, start=2):
            index_value = parse_finite_float(
                row.get("rainfall_index"), "rainfall_index", summary_path, line_number
            )
            rainfall_index = int(index_value)
            if index_value != rainfall_index:
                raise ValueError(
                    f"Non-integer rainfall_index at line {line_number} of {summary_path}"
                )
            if rainfall_index in seen_indices:
                raise ValueError(
                    f"Duplicate rainfall_index {rainfall_index} in {summary_path}"
                )
            seen_indices.add(rainfall_index)
            if rainfall_index != event:
                continue
            split = (row.get("split") or "").strip()
            if not split:
                raise ValueError(f"Empty split for R{event} in {summary_path}")
            total_depth = parse_finite_float(
                row.get("target_total_depth_mm"),
                "target_total_depth_mm",
                summary_path,
                line_number,
            )
            active_duration = parse_finite_float(
                row.get("active_duration_minutes"),
                "active_duration_minutes",
                summary_path,
                line_number,
            )
            peak_time = parse_finite_float(
                row.get("peak_time_minutes"),
                "peak_time_minutes",
                summary_path,
                line_number,
            )
            if min(total_depth, active_duration, peak_time) < 0.0:
                raise ValueError(f"Negative rainfall metadata for R{event} in {summary_path}")
            selected = EventMetadata(
                rainfall_index=event,
                split=split,
                target_total_depth_mm=total_depth,
                active_duration_minutes=active_duration,
                peak_time_minutes=peak_time,
            )
    if selected is None:
        raise ValueError(f"Rainfall index {event} is missing from {summary_path}")
    return selected


def parse_rainfall(rainfall_path: Path) -> RainfallSeries:
    """Parse the 25 CityCAT records and integrate cumulative rainfall by trapezoids."""

    lines = rainfall_path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != 5 + RAINFALL_RECORDS:
        raise ValueError(
            f"CityCAT rainfall file must contain five header lines and exactly "
            f"{RAINFALL_RECORDS} numeric records: {rainfall_path} has {len(lines)} lines"
        )
    times_seconds: list[float] = []
    intensity_mps: list[float] = []
    for record_number, line in enumerate(lines[5:], start=1):
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(
                f"Malformed rainfall record {record_number} in {rainfall_path}: {line!r}"
            )
        time_seconds = parse_finite_float(
            tokens[0], "time_seconds", rainfall_path, record_number + 5
        )
        intensity = parse_finite_float(
            tokens[1], "intensity_metres_per_second", rainfall_path, record_number + 5
        )
        if intensity < 0.0:
            raise ValueError(
                f"Negative rainfall intensity in record {record_number} of {rainfall_path}"
            )
        times_seconds.append(time_seconds)
        intensity_mps.append(intensity)

    times = np.asarray(times_seconds, dtype=np.float64)
    expected_times = np.arange(
        0,
        SIMULATION_HORIZON_S + RAINFALL_TIMESTEP_S,
        RAINFALL_TIMESTEP_S,
        dtype=np.float64,
    )
    if not np.array_equal(times, expected_times):
        raise ValueError(
            f"Rainfall timestamps must be exactly 0, 300, ..., 7200 seconds: {rainfall_path}"
        )
    mps = np.asarray(intensity_mps, dtype=np.float64)
    mm_per_hour = mps * 1000.0 * 3600.0
    interval_depths_mm = 0.5 * (mm_per_hour[:-1] + mm_per_hour[1:]) * (
        RAINFALL_TIMESTEP_S / 3600.0
    )
    cumulative_depth_mm = np.concatenate(([0.0], np.cumsum(interval_depths_mm)))
    return RainfallSeries(
        times_seconds=times,
        intensity_m_per_second=mps,
        intensity_mm_per_hour=mm_per_hour,
        cumulative_depth_mm=cumulative_depth_mm,
    )


def validate_rainfall_against_summary(
    rainfall: RainfallSeries,
    metadata: EventMetadata,
    rainfall_path: Path,
) -> None:
    """Confirm the forcing's integrated depth agrees with its summary metadata."""

    integrated_depth = float(rainfall.cumulative_depth_mm[-1])
    if not math.isclose(
        integrated_depth,
        metadata.target_total_depth_mm,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"Integrated rainfall depth {integrated_depth:.12g} mm in {rainfall_path} "
            f"does not match summary target {metadata.target_total_depth_mm:.12g} mm"
        )


def discover_rsl_files(
    results_dir: Path,
    event: int,
    requested_times_minutes: Sequence[int],
) -> dict[int, tuple[int, Path]]:
    """Discover unique RSL files by parsed minute values, not timestep assumptions."""

    event_dir = results_dir / f"R{event}C1_SurfaceMaps"
    if not event_dir.is_dir():
        raise FileNotFoundError(f"CityCAT surface-map directory not found: {event_dir}")
    pattern = re.compile(
        rf"^R{event}_C1_T(?P<timestep>\d+)_(?P<minutes>\d+)min\.rsl$"
    )
    matches_by_minute: dict[int, list[tuple[int, Path]]] = {}
    for path in sorted(event_dir.glob("*.rsl"), key=lambda item: item.name):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        time_minutes = int(match.group("minutes"))
        timestep_index = int(match.group("timestep"))
        matches_by_minute.setdefault(time_minutes, []).append((timestep_index, path))

    selected: dict[int, tuple[int, Path]] = {}
    for time_minutes in requested_times_minutes:
        matches = matches_by_minute.get(time_minutes, [])
        if not matches:
            raise FileNotFoundError(
                f"No RSL file found for R{event} at {time_minutes} minutes in {event_dir}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple RSL files match R{event} at {time_minutes} minutes: "
                f"{[str(path) for _, path in matches]}"
            )
        selected[time_minutes] = matches[0]
    return selected


def read_rsl_depth(
    path: Path,
    expected_time_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse five-column RSL rows and validate the header timestamp metadata."""

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    nonempty_lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]
    if not nonempty_lines:
        raise ValueError(f"RSL file is empty: {path}")
    header_line_number, header_line = nonempty_lines[0]
    header_tokens = header_line.split()
    expected_header_tokens = len(REQUIRED_RSL_COLUMNS) + 1
    if len(header_tokens) != expected_header_tokens:
        raise ValueError(
            f"RSL header at line {header_line_number} of {path} must contain exactly "
            f"five data-column tokens and one time-metadata token; got {len(header_tokens)}"
        )
    if tuple(header_tokens[:5]) != REQUIRED_RSL_COLUMNS:
        raise ValueError(
            f"First five RSL header tokens at line {header_line_number} of {path} must be "
            f"{' '.join(REQUIRED_RSL_COLUMNS)!r}; got {' '.join(header_tokens[:5])!r}"
        )
    time_metadata = header_tokens[5]
    time_match = RSL_TIME_METADATA_PATTERN.fullmatch(time_metadata)
    if time_match is None:
        raise ValueError(
            f"Invalid RSL time metadata at line {header_line_number} of {path}: "
            f"{time_metadata!r}; expected T_<finite non-negative seconds>_sec"
        )
    header_time_seconds = float(time_match.group("seconds"))
    if not math.isfinite(header_time_seconds) or header_time_seconds < 0.0:
        raise ValueError(
            f"Invalid RSL header time at line {header_line_number} of {path}: "
            f"{header_time_seconds}"
        )
    expected_time_seconds = expected_time_minutes * 60.0
    if not math.isclose(
        header_time_seconds,
        expected_time_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"RSL header time {header_time_seconds:g}s in {path} does not agree with "
            f"the filename time {expected_time_minutes}min ({expected_time_seconds:g}s)"
        )

    x_values: list[float] = []
    y_values: list[float] = []
    depths: list[float] = []
    coordinate_pairs: set[tuple[float, float]] = set()
    for line_number, line in nonempty_lines[1:]:
        tokens = line.split()
        if len(tokens) != len(REQUIRED_RSL_COLUMNS):
            raise ValueError(
                f"Malformed RSL row at line {line_number} of {path}: expected "
                f"{len(REQUIRED_RSL_COLUMNS)} values, got {len(tokens)}"
            )
        parsed_values = {
            name: parse_finite_float(tokens[index], name, path, line_number)
            for index, name in enumerate(REQUIRED_RSL_COLUMNS)
        }
        depth = parsed_values["Depth"]
        if depth < 0.0:
            raise ValueError(f"Negative depth at line {line_number} of {path}: {depth}")
        coordinate_pair = (parsed_values["XCen"], parsed_values["YCen"])
        if coordinate_pair in coordinate_pairs:
            raise ValueError(
                f"Duplicate coordinate pair at line {line_number} of {path}: "
                f"{coordinate_pair}"
            )
        coordinate_pairs.add(coordinate_pair)
        x_values.append(coordinate_pair[0])
        y_values.append(coordinate_pair[1])
        depths.append(depth)
    if not depths:
        raise ValueError(f"RSL file contains no hydraulic rows: {path}")
    return (
        np.asarray(x_values, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
        np.asarray(depths, dtype=np.float64),
    )


def map_coordinates_to_grid(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    dem: DemGrid,
    rsl_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Map RSL coordinates to exact active DEM cell centres."""

    from rasterio.transform import rowcol, xy

    left, bottom, right, top = dem.bounds
    in_bounds = (
        (x_coordinates >= left)
        & (x_coordinates <= right)
        & (y_coordinates >= bottom)
        & (y_coordinates <= top)
    )
    if not np.all(in_bounds):
        index = int(np.flatnonzero(~in_bounds)[0])
        raise ValueError(
            f"Coordinate outside DEM bounds in {rsl_path}: "
            f"({x_coordinates[index]}, {y_coordinates[index]})"
        )
    row_values, column_values = rowcol(dem.transform, x_coordinates, y_coordinates)
    rows = np.asarray(row_values, dtype=np.int64)
    columns = np.asarray(column_values, dtype=np.int64)
    on_grid = (rows >= 0) & (rows < dem.height) & (columns >= 0) & (columns < dem.width)
    if not np.all(on_grid):
        index = int(np.flatnonzero(~on_grid)[0])
        raise ValueError(
            f"Coordinate maps outside the DEM grid in {rsl_path}: "
            f"({x_coordinates[index]}, {y_coordinates[index]})"
        )
    centre_x_values, centre_y_values = xy(dem.transform, rows, columns, offset="center")
    centre_x = np.asarray(centre_x_values, dtype=np.float64)
    centre_y = np.asarray(centre_y_values, dtype=np.float64)
    tolerance = max(1e-7, min(dem.resolution) * 1e-6)
    centred = np.isclose(x_coordinates, centre_x, rtol=0.0, atol=tolerance) & np.isclose(
        y_coordinates, centre_y, rtol=0.0, atol=tolerance
    )
    if not np.all(centred):
        index = int(np.flatnonzero(~centred)[0])
        raise ValueError(
            f"Coordinate does not match a DEM cell centre within {tolerance:g} in "
            f"{rsl_path}: supplied=({x_coordinates[index]}, {y_coordinates[index]}), "
            f"centre=({centre_x[index]}, {centre_y[index]})"
        )
    if not np.all(dem.active_mask[rows, columns]):
        index = int(np.flatnonzero(~dem.active_mask[rows, columns])[0])
        raise ValueError(
            f"Coordinate maps to a DEM NoData cell in {rsl_path}: "
            f"({x_coordinates[index]}, {y_coordinates[index]})"
        )
    linear_cells = rows * dem.width + columns
    if np.unique(linear_cells).size != linear_cells.size:
        raise ValueError(f"Multiple coordinates map to the same DEM cell in {rsl_path}")
    return rows, columns


def load_snapshot(
    event: int,
    time_minutes: int,
    timestep_index: int,
    rsl_path: Path,
    dem: DemGrid,
) -> SnapshotDepth:
    """Read and spatially validate one requested RSL snapshot."""

    x_coordinates, y_coordinates, depths = read_rsl_depth(rsl_path, time_minutes)
    rows, columns = map_coordinates_to_grid(x_coordinates, y_coordinates, dem, rsl_path)
    if not np.isfinite(depths.astype(np.float32)).all():
        raise ValueError(f"Depths cannot be represented as Float32 in {rsl_path}")
    return SnapshotDepth(
        event=event,
        timestep_index=timestep_index,
        time_minutes=time_minutes,
        rsl_path=rsl_path,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        depths_m=depths,
        rows=rows,
        columns=columns,
        coordinate_set=frozenset(zip(x_coordinates.tolist(), y_coordinates.tolist())),
    )


def validate_common_mesh(snapshots: Sequence[SnapshotDepth]) -> None:
    """Require identical coordinate counts and sets across all snapshots."""

    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        if snapshot.depths_m.size != reference.depths_m.size:
            raise ValueError(
                f"Hydraulic coordinate count differs at {snapshot.time_minutes} minutes: "
                f"{snapshot.depths_m.size} versus {reference.depths_m.size} at "
                f"{reference.time_minutes} minutes"
            )
        if snapshot.coordinate_set != reference.coordinate_set:
            missing = len(reference.coordinate_set - snapshot.coordinate_set)
            extra = len(snapshot.coordinate_set - reference.coordinate_set)
            raise ValueError(
                f"Hydraulic coordinate set differs at {snapshot.time_minutes} minutes: "
                f"missing={missing}, extra={extra}"
            )


def build_depth_raster(snapshot: SnapshotDepth, dem: DemGrid) -> np.ndarray:
    """Populate only explicitly reported RSL cells in a NoData raster."""

    depth_raster = np.full((dem.height, dem.width), OUTPUT_NODATA, dtype=np.float32)
    depth_raster[snapshot.rows, snapshot.columns] = snapshot.depths_m.astype(np.float32)
    return depth_raster


def write_and_validate_geotiff(
    output_path: Path,
    depth_raster: np.ndarray,
    snapshot: SnapshotDepth,
    dem: DemGrid,
) -> tuple[float, float]:
    """Write one compressed depth GeoTIFF and reopen it for validation."""

    import rasterio

    profile = dem.profile.copy()
    profile.update(
        driver="GTiff",
        width=dem.width,
        height=dem.height,
        count=1,
        dtype="float32",
        nodata=OUTPUT_NODATA,
        transform=dem.transform,
        crs=dem.crs,
        compress="deflate",
    )
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(depth_raster, 1)

    with rasterio.open(output_path) as source:
        if source.count != 1 or (source.height, source.width) != (dem.height, dem.width):
            raise ValueError(f"GeoTIFF shape does not match the DEM: {output_path}")
        if source.crs != dem.crs:
            raise ValueError(f"GeoTIFF CRS does not match the DEM: {output_path}")
        if source.transform != dem.transform:
            raise ValueError(f"GeoTIFF transform does not match the DEM: {output_path}")
        if source.dtypes[0] != "float32":
            raise ValueError(f"GeoTIFF dtype is not Float32: {output_path}")
        if source.nodata is None or float(source.nodata) != OUTPUT_NODATA:
            raise ValueError(f"GeoTIFF NoData is not {OUTPUT_NODATA:g}: {output_path}")
        written = source.read(1)

    populated = np.isfinite(written) & (written != OUTPUT_NODATA)
    if int(np.count_nonzero(populated)) != snapshot.depths_m.size:
        raise ValueError(f"GeoTIFF populated-cell count is incorrect: {output_path}")
    populated_depths = written[populated]
    minimum_depth = float(np.min(populated_depths))
    maximum_depth = float(np.max(populated_depths))
    if not math.isfinite(minimum_depth) or not math.isfinite(maximum_depth):
        raise ValueError(f"GeoTIFF contains non-finite populated depths: {output_path}")
    expected_depths = snapshot.depths_m.astype(np.float32)
    if minimum_depth != float(np.min(expected_depths)) or maximum_depth != float(
        np.max(expected_depths)
    ):
        raise ValueError(f"GeoTIFF extrema changed during writing: {output_path}")
    return minimum_depth, maximum_depth


def rainfall_at_time(
    rainfall: RainfallSeries,
    time_minutes: int,
) -> tuple[float, float]:
    """Return exact-time rainfall intensity and cumulative depth."""

    time_seconds = time_minutes * 60
    matches = np.flatnonzero(rainfall.times_seconds == time_seconds)
    if matches.size != 1:
        raise ValueError(
            f"Rainfall forcing has no unique record at {time_minutes} minutes"
        )
    index = int(matches[0])
    return (
        float(rainfall.intensity_mm_per_hour[index]),
        float(rainfall.cumulative_depth_mm[index]),
    )


def calculate_metrics(
    snapshot: SnapshotDepth,
    depth_raster: np.ndarray,
    rainfall: RainfallSeries,
    dem: DemGrid,
    geotiff_path: Path,
    png_path: Path,
) -> SnapshotResult:
    """Calculate forcing values and hydraulic summary metrics."""

    rainfall_intensity, cumulative_rainfall = rainfall_at_time(
        rainfall, snapshot.time_minutes
    )
    depths = snapshot.depths_m
    wet_depths = depths[depths >= 0.01]
    wet_count = int(wet_depths.size)
    mean_wet_depth = float(np.mean(wet_depths)) if wet_count else 0.0
    areas = {
        threshold: float(np.count_nonzero(depths >= threshold) * dem.pixel_area_m2)
        for threshold in AREA_THRESHOLDS_M
    }
    return SnapshotResult(
        event=snapshot.event,
        timestep_index=snapshot.timestep_index,
        time_minutes=snapshot.time_minutes,
        rainfall_intensity_mm_per_hour=rainfall_intensity,
        cumulative_rainfall_mm=cumulative_rainfall,
        hydraulic_cell_count=int(depths.size),
        wet_cell_count_ge_0_01m=wet_count,
        maximum_depth_m=float(np.max(depths)),
        p99_depth_m=float(np.percentile(depths, 99)),
        mean_depth_wet_cells_m=mean_wet_depth,
        area_ge_0_01_m2=areas[0.01],
        area_ge_0_10_m2=areas[0.10],
        area_ge_0_30_m2=areas[0.30],
        rsl_path=snapshot.rsl_path,
        geotiff_path=geotiff_path,
        png_path=png_path,
        depth_raster=depth_raster,
    )


def masked_plot_arrays(
    depth_raster: np.ndarray,
    dem: DemGrid,
    display_min_depth: float,
) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Return terrain and display-depth arrays with appropriate masks."""

    terrain = np.ma.masked_where(~dem.active_mask, dem.data)
    water = np.ma.masked_where(
        (depth_raster == OUTPUT_NODATA) | (depth_raster < display_min_depth),
        depth_raster,
    )
    return terrain, water


def plot_individual_snapshot(
    result: SnapshotResult,
    dem: DemGrid,
    display_min_depth: float,
    display_max_depth: float,
) -> None:
    """Write one annotated temporal water-depth PNG map."""

    import matplotlib.pyplot as plt

    terrain, water = masked_plot_arrays(result.depth_raster, dem, display_min_depth)
    left, bottom, right, top = dem.bounds
    extent = (left, right, bottom, top)
    blue_colormap = plt.get_cmap("Blues").copy()
    blue_colormap.set_bad(alpha=0.0)
    figure, axis = plt.subplots(figsize=(10, 9))
    axis.imshow(terrain, cmap="gray", origin="upper", extent=extent, alpha=0.55)
    water_image = axis.imshow(
        water,
        cmap=blue_colormap,
        origin="upper",
        extent=extent,
        vmin=display_min_depth,
        vmax=display_max_depth,
    )
    colour_bar = figure.colorbar(water_image, ax=axis)
    colour_bar.set_label("Water depth (m)")
    axis.set_aspect("equal")
    axis.set_xlabel("British National Grid easting (m)")
    axis.set_ylabel("British National Grid northing (m)")
    axis.set_title(f"Stockbridge R{result.event} at {result.time_minutes} minutes")
    annotation = "\n".join(
        (
            f"Rainfall intensity: {result.rainfall_intensity_mm_per_hour:.2f} mm/h",
            f"Cumulative rainfall: {result.cumulative_rainfall_mm:.2f} mm",
            f"Maximum depth: {result.maximum_depth_m:.3f} m",
            f"99th percentile depth: {result.p99_depth_m:.3f} m",
            f"Wet area ≥ 0.01 m: {result.area_ge_0_01_m2:,.1f} m²",
            f"Wet area ≥ 0.10 m: {result.area_ge_0_10_m2:,.1f} m²",
            f"Wet area ≥ 0.30 m: {result.area_ge_0_30_m2:,.1f} m²",
            f"Colour scale capped at {display_max_depth:.2f} m; deeper cells are saturated.",
        )
    )
    axis.text(
        0.02,
        0.02,
        annotation,
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    figure.tight_layout()
    figure.savefig(result.png_path, dpi=300)
    plt.close(figure)
    if not result.png_path.is_file():
        raise ValueError(f"Snapshot PNG was not created: {result.png_path}")


def plot_comparison(
    results: Sequence[SnapshotResult],
    dem: DemGrid,
    output_path: Path,
    display_min_depth: float,
    display_max_depth: float,
) -> None:
    """Create the ordered multi-panel temporal comparison with one colour bar."""

    import matplotlib.pyplot as plt

    panel_count = len(results)
    columns = 1 if panel_count == 1 else 2
    rows = math.ceil(panel_count / columns)
    figure = plt.figure(figsize=(14, 6 * rows))
    grid = figure.add_gridspec(
        rows,
        columns + 1,
        width_ratios=[1.0] * columns + [0.045],
        left=0.07,
        right=0.94,
        bottom=0.075,
        top=0.92,
        wspace=0.24,
        hspace=0.28,
    )
    flat_axes = [
        figure.add_subplot(grid[row, column])
        for row in range(rows)
        for column in range(columns)
    ]
    colour_bar_axis = figure.add_subplot(grid[:, -1])
    left, bottom, right, top = dem.bounds
    extent = (left, right, bottom, top)
    terrain = np.ma.masked_where(~dem.active_mask, dem.data)
    blue_colormap = plt.get_cmap("Blues").copy()
    blue_colormap.set_bad(alpha=0.0)
    water_image = None
    for axis, result in zip(flat_axes, results):
        _, water = masked_plot_arrays(result.depth_raster, dem, display_min_depth)
        axis.imshow(terrain, cmap="gray", origin="upper", extent=extent, alpha=0.55)
        water_image = axis.imshow(
            water,
            cmap=blue_colormap,
            origin="upper",
            extent=extent,
            vmin=display_min_depth,
            vmax=display_max_depth,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
        axis.set_title(
            f"{result.time_minutes} min | rain "
            f"{result.rainfall_intensity_mm_per_hour:.2f} mm/h | cumulative "
            f"{result.cumulative_rainfall_mm:.2f} mm"
        )
        axis.text(
            0.02,
            0.02,
            f"Max: {result.maximum_depth_m:.3f} m\n"
            f"Area ≥ 0.10 m: {result.area_ge_0_10_m2:,.1f} m²",
            transform=axis.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for axis in flat_axes[panel_count:]:
        axis.set_visible(False)
    if water_image is None:
        raise ValueError("Cannot create a comparison plot without snapshots")
    colour_bar = figure.colorbar(water_image, cax=colour_bar_axis)
    colour_bar.set_label("Water depth (m)")
    figure.suptitle(f"Stockbridge R{results[0].event} water-depth evolution")
    figure.text(
        0.5,
        0.01,
        f"Common colour scale capped at {display_max_depth:.2f} m.",
        ha="center",
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    if not output_path.is_file():
        raise ValueError(f"Temporal comparison PNG was not created: {output_path}")


def write_diagnostic_csv(output_path: Path, results: Sequence[SnapshotResult]) -> None:
    """Write the temporal snapshot diagnostic table with Unix newlines."""

    fieldnames = [
        "event",
        "timestep_index",
        "time_minutes",
        "rainfall_intensity_mm_per_hour",
        "cumulative_rainfall_mm",
        "hydraulic_cell_count",
        "wet_cell_count_ge_0_01m",
        "maximum_depth_m",
        "p99_depth_m",
        "mean_depth_wet_cells_m",
        "area_ge_0_01_m2",
        "area_ge_0_10_m2",
        "area_ge_0_30_m2",
        "rsl_path",
        "geotiff_path",
        "png_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "event": result.event,
                    "timestep_index": result.timestep_index,
                    "time_minutes": result.time_minutes,
                    "rainfall_intensity_mm_per_hour": result.rainfall_intensity_mm_per_hour,
                    "cumulative_rainfall_mm": result.cumulative_rainfall_mm,
                    "hydraulic_cell_count": result.hydraulic_cell_count,
                    "wet_cell_count_ge_0_01m": result.wet_cell_count_ge_0_01m,
                    "maximum_depth_m": result.maximum_depth_m,
                    "p99_depth_m": result.p99_depth_m,
                    "mean_depth_wet_cells_m": result.mean_depth_wet_cells_m,
                    "area_ge_0_01_m2": result.area_ge_0_01_m2,
                    "area_ge_0_10_m2": result.area_ge_0_10_m2,
                    "area_ge_0_30_m2": result.area_ge_0_30_m2,
                    "rsl_path": str(result.rsl_path),
                    "geotiff_path": str(result.geotiff_path),
                    "png_path": str(result.png_path),
                }
            )
    if not output_path.is_file():
        raise ValueError(f"Temporal diagnostic CSV was not created: {output_path}")


def print_console_summary(results: Sequence[SnapshotResult]) -> None:
    """Print the requested concise temporal comparison table."""

    print(
        "time_min\train_mm_per_h\tcumulative_rain_mm\tmax_depth_m\tp99_depth_m\t"
        "area_ge_0.01_m2\tarea_ge_0.10_m2\tarea_ge_0.30_m2"
    )
    for result in results:
        print(
            f"{result.time_minutes}\t"
            f"{result.rainfall_intensity_mm_per_hour:.6g}\t"
            f"{result.cumulative_rainfall_mm:.6g}\t"
            f"{result.maximum_depth_m:.6g}\t"
            f"{result.p99_depth_m:.6g}\t"
            f"{result.area_ge_0_01_m2:.6g}\t"
            f"{result.area_ge_0_10_m2:.6g}\t"
            f"{result.area_ge_0_30_m2:.6g}"
        )


def main() -> None:
    """Load, validate, map, and summarize selected temporal snapshots."""

    args = parse_args()
    validate_cli_values(args)
    results_dir, dem_path, summary_path, rainfall_path, output_dir = resolve_paths(args)
    dem = load_dem(dem_path)
    metadata = read_event_metadata(summary_path, args.event)
    rainfall = parse_rainfall(rainfall_path)
    validate_rainfall_against_summary(rainfall, metadata, rainfall_path)
    discovered = discover_rsl_files(results_dir, args.event, args.times_minutes)
    snapshots = [
        load_snapshot(
            args.event,
            time_minutes,
            discovered[time_minutes][0],
            discovered[time_minutes][1],
            dem,
        )
        for time_minutes in args.times_minutes
    ]
    validate_common_mesh(snapshots)

    results: list[SnapshotResult] = []
    for snapshot in snapshots:
        depth_raster = build_depth_raster(snapshot, dem)
        geotiff_path = output_dir / (
            f"R{args.event}_T{snapshot.time_minutes}min_depth.tif"
        )
        png_path = output_dir / (
            f"R{args.event}_T{snapshot.time_minutes}min_depth_map.png"
        )
        _, geotiff_maximum = write_and_validate_geotiff(
            geotiff_path,
            depth_raster,
            snapshot,
            dem,
        )
        result = calculate_metrics(
            snapshot,
            depth_raster,
            rainfall,
            dem,
            geotiff_path,
            png_path,
        )
        if not math.isclose(
            result.maximum_depth_m,
            geotiff_maximum,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"GeoTIFF maximum differs from RSL maximum at {snapshot.time_minutes} minutes"
            )
        plot_individual_snapshot(
            result,
            dem,
            args.display_min_depth,
            args.display_max_depth,
        )
        results.append(result)

    comparison_path = output_dir / f"R{args.event}_temporal_depth_comparison.png"
    plot_comparison(
        results,
        dem,
        comparison_path,
        args.display_min_depth,
        args.display_max_depth,
    )
    diagnostic_path = output_dir / f"R{args.event}_temporal_snapshot_summary.csv"
    write_diagnostic_csv(diagnostic_path, results)
    print_console_summary(results)


if __name__ == "__main__":
    main()
