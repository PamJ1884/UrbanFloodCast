#!/usr/bin/env python3
"""Create comparable maximum-depth maps for representative Stockbridge events."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


OUTPUT_NODATA = -9999.0
AREA_THRESHOLDS_M = (0.01, 0.10, 0.30)
REQUIRED_DEPTH_COLUMNS = {"XCen", "YCen", "Depth"}
REQUIRED_SUMMARY_COLUMNS = {
    "rainfall_index",
    "split",
    "target_total_depth_mm",
    "active_duration_minutes",
    "peak_intensity_mm_per_hour",
}


@dataclass(frozen=True)
class DemGrid:
    """DEM data and spatial metadata defining the target raster grid."""

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
    """Rainfall metadata used in map annotations."""

    rainfall_index: int
    split: str
    total_depth_mm: float
    active_duration_minutes: float
    peak_intensity_mm_per_hour: float


@dataclass(frozen=True)
class HydraulicDepth:
    """Validated hydraulic mesh coordinates and maximum depths."""

    rainfall_index: int
    source_path: Path
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    depths_m: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    coordinate_set: frozenset[tuple[float, float]]


@dataclass(frozen=True)
class EventResult:
    """Output paths and comparison metrics for one event."""

    rainfall_index: int
    hydraulic_cell_count: int
    maximum_depth_m: float
    percentile_99_depth_m: float
    area_ge_001_m2: float
    area_ge_010_m2: float
    area_ge_030_m2: float
    geotiff_path: Path
    png_path: Path


def parse_args() -> argparse.Namespace:
    """Parse source, output, event, and display options."""

    parser = argparse.ArgumentParser(
        description="Map representative Stockbridge Phase 2 maximum water depths."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing R<N>C1_SurfaceMaps result folders",
    )
    parser.add_argument(
        "--dem-path",
        type=Path,
        required=True,
        help="Path to the reduced Domain_DEM.asc",
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
        help="Directory for GeoTIFF and PNG outputs",
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=[4, 60, 3],
        help="Rainfall indices to process (default: 4 60 3)",
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


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve inputs and create the requested output directory."""

    results_dir = args.results_dir.expanduser().resolve()
    dem_path = args.dem_path.expanduser().resolve()
    summary_path = args.summary_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"CityCAT results directory not found: {results_dir}")
    for label, path in (("DEM", dem_path), ("rainfall summary", summary_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise ValueError(f"Output directory is invalid: {output_dir}")
    return results_dir, dem_path, summary_path, output_dir


def validate_cli_values(
    events: Sequence[int],
    display_min_depth: float,
    display_max_depth: float,
) -> None:
    """Validate event uniqueness and the shared display range."""

    if not events:
        raise ValueError("At least one rainfall index must be supplied with --events")
    if any(event <= 0 for event in events):
        raise ValueError(f"Rainfall indices must be positive integers: {list(events)}")
    if len(set(events)) != len(events):
        raise ValueError(f"Duplicate rainfall indices supplied: {list(events)}")
    if not math.isfinite(display_min_depth) or display_min_depth < 0.0:
        raise ValueError("--display-min-depth must be finite and non-negative")
    if not math.isfinite(display_max_depth) or display_max_depth <= display_min_depth:
        raise ValueError("--display-max-depth must be finite and greater than the display minimum")


def load_dem(dem_path: Path) -> DemGrid:
    """Read and validate the DEM grid and active-terrain mask."""

    import rasterio

    with rasterio.open(dem_path) as source:
        if source.count != 1:
            raise ValueError(f"DEM must contain exactly one raster band: {dem_path}")
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
            raise ValueError(f"DEM contains no valid terrain cells: {dem_path}")
        bounds = tuple(float(value) for value in source.bounds)
        return DemGrid(
            data=data,
            active_mask=active_mask,
            width=int(source.width),
            height=int(source.height),
            transform=source.transform,
            crs=source.crs,
            bounds=bounds,
            resolution=resolution,
            pixel_area_m2=pixel_area_m2,
            profile=source.profile.copy(),
        )


def require_columns(
    fieldnames: Sequence[str] | None,
    required: set[str],
    path: Path,
) -> None:
    """Raise a clear error for missing CSV columns."""

    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise ValueError(f"Required columns missing from {path}: {missing}")


def parse_finite_float(value: str | None, column: str, path: Path, line_number: int) -> float:
    """Parse one finite numeric CSV field with line context."""

    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column} at line {line_number} of {path}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite {column} at line {line_number} of {path}: {value!r}"
        )
    return parsed


def read_summary(summary_path: Path, events: Sequence[int]) -> dict[int, EventMetadata]:
    """Read required rainfall metadata for selected event indices."""

    metadata: dict[int, EventMetadata] = {}
    seen_indices: set[int] = set()
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, REQUIRED_SUMMARY_COLUMNS, summary_path)
        for line_number, row in enumerate(reader, start=2):
            raw_index = parse_finite_float(
                row.get("rainfall_index"), "rainfall_index", summary_path, line_number
            )
            rainfall_index = int(raw_index)
            if raw_index != rainfall_index:
                raise ValueError(
                    f"Non-integer rainfall_index at line {line_number} of {summary_path}: "
                    f"{raw_index}"
                )
            if rainfall_index in seen_indices:
                raise ValueError(
                    f"Duplicate rainfall_index {rainfall_index} in {summary_path}"
                )
            seen_indices.add(rainfall_index)
            if rainfall_index not in events:
                continue
            split = (row.get("split") or "").strip()
            if not split:
                raise ValueError(
                    f"Empty split for rainfall index {rainfall_index} in {summary_path}"
                )
            values = {
                column: parse_finite_float(row.get(column), column, summary_path, line_number)
                for column in (
                    "target_total_depth_mm",
                    "active_duration_minutes",
                    "peak_intensity_mm_per_hour",
                )
            }
            if any(value < 0.0 for value in values.values()):
                raise ValueError(
                    f"Negative rainfall metadata for index {rainfall_index} in {summary_path}"
                )
            metadata[rainfall_index] = EventMetadata(
                rainfall_index=rainfall_index,
                split=split,
                total_depth_mm=values["target_total_depth_mm"],
                active_duration_minutes=values["active_duration_minutes"],
                peak_intensity_mm_per_hour=values["peak_intensity_mm_per_hour"],
            )

    missing = sorted(set(events) - set(metadata))
    if missing:
        raise ValueError(f"Selected rainfall indices missing from {summary_path}: {missing}")
    return metadata


def event_csv_path(results_dir: Path, rainfall_index: int) -> Path:
    """Return the required CityCAT maximum-depth CSV path for one event."""

    path = (
        results_dir
        / f"R{rainfall_index}C1_SurfaceMaps"
        / f"R{rainfall_index}_C1_max_depth.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Maximum-depth CSV not found for rainfall index {rainfall_index}: {path}"
        )
    return path


def read_depth_csv(path: Path, rainfall_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read finite coordinates and non-negative depths from one CityCAT CSV."""

    x_values: list[float] = []
    y_values: list[float] = []
    depths: list[float] = []
    coordinate_pairs: set[tuple[float, float]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, REQUIRED_DEPTH_COLUMNS, path)
        for line_number, row in enumerate(reader, start=2):
            x_value = parse_finite_float(row.get("XCen"), "XCen", path, line_number)
            y_value = parse_finite_float(row.get("YCen"), "YCen", path, line_number)
            depth = parse_finite_float(row.get("Depth"), "Depth", path, line_number)
            if depth < 0.0:
                raise ValueError(f"Negative depth at line {line_number} of {path}: {depth}")
            coordinate_pair = (x_value, y_value)
            if coordinate_pair in coordinate_pairs:
                raise ValueError(
                    f"Duplicate coordinate pair at line {line_number} of {path}: "
                    f"{coordinate_pair}"
                )
            coordinate_pairs.add(coordinate_pair)
            x_values.append(x_value)
            y_values.append(y_value)
            depths.append(depth)
    if not depths:
        raise ValueError(f"Maximum-depth CSV contains no hydraulic cells: {path}")
    return (
        np.asarray(x_values, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
        np.asarray(depths, dtype=np.float64),
    )


def map_coordinates_to_grid(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    dem: DemGrid,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates to exact active DEM cell centres."""

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
            f"Coordinate outside DEM bounds in {source_path}: "
            f"({x_coordinates[index]}, {y_coordinates[index]})"
        )

    row_values, column_values = rowcol(dem.transform, x_coordinates, y_coordinates)
    rows = np.asarray(row_values, dtype=np.int64)
    columns = np.asarray(column_values, dtype=np.int64)
    on_grid = (rows >= 0) & (rows < dem.height) & (columns >= 0) & (columns < dem.width)
    if not np.all(on_grid):
        index = int(np.flatnonzero(~on_grid)[0])
        raise ValueError(
            f"Coordinate maps outside the DEM grid in {source_path}: "
            f"({x_coordinates[index]}, {y_coordinates[index]})"
        )

    centre_x_values, centre_y_values = xy(dem.transform, rows, columns, offset="center")
    centre_x = np.asarray(centre_x_values, dtype=np.float64)
    centre_y = np.asarray(centre_y_values, dtype=np.float64)
    coordinate_tolerance = max(1e-7, min(dem.resolution) * 1e-6)
    centred = np.isclose(
        x_coordinates, centre_x, rtol=0.0, atol=coordinate_tolerance
    ) & np.isclose(y_coordinates, centre_y, rtol=0.0, atol=coordinate_tolerance)
    if not np.all(centred):
        index = int(np.flatnonzero(~centred)[0])
        raise ValueError(
            f"Coordinate does not match a DEM cell centre within {coordinate_tolerance:g} "
            f"in {source_path}: supplied=({x_coordinates[index]}, {y_coordinates[index]}), "
            f"centre=({centre_x[index]}, {centre_y[index]})"
        )
    if not np.all(dem.active_mask[rows, columns]):
        index = int(np.flatnonzero(~dem.active_mask[rows, columns])[0])
        raise ValueError(
            f"Coordinate maps to a DEM NoData cell in {source_path}: "
            f"({x_coordinates[index]}, {y_coordinates[index]})"
        )

    linear_cells = rows * dem.width + columns
    if np.unique(linear_cells).size != linear_cells.size:
        raise ValueError(f"Multiple coordinates map to the same DEM cell in {source_path}")
    return rows, columns


def load_hydraulic_depth(
    results_dir: Path,
    rainfall_index: int,
    dem: DemGrid,
) -> HydraulicDepth:
    """Read and spatially validate one event's hydraulic mesh depths."""

    source_path = event_csv_path(results_dir, rainfall_index)
    x_coordinates, y_coordinates, depths = read_depth_csv(source_path, rainfall_index)
    rows, columns = map_coordinates_to_grid(
        x_coordinates,
        y_coordinates,
        dem,
        source_path,
    )
    float32_depths = depths.astype(np.float32)
    if not np.isfinite(float32_depths).all():
        raise ValueError(f"Depth values cannot be represented as Float32 in {source_path}")
    return HydraulicDepth(
        rainfall_index=rainfall_index,
        source_path=source_path,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        depths_m=depths,
        rows=rows,
        columns=columns,
        coordinate_set=frozenset(zip(x_coordinates.tolist(), y_coordinates.tolist())),
    )


def validate_common_mesh(events: Sequence[HydraulicDepth]) -> None:
    """Require identical coordinate sets and counts across selected events."""

    reference = events[0]
    for event in events[1:]:
        if event.depths_m.size != reference.depths_m.size:
            raise ValueError(
                f"Hydraulic coordinate count differs between R{reference.rainfall_index} "
                f"({reference.depths_m.size}) and R{event.rainfall_index} "
                f"({event.depths_m.size})"
            )
        if event.coordinate_set != reference.coordinate_set:
            missing = len(reference.coordinate_set - event.coordinate_set)
            extra = len(event.coordinate_set - reference.coordinate_set)
            raise ValueError(
                f"Hydraulic coordinate set differs for R{event.rainfall_index}: "
                f"missing={missing}, extra={extra} relative to R{reference.rainfall_index}"
            )


def build_depth_raster(event: HydraulicDepth, dem: DemGrid) -> np.ndarray:
    """Populate only explicitly reported hydraulic cells in a NoData raster."""

    depth_raster = np.full((dem.height, dem.width), OUTPUT_NODATA, dtype=np.float32)
    depth_raster[event.rows, event.columns] = event.depths_m.astype(np.float32)
    return depth_raster


def write_and_validate_geotiff(
    output_path: Path,
    depth_raster: np.ndarray,
    event: HydraulicDepth,
    dem: DemGrid,
) -> tuple[float, float]:
    """Write one compressed GeoTIFF and verify its grid and populated values."""

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
        if source.nodata is None or not math.isclose(
            float(source.nodata), OUTPUT_NODATA, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(f"GeoTIFF NoData is not {OUTPUT_NODATA:g}: {output_path}")
        written = source.read(1)

    populated = np.isfinite(written) & (written != OUTPUT_NODATA)
    populated_count = int(np.count_nonzero(populated))
    if populated_count != event.depths_m.size:
        raise ValueError(
            f"GeoTIFF populated-cell count {populated_count} does not match source count "
            f"{event.depths_m.size}: {output_path}"
        )
    populated_depths = written[populated]
    minimum_depth = float(np.min(populated_depths))
    maximum_depth = float(np.max(populated_depths))
    if not math.isfinite(minimum_depth) or not math.isfinite(maximum_depth):
        raise ValueError(f"GeoTIFF contains non-finite populated depths: {output_path}")
    expected_depths = event.depths_m.astype(np.float32)
    if not math.isclose(
        minimum_depth,
        float(np.min(expected_depths)),
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        maximum_depth,
        float(np.max(expected_depths)),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"GeoTIFF minimum or maximum depth changed during writing: {output_path}")
    return minimum_depth, maximum_depth


def calculate_areas(depths_m: np.ndarray, pixel_area_m2: float) -> dict[float, float]:
    """Calculate inundated areas from populated hydraulic cells only."""

    return {
        threshold: float(np.count_nonzero(depths_m >= threshold) * pixel_area_m2)
        for threshold in AREA_THRESHOLDS_M
    }


def plot_depth_map(
    output_path: Path,
    depth_raster: np.ndarray,
    dem: DemGrid,
    metadata: EventMetadata,
    maximum_depth_m: float,
    areas_m2: dict[float, float],
    display_min_depth: float,
    display_max_depth: float,
) -> None:
    """Plot one maximum-depth raster using a shared clipped colour scale."""

    import matplotlib.pyplot as plt

    left, bottom, right, top = dem.bounds
    extent = (left, right, bottom, top)
    terrain = np.ma.masked_where(~dem.active_mask, dem.data)
    water = np.ma.masked_where(
        (depth_raster == OUTPUT_NODATA) | (depth_raster < display_min_depth),
        depth_raster,
    )
    blue_colormap = plt.get_cmap("Blues").copy()
    blue_colormap.set_bad(alpha=0.0)

    fig, axis = plt.subplots(figsize=(10, 9))
    axis.imshow(
        terrain,
        cmap="gray",
        origin="upper",
        extent=extent,
        alpha=0.55,
    )
    water_image = axis.imshow(
        water,
        cmap=blue_colormap,
        origin="upper",
        extent=extent,
        vmin=display_min_depth,
        vmax=display_max_depth,
    )
    colour_bar = fig.colorbar(water_image, ax=axis)
    colour_bar.set_label("Maximum water depth (m)")
    axis.set_aspect("equal")
    axis.set_xlabel("British National Grid easting (m)")
    axis.set_ylabel("British National Grid northing (m)")
    axis.set_title(f"R{metadata.rainfall_index} Stockbridge maximum water depth")
    annotation = "\n".join(
        (
            f"Split: {metadata.split}",
            f"Rainfall depth: {metadata.total_depth_mm:.2f} mm",
            f"Active duration: {metadata.active_duration_minutes:.1f} min",
            f"Peak intensity: {metadata.peak_intensity_mm_per_hour:.2f} mm/h",
            f"Maximum simulated depth: {maximum_depth_m:.3f} m",
            f"Area ≥ 0.01 m: {areas_m2[0.01]:,.1f} m²",
            f"Area ≥ 0.10 m: {areas_m2[0.10]:,.1f} m²",
            f"Area ≥ 0.30 m: {areas_m2[0.30]:,.1f} m²",
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
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    if not output_path.is_file():
        raise ValueError(f"PNG maximum-depth map was not created: {output_path}")


def process_event(
    event: HydraulicDepth,
    metadata: EventMetadata,
    dem: DemGrid,
    output_dir: Path,
    display_min_depth: float,
    display_max_depth: float,
) -> EventResult:
    """Create and validate GeoTIFF and PNG outputs for one selected event."""

    depth_raster = build_depth_raster(event, dem)
    geotiff_path = output_dir / f"R{event.rainfall_index}_max_depth.tif"
    png_path = output_dir / f"R{event.rainfall_index}_max_depth_map.png"
    _, verified_maximum_depth = write_and_validate_geotiff(
        geotiff_path,
        depth_raster,
        event,
        dem,
    )
    areas_m2 = calculate_areas(event.depths_m, dem.pixel_area_m2)
    plot_depth_map(
        png_path,
        depth_raster,
        dem,
        metadata,
        verified_maximum_depth,
        areas_m2,
        display_min_depth,
        display_max_depth,
    )
    return EventResult(
        rainfall_index=event.rainfall_index,
        hydraulic_cell_count=int(event.depths_m.size),
        maximum_depth_m=verified_maximum_depth,
        percentile_99_depth_m=float(np.percentile(event.depths_m, 99)),
        area_ge_001_m2=areas_m2[0.01],
        area_ge_010_m2=areas_m2[0.10],
        area_ge_030_m2=areas_m2[0.30],
        geotiff_path=geotiff_path,
        png_path=png_path,
    )


def print_comparison_table(results: Sequence[EventResult]) -> None:
    """Print the selected-event comparison metrics and output paths."""

    headers = (
        "event",
        "hydraulic_cells",
        "max_depth_m",
        "p99_depth_m",
        "area_ge_0.01_m2",
        "area_ge_0.10_m2",
        "area_ge_0.30_m2",
        "geotiff_path",
        "png_path",
    )
    print("\t".join(headers))
    for result in results:
        print(
            "\t".join(
                (
                    f"R{result.rainfall_index}",
                    str(result.hydraulic_cell_count),
                    f"{result.maximum_depth_m:.6g}",
                    f"{result.percentile_99_depth_m:.6g}",
                    f"{result.area_ge_001_m2:.6g}",
                    f"{result.area_ge_010_m2:.6g}",
                    f"{result.area_ge_030_m2:.6g}",
                    str(result.geotiff_path),
                    str(result.png_path),
                )
            )
        )


def main() -> None:
    """Validate selected CityCAT results and create comparable depth products."""

    args = parse_args()
    validate_cli_values(args.events, args.display_min_depth, args.display_max_depth)
    results_dir, dem_path, summary_path, output_dir = resolve_paths(args)
    dem = load_dem(dem_path)
    metadata = read_summary(summary_path, args.events)
    hydraulic_events = [
        load_hydraulic_depth(results_dir, rainfall_index, dem)
        for rainfall_index in args.events
    ]
    validate_common_mesh(hydraulic_events)
    results = [
        process_event(
            event,
            metadata[event.rainfall_index],
            dem,
            output_dir,
            args.display_min_depth,
            args.display_max_depth,
        )
        for event in hydraulic_events
    ]
    print_comparison_table(results)


if __name__ == "__main__":
    main()
