#!/usr/bin/env python3
"""Filter CityCAT polygon records by intersection with a DEM valid-cell domain."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROGRESS_INTERVAL = 5000


@dataclass(frozen=True)
class PolygonRecord:
    """One parsed CityCAT polygon and its unchanged stripped source line."""

    line_number: int
    source_line: str
    geometry: Any


@dataclass(frozen=True)
class ActiveDomain:
    """Exact vector geometry and reporting metadata for valid DEM cells."""

    geometry: Any
    crs: str
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    nodata: float
    active_cell_count: int
    inactive_cell_count: int


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""

    parser = argparse.ArgumentParser(
        description="Filter CityCAT polygons using the valid-cell domain of a DEM raster."
    )
    parser.add_argument(
        "--dem-path",
        type=Path,
        required=True,
        help="Reduced CityCAT Domain_DEM.asc or equivalent raster",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input CityCAT polygon file, such as Buildings.txt or GreenAreas.txt",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Filtered CityCAT polygon output file",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="JSON report path (default: <output-path>.json)",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve, validate, and prepare source and destination paths."""

    dem_path = args.dem_path.expanduser().resolve()
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path is not None
        else Path(f"{output_path}.json")
    )

    for label, path in (("DEM", dem_path), ("polygon input", input_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_path in {dem_path, input_path}:
        raise ValueError(f"Output path must not overwrite an input file: {output_path}")
    if report_path in {dem_path, input_path, output_path}:
        raise ValueError(f"Report path conflicts with another processing path: {report_path}")
    for path in (output_path, report_path):
        if path.exists() and not path.is_file():
            raise ValueError(f"Output path exists but is not a regular file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.parent.is_dir():
            raise ValueError(f"Output parent directory is invalid: {path.parent}")
    return dem_path, input_path, output_path, report_path


def load_active_domain(dem_path: Path) -> ActiveDomain:
    """Read the DEM and polygonize its exact finite, non-NoData cell mask."""

    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with rasterio.open(dem_path) as source:
        if source.count != 1:
            raise ValueError(f"DEM must contain exactly one raster band: {dem_path}")
        if source.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        transform_values = tuple(float(value) for value in source.transform[:6])
        if not all(math.isfinite(value) for value in transform_values):
            raise ValueError(f"DEM affine transform contains non-finite values: {dem_path}")
        determinant = source.transform.a * source.transform.e - source.transform.b * source.transform.d
        if not math.isfinite(determinant) or determinant == 0.0:
            raise ValueError(f"DEM affine transform is singular: {dem_path}")
        resolution = tuple(float(value) for value in source.res)
        if len(resolution) != 2 or not all(
            math.isfinite(value) and value > 0.0 for value in resolution
        ):
            raise ValueError(f"DEM pixel size must be finite and positive: {resolution}")
        if source.nodata is None:
            raise ValueError(f"DEM must define a NoData value: {dem_path}")

        data = source.read(1)
        nodata = float(source.nodata)
        finite_mask = np.isfinite(data)
        if math.isnan(nodata):
            active_mask = finite_mask
        else:
            active_mask = finite_mask & (data != nodata)
        active_cell_count = int(np.count_nonzero(active_mask))
        total_cell_count = int(active_mask.size)
        inactive_cell_count = total_cell_count - active_cell_count
        if active_cell_count == 0:
            raise ValueError(f"DEM contains no valid active cells: {dem_path}")
        if inactive_cell_count == 0:
            raise ValueError(
                f"DEM contains no external NoData cells; expected a reduced subcatchment mask: {dem_path}"
            )

        mask_values = active_mask.astype(np.uint8)
        cell_geometries = [
            shape(mapping)
            for mapping, value in shapes(mask_values, mask=active_mask, transform=source.transform)
            if int(value) == 1
        ]
        if not cell_geometries:
            raise ValueError(f"Failed to vectorize the active DEM cells: {dem_path}")
        active_geometry = unary_union(cell_geometries)
        if active_geometry.is_empty:
            raise ValueError(f"Active DEM domain geometry is empty: {dem_path}")

        bounds = tuple(float(value) for value in source.bounds)
        return ActiveDomain(
            geometry=active_geometry,
            crs=str(source.crs),
            width=int(source.width),
            height=int(source.height),
            bounds=bounds,
            resolution=resolution,
            nodata=nodata,
            active_cell_count=active_cell_count,
            inactive_cell_count=inactive_cell_count,
        )


def parse_declared_count(header_line: str, input_path: Path) -> int:
    """Parse the first-line CityCAT polygon count."""

    tokens = header_line.strip().split()
    if len(tokens) != 1:
        raise ValueError(f"Malformed polygon count on line 1 of {input_path}: {header_line!r}")
    try:
        declared_count = int(tokens[0])
    except ValueError as exc:
        raise ValueError(
            f"Malformed polygon count on line 1 of {input_path}: {header_line!r}"
        ) from exc
    if declared_count < 0:
        raise ValueError(f"Negative polygon count on line 1 of {input_path}: {declared_count}")
    return declared_count


def parse_polygon_line(source_line: str, line_number: int, source_path: Path) -> PolygonRecord:
    """Parse and validate one CityCAT polygon without changing its source text."""

    from shapely.geometry import Polygon

    tokens = source_line.split()
    if not tokens:
        raise ValueError(f"Empty polygon record at line {line_number} of {source_path}")
    try:
        vertex_count = int(tokens[0])
    except ValueError as exc:
        raise ValueError(
            f"Invalid vertex count at line {line_number} of {source_path}: {tokens[0]!r}"
        ) from exc
    if vertex_count < 3:
        raise ValueError(
            f"Polygon at line {line_number} of {source_path} has fewer than 3 vertices: "
            f"{vertex_count}"
        )
    expected_tokens = 1 + 2 * vertex_count
    if len(tokens) != expected_tokens:
        raise ValueError(
            f"Polygon at line {line_number} of {source_path} must contain exactly "
            f"{expected_tokens} tokens for N={vertex_count}, got {len(tokens)}"
        )
    try:
        coordinates = [float(value) for value in tokens[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric coordinate at line {line_number} of {source_path}"
        ) from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"Non-finite coordinate at line {line_number} of {source_path}")

    x_values = coordinates[:vertex_count]
    y_values = coordinates[vertex_count:]
    ring = list(zip(x_values, y_values))
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    geometry = Polygon(ring)
    if not math.isfinite(float(geometry.area)) or geometry.area <= 0.0:
        raise ValueError(f"Zero-area polygon at line {line_number} of {source_path}")
    return PolygonRecord(
        line_number=line_number,
        source_line=source_line,
        geometry=geometry,
    )


def make_geometry_valid(geometry: Any, line_number: int, source_path: Path) -> Any:
    """Repair an invalid polygon only for spatial predicate evaluation."""

    try:
        from shapely import make_valid
    except ImportError:  # Shapely versions before the top-level make_valid export.
        from shapely.validation import make_valid

    repaired = make_valid(geometry)
    if repaired.is_empty:
        raise ValueError(
            f"Invalid polygon at line {line_number} of {source_path} became empty after make_valid"
        )
    return repaired


def geometry_for_intersection(record: PolygonRecord, source_path: Path) -> tuple[Any, bool]:
    """Return a valid predicate geometry and whether repair was required."""

    if record.geometry.is_valid:
        return record.geometry, False
    return make_geometry_valid(record.geometry, record.line_number, source_path), True


def bounding_boxes_overlap(
    polygon_bounds: tuple[float, float, float, float],
    domain_bounds: tuple[float, float, float, float],
) -> bool:
    """Return whether two axis-aligned bounds overlap or touch."""

    polygon_min_x, polygon_min_y, polygon_max_x, polygon_max_y = polygon_bounds
    domain_min_x, domain_min_y, domain_max_x, domain_max_y = domain_bounds
    return not (
        polygon_max_x < domain_min_x
        or polygon_min_x > domain_max_x
        or polygon_max_y < domain_min_y
        or polygon_min_y > domain_max_y
    )


def filter_polygons(
    input_path: Path,
    active_domain: ActiveDomain,
) -> tuple[int, int, list[str], int]:
    """Parse source polygons and retain complete lines that intersect the domain."""

    from shapely.prepared import prep

    source_lines = input_path.read_text(encoding="utf-8").splitlines()
    if not source_lines:
        raise ValueError(f"Polygon input file is empty: {input_path}")
    declared_count = parse_declared_count(source_lines[0], input_path)
    polygon_lines = [
        (line_number, raw_line.strip())
        for line_number, raw_line in enumerate(source_lines[1:], start=2)
        if raw_line.strip()
    ]
    actual_count = len(polygon_lines)
    if declared_count != actual_count:
        raise ValueError(
            f"Declared polygon count {declared_count} does not match the {actual_count} "
            f"non-empty polygon records in {input_path}"
        )

    prepared_domain = prep(active_domain.geometry)
    domain_bounds = tuple(float(value) for value in active_domain.geometry.bounds)
    retained_lines: list[str] = []
    repaired_count = 0
    for processed_count, (line_number, source_line) in enumerate(polygon_lines, start=1):
        record = parse_polygon_line(source_line, line_number, input_path)
        test_geometry, repaired = geometry_for_intersection(record, input_path)
        if repaired:
            repaired_count += 1
        if bounding_boxes_overlap(
            tuple(float(value) for value in test_geometry.bounds), domain_bounds
        ) and prepared_domain.intersects(test_geometry):
            retained_lines.append(record.source_line)
        if processed_count % PROGRESS_INTERVAL == 0:
            print(
                f"Processed {processed_count}/{actual_count} polygons; "
                f"retained {len(retained_lines)}"
            )
    return declared_count, actual_count, retained_lines, repaired_count


def write_output(output_path: Path, retained_lines: Sequence[str]) -> None:
    """Write retained polygons in their original order with Unix newlines."""

    content = "\n".join([str(len(retained_lines)), *retained_lines]) + "\n"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    if not output_path.is_file():
        raise ValueError(f"Filtered polygon output was not created: {output_path}")


def validate_output(
    output_path: Path,
    expected_lines: Sequence[str],
    active_geometry: Any,
) -> int:
    """Reopen and fully validate the filtered CityCAT polygon file."""

    from shapely.prepared import prep

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    if not output_lines:
        raise ValueError(f"Filtered polygon output is empty: {output_path}")
    output_header_count = parse_declared_count(output_lines[0], output_path)
    polygon_lines = output_lines[1:]
    if output_header_count != len(polygon_lines):
        raise ValueError(
            f"Output header count {output_header_count} does not match {len(polygon_lines)} "
            f"polygon lines in {output_path}"
        )
    if output_header_count == 0:
        raise ValueError(f"Filtered polygon output contains no retained polygons: {output_path}")
    if polygon_lines != list(expected_lines):
        raise ValueError(
            f"Filtered output lines do not exactly match the retained source lines: {output_path}"
        )
    if Counter(polygon_lines) != Counter(expected_lines):
        raise ValueError(f"Duplicate polygon lines were introduced in {output_path}")

    prepared_domain = prep(active_geometry)
    for line_number, source_line in enumerate(polygon_lines, start=2):
        record = parse_polygon_line(source_line, line_number, output_path)
        test_geometry, _ = geometry_for_intersection(record, output_path)
        if not prepared_domain.intersects(test_geometry):
            raise ValueError(
                f"Output polygon at line {line_number} does not intersect the active DEM domain: "
                f"{output_path}"
            )
    return output_header_count


def nodata_for_json(nodata: float) -> float | str:
    """Return a strict-JSON representation of the raster NoData value."""

    if math.isfinite(nodata):
        return nodata
    if math.isnan(nodata):
        return "NaN"
    return "Infinity" if nodata > 0.0 else "-Infinity"


def write_report(report_path: Path, report: dict[str, object]) -> None:
    """Write the processing report as strict, readable JSON."""

    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report_path.is_file():
        raise ValueError(f"JSON processing report was not created: {report_path}")


def main() -> None:
    """Filter one CityCAT polygon file and record validation metadata."""

    args = parse_args()
    dem_path, input_path, output_path, report_path = resolve_paths(args)
    active_domain = load_active_domain(dem_path)
    declared_count, actual_count, retained_lines, repaired_count = filter_polygons(
        input_path,
        active_domain,
    )
    write_output(output_path, retained_lines)
    output_header_count = validate_output(
        output_path,
        retained_lines,
        active_domain.geometry,
    )

    retained_count = len(retained_lines)
    discarded_count = actual_count - retained_count
    total_cell_count = active_domain.active_cell_count + active_domain.inactive_cell_count
    report = {
        "source_dem_path": str(dem_path),
        "source_polygon_path": str(input_path),
        "output_polygon_path": str(output_path),
        "crs": active_domain.crs,
        "raster_width": active_domain.width,
        "raster_height": active_domain.height,
        "raster_bounds": list(active_domain.bounds),
        "raster_resolution": list(active_domain.resolution),
        "nodata_value": nodata_for_json(active_domain.nodata),
        "active_dem_cell_count": active_domain.active_cell_count,
        "inactive_dem_cell_count": active_domain.inactive_cell_count,
        "total_dem_cell_count": total_cell_count,
        "declared_input_polygon_count": declared_count,
        "actual_input_polygon_count": actual_count,
        "retained_polygon_count": retained_count,
        "discarded_polygon_count": discarded_count,
        "invalid_polygons_repaired_for_testing": repaired_count,
        "retained_percentage": 100.0 * retained_count / actual_count if actual_count else 0.0,
        "output_header_count": output_header_count,
        "validation_passed": True,
    }
    write_report(report_path, report)
    print(
        f"Completed {input_path.name}: retained {retained_count}/{actual_count} polygons "
        f"({report['retained_percentage']:.2f}%); discarded {discarded_count}; "
        f"repaired {repaired_count}"
    )


if __name__ == "__main__":
    main()
