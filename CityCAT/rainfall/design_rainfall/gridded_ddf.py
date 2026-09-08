"""Validated point extraction from explicitly catalogued DDF ASCII grids.

Spatial nearest-grid-point selection in this module is deliberately separate
from duration or return-period interpolation provided by :mod:`.ddf`.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .ddf import DDFRecord, DDFTable


ARC_ASCII_HEADER_KEYS = frozenset(
    {
        "NCOLS",
        "NROWS",
        "XLLCENTER",
        "XLLCORNER",
        "YLLCENTER",
        "YLLCORNER",
        "CELLSIZE",
        "NODATA_VALUE",
    }
)
GRID_CATALOGUE_COLUMNS = (
    "grid_path",
    "duration_minutes",
    "return_period_years",
    "value_scale_to_mm",
    "crs",
    "source",
    "source_version",
    "extraction_date",
    "climate_scenario",
    "notes",
)
EXTRACTION_METHOD = "nearest_grid_point"


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric and must not be bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_number(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _positive_integer(value: str, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if str(number) != value.strip() or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _duration_seconds(duration_minutes: float) -> int:
    seconds_float = duration_minutes * 60.0
    seconds = round(seconds_float)
    if not math.isclose(seconds_float, seconds, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"duration_minutes {duration_minutes:.12g} cannot be represented "
            "as a whole number of seconds"
        )
    return int(seconds)


def _number_text(value: float) -> str:
    return format(value, ".15g")


@dataclass(frozen=True)
class ArcInfoASCIIGrid:
    """One parsed ASCII grid with north-to-south raster row order preserved."""

    path: Path
    ncols: int
    nrows: int
    xllcenter: float
    yllcenter: float
    cellsize: float
    nodata_value: float
    origin_convention: str
    header_metadata: Mapping[str, str]
    values: np.ndarray
    nodata_mask: np.ndarray

    @property
    def minimum_x(self) -> float:
        return self.xllcenter

    @property
    def maximum_x(self) -> float:
        return self.xllcenter + (self.ncols - 1) * self.cellsize

    @property
    def minimum_y(self) -> float:
        return self.yllcenter

    @property
    def maximum_y(self) -> float:
        return self.yllcenter + (self.nrows - 1) * self.cellsize

    @property
    def geometry_signature(self) -> tuple[int, int, float, float, float]:
        """Return geometry expressed using lower-left grid-point centres."""

        return (
            self.ncols,
            self.nrows,
            self.xllcenter,
            self.yllcenter,
            self.cellsize,
        )

    def coordinates_for_index(self, row: int, column: int) -> tuple[float, float]:
        """Return coordinates for a north-to-south array row and column."""

        if row < 0 or row >= self.nrows or column < 0 or column >= self.ncols:
            raise IndexError("grid row or column is outside the raster")
        x = self.xllcenter + column * self.cellsize
        y = self.yllcenter + (self.nrows - 1 - row) * self.cellsize
        return x, y


@dataclass(frozen=True)
class GriddedDDFCatalogueEntry:
    """One explicitly described duration/return-period DDF grid."""

    grid_path: Path
    duration_minutes: float
    duration_seconds: int
    return_period_years: float
    value_scale_to_mm: float
    crs: str
    source: str
    source_version: str
    extraction_date: str
    climate_scenario: str
    notes: str
    additional_provenance: Mapping[str, str]
    grid: ArcInfoASCIIGrid


@dataclass(frozen=True)
class GriddedDDFCatalogue:
    """A coherent DDF surface described by an explicit catalogue CSV."""

    path: Path
    entries: tuple[GriddedDDFCatalogueEntry, ...]
    crs: str
    geometry_signature: tuple[int, int, float, float, float]


@dataclass(frozen=True)
class GridPointDDFExtraction:
    """Traceable extraction of one DDF value from one catalogue grid."""

    location_id: str
    duration_minutes: float
    duration_seconds: int
    return_period_years: float
    depth_mm: float
    raw_grid_value: float
    value_scale_to_mm: float
    requested_x: float
    requested_y: float
    requested_crs: str
    selected_grid_x: float
    selected_grid_y: float
    distance_to_grid_point: float
    selected_row: int
    selected_column: int
    extraction_method: str
    source_grid_path: Path
    source: str
    source_version: str
    extraction_date: str
    climate_scenario: str
    notes: str
    grid_header_metadata: Mapping[str, str]
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class GriddedDDFExtractionResult:
    """All point extractions and the corresponding validated DDF table."""

    location_id: str
    requested_x: float
    requested_y: float
    crs: str
    selected_grid_x: float
    selected_grid_y: float
    distance_to_grid_point: float
    extraction_method: str
    catalogue: GriddedDDFCatalogue
    extractions: tuple[GridPointDDFExtraction, ...]
    ddf_table: DDFTable


def read_arcinfo_ascii_grid(path: Path | str) -> ArcInfoASCIIGrid:
    """Read a strict six-line Arc/Info ASCII grid without reordering rows."""

    grid_path = Path(path).expanduser().resolve()
    try:
        lines = grid_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read Arc/Info ASCII grid: {grid_path}") from exc
    if len(lines) < 6:
        raise ValueError("Arc/Info ASCII grid has an incomplete header")

    header: dict[str, str] = {}
    for line_number, line in enumerate(lines[:6], start=1):
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(
                f"malformed Arc/Info ASCII header line {line_number}: {line!r}"
            )
        key, raw_value = tokens[0].upper(), tokens[1]
        if key not in ARC_ASCII_HEADER_KEYS:
            raise ValueError(f"unexpected Arc/Info ASCII header key: {tokens[0]!r}")
        if key in header:
            raise ValueError(f"duplicate Arc/Info ASCII header key: {key}")
        header[key] = raw_value

    required_common = {"NCOLS", "NROWS", "CELLSIZE", "NODATA_VALUE"}
    missing_common = required_common.difference(header)
    if missing_common:
        raise ValueError(
            f"Arc/Info ASCII header is missing fields: {sorted(missing_common)}"
        )
    has_center = "XLLCENTER" in header or "YLLCENTER" in header
    has_corner = "XLLCORNER" in header or "YLLCORNER" in header
    if has_center == has_corner:
        raise ValueError(
            "Arc/Info ASCII header must use exactly one CENTER or CORNER origin"
        )
    if has_center and not {"XLLCENTER", "YLLCENTER"}.issubset(header):
        raise ValueError("CENTER origin requires XLLCENTER and YLLCENTER")
    if has_corner and not {"XLLCORNER", "YLLCORNER"}.issubset(header):
        raise ValueError("CORNER origin requires XLLCORNER and YLLCORNER")
    if len(header) != 6:
        raise ValueError("Arc/Info ASCII header must contain exactly six fields")

    ncols = _positive_integer(header["NCOLS"], "NCOLS")
    nrows = _positive_integer(header["NROWS"], "NROWS")
    cellsize = _positive_number(header["CELLSIZE"], "CELLSIZE")
    nodata_value = _finite_number(header["NODATA_VALUE"], "NODATA_VALUE")
    if has_center:
        origin_convention = "center"
        xllcenter = _finite_number(header["XLLCENTER"], "XLLCENTER")
        yllcenter = _finite_number(header["YLLCENTER"], "YLLCENTER")
    else:
        origin_convention = "corner"
        xllcorner = _finite_number(header["XLLCORNER"], "XLLCORNER")
        yllcorner = _finite_number(header["YLLCORNER"], "YLLCORNER")
        xllcenter = xllcorner + cellsize / 2.0
        yllcenter = yllcorner + cellsize / 2.0
    if not math.isfinite(xllcenter) or not math.isfinite(yllcenter):
        raise ValueError("derived lower-left centre coordinates must be finite")

    raster_lines = lines[6:]
    if len(raster_lines) != nrows:
        raise ValueError(
            f"Arc/Info ASCII grid must contain exactly {nrows} raster rows; "
            f"found {len(raster_lines)}"
        )
    raster_rows: list[list[float]] = []
    for row_number, line in enumerate(raster_lines, start=1):
        tokens = line.split()
        if len(tokens) != ncols:
            raise ValueError(
                f"raster row {row_number} must contain exactly {ncols} values; "
                f"found {len(tokens)}"
            )
        try:
            row = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(f"malformed raster value in row {row_number}") from exc
        raster_rows.append(row)
    values = np.asarray(raster_rows, dtype=np.float64)
    if values.shape != (nrows, ncols):
        raise ValueError(
            f"raster must contain exactly {nrows * ncols} values"
        )
    if not np.isfinite(values).all():
        raise ValueError("Arc/Info ASCII raster values must be finite")
    nodata_mask = values == nodata_value
    values.setflags(write=False)
    nodata_mask.setflags(write=False)
    return ArcInfoASCIIGrid(
        path=grid_path,
        ncols=ncols,
        nrows=nrows,
        xllcenter=xllcenter,
        yllcenter=yllcenter,
        cellsize=cellsize,
        nodata_value=nodata_value,
        origin_convention=origin_convention,
        header_metadata=MappingProxyType(dict(header)),
        values=values,
        nodata_mask=nodata_mask,
    )


def _resolve_catalogue_grid_path(raw_path: str, catalogue_path: Path) -> Path:
    path_text = _non_empty_string(raw_path, "grid_path")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = catalogue_path.parent / path
    return path.resolve()


def load_gridded_ddf_catalogue(
    path: Path | str,
) -> GriddedDDFCatalogue:
    """Load an explicit CSV catalogue and validate its complete DDF surface."""

    catalogue_path = Path(path).expanduser().resolve()
    try:
        handle = catalogue_path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"could not open grid catalogue: {catalogue_path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("grid catalogue contains duplicate column names")
        missing = [name for name in GRID_CATALOGUE_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"grid catalogue is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("grid catalogue contains no data records")

    entries: list[GriddedDDFCatalogueEntry] = []
    ddf_keys: set[tuple[int, float]] = set()
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(row[name] is None for name in columns):
            raise ValueError(f"malformed grid catalogue row {row_number}")
        duration_minutes = _positive_number(
            row["duration_minutes"],
            f"duration_minutes at catalogue row {row_number}",
        )
        duration_seconds = _duration_seconds(duration_minutes)
        return_period_years = _positive_number(
            row["return_period_years"],
            f"return_period_years at catalogue row {row_number}",
        )
        value_scale_to_mm = _positive_number(
            row["value_scale_to_mm"],
            f"value_scale_to_mm at catalogue row {row_number}",
        )
        crs = _non_empty_string(
            row["crs"], f"crs at catalogue row {row_number}"
        )
        key = (duration_seconds, return_period_years)
        if key in ddf_keys:
            raise ValueError(
                "duplicate duration/return-period grid at catalogue row "
                f"{row_number}"
            )
        ddf_keys.add(key)
        grid_path = _resolve_catalogue_grid_path(row["grid_path"], catalogue_path)
        grid = read_arcinfo_ascii_grid(grid_path)
        additional_provenance = {
            name: value
            for name, value in row.items()
            if name not in GRID_CATALOGUE_COLUMNS and value not in (None, "")
        }
        entries.append(
            GriddedDDFCatalogueEntry(
                grid_path=grid_path,
                duration_minutes=duration_minutes,
                duration_seconds=duration_seconds,
                return_period_years=return_period_years,
                value_scale_to_mm=value_scale_to_mm,
                crs=crs,
                source=row["source"].strip(),
                source_version=row["source_version"].strip(),
                extraction_date=row["extraction_date"].strip(),
                climate_scenario=row["climate_scenario"].strip(),
                notes=row["notes"].strip(),
                additional_provenance=MappingProxyType(additional_provenance),
                grid=grid,
            )
        )

    geometry = entries[0].grid.geometry_signature
    inconsistent_geometry = [
        entry.grid_path
        for entry in entries
        if entry.grid.geometry_signature != geometry
    ]
    if inconsistent_geometry:
        raise ValueError(
            "grid catalogue contains inconsistent grid geometry: "
            f"{inconsistent_geometry}"
        )
    crs_values = {entry.crs for entry in entries}
    if len(crs_values) != 1:
        raise ValueError("grid catalogue must use one internally consistent CRS")
    climate_scenarios = {
        entry.climate_scenario
        for entry in entries
        if entry.climate_scenario
    }
    if len(climate_scenarios) > 1:
        raise ValueError(
            "grid catalogue must contain at most one non-empty climate_scenario"
        )
    return GriddedDDFCatalogue(
        path=catalogue_path,
        entries=tuple(entries),
        crs=next(iter(crs_values)),
        geometry_signature=geometry,
    )


def _nearest_axis_index(
    value: float,
    first_coordinate: float,
    count: int,
    spacing: float,
    axis_name: str,
) -> int:
    last_coordinate = first_coordinate + (count - 1) * spacing
    if value < first_coordinate or value > last_coordinate:
        raise ValueError(
            f"requested {axis_name} coordinate {value:.17g} is outside the "
            f"grid-point domain [{first_coordinate:.17g}, "
            f"{last_coordinate:.17g}]"
        )
    position = (value - first_coordinate) / spacing
    lower = max(0, min(count - 1, math.floor(position)))
    upper = max(0, min(count - 1, math.ceil(position)))
    return min(
        {lower, upper},
        key=lambda index: (
            abs(first_coordinate + index * spacing - value),
            index,
        ),
    )


def _extraction_provenance(
    entry: GriddedDDFCatalogueEntry,
    catalogue_path: Path,
    location_id: str,
    requested_x: float,
    requested_y: float,
    selected_x: float,
    selected_y: float,
    distance: float,
    row: int,
    column: int,
    raw_value: float,
) -> Mapping[str, str]:
    provenance = dict(entry.additional_provenance)
    provenance.update({
        "source": entry.source,
        "source_version": entry.source_version,
        "extraction_date": entry.extraction_date,
        "climate_scenario": entry.climate_scenario,
        "notes": entry.notes,
        "source_grid_path": str(entry.grid_path),
        "grid_catalogue_path": str(catalogue_path),
        "grid_crs": entry.crs,
        "requested_location_id": location_id,
        "requested_x": _number_text(requested_x),
        "requested_y": _number_text(requested_y),
        "selected_grid_x": _number_text(selected_x),
        "selected_grid_y": _number_text(selected_y),
        "distance_to_grid_point": _number_text(distance),
        "selected_grid_row": str(row),
        "selected_grid_column": str(column),
        "extraction_method": EXTRACTION_METHOD,
        "raw_grid_value": _number_text(raw_value),
        "value_scale_to_mm": _number_text(entry.value_scale_to_mm),
    })
    provenance.update(
        {
            f"arc_ascii_{key.lower()}": value
            for key, value in entry.grid.header_metadata.items()
        }
    )
    return MappingProxyType(provenance)


def extract_ddf_at_point(
    catalogue: GriddedDDFCatalogue,
    *,
    x: float,
    y: float,
    crs: str,
    location_id: str,
) -> GriddedDDFExtractionResult:
    """Extract every catalogue grid at one nearest native-CRS grid point.

    No coordinate transform, spatial interpolation, DDF interpolation, climate
    uplift, catchment averaging, or areal reduction is performed.
    """

    if not isinstance(catalogue, GriddedDDFCatalogue):
        raise ValueError("catalogue must be a GriddedDDFCatalogue")
    requested_x = _finite_number(x, "x")
    requested_y = _finite_number(y, "y")
    requested_crs = _non_empty_string(crs, "point CRS")
    selected_location_id = _non_empty_string(location_id, "location_id")
    if requested_crs != catalogue.crs:
        raise ValueError(
            f"point CRS {requested_crs!r} does not match catalogue CRS "
            f"{catalogue.crs!r}; coordinate transformation is not performed"
        )

    first_grid = catalogue.entries[0].grid
    column = _nearest_axis_index(
        requested_x,
        first_grid.xllcenter,
        first_grid.ncols,
        first_grid.cellsize,
        "x",
    )
    south_to_north_row = _nearest_axis_index(
        requested_y,
        first_grid.yllcenter,
        first_grid.nrows,
        first_grid.cellsize,
        "y",
    )
    row = first_grid.nrows - 1 - south_to_north_row
    selected_x, selected_y = first_grid.coordinates_for_index(row, column)
    distance = math.hypot(selected_x - requested_x, selected_y - requested_y)

    extractions: list[GridPointDDFExtraction] = []
    ddf_records: list[DDFRecord] = []
    for entry in catalogue.entries:
        if entry.grid.nodata_mask[row, column]:
            raise ValueError(
                "cannot extract NODATA cell from grid: "
                f"{entry.grid_path} at row {row}, column {column}"
            )
        raw_value = float(entry.grid.values[row, column])
        depth_mm = raw_value * entry.value_scale_to_mm
        if not math.isfinite(depth_mm) or depth_mm <= 0.0:
            raise ValueError(
                f"scaled DDF depth must be finite and positive: {depth_mm}"
            )
        frozen_provenance = _extraction_provenance(
            entry,
            catalogue.path,
            selected_location_id,
            requested_x,
            requested_y,
            selected_x,
            selected_y,
            distance,
            row,
            column,
            raw_value,
        )
        extraction = GridPointDDFExtraction(
            location_id=selected_location_id,
            duration_minutes=entry.duration_minutes,
            duration_seconds=entry.duration_seconds,
            return_period_years=entry.return_period_years,
            depth_mm=depth_mm,
            raw_grid_value=raw_value,
            value_scale_to_mm=entry.value_scale_to_mm,
            requested_x=requested_x,
            requested_y=requested_y,
            requested_crs=requested_crs,
            selected_grid_x=selected_x,
            selected_grid_y=selected_y,
            distance_to_grid_point=distance,
            selected_row=row,
            selected_column=column,
            extraction_method=EXTRACTION_METHOD,
            source_grid_path=entry.grid_path,
            source=entry.source,
            source_version=entry.source_version,
            extraction_date=entry.extraction_date,
            climate_scenario=entry.climate_scenario,
            notes=entry.notes,
            grid_header_metadata=entry.grid.header_metadata,
            provenance=frozen_provenance,
        )
        extractions.append(extraction)
        ddf_records.append(
            DDFRecord(
                location_id=selected_location_id,
                duration_minutes=entry.duration_minutes,
                duration_seconds=entry.duration_seconds,
                return_period_years=entry.return_period_years,
                depth_mm=depth_mm,
                provenance=frozen_provenance,
            )
        )

    ordered = sorted(
        zip(extractions, ddf_records),
        key=lambda pair: (
            pair[0].duration_seconds,
            pair[0].return_period_years,
        ),
    )
    ordered_extractions = tuple(pair[0] for pair in ordered)
    table = DDFTable(
        selected_location_id,
        tuple(pair[1] for pair in ordered),
    )
    return GriddedDDFExtractionResult(
        location_id=selected_location_id,
        requested_x=requested_x,
        requested_y=requested_y,
        crs=requested_crs,
        selected_grid_x=selected_x,
        selected_grid_y=selected_y,
        distance_to_grid_point=distance,
        extraction_method=EXTRACTION_METHOD,
        catalogue=catalogue,
        extractions=ordered_extractions,
        ddf_table=table,
    )
