"""Validated long-format DDF data, interpolation, and climate uplift."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


REQUIRED_DDF_COLUMNS = (
    "location_id",
    "duration_minutes",
    "return_period_years",
    "depth_mm",
)
DOCUMENTED_PROVENANCE_COLUMNS = (
    "source",
    "source_version",
    "extraction_date",
    "climate_scenario",
    "uplift_factor",
    "notes",
)
DURATION_INTERPOLATION_METHODS = ("log_log", "linear")
RETURN_PERIOD_INTERPOLATION_METHODS = ("gumbel_log_depth", "linear")


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric and must not be bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _duration_seconds(duration_minutes: float) -> int:
    seconds_float = duration_minutes * 60.0
    seconds = round(seconds_float)
    if not math.isclose(seconds_float, seconds, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"duration_minutes {duration_minutes:.12g} cannot be represented as "
            "a whole number of seconds"
        )
    return int(seconds)


@dataclass(frozen=True)
class DDFRecord:
    """One validated base-depth row from a normalised DDF CSV table."""

    location_id: str
    duration_minutes: float
    duration_seconds: int
    return_period_years: float
    depth_mm: float
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class DDFEstimate:
    """A traceable exact or one-dimensionally interpolated DDF estimate."""

    estimate_kind: str
    source_points: tuple[DDFRecord, ...]
    interpolation_method: str
    location_id: str
    requested_duration_minutes: float
    requested_return_period_years: float
    depth_mm: float


@dataclass(frozen=True)
class ClimateUpliftResult:
    """Traceable multiplication of a base rainfall depth by an uplift factor."""

    base_depth_mm: float
    uplift_factor: float
    uplifted_depth_mm: float


def apply_climate_uplift(
    base_depth_mm: float, uplift_factor: float
) -> ClimateUpliftResult:
    """Apply an explicit dimensionless uplift while preserving every component."""

    try:
        base = float(base_depth_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("base depth must be numeric") from exc
    if not math.isfinite(base) or base < 0.0:
        raise ValueError("base depth must be finite and non-negative")
    factor = _positive_number(uplift_factor, "uplift factor")
    uplifted = base * factor
    if not math.isfinite(uplifted):
        raise ValueError("uplifted depth is not finite")
    return ClimateUpliftResult(base, factor, uplifted)


def _query_values(
    location_id: str, duration_minutes: float, return_period_years: float
) -> tuple[str, float, int, float]:
    if not isinstance(location_id, str) or not location_id.strip():
        raise ValueError("location_id must be a non-empty string")
    duration = _positive_number(duration_minutes, "requested duration_minutes")
    duration_seconds = _duration_seconds(duration)
    return_period = _positive_number(
        return_period_years, "requested return_period_years"
    )
    return location_id, duration, duration_seconds, return_period


def _linear_interpolate(
    target: float, lower_x: float, upper_x: float, lower_y: float, upper_y: float
) -> float:
    weight = (target - lower_x) / (upper_x - lower_x)
    return lower_y + weight * (upper_y - lower_y)


def _gumbel_reduced_variate(return_period_years: float) -> float:
    if return_period_years <= 1.0:
        raise ValueError(
            "gumbel_log_depth interpolation requires return periods greater than "
            "one year"
        )
    non_exceedance_probability = 1.0 - 1.0 / return_period_years
    return -math.log(-math.log(non_exceedance_probability))


@dataclass(frozen=True)
class DDFTable:
    """Validated DDF records for exactly one selected location."""

    location_id: str
    records: tuple[DDFRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("selected DDF table contains no records")
        if any(record.location_id != self.location_id for record in self.records):
            raise ValueError("DDFTable records must belong to its selected location")

    def lookup_exact(
        self,
        location_id: str,
        duration_minutes: float,
        return_period_years: float,
    ) -> DDFEstimate:
        """Return an exact duration/return-period match."""

        location, duration, seconds, return_period = _query_values(
            location_id, duration_minutes, return_period_years
        )
        self._require_location(location)
        matches = [
            record
            for record in self.records
            if record.duration_seconds == seconds
            and record.return_period_years == return_period
        ]
        if not matches:
            raise ValueError(
                "no exact DDF value for location, duration, and return period"
            )
        record = matches[0]
        return DDFEstimate(
            "exact",
            (record,),
            "exact",
            location,
            duration,
            return_period,
            record.depth_mm,
        )

    def interpolate_duration(
        self,
        location_id: str,
        duration_minutes: float,
        return_period_years: float,
        *,
        method: str,
    ) -> DDFEstimate:
        """Interpolate duration at a fixed return period without extrapolation."""

        location, duration, seconds, return_period = _query_values(
            location_id, duration_minutes, return_period_years
        )
        self._require_location(location)
        if method not in DURATION_INTERPOLATION_METHODS:
            raise ValueError(
                f"duration interpolation method must be one of "
                f"{DURATION_INTERPOLATION_METHODS}"
            )
        points = sorted(
            (
                record
                for record in self.records
                if record.return_period_years == return_period
            ),
            key=lambda record: record.duration_seconds,
        )
        exact = [record for record in points if record.duration_seconds == seconds]
        if exact:
            return DDFEstimate(
                "exact",
                (exact[0],),
                "exact",
                location,
                duration,
                return_period,
                exact[0].depth_mm,
            )
        lower, upper = self._bracket(
            points,
            seconds,
            lambda record: record.duration_seconds,
            "duration",
        )
        if method == "linear":
            depth = _linear_interpolate(
                duration,
                lower.duration_minutes,
                upper.duration_minutes,
                lower.depth_mm,
                upper.depth_mm,
            )
        else:
            log_depth = _linear_interpolate(
                math.log(duration),
                math.log(lower.duration_minutes),
                math.log(upper.duration_minutes),
                math.log(lower.depth_mm),
                math.log(upper.depth_mm),
            )
            depth = math.exp(log_depth)
        return DDFEstimate(
            "interpolated",
            (lower, upper),
            method,
            location,
            duration,
            return_period,
            depth,
        )

    def interpolate_return_period(
        self,
        location_id: str,
        duration_minutes: float,
        return_period_years: float,
        *,
        method: str,
    ) -> DDFEstimate:
        """Interpolate return period at a fixed duration without extrapolation."""

        location, duration, seconds, return_period = _query_values(
            location_id, duration_minutes, return_period_years
        )
        self._require_location(location)
        if method not in RETURN_PERIOD_INTERPOLATION_METHODS:
            raise ValueError(
                f"return-period interpolation method must be one of "
                f"{RETURN_PERIOD_INTERPOLATION_METHODS}"
            )
        points = sorted(
            (
                record
                for record in self.records
                if record.duration_seconds == seconds
            ),
            key=lambda record: record.return_period_years,
        )
        exact = [
            record for record in points if record.return_period_years == return_period
        ]
        if exact:
            return DDFEstimate(
                "exact",
                (exact[0],),
                "exact",
                location,
                duration,
                return_period,
                exact[0].depth_mm,
            )
        target_y = None
        if method == "gumbel_log_depth":
            target_y = _gumbel_reduced_variate(return_period)
        lower, upper = self._bracket(
            points,
            return_period,
            lambda record: record.return_period_years,
            "return period",
        )
        if method == "linear":
            depth = _linear_interpolate(
                return_period,
                lower.return_period_years,
                upper.return_period_years,
                lower.depth_mm,
                upper.depth_mm,
            )
        else:
            assert target_y is not None
            lower_y = _gumbel_reduced_variate(lower.return_period_years)
            upper_y = _gumbel_reduced_variate(upper.return_period_years)
            log_depth = _linear_interpolate(
                target_y,
                lower_y,
                upper_y,
                math.log(lower.depth_mm),
                math.log(upper.depth_mm),
            )
            depth = math.exp(log_depth)
        return DDFEstimate(
            "interpolated",
            (lower, upper),
            method,
            location,
            duration,
            return_period,
            depth,
        )

    def _require_location(self, location_id: str) -> None:
        if location_id != self.location_id:
            raise ValueError(
                f"DDF table contains location {self.location_id!r}, not "
                f"{location_id!r}"
            )

    @staticmethod
    def _bracket(records, target, coordinate, dimension):
        if not records:
            raise ValueError(f"no DDF points available for fixed {dimension} lookup")
        minimum = coordinate(records[0])
        maximum = coordinate(records[-1])
        if target < minimum or target > maximum:
            raise ValueError(
                f"requested {dimension} lies outside the available DDF domain; "
                "extrapolation is disabled"
            )
        lower = max(
            (record for record in records if coordinate(record) < target),
            key=coordinate,
        )
        upper = min(
            (record for record in records if coordinate(record) > target),
            key=coordinate,
        )
        return lower, upper


def load_ddf_csv(
    path: Path | str, *, location_id: str | None = None
) -> DDFTable:
    """Load, validate, and select one location from a normalised DDF CSV."""

    source_path = Path(path).expanduser().resolve()
    try:
        handle = source_path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"could not open DDF CSV: {source_path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [name for name in REQUIRED_DDF_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"DDF CSV is missing required columns: {missing}")
        records: list[DDFRecord] = []
        keys: set[tuple[str, int, float]] = set()
        for row_number, row in enumerate(reader, start=2):
            row_location = row["location_id"]
            if not isinstance(row_location, str) or not row_location.strip():
                raise ValueError(f"location_id must be non-empty at CSV row {row_number}")
            duration = _positive_number(
                row["duration_minutes"],
                f"duration_minutes at CSV row {row_number}",
            )
            seconds = _duration_seconds(duration)
            return_period = _positive_number(
                row["return_period_years"],
                f"return_period_years at CSV row {row_number}",
            )
            depth = _positive_number(
                row["depth_mm"], f"depth_mm at CSV row {row_number}"
            )
            provenance = {
                name: value
                for name, value in row.items()
                if name not in REQUIRED_DDF_COLUMNS and value not in (None, "")
            }
            if "uplift_factor" in provenance:
                _positive_number(
                    provenance["uplift_factor"],
                    f"uplift_factor at CSV row {row_number}",
                )
            key = (row_location, seconds, return_period)
            if key in keys:
                raise ValueError(
                    "duplicate DDF key for location, duration, and return period "
                    f"at CSV row {row_number}"
                )
            keys.add(key)
            records.append(
                DDFRecord(
                    row_location,
                    duration,
                    seconds,
                    return_period,
                    depth,
                    MappingProxyType(provenance),
                )
            )

    if not records:
        raise ValueError("DDF CSV contains no data records")
    locations = sorted({record.location_id for record in records})
    if location_id is None:
        if len(locations) != 1:
            raise ValueError(
                "DDF CSV contains multiple locations; select location_id explicitly"
            )
        selected_location = locations[0]
    else:
        if not isinstance(location_id, str) or not location_id.strip():
            raise ValueError("selected location_id must be a non-empty string")
        selected_location = location_id
        if selected_location not in locations:
            raise ValueError(f"selected location_id {selected_location!r} was not found")
    selected_records = tuple(
        record for record in records if record.location_id == selected_location
    )
    climate_scenarios = {
        scenario.strip()
        for record in selected_records
        if (scenario := record.provenance.get("climate_scenario"))
        and scenario.strip()
    }
    if len(climate_scenarios) > 1:
        raise ValueError(
            "interpolation across climate scenarios is not permitted; the input "
            "must be separated into coherent DDF tables"
        )
    return DDFTable(selected_location, selected_records)
