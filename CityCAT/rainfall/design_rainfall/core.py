"""Generic temporal, validation, and CityCAT rainfall-file primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TemporalConfiguration:
    """Time-grid and numerical tolerances shared by rainfall workflows."""

    rainfall_timestep_seconds: int
    simulation_horizon_seconds: int
    maximum_active_duration_seconds: int
    depth_tolerance_mm: float = 1e-6
    citycat_value_tolerance: float = 1e-15

    def __post_init__(self) -> None:
        for name in (
            "rainfall_timestep_seconds",
            "simulation_horizon_seconds",
            "maximum_active_duration_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.simulation_horizon_seconds % self.rainfall_timestep_seconds:
            raise ValueError("simulation horizon must be divisible by the rainfall timestep")
        if self.maximum_active_duration_seconds > self.simulation_horizon_seconds:
            raise ValueError("maximum active duration must not exceed the simulation horizon")
        if self.maximum_active_duration_seconds % self.rainfall_timestep_seconds:
            raise ValueError("maximum active duration must align with the rainfall timestep")
        for name in ("depth_tolerance_mm", "citycat_value_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def number_of_records(self) -> int:
        return self.simulation_horizon_seconds // self.rainfall_timestep_seconds + 1

    def validate_grid_time(self, value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value % self.rainfall_timestep_seconds:
            raise ValueError(f"{name} must align with the rainfall timestep")


def generate_timeline(configuration: TemporalConfiguration) -> np.ndarray:
    """Generate the inclusive simulation timeline."""

    return (
        np.arange(configuration.number_of_records, dtype=np.int64)
        * configuration.rainfall_timestep_seconds
    )


@dataclass(frozen=True)
class RainfallEvent:
    """A discretised rainfall event with profile-independent metadata.

    ``active_duration_seconds`` is the endpoint of active rainfall, not the
    timestamp of the final positive value. Intensity is zero at time zero, may
    be positive strictly before the endpoint, and is zero at and after it.
    """

    rainfall_index: int
    phase_event_id: str
    storm_id: str
    role: str
    profile_family: str
    target_depth_mm: float
    active_duration_seconds: int
    intensities_mm_per_hour: np.ndarray
    peak_metadata: Mapping[str, Any] | None = None


def validate_rainfall_event(
    event: RainfallEvent, configuration: TemporalConfiguration
) -> None:
    """Validate generic physical and temporal properties of a rainfall event."""

    if (
        isinstance(event.rainfall_index, bool)
        or not isinstance(event.rainfall_index, int)
        or event.rainfall_index <= 0
    ):
        raise ValueError("rainfall_index must be a positive integer")
    for name in ("phase_event_id", "storm_id", "role", "profile_family"):
        value = getattr(event, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    configuration.validate_grid_time(event.active_duration_seconds, "active duration")
    if event.active_duration_seconds > configuration.maximum_active_duration_seconds:
        raise ValueError("active duration exceeds the configured maximum")
    if event.active_duration_seconds > configuration.simulation_horizon_seconds:
        raise ValueError("rainfall is not fully contained within the simulation horizon")
    intensities = np.asarray(event.intensities_mm_per_hour, dtype=np.float64)
    if intensities.ndim != 1 or intensities.size != configuration.number_of_records:
        raise ValueError(
            f"rainfall series must contain exactly "
            f"{configuration.number_of_records} records"
        )
    if not np.isfinite(intensities).all() or np.any(intensities < 0.0):
        raise ValueError("rainfall intensities must be finite and non-negative")
    if float(intensities[0]) != 0.0:
        raise ValueError("rainfall intensity at time zero must be zero")
    endpoint = event.active_duration_seconds // configuration.rainfall_timestep_seconds
    if np.any(intensities[endpoint:] != 0.0):
        raise ValueError("rainfall must be zero at and after the active-duration endpoint")
    if not math.isfinite(event.target_depth_mm) or event.target_depth_mm < 0.0:
        raise ValueError("target depth must be finite and non-negative")
    # NumPy 1.x, used by this repository, provides trapz but not trapezoid.
    integrated_depth_mm = float(
        np.trapz(intensities, generate_timeline(configuration) / 3600.0)
    )
    if not math.isclose(
        integrated_depth_mm,
        event.target_depth_mm,
        rel_tol=0.0,
        abs_tol=configuration.depth_tolerance_mm,
    ):
        raise ValueError(
            f"integrated depth {integrated_depth_mm:.12g} mm does not match "
            f"target {event.target_depth_mm:.12g} mm"
        )
    if event.peak_metadata is not None:
        peak_time = event.peak_metadata.get("peak_time_seconds")
        if peak_time is not None:
            configuration.validate_grid_time(peak_time, "peak time")
            if peak_time >= event.active_duration_seconds:
                raise ValueError("peak time must precede the active-duration endpoint")


def export_citycat_rainfall(
    output_directory: Path | str,
    event: RainfallEvent,
    configuration: TemporalConfiguration,
) -> Path:
    """Validate and export one event in the Phase 2 CityCAT text format."""

    validate_rainfall_event(event, configuration)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"Rainfall_Data_{event.rainfall_index}.txt"
    values = np.asarray(event.intensities_mm_per_hour, dtype=np.float64) / 1000.0 / 3600.0
    lines = [
        f"* * * {event.storm_id}",
        "* * * rainfall ***",
        "* * *",
        str(configuration.number_of_records),
        "* * *",
    ]
    lines.extend(
        f"{int(time)} {float(value):.17g}"
        for time, value in zip(generate_timeline(configuration), values)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def validate_citycat_rainfall(
    path: Path | str,
    event: RainfallEvent,
    configuration: TemporalConfiguration,
) -> None:
    """Validate CityCAT structure, exact timestamps, and converted values."""

    validate_rainfall_event(event, configuration)
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    expected_header = [
        f"* * * {event.storm_id}",
        "* * * rainfall ***",
        "* * *",
        str(configuration.number_of_records),
        "* * *",
    ]
    if lines[:5] != expected_header:
        raise ValueError(f"CityCAT header validation failed: {path}")
    records = lines[5:]
    if len(records) != configuration.number_of_records:
        raise ValueError(f"CityCAT file must contain exactly {configuration.number_of_records} records")
    parsed_times, parsed_values = [], []
    for number, line in enumerate(records, start=1):
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(f"malformed CityCAT record {number}: {line!r}")
        try:
            parsed_times.append(int(tokens[0]))
            parsed_values.append(float(tokens[1]))
        except ValueError as exc:
            raise ValueError(f"malformed CityCAT record {number}: {line!r}") from exc
    if not np.array_equal(
        np.asarray(parsed_times, dtype=np.int64), generate_timeline(configuration)
    ):
        raise ValueError("CityCAT timestamps do not match the configured timeline")
    parsed = np.asarray(parsed_values, dtype=np.float64)
    expected = np.asarray(event.intensities_mm_per_hour, dtype=np.float64) / 1000.0 / 3600.0
    if not np.isfinite(parsed).all() or np.any(parsed < 0.0):
        raise ValueError("CityCAT rainfall values must be finite and non-negative")
    if not np.allclose(parsed, expected, rtol=0.0, atol=configuration.citycat_value_tolerance):
        raise ValueError("CityCAT metres-per-second conversion is incorrect")
