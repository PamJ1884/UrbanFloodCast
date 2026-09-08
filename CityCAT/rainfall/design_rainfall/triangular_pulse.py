"""Reusable, discretely depth-conserving single triangular rainfall pulses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .core import (
    RainfallEvent,
    TemporalConfiguration,
    generate_timeline,
    validate_rainfall_event,
)


SHAPE_NORMALISATION_MEANING = (
    "millimetres per hour applied to the dimensionless sampled triangular "
    "shape so its trapezoidal piecewise-linear integral equals target_depth_mm"
)
DEPTH_INTEGRATION_METHOD = "trapezoidal_sampled_piecewise_linear"


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


@dataclass(frozen=True)
class TriangularPulse:
    """Continuous triangle parameters to sample on a configured time grid."""

    start_time_seconds: int
    duration_seconds: int
    target_depth_mm: float
    peak_position_ratio: float

    def __post_init__(self) -> None:
        _non_negative_integer(self.start_time_seconds, "start_time_seconds")
        _positive_integer(self.duration_seconds, "duration_seconds")
        target_depth = _finite_number(self.target_depth_mm, "target_depth_mm")
        if target_depth <= 0.0:
            raise ValueError("target_depth_mm must be finite and strictly positive")
        peak_ratio = _finite_number(
            self.peak_position_ratio, "peak_position_ratio"
        )
        if not 0.0 < peak_ratio < 1.0:
            raise ValueError("peak_position_ratio must be strictly between 0 and 1")
        object.__setattr__(self, "target_depth_mm", target_depth)
        object.__setattr__(self, "peak_position_ratio", peak_ratio)

    @property
    def end_time_seconds(self) -> int:
        """Return the absolute pulse endpoint measured from time zero."""

        return self.start_time_seconds + self.duration_seconds

    @property
    def peak_time_seconds(self) -> float:
        """Return the requested peak time without snapping it to the grid."""

        return (
            self.start_time_seconds
            + self.peak_position_ratio * self.duration_seconds
        )


def _validate_pulse_timing(
    configuration: TemporalConfiguration,
    pulse: TriangularPulse,
) -> None:
    timestep = configuration.rainfall_timestep_seconds
    if pulse.start_time_seconds % timestep:
        raise ValueError("start_time_seconds must align with the rainfall timestep")
    if pulse.duration_seconds % timestep:
        raise ValueError("duration_seconds must align with the rainfall timestep")
    if pulse.end_time_seconds > configuration.simulation_horizon_seconds:
        raise ValueError("pulse endpoint exceeds the simulation horizon")
    if pulse.end_time_seconds > configuration.maximum_active_duration_seconds:
        raise ValueError("pulse endpoint exceeds the maximum active endpoint")


def _sample_dimensionless_shape(
    timeline_seconds: np.ndarray,
    pulse: TriangularPulse,
) -> np.ndarray:
    shape = np.zeros(timeline_seconds.size, dtype=np.float64)
    peak_time = pulse.peak_time_seconds
    rising = (
        (timeline_seconds > pulse.start_time_seconds)
        & (timeline_seconds <= peak_time)
        & (timeline_seconds < pulse.end_time_seconds)
    )
    falling = (
        (timeline_seconds > peak_time)
        & (timeline_seconds < pulse.end_time_seconds)
    )
    shape[rising] = (
        timeline_seconds[rising] - pulse.start_time_seconds
    ) / (peak_time - pulse.start_time_seconds)
    shape[falling] = (
        pulse.end_time_seconds - timeline_seconds[falling]
    ) / (pulse.end_time_seconds - peak_time)
    return shape


def generate_triangular_pulse(
    configuration: TemporalConfiguration,
    pulse: TriangularPulse,
    *,
    rainfall_index: int,
    phase_event_id: str,
    storm_id: str,
    role: str,
    profile_family: str,
) -> RainfallEvent:
    """Sample, depth-normalise, and validate one triangular rainfall pulse.

    The requested peak remains continuous and may fall between samples. The
    returned event uses the full configured timeline, and its active duration
    is the pulse's absolute endpoint rather than its pulse duration.
    """

    if not isinstance(configuration, TemporalConfiguration):
        raise ValueError("configuration must be a TemporalConfiguration")
    if not isinstance(pulse, TriangularPulse):
        raise ValueError("pulse must be a TriangularPulse")
    _validate_pulse_timing(configuration, pulse)

    timeline = generate_timeline(configuration)
    shape = _sample_dimensionless_shape(timeline, pulse)
    timeline_hours = timeline / 3600.0
    sampled_shape_integral_hours = float(np.trapz(shape, timeline_hours))
    if (
        not math.isfinite(sampled_shape_integral_hours)
        or sampled_shape_integral_hours <= 0.0
    ):
        raise ValueError(
            "sampled triangular pulse has zero integral; at least one positive "
            "interior timeline sample is required"
        )

    normalisation_factor = pulse.target_depth_mm / sampled_shape_integral_hours
    intensities = shape * normalisation_factor
    if not math.isfinite(normalisation_factor) or not np.isfinite(intensities).all():
        raise ValueError("triangular-pulse normalisation produced non-finite intensity")
    integrated_depth_mm = float(np.trapz(intensities, timeline_hours))
    sampled_maximum = float(np.max(intensities))
    sampled_maximum_times = tuple(
        int(value) for value in timeline[intensities == sampled_maximum]
    )

    metadata = MappingProxyType(
        {
            "pulse_start_time_seconds": pulse.start_time_seconds,
            "pulse_duration_seconds": pulse.duration_seconds,
            "pulse_end_time_seconds": pulse.end_time_seconds,
            "target_depth_mm": pulse.target_depth_mm,
            "integrated_depth_mm": integrated_depth_mm,
            "requested_peak_position_ratio": pulse.peak_position_ratio,
            "requested_peak_time_seconds": pulse.peak_time_seconds,
            "sampled_maximum_intensity_mm_per_hour": sampled_maximum,
            "sampled_maximum_times_seconds": sampled_maximum_times,
            "rainfall_timestep_seconds": configuration.rainfall_timestep_seconds,
            "sampled_shape_integral_hours": sampled_shape_integral_hours,
            "shape_normalisation_factor_mm_per_hour": normalisation_factor,
            "shape_normalisation_factor_meaning": SHAPE_NORMALISATION_MEANING,
            "depth_integration_method": DEPTH_INTEGRATION_METHOD,
        }
    )
    event = RainfallEvent(
        rainfall_index=rainfall_index,
        phase_event_id=phase_event_id,
        storm_id=storm_id,
        role=role,
        profile_family=profile_family,
        target_depth_mm=pulse.target_depth_mm,
        active_duration_seconds=pulse.end_time_seconds,
        intensities_mm_per_hour=intensities,
        peak_metadata=metadata,
    )
    validate_rainfall_event(event, configuration)
    return event
