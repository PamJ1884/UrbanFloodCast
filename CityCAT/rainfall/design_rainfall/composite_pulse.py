"""Reusable two-pulse triangular rainfall events with an explicit dry gap."""

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
from .triangular_pulse import (
    DEPTH_INTEGRATION_METHOD,
    TriangularPulse,
    generate_triangular_pulse,
)


@dataclass(frozen=True)
class CompositeTriangularPulse:
    """Two non-overlapping triangular pulses separated by a dry gap."""

    first_pulse: TriangularPulse
    second_pulse: TriangularPulse

    def __post_init__(self) -> None:
        if not isinstance(self.first_pulse, TriangularPulse):
            raise ValueError("first_pulse must be a TriangularPulse")
        if not isinstance(self.second_pulse, TriangularPulse):
            raise ValueError("second_pulse must be a TriangularPulse")

        if (
            self.second_pulse.start_time_seconds
            <= self.first_pulse.end_time_seconds
        ):
            raise ValueError(
                "second_pulse must start after first_pulse ends so that "
                "the composite event contains a strictly positive dry gap"
            )

    @property
    def target_depth_mm(self) -> float:
        """Return the total target depth of both pulses."""

        return (
            self.first_pulse.target_depth_mm
            + self.second_pulse.target_depth_mm
        )

    @property
    def dry_gap_seconds(self) -> int:
        """Return the dry interval between the two pulses."""

        return (
            self.second_pulse.start_time_seconds
            - self.first_pulse.end_time_seconds
        )

    @property
    def active_endpoint_seconds(self) -> int:
        """Return the absolute endpoint of the second pulse."""

        return self.second_pulse.end_time_seconds


def _integrated_depth_mm(
    intensities_mm_per_hour: np.ndarray,
    timeline_seconds: np.ndarray,
) -> float:
    """Return trapezoidal depth of a sampled rainfall series."""

    return float(
        np.trapz(
            intensities_mm_per_hour,
            timeline_seconds / 3600.0,
        )
    )


def generate_composite_triangular_pulse(
    configuration: TemporalConfiguration,
    composite: CompositeTriangularPulse,
    *,
    rainfall_index: int,
    phase_event_id: str,
    storm_id: str,
    role: str,
    profile_family: str,
) -> RainfallEvent:
    """Generate and validate a two-pulse triangular rainfall event.

    Each constituent pulse is generated using the Stage 3A triangular-pulse
    implementation and therefore conserves its own requested discrete
    trapezoidal depth. The two sampled intensity series are then summed and
    total event depth is independently checked.
    """

    if not isinstance(configuration, TemporalConfiguration):
        raise ValueError("configuration must be a TemporalConfiguration")
    if not isinstance(composite, CompositeTriangularPulse):
        raise ValueError("composite must be a CompositeTriangularPulse")

    # Reuse Stage 3A exactly for each constituent pulse.
    first_event = generate_triangular_pulse(
        configuration,
        composite.first_pulse,
        rainfall_index=rainfall_index,
        phase_event_id=phase_event_id,
        storm_id=storm_id,
        role=role,
        profile_family=f"{profile_family}_PULSE1",
    )

    second_event = generate_triangular_pulse(
        configuration,
        composite.second_pulse,
        rainfall_index=rainfall_index,
        phase_event_id=phase_event_id,
        storm_id=storm_id,
        role=role,
        profile_family=f"{profile_family}_PULSE2",
    )

    first_intensities = np.asarray(
        first_event.intensities_mm_per_hour,
        dtype=np.float64,
    )
    second_intensities = np.asarray(
        second_event.intensities_mm_per_hour,
        dtype=np.float64,
    )

    if first_intensities.shape != second_intensities.shape:
        raise ValueError(
            "constituent triangular pulses produced incompatible timelines"
        )

    # The dry-gap design forbids simultaneous positive rainfall.
    overlap = (
        (first_intensities > 0.0)
        & (second_intensities > 0.0)
    )
    if np.any(overlap):
        raise ValueError(
            "constituent triangular pulses overlap on the sampled timeline"
        )

    intensities = first_intensities + second_intensities

    if not np.isfinite(intensities).all():
        raise ValueError(
            "composite triangular pulse produced non-finite intensity"
        )

    timeline = generate_timeline(configuration)

    first_integrated_depth_mm = _integrated_depth_mm(
        first_intensities,
        timeline,
    )
    second_integrated_depth_mm = _integrated_depth_mm(
        second_intensities,
        timeline,
    )
    integrated_depth_mm = _integrated_depth_mm(
        intensities,
        timeline,
    )

    expected_total_depth_mm = composite.target_depth_mm

    if not math.isclose(
        integrated_depth_mm,
        expected_total_depth_mm,
        rel_tol=0.0,
        abs_tol=configuration.depth_tolerance_mm,
    ):
        raise ValueError(
            "composite triangular pulse does not conserve total target depth: "
            f"expected {expected_total_depth_mm}, "
            f"integrated {integrated_depth_mm}"
        )

    if not math.isclose(
        integrated_depth_mm,
        first_integrated_depth_mm + second_integrated_depth_mm,
        rel_tol=0.0,
        abs_tol=configuration.depth_tolerance_mm,
    ):
        raise ValueError(
            "composite depth does not equal the sum of constituent depths"
        )

    dry_gap_start_seconds = composite.first_pulse.end_time_seconds
    dry_gap_end_seconds = composite.second_pulse.start_time_seconds

    dry_gap_mask = (
        (timeline >= dry_gap_start_seconds)
        & (timeline <= dry_gap_end_seconds)
    )

    if np.any(intensities[dry_gap_mask] != 0.0):
        raise ValueError(
            "composite triangular pulse contains rainfall inside the dry gap"
        )

    sampled_maximum = float(np.max(intensities))
    sampled_maximum_times = tuple(
        int(value)
        for value in timeline[intensities == sampled_maximum]
    )

    metadata = MappingProxyType(
        {
            "number_of_pulses": 2,
            "target_depth_mm": expected_total_depth_mm,
            "integrated_depth_mm": integrated_depth_mm,
            "first_pulse_peak_metadata": first_event.peak_metadata,
            "second_pulse_peak_metadata": second_event.peak_metadata,
            "first_pulse_target_depth_mm": (
                composite.first_pulse.target_depth_mm
            ),
            "second_pulse_target_depth_mm": (
                composite.second_pulse.target_depth_mm
            ),
            "first_pulse_integrated_depth_mm": (
                first_integrated_depth_mm
            ),
            "second_pulse_integrated_depth_mm": (
                second_integrated_depth_mm
            ),
            "first_pulse_start_time_seconds": (
                composite.first_pulse.start_time_seconds
            ),
            "first_pulse_duration_seconds": (
                composite.first_pulse.duration_seconds
            ),
            "first_pulse_end_time_seconds": (
                composite.first_pulse.end_time_seconds
            ),
            "first_pulse_requested_peak_position_ratio": (
                composite.first_pulse.peak_position_ratio
            ),
            "first_pulse_requested_peak_time_seconds": (
                composite.first_pulse.peak_time_seconds
            ),
            "second_pulse_start_time_seconds": (
                composite.second_pulse.start_time_seconds
            ),
            "second_pulse_duration_seconds": (
                composite.second_pulse.duration_seconds
            ),
            "second_pulse_end_time_seconds": (
                composite.second_pulse.end_time_seconds
            ),
            "second_pulse_requested_peak_position_ratio": (
                composite.second_pulse.peak_position_ratio
            ),
            "second_pulse_requested_peak_time_seconds": (
                composite.second_pulse.peak_time_seconds
            ),
            "dry_gap_start_seconds": dry_gap_start_seconds,
            "dry_gap_end_seconds": dry_gap_end_seconds,
            "dry_gap_seconds": composite.dry_gap_seconds,
            "sampled_maximum_intensity_mm_per_hour": sampled_maximum,
            "sampled_maximum_times_seconds": sampled_maximum_times,
            "rainfall_timestep_seconds": (
                configuration.rainfall_timestep_seconds
            ),
            "active_endpoint_seconds": (
                composite.active_endpoint_seconds
            ),
            "depth_integration_method": DEPTH_INTEGRATION_METHOD,
        }
    )

    event = RainfallEvent(
        rainfall_index=rainfall_index,
        phase_event_id=phase_event_id,
        storm_id=storm_id,
        role=role,
        profile_family=profile_family,
        target_depth_mm=expected_total_depth_mm,
        active_duration_seconds=composite.active_endpoint_seconds,
        intensities_mm_per_hour=intensities,
        peak_metadata=metadata,
    )

    validate_rainfall_event(event, configuration)
    return event
