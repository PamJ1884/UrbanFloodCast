"""TEST / experimental interval-to-node adaptation; not an official ReFH method.

This module is deliberately absent from the package's public exports. Reference
interval depths remain unchanged. Conservation of total depth does not imply
conservation of the reference depth in each interval.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .core import RainfallEvent, validate_rainfall_event
from .fsr_feh_profiles import FSRFEHReferenceProfile, _finite_real


@dataclass(frozen=True)
class ExperimentalFSRFEHAdaptation:
    """TEST adaptation and diagnostics, with the exact reference attached.

    Maximum cumulative error is measured at *reference interval edges*, not
    asserted to be the supremum of the difference between continuous curves.
    Use cumulative_depth_at for additional between-edge evaluation and plots.
    """

    reference: FSRFEHReferenceProfile
    event: RainfallEvent
    depth_before_normalization_mm: float
    normalization_factor: float
    final_total_depth_mm: float
    adapted_interval_depths_mm: tuple[float, ...]
    adapted_cumulative_depths_mm: tuple[float, ...]
    maximum_absolute_cumulative_depth_error_mm: float
    maximum_absolute_cumulative_depth_error_fraction: float
    reference_maximum_interval_average_intensity_mm_per_hour: float
    adapted_maximum_nodal_intensity_mm_per_hour: float
    adapted_maximum_nodal_times_seconds: tuple[int, ...]
    cumulative_error_evaluation: str = "reference_interval_edges"

    def cumulative_depth_at(self, time_seconds: float) -> float:
        """Integrate the sampled linear intensity, including a partial interval.

        The within-interval cumulative curve is quadratic, not a straight
        interpolation between cumulative values at interval edges.
        """
        time = _finite_real(time_seconds, "time_seconds")
        edges = self.reference.interval_edges_seconds
        if time <= edges[0]:
            return 0.0
        if time >= edges[-1]:
            return self.adapted_cumulative_depths_mm[-1]
        index = bisect_right(edges, time) - 1
        length = time - edges[index]
        fraction = length / (edges[index + 1] - edges[index])
        values = self.event.intensities_mm_per_hour
        intensity_at_time = (1.0 - fraction) * values[index] + fraction * values[index + 1]
        partial = length / 3600.0 * (values[index] / 2.0 + intensity_at_time / 2.0)
        return self.adapted_cumulative_depths_mm[index] + float(partial)


def adapt_fsr_feh_reference_experimental(
    reference: FSRFEHReferenceProfile,
    *,
    rainfall_index: int,
    phase_event_id: str,
    storm_id: str,
    role: str,
    profile_family: str,
) -> ExperimentalFSRFEHAdaptation:
    """TEST candidate: adjacent interval means, zero endpoints, global scaling.

    At least two rainfall intervals (one positive interior node) are needed.
    This structural minimum does not establish adequate temporal resolution.
    No limiter, seasonal correction or adjustment of the reference is applied.
    """
    if not isinstance(reference, FSRFEHReferenceProfile):
        raise ValueError("reference must be an FSRFEHReferenceProfile")
    configuration = reference.configuration
    timestep = configuration.rainfall_timestep_seconds
    start = reference.start_time_seconds // timestep
    end = reference.end_time_seconds // timestep
    if end - start < 2:
        raise ValueError("insufficient temporal support: at least two rainfall intervals required")
    average = np.asarray(reference.interval_average_intensities_mm_per_hour)
    timeline = np.asarray(reference.interval_edges_seconds)
    if average.shape != (timeline.size - 1,) or not np.isfinite(average).all() or np.any(average < 0):
        raise ValueError("reference interval intensities must be finite and non-negative")
    candidate = np.zeros(timeline.size, dtype=np.float64)
    candidate[start + 1:end] = average[start:end - 1] / 2.0 + average[start + 1:end] / 2.0
    before = float(np.trapz(candidate, timeline / 3600.0))
    if not math.isfinite(before) or before <= 0:
        raise ValueError("experimental candidate integral must be finite and positive")
    factor = reference.target_depth_mm / before
    with np.errstate(over="ignore", invalid="ignore"):
        intensities = candidate * factor
    if not math.isfinite(factor) or not np.isfinite(intensities).all() or np.any(intensities < 0):
        raise ValueError("experimental normalization produced invalid intensities")
    final = float(np.trapz(intensities, timeline / 3600.0))
    intervals = tuple(float(value) for value in (
        (intensities[:-1] / 2.0 + intensities[1:] / 2.0) * (timestep / 3600.0)
    ))
    cumulative = (0.0,) + tuple(float(value) for value in np.cumsum(intervals))
    maximum_error = max(abs(left - right) for left, right in zip(
        reference.cumulative_depths_mm, cumulative,
    ))
    maximum = float(np.max(intensities))
    maximum_times = tuple(int(t) for t in timeline[intensities == maximum])
    reference_maximum = float(np.max(average))
    metadata = MappingProxyType({
        "status": "TEST / experimental; not an official ReFH conversion",
        "profile_name": reference.profile_name,
        "parameter_a": reference.parameter_a,
        "parameter_b": reference.parameter_b,
        "source_reference": reference.source_reference,
        "pulse_start_time_seconds": reference.start_time_seconds,
        "pulse_duration_seconds": reference.duration_seconds,
        "pulse_end_time_seconds": reference.end_time_seconds,
        "target_depth_mm": reference.target_depth_mm,
        "reference_central_time_seconds": reference.start_time_seconds + reference.duration_seconds / 2.0,
        "depth_before_normalization_mm": before,
        "normalization_factor": factor,
        "normalization_factor_units": "dimensionless",
        "integrated_depth_mm": final,
        "depth_integration_method": "trapezoidal_sampled_piecewise_linear",
        "rainfall_timestep_seconds": timestep,
        "maximum_absolute_cumulative_depth_error_mm": maximum_error,
        "maximum_absolute_cumulative_depth_error_fraction": maximum_error / reference.target_depth_mm,
        "cumulative_error_evaluation": "reference_interval_edges",
        "reference_maximum_interval_average_intensity_mm_per_hour": reference_maximum,
        "adapted_maximum_nodal_intensity_mm_per_hour": maximum,
        "adapted_maximum_nodal_times_seconds": maximum_times,
    })
    # Immutable samples keep the cumulative diagnostics consistent with the event.
    intensities = np.frombuffer(intensities.tobytes(), dtype=np.float64)
    event = RainfallEvent(
        rainfall_index=rainfall_index, phase_event_id=phase_event_id,
        storm_id=storm_id, role=role, profile_family=profile_family,
        target_depth_mm=reference.target_depth_mm,
        active_duration_seconds=reference.end_time_seconds,
        intensities_mm_per_hour=intensities, peak_metadata=metadata,
    )
    validate_rainfall_event(event, configuration)
    return ExperimentalFSRFEHAdaptation(
        reference, event, before, factor, final, intervals, cumulative,
        maximum_error, maximum_error / reference.target_depth_mm,
        reference_maximum, maximum, maximum_times,
    )
