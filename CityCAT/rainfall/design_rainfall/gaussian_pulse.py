"""Finite-duration, tapered skew-normal-based rainfall with discrete conservation.

The requested peak is the mode of the *unmodified* distribution. Tapering and
sampling can move the final maximum. This module uses NumPy and the standard
library only; no DDF lookup, frequency assignment or scientific defaults occur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np

from .core import (
    RainfallEvent,
    TemporalConfiguration,
    generate_timeline,
    validate_rainfall_event,
)


MODE_RELATIVE_TOLERANCE = 1e-13
MODE_MAX_ITERATIONS = 128
TAPER_DEFINITION = (
    "u=(t-start)/duration; e=edge_taper_ratio; "
    "w(u)=sin((pi/2)*min(u/e,(1-u)/e,1))**2 for 0<u<1; zero otherwise"
)
DEPTH_INTEGRATION_METHOD = "trapezoidal_sampled_piecewise_linear"


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


@dataclass(frozen=True)
class GaussianPulse:
    """Explicit parameters for a tapered Gaussian/skew-normal-based pulse.

    ``scale_ratio * duration_seconds`` is the unmodified distribution scale,
    not the effective temporal standard deviation of the sampled rainfall.
    ``skewness_parameter`` is skew-normal alpha, not measured skewness.
    """

    start_time_seconds: int
    duration_seconds: int
    target_depth_mm: float
    peak_position_ratio: float
    scale_ratio: float
    skewness_parameter: float
    edge_taper_ratio: float

    def __post_init__(self) -> None:
        for name, minimum in (("start_time_seconds", 0), ("duration_seconds", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
        for name in (
            "target_depth_mm", "peak_position_ratio", "scale_ratio",
            "skewness_parameter", "edge_taper_ratio",
        ):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        for name in ("target_depth_mm", "scale_ratio"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
        if not 0.0 < self.peak_position_ratio < 1.0:
            raise ValueError("peak_position_ratio must be strictly between 0 and 1")
        if not 0.0 < self.edge_taper_ratio < 0.5:
            raise ValueError("edge_taper_ratio must be strictly between 0 and 0.5")

    @property
    def end_time_seconds(self) -> int:
        """Absolute endpoint, consistent with RainfallEvent active duration."""
        return self.start_time_seconds + self.duration_seconds

    @property
    def peak_time_seconds(self) -> float:
        """Requested unmodified mode, without timestep snapping."""
        return self.start_time_seconds + self.peak_position_ratio * self.duration_seconds


def _bisect_mode_root(function, lower: float, upper: float) -> float:
    """Bisect a finite, increasing, sign-bracketed mode equation."""
    left, right = function(lower), function(upper)
    if not (math.isfinite(left) and math.isfinite(right) and left <= 0 <= right):
        raise ValueError("skew-normal mode calculation failed: invalid root bracket")
    if left == 0.0:
        return lower
    if right == 0.0:
        return upper
    for _ in range(MODE_MAX_ITERATIONS):
        middle = lower + (upper - lower) / 2.0
        value = function(middle)
        if not math.isfinite(value):
            raise ValueError("skew-normal mode calculation failed: non-finite residual")
        if value == 0.0 or upper - lower <= MODE_RELATIVE_TOLERANCE * abs(middle):
            return middle
        if value > 0.0:
            upper = middle
        else:
            lower = middle
    raise ValueError("skew-normal mode calculation failed: bisection did not converge")


def _standardized_mode(alpha: float) -> float:
    """Solve m*Phi(a*m) = a*phi(a*m), using m(-a)=-m(a).

    For 0<a<=1 solve q=phi(a*a*q)/Phi(a*a*q), where m=a*q and
    0<q<=sqrt(2/pi). For a>1 solve in v=a*m:
    log(v)+log(Phi(v))+v*v/2+log(2*pi)/2-2*log(a)=0.
    These increasing equations avoid tiny root brackets and a*a overflow.
    Bisection bounds the relative root error by the final bracket width
    (1e-13), with 128 iterations at most. No approximate-location fallback.
    """
    if alpha == 0.0:
        return 0.0
    a = abs(alpha)
    if a <= 1.0:
        def equation(q):
            v = a * a * q
            phi = math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)
            cdf = 0.5 * math.erfc(-v / math.sqrt(2.0))
            return q - phi / cdf

        q = _bisect_mode_root(equation, 0.0, math.sqrt(2.0 / math.pi))
        mode = a * q
    else:
        def equation(v):
            return (
                math.log(v) + math.log(0.5 * math.erfc(-v / math.sqrt(2.0)))
                + 0.5 * v * v + 0.5 * math.log(2.0 * math.pi) - 2.0 * math.log(a)
            )

        # At v=0.25 this is negative for every a>1. A finite float alpha
        # needs v<64; a bounded doubling search still checks the bracket.
        upper = 1.0
        for _ in range(7):
            if equation(upper) >= 0.0:
                break
            upper *= 2.0
        mode = _bisect_mode_root(equation, 0.25, upper) / a
    if not math.isfinite(mode) or mode <= 0.0:
        raise ValueError("skew-normal mode calculation failed: unrepresentable mode")
    return math.copysign(mode, alpha)


def _rainfall_diagnostics(
    timeline: np.ndarray, intensities: np.ndarray, pulse: GaussianPulse,
) -> tuple[float, float, float]:
    """Exact segment moments and partial depth of the linear interpolant.

    Use dimensionless pulse time and intensities divided by their maximum:
    both scale factors cancel in moment/depth ratios, avoiding large raw
    powers or near-zero rainfall weights. Integrate each linear segment
    analytically, then calculate the second moment about the centroid.
    """
    selected = (timeline >= pulse.start_time_seconds) & (timeline <= pulse.end_time_seconds)
    u = (timeline[selected] - pulse.start_time_seconds) / pulse.duration_seconds
    weights = intensities[selected] / float(np.max(intensities))
    segments = list(zip(u[:-1], u[1:], weights[:-1], weights[1:]))
    area = math.fsum((b - a) * (y0 + y1) / 2.0 for a, b, y0, y1 in segments)
    first = math.fsum(
        (b - a) * ((2.0 * a + b) * y0 + (a + 2.0 * b) * y1) / 6.0
        for a, b, y0, y1 in segments
    )
    mean = first / area
    variance_terms, partial_areas = [], []
    for a, b, y0, y1 in segments:
        h = b - a
        x0, x1 = a - mean, b - mean
        variance_terms.append(h / 12.0 * (
            (3.0 * x0 * x0 + 2.0 * x0 * x1 + x1 * x1) * y0
            + (x0 * x0 + 2.0 * x0 * x1 + 3.0 * x1 * x1) * y1
        ))
        length = min(max(pulse.peak_position_ratio - a, 0.0), h)
        fraction = length / h
        y_end = (1.0 - fraction) * y0 + fraction * y1
        partial_areas.append(length * (y0 + y_end) / 2.0)
    variance = math.fsum(variance_terms) / area
    centroid = pulse.start_time_seconds + pulse.duration_seconds * mean
    deviation = pulse.duration_seconds * math.sqrt(variance)
    fraction_before_peak = math.fsum(partial_areas) / area
    if not all(math.isfinite(value) for value in (centroid, deviation, fraction_before_peak)):
        raise ValueError("Gaussian rainfall diagnostics are non-finite")
    return centroid, deviation, fraction_before_peak


def generate_gaussian_pulse(
    configuration: TemporalConfiguration,
    pulse: GaussianPulse,
    *,
    rainfall_index: int,
    phase_event_id: str,
    storm_id: str,
    role: str,
    profile_family: str,
) -> RainfallEvent:
    """Sample, taper, depth-normalise and validate one Gaussian-family pulse.

    Sample 2*phi(z)*Phi(alpha*z), with z=(t-requested_peak)/scale+m(alpha).
    This is equivalent to location=requested_peak-scale*m, while avoiding
    subtraction of a potentially large location during shape evaluation.
    Phi uses erfc to avoid cancellation in the negative tail. Underflowed
    tails may become zero; fewer than two positive interior samples is a
    temporal-resolution error, not permission to snap or widen the pulse.
    """
    if not isinstance(configuration, TemporalConfiguration):
        raise ValueError("configuration must be a TemporalConfiguration")
    if not isinstance(pulse, GaussianPulse):
        raise ValueError("pulse must be a GaussianPulse")
    timestep = configuration.rainfall_timestep_seconds
    for name in ("start_time_seconds", "duration_seconds"):
        if getattr(pulse, name) % timestep:
            raise ValueError(f"{name} must align with the rainfall timestep")
    if pulse.end_time_seconds > configuration.simulation_horizon_seconds:
        raise ValueError("pulse endpoint exceeds the simulation horizon")
    if pulse.end_time_seconds > configuration.maximum_active_duration_seconds:
        raise ValueError("pulse endpoint exceeds the maximum active endpoint")

    scale = pulse.scale_ratio * pulse.duration_seconds
    mode = _standardized_mode(pulse.skewness_parameter)
    location = pulse.peak_time_seconds - scale * mode
    if not (math.isfinite(scale) and scale > 0 and math.isfinite(location)):
        raise ValueError("Gaussian scale/location is not representable as finite seconds")
    timeline = generate_timeline(configuration)
    shape = np.zeros(timeline.size, dtype=np.float64)
    interior = (timeline > pulse.start_time_seconds) & (timeline < pulse.end_time_seconds)
    for index in np.flatnonzero(interior):
        elapsed = float(timeline[index] - pulse.start_time_seconds)
        z = (elapsed - pulse.peak_position_ratio * pulse.duration_seconds) / scale + mode
        # erfc gives 2*Phi(alpha*z); handle alpha=0 separately if z overflows.
        cdf_twice = (
            1.0 if pulse.skewness_parameter == 0.0
            else math.erfc(-pulse.skewness_parameter * z / math.sqrt(2.0))
        )
        density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * cdf_twice
        taper_argument = min(
            min(elapsed, pulse.duration_seconds - elapsed)
            / (pulse.edge_taper_ratio * pulse.duration_seconds),
            1.0,
        )
        shape[index] = density * math.sin(math.pi / 2.0 * taper_argument) ** 2
    positive_samples = int(np.count_nonzero(shape[interior] > 0.0))
    if positive_samples < 2:
        raise ValueError(
            "Gaussian temporal-resolution error: at least two strictly positive "
            "interior samples are required; choose a finer timestep or another profile"
        )
    timeline_hours = timeline / 3600.0
    integral = float(np.trapz(shape, timeline_hours))
    if not math.isfinite(integral) or integral <= 0.0:
        raise ValueError("Gaussian sampled shape integral must be finite and positive")
    factor = pulse.target_depth_mm / integral
    with np.errstate(over="ignore", invalid="ignore"):
        intensities = shape * factor
    if not math.isfinite(factor) or not np.isfinite(intensities).all() or np.any(intensities < 0):
        raise ValueError("Gaussian normalisation produced non-finite or negative intensity")
    if np.count_nonzero(intensities[interior] > 0) < 2:
        raise ValueError("Gaussian temporal-resolution error after intensity underflow")
    integrated_depth = float(np.trapz(intensities, timeline_hours))
    if not math.isfinite(integrated_depth) or not math.isclose(
        integrated_depth, pulse.target_depth_mm, rel_tol=0.0,
        abs_tol=configuration.depth_tolerance_mm,
    ):
        raise ValueError("Gaussian integrated depth does not match target within depth tolerance")
    centroid, deviation, fraction = _rainfall_diagnostics(timeline, intensities, pulse)
    maximum = float(np.max(intensities))
    metadata = MappingProxyType({
        "pulse_start_time_seconds": pulse.start_time_seconds,
        "pulse_duration_seconds": pulse.duration_seconds,
        "pulse_end_time_seconds": pulse.end_time_seconds,
        "target_depth_mm": pulse.target_depth_mm,
        "requested_peak_position_ratio": pulse.peak_position_ratio,
        "scale_ratio": pulse.scale_ratio,
        "skewness_parameter": pulse.skewness_parameter,
        "edge_taper_ratio": pulse.edge_taper_ratio,
        "scale_seconds": scale,
        "standardized_mode": mode,
        "mode_calculation_method": "symmetry_and_bracketed_bisection_of_scaled_mode_equation",
        "mode_relative_tolerance": MODE_RELATIVE_TOLERANCE,
        "distribution_location_seconds": location,
        "requested_peak_time_seconds": pulse.peak_time_seconds,
        "sampled_maximum_intensity_mm_per_hour": maximum,
        "sampled_maximum_times_seconds": tuple(int(t) for t in timeline[intensities == maximum]),
        "sampled_shape_integral_hours": integral,
        "shape_normalisation_factor_mm_per_hour": factor,
        "shape_normalisation_factor_meaning": (
            "millimetres per hour applied to the dimensionless tapered skew-normal "
            "shape so its trapezoidal piecewise-linear integral equals target_depth_mm"
        ),
        "integrated_depth_mm": integrated_depth,
        "depth_integration_method": DEPTH_INTEGRATION_METHOD,
        "rainfall_timestep_seconds": timestep,
        "positive_interior_samples": int(np.count_nonzero(intensities[interior] > 0)),
        "taper_definition": TAPER_DEFINITION,
        "temporal_centroid_seconds": centroid,
        "effective_temporal_standard_deviation_seconds": deviation,
        "depth_fraction_up_to_requested_peak": fraction,
        "diagnostics_method": "analytic_segments_of_sampled_piecewise_linear_intensity",
    })
    event = RainfallEvent(
        rainfall_index=rainfall_index, phase_event_id=phase_event_id,
        storm_id=storm_id, role=role, profile_family=profile_family,
        target_depth_mm=pulse.target_depth_mm,
        active_duration_seconds=pulse.end_time_seconds,
        intensities_mm_per_hour=intensities, peak_metadata=metadata,
    )
    validate_rainfall_event(event, configuration)
    return event
