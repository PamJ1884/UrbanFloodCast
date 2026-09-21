"""Published FSR/FEH profile approximation and its interval-depth reference.

Y is central depth, not cumulative depth from storm start. Chronological C is
derived from symmetry. This reference does not implement full ReFH2 behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .core import TemporalConfiguration, generate_timeline


PROFILE_SOURCE_REFERENCE = (
    "https://refhdocs.hydrosolutions.co.uk/Design-DDF-Rainfall-Hyetographs/"
    "Design-Storm-Profiles/"
)
_PROFILE_PARAMETERS = {"summer": (0.100, 0.815), "winter": (0.060, 1.026)}


def _parameters(profile_name: str) -> tuple[float, float]:
    if not isinstance(profile_name, str) or profile_name not in _PROFILE_PARAMETERS:
        raise ValueError("profile_name must be 'summer' or 'winter'")
    return _PROFILE_PARAMETERS[profile_name]


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


def central_depth_fraction(x: float, *, profile_name: str) -> float:
    """Return Y(x)=(1-a**(x**b))/(1-a) for a central duration fraction.

    Evaluate with expm1 to avoid cancellation for small x, with exact values
    at 0 and 1. Only the published summer/winter parameter pairs are used.
    """
    a, b = _parameters(profile_name)
    x = _finite_real(x, "x")
    if not 0.0 <= x <= 1.0:
        raise ValueError("central duration fraction x must be between 0 and 1")
    if x in (0.0, 1.0):
        return x
    return -math.expm1(math.log(a) * x ** b) / (1.0 - a)


def chronological_cumulative_fraction(u: float, *, profile_name: str) -> float:
    """Return cumulative depth fraction from storm start, derived by symmetry.

    C(u)=[1-Y(1-2u)]/2 on [0,1/2], [1+Y(2u-1)]/2 on [1/2,1].
    C=0 before the event and C=1 after it. The left branch evaluates 1-Y
    using expm1/log1p rather than subtracting nearly equal numbers near u=0.
    No instantaneous derivative is evaluated, including at the centre.
    """
    a, b = _parameters(profile_name)
    u = _finite_real(u, "u")
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    if u == 0.5:
        return 0.5
    if u < 0.5:
        power_minus_one = math.expm1(b * math.log1p(-2.0 * u))
        return a * math.expm1(math.log(a) * power_minus_one) / (2.0 * (1.0 - a))
    return (1.0 + central_depth_fraction(2.0 * u - 1.0, profile_name=profile_name)) / 2.0


@dataclass(frozen=True)
class FSRFEHReferenceProfile:
    """Exact interval differences of the published approximation, in float64.

    Edges span the full configured timeline in absolute seconds. Cumulative
    and interval depths are mm; interval-average intensities are mm/hour.
    Tuples keep the reference values separate from any numerical adaptation.
    """

    configuration: TemporalConfiguration
    profile_name: str
    start_time_seconds: int
    duration_seconds: int
    target_depth_mm: float
    parameter_a: float
    parameter_b: float
    source_reference: str
    interval_edges_seconds: tuple[int, ...]
    cumulative_depths_mm: tuple[float, ...]
    interval_depths_mm: tuple[float, ...]
    interval_average_intensities_mm_per_hour: tuple[float, ...]
    intensity_units: str = "mm/hour"
    depth_units: str = "mm"
    time_units: str = "seconds"

    @property
    def end_time_seconds(self) -> int:
        return self.start_time_seconds + self.duration_seconds

    def cumulative_depth_at(self, time_seconds: float) -> float:
        """Evaluate the chronological reference curve at any finite time."""
        time = _finite_real(time_seconds, "time_seconds")
        return self.target_depth_mm * chronological_cumulative_fraction(
            (time - self.start_time_seconds) / self.duration_seconds,
            profile_name=self.profile_name,
        )


def build_fsr_feh_reference(
    configuration: TemporalConfiguration,
    *,
    profile_name: str,
    start_time_seconds: int,
    duration_seconds: int,
    target_depth_mm: float,
) -> FSRFEHReferenceProfile:
    """Discretize C by exact differences at aligned interval edges.

    Require explicit timing, positive depth and configuration. Apply no DDF
    lookup, ARF, seasonal correction, uplift, clipping or ReFH2.3 peak limiter.
    Summer/winter select temporal shape only, without a frequency inference.
    """
    if not isinstance(configuration, TemporalConfiguration):
        raise ValueError("configuration must be a TemporalConfiguration")
    a, b = _parameters(profile_name)
    for name, value, minimum in (
        ("start_time_seconds", start_time_seconds, 0),
        ("duration_seconds", duration_seconds, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
        if value % configuration.rainfall_timestep_seconds:
            raise ValueError(f"{name} must align with the rainfall timestep")
    endpoint = start_time_seconds + duration_seconds
    if endpoint > configuration.simulation_horizon_seconds:
        raise ValueError("profile endpoint exceeds the simulation horizon")
    if endpoint > configuration.maximum_active_duration_seconds:
        raise ValueError("profile endpoint exceeds the maximum active endpoint")
    depth = _finite_real(target_depth_mm, "target_depth_mm")
    if depth <= 0.0:
        raise ValueError("target_depth_mm must be strictly positive")
    edges = tuple(int(t) for t in generate_timeline(configuration))
    cumulative = tuple(depth * chronological_cumulative_fraction(
        (t - start_time_seconds) / duration_seconds, profile_name=profile_name,
    ) for t in edges)
    intervals = tuple(right - left for left, right in zip(cumulative, cumulative[1:]))
    timestep_hours = configuration.rainfall_timestep_seconds / 3600.0
    intensities = tuple(value / timestep_hours for value in intervals)
    if any(not math.isfinite(v) or v < 0 for v in intervals + intensities):
        raise ValueError("reference interval depths/intensities must be finite and non-negative")
    if not math.isclose(math.fsum(intervals), depth, rel_tol=0.0,
                        abs_tol=configuration.depth_tolerance_mm):
        raise ValueError("reference interval depths do not conserve target within tolerance")
    return FSRFEHReferenceProfile(
        configuration, profile_name, start_time_seconds, duration_seconds, depth,
        a, b, PROFILE_SOURCE_REFERENCE, edges, cumulative, intervals, intensities,
    )
