"""External experiment configuration for reusable design-rainfall workflows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .core import TemporalConfiguration


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class DurationRequest:
    """Requested durations as either explicit seconds or an inclusive range."""

    values_seconds: tuple[int, ...] | None = None
    range_start_seconds: int | None = None
    range_stop_seconds: int | None = None
    range_step_seconds: int | None = None

    def __post_init__(self) -> None:
        has_values = self.values_seconds is not None
        range_values = (
            self.range_start_seconds,
            self.range_stop_seconds,
            self.range_step_seconds,
        )
        has_any_range = any(value is not None for value in range_values)
        has_complete_range = all(value is not None for value in range_values)
        if has_values == has_any_range:
            raise ValueError(
                "requested durations must use exactly one of values_seconds or "
                "range_seconds"
            )
        if has_any_range and not has_complete_range:
            raise ValueError("duration range requires start, stop, and step seconds")

        if has_values:
            assert self.values_seconds is not None
            if not self.values_seconds:
                raise ValueError("requested duration values must not be empty")
            durations = self.values_seconds
        else:
            start = _positive_integer(self.range_start_seconds, "duration range start")
            stop = _positive_integer(self.range_stop_seconds, "duration range stop")
            step = _positive_integer(self.range_step_seconds, "duration range step")
            if stop < start:
                raise ValueError("duration range stop must not precede its start")
            if (stop - start) % step:
                raise ValueError("duration range stop must be reached exactly by its step")
            durations = tuple(range(start, stop + 1, step))

        for duration in durations:
            _positive_integer(duration, "requested duration")
        if len(set(durations)) != len(durations):
            raise ValueError("requested durations must be unique")

    @property
    def durations_seconds(self) -> tuple[int, ...]:
        """Resolve explicit values or an inclusive range to duration seconds."""

        if self.values_seconds is not None:
            return self.values_seconds
        assert self.range_start_seconds is not None
        assert self.range_stop_seconds is not None
        assert self.range_step_seconds is not None
        return tuple(
            range(
                self.range_start_seconds,
                self.range_stop_seconds + 1,
                self.range_step_seconds,
            )
        )

    def validate_against(self, temporal: TemporalConfiguration) -> None:
        """Validate resolved durations against a rainfall time grid."""

        for duration in self.durations_seconds:
            if duration % temporal.rainfall_timestep_seconds:
                raise ValueError(
                    f"requested duration {duration} does not align with the "
                    "rainfall timestep"
                )
            if duration > temporal.maximum_active_duration_seconds:
                raise ValueError(
                    f"requested duration {duration} exceeds maximum active duration"
                )


@dataclass(frozen=True)
class EvaluationRoleConfiguration:
    """Generic evaluation-role metadata without prescribing a split design."""

    strategy: str
    roles: tuple[str, ...]
    provisional: bool

    def __post_init__(self) -> None:
        _non_empty_string(self.strategy, "evaluation-role strategy")
        if not isinstance(self.provisional, bool):
            raise ValueError("evaluation-role provisional flag must be boolean")
        for role in self.roles:
            _non_empty_string(role, "evaluation role")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("evaluation roles must be unique")


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Validated external configuration for one rainfall experiment."""

    experiment_name: str
    experiment_version: str
    phase: str
    rainfall_index_start: int
    number_of_events: int
    random_seed: int
    temporal: TemporalConfiguration
    requested_durations: DurationRequest
    requested_return_periods_years: tuple[float, ...]
    permitted_profile_families: tuple[str, ...]
    evaluation_roles: EvaluationRoleConfiguration
    source_ddf_table_path: Path
    output_root: Path
    data_status: str

    def __post_init__(self) -> None:
        for name in ("experiment_name", "experiment_version", "phase"):
            _non_empty_string(getattr(self, name), name)
        _positive_integer(self.rainfall_index_start, "rainfall_index_start")
        _positive_integer(self.number_of_events, "number_of_events")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer and must not be bool")

        rainfall_index_end = self.rainfall_index_end
        if rainfall_index_end < self.rainfall_index_start or rainfall_index_end <= 0:
            raise ValueError("resulting rainfall-index range is invalid")
        if not isinstance(self.temporal, TemporalConfiguration):
            raise ValueError("temporal must be a TemporalConfiguration")
        if not isinstance(self.requested_durations, DurationRequest):
            raise ValueError("requested_durations must be a DurationRequest")
        self.requested_durations.validate_against(self.temporal)

        return_periods: list[float] = []
        for value in self.requested_return_periods_years:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("requested return periods must be numeric")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value <= 0.0:
                raise ValueError("requested return periods must be finite and positive")
            return_periods.append(numeric_value)
        if len(set(return_periods)) != len(return_periods):
            raise ValueError("requested return periods must be unique")

        if not self.permitted_profile_families:
            raise ValueError("at least one profile family must be permitted")
        for family in self.permitted_profile_families:
            _non_empty_string(family, "profile-family name")
        if len(set(self.permitted_profile_families)) != len(
            self.permitted_profile_families
        ):
            raise ValueError("permitted profile families must be unique")
        if not isinstance(self.evaluation_roles, EvaluationRoleConfiguration):
            raise ValueError(
                "evaluation_roles must be an EvaluationRoleConfiguration"
            )
        for name in ("source_ddf_table_path", "output_root"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be represented as a pathlib.Path")
        if self.data_status not in {"provisional", "final"}:
            raise ValueError("data_status must be 'provisional' or 'final'")

    @property
    def rainfall_index_end(self) -> int:
        """Return the inclusive final rainfall index."""

        return self.rainfall_index_start + self.number_of_events - 1


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"missing required configuration field: {name}")
    return mapping[name]


def _load_duration_request(value: object) -> DurationRequest:
    request = _mapping(value, "requested_durations")
    has_values = "values_seconds" in request
    has_range = "range_seconds" in request
    if has_values == has_range:
        raise ValueError(
            "requested_durations must contain exactly one of values_seconds or "
            "range_seconds"
        )
    if has_values:
        values = request["values_seconds"]
        if not isinstance(values, list):
            raise ValueError("requested duration values_seconds must be a JSON array")
        return DurationRequest(values_seconds=tuple(values))
    duration_range = _mapping(request["range_seconds"], "duration range_seconds")
    return DurationRequest(
        range_start_seconds=_required(duration_range, "start"),
        range_stop_seconds=_required(duration_range, "stop"),
        range_step_seconds=_required(duration_range, "step"),
    )


def _resolve_path(value: object, name: str, base_directory: Path) -> Path:
    text = _non_empty_string(value, name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def load_experiment_configuration(path: Path | str) -> ExperimentConfiguration:
    """Load and validate a JSON experiment configuration."""

    configuration_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(configuration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not load experiment configuration: {configuration_path}"
        ) from exc
    payload = _mapping(raw, "experiment configuration")
    temporal_raw = _mapping(_required(payload, "temporal"), "temporal")
    temporal = TemporalConfiguration(
        rainfall_timestep_seconds=_required(
            temporal_raw, "rainfall_timestep_seconds"
        ),
        simulation_horizon_seconds=_required(
            temporal_raw, "simulation_horizon_seconds"
        ),
        maximum_active_duration_seconds=_required(
            temporal_raw, "maximum_active_duration_seconds"
        ),
        depth_tolerance_mm=temporal_raw.get("depth_tolerance_mm", 1e-6),
        citycat_value_tolerance=temporal_raw.get(
            "citycat_value_tolerance", 1e-15
        ),
    )
    roles_raw = _mapping(
        _required(payload, "evaluation_roles"), "evaluation_roles"
    )
    roles = _required(roles_raw, "roles")
    if not isinstance(roles, list):
        raise ValueError("evaluation role names must be a JSON array")
    return_periods = _required(payload, "requested_return_periods_years")
    if not isinstance(return_periods, list):
        raise ValueError("requested_return_periods_years must be a JSON array")
    profile_families = _required(payload, "permitted_profile_families")
    if not isinstance(profile_families, list):
        raise ValueError("permitted_profile_families must be a JSON array")

    return ExperimentConfiguration(
        experiment_name=_required(payload, "experiment_name"),
        experiment_version=_required(payload, "experiment_version"),
        phase=_required(payload, "phase"),
        rainfall_index_start=_required(payload, "rainfall_index_start"),
        number_of_events=_required(payload, "number_of_events"),
        random_seed=_required(payload, "random_seed"),
        temporal=temporal,
        requested_durations=_load_duration_request(
            _required(payload, "requested_durations")
        ),
        requested_return_periods_years=tuple(return_periods),
        permitted_profile_families=tuple(profile_families),
        evaluation_roles=EvaluationRoleConfiguration(
            strategy=_required(roles_raw, "strategy"),
            roles=tuple(roles),
            provisional=_required(roles_raw, "provisional"),
        ),
        source_ddf_table_path=_resolve_path(
            _required(payload, "source_ddf_table_path"),
            "source_ddf_table_path",
            configuration_path.parent,
        ),
        output_root=_resolve_path(
            _required(payload, "output_root"),
            "output_root",
            configuration_path.parent,
        ),
        data_status=_required(payload, "data_status"),
    )
