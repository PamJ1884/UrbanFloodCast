"""Reusable building blocks for Stockbridge design-rainfall workflows."""

from .config import (
    DurationRequest,
    EvaluationRoleConfiguration,
    ExperimentConfiguration,
    load_experiment_configuration,
)
from .core import (
    RainfallEvent,
    TemporalConfiguration,
    export_citycat_rainfall,
    generate_timeline,
    validate_citycat_rainfall,
    validate_rainfall_event,
)
from .ddf import (
    ClimateUpliftResult,
    DDFEstimate,
    DDFRecord,
    DDFTable,
    apply_climate_uplift,
    load_ddf_csv,
)

__all__ = [
    "ClimateUpliftResult",
    "DDFEstimate",
    "DDFRecord",
    "DDFTable",
    "DurationRequest",
    "EvaluationRoleConfiguration",
    "ExperimentConfiguration",
    "RainfallEvent",
    "TemporalConfiguration",
    "apply_climate_uplift",
    "export_citycat_rainfall",
    "generate_timeline",
    "validate_citycat_rainfall",
    "validate_rainfall_event",
    "load_ddf_csv",
    "load_experiment_configuration",
]
