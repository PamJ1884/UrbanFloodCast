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
from .gridded_ddf import (
    ArcInfoASCIIGrid,
    GridPointDDFExtraction,
    GriddedDDFCatalogue,
    GriddedDDFCatalogueEntry,
    GriddedDDFExtractionResult,
    extract_ddf_at_point,
    load_gridded_ddf_catalogue,
    read_arcinfo_ascii_grid,
)
from .triangular_pulse import TriangularPulse, generate_triangular_pulse

__all__ = [
    "ArcInfoASCIIGrid",
    "ClimateUpliftResult",
    "DDFEstimate",
    "DDFRecord",
    "DDFTable",
    "DurationRequest",
    "EvaluationRoleConfiguration",
    "ExperimentConfiguration",
    "GridPointDDFExtraction",
    "GriddedDDFCatalogue",
    "GriddedDDFCatalogueEntry",
    "GriddedDDFExtractionResult",
    "RainfallEvent",
    "TemporalConfiguration",
    "TriangularPulse",
    "apply_climate_uplift",
    "export_citycat_rainfall",
    "extract_ddf_at_point",
    "generate_timeline",
    "generate_triangular_pulse",
    "validate_citycat_rainfall",
    "validate_rainfall_event",
    "load_ddf_csv",
    "load_experiment_configuration",
    "load_gridded_ddf_catalogue",
    "read_arcinfo_ascii_grid",
]
