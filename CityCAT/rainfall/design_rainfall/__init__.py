"""Reusable building blocks for Stockbridge design-rainfall workflows."""

from .config import (
    DurationRequest,
    EvaluationRoleConfiguration,
    ExperimentConfiguration,
    load_experiment_configuration,
)
from .composite_pulse import (
    CompositeTriangularPulse,
    generate_composite_triangular_pulse,
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
from .feh_point_ddf import (
    FEHPointDDFMetadata,
    FEHPointDDFResult,
    load_feh_point_ddf_csv,
)
from .gaussian_pulse import GaussianPulse, generate_gaussian_pulse
from .fsr_feh_profiles import (
    FSRFEHReferenceProfile,
    build_fsr_feh_reference,
    central_depth_fraction,
    chronological_cumulative_fraction,
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
    "CompositeTriangularPulse",
    "DDFEstimate",
    "DDFRecord",
    "DDFTable",
    "DurationRequest",
    "EvaluationRoleConfiguration",
    "ExperimentConfiguration",
    "FEHPointDDFMetadata",
    "FEHPointDDFResult",
    "FSRFEHReferenceProfile",
    "GaussianPulse",
    "GridPointDDFExtraction",
    "GriddedDDFCatalogue",
    "GriddedDDFCatalogueEntry",
    "GriddedDDFExtractionResult",
    "RainfallEvent",
    "TemporalConfiguration",
    "TriangularPulse",
    "apply_climate_uplift",
    "build_fsr_feh_reference",
    "central_depth_fraction",
    "chronological_cumulative_fraction",
    "export_citycat_rainfall",
    "extract_ddf_at_point",
    "generate_timeline",
    "generate_composite_triangular_pulse",
    "generate_gaussian_pulse",
    "generate_triangular_pulse",
    "validate_citycat_rainfall",
    "validate_rainfall_event",
    "load_ddf_csv",
    "load_experiment_configuration",
    "load_feh_point_ddf_csv",
    "load_gridded_ddf_catalogue",
    "read_arcinfo_ascii_grid",
]
