"""Generate the Stockbridge Phase 3 CityCAT rainfall production dataset.

The Phase 3 master manifest is the single source of truth for event identity,
hydrological design values, temporal-profile family, and profile parameters.

Expected design:
    40 triangular
    80 Gaussian
    80 FSR/FEH experimental adaptations
   200 events total

Outputs:
    <output-dir>/citycat_txt/Rainfall_Data_1.txt ... Rainfall_Data_200.txt
    <output-dir>/generation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    GaussianPulse,
    TemporalConfiguration,
    TriangularPulse,
    build_fsr_feh_reference,
    export_citycat_rainfall,
    generate_gaussian_pulse,
    generate_timeline,
    generate_triangular_pulse,
    validate_citycat_rainfall,
)

from CityCAT.rainfall.design_rainfall.experimental_fsr_feh import (
    adapt_fsr_feh_reference_experimental,
)


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "design_rainfall"
    / "manifests"
    / "phase3_stockbridge_manifest.csv"
)

EXPECTED_TOTAL = 200

EXPECTED_FAMILY_COUNTS = {
    "triangular": 40,
    "gaussian": 80,
    "fsr_feh": 80,
}

DATA_HEADER = "time_s rainfall_m_s"

SUMMARY_FIELDS = [
    "phase3_sequence",
    "phase3_event_id",
    "family",
    "profile_subtype",
    "backbone_id",
    "filename",
    "target_depth_mm",
    "integrated_depth_mm",
    "depth_error_mm",
    "maximum_intensity_mm_per_hour",
    "maximum_intensity_times_seconds",
    "citycat_sha256",
    "source_manifest_sha256",
    "design_qa_status",
    "generation_qa_status",
    "fsr_feh_maximum_cumulative_error_mm",
    "fsr_feh_maximum_cumulative_error_fraction",
    "fsr_feh_reference_maximum_interval_average_intensity_mm_per_hour",
    "fsr_feh_adapted_maximum_nodal_intensity_mm_per_hour",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def finite_float(value: str, field_name: str, event_id: str) -> float:
    """Parse one finite floating-point manifest value."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{event_id}: {field_name} must be numeric, got {value!r}"
        ) from exc

    if not math.isfinite(parsed):
        raise ValueError(
            f"{event_id}: {field_name} must be finite, got {value!r}"
        )

    return parsed


def integer_value(value: str, field_name: str, event_id: str) -> int:
    """Parse one integer manifest value."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{event_id}: {field_name} must be an integer, got {value!r}"
        ) from exc

    return parsed


def require_blank(row: dict[str, str], fields: tuple[str, ...]) -> None:
    """Require family-inapplicable manifest fields to be empty."""

    event_id = row["phase3_event_id"]

    for field_name in fields:
        if row[field_name].strip():
            raise ValueError(
                f"{event_id}: {field_name} must be blank for "
                f"family={row['family']}"
            )


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the Phase 3 master manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("manifest has no header")

        required = {
            "phase3_sequence",
            "phase3_event_id",
            "family",
            "profile_subtype",
            "backbone_id",
            "duration_min",
            "return_period_years",
            "feh_depth_mm",
            "ddf_kind",
            "ddf_method",
            "ddf_source_durations_min",
            "start_time_seconds",
            "rainfall_timestep_seconds",
            "simulation_horizon_seconds",
            "peak_position_ratio",
            "gaussian_alpha",
            "gaussian_scale_ratio",
            "gaussian_edge_taper_ratio",
            "fsr_feh_parameter_a",
            "fsr_feh_parameter_b",
            "qa_status",
        }

        missing = sorted(required - set(reader.fieldnames))

        if missing:
            raise ValueError(
                "manifest is missing required columns: "
                + ", ".join(missing)
            )

        rows = list(reader)

    return rows


def validate_manifest(
    rows: list[dict[str, str]],
) -> TemporalConfiguration:
    """Validate the complete Phase 3 design before generating any files."""

    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(
            f"manifest must contain exactly {EXPECTED_TOTAL} rows; "
            f"found {len(rows)}"
        )

    sequences = [
        integer_value(
            row["phase3_sequence"],
            "phase3_sequence",
            row["phase3_event_id"],
        )
        for row in rows
    ]

    if sequences != list(range(1, EXPECTED_TOTAL + 1)):
        raise ValueError(
            "phase3_sequence must be ordered exactly from 1 to 200"
        )

    event_ids = [row["phase3_event_id"].strip() for row in rows]

    if any(not event_id for event_id in event_ids):
        raise ValueError("phase3_event_id must not be blank")

    if len(set(event_ids)) != EXPECTED_TOTAL:
        raise ValueError("phase3_event_id values must be unique")

    family_counts = Counter(row["family"].strip() for row in rows)

    if dict(family_counts) != EXPECTED_FAMILY_COUNTS:
        raise ValueError(
            "unexpected family counts: "
            f"{dict(family_counts)}"
        )

    first = rows[0]

    timestep = integer_value(
        first["rainfall_timestep_seconds"],
        "rainfall_timestep_seconds",
        first["phase3_event_id"],
    )

    horizon = integer_value(
        first["simulation_horizon_seconds"],
        "simulation_horizon_seconds",
        first["phase3_event_id"],
    )

    configuration = TemporalConfiguration(
        rainfall_timestep_seconds=timestep,
        simulation_horizon_seconds=horizon,
        maximum_active_duration_seconds=horizon,
        depth_tolerance_mm=1e-6,
    )

    backbone_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        event_id = row["phase3_event_id"].strip()
        family = row["family"].strip()
        subtype = row["profile_subtype"].strip()
        backbone = row["backbone_id"].strip()

        if row["qa_status"].strip() != "PASS":
            raise ValueError(
                f"{event_id}: design qa_status must be PASS"
            )

        if not subtype:
            raise ValueError(
                f"{event_id}: profile_subtype must not be blank"
            )

        if not backbone:
            raise ValueError(
                f"{event_id}: backbone_id must not be blank"
            )

        row_timestep = integer_value(
            row["rainfall_timestep_seconds"],
            "rainfall_timestep_seconds",
            event_id,
        )

        row_horizon = integer_value(
            row["simulation_horizon_seconds"],
            "simulation_horizon_seconds",
            event_id,
        )

        if row_timestep != timestep or row_horizon != horizon:
            raise ValueError(
                f"{event_id}: all events must use the same "
                "timestep and simulation horizon"
            )

        start = integer_value(
            row["start_time_seconds"],
            "start_time_seconds",
            event_id,
        )

        duration_min = integer_value(
            row["duration_min"],
            "duration_min",
            event_id,
        )

        if start < 0:
            raise ValueError(
                f"{event_id}: start_time_seconds must be non-negative"
            )

        if duration_min <= 0:
            raise ValueError(
                f"{event_id}: duration_min must be positive"
            )

        duration_seconds = duration_min * 60

        if start % timestep:
            raise ValueError(
                f"{event_id}: start time must align with timestep"
            )

        if duration_seconds % timestep:
            raise ValueError(
                f"{event_id}: duration must align with timestep"
            )

        if start + duration_seconds > horizon:
            raise ValueError(
                f"{event_id}: rainfall extends beyond simulation horizon"
            )

        return_period = finite_float(
            row["return_period_years"],
            "return_period_years",
            event_id,
        )

        depth = finite_float(
            row["feh_depth_mm"],
            "feh_depth_mm",
            event_id,
        )

        if return_period <= 0.0:
            raise ValueError(
                f"{event_id}: return period must be positive"
            )

        if depth <= 0.0:
            raise ValueError(
                f"{event_id}: FEH depth must be positive"
            )

        if family == "triangular":
            peak_ratio = finite_float(
                row["peak_position_ratio"],
                "peak_position_ratio",
                event_id,
            )

            if not 0.0 < peak_ratio < 1.0:
                raise ValueError(
                    f"{event_id}: triangular peak ratio must be in (0, 1)"
                )

            require_blank(
                row,
                (
                    "gaussian_alpha",
                    "gaussian_scale_ratio",
                    "gaussian_edge_taper_ratio",
                    "fsr_feh_parameter_a",
                    "fsr_feh_parameter_b",
                ),
            )

        elif family == "gaussian":
            peak_ratio = finite_float(
                row["peak_position_ratio"],
                "peak_position_ratio",
                event_id,
            )

            scale_ratio = finite_float(
                row["gaussian_scale_ratio"],
                "gaussian_scale_ratio",
                event_id,
            )

            finite_float(
                row["gaussian_alpha"],
                "gaussian_alpha",
                event_id,
            )

            taper_ratio = finite_float(
                row["gaussian_edge_taper_ratio"],
                "gaussian_edge_taper_ratio",
                event_id,
            )

            if not 0.0 < peak_ratio < 1.0:
                raise ValueError(
                    f"{event_id}: Gaussian peak ratio must be in (0, 1)"
                )

            if scale_ratio <= 0.0:
                raise ValueError(
                    f"{event_id}: Gaussian scale ratio must be positive"
                )

            if taper_ratio < 0.0:
                raise ValueError(
                    f"{event_id}: Gaussian taper ratio must be non-negative"
                )

            require_blank(
                row,
                (
                    "fsr_feh_parameter_a",
                    "fsr_feh_parameter_b",
                ),
            )

        elif family == "fsr_feh":
            if subtype not in {"summer", "winter"}:
                raise ValueError(
                    f"{event_id}: FSR/FEH subtype must be summer or winter"
                )

            parameter_a = finite_float(
                row["fsr_feh_parameter_a"],
                "fsr_feh_parameter_a",
                event_id,
            )

            parameter_b = finite_float(
                row["fsr_feh_parameter_b"],
                "fsr_feh_parameter_b",
                event_id,
            )

            if parameter_a <= 0.0 or parameter_b <= 0.0:
                raise ValueError(
                    f"{event_id}: FSR/FEH parameters must be positive"
                )

            require_blank(
                row,
                (
                    "peak_position_ratio",
                    "gaussian_alpha",
                    "gaussian_scale_ratio",
                    "gaussian_edge_taper_ratio",
                ),
            )

        else:
            raise ValueError(
                f"{event_id}: unsupported family {family!r}"
            )

        backbone_rows[backbone].append(row)

    if len(backbone_rows) != 40:
        raise ValueError(
            f"expected 40 unique backbones; found {len(backbone_rows)}"
        )

    for backbone, group in sorted(backbone_rows.items()):
        if len(group) != 5:
            raise ValueError(
                f"{backbone}: expected 5 Phase 3 events; found {len(group)}"
            )

        counts = Counter(row["family"] for row in group)

        if counts != Counter(
            {
                "triangular": 1,
                "gaussian": 2,
                "fsr_feh": 2,
            }
        ):
            raise ValueError(
                f"{backbone}: unexpected family composition {dict(counts)}"
            )

        signatures = {
            (
                integer_value(
                    row["duration_min"],
                    "duration_min",
                    row["phase3_event_id"],
                ),
                finite_float(
                    row["return_period_years"],
                    "return_period_years",
                    row["phase3_event_id"],
                ),
                finite_float(
                    row["feh_depth_mm"],
                    "feh_depth_mm",
                    row["phase3_event_id"],
                ),
                row["ddf_kind"],
                row["ddf_method"],
                row["ddf_source_durations_min"],
            )
            for row in group
        }

        if len(signatures) != 1:
            raise ValueError(
                f"{backbone}: hydrological design differs across profiles"
            )

    return configuration


def build_metadata_lines(
    row: dict[str, str],
) -> tuple[str, str]:
    """Build the two optional CityCAT metadata comment lines."""

    family = row["family"]
    subtype = row["profile_subtype"]

    line_1 = (
        f"backbone={row['backbone_id']} "
        f"family={family} "
        f"subtype={subtype}"
    )

    base = (
        f"D_min={row['duration_min']} "
        f"T_years={row['return_period_years']} "
        f"depth_mm={row['feh_depth_mm']}"
    )

    if family == "triangular":
        line_2 = (
            f"{base} "
            f"peak_ratio={row['peak_position_ratio']}"
        )

    elif family == "gaussian":
        line_2 = (
            f"{base} "
            f"peak_ratio={row['peak_position_ratio']} "
            f"alpha={row['gaussian_alpha']} "
            f"scale={row['gaussian_scale_ratio']} "
            f"taper={row['gaussian_edge_taper_ratio']}"
        )

    elif family == "fsr_feh":
        line_2 = (
            f"{base} "
            f"a={row['fsr_feh_parameter_a']} "
            f"b={row['fsr_feh_parameter_b']}"
        )

    else:
        raise ValueError(
            f"unsupported family: {family}"
        )

    return line_1, line_2


def generate_event(
    row: dict[str, str],
    configuration: TemporalConfiguration,
):
    """Generate one RainfallEvent from one validated manifest row."""

    sequence = int(row["phase3_sequence"])
    event_id = row["phase3_event_id"]
    family = row["family"]

    start = int(row["start_time_seconds"])
    duration_seconds = int(row["duration_min"]) * 60
    depth = float(row["feh_depth_mm"])

    identity = {
        "rainfall_index": sequence,
        "phase_event_id": event_id,
        "storm_id": event_id,
        "role": "phase3_production",
        "profile_family": family,
    }

    diagnostics = {
        "fsr_feh_maximum_cumulative_error_mm": "",
        "fsr_feh_maximum_cumulative_error_fraction": "",
        "fsr_feh_reference_maximum_interval_average_intensity_mm_per_hour": "",
        "fsr_feh_adapted_maximum_nodal_intensity_mm_per_hour": "",
    }

    if family == "triangular":
        pulse = TriangularPulse(
            start_time_seconds=start,
            duration_seconds=duration_seconds,
            target_depth_mm=depth,
            peak_position_ratio=float(
                row["peak_position_ratio"]
            ),
        )

        event = generate_triangular_pulse(
            configuration,
            pulse,
            **identity,
        )

    elif family == "gaussian":
        pulse = GaussianPulse(
            start_time_seconds=start,
            duration_seconds=duration_seconds,
            target_depth_mm=depth,
            peak_position_ratio=float(
                row["peak_position_ratio"]
            ),
            scale_ratio=float(
                row["gaussian_scale_ratio"]
            ),
            skewness_parameter=float(
                row["gaussian_alpha"]
            ),
            edge_taper_ratio=float(
                row["gaussian_edge_taper_ratio"]
            ),
        )

        event = generate_gaussian_pulse(
            configuration,
            pulse,
            **identity,
        )

    elif family == "fsr_feh":
        reference = build_fsr_feh_reference(
            configuration,
            profile_name=row["profile_subtype"],
            start_time_seconds=start,
            duration_seconds=duration_seconds,
            target_depth_mm=depth,
        )

        manifest_a = float(
            row["fsr_feh_parameter_a"]
        )
        manifest_b = float(
            row["fsr_feh_parameter_b"]
        )

        if not math.isclose(
            reference.parameter_a,
            manifest_a,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{event_id}: FSR/FEH parameter a differs "
                "between manifest and reference builder"
            )

        if not math.isclose(
            reference.parameter_b,
            manifest_b,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{event_id}: FSR/FEH parameter b differs "
                "between manifest and reference builder"
            )

        adaptation = adapt_fsr_feh_reference_experimental(
            reference,
            **identity,
        )

        event = adaptation.event

        diagnostics = {
            "fsr_feh_maximum_cumulative_error_mm": (
                f"{adaptation.maximum_absolute_cumulative_depth_error_mm:.17g}"
            ),
            "fsr_feh_maximum_cumulative_error_fraction": (
                f"{adaptation.maximum_absolute_cumulative_depth_error_fraction:.17g}"
            ),
            "fsr_feh_reference_maximum_interval_average_intensity_mm_per_hour": (
                f"{adaptation.reference_maximum_interval_average_intensity_mm_per_hour:.17g}"
            ),
            "fsr_feh_adapted_maximum_nodal_intensity_mm_per_hour": (
                f"{adaptation.adapted_maximum_nodal_intensity_mm_per_hour:.17g}"
            ),
        }

    else:
        raise ValueError(
            f"{event_id}: unsupported family {family!r}"
        )

    return event, diagnostics


def write_summary(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Write the production-generation QA summary."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    manifest_path: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    """Generate and validate the complete Phase 3 production dataset."""

    rows = read_manifest(manifest_path)

    configuration = validate_manifest(rows)

    manifest_sha256 = sha256_file(
        manifest_path
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_txt_directory = (
        output_root / "citycat_txt"
    )

    final_summary = (
        output_root / "generation_summary.csv"
    )

    if (
        final_txt_directory.exists()
        or final_summary.exists()
    ) and not overwrite:
        raise FileExistsError(
            "Phase 3 output already exists. "
            "Use --overwrite only if replacement is intentional."
        )

    with tempfile.TemporaryDirectory(
        dir=output_root,
        prefix=".phase3_build_",
    ) as temporary_directory:
        staging_root = Path(
            temporary_directory
        )

        staging_txt = (
            staging_root / "citycat_txt"
        )

        staging_txt.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_rows = []

        timeline = generate_timeline(
            configuration
        )

        for row in rows:
            event, diagnostics = generate_event(
                row,
                configuration,
            )

            metadata_lines = build_metadata_lines(
                row
            )

            path = export_citycat_rainfall(
                staging_txt,
                event,
                configuration,
                metadata_lines=metadata_lines,
                data_header=DATA_HEADER,
            )

            validate_citycat_rainfall(
                path,
                event,
                configuration,
                metadata_lines=metadata_lines,
                data_header=DATA_HEADER,
            )

            intensities = np.asarray(
                event.intensities_mm_per_hour,
                dtype=np.float64,
            )

            integrated_depth = float(
                np.trapz(
                    intensities,
                    timeline / 3600.0,
                )
            )

            target_depth = float(
                row["feh_depth_mm"]
            )

            depth_error = (
                integrated_depth - target_depth
            )

            if abs(depth_error) > configuration.depth_tolerance_mm:
                raise ValueError(
                    f"{row['phase3_event_id']}: "
                    f"production depth error {depth_error:.12g} mm "
                    "exceeds tolerance"
                )

            maximum = float(
                np.max(intensities)
            )

            maximum_times = tuple(
                int(value)
                for value in timeline[
                    intensities == maximum
                ]
            )

            summary = {
                "phase3_sequence": row["phase3_sequence"],
                "phase3_event_id": row["phase3_event_id"],
                "family": row["family"],
                "profile_subtype": row["profile_subtype"],
                "backbone_id": row["backbone_id"],
                "filename": path.name,
                "target_depth_mm": f"{target_depth:.17g}",
                "integrated_depth_mm": f"{integrated_depth:.17g}",
                "depth_error_mm": f"{depth_error:.17g}",
                "maximum_intensity_mm_per_hour": f"{maximum:.17g}",
                "maximum_intensity_times_seconds": ";".join(
                    str(value)
                    for value in maximum_times
                ),
                "citycat_sha256": sha256_file(path),
                "source_manifest_sha256": manifest_sha256,
                "design_qa_status": row["qa_status"],
                "generation_qa_status": "PASS",
                **diagnostics,
            }

            summary_rows.append(summary)

        expected_names = {
            f"Rainfall_Data_{index}.txt"
            for index in range(
                1,
                EXPECTED_TOTAL + 1,
            )
        }

        actual_names = {
            path.name
            for path in staging_txt.glob(
                "Rainfall_Data_*.txt"
            )
        }

        if actual_names != expected_names:
            missing = sorted(
                expected_names - actual_names
            )

            unexpected = sorted(
                actual_names - expected_names
            )

            raise ValueError(
                "generated CityCAT file set is incomplete or unexpected; "
                f"missing={missing}, unexpected={unexpected}"
            )

        if len(summary_rows) != EXPECTED_TOTAL:
            raise ValueError(
                "generation summary does not contain exactly 200 events"
            )

        staging_summary = (
            staging_root / "generation_summary.csv"
        )

        write_summary(
            staging_summary,
            summary_rows,
        )

        if final_txt_directory.exists():
            shutil.rmtree(
                final_txt_directory
            )

        if final_summary.exists():
            final_summary.unlink()

        shutil.move(
            str(staging_txt),
            str(final_txt_directory),
        )

        shutil.move(
            str(staging_summary),
            str(final_summary),
        )

    family_counts = Counter(
        row["family"]
        for row in rows
    )

    print(
        f"Manifest validated: {len(rows)} events"
    )

    print(
        "Families: "
        + ", ".join(
            f"{family}={family_counts[family]}"
            for family in (
                "triangular",
                "gaussian",
                "fsr_feh",
            )
        )
    )

    print(
        f"Manifest SHA-256: {manifest_sha256}"
    )

    print(
        f"CityCAT files: {final_txt_directory}"
    )

    print(
        f"Generation summary: {final_summary}"
    )

    print(
        "Generation QA: PASS"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the 200-event Stockbridge Phase 3 "
            "CityCAT rainfall production dataset."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "Phase 3 master manifest CSV "
            f"(default: {DEFAULT_MANIFEST})"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Dedicated output directory. "
            "A citycat_txt directory and generation_summary.csv "
            "will be created inside it."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing Phase 3 generated dataset "
            "in the selected output directory."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    arguments = parse_arguments()

    generate_dataset(
        arguments.manifest.expanduser().resolve(),
        arguments.output_dir.expanduser().resolve(),
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
