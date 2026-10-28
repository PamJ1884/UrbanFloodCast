#!/usr/bin/env python3
"""Generate deterministic TEST visual-QA artifacts for Stage 3A and Stage 3B.

Run from the repository root, for example::

    python3 CityCAT/rainfall/generate_design_rainfall_visual_qa.py \
        --output-dir /path/to/new/output_directory

The output directory must not already exist, preventing silent replacement of
any prior QA artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Support direct execution without an editable installation or PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from CityCAT.rainfall.design_rainfall import (
    CompositeTriangularPulse,
    RainfallEvent,
    TemporalConfiguration,
    TriangularPulse,
    export_citycat_rainfall,
    generate_composite_triangular_pulse,
    generate_timeline,
    generate_triangular_pulse,
    validate_citycat_rainfall,
    validate_rainfall_event,
)


DATA_STATUS = "TEST_SYNTHETIC_SOFTWARE_VISUAL_QA_ONLY"
QA_CONFIGURATION = TemporalConfiguration(
    rainfall_timestep_seconds=300,
    simulation_horizon_seconds=7200,
    maximum_active_duration_seconds=5400,
    depth_tolerance_mm=1e-9,
    citycat_value_tolerance=1e-15,
)
FIXED_PDF_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class VisualQACase:
    """One labelled synthetic event and its plotting annotations."""

    case_id: str
    output_slug: str
    title: str
    event: RainfallEvent
    requested_peaks_seconds: tuple[tuple[str, float], ...]
    dry_gap_bounds_seconds: tuple[int, int] | None = None


@dataclass(frozen=True)
class CaseDiagnostics:
    """Validated numerical QA values for one event."""

    case: VisualQACase
    cumulative_depth_mm: np.ndarray
    integrated_depth_mm: float
    depth_error_mm: float
    sampled_peak_mm_per_hour: float
    sampled_peak_times_seconds: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    """Parse the required artifact output directory."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate TEST synthetic Stage 3A/3B rainfall visual-QA artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory in which all TEST QA artifacts will be created",
    )
    return parser.parse_args()


def cumulative_depth_mm(
    timeline_seconds: np.ndarray,
    intensities_mm_per_hour: np.ndarray,
) -> np.ndarray:
    """Integrate sampled intensities using weighted trapezoidal increments."""

    intervals_seconds = np.diff(timeline_seconds)
    increments_mm = (
        0.5
        * (intensities_mm_per_hour[:-1] + intensities_mm_per_hour[1:])
        * intervals_seconds
        / 3600.0
    )
    return np.concatenate(
        (np.asarray([0.0], dtype=np.float64), np.cumsum(increments_mm))
    )


def _generate_single(
    pulse: TriangularPulse,
    *,
    rainfall_index: int,
    case_id: str,
    profile_family: str,
) -> RainfallEvent:
    return generate_triangular_pulse(
        QA_CONFIGURATION,
        pulse,
        rainfall_index=rainfall_index,
        phase_event_id=f"{case_id}_STAGE3A",
        storm_id=f"{case_id}_STORM",
        role="TEST_SYNTHETIC_VISUAL_QA",
        profile_family=profile_family,
    )


def _generate_composite(
    first_pulse: TriangularPulse,
    second_pulse: TriangularPulse,
    *,
    rainfall_index: int,
    case_id: str,
) -> RainfallEvent:
    return generate_composite_triangular_pulse(
        QA_CONFIGURATION,
        CompositeTriangularPulse(first_pulse, second_pulse),
        rainfall_index=rainfall_index,
        phase_event_id=f"{case_id}_STAGE3B",
        storm_id=f"{case_id}_STORM",
        role="TEST_SYNTHETIC_VISUAL_QA",
        profile_family="TEST_TWO_PULSE_TRIANGULAR",
    )


def generate_cases() -> tuple[VisualQACase, ...]:
    """Generate the four fixed TEST events through the approved public APIs."""

    pulse_a = TriangularPulse(0, 1800, 20.0, 0.5)
    pulse_b = TriangularPulse(600, 1800, 20.0, 0.37)
    first_c = TriangularPulse(0, 1800, 12.0, 0.5)
    second_c = TriangularPulse(3000, 900, 8.0, 0.5)
    first_d = TriangularPulse(0, 1800, 8.0, 0.5)
    second_d = TriangularPulse(3000, 900, 12.0, 0.5)

    return (
        VisualQACase(
            case_id="TEST_A",
            output_slug="TEST_A_single_symmetric",
            title="A. TEST SYNTHETIC single pulse: 20 mm / 30 min",
            event=_generate_single(
                pulse_a,
                rainfall_index=901,
                case_id="TEST_A",
                profile_family="TEST_SINGLE_SYMMETRIC_TRIANGULAR",
            ),
            requested_peaks_seconds=(("pulse", pulse_a.peak_time_seconds),),
        ),
        VisualQACase(
            case_id="TEST_B",
            output_slug="TEST_B_single_asymmetric_delayed",
            title=(
                "B. TEST SYNTHETIC delayed pulse: 20 mm / 30 min; "
                "start 10 min"
            ),
            event=_generate_single(
                pulse_b,
                rainfall_index=902,
                case_id="TEST_B",
                profile_family="TEST_SINGLE_ASYMMETRIC_TRIANGULAR",
            ),
            requested_peaks_seconds=(("pulse", pulse_b.peak_time_seconds),),
        ),
        VisualQACase(
            case_id="TEST_C",
            output_slug="TEST_C_two_pulse_12mm_then_8mm",
            title=(
                "C. TEST SYNTHETIC two pulses: P1 12 mm / 30 min; "
                "P2 8 mm / 15 min"
            ),
            event=_generate_composite(
                first_c,
                second_c,
                rainfall_index=903,
                case_id="TEST_C",
            ),
            requested_peaks_seconds=(
                ("P1", first_c.peak_time_seconds),
                ("P2", second_c.peak_time_seconds),
            ),
            dry_gap_bounds_seconds=(1800, 3000),
        ),
        VisualQACase(
            case_id="TEST_D",
            output_slug="TEST_D_two_pulse_8mm_then_12mm",
            title=(
                "D. TEST SYNTHETIC two pulses: P1 8 mm / 30 min; "
                "P2 12 mm / 15 min"
            ),
            event=_generate_composite(
                first_d,
                second_d,
                rainfall_index=904,
                case_id="TEST_D",
            ),
            requested_peaks_seconds=(
                ("P1", first_d.peak_time_seconds),
                ("P2", second_d.peak_time_seconds),
            ),
            dry_gap_bounds_seconds=(1800, 3000),
        ),
    )


def validate_case(case: VisualQACase) -> CaseDiagnostics:
    """Validate the event, cumulative series, total depth, and dry gap."""

    validate_rainfall_event(case.event, QA_CONFIGURATION)
    timeline = generate_timeline(QA_CONFIGURATION)
    intensities = np.asarray(case.event.intensities_mm_per_hour, dtype=np.float64)
    cumulative = cumulative_depth_mm(timeline, intensities)
    integrated_depth = float(cumulative[-1])
    depth_error = integrated_depth - case.event.target_depth_mm
    if not math.isclose(
        integrated_depth,
        20.0,
        rel_tol=0.0,
        abs_tol=QA_CONFIGURATION.depth_tolerance_mm,
    ):
        raise ValueError(
            f"{case.case_id} integrated depth {integrated_depth} is not 20 mm"
        )
    if np.any(np.diff(cumulative) < 0.0):
        raise ValueError(f"{case.case_id} cumulative depth is not non-decreasing")
    if not math.isclose(
        integrated_depth,
        float(
            np.trapz(
                intensities,
                timeline / 3600.0,
            )
        ),
        rel_tol=0.0,
        abs_tol=QA_CONFIGURATION.depth_tolerance_mm,
    ):
        raise ValueError(f"{case.case_id} cumulative endpoint is inconsistent")
    if case.dry_gap_bounds_seconds is not None:
        gap_start, gap_end = case.dry_gap_bounds_seconds
        dry_gap = (timeline >= gap_start) & (timeline <= gap_end)
        if np.any(intensities[dry_gap] != 0.0):
            raise ValueError(f"{case.case_id} dry gap is not exactly zero")

    sampled_peak = float(np.max(intensities))
    sampled_peak_times = tuple(
        int(value) for value in timeline[intensities == sampled_peak]
    )
    return CaseDiagnostics(
        case=case,
        cumulative_depth_mm=cumulative,
        integrated_depth_mm=integrated_depth,
        depth_error_mm=depth_error,
        sampled_peak_mm_per_hour=sampled_peak,
        sampled_peak_times_seconds=sampled_peak_times,
    )


def json_compatible(value: Any) -> Any:
    """Recursively convert event metadata without discarding nested fields."""

    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON metadata values must be finite")
        return value
    raise TypeError(f"cannot convert metadata value to JSON: {type(value).__name__}")


def _write_timeseries(
    path: Path,
    diagnostics: CaseDiagnostics,
) -> None:
    timeline = generate_timeline(QA_CONFIGURATION)
    intensities = diagnostics.case.event.intensities_mm_per_hour
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "time_seconds",
                "time_minutes",
                "intensity_mm_per_hour",
                "cumulative_depth_mm",
            )
        )
        for seconds, intensity, cumulative in zip(
            timeline,
            intensities,
            diagnostics.cumulative_depth_mm,
        ):
            writer.writerow(
                (
                    int(seconds),
                    format(float(seconds) / 60.0, ".17g"),
                    format(float(intensity), ".17g"),
                    format(float(cumulative), ".17g"),
                )
            )


def _event_metadata_payload(diagnostics: CaseDiagnostics) -> dict[str, Any]:
    event = diagnostics.case.event
    return {
        "data_status": DATA_STATUS,
        "case_id": diagnostics.case.case_id,
        "case_title": diagnostics.case.title,
        "configuration": {
            "rainfall_timestep_seconds": QA_CONFIGURATION.rainfall_timestep_seconds,
            "simulation_horizon_seconds": QA_CONFIGURATION.simulation_horizon_seconds,
            "maximum_active_duration_seconds": (
                QA_CONFIGURATION.maximum_active_duration_seconds
            ),
            "depth_tolerance_mm": QA_CONFIGURATION.depth_tolerance_mm,
            "citycat_value_tolerance": QA_CONFIGURATION.citycat_value_tolerance,
        },
        "event": {
            "rainfall_index": event.rainfall_index,
            "phase_event_id": event.phase_event_id,
            "storm_id": event.storm_id,
            "role": event.role,
            "profile_family": event.profile_family,
            "target_depth_mm": event.target_depth_mm,
            "active_duration_seconds": event.active_duration_seconds,
            "peak_metadata": json_compatible(event.peak_metadata),
        },
    }


def _write_metadata(path: Path, diagnostics: CaseDiagnostics) -> None:
    payload = _event_metadata_payload(diagnostics)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"JSON metadata did not round-trip exactly: {path}")


def _write_summary(path: Path, diagnostics: tuple[CaseDiagnostics, ...]) -> None:
    fields = (
        "data_status",
        "case_id",
        "case_title",
        "target_depth_mm",
        "integrated_depth_mm",
        "depth_error_mm",
        "sampled_peak_mm_per_hour",
        "sampled_peak_times_seconds",
        "active_endpoint_seconds",
        "dry_gap_duration_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in diagnostics:
            gap = result.case.dry_gap_bounds_seconds
            writer.writerow(
                {
                    "data_status": DATA_STATUS,
                    "case_id": result.case.case_id,
                    "case_title": result.case.title,
                    "target_depth_mm": format(
                        result.case.event.target_depth_mm, ".17g"
                    ),
                    "integrated_depth_mm": format(
                        result.integrated_depth_mm, ".17g"
                    ),
                    "depth_error_mm": format(result.depth_error_mm, ".17g"),
                    "sampled_peak_mm_per_hour": format(
                        result.sampled_peak_mm_per_hour, ".17g"
                    ),
                    "sampled_peak_times_seconds": json.dumps(
                        result.sampled_peak_times_seconds
                    ),
                    "active_endpoint_seconds": (
                        result.case.event.active_duration_seconds
                    ),
                    "dry_gap_duration_seconds": (
                        "" if gap is None else gap[1] - gap[0]
                    ),
                }
            )


def _requested_peak_label(label: str, seconds: float) -> str:
    timestep = QA_CONFIGURATION.rainfall_timestep_seconds
    is_sampled = math.isclose(
        seconds / timestep,
        round(seconds / timestep),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    grid_note = "" if is_sampled else " (off-grid)"
    return f"Requested peak {label}: {seconds / 60.0:g} min{grid_note}"


def create_overview_figure(
    diagnostics: tuple[CaseDiagnostics, ...],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot sampled intensity and trapezoidal cumulative depth for all cases."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timeline_minutes = generate_timeline(QA_CONFIGURATION) / 60.0
    common_intensity_max = max(
        result.sampled_peak_mm_per_hour for result in diagnostics
    )
    common_depth_max = max(
        result.integrated_depth_mm for result in diagnostics
    )
    peak_colours = ("tab:red", "tab:purple")

    with plt.rc_context({"font.size": 9.0, "axes.titleweight": "bold"}):
        figure, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=True)
        for row, result in enumerate(diagnostics):
            intensity_axis, cumulative_axis = axes[row]
            event = result.case.event
            intensity_axis.plot(
                timeline_minutes,
                event.intensities_mm_per_hour,
                color="tab:blue",
                marker="o",
                markersize=4,
                linewidth=1.7,
                label="Sampled intensity (straight connections)",
            )
            cumulative_axis.plot(
                timeline_minutes,
                result.cumulative_depth_mm,
                color="tab:green",
                marker="o",
                markersize=4,
                linewidth=1.7,
                label="Trapezoidal cumulative depth at samples",
            )

            for peak_number, (label, seconds) in enumerate(
                result.case.requested_peaks_seconds
            ):
                intensity_axis.axvline(
                    seconds / 60.0,
                    color=peak_colours[peak_number],
                    linestyle="--",
                    linewidth=1.4,
                    label=_requested_peak_label(label, seconds),
                )

            if result.case.dry_gap_bounds_seconds is not None:
                gap_start, gap_end = result.case.dry_gap_bounds_seconds
                for axis in (intensity_axis, cumulative_axis):
                    axis.axvspan(
                        gap_start / 60.0,
                        gap_end / 60.0,
                        color="0.5",
                        alpha=0.16,
                        label="Dry gap: 30-50 min",
                    )

            intensity_axis.set_title(
                f"{result.case.title}\n"
                f"Target {event.target_depth_mm:.3f} mm | "
                f"integrated {result.integrated_depth_mm:.3f} mm | "
                f"sampled max {result.sampled_peak_mm_per_hour:.3f} mm/h"
            )
            cumulative_axis.set_title(
                f"{result.case.case_id} TEST SYNTHETIC cumulative rainfall\n"
                "Straight lines are visual guides between sampled values"
            )
            intensity_axis.set_ylabel("Sampled intensity (mm/h)")
            cumulative_axis.set_ylabel("Cumulative depth (mm)")
            for axis in (intensity_axis, cumulative_axis):
                axis.set_xlim(0.0, 120.0)
                axis.set_xticks(np.arange(0.0, 121.0, 20.0))
                axis.set_xlabel("Time (minutes)")
                axis.tick_params(labelbottom=True)
                axis.grid(True, linewidth=0.5, alpha=0.35)
                axis.legend(loc="best", fontsize=8, framealpha=0.9)
            intensity_axis.set_ylim(0.0, common_intensity_max * 1.15)
            cumulative_axis.set_ylim(0.0, common_depth_max * 1.08)

        figure.suptitle(
            "TEST SYNTHETIC Stage 3A and Stage 3B rainfall visual QA",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.006,
            (
                "All curves connect sampled values with straight visual guides; "
                "cumulative depth uses weighted trapezoidal increments."
            ),
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.975))
        figure.savefig(
            png_path,
            dpi=300,
            metadata={
                "Title": "TEST SYNTHETIC Stage 3A/3B rainfall visual QA",
                "Description": DATA_STATUS,
                "Software": "UrbanFloodCast design-rainfall visual QA",
            },
        )
        figure.savefig(
            pdf_path,
            metadata={
                "Title": "TEST SYNTHETIC Stage 3A/3B rainfall visual QA",
                "Author": "UrbanFloodCast TEST synthetic QA",
                "Subject": DATA_STATUS,
                "Creator": "UrbanFloodCast design-rainfall visual QA",
                "CreationDate": FIXED_PDF_DATE,
                "ModDate": FIXED_PDF_DATE,
            },
        )
        plt.close(figure)


def write_case_artifacts(
    output_directory: Path,
    diagnostics: CaseDiagnostics,
) -> tuple[Path, Path, Path]:
    """Write one case's CSV, metadata, and validated CityCAT file."""

    case_directory = output_directory / diagnostics.case.output_slug
    case_directory.mkdir()
    timeseries_path = case_directory / "TEST_timeseries.csv"
    metadata_path = case_directory / "TEST_event_metadata.json"
    _write_timeseries(timeseries_path, diagnostics)
    _write_metadata(metadata_path, diagnostics)
    citycat_path = export_citycat_rainfall(
        case_directory / "citycat",
        diagnostics.case.event,
        QA_CONFIGURATION,
    )
    validate_citycat_rainfall(
        citycat_path,
        diagnostics.case.event,
        QA_CONFIGURATION,
    )
    return timeseries_path, metadata_path, citycat_path


def run_visual_qa(output_directory: Path) -> tuple[CaseDiagnostics, ...]:
    """Generate, validate, export, and plot the complete fixed QA demonstration."""

    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {output_directory}"
        )
    output_directory.mkdir(parents=True)

    cases = generate_cases()
    diagnostics = tuple(validate_case(case) for case in cases)
    for result in diagnostics:
        write_case_artifacts(output_directory, result)

    summary_path = output_directory / "TEST_SYNTHETIC_visual_qa_summary.csv"
    png_path = output_directory / "TEST_SYNTHETIC_visual_qa_overview.png"
    pdf_path = output_directory / "TEST_SYNTHETIC_visual_qa_overview.pdf"
    _write_summary(summary_path, diagnostics)
    create_overview_figure(diagnostics, png_path, pdf_path)

    print(f"TEST synthetic visual QA written to: {output_directory}")
    for result in diagnostics:
        print(
            f"{result.case.case_id}: target={result.case.event.target_depth_mm:.12g} "
            f"mm, integrated={result.integrated_depth_mm:.12g} mm, "
            f"error={result.depth_error_mm:.3g} mm, "
            f"sampled_peak={result.sampled_peak_mm_per_hour:.12g} mm/h, "
            f"peak_times_s={result.sampled_peak_times_seconds}, "
            f"endpoint_s={result.case.event.active_duration_seconds}"
        )
    print(f"Overview PNG: {png_path}")
    print(f"Overview PDF: {pdf_path}")
    print(f"Summary CSV: {summary_path}")
    return diagnostics


def main() -> None:
    args = parse_args()
    run_visual_qa(args.output_dir)


if __name__ == "__main__":
    main()
