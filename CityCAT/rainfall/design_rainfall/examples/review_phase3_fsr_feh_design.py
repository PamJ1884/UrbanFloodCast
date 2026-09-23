"""Reproducible QA for the Phase 3 FSR/FEH rainfall design.

Builds the 40 FEH22 duration/return-period backbone scenarios and two
FSR/FEH temporal profiles for every backbone:

    - Summer
    - Winter

Total:
    40 backbones x 2 seasonal profiles = 80 candidate events

The total storm depth comes from the FEH22 point DDF table for
Stockbridge. The FSR/FEH profile controls the temporal distribution.

The published/reference FSR/FEH temporal profile is first constructed
using build_fsr_feh_reference(). It is then converted to the 5-minute
CityCAT nodal representation using the explicitly experimental
adapt_fsr_feh_reference_experimental() adapter.

The adapter:
- preserves final total storm depth through global normalization;
- does not preserve every individual FSR/FEH interval depth exactly;
- therefore introduces temporal-shape discretisation error;
- is expected to show the largest relative errors for short storms.

This script quantifies that error rather than hiding it.

This is a QA/review script, not the final Phase 3 production generator.
Outputs are written to a temporary TEST_phase3_fsr_feh_qa_* directory.
"""

from pathlib import Path
import csv
import tempfile

from CityCAT.rainfall.design_rainfall import (
    TemporalConfiguration,
    build_fsr_feh_reference,
    load_feh_point_ddf_csv,
)

from CityCAT.rainfall.design_rainfall.experimental_fsr_feh import (
    adapt_fsr_feh_reference_experimental,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

configuration = TemporalConfiguration(
    rainfall_timestep_seconds=300,
    simulation_horizon_seconds=7200,
    maximum_active_duration_seconds=7200,
    depth_tolerance_mm=1e-6,
)

feh_path = (
    Path.home()
    / "flood/02_data_raw/DDFs/FEH_Point_Rainfall_FEH22_AM_447157_129992.csv"
)

feh = load_feh_point_ddf_csv(
    feh_path,
    location_id="STOCKBRIDGE",
)


# ---------------------------------------------------------------------
# 40 FEH22 backbone storms
#
# (backbone_id, duration_minutes, return_period_years)
# ---------------------------------------------------------------------

backbones = [
    ("B01", 15, 2),
    ("B02", 15, 10),
    ("B03", 15, 30),
    ("B04", 15, 200),
    ("B05", 15, 500),
    ("B06", 20, 75),
    ("B07", 25, 5),
    ("B08", 30, 20),
    ("B09", 30, 150),
    ("B10", 35, 5),
    ("B11", 35, 50),
    ("B12", 35, 100),
    ("B13", 40, 20),
    ("B14", 40, 200),
    ("B15", 45, 50),
    ("B16", 45, 100),
    ("B17", 50, 30),
    ("B18", 50, 500),
    ("B19", 55, 2),
    ("B20", 55, 10),
    ("B21", 60, 75),
    ("B22", 60, 200),
    ("B23", 65, 30),
    ("B24", 70, 10),
    ("B25", 70, 50),
    ("B26", 70, 100),
    ("B27", 70, 500),
    ("B28", 75, 20),
    ("B29", 75, 150),
    ("B30", 80, 5),
    ("B31", 80, 50),
    ("B32", 80, 100),
    ("B33", 85, 150),
    ("B34", 90, 2),
    ("B35", 90, 10),
    ("B36", 90, 30),
    ("B37", 90, 75),
    ("B38", 90, 500),
    ("B39", 30, 1000),
    ("B40", 90, 10000),
]


# ---------------------------------------------------------------------
# FSR/FEH recipe
#
# Each backbone generates exactly:
#   1 Summer profile
#   1 Winter profile
# ---------------------------------------------------------------------

profile_names = (
    "summer",
    "winter",
)

assert len(backbones) == 40
assert len(profile_names) == 2


# ---------------------------------------------------------------------
# Generate reference profiles, adapt them, and perform QA
# ---------------------------------------------------------------------

rows = []

rainfall_index = 1

for backbone_id, duration_min, return_period in backbones:

    # -------------------------------------------------------------
    # FEH22 target depth
    # -------------------------------------------------------------

    ddf = feh.table.interpolate_duration(
        "STOCKBRIDGE",
        duration_minutes=duration_min,
        return_period_years=return_period,
        method="log_log",
    )

    depth_mm = float(ddf.depth_mm)

    mean_intensity_mm_h = (
        depth_mm / (duration_min / 60.0)
    )

    source_durations = ",".join(
        f"{point.duration_minutes:g}"
        for point in ddf.source_points
    )

    # -------------------------------------------------------------
    # Summer + Winter
    # -------------------------------------------------------------

    for profile_name in profile_names:

        fsr_id = f"F{rainfall_index:03d}"

        reference = build_fsr_feh_reference(
            configuration,
            profile_name=profile_name,
            start_time_seconds=0,
            duration_seconds=duration_min * 60,
            target_depth_mm=depth_mm,
        )

        adaptation = adapt_fsr_feh_reference_experimental(
            reference,
            rainfall_index=rainfall_index,
            phase_event_id=f"P3_{fsr_id}",
            storm_id=fsr_id,
            role="phase3_candidate",
            profile_family=f"fsr_feh_{profile_name}",
        )

        final_depth_mm = float(
            adaptation.final_total_depth_mm
        )

        depth_error_mm = (
            final_depth_mm - depth_mm
        )

        maximum_cumulative_error_mm = float(
            adaptation.maximum_absolute_cumulative_depth_error_mm
        )

        maximum_cumulative_error_fraction = float(
            adaptation.maximum_absolute_cumulative_depth_error_fraction
        )

        maximum_cumulative_error_percent = (
            100.0 * maximum_cumulative_error_fraction
        )

        adapted_peak_times_min = ",".join(
            f"{seconds / 60.0:g}"
            for seconds
            in adaptation.adapted_maximum_nodal_times_seconds
        )

        rows.append({
            "fsr_feh_id": fsr_id,
            "backbone_id": backbone_id,
            "profile_name": profile_name,
            "duration_min": duration_min,
            "return_period_years": return_period,
            "feh_depth_mm": depth_mm,
            "feh_mean_intensity_mm_h": mean_intensity_mm_h,
            "ddf_kind": ddf.estimate_kind,
            "ddf_method": ddf.interpolation_method,
            "ddf_source_durations_min": source_durations,
            "parameter_a": reference.parameter_a,
            "parameter_b": reference.parameter_b,
            "source_reference": reference.source_reference,
            "reference_interval_count":
                len(reference.interval_depths_mm),
            "reference_maximum_interval_average_intensity_mm_h":
                adaptation.reference_maximum_interval_average_intensity_mm_per_hour,
            "depth_before_normalization_mm":
                adaptation.depth_before_normalization_mm,
            "normalization_factor":
                adaptation.normalization_factor,
            "adapted_maximum_nodal_intensity_mm_h":
                adaptation.adapted_maximum_nodal_intensity_mm_per_hour,
            "adapted_maximum_nodal_times_min":
                adapted_peak_times_min,
            "final_total_depth_mm":
                final_depth_mm,
            "depth_error_mm":
                depth_error_mm,
            "maximum_absolute_cumulative_depth_error_mm":
                maximum_cumulative_error_mm,
            "maximum_absolute_cumulative_depth_error_fraction":
                maximum_cumulative_error_fraction,
            "maximum_absolute_cumulative_depth_error_percent":
                maximum_cumulative_error_percent,
            "cumulative_error_evaluation":
                adaptation.cumulative_error_evaluation,
        })

        rainfall_index += 1


assert len(rows) == 80

assert sum(
    row["profile_name"] == "summer"
    for row in rows
) == 40

assert sum(
    row["profile_name"] == "winter"
    for row in rows
) == 40


# ---------------------------------------------------------------------
# Output CSV
# ---------------------------------------------------------------------

output_dir = Path(
    tempfile.mkdtemp(
        prefix="TEST_phase3_fsr_feh_qa_"
    )
)

csv_path = (
    output_dir
    / "TEST_phase3_fsr_feh_80_QA.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

maximum_depth_error_mm = max(
    abs(row["depth_error_mm"])
    for row in rows
)

minimum_peak_intensity = min(
    row["adapted_maximum_nodal_intensity_mm_h"]
    for row in rows
)

maximum_peak_intensity = max(
    row["adapted_maximum_nodal_intensity_mm_h"]
    for row in rows
)

maximum_cumulative_error_mm = max(
    row["maximum_absolute_cumulative_depth_error_mm"]
    for row in rows
)

maximum_cumulative_error_percent = max(
    row["maximum_absolute_cumulative_depth_error_percent"]
    for row in rows
)


# ---------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("PHASE 3 FSR/FEH QA")
print("=" * 78)

print(f"Output directory : {output_dir}")
print(f"CSV              : {csv_path}")
print(f"Events generated : {len(rows)}")
print("Profiles         : 40 Summer + 40 Winter")

print()
print(
    "Maximum absolute final-depth error : "
    f"{maximum_depth_error_mm:.12g} mm"
)

print()
print("Adapted peak-intensity range:")
print(
    f"  min = {minimum_peak_intensity:.2f} mm/h"
)
print(
    f"  max = {maximum_peak_intensity:.2f} mm/h"
)

print()
print("Maximum cumulative-profile discrepancy:")
print(
    f"  absolute = {maximum_cumulative_error_mm:.3f} mm"
)
print(
    f"  relative = {maximum_cumulative_error_percent:.2f} %"
)


# ---------------------------------------------------------------------
# Largest cumulative-profile errors
# ---------------------------------------------------------------------

print()
print("TOP 20 CUMULATIVE-PROFILE DISCRETISATION ERRORS")
print(
    "ID   Backbone Profile   D      T      Depth"
    "   max_err(mm)   max_err(%)"
)
print("-" * 86)

for row in sorted(
    rows,
    key=lambda item:
        item[
            "maximum_absolute_cumulative_depth_error_fraction"
        ],
    reverse=True,
)[:20]:

    print(
        f"{row['fsr_feh_id']:<5}"
        f"{row['backbone_id']:<9}"
        f"{row['profile_name']:<9}"
        f"{row['duration_min']:>4}"
        f"{row['return_period_years']:>7}"
        f"{row['feh_depth_mm']:>10.2f}"
        f"{row['maximum_absolute_cumulative_depth_error_mm']:>14.3f}"
        f"{row['maximum_absolute_cumulative_depth_error_percent']:>13.2f}"
    )


# ---------------------------------------------------------------------
# Short-duration storms
# ---------------------------------------------------------------------

print()
print("SHORT EVENTS (D <= 30 min)")
print(
    "ID   Backbone Profile   D      T      Depth"
    "   peak_I   error(mm)   error(%)"
)
print("-" * 94)

for row in rows:

    if row["duration_min"] <= 30:

        print(
            f"{row['fsr_feh_id']:<5}"
            f"{row['backbone_id']:<9}"
            f"{row['profile_name']:<9}"
            f"{row['duration_min']:>4}"
            f"{row['return_period_years']:>7}"
            f"{row['feh_depth_mm']:>10.2f}"
            f"{row['adapted_maximum_nodal_intensity_mm_h']:>10.2f}"
            f"{row['maximum_absolute_cumulative_depth_error_mm']:>12.3f}"
            f"{row['maximum_absolute_cumulative_depth_error_percent']:>11.2f}"
        )


# ---------------------------------------------------------------------
# Highest adapted peak intensities
# ---------------------------------------------------------------------

print()
print("TOP 15 ADAPTED PEAK INTENSITIES")
print(
    "ID   Backbone Profile   D      T      Depth"
    "   ref_peak   adapted_peak   peak_time"
)
print("-" * 100)

for row in sorted(
    rows,
    key=lambda item:
        item[
            "adapted_maximum_nodal_intensity_mm_h"
        ],
    reverse=True,
)[:15]:

    print(
        f"{row['fsr_feh_id']:<5}"
        f"{row['backbone_id']:<9}"
        f"{row['profile_name']:<9}"
        f"{row['duration_min']:>4}"
        f"{row['return_period_years']:>7}"
        f"{row['feh_depth_mm']:>10.2f}"
        f"{row['reference_maximum_interval_average_intensity_mm_h']:>11.2f}"
        f"{row['adapted_maximum_nodal_intensity_mm_h']:>15.2f}"
        f"{row['adapted_maximum_nodal_times_min']:>12}"
    )


# ---------------------------------------------------------------------
# Mean cumulative error by duration and profile
# ---------------------------------------------------------------------

print()
print("CUMULATIVE ERROR BY DURATION / PROFILE")
print(
    "D(min)   profile    n    mean_error(%)    max_error(%)"
)
print("-" * 64)

durations = sorted(
    set(row["duration_min"] for row in rows)
)

for duration_min in durations:

    for profile_name in profile_names:

        subset = [
            row
            for row in rows
            if (
                row["duration_min"] == duration_min
                and row["profile_name"] == profile_name
            )
        ]

        if not subset:
            continue

        errors = [
            row[
                "maximum_absolute_cumulative_depth_error_percent"
            ]
            for row in subset
        ]

        mean_error = sum(errors) / len(errors)

        print(
            f"{duration_min:>6}"
            f"{profile_name:>10}"
            f"{len(subset):>6}"
            f"{mean_error:>17.2f}"
            f"{max(errors):>16.2f}"
        )


print()
print("QA complete.")
