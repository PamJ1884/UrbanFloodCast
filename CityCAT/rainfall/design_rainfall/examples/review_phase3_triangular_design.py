"""Reproducible QA for the Phase 3 triangular rainfall design.

Builds the 40 FEH22 duration/return-period backbone scenarios and one
triangular rainfall profile for each backbone.

Design:
- 40 triangular events
- FEH22 point rainfall depth defines total storm depth
- peak_position_ratio controls temporal peak location
- main peak-position range: 0.30–0.70
- limited extreme peak positions: 0.20 and 0.80
- 5-minute CityCAT rainfall timestep
- 2-hour simulation horizon

Peak-position distribution:
    0.20 -> 2 events
    0.30 -> 7 events
    0.40 -> 7 events
    0.50 -> 8 events
    0.60 -> 7 events
    0.70 -> 7 events
    0.80 -> 2 events

This is a QA/review script, not the final Phase 3 production generator.
Outputs are written to a temporary TEST_phase3_triangular_qa_* directory.
"""

from collections import Counter
from pathlib import Path
import csv
import tempfile

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    TemporalConfiguration,
    TriangularPulse,
    generate_timeline,
    generate_triangular_pulse,
    load_feh_point_ddf_csv,
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
# Phase 3 triangular recipe
#
# One peak_position_ratio per backbone.
# ---------------------------------------------------------------------

peak_design = {
    "B01": 0.30,
    "B02": 0.50,
    "B03": 0.70,
    "B04": 0.40,
    "B05": 0.60,

    "B06": 0.30,
    "B07": 0.70,
    "B08": 0.40,
    "B09": 0.60,

    "B10": 0.50,
    "B11": 0.30,
    "B12": 0.70,
    "B13": 0.40,
    "B14": 0.60,

    "B15": 0.50,
    "B16": 0.30,
    "B17": 0.70,
    "B18": 0.40,
    "B19": 0.60,

    "B20": 0.50,
    "B21": 0.30,
    "B22": 0.70,
    "B23": 0.40,
    "B24": 0.60,

    "B25": 0.50,
    "B26": 0.30,
    "B27": 0.70,
    "B28": 0.40,
    "B29": 0.60,

    # Deliberate extreme peak-position cases.
    # Assigned only to relatively long storms.
    "B30": 0.20,
    "B31": 0.80,

    "B32": 0.50,
    "B33": 0.30,
    "B34": 0.70,
    "B35": 0.40,
    "B36": 0.60,

    # Second pair of deliberately extreme timing cases.
    "B37": 0.20,
    "B38": 0.80,

    # Extreme return periods are kept temporally central.
    "B39": 0.50,
    "B40": 0.50,
}


# ---------------------------------------------------------------------
# Sanity checks on recipe
# ---------------------------------------------------------------------

assert len(backbones) == 40
assert set(peak_design) == {item[0] for item in backbones}

peak_counts = Counter(peak_design.values())

expected_peak_counts = {
    0.20: 2,
    0.30: 7,
    0.40: 7,
    0.50: 8,
    0.60: 7,
    0.70: 7,
    0.80: 2,
}

assert dict(sorted(peak_counts.items())) == expected_peak_counts, peak_counts


# ---------------------------------------------------------------------
# Generate and QA the 40 candidate triangular events
# ---------------------------------------------------------------------

timeline_seconds = np.asarray(
    generate_timeline(configuration),
    dtype=float,
)

timeline_hours = timeline_seconds / 3600.0

rows = []

for rainfall_index, (
    backbone_id,
    duration_min,
    return_period,
) in enumerate(backbones, start=1):

    # FEH22 target storm depth.
    ddf = feh.table.interpolate_duration(
        "STOCKBRIDGE",
        duration_minutes=duration_min,
        return_period_years=return_period,
        method="log_log",
    )

    depth_mm = float(ddf.depth_mm)

    duration_hours = duration_min / 60.0

    mean_intensity_mm_h = depth_mm / duration_hours

    # Continuous triangular-theory peak:
    #
    # area = 0.5 * duration * I_peak
    #
    # therefore:
    #
    # I_peak = 2 * depth / duration
    theoretical_peak_intensity_mm_h = (
        2.0 * depth_mm / duration_hours
    )

    peak_ratio = peak_design[backbone_id]

    requested_peak_time_min = (
        duration_min * peak_ratio
    )

    triangular_id = f"T{rainfall_index:03d}"

    pulse = TriangularPulse(
        start_time_seconds=0,
        duration_seconds=duration_min * 60,
        target_depth_mm=depth_mm,
        peak_position_ratio=peak_ratio,
    )

    event = generate_triangular_pulse(
        configuration,
        pulse,
        rainfall_index=rainfall_index,
        phase_event_id=f"P3_{triangular_id}",
        storm_id=triangular_id,
        role="phase3_candidate",
        profile_family="triangular",
    )

    intensities = np.asarray(
        event.intensities_mm_per_hour,
        dtype=float,
    )

    integrated_depth_mm = float(
        np.trapz(
            intensities,
            timeline_hours,
        )
    )

    depth_error_mm = (
        integrated_depth_mm - depth_mm
    )

    sampled_peak_intensity_mm_h = float(
        np.max(intensities)
    )

    peak_indices = np.flatnonzero(
        np.isclose(
            intensities,
            sampled_peak_intensity_mm_h,
            rtol=0.0,
            atol=1e-12,
        )
    )

    sampled_peak_times_min = tuple(
        float(timeline_seconds[index] / 60.0)
        for index in peak_indices
    )

    nearest_sampled_peak_error_min = min(
        abs(
            sampled_time
            - requested_peak_time_min
        )
        for sampled_time in sampled_peak_times_min
    )

    sampled_vs_theoretical_peak_difference_mm_h = (
        sampled_peak_intensity_mm_h
        - theoretical_peak_intensity_mm_h
    )

    source_durations = ",".join(
        f"{point.duration_minutes:g}"
        for point in ddf.source_points
    )

    rows.append({
        "triangular_id": triangular_id,
        "backbone_id": backbone_id,
        "duration_min": duration_min,
        "return_period_years": return_period,
        "feh_depth_mm": depth_mm,
        "feh_mean_intensity_mm_h": mean_intensity_mm_h,
        "ddf_kind": ddf.estimate_kind,
        "ddf_method": ddf.interpolation_method,
        "ddf_source_durations_min": source_durations,
        "requested_peak_ratio": peak_ratio,
        "requested_peak_time_min": requested_peak_time_min,
        "sampled_peak_times_min": ",".join(
            f"{value:g}"
            for value in sampled_peak_times_min
        ),
        "nearest_sampled_peak_error_min":
            nearest_sampled_peak_error_min,
        "theoretical_peak_intensity_mm_h":
            theoretical_peak_intensity_mm_h,
        "sampled_peak_intensity_mm_h":
            sampled_peak_intensity_mm_h,
        "sampled_minus_theoretical_peak_mm_h":
            sampled_vs_theoretical_peak_difference_mm_h,
        "integrated_depth_mm": integrated_depth_mm,
        "depth_error_mm": depth_error_mm,
    })


assert len(rows) == 40


# ---------------------------------------------------------------------
# Output CSV
# ---------------------------------------------------------------------

output_dir = Path(
    tempfile.mkdtemp(
        prefix="TEST_phase3_triangular_qa_"
    )
)

csv_path = (
    output_dir
    / "TEST_phase3_triangular_40_QA.csv"
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
# Console QA summary
# ---------------------------------------------------------------------

max_depth_error_mm = max(
    abs(row["depth_error_mm"])
    for row in rows
)

largest_peak_timing_error_min = max(
    row["nearest_sampled_peak_error_min"]
    for row in rows
)

minimum_sampled_peak = min(
    row["sampled_peak_intensity_mm_h"]
    for row in rows
)

maximum_sampled_peak = max(
    row["sampled_peak_intensity_mm_h"]
    for row in rows
)


print()
print("=" * 78)
print("PHASE 3 TRIANGULAR QA")
print("=" * 78)

print(f"Output directory : {output_dir}")
print(f"CSV              : {csv_path}")
print(f"Events generated : {len(rows)}")

print()
print("Peak-position distribution:")

for peak_ratio in sorted(peak_counts):
    print(
        f"  {peak_ratio:.2f} : "
        f"{peak_counts[peak_ratio]}"
    )

print()
print(
    "Maximum absolute depth error : "
    f"{max_depth_error_mm:.12g} mm"
)

print()
print("Sampled peak-intensity range:")
print(
    f"  min = {minimum_sampled_peak:.2f} mm/h"
)
print(
    f"  max = {maximum_sampled_peak:.2f} mm/h"
)

print()
print(
    "Largest sampled-peak timing error:"
)
print(
    f"  {largest_peak_timing_error_min:.2f} min"
)


# ---------------------------------------------------------------------
# Highest peak intensities
# ---------------------------------------------------------------------

print()
print("TOP 15 SAMPLED PEAK INTENSITIES")
print(
    "ID   Backbone   D      T      Depth  peak_r"
    "   I_theory   I_sample"
)
print("-" * 78)

for row in sorted(
    rows,
    key=lambda item:
        item["sampled_peak_intensity_mm_h"],
    reverse=True,
)[:15]:

    print(
        f"{row['triangular_id']:<5}"
        f"{row['backbone_id']:<9}"
        f"{row['duration_min']:>4}"
        f"{row['return_period_years']:>7}"
        f"{row['feh_depth_mm']:>10.2f}"
        f"{row['requested_peak_ratio']:>8.2f}"
        f"{row['theoretical_peak_intensity_mm_h']:>11.2f}"
        f"{row['sampled_peak_intensity_mm_h']:>11.2f}"
    )


# ---------------------------------------------------------------------
# Short-duration events
# ---------------------------------------------------------------------

print()
print("SHORT EVENTS (D <= 30 min)")
print(
    "ID   Backbone   D      T      Depth  peak_r"
    "   req_peak   sampled_peak   I_sample"
)
print("-" * 88)

for row in rows:

    if row["duration_min"] <= 30:

        print(
            f"{row['triangular_id']:<5}"
            f"{row['backbone_id']:<9}"
            f"{row['duration_min']:>4}"
            f"{row['return_period_years']:>7}"
            f"{row['feh_depth_mm']:>10.2f}"
            f"{row['requested_peak_ratio']:>8.2f}"
            f"{row['requested_peak_time_min']:>11.2f}"
            f"{row['sampled_peak_times_min']:>15}"
            f"{row['sampled_peak_intensity_mm_h']:>11.2f}"
        )


# ---------------------------------------------------------------------
# Largest peak-timing discrepancies
# ---------------------------------------------------------------------

print()
print("TOP 10 PEAK-TIMING DISCRETISATION ERRORS")
print(
    "ID   Backbone   D   peak_r   requested(min)"
    "   sampled(min)   error(min)"
)
print("-" * 78)

for row in sorted(
    rows,
    key=lambda item:
        item["nearest_sampled_peak_error_min"],
    reverse=True,
)[:10]:

    print(
        f"{row['triangular_id']:<5}"
        f"{row['backbone_id']:<9}"
        f"{row['duration_min']:>4}"
        f"{row['requested_peak_ratio']:>9.2f}"
        f"{row['requested_peak_time_min']:>16.2f}"
        f"{row['sampled_peak_times_min']:>15}"
        f"{row['nearest_sampled_peak_error_min']:>13.2f}"
    )


print()
print("QA complete.")
