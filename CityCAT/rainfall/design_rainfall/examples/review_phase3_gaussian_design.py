"""Reproducible QA for the Phase 3 Gaussian rainfall design.

Builds the 40 FEH22 duration/return-period backbone scenarios and the
80 proposed Gaussian variants (30 left-skewed, 30 right-skewed,
20 symmetric/central).

Uses:
- FEH22 point rainfall depths for Stockbridge
- duration interpolation with log-log DDF interpolation where required
- GaussianPulse / generate_gaussian_pulse from design_rainfall
- fixed scale_ratio=0.18
- fixed edge_taper_ratio=0.12
- 5-minute CityCAT rainfall timestep
- 2-hour simulation horizon

This is a QA/review script, not the final Phase 3 production generator.
Outputs are written to a temporary TEST_phase3_gaussian_qa_* directory.
"""


from pathlib import Path
import csv
import tempfile

import numpy as np

from CityCAT.rainfall.design_rainfall import (
    GaussianPulse,
    TemporalConfiguration,
    generate_gaussian_pulse,
    generate_timeline,
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

scale_ratio = 0.18
edge_taper_ratio = 0.12

feh_path = (
    Path.home()
    / "flood/02_data_raw/DDFs/FEH_Point_Rainfall_FEH22_AM_447157_129992.csv"
)

feh = load_feh_point_ddf_csv(
    feh_path,
    location_id="STOCKBRIDGE",
)

# ---------------------------------------------------------------------
# 40 backbone storms
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
# Two Gaussian variants for every backbone
#
# (shape, alpha, peak_position_ratio)
# ---------------------------------------------------------------------

design = {
    "B01": [("L", -1.5, 0.30), ("C",  0.0, 0.30)],
    "B02": [("L", -1.5, 0.30), ("R", +1.5, 0.70)],
    "B03": [("R", +1.5, 0.70), ("C",  0.0, 0.70)],
    "B04": [("L", -3.0, 0.35), ("R", +3.0, 0.65)],
    "B05": [("L", -3.0, 0.70), ("C",  0.0, 0.70)],
    "B06": [("L", -5.0, 0.40), ("R", +5.0, 0.60)],
    "B07": [("R", +3.0, 0.30), ("C",  0.0, 0.30)],
    "B08": [("L", -5.0, 0.35), ("C",  0.0, 0.35)],
    "B09": [("L", -1.5, 0.45), ("R", +1.5, 0.55)],
    "B10": [("R", +5.0, 0.65), ("C",  0.0, 0.65)],
    "B11": [("L", -1.5, 0.20), ("R", +1.5, 0.80)],
    "B12": [("L", -1.5, 0.65), ("C",  0.0, 0.65)],
    "B13": [("L", -3.0, 0.50), ("R", +3.0, 0.50)],
    "B14": [("R", +1.5, 0.35), ("C",  0.0, 0.35)],
    "B15": [("L", -5.0, 0.30), ("R", +5.0, 0.70)],
    "B16": [("L", -3.0, 0.40), ("C",  0.0, 0.40)],
    "B17": [("L", -1.5, 0.35), ("R", +1.5, 0.65)],
    "B18": [("R", +3.0, 0.60), ("C",  0.0, 0.60)],
    "B19": [("L", -3.0, 0.40), ("R", +3.0, 0.60)],
    "B20": [("L", -5.0, 0.60), ("C",  0.0, 0.60)],
    "B21": [("L", -5.0, 0.45), ("R", +5.0, 0.55)],
    "B22": [("R", +5.0, 0.40), ("C",  0.0, 0.40)],
    "B23": [("L", -1.5, 0.50), ("R", +1.5, 0.50)],
    "B24": [("L", -1.5, 0.45), ("C",  0.0, 0.45)],
    "B25": [("L", -3.0, 0.30), ("R", +3.0, 0.70)],
    "B26": [("R", +1.5, 0.55), ("C",  0.0, 0.55)],
    "B27": [("L", -5.0, 0.35), ("R", +5.0, 0.65)],
    "B28": [("L", -3.0, 0.55), ("C",  0.0, 0.55)],
    "B29": [("L", -1.5, 0.40), ("R", +1.5, 0.60)],
    "B30": [("R", +3.0, 0.45), ("C",  0.0, 0.45)],
    "B31": [("L", -3.0, 0.20), ("R", +3.0, 0.80)],
    "B32": [("L", -5.0, 0.50), ("C",  0.0, 0.50)],
    "B33": [("L", -5.0, 0.45), ("R", +5.0, 0.55)],
    "B34": [("R", +5.0, 0.50), ("C",  0.0, 0.50)],
    "B35": [("L", -1.5, 0.50), ("R", +1.5, 0.50)],
    "B36": [("L", -3.0, 0.50), ("C",  0.0, 0.50)],
    "B37": [("L", -3.0, 0.30), ("R", +3.0, 0.70)],
    "B38": [("R", +3.0, 0.50), ("C",  0.0, 0.50)],
    "B39": [("L", -3.0, 0.40), ("R", +3.0, 0.60)],
    "B40": [("L", -1.5, 0.45), ("R", +1.5, 0.55)],
}

# ---------------------------------------------------------------------
# Sanity checks on design
# ---------------------------------------------------------------------

assert len(backbones) == 40
assert set(design) == {b[0] for b in backbones}
assert all(len(design[b]) == 2 for b in design)

shape_counts = {"L": 0, "R": 0, "C": 0}
for variants in design.values():
    for shape, _, _ in variants:
        shape_counts[shape] += 1

assert shape_counts == {"L": 30, "R": 30, "C": 20}, shape_counts

# ---------------------------------------------------------------------
# Generate and QA all 80 events
# ---------------------------------------------------------------------

timeline_seconds = np.asarray(generate_timeline(configuration), dtype=float)
timeline_hours = timeline_seconds / 3600.0

rows = []
gaussian_index = 1

for backbone_id, duration_min, return_period in backbones:

    ddf = feh.table.interpolate_duration(
        "STOCKBRIDGE",
        duration_minutes=duration_min,
        return_period_years=return_period,
        method="log_log",
    )

    depth_mm = float(ddf.depth_mm)
    mean_intensity = depth_mm / (duration_min / 60.0)

    source_durations = ",".join(
        f"{point.duration_minutes:g}"
        for point in ddf.source_points
    )

    for shape, alpha, peak_ratio in design[backbone_id]:

        gaussian_id = f"G{gaussian_index:03d}"

        pulse = GaussianPulse(
            start_time_seconds=0,
            duration_seconds=duration_min * 60,
            target_depth_mm=depth_mm,
            peak_position_ratio=peak_ratio,
            scale_ratio=scale_ratio,
            skewness_parameter=alpha,
            edge_taper_ratio=edge_taper_ratio,
        )

        event = generate_gaussian_pulse(
            configuration,
            pulse,
            rainfall_index=gaussian_index,
            phase_event_id=f"P3_{gaussian_id}",
            storm_id=gaussian_id,
            role="phase3_candidate",
            profile_family="gaussian",
        )

        intensities = np.asarray(
            event.intensities_mm_per_hour,
            dtype=float,
        )

        integrated_depth = float(
            np.trapz(intensities, timeline_hours)
        )

        depth_error = integrated_depth - depth_mm

        maximum = float(np.max(intensities))

        max_indices = np.flatnonzero(intensities == maximum)
        sampled_peak_times_min = tuple(
            float(timeline_seconds[i] / 60.0)
            for i in max_indices
        )

        requested_peak_time_min = (
            duration_min * peak_ratio
        )

        nearest_sampled_peak_error_min = min(
            abs(value - requested_peak_time_min)
            for value in sampled_peak_times_min
        )

        metadata = dict(event.peak_metadata)

        normalization_factor = None
        for key, value in metadata.items():
            key_lower = str(key).lower()
            if (
                "normal" in key_lower
                and "factor" in key_lower
            ):
                try:
                    normalization_factor = float(value)
                except (TypeError, ValueError):
                    pass

        rows.append({
            "gaussian_id": gaussian_id,
            "backbone_id": backbone_id,
            "shape": shape,
            "duration_min": duration_min,
            "return_period_years": return_period,
            "feh_depth_mm": depth_mm,
            "feh_mean_intensity_mm_h": mean_intensity,
            "ddf_kind": ddf.estimate_kind,
            "ddf_method": ddf.interpolation_method,
            "ddf_source_durations_min": source_durations,
            "alpha": alpha,
            "requested_peak_ratio": peak_ratio,
            "requested_peak_time_min": requested_peak_time_min,
            "sampled_peak_times_min": ",".join(
                f"{value:g}"
                for value in sampled_peak_times_min
            ),
            "nearest_sampled_peak_error_min":
                nearest_sampled_peak_error_min,
            "scale_ratio": scale_ratio,
            "edge_taper_ratio": edge_taper_ratio,
            "peak_intensity_mm_h": maximum,
            "integrated_depth_mm": integrated_depth,
            "depth_error_mm": depth_error,
            "normalization_factor": normalization_factor,
        })

        gaussian_index += 1

assert len(rows) == 80

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

output_dir = Path(
    tempfile.mkdtemp(prefix="TEST_phase3_gaussian_qa_")
)

csv_path = output_dir / "TEST_phase3_gaussian_80_QA.csv"

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
# Console summary
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("PHASE 3 GAUSSIAN QA")
print("=" * 78)

print(f"Output directory : {output_dir}")
print(f"CSV              : {csv_path}")
print(f"Events generated : {len(rows)}")
print(f"Shape counts     : {shape_counts}")

max_depth_error = max(
    abs(row["depth_error_mm"])
    for row in rows
)

print(
    f"Maximum absolute depth error : "
    f"{max_depth_error:.12g} mm"
)

print()
print("Peak-intensity range:")
print(
    f"  min = {min(r['peak_intensity_mm_h'] for r in rows):.2f} mm/h"
)
print(
    f"  max = {max(r['peak_intensity_mm_h'] for r in rows):.2f} mm/h"
)

print()
print("Largest sampled-peak timing error:")
print(
    f"  {max(r['nearest_sampled_peak_error_min'] for r in rows):.2f} min"
)

print()
print("TOP 15 PEAK INTENSITIES")
print(
    "ID   Backbone Shape   D   T       Depth   alpha   "
    "peak_r   peak_I(mm/h)"
)
print("-" * 78)

for row in sorted(
    rows,
    key=lambda r: r["peak_intensity_mm_h"],
    reverse=True,
)[:15]:

    print(
        f"{row['gaussian_id']:<5}"
        f"{row['backbone_id']:<9}"
        f"{row['shape']:^6}"
        f"{row['duration_min']:>4}"
        f"{row['return_period_years']:>7}"
        f"{row['feh_depth_mm']:>9.2f}"
        f"{row['alpha']:>8.1f}"
        f"{row['requested_peak_ratio']:>9.2f}"
        f"{row['peak_intensity_mm_h']:>13.2f}"
    )

print()
print("SHORT EVENTS (D <= 30 min)")
print(
    "ID   Backbone Shape   D   T       Depth   alpha   "
    "peak_r   peak_I(mm/h)"
)
print("-" * 78)

for row in rows:
    if row["duration_min"] <= 30:
        print(
            f"{row['gaussian_id']:<5}"
            f"{row['backbone_id']:<9}"
            f"{row['shape']:^6}"
            f"{row['duration_min']:>4}"
            f"{row['return_period_years']:>7}"
            f"{row['feh_depth_mm']:>9.2f}"
            f"{row['alpha']:>8.1f}"
            f"{row['requested_peak_ratio']:>9.2f}"
            f"{row['peak_intensity_mm_h']:>13.2f}"
        )

print()
print("QA complete.")

