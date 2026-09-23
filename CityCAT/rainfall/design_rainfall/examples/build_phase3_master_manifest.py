"""Build the approved Stockbridge Phase 3 rainfall master manifest.

The Phase 3 design contains 200 rainfall scenarios generated from
40 common FEH22 rainfall backbones:

    40 Triangular
    80 Gaussian / skew-normal
    80 FSR/FEH
    ----------------
    200 total

Every backbone is defined by:
- storm duration;
- return period;
- FEH22 point rainfall depth for Stockbridge.

The rainfall families then vary the temporal distribution while retaining
the same backbone duration, return period, and target depth.

Approved family recipes:
- Triangular:
    40 profiles with peak_position_ratio mainly between 0.30 and 0.70,
    plus four deliberately extreme 0.20/0.80 cases.

- Gaussian:
    80 profiles:
        30 left-skewed
        20 central/symmetric
        30 right-skewed
    with fixed scale_ratio=0.18 and edge_taper_ratio=0.12.

- FSR/FEH:
    40 Summer + 40 Winter profiles.

This script creates the recipe manifest only. It does NOT create the final
CityCAT rainfall files.

Output:
    CityCAT/rainfall/design_rainfall/manifests/
        phase3_stockbridge_manifest.csv
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv

from CityCAT.rainfall.design_rainfall import (
    TemporalConfiguration,
    build_fsr_feh_reference,
    load_feh_point_ddf_csv,
)


# ---------------------------------------------------------------------
# Paths and common configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]

OUTPUT_PATH = (
    REPO_ROOT
    / "CityCAT"
    / "rainfall"
    / "design_rainfall"
    / "manifests"
    / "phase3_stockbridge_manifest.csv"
)

FEH_PATH = (
    Path.home()
    / "flood"
    / "02_data_raw"
    / "DDFs"
    / "FEH_Point_Rainfall_FEH22_AM_447157_129992.csv"
)

LOCATION_ID = "STOCKBRIDGE"

RAINFALL_TIMESTEP_SECONDS = 300
SIMULATION_HORIZON_SECONDS = 7200
START_TIME_SECONDS = 0

configuration = TemporalConfiguration(
    rainfall_timestep_seconds=RAINFALL_TIMESTEP_SECONDS,
    simulation_horizon_seconds=SIMULATION_HORIZON_SECONDS,
    maximum_active_duration_seconds=SIMULATION_HORIZON_SECONDS,
    depth_tolerance_mm=1e-6,
)

feh = load_feh_point_ddf_csv(
    FEH_PATH,
    location_id=LOCATION_ID,
)


# ---------------------------------------------------------------------
# 40 approved FEH22 backbone scenarios
#
# (backbone_id, duration_minutes, return_period_years)
# ---------------------------------------------------------------------

BACKBONES = [
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
# Approved triangular recipe
# ---------------------------------------------------------------------

TRIANGULAR_PEAK_DESIGN = {
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
    "B30": 0.20,
    "B31": 0.80,
    "B32": 0.50,
    "B33": 0.30,
    "B34": 0.70,
    "B35": 0.40,
    "B36": 0.60,
    "B37": 0.20,
    "B38": 0.80,
    "B39": 0.50,
    "B40": 0.50,
}


# ---------------------------------------------------------------------
# Approved Gaussian recipe
#
# Entries:
#   (shape, alpha, peak_position_ratio)
# ---------------------------------------------------------------------

GAUSSIAN_SCALE_RATIO = 0.18
GAUSSIAN_EDGE_TAPER_RATIO = 0.12

GAUSSIAN_DESIGN = {
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
# FSR/FEH recipe
# ---------------------------------------------------------------------

FSR_FEH_PROFILES = (
    "summer",
    "winter",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

BACKBONE_LOOKUP = {
    backbone_id: (duration_min, return_period)
    for backbone_id, duration_min, return_period in BACKBONES
}


def get_ddf_record(backbone_id):
    duration_min, return_period = BACKBONE_LOOKUP[backbone_id]

    estimate = feh.table.interpolate_duration(
        LOCATION_ID,
        duration_minutes=duration_min,
        return_period_years=return_period,
        method="log_log",
    )

    source_durations = ",".join(
        f"{point.duration_minutes:g}"
        for point in estimate.source_points
    )

    return {
        "duration_min": duration_min,
        "return_period_years": return_period,
        "feh_depth_mm": float(estimate.depth_mm),
        "ddf_kind": estimate.estimate_kind,
        "ddf_method": (
            ""
            if estimate.interpolation_method is None
            else estimate.interpolation_method
        ),
        "ddf_source_durations_min": source_durations,
    }


def base_row(
    *,
    phase3_sequence,
    phase3_event_id,
    family,
    profile_subtype,
    backbone_id,
):
    ddf = get_ddf_record(backbone_id)

    return {
        "phase3_sequence": phase3_sequence,
        "phase3_event_id": phase3_event_id,
        "family": family,
        "profile_subtype": profile_subtype,
        "backbone_id": backbone_id,
        "duration_min": ddf["duration_min"],
        "return_period_years": ddf["return_period_years"],
        "feh_depth_mm": ddf["feh_depth_mm"],
        "ddf_kind": ddf["ddf_kind"],
        "ddf_method": ddf["ddf_method"],
        "ddf_source_durations_min":
            ddf["ddf_source_durations_min"],
        "start_time_seconds": START_TIME_SECONDS,
        "rainfall_timestep_seconds":
            RAINFALL_TIMESTEP_SECONDS,
        "simulation_horizon_seconds":
            SIMULATION_HORIZON_SECONDS,
        "peak_position_ratio": "",
        "gaussian_alpha": "",
        "gaussian_scale_ratio": "",
        "gaussian_edge_taper_ratio": "",
        "fsr_feh_parameter_a": "",
        "fsr_feh_parameter_b": "",
        "qa_status": "PASS",
    }


# ---------------------------------------------------------------------
# Build manifest
# ---------------------------------------------------------------------

rows = []
sequence = 1


# ---------------------------------------------------------------------
# 1. Triangular: 40
# ---------------------------------------------------------------------

for triangular_index, (
    backbone_id,
    _,
    _,
) in enumerate(BACKBONES, start=1):

    row = base_row(
        phase3_sequence=sequence,
        phase3_event_id=f"P3_TRI_{triangular_index:03d}",
        family="triangular",
        profile_subtype="triangular",
        backbone_id=backbone_id,
    )

    row["peak_position_ratio"] = (
        TRIANGULAR_PEAK_DESIGN[backbone_id]
    )

    rows.append(row)
    sequence += 1


# ---------------------------------------------------------------------
# 2. Gaussian: 80
# ---------------------------------------------------------------------

gaussian_index = 1

shape_name = {
    "L": "left_skewed",
    "C": "central",
    "R": "right_skewed",
}

for backbone_id, _, _ in BACKBONES:

    for shape, alpha, peak_ratio in GAUSSIAN_DESIGN[backbone_id]:

        row = base_row(
            phase3_sequence=sequence,
            phase3_event_id=f"P3_GAU_{gaussian_index:03d}",
            family="gaussian",
            profile_subtype=shape_name[shape],
            backbone_id=backbone_id,
        )

        row["peak_position_ratio"] = peak_ratio
        row["gaussian_alpha"] = alpha
        row["gaussian_scale_ratio"] = GAUSSIAN_SCALE_RATIO
        row["gaussian_edge_taper_ratio"] = (
            GAUSSIAN_EDGE_TAPER_RATIO
        )

        rows.append(row)

        gaussian_index += 1
        sequence += 1


# ---------------------------------------------------------------------
# 3. FSR/FEH: 80
# ---------------------------------------------------------------------

fsr_index = 1

for backbone_id, duration_min, _ in BACKBONES:

    ddf = get_ddf_record(backbone_id)

    for profile_name in FSR_FEH_PROFILES:

        reference = build_fsr_feh_reference(
            configuration,
            profile_name=profile_name,
            start_time_seconds=START_TIME_SECONDS,
            duration_seconds=duration_min * 60,
            target_depth_mm=ddf["feh_depth_mm"],
        )

        row = base_row(
            phase3_sequence=sequence,
            phase3_event_id=f"P3_FSR_{fsr_index:03d}",
            family="fsr_feh",
            profile_subtype=profile_name,
            backbone_id=backbone_id,
        )

        row["fsr_feh_parameter_a"] = reference.parameter_a
        row["fsr_feh_parameter_b"] = reference.parameter_b

        rows.append(row)

        fsr_index += 1
        sequence += 1


# ---------------------------------------------------------------------
# Strong validation checks
# ---------------------------------------------------------------------

assert len(rows) == 200
assert sequence == 201

event_ids = [
    row["phase3_event_id"]
    for row in rows
]

assert len(event_ids) == len(set(event_ids))

family_counts = Counter(
    row["family"]
    for row in rows
)

assert family_counts == {
    "triangular": 40,
    "gaussian": 80,
    "fsr_feh": 80,
}, family_counts


gaussian_subtype_counts = Counter(
    row["profile_subtype"]
    for row in rows
    if row["family"] == "gaussian"
)

assert gaussian_subtype_counts == {
    "left_skewed": 30,
    "central": 20,
    "right_skewed": 30,
}, gaussian_subtype_counts


fsr_subtype_counts = Counter(
    row["profile_subtype"]
    for row in rows
    if row["family"] == "fsr_feh"
)

assert fsr_subtype_counts == {
    "summer": 40,
    "winter": 40,
}, fsr_subtype_counts


triangular_peak_counts = Counter(
    row["peak_position_ratio"]
    for row in rows
    if row["family"] == "triangular"
)

assert triangular_peak_counts == {
    0.20: 2,
    0.30: 7,
    0.40: 7,
    0.50: 8,
    0.60: 7,
    0.70: 7,
    0.80: 2,
}, triangular_peak_counts


# Every backbone must appear exactly five times:
#   1 Triangular
#   2 Gaussian
#   2 FSR/FEH

backbone_counts = Counter(
    row["backbone_id"]
    for row in rows
)

assert set(backbone_counts.values()) == {5}, backbone_counts


# Verify family composition within each backbone.

composition = defaultdict(Counter)

for row in rows:
    composition[row["backbone_id"]][row["family"]] += 1

for backbone_id in BACKBONE_LOOKUP:

    assert composition[backbone_id] == {
        "triangular": 1,
        "gaussian": 2,
        "fsr_feh": 2,
    }, (backbone_id, composition[backbone_id])


# Verify that all five rows for a backbone use exactly the same
# duration, return period, and FEH22 depth.

for backbone_id in BACKBONE_LOOKUP:

    subset = [
        row
        for row in rows
        if row["backbone_id"] == backbone_id
    ]

    durations = {
        row["duration_min"]
        for row in subset
    }

    return_periods = {
        row["return_period_years"]
        for row in subset
    }

    depths = {
        row["feh_depth_mm"]
        for row in subset
    }

    assert len(durations) == 1
    assert len(return_periods) == 1
    assert len(depths) == 1


# ---------------------------------------------------------------------
# Write deterministic CSV
# ---------------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [
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
]

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("PHASE 3 STOCKBRIDGE MASTER MANIFEST")
print("=" * 78)

print(f"Output : {OUTPUT_PATH}")
print(f"Rows   : {len(rows)}")

print()
print("Family counts:")
for family in (
    "triangular",
    "gaussian",
    "fsr_feh",
):
    print(
        f"  {family:<12}: "
        f"{family_counts[family]}"
    )

print()
print("Gaussian subtypes:")
for subtype in (
    "left_skewed",
    "central",
    "right_skewed",
):
    print(
        f"  {subtype:<14}: "
        f"{gaussian_subtype_counts[subtype]}"
    )

print()
print("FSR/FEH subtypes:")
print(
    f"  summer        : "
    f"{fsr_subtype_counts['summer']}"
)
print(
    f"  winter        : "
    f"{fsr_subtype_counts['winter']}"
)

print()
print("Backbone validation:")
print("  40 / 40 backbones represented")
print("  5 scenarios per backbone")
print("  1 Triangular + 2 Gaussian + 2 FSR/FEH")

print()
print("All manifest validation checks passed.")
