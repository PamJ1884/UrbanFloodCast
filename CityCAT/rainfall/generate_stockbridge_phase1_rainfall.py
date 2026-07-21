#!/usr/bin/env python3
"""Generate deterministic Phase 1 rainfall inputs for the Stockbridge domain."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DT_S = 300
ACTIVE_DURATION_S = 3600
SIMULATION_HORIZON_S = 7200
PEAK_RATIO = 0.50
PURPOSE = "Phase 1 reduced Stockbridge versus full Winchester domain validation"
EXPECTED_RECORDS = SIMULATION_HORIZON_S // DT_S + 1
DEPTH_TOLERANCE_MM = 1e-6


@dataclass(frozen=True)
class StormDefinition:
    """Definition of one controlled Phase 1 rainfall event."""

    storm_id: str
    total_depth_mm: float


STORMS = (
    StormDefinition("stockbridge_phase1_weak_15mm", 15.0),
    StormDefinition("stockbridge_phase1_medium_35p5mm", 35.5),
    StormDefinition("stockbridge_phase1_intense_60mm", 60.0),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic Stockbridge Phase 1 CityCAT rainfall files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stockbridge_phase1_rainfall_v01"),
        help="Output directory (default: %(default)s)",
    )
    return parser.parse_args()


def make_timestamps() -> np.ndarray:
    """Return the complete 0--7200 second simulation timeline."""

    return np.arange(0, SIMULATION_HORIZON_S + DT_S, DT_S, dtype=np.int64)


def make_triangular_profile(total_depth_mm: float, times_s: np.ndarray) -> np.ndarray:
    """Create a discrete triangular profile normalized to the requested depth."""

    peak_time_s = int(ACTIVE_DURATION_S * PEAK_RATIO)
    shape = np.zeros(times_s.size, dtype=np.float64)
    rising = (times_s >= 0) & (times_s <= peak_time_s)
    falling = (times_s > peak_time_s) & (times_s < ACTIVE_DURATION_S)
    shape[rising] = times_s[rising] / peak_time_s
    shape[falling] = (ACTIVE_DURATION_S - times_s[falling]) / (
        ACTIVE_DURATION_S - peak_time_s
    )

    sampled_area_h = float(np.sum(shape) * DT_S / 3600.0)
    if sampled_area_h <= 0.0:
        raise ValueError("Triangular rainfall shape has a non-positive sampled area")
    return shape * (total_depth_mm / sampled_area_h)


def validate_profile(
    storm: StormDefinition,
    times_s: np.ndarray,
    intensity_mmhr: np.ndarray,
) -> float:
    """Validate temporal, physical, peak, and depth requirements."""

    if times_s.size != EXPECTED_RECORDS:
        raise ValueError(
            f"{storm.storm_id}: expected exactly {EXPECTED_RECORDS} timestamps, got {times_s.size}"
        )
    if int(times_s[0]) != 0 or int(times_s[-1]) != SIMULATION_HORIZON_S:
        raise ValueError(
            f"{storm.storm_id}: timestamps must begin at 0 and end at {SIMULATION_HORIZON_S}"
        )
    if not np.all(np.diff(times_s) == DT_S):
        raise ValueError(f"{storm.storm_id}: timestamps must be spaced exactly {DT_S} seconds apart")
    if intensity_mmhr.shape != times_s.shape:
        raise ValueError(f"{storm.storm_id}: rainfall and timestamp arrays do not align")
    if not np.isfinite(intensity_mmhr).all() or np.any(intensity_mmhr < 0.0):
        raise ValueError(f"{storm.storm_id}: rainfall values must be finite and non-negative")
    if float(intensity_mmhr[0]) != 0.0:
        raise ValueError(f"{storm.storm_id}: first rainfall value must be zero")
    if not np.all(intensity_mmhr[times_s >= ACTIVE_DURATION_S] == 0.0):
        raise ValueError(
            f"{storm.storm_id}: rainfall at and after {ACTIVE_DURATION_S} seconds must be zero"
        )

    peak_indices = np.flatnonzero(intensity_mmhr == np.max(intensity_mmhr))
    peak_time_s = int(ACTIVE_DURATION_S * PEAK_RATIO)
    if peak_indices.size != 1 or int(times_s[peak_indices[0]]) != peak_time_s:
        observed_times = times_s[peak_indices].tolist()
        raise ValueError(
            f"{storm.storm_id}: peak must occur uniquely at {peak_time_s} seconds; "
            f"observed {observed_times}"
        )

    depth_mm = float(np.sum(intensity_mmhr) * DT_S / 3600.0)
    depth_error_mm = depth_mm - storm.total_depth_mm
    if not math.isclose(depth_mm, storm.total_depth_mm, rel_tol=0.0, abs_tol=DEPTH_TOLERANCE_MM):
        raise ValueError(
            f"{storm.storm_id}: integrated depth {depth_mm:.12g} mm differs from target "
            f"{storm.total_depth_mm:.12g} mm by {depth_error_mm:.12g} mm"
        )
    return depth_mm


def make_timeseries(
    storm: StormDefinition,
    times_s: np.ndarray,
    intensity_mmhr: np.ndarray,
) -> pd.DataFrame:
    """Build the per-storm rainfall time-series table."""

    intensity_mps = intensity_mmhr / 1000.0 / 3600.0
    cumulative_depth_mm = np.cumsum(intensity_mmhr * DT_S / 3600.0)
    return pd.DataFrame(
        {
            "storm_id": storm.storm_id,
            "time_s": times_s,
            "time_min": times_s / 60.0,
            "time_h": times_s / 3600.0,
            "intensity_mmhr": intensity_mmhr,
            "intensity_mps": intensity_mps,
            "cumulative_depth_mm": cumulative_depth_mm,
        }
    )


def write_citycat_txt(path: Path, storm_id: str, frame: pd.DataFrame) -> None:
    """Write and validate one CityCAT rainfall text file."""

    if not path.parent.is_dir():
        raise ValueError(f"CityCAT output directory is invalid: {path.parent}")
    lines = [
        f"* * * {storm_id}",
        "* * * rainfall ***",
        "* * *",
        str(EXPECTED_RECORDS),
        "* * *",
    ]
    lines.extend(
        f"{int(time_s)} {float(intensity_mps):.17g}"
        for time_s, intensity_mps in zip(frame["time_s"], frame["intensity_mps"])
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate_citycat_txt(path)


def validate_citycat_txt(path: Path) -> None:
    """Confirm the exact CityCAT header and following numeric records."""

    if not path.is_file():
        raise ValueError(f"CityCAT rainfall file was not created: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"CityCAT rainfall file has an incomplete header: {path}")
    if not lines[0].startswith("* * * "):
        raise ValueError(f"Invalid CityCAT metadata line 1 in {path}: {lines[0]!r}")
    expected_header_lines = {
        1: "* * * rainfall ***",
        2: "* * *",
        3: str(EXPECTED_RECORDS),
        4: "* * *",
    }
    for line_index, expected_line in expected_header_lines.items():
        if lines[line_index] != expected_line:
            raise ValueError(
                f"Invalid CityCAT header line {line_index + 1} in {path}: "
                f"expected {expected_line!r}, got {lines[line_index]!r}"
            )
    try:
        header_count = int(lines[3].strip())
    except ValueError as exc:
        raise ValueError(f"CityCAT header N is not an integer in {path}: {lines[3]!r}") from exc
    if header_count != EXPECTED_RECORDS:
        raise ValueError(
            f"CityCAT header N must equal {EXPECTED_RECORDS} in {path}, got {header_count}"
        )

    rainfall_lines = lines[5:]
    if len(rainfall_lines) != EXPECTED_RECORDS:
        raise ValueError(
            f"CityCAT rainfall file must contain exactly {EXPECTED_RECORDS} records after "
            f"the header: {path} contains {len(rainfall_lines)}"
        )

    numeric_records = []
    for record_number, line in enumerate(rainfall_lines, start=1):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Invalid numeric rainfall record {record_number} in {path}: {line!r}"
            )
        try:
            numeric_records.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric rainfall record {record_number} in {path}: {line!r}"
            ) from exc
    if numeric_records[-1][0] != SIMULATION_HORIZON_S:
        raise ValueError(
            f"Final CityCAT rainfall timestamp must be {SIMULATION_HORIZON_S} seconds in "
            f"{path}, got {numeric_records[-1][0]:g}"
        )


def prepare_output_paths(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Create and validate the requested output directory structure."""

    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {out_dir}")
    citycat_dir = out_dir / "citycat_txt"
    csv_dir = out_dir / "csv_timeseries"
    figures_dir = out_dir / "figures"
    for directory in (out_dir, citycat_dir, csv_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValueError(f"Output directory is invalid: {directory}")
    return citycat_dir, csv_dir, figures_dir, out_dir / "phase1_rainfall_summary.csv"


def plot_hyetographs(series: Sequence[pd.DataFrame], figure_path: Path) -> None:
    """Plot all three controlled rainfall hyetographs in one figure."""

    if not figure_path.parent.is_dir():
        raise ValueError(f"Figure output directory is invalid: {figure_path.parent}")
    fig, ax = plt.subplots()
    for frame in series:
        ax.step(
            frame["time_h"],
            frame["intensity_mmhr"],
            where="post",
            label=str(frame["storm_id"].iloc[0]),
        )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Rainfall intensity (mm/h)")
    ax.set_title("Stockbridge Phase 1 rainfall hyetographs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    if not figure_path.is_file():
        raise ValueError(f"Hyetograph figure was not created: {figure_path}")


def write_summary(rows: Sequence[dict[str, object]], summary_path: Path) -> None:
    """Write the three-row Phase 1 storm summary."""

    if not rows:
        raise ValueError("Cannot write an empty rainfall summary")
    if not summary_path.parent.is_dir():
        raise ValueError(f"Summary output directory is invalid: {summary_path.parent}")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not summary_path.is_file():
        raise ValueError(f"Rainfall summary was not created: {summary_path}")


def main() -> None:
    """Generate rainfall text files, tables, summary metadata, and plot."""

    args = parse_args()
    citycat_dir, csv_dir, figures_dir, summary_path = prepare_output_paths(args.out_dir)
    times_s = make_timestamps()
    summary_rows: list[dict[str, object]] = []
    all_series: list[pd.DataFrame] = []

    for storm in STORMS:
        intensity_mmhr = make_triangular_profile(storm.total_depth_mm, times_s)
        depth_mm = validate_profile(storm, times_s, intensity_mmhr)
        frame = make_timeseries(storm, times_s, intensity_mmhr)
        txt_path = citycat_dir / f"{storm.storm_id}.txt"
        csv_path = csv_dir / f"{storm.storm_id}.csv"
        write_citycat_txt(txt_path, storm.storm_id, frame)
        frame.to_csv(csv_path, index=False)
        if not csv_path.is_file():
            raise ValueError(f"Rainfall CSV was not created: {csv_path}")

        peak_index = int(np.argmax(intensity_mmhr))
        peak_mmhr = float(intensity_mmhr[peak_index])
        summary_rows.append(
            {
                "storm_id": storm.storm_id,
                "purpose": PURPOSE,
                "profile": "triangular",
                "active_duration_s": ACTIVE_DURATION_S,
                "active_duration_h": ACTIVE_DURATION_S / 3600.0,
                "simulation_horizon_s": SIMULATION_HORIZON_S,
                "simulation_horizon_h": SIMULATION_HORIZON_S / 3600.0,
                "dt_s": DT_S,
                "dt_min": DT_S / 60.0,
                "total_records": len(times_s),
                "total_depth_mm_target": storm.total_depth_mm,
                "total_depth_mm_check": depth_mm,
                "depth_error_mm": depth_mm - storm.total_depth_mm,
                "peak_ratio": PEAK_RATIO,
                "peak_time_s": int(times_s[peak_index]),
                "peak_mmhr": peak_mmhr,
                "peak_mps": peak_mmhr / 1000.0 / 3600.0,
                "dry_period_s": SIMULATION_HORIZON_S - ACTIVE_DURATION_S,
                "txt_path": str(txt_path),
                "csv_path": str(csv_path),
            }
        )
        all_series.append(frame)

    write_summary(summary_rows, summary_path)
    plot_hyetographs(
        all_series,
        figures_dir / "stockbridge_phase1_hyetographs.png",
    )
    print(f"Generated {len(STORMS)} Phase 1 rainfall events in {summary_path.parent}")


if __name__ == "__main__":
    main()
