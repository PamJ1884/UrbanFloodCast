#!/usr/bin/env python3
"""Plot production Stockbridge Phase 3 rainfall profiles by common backbone.

The plots are generated from the final CityCAT rainfall files, not from
reconstructed or illustrative profiles. Manifest metadata is used for event
identity and hydrological design information.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


DEFAULT_MANIFEST = Path(
    "~/flood/02_data/CityCAT_Winchester/manifests/"
    "phase3_stockbridge_manifest.csv"
).expanduser()

DEFAULT_RAINFALL_DIR = Path(
    "~/flood/02_data/CityCAT_Winchester/rainfall/"
    "stockbridge_phase3_v01/citycat_txt"
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot final Stockbridge Phase 3 CityCAT rainfall profiles "
            "for selected common backbones."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--rainfall-dir",
        type=Path,
        default=DEFAULT_RAINFALL_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["B05", "B40"],
        help="Backbone IDs to plot, e.g. B05 B40",
    )

    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"manifest is empty: {path}")

    return rows


def read_citycat_rainfall(
    path: Path,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Read one final CityCAT rainfall file.

    Returns
    -------
    storm_id
        Event ID stored in the first CityCAT comment line.
    time_seconds
        CityCAT rainfall timestamps.
    intensity_mm_per_hour
        Rainfall intensity converted from CityCAT m/s to mm/h.
    """

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(lines) < 6:
        raise ValueError(
            f"CityCAT file is too short: {path}"
        )

    storm_id = lines[0].replace(
        "* * *",
        "",
        1,
    ).strip()

    try:
        number_of_records = int(lines[3])
    except ValueError as exc:
        raise ValueError(
            f"invalid CityCAT record count: {path}"
        ) from exc

    records = lines[5:]

    if len(records) != number_of_records:
        raise ValueError(
            f"{path.name}: expected {number_of_records} records, "
            f"found {len(records)}"
        )

    times = []
    values_m_per_s = []

    for line in records:
        tokens = line.split()

        if len(tokens) != 2:
            raise ValueError(
                f"malformed rainfall record in {path}: {line!r}"
            )

        times.append(int(tokens[0]))
        values_m_per_s.append(float(tokens[1]))

    time_seconds = np.asarray(
        times,
        dtype=np.float64,
    )

    intensity_mm_per_hour = (
        np.asarray(
            values_m_per_s,
            dtype=np.float64,
        )
        * 1000.0
        * 3600.0
    )

    return (
        storm_id,
        time_seconds,
        intensity_mm_per_hour,
    )


def family_sort_key(
    row: dict[str, str],
) -> tuple[int, int]:
    family_rank = {
        "triangular": 0,
        "gaussian": 1,
        "fsr_feh": 2,
    }

    subtype_rank = {
        "triangular": 0,
        "left_skewed": 0,
        "central": 1,
        "symmetric": 1,
        "right_skewed": 2,
        "summer": 0,
        "winter": 1,
    }

    return (
        family_rank[row["family"]],
        subtype_rank.get(
            row["profile_subtype"],
            99,
        ),
    )


def format_number(value: str) -> str:
    number = float(value)

    if number.is_integer():
        return str(int(number))

    return f"{number:g}"


def profile_label(
    row: dict[str, str],
    peak_mm_per_hour: float,
) -> str:
    event_id = row["phase3_event_id"]
    family = row["family"]

    if family == "triangular":
        return (
            f"{event_id} — Triangular "
            f"(p={float(row['peak_position_ratio']):.2f}; "
            f"peak={peak_mm_per_hour:.1f} mm/h)"
        )

    if family == "gaussian":
        subtype = row["profile_subtype"].replace(
            "_",
            " ",
        )

        return (
            f"{event_id} — Gaussian {subtype} "
            f"(α={float(row['gaussian_alpha']):g}, "
            f"p={float(row['peak_position_ratio']):.2f}; "
            f"peak={peak_mm_per_hour:.1f} mm/h)"
        )

    if family == "fsr_feh":
        return (
            f"{event_id} — FSR/FEH "
            f"{row['profile_subtype']} "
            f"(peak={peak_mm_per_hour:.1f} mm/h)"
        )

    raise ValueError(
        f"unsupported family: {family}"
    )


def validate_common_backbone(
    rows: list[dict[str, str]],
    backbone: str,
) -> tuple[float, float, float]:
    if len(rows) != 5:
        raise ValueError(
            f"{backbone}: expected 5 profiles, found {len(rows)}"
        )

    durations = {
        float(row["duration_min"])
        for row in rows
    }

    return_periods = {
        float(row["return_period_years"])
        for row in rows
    }

    depths = {
        float(row["feh_depth_mm"])
        for row in rows
    }

    if (
        len(durations) != 1
        or len(return_periods) != 1
        or len(depths) != 1
    ):
        raise ValueError(
            f"{backbone}: profiles do not share a common hydrological design"
        )

    return (
        durations.pop(),
        return_periods.pop(),
        depths.pop(),
    )


def create_backbone_figure(
    rows: list[dict[str, str]],
    backbone: str,
    rainfall_directory: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    duration_min, return_period, depth_mm = (
        validate_common_backbone(
            rows,
            backbone,
        )
    )

    rows = sorted(
        rows,
        key=family_sort_key,
    )

    figure, axis = plt.subplots(
        figsize=(15.5, 7.0)
    )

    global_maximum = 0.0

    for row in rows:
        sequence = int(
            row["phase3_sequence"]
        )

        path = (
            rainfall_directory
            / f"Rainfall_Data_{sequence}.txt"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"missing production rainfall file: {path}"
            )

        (
            storm_id,
            time_seconds,
            intensity_mm_per_hour,
        ) = read_citycat_rainfall(path)

        if storm_id != row["phase3_event_id"]:
            raise ValueError(
                f"{path.name}: storm ID {storm_id!r} "
                f"does not match manifest event "
                f"{row['phase3_event_id']!r}"
            )

        integrated_depth = float(
            np.trapz(
                intensity_mm_per_hour,
                time_seconds / 3600.0,
            )
        )

        if not math.isclose(
            integrated_depth,
            depth_mm,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{storm_id}: integrated depth "
                f"{integrated_depth:.12g} mm "
                f"does not match {depth_mm:.12g} mm"
            )

        peak = float(
            np.max(
                intensity_mm_per_hour
            )
        )

        global_maximum = max(
            global_maximum,
            peak,
        )

        axis.plot(
            time_seconds / 60.0,
            intensity_mm_per_hour,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=profile_label(
                row,
                peak,
            ),
        )

    axis.axvline(
        duration_min,
        linestyle=":",
        linewidth=1.2,
    )

    axis.text(
        duration_min,
        global_maximum * 1.03,
        f"D = {duration_min:g} min",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    x_limit = min(
        120.0,
        max(
            30.0,
            duration_min + 15.0,
            duration_min * 1.15,
        ),
    )

    axis.set_xlim(
        0.0,
        x_limit,
    )

    axis.set_ylim(
        0.0,
        global_maximum * 1.14,
    )

    axis.set_xlabel(
        "Time (min)"
    )

    axis.set_ylabel(
        "Rainfall intensity (mm/h)"
    )

    axis.grid(
        True,
        linewidth=0.6,
        alpha=0.35,
    )

    axis.set_title(
        (
            f"Stockbridge Phase 3 — common backbone {backbone}\n"
            f"D = {duration_min:g} min | "
            f"T = {return_period:g} yr | "
            f"FEH22 depth = {depth_mm:g} mm"
        ),
        fontweight="bold",
        fontsize=14,
    )

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=8.5,
        framealpha=0.95,
    )

    figure.text(
        0.5,
        0.015,
        (
            "Same backbone: duration, return period and FEH22 total depth "
            "are fixed; only temporal rainfall distribution changes. "
            "Markers are the 5-minute CityCAT forcing samples; "
            "lines are visual guides between samples."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(0.0, 0.08, 1.0, 1.0)
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_directory
        / f"phase3_{backbone}_common_backbone.png"
    )

    pdf_path = (
        output_directory
        / f"phase3_{backbone}_common_backbone.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return png_path, pdf_path


def main() -> None:
    args = parse_args()

    manifest_path = (
        args.manifest
        .expanduser()
        .resolve()
    )

    rainfall_directory = (
        args.rainfall_dir
        .expanduser()
        .resolve()
    )

    output_directory = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    rows = read_manifest(
        manifest_path
    )

    for backbone in args.backbones:
        selected = [
            row
            for row in rows
            if row["backbone_id"] == backbone
        ]

        if not selected:
            raise ValueError(
                f"backbone not found in manifest: {backbone}"
            )

        png_path, pdf_path = (
            create_backbone_figure(
                selected,
                backbone,
                rainfall_directory,
                output_directory,
            )
        )

        print(
            f"{backbone}: PASS"
        )

        print(
            f"  PNG: {png_path}"
        )

        print(
            f"  PDF: {pdf_path}"
        )


if __name__ == "__main__":
    main()
