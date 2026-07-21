#!/usr/bin/env python3
"""Generate a deterministic, space-filling Phase 2 rainfall ensemble."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_N_STORMS = 100
DEFAULT_SEED = 1884
SIMULATION_HORIZON_S = 7200
TIMESTEP_S = 300
NUMBER_OF_RECORDS = SIMULATION_HORIZON_S // TIMESTEP_S + 1
MIN_DEPTH_MM = 10.0
MAX_DEPTH_MM = 60.0
MIN_DURATION_S = 30 * 60
MAX_DURATION_S = 90 * 60
MIN_PEAK_RATIO = 0.25
MAX_PEAK_RATIO = 0.75
MAX_PEAK_MM_PER_HOUR = 120.0
DEPTH_TOLERANCE_MM = 1e-6
PROFILE_TYPE = "asymmetric_triangular_pulse"
CANDIDATE_POOL_MINIMUM = 20_000
SPLIT_NAMES = ("train", "validation", "test")
PHASE1_BENCHMARKS = (
    (15.0, 3600, 1800),
    (35.5, 3600, 1800),
    (60.0, 3600, 1800),
)


@dataclass(frozen=True)
class StormCandidate:
    """One feasible rainfall design and its discretised time series."""

    total_depth_mm: float
    active_duration_s: int
    peak_time_s: int
    peak_time_ratio: float
    peak_intensity_mm_per_hour: float
    intensity_mm_per_hour: np.ndarray
    rainfall_series_sha256: str


@dataclass(frozen=True)
class SelectedStorm:
    """A selected candidate with sequential CityCAT metadata."""

    rainfall_index: int
    storm_id: str
    split: str
    candidate: StormCandidate


def parse_args() -> argparse.Namespace:
    """Parse ensemble generation options."""

    parser = argparse.ArgumentParser(
        description="Generate the Stockbridge Phase 2 core synthetic rainfall ensemble."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for the generated ensemble",
    )
    parser.add_argument(
        "--n-storms",
        type=int,
        default=DEFAULT_N_STORMS,
        help="Number of storms to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated ensemble in the output directory",
    )
    return parser.parse_args()


def simulation_times() -> np.ndarray:
    """Return the fixed 0--7200 second CityCAT timeline."""

    return np.arange(0, SIMULATION_HORIZON_S + TIMESTEP_S, TIMESTEP_S, dtype=np.int64)


def rainfall_hash(intensity_mm_per_hour: np.ndarray) -> str:
    """Return a stable hash of a rainfall intensity series."""

    values = np.asarray(intensity_mm_per_hour, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_rainfall_series(
    total_depth_mm: float,
    active_duration_s: int,
    peak_time_s: int,
) -> np.ndarray:
    """Construct and depth-normalise one asymmetric triangular pulse."""

    times_s = simulation_times()
    intensity = np.zeros(times_s.size, dtype=np.float64)
    rising = (times_s > 0) & (times_s <= peak_time_s)
    falling = (times_s > peak_time_s) & (times_s < active_duration_s)
    intensity[rising] = times_s[rising] / peak_time_s
    intensity[falling] = (active_duration_s - times_s[falling]) / (
        active_duration_s - peak_time_s
    )
    unscaled_depth_mm = float(np.trapz(intensity, times_s / 3600.0))
    if unscaled_depth_mm <= 0.0:
        raise ValueError("Candidate triangular pulse has a non-positive integrated area")
    intensity *= total_depth_mm / unscaled_depth_mm
    return intensity


def is_phase1_benchmark(total_depth_mm: float, duration_s: int, peak_time_s: int) -> bool:
    """Return whether a parameter tuple exactly matches a Phase 1 event."""

    return any(
        math.isclose(total_depth_mm, benchmark_depth, rel_tol=0.0, abs_tol=1e-12)
        and duration_s == benchmark_duration
        and peak_time_s == benchmark_peak
        for benchmark_depth, benchmark_duration, benchmark_peak in PHASE1_BENCHMARKS
    )


def valid_peak_steps(duration_steps: int) -> np.ndarray:
    """Return internal grid steps within the allowed peak-ratio interval."""

    minimum_step = max(1, math.ceil(MIN_PEAK_RATIO * duration_steps))
    maximum_step = min(duration_steps - 1, math.floor(MAX_PEAK_RATIO * duration_steps))
    return np.arange(minimum_step, maximum_step + 1, dtype=np.int64)


def generate_candidate_pool(n_storms: int, rng: np.random.Generator) -> list[StormCandidate]:
    """Generate a large unique pool of feasible gridded rainfall candidates."""

    target_pool_size = max(CANDIDATE_POOL_MINIMUM, n_storms * 200)
    duration_steps_options = np.arange(
        MIN_DURATION_S // TIMESTEP_S,
        MAX_DURATION_S // TIMESTEP_S + 1,
        dtype=np.int64,
    )
    candidates: list[StormCandidate] = []
    parameter_keys: set[tuple[float, int, int]] = set()
    series_hashes: set[str] = set()
    maximum_attempts = target_pool_size * 50

    for _ in range(maximum_attempts):
        duration_steps = int(rng.choice(duration_steps_options))
        peak_step = int(rng.choice(valid_peak_steps(duration_steps)))
        active_duration_s = duration_steps * TIMESTEP_S
        peak_time_s = peak_step * TIMESTEP_S
        total_depth_mm = float(rng.uniform(MIN_DEPTH_MM, MAX_DEPTH_MM))
        if is_phase1_benchmark(total_depth_mm, active_duration_s, peak_time_s):
            continue

        intensity = make_rainfall_series(total_depth_mm, active_duration_s, peak_time_s)
        peak_intensity = float(np.max(intensity))
        if peak_intensity > MAX_PEAK_MM_PER_HOUR + 1e-12:
            continue
        parameter_key = (round(total_depth_mm, 12), active_duration_s, peak_time_s)
        series_sha256 = rainfall_hash(intensity)
        if parameter_key in parameter_keys or series_sha256 in series_hashes:
            continue

        parameter_keys.add(parameter_key)
        series_hashes.add(series_sha256)
        candidates.append(
            StormCandidate(
                total_depth_mm=total_depth_mm,
                active_duration_s=active_duration_s,
                peak_time_s=peak_time_s,
                peak_time_ratio=peak_time_s / active_duration_s,
                peak_intensity_mm_per_hour=peak_intensity,
                intensity_mm_per_hour=intensity,
                rainfall_series_sha256=series_sha256,
            )
        )
        if len(candidates) == target_pool_size:
            return candidates

    raise ValueError(
        f"Could not generate the requested candidate pool of {target_pool_size} unique "
        f"feasible storms after {maximum_attempts} attempts"
    )


def normalised_parameters(candidates: Sequence[StormCandidate]) -> np.ndarray:
    """Map design parameters to the shared unit cube."""

    return np.asarray(
        [
            (
                (candidate.total_depth_mm - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM),
                (candidate.active_duration_s - MIN_DURATION_S)
                / (MAX_DURATION_S - MIN_DURATION_S),
                (candidate.peak_time_ratio - MIN_PEAK_RATIO)
                / (MAX_PEAK_RATIO - MIN_PEAK_RATIO),
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )


def greedy_maximin_select(
    candidates: Sequence[StormCandidate],
    n_storms: int,
    rng: np.random.Generator,
) -> list[StormCandidate]:
    """Select a seeded space-filling subset using greedy maximin distance."""

    if n_storms > len(candidates):
        raise ValueError(f"Cannot select {n_storms} storms from {len(candidates)} candidates")
    points = normalised_parameters(candidates)
    selected_mask = np.zeros(len(candidates), dtype=bool)
    first_index = int(rng.integers(0, len(candidates)))
    selected_indices = [first_index]
    selected_mask[first_index] = True
    minimum_squared_distance = np.sum((points - points[first_index]) ** 2, axis=1)

    while len(selected_indices) < n_storms:
        scores = minimum_squared_distance.copy()
        scores[selected_mask] = -1.0
        next_index = int(np.argmax(scores))
        selected_indices.append(next_index)
        selected_mask[next_index] = True
        squared_distance = np.sum((points - points[next_index]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, squared_distance)
    return [candidates[index] for index in selected_indices]


def split_targets(n_storms: int) -> dict[str, int]:
    """Return deterministic 70/15/15 split quotas using largest remainders."""

    proportions = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)
    exact_counts = proportions * n_storms
    counts = np.floor(exact_counts).astype(int)
    remainder = n_storms - int(np.sum(counts))
    fractional_order = np.argsort(-(exact_counts - counts), kind="stable")
    for index in fractional_order[:remainder]:
        counts[index] += 1
    for empty_index in np.flatnonzero(counts == 0):
        donor_index = int(np.argmax(counts))
        if counts[donor_index] <= 1:
            raise ValueError(f"Cannot represent all three splits with {n_storms} storms")
        counts[donor_index] -= 1
        counts[empty_index] += 1
    return {name: int(count) for name, count in zip(SPLIT_NAMES, counts)}


def assign_space_filling_splits(
    candidates: Sequence[StormCandidate],
    targets: dict[str, int],
) -> list[str]:
    """Assign quota-constrained splits with maximin coverage inside each split."""

    points = normalised_parameters(candidates)
    centre = np.full(points.shape[1], 0.5, dtype=np.float64)
    remaining = np.ones(len(candidates), dtype=bool)
    members: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}

    while np.any(remaining):
        eligible_splits = [name for name in SPLIT_NAMES if len(members[name]) < targets[name]]
        if not eligible_splits:
            raise ValueError("Split quotas were exhausted before all storms were assigned")
        split = min(
            eligible_splits,
            key=lambda name: (len(members[name]) / targets[name], SPLIT_NAMES.index(name)),
        )
        available_indices = np.flatnonzero(remaining)
        if not members[split]:
            distances = np.sum((points[available_indices] - centre) ** 2, axis=1)
            chosen_index = int(available_indices[np.argmin(distances)])
        else:
            assigned_points = points[np.asarray(members[split], dtype=np.int64)]
            differences = points[available_indices, None, :] - assigned_points[None, :, :]
            minimum_distances = np.min(np.sum(differences**2, axis=2), axis=1)
            chosen_index = int(available_indices[np.argmax(minimum_distances)])
        members[split].append(chosen_index)
        remaining[chosen_index] = False

    assignments = [""] * len(candidates)
    for split, indices in members.items():
        for index in indices:
            assignments[index] = split
    if any(not split for split in assignments):
        raise ValueError("At least one selected storm did not receive a split")
    return assignments


def generated_artifacts(output_dir: Path) -> list[Path]:
    """List only files owned by this generator within an output directory."""

    artifacts = [
        output_dir / "phase2_core_rainfall_summary.csv",
        output_dir / "phase2_core_manifest.json",
    ]
    artifacts.extend((output_dir / "citycat_txt").glob("Rainfall_Data_*.txt"))
    artifacts.extend((output_dir / "csv_timeseries").glob("Rainfall_Data_*.csv"))
    artifacts.extend(
        [
            output_dir / "qc" / "depth_vs_duration_by_split.png",
            output_dir / "qc" / "peak_ratio_vs_depth.png",
            output_dir / "qc" / "depth_histogram.png",
            output_dir / "qc" / "duration_histogram.png",
            output_dir / "qc" / "peak_intensity_histogram.png",
            output_dir / "qc" / "parameter_histograms.png",
            output_dir / "qc" / "representative_hyetographs.png",
        ]
    )
    return sorted({path for path in artifacts if path.exists()}, key=str)


def prepare_output_directories(
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    """Create output directories and safely handle known generated artifacts."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")
    existing_artifacts = generated_artifacts(output_dir)
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            f"Generated ensemble already exists in {output_dir}; use --overwrite to replace it"
        )
    if overwrite:
        for path in existing_artifacts:
            if not path.is_file():
                raise ValueError(f"Expected generated artifact is not a regular file: {path}")
            path.unlink()

    citycat_dir = output_dir / "citycat_txt"
    csv_dir = output_dir / "csv_timeseries"
    qc_dir = output_dir / "qc"
    for directory in (output_dir, citycat_dir, csv_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValueError(f"Output directory is invalid: {directory}")
    return citycat_dir, csv_dir, qc_dir


def validate_storm(candidate: StormCandidate) -> None:
    """Validate one selected storm against all physical and temporal constraints."""

    times_s = simulation_times()
    intensity = candidate.intensity_mm_per_hour
    if times_s.size != NUMBER_OF_RECORDS or intensity.size != NUMBER_OF_RECORDS:
        raise ValueError("Rainfall series must contain exactly 25 records")
    if not np.array_equal(times_s, np.arange(0, 7201, 300, dtype=np.int64)):
        raise ValueError("Rainfall timestamps are not exactly 0, 300, ..., 7200 seconds")
    if not np.isfinite(intensity).all() or np.any(intensity < 0.0):
        raise ValueError("Rainfall intensity contains negative or non-finite values")
    if float(intensity[0]) != 0.0:
        raise ValueError("Rainfall intensity at time zero must be zero")
    end_index = candidate.active_duration_s // TIMESTEP_S
    if float(intensity[end_index]) != 0.0 or np.any(intensity[end_index:] != 0.0):
        raise ValueError("Rainfall must be zero at and after the active-duration endpoint")
    maximum = float(np.max(intensity))
    peak_indices = np.flatnonzero(intensity == maximum)
    if maximum <= 0.0 or peak_indices.size != 1:
        raise ValueError("Rainfall series must contain exactly one positive peak")
    peak_time_s = int(times_s[peak_indices[0]])
    if peak_time_s != candidate.peak_time_s or not 0 < peak_time_s < candidate.active_duration_s:
        raise ValueError("Rainfall peak time is not strictly inside the active duration")
    if candidate.active_duration_s % TIMESTEP_S != 0 or peak_time_s % TIMESTEP_S != 0:
        raise ValueError("Active duration and peak time must align to 300-second intervals")
    if not MIN_DURATION_S <= candidate.active_duration_s <= MAX_DURATION_S:
        raise ValueError("Active duration lies outside the 30--90 minute limits")
    if not MIN_PEAK_RATIO <= candidate.peak_time_ratio <= MAX_PEAK_RATIO:
        raise ValueError("Peak-time ratio lies outside the 0.25--0.75 limits")
    integrated_depth = float(np.trapz(intensity, times_s / 3600.0))
    if not math.isclose(
        integrated_depth,
        candidate.total_depth_mm,
        rel_tol=0.0,
        abs_tol=DEPTH_TOLERANCE_MM,
    ):
        raise ValueError(
            f"Integrated depth {integrated_depth:.12g} mm does not match target "
            f"{candidate.total_depth_mm:.12g} mm"
        )
    if maximum > MAX_PEAK_MM_PER_HOUR + 1e-12:
        raise ValueError(f"Peak intensity exceeds {MAX_PEAK_MM_PER_HOUR:g} mm/hour")
    if rainfall_hash(intensity) != candidate.rainfall_series_sha256:
        raise ValueError("Rainfall-series SHA-256 does not match the intensity values")
    if is_phase1_benchmark(
        candidate.total_depth_mm,
        candidate.active_duration_s,
        candidate.peak_time_s,
    ):
        raise ValueError("Selected event duplicates an excluded Phase 1 benchmark")


def write_citycat_file(path: Path, storm: SelectedStorm) -> None:
    """Write one CityCAT rainfall file with the exact required header."""

    times_s = simulation_times()
    intensity_m_per_second = storm.candidate.intensity_mm_per_hour / 1000.0 / 3600.0
    lines = [
        f"* * * {storm.storm_id}",
        "* * * rainfall ***",
        "* * *",
        str(NUMBER_OF_RECORDS),
        "* * *",
    ]
    lines.extend(
        f"{int(time_s)} {float(intensity_mps):.17g}"
        for time_s, intensity_mps in zip(times_s, intensity_m_per_second)
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def validate_citycat_file(path: Path, storm: SelectedStorm) -> None:
    """Reopen and validate one CityCAT TXT file and its unit conversion."""

    lines = path.read_text(encoding="utf-8").splitlines()
    expected_header = [
        f"* * * {storm.storm_id}",
        "* * * rainfall ***",
        "* * *",
        str(NUMBER_OF_RECORDS),
        "* * *",
    ]
    if lines[:5] != expected_header:
        raise ValueError(f"CityCAT header validation failed: {path}")
    numeric_lines = lines[5:]
    if len(numeric_lines) != NUMBER_OF_RECORDS:
        raise ValueError(f"CityCAT file must contain exactly 25 numeric records: {path}")
    parsed_times: list[int] = []
    parsed_mps: list[float] = []
    for record_number, line in enumerate(numeric_lines, start=1):
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(f"Malformed rainfall record {record_number} in {path}: {line!r}")
        try:
            parsed_times.append(int(tokens[0]))
            parsed_mps.append(float(tokens[1]))
        except ValueError as exc:
            raise ValueError(
                f"Malformed rainfall record {record_number} in {path}: {line!r}"
            ) from exc
    if not np.array_equal(np.asarray(parsed_times, dtype=np.int64), simulation_times()):
        raise ValueError(f"CityCAT timestamps are not exactly 0, 300, ..., 7200: {path}")
    parsed_mps_array = np.asarray(parsed_mps, dtype=np.float64)
    expected_mps = storm.candidate.intensity_mm_per_hour / 1000.0 / 3600.0
    if not np.isfinite(parsed_mps_array).all() or np.any(parsed_mps_array < 0.0):
        raise ValueError(f"CityCAT rainfall values are negative or non-finite: {path}")
    if not np.array_equal(parsed_mps_array, expected_mps):
        raise ValueError(f"CityCAT metres-per-second conversion is incorrect: {path}")


def cumulative_depth_mm(intensity_mm_per_hour: np.ndarray) -> np.ndarray:
    """Return trapezoidally integrated depth at every timestamp."""

    interval_depths = (
        0.5
        * (intensity_mm_per_hour[:-1] + intensity_mm_per_hour[1:])
        * TIMESTEP_S
        / 3600.0
    )
    return np.concatenate(([0.0], np.cumsum(interval_depths)))


def write_timeseries_csv(path: Path, storm: SelectedStorm) -> None:
    """Write one dependency-free rainfall time-series CSV."""

    times_s = simulation_times()
    intensity = storm.candidate.intensity_mm_per_hour
    intensity_mps = intensity / 1000.0 / 3600.0
    cumulative = cumulative_depth_mm(intensity)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "time_seconds",
                "time_minutes",
                "intensity_mm_per_hour",
                "intensity_m_per_second",
                "cumulative_depth_mm",
            ]
        )
        for time_s, mm_per_hour, m_per_second, depth_mm in zip(
            times_s,
            intensity,
            intensity_mps,
            cumulative,
        ):
            writer.writerow(
                [
                    int(time_s),
                    f"{time_s / 60.0:.10g}",
                    f"{float(mm_per_hour):.17g}",
                    f"{float(m_per_second):.17g}",
                    f"{float(depth_mm):.17g}",
                ]
            )


def summary_row(storm: SelectedStorm) -> dict[str, object]:
    """Return summary metadata for one selected event."""

    candidate = storm.candidate
    integrated_depth_mm = float(
        np.trapz(candidate.intensity_mm_per_hour, simulation_times() / 3600.0)
    )
    return {
        "rainfall_index": storm.rainfall_index,
        "storm_id": storm.storm_id,
        "split": storm.split,
        "profile_type": PROFILE_TYPE,
        "target_total_depth_mm": candidate.total_depth_mm,
        "integrated_total_depth_mm": integrated_depth_mm,
        "active_duration_seconds": candidate.active_duration_s,
        "active_duration_minutes": candidate.active_duration_s / 60.0,
        "peak_time_seconds": candidate.peak_time_s,
        "peak_time_minutes": candidate.peak_time_s / 60.0,
        "peak_time_ratio": candidate.peak_time_ratio,
        "peak_intensity_mm_per_hour": candidate.peak_intensity_mm_per_hour,
        "recession_duration_minutes": (
            candidate.active_duration_s - candidate.peak_time_s
        )
        / 60.0,
        "number_of_records": NUMBER_OF_RECORDS,
        "timestep_seconds": TIMESTEP_S,
        "rainfall_series_sha256": candidate.rainfall_series_sha256,
    }


def write_summary_csv(path: Path, storms: Sequence[SelectedStorm]) -> None:
    """Write ensemble metadata in sequential rainfall-index order."""

    rows = [summary_row(storm) for storm in storms]
    if not rows:
        raise ValueError("Cannot write an empty ensemble summary")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def representative_storms(storms: Sequence[SelectedStorm]) -> list[SelectedStorm]:
    """Choose the event nearest each split centroid for QC plotting."""

    representatives: list[SelectedStorm] = []
    for split in SPLIT_NAMES:
        split_storms = [storm for storm in storms if storm.split == split]
        if not split_storms:
            continue
        points = normalised_parameters([storm.candidate for storm in split_storms])
        centroid = np.mean(points, axis=0)
        index = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
        representatives.append(split_storms[index])
    return representatives


def create_qc_plots(qc_dir: Path, storms: Sequence[SelectedStorm]) -> None:
    """Create four standard-matplotlib ensemble quality-control figures."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for split in SPLIT_NAMES:
        split_storms = [storm for storm in storms if storm.split == split]
        ax.scatter(
            [storm.candidate.active_duration_s / 60.0 for storm in split_storms],
            [storm.candidate.total_depth_mm for storm in split_storms],
            label=split,
        )
    ax.set_xlabel("Active rainfall duration (minutes)")
    ax.set_ylabel("Total rainfall depth (mm)")
    ax.set_title("Phase 2 depth versus active duration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(qc_dir / "depth_vs_duration_by_split.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for split in SPLIT_NAMES:
        split_storms = [storm for storm in storms if storm.split == split]
        ax.scatter(
            [storm.candidate.peak_time_ratio for storm in split_storms],
            [storm.candidate.total_depth_mm for storm in split_storms],
            label=split,
        )
    ax.set_xlabel("Peak-time ratio")
    ax.set_ylabel("Total rainfall depth (mm)")
    ax.set_title("Phase 2 peak position versus rainfall depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(qc_dir / "peak_ratio_vs_depth.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist([storm.candidate.total_depth_mm for storm in storms])
    ax.set_xlabel("Total depth (mm)")
    ax.set_ylabel("Storm count")
    ax.set_title("Phase 2 total-depth distribution")
    fig.tight_layout()
    fig.savefig(qc_dir / "depth_histogram.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist([storm.candidate.active_duration_s / 60.0 for storm in storms])
    ax.set_xlabel("Active duration (minutes)")
    ax.set_ylabel("Storm count")
    ax.set_title("Phase 2 active-duration distribution")
    fig.tight_layout()
    fig.savefig(qc_dir / "duration_histogram.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist([storm.candidate.peak_intensity_mm_per_hour for storm in storms])
    ax.set_xlabel("Peak intensity (mm/hour)")
    ax.set_ylabel("Storm count")
    ax.set_title("Phase 2 peak-intensity distribution")
    fig.tight_layout()
    fig.savefig(qc_dir / "peak_intensity_histogram.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    times_minutes = simulation_times() / 60.0
    for storm in representative_storms(storms):
        ax.plot(
            times_minutes,
            storm.candidate.intensity_mm_per_hour,
            label=f"{storm.split}: {storm.storm_id}",
        )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Rainfall intensity (mm/hour)")
    ax.set_title("Representative Phase 2 hyetographs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(qc_dir / "representative_hyetographs.png")
    plt.close(fig)

    expected_plots = (
        "depth_vs_duration_by_split.png",
        "peak_ratio_vs_depth.png",
        "depth_histogram.png",
        "duration_histogram.png",
        "peak_intensity_histogram.png",
        "representative_hyetographs.png",
    )
    missing = [name for name in expected_plots if not (qc_dir / name).is_file()]
    if missing:
        raise ValueError(f"QC plots were not created: {missing}")


def validate_ensemble(
    storms: Sequence[SelectedStorm],
    citycat_dir: Path,
    expected_split_counts: dict[str, int],
) -> dict[str, object]:
    """Validate cross-ensemble uniqueness, splits, exclusions, and numbering."""

    parameter_keys: set[tuple[float, int, int]] = set()
    series_hashes: set[str] = set()
    for storm in storms:
        validate_storm(storm.candidate)
        parameter_key = (
            round(storm.candidate.total_depth_mm, 12),
            storm.candidate.active_duration_s,
            storm.candidate.peak_time_s,
        )
        if parameter_key in parameter_keys:
            raise ValueError(f"Duplicate rainfall parameter combination: {storm.storm_id}")
        if storm.candidate.rainfall_series_sha256 in series_hashes:
            raise ValueError(f"Duplicate rainfall-series hash: {storm.storm_id}")
        parameter_keys.add(parameter_key)
        series_hashes.add(storm.candidate.rainfall_series_sha256)

    expected_indices = list(range(1, len(storms) + 1))
    actual_indices = [storm.rainfall_index for storm in storms]
    if actual_indices != expected_indices:
        raise ValueError("Rainfall indices are not continuous and sequential from 1")
    expected_files = [citycat_dir / f"Rainfall_Data_{index}.txt" for index in expected_indices]
    if any(not path.is_file() for path in expected_files):
        raise ValueError("At least one sequential CityCAT rainfall file is missing")
    discovered_files = sorted(citycat_dir.glob("Rainfall_Data_*.txt"))
    if {path.resolve() for path in discovered_files} != {path.resolve() for path in expected_files}:
        raise ValueError("CityCAT file numbering is not a continuous 1--N sequence")

    observed_split_counts = {
        split: sum(storm.split == split for storm in storms) for split in SPLIT_NAMES
    }
    if observed_split_counts != expected_split_counts:
        raise ValueError(
            f"Split counts do not match targets: observed={observed_split_counts}, "
            f"expected={expected_split_counts}"
        )
    if len(storms) == 100 and observed_split_counts != {
        "train": 70,
        "validation": 15,
        "test": 15,
    }:
        raise ValueError("A 100-storm ensemble must use an exact 70/15/15 split")
    return {
        "all_storms_passed": True,
        "record_count_per_storm": NUMBER_OF_RECORDS,
        "timestamps_passed": True,
        "non_negative_finite_intensity_passed": True,
        "zero_tail_passed": True,
        "unique_peak_passed": True,
        "depth_tolerance_mm": DEPTH_TOLERANCE_MM,
        "depth_integration_passed": True,
        "intensity_cap_passed": True,
        "citycat_unit_conversion_passed": True,
        "unique_parameter_combinations_passed": True,
        "unique_rainfall_hashes_passed": True,
        "phase1_benchmarks_excluded": True,
        "continuous_file_numbering_passed": True,
        "split_counts_passed": True,
        "observed_split_counts": observed_split_counts,
    }


def write_manifest(
    path: Path,
    seed: int,
    requested_count: int,
    storms: Sequence[SelectedStorm],
    validation_results: dict[str, object],
) -> None:
    """Write generation settings, exclusions, split counts, and validation results."""

    split_counts = {
        split: sum(storm.split == split for storm in storms) for split in SPLIT_NAMES
    }
    manifest = {
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_storm_count": requested_count,
        "generated_storm_count": len(storms),
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "parameter_limits": {
            "total_depth_mm": [MIN_DEPTH_MM, MAX_DEPTH_MM],
            "active_duration_seconds": [MIN_DURATION_S, MAX_DURATION_S],
            "peak_time_ratio": [MIN_PEAK_RATIO, MAX_PEAK_RATIO],
        },
        "maximum_peak_intensity_mm_per_hour": MAX_PEAK_MM_PER_HOUR,
        "simulation_horizon_seconds": SIMULATION_HORIZON_S,
        "timestep_seconds": TIMESTEP_S,
        "citycat_intensity_units": "metres per second",
        "excluded_phase1_benchmark_combinations": [
            {
                "total_depth_mm": depth_mm,
                "active_duration_seconds": duration_s,
                "peak_time_seconds": peak_time_s,
            }
            for depth_mm, duration_s, peak_time_s in PHASE1_BENCHMARKS
        ],
        "validation_results": validation_results,
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Generate, write, validate, and summarize the Phase 2 core ensemble."""

    args = parse_args()
    if args.n_storms < 3:
        raise ValueError("--n-storms must be at least 3 so every split is represented")
    citycat_dir, csv_dir, qc_dir = prepare_output_directories(
        args.output_dir,
        args.overwrite,
    )
    output_dir = citycat_dir.parent
    rng = np.random.default_rng(args.seed)
    candidates = generate_candidate_pool(args.n_storms, rng)
    selected_candidates = greedy_maximin_select(candidates, args.n_storms, rng)
    targets = split_targets(args.n_storms)
    assignments = assign_space_filling_splits(selected_candidates, targets)
    storms = [
        SelectedStorm(
            rainfall_index=index,
            storm_id=f"stockbridge_phase2_core_{index:03d}",
            split=assignments[index - 1],
            candidate=candidate,
        )
        for index, candidate in enumerate(selected_candidates, start=1)
    ]

    for storm in storms:
        validate_storm(storm.candidate)
        citycat_path = citycat_dir / f"Rainfall_Data_{storm.rainfall_index}.txt"
        csv_path = csv_dir / f"Rainfall_Data_{storm.rainfall_index}.csv"
        write_citycat_file(citycat_path, storm)
        validate_citycat_file(citycat_path, storm)
        write_timeseries_csv(csv_path, storm)
        if not csv_path.is_file():
            raise ValueError(f"Rainfall CSV was not created: {csv_path}")

    validation_results = validate_ensemble(storms, citycat_dir, targets)
    summary_path = output_dir / "phase2_core_rainfall_summary.csv"
    manifest_path = output_dir / "phase2_core_manifest.json"
    write_summary_csv(summary_path, storms)
    create_qc_plots(qc_dir, storms)
    write_manifest(
        manifest_path,
        args.seed,
        args.n_storms,
        storms,
        validation_results,
    )
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError("Summary CSV or manifest JSON was not created")
    print(
        f"Generated and validated {len(storms)} Phase 2 storms in {output_dir} "
        f"with splits {targets}"
    )


if __name__ == "__main__":
    main()
