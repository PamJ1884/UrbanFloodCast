#!/usr/bin/env python3
"""Batch-convert Stockbridge Phase 2 CityCAT results with the existing converter."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPECTED_EVENT_COUNT = 100
EXPECTED_RSL_COUNT = 25
EXPECTED_TIMESTEPS = tuple(range(25))
EXPECTED_TENSOR_SHAPE = (305, 326, 25, 5)
EXPECTED_ACTIVE_CELLS = 32249
EXPECTED_INACTIVE_CELLS = 67181
EXPECTED_SPLIT_COUNTS = {"train": 70, "validation": 15, "test": 15}
SPLIT_DIRECTORY_MAP = {"train": "train", "validation": "valid", "test": "test"}
REQUIRED_SUMMARY_FIELDS = {"rainfall_index", "storm_id", "split"}
CHANNEL_NAMES = ("H", "Vx", "Vy", "rainfall", "DEM")


@dataclass(frozen=True)
class EventMetadata:
    """Summary metadata for one continuous rainfall index."""

    rainfall_index: int
    storm_id: str
    source_split: str


@dataclass(frozen=True)
class EventPlan:
    """Resolved source and destination paths for one conversion."""

    metadata: EventMetadata
    output_split: str
    rsl_dir: Path
    rainfall_path: Path
    output_path: Path


@dataclass(frozen=True)
class TensorStatistics:
    """Validated tensor structure and channel statistics."""

    output_size_bytes: int
    tensor_shape: list[int]
    dtype: str
    active_cells: int
    inactive_cells: int
    barrier_elevation: float
    h_min: float
    h_max: float
    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float
    rainfall_min: float
    rainfall_max: float
    dem_min: float
    dem_max: float


@dataclass(frozen=True)
class ManifestRecord:
    """One converted or resumed event manifest entry."""

    rainfall_index: int
    storm_id: str
    source_split: str
    output_split: str
    status: str
    rsl_directory: str
    rainfall_path: str
    output_pt_path: str
    output_size_bytes: int
    tensor_shape: list[int]
    dtype: str
    active_cells: int
    inactive_cells: int
    barrier_elevation: float
    h_min: float
    h_max: float
    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float
    rainfall_min: float
    rainfall_max: float
    dem_min: float
    dem_max: float
    conversion_duration_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse batch paths, converter options, and execution controls."""

    parser = argparse.ArgumentParser(
        description="Convert Stockbridge Phase 2 CityCAT simulations to DNO PT tensors."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing all R<N>C1_SurfaceMaps folders and rainfall files",
    )
    parser.add_argument(
        "--dem-path",
        type=Path,
        required=True,
        help="Path to the georeferenced Stockbridge DEM GeoTIFF",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        required=True,
        help="Path to phase2_core_rainfall_summary.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for train, valid, test, and manifest outputs",
    )
    parser.add_argument(
        "--converter-path",
        type=Path,
        default=Path(__file__).with_name("convert_citycat_rsl_to_pt.py"),
        help="Existing single-event converter (default: beside this script)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First CityCAT timestep index (default: %(default)s)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=24,
        help="Last CityCAT timestep index (default: %(default)s)",
    )
    parser.add_argument(
        "--rain-alignment",
        choices=("future", "legacy"),
        default="future",
        help="Rainfall alignment passed to the converter (default: %(default)s)",
    )
    parser.add_argument(
        "--expected-step-seconds",
        type=int,
        default=300,
        help="Expected CityCAT timestep in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert and replace existing PT tensors",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight checks and print the conversion plan only",
    )
    parser.add_argument(
        "--event-indexes",
        type=int,
        nargs="+",
        help="Optional rainfall indices to convert (default: all 1 through 100)",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve required inputs, converter, and output root."""

    results_dir = args.results_dir.expanduser().resolve()
    dem_path = args.dem_path.expanduser().resolve()
    summary_path = args.summary_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    converter_path = args.converter_path.expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Stockbridge results directory not found: {results_dir}")
    for label, path in (
        ("DEM", dem_path),
        ("Phase 2 summary", summary_path),
        ("RSL-to-PT converter", converter_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output root exists but is not a directory: {output_root}")
    return results_dir, dem_path, summary_path, output_root, converter_path


def validate_options(args: argparse.Namespace) -> list[int]:
    """Validate fixed tensor-contract options and selected event indices."""

    if args.start_index != 0 or args.end_index != 24:
        raise ValueError(
            "Stockbridge Phase 2 tensor validation requires --start-index 0 and --end-index 24"
        )
    if args.expected_step_seconds <= 0:
        raise ValueError("--expected-step-seconds must be positive")
    selected = list(
        args.event_indexes
        if args.event_indexes is not None
        else range(1, EXPECTED_EVENT_COUNT + 1)
    )
    if not selected:
        raise ValueError("--event-indexes must contain at least one rainfall index")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate values supplied with --event-indexes: {selected}")
    invalid = sorted(index for index in selected if not 1 <= index <= EXPECTED_EVENT_COUNT)
    if invalid:
        raise ValueError(f"Event indices must lie between 1 and 100: {invalid}")
    return sorted(selected)


def require_summary_fields(fieldnames: Sequence[str] | None, summary_path: Path) -> None:
    """Require the three summary fields used by the batch orchestrator."""

    missing = sorted(REQUIRED_SUMMARY_FIELDS - set(fieldnames or []))
    if missing:
        raise ValueError(f"Required summary fields missing from {summary_path}: {missing}")


def read_summary(summary_path: Path) -> dict[int, EventMetadata]:
    """Read and validate exactly 100 unique, continuously indexed split records."""

    events: dict[int, EventMetadata] = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_summary_fields(reader.fieldnames, summary_path)
        for line_number, row in enumerate(reader, start=2):
            raw_index = (row.get("rainfall_index") or "").strip()
            try:
                rainfall_index = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid rainfall_index at line {line_number} of {summary_path}: "
                    f"{raw_index!r}"
                ) from exc
            if rainfall_index in events:
                raise ValueError(
                    f"Duplicate rainfall_index {rainfall_index} in {summary_path}"
                )
            storm_id = (row.get("storm_id") or "").strip()
            source_split = (row.get("split") or "").strip()
            if not storm_id:
                raise ValueError(
                    f"Empty storm_id at line {line_number} of {summary_path}"
                )
            if source_split not in SPLIT_DIRECTORY_MAP:
                raise ValueError(
                    f"Invalid split at line {line_number} of {summary_path}: "
                    f"{source_split!r}"
                )
            events[rainfall_index] = EventMetadata(
                rainfall_index=rainfall_index,
                storm_id=storm_id,
                source_split=source_split,
            )

    expected_indices = set(range(1, EXPECTED_EVENT_COUNT + 1))
    missing = sorted(expected_indices - set(events))
    extra = sorted(set(events) - expected_indices)
    if len(events) != EXPECTED_EVENT_COUNT or missing or extra:
        raise ValueError(
            f"Summary rainfall indices must be continuous from 1 to 100: "
            f"rows={len(events)}, missing={missing}, extra={extra}"
        )
    split_counts = {
        split: sum(event.source_split == split for event in events.values())
        for split in SPLIT_DIRECTORY_MAP
    }
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Summary split counts must be exactly 70/15/15: observed={split_counts}"
        )
    return events


def validate_event_sources(
    results_dir: Path,
    event_index: int,
    expected_step_seconds: int,
) -> tuple[Path, Path]:
    """Validate one event's 25 continuous RSL files and rainfall input."""

    rsl_dir = results_dir / f"R{event_index}C1_SurfaceMaps"
    if not rsl_dir.is_dir():
        raise FileNotFoundError(f"RSL directory missing for R{event_index}: {rsl_dir}")
    rsl_files = sorted(
        (path for path in rsl_dir.glob("*.rsl") if path.is_file()),
        key=lambda path: path.name,
    )
    if len(rsl_files) != EXPECTED_RSL_COUNT:
        raise ValueError(
            f"R{event_index} must contain exactly {EXPECTED_RSL_COUNT} RSL files, "
            f"found {len(rsl_files)} in {rsl_dir}"
        )
    pattern = re.compile(
        rf"^R{event_index}_C1_T(?P<timestep>\d+)_(?P<minutes>\d+)min\.rsl$"
    )
    timestep_minutes: dict[int, int] = {}
    for path in rsl_files:
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected RSL filename for R{event_index}: {path.name}")
        timestep = int(match.group("timestep"))
        minutes = int(match.group("minutes"))
        if timestep in timestep_minutes:
            raise ValueError(f"Duplicate RSL timestep T{timestep} for R{event_index}")
        if minutes * 60 != timestep * expected_step_seconds:
            raise ValueError(
                f"RSL filename time is inconsistent for R{event_index} T{timestep}: "
                f"{minutes}min versus expected {timestep * expected_step_seconds}s"
            )
        timestep_minutes[timestep] = minutes
    if tuple(sorted(timestep_minutes)) != EXPECTED_TIMESTEPS:
        missing = sorted(set(EXPECTED_TIMESTEPS) - set(timestep_minutes))
        extra = sorted(set(timestep_minutes) - set(EXPECTED_TIMESTEPS))
        raise ValueError(
            f"R{event_index} RSL timesteps must be continuous T0--T24: "
            f"missing={missing}, extra={extra}"
        )

    rainfall_path = results_dir / f"Rainfall_Data_{event_index}.txt"
    if not rainfall_path.is_file():
        raise FileNotFoundError(
            f"Rainfall input missing for R{event_index}: {rainfall_path}"
        )
    return rsl_dir, rainfall_path


def output_filename(event_index: int, start_index: int, end_index: int) -> str:
    """Return the zero-padded Stockbridge tensor filename."""

    return (
        f"R{event_index:03d}_stockbridge_"
        f"T{start_index:03d}_T{end_index:03d}.pt"
    )


def build_plans(
    events: dict[int, EventMetadata],
    selected_indexes: Sequence[int],
    results_dir: Path,
    output_root: Path,
    start_index: int,
    end_index: int,
    expected_step_seconds: int,
) -> list[EventPlan]:
    """Preflight all sources, then return collision-free selected-event plans."""

    source_paths: dict[int, tuple[Path, Path]] = {}
    for event_index in range(1, EXPECTED_EVENT_COUNT + 1):
        source_paths[event_index] = validate_event_sources(
            results_dir,
            event_index,
            expected_step_seconds,
        )

    plans: list[EventPlan] = []
    output_paths: set[Path] = set()
    for event_index in selected_indexes:
        metadata = events[event_index]
        output_split = SPLIT_DIRECTORY_MAP[metadata.source_split]
        output_path = (
            output_root
            / output_split
            / output_filename(event_index, start_index, end_index)
        )
        if output_path in output_paths:
            raise ValueError(f"Multiple events resolve to the same output path: {output_path}")
        output_paths.add(output_path)
        rsl_dir, rainfall_path = source_paths[event_index]
        plans.append(
            EventPlan(
                metadata=metadata,
                output_split=output_split,
                rsl_dir=rsl_dir,
                rainfall_path=rainfall_path,
                output_path=output_path,
            )
        )
    return plans


def print_preflight_summary(
    plans: Sequence[EventPlan],
    results_dir: Path,
    dem_path: Path,
    converter_path: Path,
    output_root: Path,
) -> None:
    """Print validated source and selected conversion counts."""

    selected_split_counts = {
        split: sum(plan.output_split == split for plan in plans)
        for split in SPLIT_DIRECTORY_MAP.values()
    }
    print(
        f"Preflight passed: 100 events, 2500 continuous RSL files, split counts "
        f"train=70 validation=15 test=15"
    )
    print(f"Results: {results_dir}")
    print(f"DEM: {dem_path}")
    print(f"Converter: {converter_path}")
    print(f"Output root: {output_root}")
    print(
        f"Selected conversions: {len(plans)} "
        f"(train={selected_split_counts['train']}, "
        f"valid={selected_split_counts['valid']}, "
        f"test={selected_split_counts['test']})"
    )


def converter_command(
    plan: EventPlan,
    converter_path: Path,
    dem_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build one subprocess invocation of the existing converter."""

    return [
        sys.executable,
        str(converter_path),
        "--rsl-dir",
        str(plan.rsl_dir),
        "--dem-path",
        str(dem_path),
        "--rainfall-path",
        str(plan.rainfall_path),
        "--output-path",
        str(plan.output_path),
        "--start-index",
        str(args.start_index),
        "--end-index",
        str(args.end_index),
        "--rain-alignment",
        args.rain_alignment,
        "--expected-step-seconds",
        str(args.expected_step_seconds),
    ]


def print_dry_run(
    plans: Sequence[EventPlan],
    converter_path: Path,
    dem_path: Path,
    args: argparse.Namespace,
) -> None:
    """Print every planned conversion without creating or replacing outputs."""

    for plan in plans:
        if plan.output_path.exists() and not args.overwrite:
            action = "validate-and-skip-if-valid"
        elif plan.output_path.exists():
            action = "reconvert-and-replace"
        else:
            action = "convert"
        command = converter_command(plan, converter_path, dem_path, args)
        print(
            f"DRY RUN R{plan.metadata.rainfall_index:03d} "
            f"{plan.output_split}: {action}\n  {shlex.join(command)}"
        )


def validate_tensor(output_path: Path) -> TensorStatistics:
    """Load one PT tensor on CPU and enforce the Stockbridge DNO contract."""

    import torch

    if not output_path.is_file():
        raise FileNotFoundError(f"PT tensor not found: {output_path}")
    output_size_bytes = output_path.stat().st_size
    if output_size_bytes <= 0:
        raise ValueError(f"PT tensor file is empty: {output_path}")
    try:
        tensor = torch.load(output_path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not load PT tensor {output_path}: {exc}") from exc
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"PT object is not a torch.Tensor: {output_path} contains {type(tensor).__name__}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(f"Tensor dtype must be torch.float32 in {output_path}: {tensor.dtype}")
    if tuple(tensor.shape) != EXPECTED_TENSOR_SHAPE:
        raise ValueError(
            f"Tensor shape must be {EXPECTED_TENSOR_SHAPE} in {output_path}: "
            f"{tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"Tensor contains non-finite values: {output_path}")

    h_values = tensor[..., 0]
    vx_values = tensor[..., 1]
    vy_values = tensor[..., 2]
    rainfall_values = tensor[..., 3]
    dem_values = tensor[..., 4]
    first_dem = dem_values[..., 0]
    if not torch.equal(dem_values, first_dem.unsqueeze(-1).expand_as(dem_values)):
        raise ValueError(f"DEM channel is not constant through time: {output_path}")
    barrier_elevation = float(first_dem.max().item())
    if not math.isfinite(barrier_elevation):
        raise ValueError(f"Barrier elevation is non-finite: {output_path}")
    inactive_mask = first_dem == barrier_elevation
    inactive_cells = int(torch.count_nonzero(inactive_mask).item())
    active_cells = int(inactive_mask.numel() - inactive_cells)
    if active_cells != EXPECTED_ACTIVE_CELLS or inactive_cells != EXPECTED_INACTIVE_CELLS:
        raise ValueError(
            f"Hydraulic cell counts are invalid in {output_path}: active={active_cells} "
            f"(expected {EXPECTED_ACTIVE_CELLS}), inactive={inactive_cells} "
            f"(expected {EXPECTED_INACTIVE_CELLS})"
        )
    inactive_time_mask = inactive_mask.unsqueeze(-1).expand_as(h_values)
    for channel_name, channel_values in (
        ("H", h_values),
        ("Vx", vx_values),
        ("Vy", vy_values),
    ):
        if bool(torch.any(channel_values[inactive_time_mask] != 0.0).item()):
            raise ValueError(
                f"{channel_name} is non-zero on inactive hydraulic cells: {output_path}"
            )
    if bool(torch.any(h_values < 0.0).item()):
        raise ValueError(f"H channel contains negative depths: {output_path}")

    channel_ranges = []
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        channel = tensor[..., channel_index]
        channel_minimum = float(channel.min().item())
        channel_maximum = float(channel.max().item())
        if not math.isfinite(channel_minimum) or not math.isfinite(channel_maximum):
            raise ValueError(
                f"{channel_name} channel extrema are non-finite: {output_path}"
            )
        channel_ranges.append((channel_minimum, channel_maximum))
    return TensorStatistics(
        output_size_bytes=output_size_bytes,
        tensor_shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        active_cells=active_cells,
        inactive_cells=inactive_cells,
        barrier_elevation=barrier_elevation,
        h_min=channel_ranges[0][0],
        h_max=channel_ranges[0][1],
        vx_min=channel_ranges[1][0],
        vx_max=channel_ranges[1][1],
        vy_min=channel_ranges[2][0],
        vy_max=channel_ranges[2][1],
        rainfall_min=channel_ranges[3][0],
        rainfall_max=channel_ranges[3][1],
        dem_min=channel_ranges[4][0],
        dem_max=channel_ranges[4][1],
    )


def make_manifest_record(
    plan: EventPlan,
    status: str,
    statistics: TensorStatistics,
    duration_seconds: float,
) -> ManifestRecord:
    """Combine event paths, status, timing, and tensor statistics."""

    return ManifestRecord(
        rainfall_index=plan.metadata.rainfall_index,
        storm_id=plan.metadata.storm_id,
        source_split=plan.metadata.source_split,
        output_split=plan.output_split,
        status=status,
        rsl_directory=str(plan.rsl_dir),
        rainfall_path=str(plan.rainfall_path),
        output_pt_path=str(plan.output_path),
        output_size_bytes=statistics.output_size_bytes,
        tensor_shape=statistics.tensor_shape,
        dtype=statistics.dtype,
        active_cells=statistics.active_cells,
        inactive_cells=statistics.inactive_cells,
        barrier_elevation=statistics.barrier_elevation,
        h_min=statistics.h_min,
        h_max=statistics.h_max,
        vx_min=statistics.vx_min,
        vx_max=statistics.vx_max,
        vy_min=statistics.vy_min,
        vy_max=statistics.vy_max,
        rainfall_min=statistics.rainfall_min,
        rainfall_max=statistics.rainfall_max,
        dem_min=statistics.dem_min,
        dem_max=statistics.dem_max,
        conversion_duration_seconds=duration_seconds,
    )


def process_plan(
    plan: EventPlan,
    converter_path: Path,
    dem_path: Path,
    args: argparse.Namespace,
) -> ManifestRecord:
    """Validate-and-skip or sequentially convert and validate one event."""

    event_start = time.monotonic()
    event_label = f"R{plan.metadata.rainfall_index:03d}"
    print(f"Processing {event_label} -> {plan.output_split}")
    if plan.output_path.exists() and not args.overwrite:
        try:
            statistics = validate_tensor(plan.output_path)
        except Exception as exc:
            raise ValueError(
                f"Existing output for {event_label} is invalid and cannot be skipped: {exc}. "
                f"Remove it or rerun with --overwrite to replace it."
            ) from exc
        duration = time.monotonic() - event_start
        print(f"Skipped {event_label}: existing tensor valid ({duration:.2f}s)")
        return make_manifest_record(plan, "skipped", statistics, duration)

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = converter_command(plan, converter_path, dem_path, args)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Conversion failed for {event_label} with exit code {exc.returncode}: "
            f"{shlex.join(command)}"
        ) from exc
    statistics = validate_tensor(plan.output_path)
    duration = time.monotonic() - event_start
    print(f"Converted {event_label}: tensor valid ({duration:.2f}s)")
    return make_manifest_record(plan, "converted", statistics, duration)


def prepare_output_directories(output_root: Path) -> None:
    """Create and validate the fixed train/valid/test output structure."""

    for directory in (
        output_root,
        output_root / "train",
        output_root / "valid",
        output_root / "test",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValueError(f"Output directory is invalid: {directory}")


def manifest_csv_row(record: ManifestRecord) -> dict[str, Any]:
    """Flatten list-valued manifest data for CSV output."""

    row = asdict(record)
    row["tensor_shape"] = "x".join(str(value) for value in record.tensor_shape)
    return row


def write_manifests(
    output_root: Path,
    records: Sequence[ManifestRecord],
    converter_path: Path,
    dem_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Write per-event CSV records and batch-level JSON metadata."""

    if not records:
        raise ValueError("Cannot write empty PT conversion manifests")
    csv_path = output_root / "stockbridge_phase2_pt_manifest.csv"
    json_path = output_root / "stockbridge_phase2_pt_manifest.json"
    csv_rows = [manifest_csv_row(record) for record in records]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(csv_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    converted_count = sum(record.status == "converted" for record in records)
    skipped_count = sum(record.status == "skipped" for record in records)
    output_split_counts = {
        split: sum(record.output_split == split for record in records)
        for split in SPLIT_DIRECTORY_MAP.values()
    }
    manifest = {
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "converter_path": str(converter_path),
        "dem_path": str(dem_path),
        "summary_path": str(summary_path),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "rain_alignment": args.rain_alignment,
        "expected_timestep_seconds": args.expected_step_seconds,
        "total_converted_count": converted_count,
        "total_skipped_count": skipped_count,
        "train_count": output_split_counts["train"],
        "valid_count": output_split_counts["valid"],
        "test_count": output_split_counts["test"],
        "validation_passed": True,
        "events": [asdict(record) for record in records],
    }
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if not csv_path.is_file() or not json_path.is_file():
        raise ValueError("PT conversion manifest files were not created")
    return csv_path, json_path


def print_final_summary(records: Sequence[ManifestRecord], elapsed_seconds: float) -> None:
    """Print conversion counts, elapsed time, and approximate PT storage."""

    converted_count = sum(record.status == "converted" for record in records)
    skipped_count = sum(record.status == "skipped" for record in records)
    total_size_bytes = sum(record.output_size_bytes for record in records)
    print(
        f"Completed: converted={converted_count}, skipped={skipped_count}, "
        f"total_elapsed={elapsed_seconds:.2f}s, "
        f"PT_size={total_size_bytes / (1024**3):.3f} GiB"
    )


def main() -> None:
    """Preflight, convert sequentially, validate, and write batch manifests."""

    args = parse_args()
    selected_indexes = validate_options(args)
    results_dir, dem_path, summary_path, output_root, converter_path = resolve_paths(args)
    events = read_summary(summary_path)
    plans = build_plans(
        events,
        selected_indexes,
        results_dir,
        output_root,
        args.start_index,
        args.end_index,
        args.expected_step_seconds,
    )
    print_preflight_summary(
        plans,
        results_dir,
        dem_path,
        converter_path,
        output_root,
    )
    if args.dry_run:
        print_dry_run(plans, converter_path, dem_path, args)
        print("Dry run complete; no converters were executed and no outputs were written.")
        return

    prepare_output_directories(output_root)
    batch_start = time.monotonic()
    records = [
        process_plan(plan, converter_path, dem_path, args)
        for plan in plans
    ]
    write_manifests(
        output_root,
        records,
        converter_path,
        dem_path,
        summary_path,
        args,
    )
    print_final_summary(records, time.monotonic() - batch_start)


if __name__ == "__main__":
    main()
