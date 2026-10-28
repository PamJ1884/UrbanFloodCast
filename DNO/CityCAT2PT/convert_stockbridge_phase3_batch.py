#!/usr/bin/env python3
"""Sequential canonical Phase 3 CityCAT-to-DNO conversion with safe resume."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


DATA = Path.home() / "flood/02_data/CityCAT_Winchester"
MASTER = DATA / "manifests/stockbridge_phase2plus3_300_v01/master_manifest_R001_R300_v01.csv"
REFERENCE_PT = DATA / "pt/stockbridge_phase2_core_v01/train/R001_stockbridge_T000_T024.pt"
REFERENCE_JSON = REFERENCE_PT.with_suffix(".pt.json")
DEM = Path.home() / "flood/10_qgis_projects/Winchester/02_rasters/stockbridge_dem_subcatchment_candidate_v01.tif"
DEM_SHA256 = "d0aa9cbeb0664ab13b269857e9ab8b24e4bb58468fef2488801683e9568a5785"
CONVERTER = Path(__file__).with_name("convert_citycat_rsl_to_pt.py")
OUTPUT = DATA / "pt/stockbridge_phase3_core_v01"
STAGING = DATA / "staging/stockbridge_phase3_core_v01"
DEFAULT_SOURCE = (Path(f"/run/user/{os.getuid()}/gvfs") /
                  "smb-share:server=campus.ncl.ac.uk,share=rdw" /
                  "pamela-phd-flood/02_data/CityCAT_Winchester/results/stockbridge_phase3_v01")
RAINFALL_ROOT = DATA / "rainfall/stockbridge_phase3_v01/rainfall_data_phase3"
EXPECTED_IDS = {f"R{index:03d}" for index in range(101, 301)}
TIMES = list(range(0, 7201, 300))
INDICES = list(range(25))
SOURCE_TIMES = [None, *TIMES[:-1]]
SHAPE = [305, 326, 25, 5]
CHANNELS = ["H", "U", "V", "rainfall", "DEM"]
SPLITS = {"train": 140, "valid": 30, "test": 30}
STATIC_KEYS = ("raster_width", "raster_height", "raster_bounds", "raster_resolution",
               "affine_transform", "crs", "hydraulic_active_cell_count",
               "hydraulic_inactive_cell_count", "hydraulic_total_cell_count",
               "barrier_offset", "barrier_elevation")
CSV_FIELDS = ("event_id", "event_index", "phase_sequence", "backbone_id", "family",
              "profile_subtype", "split", "source_citycat_dir", "rainfall_path",
              "rainfall_sha256", "dem_path", "dem_sha256", "output_pt_path",
              "output_sidecar_path", "tensor_shape", "dtype", "active_cells",
              "inactive_cells", "barrier_elevation", "stage_seconds",
              "conversion_seconds", "validation_seconds", "total_seconds",
              "status", "error")


@dataclass
class Event:
    event_id: str
    event_index: int
    phase_sequence: int
    split: str
    backbone_id: str
    family: str
    profile_subtype: str
    rainfall_path: Path
    rainfall_sha256: str
    source_dir: Path | None = None
    rsl_files: list[Path] | None = None
    rain_values: list[float] | None = None
    output_pt: Path | None = None
    staged_dir: Path | None = None
    action: str = "NOT_SELECTED"
    existing_stats: dict | None = None
    existing_validation_seconds: float = 0.0


class BatchError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BatchError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE,
                        help="Mounted Phase 3 CityCAT results root")
    parser.add_argument("--event", action="append", default=[], metavar="R101",
                        help="Select one event; repeatable")
    parser.add_argument("--backbone", action="append", default=[], metavar="B01",
                        help="Select one backbone; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Preflight and plan without writing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacement of invalid or incomplete PT/sidecar pairs")
    parser.add_argument("--replace-valid", action="store_true",
                        help="With --overwrite, also reconvert validated existing pairs")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Retain staged RSL files after a successful conversion")
    args = parser.parse_args()
    if args.replace_valid and not args.overwrite:
        parser.error("--replace-valid requires --overwrite")
    return args


def load_manifest() -> dict[str, Event]:
    needed = {"event_id", "event_index", "phase", "phase_sequence", "split_300",
              "backbone_id", "family", "profile_subtype", "canonical_rainfall_filename",
              "canonical_rainfall_path", "rainfall_sha256", "n_timesteps",
              "timestep_seconds", "simulation_horizon_seconds"}
    events: dict[str, Event] = {}
    sequences: set[int] = set()
    with MASTER.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(needed <= set(reader.fieldnames or []),
                f"Master manifest missing columns: {sorted(needed - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            if row["phase"].strip() != "phase3":
                continue
            event_id = row["event_id"].strip()
            require(event_id in EXPECTED_IDS and event_id not in events,
                    f"Invalid or duplicate Phase 3 event at manifest line {line}: {event_id!r}")
            try:
                index = int(row["event_index"])
                sequence = int(row["phase_sequence"])
                n_times = int(row["n_timesteps"])
                step = int(row["timestep_seconds"])
                horizon = int(row["simulation_horizon_seconds"])
            except ValueError as exc:
                raise BatchError(f"Invalid numeric Phase 3 metadata at manifest line {line}") from exc
            require(index == int(event_id[1:]) and sequence not in sequences,
                    f"{event_id}: event_index or phase_sequence invalid/duplicate")
            require((n_times, step, horizon) == (25, 300, 7200),
                    f"{event_id}: rainfall timestep metadata differs from 25/300/7200")
            sequences.add(sequence)
            split = row["split_300"].strip()
            require(split in SPLITS, f"{event_id}: invalid frozen split {split!r}")
            backbone = row["backbone_id"].strip()
            require(re.fullmatch(r"B\d{2}", backbone) is not None,
                    f"{event_id}: invalid backbone {backbone!r}")
            family = row["family"].strip()
            subtype = row["profile_subtype"].strip()
            require(bool(family and subtype), f"{event_id}: missing family/subtype")
            filename = row["canonical_rainfall_filename"].strip()
            rainfall = Path(row["canonical_rainfall_path"].strip()).expanduser()
            require(filename == f"Rainfall_Data_{index}.txt" and rainfall.is_absolute() and
                    rainfall.name == filename and rainfall.resolve().parent == RAINFALL_ROOT.resolve(),
                    f"{event_id}: canonical rainfall filename/path is invalid")
            rain_hash = row["rainfall_sha256"].strip().lower()
            require(re.fullmatch(r"[0-9a-f]{64}", rain_hash) is not None,
                    f"{event_id}: invalid rainfall SHA256 in manifest")
            events[event_id] = Event(event_id, index, sequence, split, backbone,
                                     family, subtype, rainfall, rain_hash)
    require(set(events) == EXPECTED_IDS,
            f"Manifest Phase 3 IDs differ from R101–R300: count={len(events)}, "
            f"missing={sorted(EXPECTED_IDS - set(events))}")
    require(sequences == set(range(1, 201)),
            f"Phase 3 phase_sequence must be 1–200, found {len(sequences)} unique")
    split_counts = {split: sum(event.split == split for event in events.values())
                    for split in SPLITS}
    require(split_counts == SPLITS, f"Frozen split counts differ: {split_counts}")
    backbones: dict[str, list[Event]] = {}
    for event in events.values():
        backbones.setdefault(event.backbone_id, []).append(event)
    require(len(backbones) == 40, f"Expected 40 backbones, found {len(backbones)}")
    for backbone, members in backbones.items():
        require(len(members) == 5 and len({member.split for member in members}) == 1,
                f"{backbone}: expected five events in exactly one frozen split")
    return events


def choose_events(events: dict[str, Event], args: argparse.Namespace) -> list[Event]:
    requested_events = set(args.event)
    requested_backbones = set(args.backbone)
    require(len(requested_events) == len(args.event), "Duplicate --event selector")
    require(len(requested_backbones) == len(args.backbone), "Duplicate --backbone selector")
    require(requested_events <= set(events),
            f"Unknown Phase 3 event selector(s): {sorted(requested_events - set(events))}")
    known_backbones = {event.backbone_id for event in events.values()}
    require(requested_backbones <= known_backbones,
            f"Unknown backbone selector(s): {sorted(requested_backbones - known_backbones)}")
    if not requested_events and not requested_backbones:
        selected = list(events.values())
    else:
        selected = [event for event in events.values()
                    if event.event_id in requested_events or event.backbone_id in requested_backbones]
    return sorted(selected, key=lambda event: event.event_index)


def validate_rsl(directory: Path, event_id: str) -> list[Path]:
    require(directory.is_dir(), f"{event_id}: CityCAT source directory missing: {directory}")
    files = [path for path in directory.iterdir() if path.suffix.lower() == ".rsl"]
    require(len(files) == 25, f"{event_id}: expected 25 .rsl files, found {len(files)}")
    pattern = re.compile(rf"{event_id}_C1_T(\d+)_(\d+)min\.rsl")
    by_index: dict[int, Path] = {}
    for path in files:
        require(path.is_file() and not path.is_symlink(), f"{event_id}: invalid RSL: {path}")
        match = pattern.fullmatch(path.name)
        require(match is not None, f"{event_id}: unexpected RSL filename: {path.name}")
        index, minutes = map(int, match.groups())
        require(index not in by_index and minutes == index * 5,
                f"{event_id}: duplicate or mis-timed T{index} ({minutes}min)")
        by_index[index] = path
    require(sorted(by_index) == INDICES,
            f"{event_id}: RSL timesteps differ from T0–T24: {sorted(by_index)}")
    return [by_index[index] for index in INDICES]


def validate_rainfall(event: Event) -> list[float]:
    digest = sha256(event.rainfall_path)
    require(digest == event.rainfall_sha256,
            f"{event.event_id}: rainfall SHA256 differs from master manifest: {digest}")
    records: list[tuple[float, float]] = []
    count_header = None
    for line_no, raw in enumerate(event.rainfall_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0] == "*":
            continue
        if len(fields) == 1 and count_header is None and not records:
            try:
                count_header = int(fields[0])
            except ValueError as exc:
                raise BatchError(f"{event.event_id}: malformed rainfall header line {line_no}") from exc
            continue
        require(len(fields) == 2,
                f"{event.event_id}: malformed rainfall record at line {line_no}")
        try:
            time_seconds, value = map(float, fields)
        except ValueError as exc:
            raise BatchError(f"{event.event_id}: nonnumeric rainfall at line {line_no}") from exc
        require(math.isfinite(time_seconds) and math.isfinite(value) and value >= 0,
                f"{event.event_id}: non-finite or negative rainfall at line {line_no}")
        records.append((time_seconds, value))
    require(count_header == 25 and len(records) == 25,
            f"{event.event_id}: expected 25 declared/numeric rainfall records, got "
            f"{count_header}/{len(records)}")
    require([time for time, _ in records] == TIMES,
            f"{event.event_id}: rainfall times differ from 0,300,...,7200s")
    return [value for _, value in records]


def validate_reference():
    import torch

    require(CONVERTER.is_file(), f"Canonical converter missing: {CONVERTER}")
    reference = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    for key, expected in (("tensor_shape", SHAPE), ("dtype", "torch.float32"),
                          ("channel_order", CHANNELS), ("rainfall_units", "m/s"),
                          ("rainfall_alignment", "future"),
                          ("selected_timestep_indices", INDICES),
                          ("selected_times_seconds", TIMES),
                          ("rainfall_source_times_seconds", SOURCE_TIMES),
                          ("hydraulic_active_cell_count", 32249),
                          ("hydraulic_inactive_cell_count", 67181),
                          ("hydraulic_total_cell_count", 99430),
                          ("barrier_offset", 30.0),
                          ("barrier_elevation", 152.61531829833984)):
        require(reference.get(key) == expected,
                f"Phase 2 reference {key} differs: {reference.get(key)!r}")
    source_dem = reference.get("source_dem_path")
    require(isinstance(source_dem, str) and
            Path(source_dem).expanduser().resolve() == DEM.resolve(),
            "Phase 2 reference source_dem_path differs from canonical DEM")
    dem_sha = sha256(DEM)
    require(dem_sha == DEM_SHA256, f"DEM SHA256 mismatch: {dem_sha}")
    require(REFERENCE_PT.is_file(), f"Phase 2 reference tensor missing: {REFERENCE_PT}")
    tensor = torch.load(REFERENCE_PT, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == SHAPE and
            tensor.dtype == torch.float32, "Phase 2 reference tensor shape/dtype differs")
    reference_dem = tensor[..., 4].clone()
    require(bool(torch.isfinite(reference_dem).all()), "Phase 2 reference DEM is non-finite")
    require(torch.equal(reference_dem, reference_dem[..., :1].expand_as(reference_dem)),
            "Phase 2 reference DEM varies over time")
    barrier_value = torch.tensor(reference["barrier_elevation"], dtype=torch.float32).item()
    inactive = reference_dem[..., 0] == barrier_value
    require((int((~inactive).sum()), int(inactive.sum())) == (32249, 67181),
            "Phase 2 reference active/inactive spatial mask differs")
    return reference, reference_dem, inactive, dem_sha


def validate_output(path: Path, event: Event, reference: dict, reference_dem,
                    inactive) -> dict:
    import torch

    sidecar_path = path.with_suffix(".pt.json")
    require(path.is_file() and sidecar_path.is_file(),
            f"{event.event_id}: PT/sidecar pair missing: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for key in STATIC_KEYS:
        require(sidecar.get(key) == reference.get(key),
                f"{event.event_id}: sidecar {key} differs from Phase 2")
    for key, expected in (("tensor_shape", SHAPE), ("dtype", "torch.float32"),
                          ("channel_order", CHANNELS), ("rainfall_units", "m/s"),
                          ("rainfall_alignment", "future"),
                          ("selected_timestep_indices", INDICES),
                          ("selected_times_seconds", TIMES),
                          ("rainfall_source_times_seconds", SOURCE_TIMES),
                          ("selected_rsl_filenames", [p.name for p in event.rsl_files])):
        require(sidecar.get(key) == expected,
                f"{event.event_id}: sidecar {key} differs")
    require(sidecar.get("source_dem_path") == str(DEM.resolve()) and
            sidecar.get("source_rainfall_path") == str(event.rainfall_path.resolve()) and
            sidecar.get("source_rsl_dir") == str(event.staged_dir.resolve()),
            f"{event.event_id}: sidecar source paths differ")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == SHAPE and
            tensor.dtype == torch.float32, f"{event.event_id}: tensor shape/dtype differs")
    require(bool(torch.isfinite(tensor).all()), f"{event.event_id}: non-finite tensor")
    h, rain, dem = tensor[..., 0], tensor[..., 3], tensor[..., 4]
    require(not bool(torch.any(h < 0)), f"{event.event_id}: negative H")
    require(torch.equal(dem, dem[..., :1].expand_as(dem)),
            f"{event.event_id}: DEM varies over time")
    require(torch.equal(dem, reference_dem),
            f"{event.event_id}: DEM differs pixel-by-pixel from Phase 2")
    require(not bool(torch.any(tensor[..., :3][inactive] != 0)),
            f"{event.event_id}: H/U/V nonzero in inactive cells")
    require(torch.equal(rain, rain[:1, :1, :].expand_as(rain)),
            f"{event.event_id}: rainfall is not spatially uniform")
    expected_rain = torch.tensor([0.0, *event.rain_values[:-1]], dtype=torch.float32)
    require(torch.equal(rain, expected_rain.view(1, 1, 25).expand_as(rain)),
            f"{event.event_id}: rainfall channel differs from future-aligned source")
    return {"tensor_shape": "x".join(map(str, SHAPE)), "dtype": "torch.float32",
            "active_cells": int((~inactive).sum()), "inactive_cells": int(inactive.sum()),
            "barrier_elevation": reference["barrier_elevation"]}


def sidecar_path(pt: Path) -> Path:
    return pt.with_suffix(".pt.json")


def preflight(events: dict[str, Event], selected: list[Event], args: argparse.Namespace,
              reference: dict, reference_dem, inactive) -> None:
    source_root = args.source_root.expanduser().resolve()
    require(source_root.is_dir(), f"CityCAT source root missing: {source_root}")
    for write_root in (OUTPUT.resolve(), STAGING.resolve()):
        require(not write_root.is_relative_to(source_root) and
                not source_root.is_relative_to(write_root),
                f"CityCAT source root overlaps a write root: {source_root}")
    for event in events.values():
        event.source_dir = source_root / f"{event.event_id}C1_SurfaceMaps"
        event.output_pt = OUTPUT / event.split / f"{event.event_id}_stockbridge_T000_T024.pt"
        event.staged_dir = STAGING / f"{event.event_id}C1_SurfaceMaps"
    for event in selected:
        event.rsl_files = validate_rsl(event.source_dir, event.event_id)
        event.rain_values = validate_rainfall(event)
        pt, sidecar = event.output_pt, sidecar_path(event.output_pt)
        require(not pt.is_symlink() and not sidecar.is_symlink(),
                f"{event.event_id}: refusing symlink output")
        pt_exists, sidecar_exists = pt.exists(), sidecar.exists()
        if pt_exists and sidecar_exists:
            start = time.monotonic()
            try:
                event.existing_stats = validate_output(pt, event, reference,
                                                       reference_dem, inactive)
            except Exception as exc:
                if not args.overwrite:
                    raise BatchError(f"{event.event_id}: existing PT/sidecar invalid; "
                                     f"use --overwrite to replace: {exc}") from exc
                event.action = "REPLACE"
            else:
                event.existing_validation_seconds = time.monotonic() - start
                event.action = "REPLACE" if args.replace_valid else "VALID_EXISTING"
        elif pt_exists or sidecar_exists:
            require(args.overwrite,
                    f"{event.event_id}: incomplete PT/sidecar pair; use --overwrite")
            event.action = "REPLACE"
        else:
            event.action = "CONVERT"
        if event.action != "VALID_EXISTING":
            require(not event.staged_dir.is_symlink() and
                    (not event.staged_dir.exists() or event.staged_dir.is_dir()),
                    f"{event.event_id}: invalid/symlink staging directory: "
                    f"{event.staged_dir}")


def check_write_roots() -> None:
    forbidden = (DATA / "pt/stockbridge_phase2_core_v01",
                 DATA / "pt/stockbridge_phase2plus3_300_v01",
                 RAINFALL_ROOT, DATA / "results", MASTER.parent, DEFAULT_SOURCE)
    for root in (OUTPUT, STAGING):
        require(not root.is_symlink(), f"Refusing symlink write root: {root}")
        resolved = root.resolve()
        for protected in forbidden:
            require(not resolved.is_relative_to(protected.resolve()),
                    f"Write root resolves into protected location: {root} -> {resolved}")
        require(not root.exists() or root.is_dir(), f"Write root is not a directory: {root}")
    require(not OUTPUT.resolve().is_relative_to(STAGING.resolve()) and
            not STAGING.resolve().is_relative_to(OUTPUT.resolve()),
            "Output and staging roots overlap")
    for split in SPLITS:
        path = OUTPUT / split
        require(not path.is_symlink() and (not path.exists() or path.is_dir()),
                f"Invalid/symlink output split directory: {path}")


def stage(event: Event) -> None:
    staged = event.staged_dir
    require(not staged.is_symlink(), f"{event.event_id}: refusing symlink staging directory")
    if staged.exists():
        require(staged.is_dir(), f"{event.event_id}: staging path is not a directory")
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for source in event.rsl_files:
        target = staged / source.name
        shutil.copy2(source, target)
        require(target.stat().st_size == source.stat().st_size,
                f"{event.event_id}: staged RSL size differs: {source.name}")
    validate_rsl(staged, event.event_id)


def converter_command(event: Event, temporary_pt: Path) -> list[str]:
    return [sys.executable, str(CONVERTER), "--rsl-dir", str(event.staged_dir),
            "--dem-path", str(DEM), "--rainfall-path", str(event.rainfall_path),
            "--output-path", str(temporary_pt), "--start-index", "0",
            "--end-index", "24", "--rain-alignment", "future",
            "--barrier-offset", "30", "--expected-step-seconds", "300"]


def publish_pair(temporary_pt: Path, final_pt: Path, temporary_dir: Path) -> None:
    pairs = ((temporary_pt, final_pt), (sidecar_path(temporary_pt), sidecar_path(final_pt)))
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for _, destination in pairs:
            if destination.exists():
                backup = temporary_dir / f"backup_{destination.name}"
                os.replace(destination, backup)
                backups[destination] = backup
        for source, destination in pairs:
            os.replace(source, destination)
            installed.append(destination)
    except Exception:
        for destination in installed:
            destination.unlink()
        for destination, backup in backups.items():
            os.replace(backup, destination)
        raise


def blank_record(event: Event, dem_sha: str) -> dict:
    output_pt = OUTPUT / event.split / f"{event.event_id}_stockbridge_T000_T024.pt"
    return {"event_id": event.event_id, "event_index": event.event_index,
            "phase_sequence": event.phase_sequence, "backbone_id": event.backbone_id,
            "family": event.family, "profile_subtype": event.profile_subtype,
            "split": event.split, "source_citycat_dir": str(event.source_dir or ""),
            "rainfall_path": str(event.rainfall_path),
            "rainfall_sha256": event.rainfall_sha256, "dem_path": str(DEM),
            "dem_sha256": dem_sha, "output_pt_path": str(output_pt),
            "output_sidecar_path": str(sidecar_path(output_pt)), "tensor_shape": "",
            "dtype": "", "active_cells": "", "inactive_cells": "",
            "barrier_elevation": "", "stage_seconds": 0.0,
            "conversion_seconds": 0.0, "validation_seconds": 0.0,
            "total_seconds": 0.0, "status": "NOT_SELECTED", "error": ""}


def atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     prefix=f".{path.name}.", dir=path.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def write_reports(records: dict[str, dict], selected: list[Event], dem_sha: str,
                  hashes: dict, started: float) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records[event_id] for event_id in sorted(records))
    atomic_write(OUTPUT / "stockbridge_phase3_pt_manifest.csv", buffer.getvalue())
    selected_records = [records[event.event_id] for event in selected]
    statuses = {status: sum(row["status"] == status for row in selected_records)
                for status in ("CONVERTED", "VALID_EXISTING", "FAILED", "PENDING")}
    durations = [float(row["total_seconds"]) for row in selected_records
                 if row["status"] in ("CONVERTED", "VALID_EXISTING", "FAILED")]
    full_success = (len(selected) == 200 and statuses["PENDING"] == 0 and
                    statuses["FAILED"] == 0)
    summary = {
        "dataset_name": "stockbridge_phase3_core_v01",
        "event_range": "R101-R300", "total_events": 200,
        "selected_event_count": len(selected),
        "not_selected_count": 200 - len(selected),
        "selected_events": [event.event_id for event in selected],
        "execution_scope": "full" if len(selected) == 200 else "subset",
        "status": "COMPLETE" if full_success else
                  "FAILED" if statuses["FAILED"] else "PARTIAL",
        "split_counts": SPLITS,
        "selected_split_counts": {split: sum(event.split == split for event in selected)
                                  for split in SPLITS},
        "backbone_count": 40, "events_per_backbone": 5,
        "tensor_contract": {
            "tensor_shape": SHAPE, "dtype": "torch.float32",
            "channel_order": CHANNELS, "rainfall_units": "m/s",
            "rainfall_alignment": "future", "selected_timestep_indices": INDICES,
            "selected_times_seconds": TIMES,
            "rainfall_source_times_seconds": SOURCE_TIMES,
            "active_cells": 32249, "inactive_cells": 67181,
            "total_cells": 99430, "barrier_offset": 30.0,
            "barrier_elevation": 152.61531829833984,
        },
        "dem_sha256": dem_sha, **hashes,
        "converted_count": statuses["CONVERTED"],
        "validated_existing_count": statuses["VALID_EXISTING"],
        "failed_count": statuses["FAILED"],
        "pending_count": statuses["PENDING"],
        "pass_count": statuses["CONVERTED"] + statuses["VALID_EXISTING"],
        "fail_count": statuses["FAILED"],
        "skip_count": statuses["VALID_EXISTING"],
        "timing_seconds": {
            "total_batch_runtime": time.monotonic() - started,
            "mean_event_runtime": statistics.mean(durations) if durations else None,
            "median_event_runtime": statistics.median(durations) if durations else None,
            "min_event_runtime": min(durations) if durations else None,
            "max_event_runtime": max(durations) if durations else None,
        },
    }
    atomic_write(OUTPUT / "stockbridge_phase3_pt_summary.json",
                 json.dumps(summary, indent=2, allow_nan=False) + "\n")


def process(event: Event, record: dict, args: argparse.Namespace,
            reference: dict, reference_dem, inactive) -> None:
    if event.action == "VALID_EXISTING":
        record.update(event.existing_stats)
        record["validation_seconds"] = event.existing_validation_seconds
        record["total_seconds"] = event.existing_validation_seconds
        record["status"] = "VALID_EXISTING"
        print(f"  stage=0.00s convert=0.00s validate={event.existing_validation_seconds:.2f}s "
              f"total={event.existing_validation_seconds:.2f}s VALID_EXISTING", flush=True)
        return
    started = time.monotonic()
    stage_seconds = conversion_seconds = validation_seconds = 0.0
    try:
        start = time.monotonic()
        try:
            stage(event)
        finally:
            stage_seconds = time.monotonic() - start
        event.output_pt.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{event.event_id}_", dir=event.output_pt.parent) as temporary:
            temporary_dir = Path(temporary)
            temporary_pt = temporary_dir / event.output_pt.name
            start = time.monotonic()
            try:
                result = subprocess.run(converter_command(event, temporary_pt),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, check=False)
                require(result.returncode == 0,
                        f"{event.event_id}: converter exited {result.returncode}: "
                        f"{result.stdout[-1500:]}")
            finally:
                conversion_seconds = time.monotonic() - start
            start = time.monotonic()
            try:
                stats = validate_output(temporary_pt, event, reference,
                                        reference_dem, inactive)
            finally:
                validation_seconds = time.monotonic() - start
            publish_pair(temporary_pt, event.output_pt, temporary_dir)
            record.update(stats)
            record["status"] = "CONVERTED"
        if not args.keep_staging:
            try:
                shutil.rmtree(event.staged_dir)
            except Exception as exc:
                print(f"  WARNING: published PT is valid; staging cleanup failed: {exc}",
                      file=sys.stderr, flush=True)
    except Exception as exc:
        if record["status"] == "CONVERTED":
            print(f"  WARNING: published PT is valid; post-publication cleanup failed: {exc}",
                  file=sys.stderr, flush=True)
        else:
            record["status"] = "FAILED"
            record["error"] = str(exc)
            print(f"  FAIL: {exc} (staging retained if present)", file=sys.stderr, flush=True)
    finally:
        record["stage_seconds"] = stage_seconds
        record["conversion_seconds"] = conversion_seconds
        record["validation_seconds"] = validation_seconds
        record["total_seconds"] = time.monotonic() - started
        if record["status"] == "CONVERTED":
            print(f"  stage={stage_seconds:.2f}s convert={conversion_seconds:.2f}s "
                  f"validate={validation_seconds:.2f}s "
                  f"total={record['total_seconds']:.2f}s PASS", flush=True)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    check_write_roots()
    events = load_manifest()
    selected = choose_events(events, args)
    reference, reference_dem, inactive, dem_sha = validate_reference()
    preflight(events, selected, args, reference, reference_dem, inactive)
    print(f"Preflight PASS: 200 Phase 3 events; 40 backbones x 5; "
          f"splits train=140 valid=30 test=30; selected={len(selected)}", flush=True)
    for index, event in enumerate(selected, 1):
        print(f"[{index:03d}/{len(selected):03d}] {event.event_id} {event.split} "
              f"{event.backbone_id} {event.family}/{event.profile_subtype} "
              f"action={event.action}", flush=True)
    hashes = {"converter_sha256": sha256(CONVERTER),
              "master_manifest_sha256": sha256(MASTER),
              "phase2_reference_tensor_sha256": sha256(REFERENCE_PT),
              "phase2_reference_sidecar_sha256": sha256(REFERENCE_JSON)}
    if args.dry_run:
        print("Dry run complete; no files written.")
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (OUTPUT / split).mkdir(exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    records = {event.event_id: blank_record(event, dem_sha) for event in events.values()}
    for event in selected:
        records[event.event_id]["status"] = "PENDING"
    write_reports(records, selected, dem_sha, hashes, started)
    for index, event in enumerate(selected, 1):
        print(f"[{index:03d}/{len(selected):03d}] {event.event_id} {event.split} "
              f"{event.backbone_id} {event.family}/{event.profile_subtype}", flush=True)
        process(event, records[event.event_id], args, reference, reference_dem, inactive)
        write_reports(records, selected, dem_sha, hashes, started)
    failed = sum(records[event.event_id]["status"] == "FAILED" for event in selected)
    converted = sum(records[event.event_id]["status"] == "CONVERTED" for event in selected)
    existing = len(selected) - failed - converted
    print(f"Completed: converted={converted} validated-existing={existing} "
          f"failed={failed} runtime={time.monotonic() - started:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Phase 3 batch FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
