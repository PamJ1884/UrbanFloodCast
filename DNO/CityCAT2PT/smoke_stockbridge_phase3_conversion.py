#!/usr/bin/env python3
"""Preflight and smoke-convert three Phase 3 TRAIN events with the canonical converter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DATA = Path.home() / "flood/02_data/CityCAT_Winchester"
MANIFEST = DATA / "manifests/stockbridge_phase2plus3_300_v01/master_manifest_R001_R300_v01.csv"
REFERENCE = DATA / "pt/stockbridge_phase2_core_v01/train/R001_stockbridge_T000_T024.pt.json"
REFERENCE_PT = REFERENCE.with_suffix("")
RAINFALL_ROOT = DATA / "rainfall/stockbridge_phase3_v01/rainfall_data_phase3"
SMOKE = DATA / "smoke/phase3_dno_smoke_v01"
DEM = Path.home() / "flood/10_qgis_projects/Winchester/02_rasters/stockbridge_dem_subcatchment_candidate_v01.tif"
DEM_SHA256 = "d0aa9cbeb0664ab13b269857e9ab8b24e4bb58468fef2488801683e9568a5785"
COMBINED_ROOT = DATA / "pt/stockbridge_phase2plus3_300_v01"
CONVERTER = Path(__file__).with_name("convert_citycat_rsl_to_pt.py")
SOURCE_ROOT = (Path(f"/run/user/{os.getuid()}/gvfs") /
               "smb-share:server=campus.ncl.ac.uk,share=rdw" /
               "pamela-phd-flood/02_data/CityCAT_Winchester/results/stockbridge_phase3_v01")
EVENTS = {"R101": ("triangular", "triangular"),
          "R142": ("gaussian", "central"),
          "R221": ("fsr_feh", "summer")}
INDICES = list(range(25))
TIMES = [index * 300 for index in INDICES]
SOURCE_TIMES = [None, *TIMES[:-1]]
SHAPE = [305, 326, 25, 5]
CHANNELS = ["H", "U", "V", "rainfall", "DEM"]
STATIC_KEYS = ("raster_width", "raster_height", "raster_bounds", "raster_resolution",
               "affine_transform", "crs", "hydraulic_active_cell_count",
               "hydraulic_inactive_cell_count", "barrier_offset", "barrier_elevation")
CSV_FIELDS = ("event_id", "backbone_id", "family", "profile_subtype", "split",
              "source_citycat_dir", "rainfall_path", "rainfall_sha256", "dem_path",
              "dem_sha256", "output_pt_path", "output_sidecar_path", "tensor_shape",
              "dtype", "active_cells", "inactive_cells", "barrier_elevation", "status", "error")


@dataclass
class Plan:
    event_id: str
    row: dict[str, str]
    source_dir: Path
    rsl_files: list[Path]
    rainfall_path: Path
    rainfall_sha256: str
    rainfall_values: list[float]
    staged_dir: Path
    output_pt: Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"event_id", "phase", "split_300", "backbone_id", "family",
                  "profile_subtype", "canonical_rainfall_filename", "canonical_rainfall_path",
                  "rainfall_sha256", "target_combined_pt_path", "duration_min",
                  "return_period_years", "target_depth_mm", "n_timesteps",
                  "timestep_seconds", "simulation_horizon_seconds"}
        require(needed <= set(reader.fieldnames or []),
                f"Master manifest missing columns: {sorted(needed - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            event_id = row["event_id"].strip()
            if event_id in EVENTS:
                require(event_id not in selected, f"Duplicate {event_id} in master manifest at line {line}")
                selected[event_id] = row
    require(set(selected) == set(EVENTS), f"Missing selected events: {sorted(set(EVENTS) - set(selected))}")
    return selected


def validate_rsl(directory: Path, event_id: str) -> list[Path]:
    require(directory.is_dir(), f"CityCAT directory missing: {directory}")
    files = sorted((p for p in directory.iterdir() if p.suffix.lower() == ".rsl"),
                   key=lambda p: p.name)
    require(len(files) == 25, f"{event_id}: expected 25 .rsl files, found {len(files)} in {directory}")
    pattern = re.compile(rf"{event_id}_C1_T(\d+)_(\d+)min\.rsl")
    by_index: dict[int, Path] = {}
    for path in files:
        require(path.is_file() and not path.is_symlink(), f"{event_id}: invalid RSL file: {path}")
        match = pattern.fullmatch(path.name)
        require(match is not None, f"{event_id}: unexpected RSL filename: {path.name}")
        index, minutes = map(int, match.groups())
        require(index not in by_index, f"{event_id}: duplicate timestep T{index}")
        require(minutes * 60 == index * 300,
                f"{event_id}: T{index} filename time is {minutes}min, expected {index * 5}min")
        by_index[index] = path
    require(sorted(by_index) == INDICES,
            f"{event_id}: timestep set differs from T0–T24: {sorted(by_index)}")
    require(by_index[24].name.endswith("T24_120min.rsl"), f"{event_id}: T24 is not 120min")
    return [by_index[index] for index in INDICES]


def validate_rainfall(path: Path, event_id: str) -> tuple[str, list[float]]:
    digest = sha256(path)
    records: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            fields = raw.split()
            if not fields or fields[0] == "*":
                continue
            if len(fields) == 1 and fields[0] == "25":
                continue
            require(len(fields) == 2, f"{event_id}: malformed rainfall line {line_no}: {raw.strip()}")
            try:
                time, value = map(float, fields)
            except ValueError as exc:
                raise ValueError(f"{event_id}: non-numeric rainfall line {line_no}") from exc
            require(math.isfinite(time) and math.isfinite(value) and value >= 0,
                    f"{event_id}: non-finite or negative rainfall at line {line_no}")
            records.append((time, value))
    require(len(records) == 25, f"{event_id}: expected 25 rainfall records, found {len(records)}")
    require([time for time, _ in records] == TIMES,
            f"{event_id}: rainfall times must be exactly 0,300,...,7200s")
    return digest, [value for _, value in records]


def build_plans(source_root: Path, reference: dict) -> tuple[list[Plan], str]:
    require(CONVERTER.is_file(), f"Converter missing: {CONVERTER}")
    require(REFERENCE.is_file(), f"Phase 2 reference sidecar missing: {REFERENCE}")
    for key, expected in (("tensor_shape", SHAPE), ("dtype", "torch.float32"),
                          ("channel_order", CHANNELS), ("rainfall_units", "m/s"),
                          ("rainfall_alignment", "future"),
                          ("hydraulic_active_cell_count", 32249),
                          ("hydraulic_inactive_cell_count", 67181),
                          ("hydraulic_total_cell_count", 99430),
                          ("selected_timestep_indices", INDICES),
                          ("selected_times_seconds", TIMES),
                          ("rainfall_source_times_seconds", SOURCE_TIMES),
                          ("barrier_offset", 30.0)):
        require(reference.get(key) == expected, f"Phase 2 reference {key} differs: {reference.get(key)!r}")
    require(reference.get("raster_width") == 326 and reference.get("raster_height") == 305
            and reference.get("raster_resolution") == [5.0, 5.0],
            "Phase 2 reference raster geometry differs")
    source_dem = reference.get("source_dem_path")
    require(isinstance(source_dem, str) and
            Path(source_dem).expanduser().resolve() == DEM.resolve(),
            f"Phase 2 reference source_dem_path differs from smoke DEM: {source_dem!r}")
    dem_sha = sha256(DEM)
    require(dem_sha == DEM_SHA256, f"DEM SHA256 mismatch: {dem_sha} != {DEM_SHA256}")
    rows = read_manifest()
    plans = []
    for event_id, (family, subtype) in EVENTS.items():
        row = rows[event_id]
        for key, expected in (("phase", "phase3"), ("split_300", "train"),
                              ("backbone_id", "B01"), ("family", family),
                              ("profile_subtype", subtype), ("duration_min", "15"),
                              ("return_period_years", "2"), ("n_timesteps", "25"),
                              ("timestep_seconds", "300"),
                              ("simulation_horizon_seconds", "7200")):
            require(row[key] == expected, f"{event_id}: manifest {key}={row[key]!r}, expected {expected!r}")
        require(math.isclose(float(row["target_depth_mm"]), 8.3, abs_tol=1e-9, rel_tol=0),
                f"{event_id}: target depth is not 8.3mm")
        number = int(event_id[1:])
        rainfall_name = f"Rainfall_Data_{number}.txt"
        rainfall_path = RAINFALL_ROOT / rainfall_name
        require(row["canonical_rainfall_filename"] == rainfall_name and
                Path(row["canonical_rainfall_path"]).resolve() == rainfall_path.resolve(),
                f"{event_id}: canonical rainfall path differs from required local source")
        target_name = f"{event_id}_stockbridge_T000_T024.pt"
        require(Path(row["target_combined_pt_path"]) == COMBINED_ROOT / "train" / target_name,
                f"{event_id}: unexpected target combined PT metadata")
        source_dir = source_root / f"{event_id}C1_SurfaceMaps"
        rsl_files = validate_rsl(source_dir, event_id)
        rain_sha, rain_values = validate_rainfall(rainfall_path, event_id)
        require(rain_sha == row["rainfall_sha256"],
                f"{event_id}: rainfall SHA256 mismatch with manifest: {rain_sha}")
        plans.append(Plan(event_id, row, source_dir, rsl_files, rainfall_path,
                          rain_sha, rain_values, SMOKE / "citycat" / source_dir.name,
                          SMOKE / "pt" / target_name))
    return plans, dem_sha


def load_reference_dem():
    import torch

    require(REFERENCE_PT.is_file(), f"Phase 2 reference tensor missing: {REFERENCE_PT}")
    tensor = torch.load(REFERENCE_PT, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == SHAPE and
            tensor.dtype == torch.float32, "Phase 2 reference tensor type, shape, or dtype differs")
    dem = tensor[..., 4].clone()
    require(bool(torch.isfinite(dem).all()), "Phase 2 reference DEM has non-finite values")
    return dem


def command(plan: Plan) -> list[str]:
    return [sys.executable, str(CONVERTER), "--rsl-dir", str(plan.staged_dir),
            "--dem-path", str(DEM), "--rainfall-path", str(plan.rainfall_path),
            "--output-path", str(plan.output_pt), "--start-index", "0",
            "--end-index", "24", "--rain-alignment", "future",
            "--barrier-offset", "30", "--expected-step-seconds", "300"]


def validate_output(plan: Plan, reference: dict, reference_dem) -> dict:
    import torch

    sidecar_path = plan.output_pt.with_suffix(".pt.json")
    require(plan.output_pt.is_file() and sidecar_path.is_file(),
            f"{plan.event_id}: PT or sidecar missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for key in STATIC_KEYS:
        require(sidecar.get(key) == reference.get(key),
                f"{plan.event_id}: {key} differs from Phase 2: {sidecar.get(key)!r} != {reference.get(key)!r}")
    for key, expected in (("tensor_shape", SHAPE), ("dtype", "torch.float32"),
                          ("channel_order", CHANNELS), ("rainfall_units", "m/s"),
                          ("rainfall_alignment", "future"),
                          ("selected_timestep_indices", INDICES),
                          ("selected_times_seconds", TIMES),
                          ("rainfall_source_times_seconds", SOURCE_TIMES),
                          ("hydraulic_total_cell_count", 99430),
                          ("selected_rsl_filenames", [p.name for p in plan.rsl_files])):
        require(sidecar.get(key) == expected, f"{plan.event_id}: sidecar {key} differs")
    require(sidecar.get("source_rsl_dir") == str(plan.staged_dir) and
            sidecar.get("source_dem_path") == str(DEM) and
            sidecar.get("source_rainfall_path") == str(plan.rainfall_path),
            f"{plan.event_id}: sidecar input paths differ")
    require(sidecar.get("dem_min_inside_mask") == reference.get("dem_min_inside_mask") and
            sidecar.get("dem_max_inside_mask") == reference.get("dem_max_inside_mask"),
            f"{plan.event_id}: DEM elevation range differs from Phase 2")
    tensor = torch.load(plan.output_pt, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == SHAPE and
            tensor.dtype == torch.float32, f"{plan.event_id}: tensor type, shape, or dtype differs")
    require(bool(torch.isfinite(tensor).all()), f"{plan.event_id}: tensor has non-finite values")
    dem = tensor[..., 4]
    require(torch.equal(dem, reference_dem),
            f"{plan.event_id}: DEM channel differs pixel-by-pixel from Phase 2 reference")
    require(bool(torch.equal(dem, dem[..., :1].expand_as(dem))),
            f"{plan.event_id}: DEM changes over time")
    elevation = float(reference["barrier_elevation"])
    barrier_value = torch.tensor(elevation, dtype=torch.float32).item()
    barrier = dem[..., 0] == barrier_value
    inactive = int(barrier.sum())
    active = barrier.numel() - inactive
    require((active, inactive) == (32249, 67181),
            f"{plan.event_id}: active/inactive cells differ: {active}/{inactive}")
    require(float(dem[..., 0].max()) == barrier_value and
            float(dem[..., 0][~barrier].min()) == reference["dem_min_inside_mask"] and
            float(dem[..., 0][~barrier].max()) == reference["dem_max_inside_mask"],
            f"{plan.event_id}: barrier or active DEM elevations differ")
    require(not bool(torch.any(tensor[..., :3][barrier] != 0)),
            f"{plan.event_id}: H/U/V nonzero in inactive cells")
    require(not bool(torch.any(tensor[..., 0] < 0)), f"{plan.event_id}: negative H")
    expected_rain = torch.tensor([0.0, *plan.rainfall_values[:-1]], dtype=torch.float32)
    require(bool(torch.equal(tensor[..., 3], expected_rain.view(1, 1, 25).expand(305, 326, 25))),
            f"{plan.event_id}: rainfall channel is not spatially uniform or correctly aligned")
    require(sidecar.get("rainfall_vector") == expected_rain.tolist(),
            f"{plan.event_id}: sidecar rainfall vector differs from source")
    return {"tensor_shape": "x".join(map(str, SHAPE)), "dtype": "torch.float32",
            "active_cells": active, "inactive_cells": inactive,
            "barrier_elevation": elevation}


def provenance_row(plan: Plan, dem_sha: str) -> dict:
    return {"event_id": plan.event_id, "backbone_id": plan.row["backbone_id"],
            "family": plan.row["family"], "profile_subtype": plan.row["profile_subtype"],
            "split": plan.row["split_300"], "source_citycat_dir": str(plan.source_dir),
            "rainfall_path": str(plan.rainfall_path), "rainfall_sha256": plan.rainfall_sha256,
            "dem_path": str(DEM), "dem_sha256": dem_sha,
            "output_pt_path": str(plan.output_pt),
            "output_sidecar_path": str(plan.output_pt.with_suffix(".pt.json")),
            "tensor_shape": "", "dtype": "", "active_cells": "", "inactive_cells": "",
            "barrier_elevation": "", "status": "PENDING", "error": ""}


def write_reports(records: list[dict], dem_sha: str, reference: dict, hashes: dict) -> None:
    with (SMOKE / "smoke_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    contract = {key: reference[key] for key in (*STATIC_KEYS, "tensor_shape", "dtype",
                                               "channel_order", "rainfall_units", "rainfall_alignment")}
    summary = {"event_count": len(records),
               "pass_count": sum(row["status"] == "PASS" for row in records),
               "fail_count": sum(row["status"] == "FAIL" for row in records),
               "common_tensor_contract": contract,
               "selected_events": [row["event_id"] for row in records],
               "reference_phase2_sidecar": str(REFERENCE), "dem_sha256": dem_sha,
               **hashes}
    (SMOKE / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT,
                        help="Mounted Phase 3 CityCAT results root")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace a prior smoke workspace")
    args = parser.parse_args()
    require(not SMOKE.is_symlink(), f"Refusing symlink smoke workspace: {SMOKE}")
    require(not SMOKE.exists() or args.overwrite,
            f"Smoke workspace already exists: {SMOKE}; use --overwrite to replace it")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    plans, dem_sha = build_plans(args.source_root.expanduser().resolve(), reference)
    reference_dem = load_reference_dem()
    print(f"Preflight PASS: {len(plans)} TRAIN Phase 3 B01 events; DEM SHA256 {dem_sha}", flush=True)
    for plan in plans:
        print(f"{plan.event_id} {plan.row['family']}/{plan.row['profile_subtype']}: "
              f"25 RSL files, 25 rainfall records; {shlex.join(command(plan))}", flush=True)
    if args.dry_run:
        print("Dry run complete; no files written.")
        return 0
    hashes = {"converter_sha256": sha256(CONVERTER),
              "reference_phase2_sidecar_sha256": sha256(REFERENCE),
              "reference_phase2_tensor_sha256": sha256(REFERENCE_PT)}
    if SMOKE.exists():
        require(SMOKE.is_dir(), f"Smoke workspace is not a directory: {SMOKE}")
        shutil.rmtree(SMOKE)
    (SMOKE / "citycat").mkdir(parents=True)
    (SMOKE / "pt").mkdir()
    records = []
    for plan in plans:
        record = provenance_row(plan, dem_sha)
        try:
            plan.staged_dir.mkdir()
            for source in plan.rsl_files:
                shutil.copy2(source, plan.staged_dir / source.name)
            validate_rsl(plan.staged_dir, plan.event_id)
            subprocess.run(command(plan), check=True)
            record.update(validate_output(plan, reference, reference_dem))
            record["status"] = "PASS"
            print(f"{plan.event_id}: PASS", flush=True)
        except Exception as exc:
            record["status"] = "FAIL"
            record["error"] = str(exc)
            print(f"{plan.event_id}: FAIL: {exc}", file=sys.stderr, flush=True)
        records.append(record)
        write_reports(records, dem_sha, reference, hashes)
    failed = sum(record["status"] == "FAIL" for record in records)
    print(f"Smoke complete: PASS={len(records) - failed} FAIL={failed}; reports in {SMOKE}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Smoke preflight FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
