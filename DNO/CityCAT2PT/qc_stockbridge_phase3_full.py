#!/usr/bin/env python3
"""Independent, read-only global QC of canonical Stockbridge Phase 3 PT files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

sys.dont_write_bytecode = True
import torch


HOME = Path.home()
DATA = HOME / "flood/02_data/CityCAT_Winchester"
DATASET = DATA / "pt/stockbridge_phase3_core_v01"
MASTER = DATA / "manifests/stockbridge_phase2plus3_300_v01/master_manifest_R001_R300_v01.csv"
PRODUCTION = DATASET / "stockbridge_phase3_pt_manifest.csv"
REFERENCE_PT = DATA / "pt/stockbridge_phase2_core_v01/train/R001_stockbridge_T000_T024.pt"
DEM = HOME / "flood/10_qgis_projects/Winchester/02_rasters/stockbridge_dem_subcatchment_candidate_v01.tif"
DEM_SHA256 = "d0aa9cbeb0664ab13b269857e9ab8b24e4bb58468fef2488801683e9568a5785"
RAIN_ROOT = DATA / "rainfall/stockbridge_phase3_v01/rainfall_data_phase3"
STAGING_ROOT = DATA / "staging/stockbridge_phase3_core_v01"
IDS = {f"R{number:03d}" for number in range(101, 301)}
SPLIT_COUNTS = {"train": 140, "valid": 30, "test": 30}
SHAPE = (305, 326, 25, 5)
CHANNELS = ["H", "U", "V", "rainfall", "DEM"]
TIMES = list(range(0, 7201, 300))
SOURCE_TIMES = [None, *TIMES[:-1]]
BARRIER = 152.61531829833984
ACTIVE, INACTIVE, TOTAL = 32249, 67181, 99430
THRESHOLDS = (0.01, 0.05, 0.10)
STATIC_KEYS = ("raster_width", "raster_height", "raster_bounds", "raster_resolution",
               "affine_transform", "crs")
PT_NAME = re.compile(r"(R\d{3})_stockbridge_T000_T024\.pt")
MASTER_FIELDS = {"event_id", "event_index", "phase", "phase_sequence", "split_300",
                 "backbone_id", "family", "profile_subtype", "canonical_rainfall_filename",
                 "canonical_rainfall_path", "rainfall_sha256", "citycat_results_location",
                 "n_timesteps", "timestep_seconds", "simulation_horizon_seconds"}
PRODUCTION_FIELDS = {"event_id", "event_index", "phase_sequence", "backbone_id",
                     "family", "profile_subtype", "split", "source_citycat_dir",
                     "rainfall_path", "rainfall_sha256", "dem_path", "dem_sha256",
                     "output_pt_path", "output_sidecar_path", "status", "error"}


class QCError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QCError(message)


def file_sha256(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, fields: set[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = fields - set(reader.fieldnames or [])
        require(not missing, f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def load_manifests() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    master: dict[str, dict[str, str]] = {}
    sequences: set[int] = set()
    for row in read_csv(MASTER, MASTER_FIELDS):
        if row["phase"].strip() != "phase3":
            continue
        event_id = row["event_id"].strip()
        require(event_id in IDS and event_id not in master,
                f"Master has unexpected/duplicate Phase 3 event {event_id!r}")
        try:
            index, sequence = int(row["event_index"]), int(row["phase_sequence"])
            n_timesteps = int(row["n_timesteps"])
            timestep_seconds = int(row["timestep_seconds"])
            horizon_seconds = int(row["simulation_horizon_seconds"])
        except ValueError as exc:
            raise QCError(f"{event_id}: invalid master index/sequence/temporal metadata") from exc
        require(index == int(event_id[1:]) and sequence not in sequences,
                f"{event_id}: master index or phase sequence invalid/duplicate")
        sequences.add(sequence)
        require((n_timesteps, timestep_seconds, horizon_seconds) == (25, 300, 7200),
                f"{event_id}: master temporal metadata differs from 25/300/7200")
        require(row["split_300"] in SPLIT_COUNTS,
                f"{event_id}: invalid master split {row['split_300']!r}")
        require(bool(row["backbone_id"] and row["family"] and row["profile_subtype"]),
                f"{event_id}: missing master backbone/family/subtype")
        filename = f"Rainfall_Data_{index}.txt"
        rain_path = Path(row["canonical_rainfall_path"])
        require(row["canonical_rainfall_filename"] == filename and
                rain_path.is_absolute() and rain_path.name == filename and
                rain_path.resolve().parent == RAIN_ROOT.resolve(),
                f"{event_id}: invalid canonical rainfall path/filename")
        require(re.fullmatch(r"[0-9a-fA-F]{64}", row["rainfall_sha256"]) is not None,
                f"{event_id}: invalid master rainfall SHA256")
        master[event_id] = row
    require(set(master) == IDS and sequences == set(range(1, 201)),
            f"Master Phase 3 inventory differs from R101–R300/sequence 1–200: {len(master)} rows")
    counts = {split: sum(row["split_300"] == split for row in master.values())
              for split in SPLIT_COUNTS}
    require(counts == SPLIT_COUNTS, f"Master split counts differ: {counts}")
    backbones: dict[str, list[str]] = defaultdict(list)
    for event_id, row in master.items():
        backbones[row["backbone_id"]].append(event_id)
    require(len(backbones) == 40, f"Master has {len(backbones)} backbones, expected 40")
    for backbone, members in backbones.items():
        require(len(members) == 5 and
                len({master[event_id]["split_300"] for event_id in members}) == 1,
                f"{backbone}: expected five events in one frozen split")

    production: dict[str, dict[str, str]] = {}
    rows = read_csv(PRODUCTION, PRODUCTION_FIELDS)
    for row in rows:
        event_id = row["event_id"].strip()
        require(event_id in IDS and event_id not in production,
                f"Production manifest has unexpected/duplicate event {event_id!r}")
        production[event_id] = row
    require(len(rows) == 200 and set(production) == IDS,
            f"Production manifest must contain exactly R101–R300: {len(rows)} rows")
    return master, production


def check_source_provenance(event_id: str, master: dict, production: dict) -> None:
    uri = urlparse(master["citycat_results_location"])
    parts = uri.path.strip("/").split("/")
    require(uri.scheme == "smb" and uri.netloc == "campus.ncl.ac.uk" and
            len(parts) >= 2 and parts[0] == "rdw" and
            parts[-1] == f"{event_id}C1_SurfaceMaps",
            f"{event_id}: invalid original CityCAT SMB location in master")
    source = Path(production["source_citycat_dir"])
    expected_suffix = ("/smb-share:server=campus.ncl.ac.uk,share=rdw/" +
                       "/".join(parts[1:]))
    require(source.is_absolute() and source.as_posix().endswith(expected_suffix),
            f"{event_id}: production source_citycat_dir differs from master SMB provenance")
    # The original GVFS path is provenance; neither it nor temporary staging must be mounted now.


def check_manifest_row(event_id: str, master: dict, production: dict) -> tuple[Path, Path]:
    for product_key, master_key in (("event_index", "event_index"),
                                    ("phase_sequence", "phase_sequence"),
                                    ("backbone_id", "backbone_id"),
                                    ("family", "family"),
                                    ("profile_subtype", "profile_subtype"),
                                    ("split", "split_300"),
                                    ("rainfall_sha256", "rainfall_sha256")):
        require(production[product_key] == master[master_key],
                f"{event_id}: production {product_key} differs from master")
    rain = Path(master["canonical_rainfall_path"])
    require(Path(production["rainfall_path"]) == rain,
            f"{event_id}: production rainfall path differs from master")
    split = master["split_300"]
    pt = DATASET / split / f"{event_id}_stockbridge_T000_T024.pt"
    sidecar = pt.with_suffix(".pt.json")
    require(Path(production["output_pt_path"]) == pt and
            Path(production["output_sidecar_path"]) == sidecar,
            f"{event_id}: production PT/sidecar paths differ from canonical Phase 3 output")
    require(Path(production["dem_path"]).resolve() == DEM.resolve() and
            production["dem_sha256"] == DEM_SHA256,
            f"{event_id}: production DEM provenance differs")
    require(production["status"] in {"CONVERTED", "VALID_EXISTING"} and
            not production["error"].strip(),
            f"{event_id}: production status/error indicates incomplete or failed conversion")
    check_source_provenance(event_id, master, production)
    return pt, sidecar


def check_inventory(master: dict[str, dict]) -> None:
    require(DATASET.is_dir(), f"Missing Phase 3 dataset: {DATASET}")
    seen: dict[str, str] = {}
    for split, expected_count in SPLIT_COUNTS.items():
        folder = DATASET / split
        require(folder.is_dir() and not folder.is_symlink(),
                f"Missing/symlink split directory: {folder}")
        pts: set[str] = set()
        sidecars: set[str] = set()
        for path in folder.iterdir():
            name = path.name
            if ".pt" not in name:
                continue
            require(path.is_file() and not path.is_symlink(),
                    f"Invalid PT/sidecar inventory item: {path}")
            if name.endswith(".pt.json"):
                match = PT_NAME.fullmatch(name[:-5])
                require(match is not None, f"Unexpected PT sidecar: {path}")
                sidecars.add(match.group(1))
            elif name.endswith(".pt"):
                match = PT_NAME.fullmatch(name)
                require(match is not None, f"Unexpected PT file: {path}")
                event_id = match.group(1)

                if event_id in seen:
                    raise QCError(
                        f"Duplicate event PT across splits: "
                        f"{event_id} in {seen[event_id]} and {split}"
                    )

                seen[event_id] = split
                pts.add(event_id)
            else:
                raise QCError(f"Unexpected PT-related file: {path}")
        expected_ids = {event_id for event_id, row in master.items()
                        if row["split_300"] == split}
        require(len(pts) == expected_count and len(sidecars) == expected_count and
                pts == expected_ids and sidecars == expected_ids,
                f"{split}: PT/sidecar inventory differs from frozen split; "
                f"PT={len(pts)} sidecars={len(sidecars)} "
                f"missing_PT={sorted(expected_ids - pts)} "
                f"missing_sidecar={sorted(expected_ids - sidecars)} "
                f"unexpected={sorted((pts | sidecars) - expected_ids)}")
    require(set(seen) == IDS, f"Dataset PT inventory misses events: {sorted(IDS - set(seen))}")


def load_reference():
    require(file_sha256(DEM) == DEM_SHA256, "Canonical DEM SHA256 differs from expected")
    reference_json = REFERENCE_PT.with_suffix(".pt.json")
    require(reference_json.is_file(), f"Missing Phase 2 sidecar: {reference_json}")
    sidecar = json.loads(reference_json.read_text(encoding="utf-8"))
    require(sidecar.get("barrier_elevation") == BARRIER and
            sidecar.get("hydraulic_active_cell_count") == ACTIVE and
            sidecar.get("hydraulic_inactive_cell_count") == INACTIVE and
            sidecar.get("hydraulic_total_cell_count") == TOTAL,
            "Phase 2 reference sidecar spatial contract differs")
    require(Path(sidecar.get("source_dem_path", "")).resolve() == DEM.resolve(),
            "Phase 2 reference sidecar uses a different DEM")
    require(REFERENCE_PT.is_file(), f"Missing Phase 2 tensor: {REFERENCE_PT}")
    tensor = torch.load(REFERENCE_PT, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == SHAPE and
            tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
            "Phase 2 reference tensor has wrong shape/dtype or non-finite values")
    dem = tensor[..., 4].clone()
    require(torch.equal(dem, dem[..., :1].expand_as(dem)),
            "Phase 2 reference DEM is not static through time")
    barrier_f32 = torch.tensor(BARRIER, dtype=torch.float32).item()
    inactive = dem[..., 0] == barrier_f32
    require((int((~inactive).sum()), int(inactive.sum()), inactive.numel()) ==
            (ACTIVE, INACTIVE, TOTAL), "Phase 2 reference hydraulic mask counts differ")
    return sidecar, dem, inactive


def parse_rainfall(path: Path, event_id: str, expected_hash: str):
    require(file_sha256(path).lower() == expected_hash.lower(),
            f"{event_id}: canonical rainfall SHA256 differs from master")
    records: list[tuple[float, float]] = []
    declared = None
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0] == "*":
            continue
        if len(fields) == 1 and declared is None and not records:
            try:
                declared = int(fields[0])
            except ValueError as exc:
                raise QCError(f"{event_id}: invalid rainfall count line {line_no}") from exc
            continue
        require(len(fields) == 2, f"{event_id}: malformed rainfall line {line_no}")
        try:
            time_seconds, value = map(float, fields)
        except ValueError as exc:
            raise QCError(f"{event_id}: non-numeric rainfall line {line_no}") from exc
        require(math.isfinite(time_seconds) and math.isfinite(value) and value >= 0,
                f"{event_id}: non-finite or negative rainfall at line {line_no}")
        records.append((time_seconds, value))
    require(declared == 25 and len(records) == 25 and
            [time for time, _ in records] == TIMES,
            f"{event_id}: rainfall must have 25 records at 0,300,...,7200s")
    values = [value for _, value in records]
    peak = max(range(25), key=values.__getitem__)
    expected = torch.tensor([0.0, *values[:-1]], dtype=torch.float32)
    depth_mm = math.fsum(values) * 300 * 1000
    peak_mm_h = values[peak] * 3_600_000
    return expected, depth_mm, peak_mm_h, peak


@dataclass
class Diagnostics:
    event_id: str
    split: str
    family: str
    subtype: str
    h_min: float
    h_max: float
    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float
    flooded_max: dict[float, int]
    flooded_t24: dict[float, int]
    rain_depth_mm: float
    peak_mm_h: float
    peak_source_timestep: int


def check_event(event_id: str, master: dict, production: dict,
                reference_sidecar: dict, reference_dem: torch.Tensor,
                inactive: torch.Tensor) -> tuple[Diagnostics, bool, str]:
    pt, sidecar_path = check_manifest_row(event_id, master, production)
    expected_rain, depth_mm, peak_mm_h, peak = parse_rainfall(
        Path(master["canonical_rainfall_path"]), event_id, master["rainfall_sha256"])
    require(pt.is_file() and sidecar_path.is_file(),
            f"{event_id}: missing PT or sidecar")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "tensor_shape": list(SHAPE), "dtype": "torch.float32",
        "channel_order": CHANNELS, "rainfall_units": "m/s",
        "rainfall_alignment": "future", "selected_timestep_indices": list(range(25)),
        "selected_times_seconds": TIMES,
        "rainfall_source_times_seconds": SOURCE_TIMES,
        "hydraulic_active_cell_count": ACTIVE,
        "hydraulic_inactive_cell_count": INACTIVE,
        "hydraulic_total_cell_count": TOTAL,
        "barrier_offset": 30.0, "barrier_elevation": BARRIER,
    }
    for key, expected in expected_metadata.items():
        require(metadata.get(key) == expected,
                f"{event_id}: sidecar {key} differs from canonical contract")
    for key in STATIC_KEYS:
        require(metadata.get(key) == reference_sidecar.get(key),
                f"{event_id}: sidecar {key} differs from Phase 2 reference")
    require(Path(metadata.get("source_dem_path", "")).resolve() == DEM.resolve() and
            Path(metadata.get("source_rainfall_path", "")) ==
            Path(master["canonical_rainfall_path"]),
            f"{event_id}: sidecar DEM/rainfall provenance differs")
    rsl_names = [f"{event_id}_C1_T{index}_{index * 5}min.rsl" for index in range(25)]
    require(metadata.get("selected_rsl_filenames") == rsl_names,
            f"{event_id}: sidecar timestep filenames differ")
    staged_source = metadata.get("source_rsl_dir")
    expected_staging = STAGING_ROOT / f"{event_id}C1_SurfaceMaps"
    require(isinstance(staged_source, str) and staged_source == str(expected_staging),
            f"{event_id}: sidecar source_rsl_dir {staged_source!r} differs from "
            f"historical staging path {str(expected_staging)!r}")
    staging_missing = not expected_staging.exists()
    tensor = torch.load(pt, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == SHAPE and
            tensor.dtype == torch.float32,
            f"{event_id}: tensor type/shape/dtype differs")
    require(bool(torch.isfinite(tensor).all()), f"{event_id}: tensor has non-finite values")
    h, vx, vy, rain, dem = (tensor[..., index] for index in range(5))
    require(not bool(torch.any(h < 0)), f"{event_id}: H contains negative depths")
    require(torch.equal(dem, dem[..., :1].expand_as(dem)),
            f"{event_id}: DEM changes through time")
    require(torch.equal(dem, reference_dem),
            f"{event_id}: DEM differs pixel-by-pixel from Phase 2")
    require(not bool(torch.any(tensor[..., :3][inactive] != 0)),
            f"{event_id}: H/U/V nonzero on inactive cells")
    require(torch.equal(rain, rain[:1, :1, :].expand_as(rain)),
            f"{event_id}: rainfall is not spatially uniform")
    require(torch.equal(rain, expected_rain.view(1, 1, 25).expand_as(rain)),
            f"{event_id}: tensor rainfall differs from canonical future alignment")

    active = ~inactive
    def extrema(channel: torch.Tensor) -> tuple[float, float]:
        cells = channel[active]
        return float(cells.min()), float(cells.max())
    h_min, h_max = extrema(h)
    vx_min, vx_max = extrema(vx)
    vy_min, vy_max = extrema(vy)
    flooded_max, flooded_t24 = {}, {}
    active_h = h[active]
    for threshold in THRESHOLDS:
        counts = (active_h > threshold).sum(dim=0)
        flooded_max[threshold] = int(counts.max())
        flooded_t24[threshold] = int(counts[-1])
    return (Diagnostics(event_id, master["split_300"], master["family"],
                        master["profile_subtype"], h_min, h_max,
                        vx_min, vx_max, vy_min, vy_max,
                        flooded_max, flooded_t24, depth_mm, peak_mm_h, peak),
            staging_missing, staged_source)


def aggregate(records: list[Diagnostics], field: str) -> list[tuple[str, list[Diagnostics]]]:
    groups: dict[str, list[Diagnostics]] = defaultdict(list)
    for record in records:
        groups[getattr(record, field)].append(record)
    return sorted(groups.items())


def print_aggregates(records: list[Diagnostics]) -> None:
    for label, field in (("Split", "split"), ("Rainfall family", "family"),
                         ("Profile subtype", "subtype")):
        print(f"\n{label} diagnostics:")
        print("Group | N | H[min,max] m | Vx[min,max] m/s | Vy[min,max] m/s | "
              "Rain depth mm [min,max] | Peak mm/h max | Peak source T[min,max]")
        grouped = aggregate(records, field)
        for name, group in grouped:
            def pair(low: float, high: float) -> str:
                return f"[{low:.4g},{high:.4g}]"
            print(f"{name} | {len(group)} | "
                  f"{pair(min(r.h_min for r in group), max(r.h_max for r in group))} | "
                  f"{pair(min(r.vx_min for r in group), max(r.vx_max for r in group))} | "
                  f"{pair(min(r.vy_min for r in group), max(r.vy_max for r in group))} | "
                  f"{pair(min(r.rain_depth_mm for r in group), max(r.rain_depth_mm for r in group))} | "
                  f"{max(r.peak_mm_h for r in group):.4g} | "
                  f"T{min(r.peak_source_timestep for r in group)}-"
                  f"T{max(r.peak_source_timestep for r in group)}")
        print("Flooded active cells: maximum across events/timesteps / maximum at T24")
        print("Group | H>0.01 m | H>0.05 m | H>0.10 m")
        for name, group in grouped:
            counts = [f"{max(r.flooded_max[t] for r in group)}/"
                      f"{max(r.flooded_t24[t] for r in group)}" for t in THRESHOLDS]
            print(f"{name} | " + " | ".join(counts))


def main() -> int:
    try:
        master, production = load_manifests()
        check_inventory(master)
        reference_sidecar, reference_dem, inactive = load_reference()
    except Exception as exc:
        print(f"QC FAIL: global structure/reference: {exc}", file=sys.stderr)
        return 1
    issues: list[str] = []
    records: list[Diagnostics] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    missing_staging: list[tuple[str, str]] = []
    for number in range(101, 301):
        event_id = f"R{number:03d}"
        pt = DATASET / master[event_id]["split_300"] / f"{event_id}_stockbridge_T000_T024.pt"
        try:
            hashes[file_sha256(pt)].append(event_id)
            record, stage_missing, staged_source = check_event(
                event_id, master[event_id], production[event_id],
                reference_sidecar, reference_dem, inactive)
            records.append(record)
            if stage_missing:
                missing_staging.append((event_id, staged_source))
        except Exception as exc:
            issues.append(f"{event_id}: {exc}")
        if number % 25 == 0:
            print(f"Checked through R{number:03d}: {len(records)}/200 events passed", flush=True)
    duplicates = [(digest, members) for digest, members in sorted(hashes.items())
                  if len(members) > 1]
    print("\nInventory: 200 unique events; train=140 valid=30 test=30; "
          "40 backbones x 5; one split per backbone")
    print(f"Tensor/rainfall contract: {len(records)}/200 passed")
    if records:
        print_aggregates(records)
    if duplicates:
        print("\nDuplicate PT SHA256 values (diagnostic; review required):")
        for digest, members in duplicates:
            print(f"  {digest}: {', '.join(members)}")
    else:
        print("\nDuplicate PT SHA256 values: none")
    print(f"INFO: {len(missing_staging)} sidecars point to deleted temporary staging directories "
          "(not a QC failure).")
    if missing_staging:
        print(f"INFO example: {missing_staging[0][0]} -> {missing_staging[0][1]}")
    if issues:
        print(f"\nQC FAIL: {len(issues)} event(s) failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("\nQC PASS: 200/200 events; canonical tensor, rainfall, spatial, "
          "manifest, and inventory checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
