#!/usr/bin/env python3
"""Convert Berlin II UrbanFloodCast hydraulic TIFFs and rainfall TXT data into a PT tensor."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import tifffile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Berlin II hydraulic TIFFs to a PT tensor")
    parser.add_argument("--scenario-dir", required=True, help="Directory containing the scenario TIFFs and rainfall TXT")
    parser.add_argument("--dem-path", required=True, help="Path to the source DEM TIFF")
    parser.add_argument("--output-path", required=True, help="Path for the output .pt tensor")
    parser.add_argument("--row-offset", type=int, default=3, help="DEM row start offset")
    parser.add_argument("--col-offset", type=int, default=1, help="DEM column start offset")
    parser.add_argument("--factor", type=int, default=6, help="DEM downsampling factor")
    parser.add_argument("--barrier-offset", type=float, default=30.0, help="Barrier elevation offset")
    parser.add_argument(
        "--rain-alignment",
        choices=["future", "legacy"],
        default="future",
        help="Rainfall alignment mode",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file")
    return parser.parse_args()


def _load_tiff_array(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if isinstance(arr, list):
        raise TypeError(f"Expected a single array from {path}, but got a list")
    return np.asarray(arr, dtype=np.float32)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _find_rainfall_file(scenario_dir: Path) -> Path:
    txt_files = sorted(
        [path for path in scenario_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"],
        key=lambda p: p.name,
    )
    if len(txt_files) != 1:
        raise FileNotFoundError(f"Expected exactly one rainfall .txt file in {scenario_dir}, found {len(txt_files)}")
    return txt_files[0]


def _load_hydraulic_arrays(scenario_dir: Path, scenario_name: str) -> Tuple[List[np.ndarray], List[int]]:
    times_seconds = list(range(0, 7201, 300))
    arrays: List[np.ndarray] = []
    for time_seconds in times_seconds:
        for suffix in ("H", "U", "V"):
            file_name = f"{scenario_name}_{time_seconds}{suffix}.tif"
            path = _require_file(scenario_dir / file_name)
            arrays.append(_load_tiff_array(path))
    return arrays, times_seconds


def _validate_hydraulic_arrays(arrays: List[np.ndarray], scenario_dir: Path, scenario_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_shape = (433, 692)
    if not arrays:
        raise FileNotFoundError(f"No hydraulic arrays found for scenario {scenario_name} in {scenario_dir}")
    first_arr = np.asarray(arrays[0], dtype=np.float32)
    if first_arr.shape != target_shape:
        raise ValueError(f"Hydraulic array shape mismatch: expected {target_shape}, got {first_arr.shape}")
    mask = np.isfinite(first_arr)
    for idx, arr in enumerate(arrays):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != target_shape:
            raise ValueError(f"Hydraulic array {idx} has shape {arr.shape}, expected {target_shape}")
        if np.array_equal(np.isfinite(arr), mask) is False:
            raise ValueError(f"Hydraulic array {idx} has a different finite mask than the reference raster")
    h_arrays = arrays[0::3]
    u_arrays = arrays[1::3]
    v_arrays = arrays[2::3]
    return np.stack(h_arrays, axis=2), np.stack(u_arrays, axis=2), np.stack(v_arrays, axis=2), mask


def _prepare_dem(
    dem_path: Path,
    target_rows: int,
    target_cols: int,
    row_offset: int,
    col_offset: int,
    factor: int,
    hydraulic_mask: np.ndarray,
) -> np.ndarray:
    if hydraulic_mask.shape != (target_rows, target_cols):
        raise ValueError(
            f"Hydraulic mask shape mismatch: expected {(target_rows, target_cols)}, got {hydraulic_mask.shape}"
        )

    dem = _load_tiff_array(dem_path)
    if dem.shape != (2602, 4153):
        raise ValueError(f"DEM shape mismatch: expected (2602, 4153), got {dem.shape}")

    row_start = row_offset
    row_end = row_start + target_rows * factor
    col_start = col_offset
    col_end = col_start + target_cols * factor
    cropped = dem[row_start:row_end, col_start:col_end]
    expected_shape = (target_rows * factor, target_cols * factor)
    if cropped.shape != expected_shape:
        raise ValueError(f"DEM crop shape mismatch: expected {expected_shape}, got {cropped.shape}")

    reshaped = cropped.reshape(target_rows, factor, target_cols, factor)
    finite_mask = np.isfinite(reshaped)
    finite_counts = finite_mask.sum(axis=(1, 3))

    invalid_active = hydraulic_mask & (finite_counts != factor * factor)
    if np.any(invalid_active):
        first_idx = np.argwhere(invalid_active)[0]
        first_row, first_col = first_idx[0], first_idx[1]
        raise ValueError(
            f"DEM validation failed for {invalid_active.sum()} hydraulic cells; "
            f"first affected cell at ({first_row}, {first_col}) has {finite_counts[first_row, first_col]} finite source values; "
            f"expected {factor * factor}"
        )

    block_sums = np.nansum(reshaped, axis=(1, 3), dtype=np.float64)
    dem_target = np.full((target_rows, target_cols), np.nan, dtype=np.float32)
    dem_target[hydraulic_mask] = (
        block_sums[hydraulic_mask] / finite_counts[hydraulic_mask]
    ).astype(np.float32)

    if not np.isfinite(dem_target[hydraulic_mask]).all():
        raise ValueError("DEM target contains non-finite values for hydraulically active cells")

    return dem_target


def _prepare_rainfall(scenario_dir: Path, scenario_name: str, times_seconds: List[int], rain_alignment: str) -> Tuple[np.ndarray, List[float], str]:
    rainfall_path = _find_rainfall_file(scenario_dir)
    rainfall_vector = np.zeros(len(times_seconds), dtype=np.float32)
    used_indices: List[int] = []
    with rainfall_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid rainfall line in {rainfall_path}: {raw_line.rstrip()}")
            time_seconds = float(parts[0])
            rainfall_value = float(parts[1])
            if not np.isfinite(time_seconds) or not np.isfinite(rainfall_value):
                raise ValueError(f"Non-finite rainfall value in {rainfall_path}: {raw_line.rstrip()}")
            if rainfall_value < 0.0:
                raise ValueError(f"Negative rainfall value in {rainfall_path}: {raw_line.rstrip()}")
            if rain_alignment == "future":
                tensor_index = int(time_seconds // 300 + 1)
            else:
                tensor_index = int(time_seconds // 300)
            if 0 <= tensor_index <= len(times_seconds) - 1:
                if tensor_index in used_indices:
                    raise ValueError(f"Duplicate rainfall tensor index {tensor_index} in {rainfall_path}")
                used_indices.append(tensor_index)
                rainfall_vector[tensor_index] = rainfall_value
            else:
                continue
    if not np.isfinite(rainfall_vector).all():
        raise ValueError(f"Rainfall vector contains non-finite values for {scenario_name}")
    return rainfall_vector, [float(v) for v in rainfall_vector], rainfall_path.name


def _assemble_tensor(
    h_tensor: np.ndarray,
    u_tensor: np.ndarray,
    v_tensor: np.ndarray,
    dem_tensor: np.ndarray,
    rainfall_vector: np.ndarray,
    mask: np.ndarray,
    barrier_offset: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    target_rows, target_cols = h_tensor.shape[:2]
    rainfall_spatial = np.broadcast_to(rainfall_vector.reshape(1, 1, -1), (target_rows, target_cols, rainfall_vector.shape[0]))
    dem_target = dem_tensor.astype(np.float32)
    valid_target = mask.astype(bool)
    barrier_elevation = float(np.nanmax(dem_target[valid_target])) + barrier_offset
    dem_out = np.where(valid_target, dem_target, barrier_elevation).astype(np.float32)
    if not np.isfinite(dem_out).all():
        raise ValueError("DEM output contains non-finite values")
    channel_h = h_tensor.astype(np.float32)
    channel_u = u_tensor.astype(np.float32)
    channel_v = v_tensor.astype(np.float32)
    channel_rain = rainfall_spatial.astype(np.float32)
    dem_repeat = np.repeat(dem_out[..., None], rainfall_vector.shape[0], axis=-1)
    output = np.stack([channel_h, channel_u, channel_v, channel_rain, dem_repeat], axis=-1)
    return output, dem_out, barrier_elevation


def main() -> None:
    args = _parse_args()
    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    dem_path = Path(args.dem_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    dem_path = _require_file(dem_path)

    scenario_name = scenario_dir.name
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output path already exists: {output_path}")

    arrays, times_seconds = _load_hydraulic_arrays(scenario_dir, scenario_name)
    h_tensor, u_tensor, v_tensor, mask = _validate_hydraulic_arrays(arrays, scenario_dir, scenario_name)
    dem_tensor = _prepare_dem(
        dem_path,
        target_rows=433,
        target_cols=692,
        row_offset=args.row_offset,
        col_offset=args.col_offset,
        factor=args.factor,
        hydraulic_mask=mask,
    )
    rainfall_vector, rainfall_values, rainfall_file = _prepare_rainfall(
        scenario_dir,
        scenario_name,
        times_seconds,
        args.rain_alignment,
    )

    output, dem_out, barrier_elevation = _assemble_tensor(
        h_tensor=h_tensor,
        u_tensor=u_tensor,
        v_tensor=v_tensor,
        dem_tensor=dem_tensor,
        rainfall_vector=rainfall_vector,
        mask=mask,
        barrier_offset=args.barrier_offset,
    )
    output_tensor = torch.from_numpy(np.ascontiguousarray(output, dtype=np.float32))
    if output_tensor.shape != (433, 692, 25, 5):
        raise ValueError(f"Unexpected output tensor shape: {tuple(output_tensor.shape)}")
    if output_tensor.dtype != torch.float32:
        raise ValueError(f"Unexpected output dtype: {output_tensor.dtype}")

    if output_path.exists() and args.overwrite:
        output_path.unlink()
    torch.save(output_tensor, output_path)

    metadata = {
        "scenario_name": scenario_name,
        "scenario_dir": str(scenario_dir),
        "dem_path": str(dem_path),
        "output_path": str(output_path),
        "tensor_shape": list(output_tensor.shape),
        "dtype": str(output_tensor.dtype),
        "channel_order": ["H", "U", "V", "rainfall", "DEM"],
        "times_seconds": times_seconds,
        "hydraulic_valid_cells": int(mask.sum()),
        "hydraulic_total_cells": int(mask.size),
        "row_offset": args.row_offset,
        "col_offset": args.col_offset,
        "factor": args.factor,
        "barrier_offset": args.barrier_offset,
        "barrier_elevation": float(barrier_elevation),
        "rainfall_file": rainfall_file,
        "rainfall_alignment": args.rain_alignment,
        "rainfall_vector": [float(value) for value in rainfall_values],
        "rainfall_total": float(np.sum(rainfall_vector)),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    channel_nan_counts = [int(np.isnan(channel).sum()) for channel in [output[..., 0], output[..., 1], output[..., 2], output[..., 3], output[..., 4]]]
    summary = (
        f"scenario={scenario_name} output={output_path} "
        f"shape={tuple(output_tensor.shape)} dtype={output_tensor.dtype} "
        f"valid_cells={int(mask.sum())} rain_alignment={args.rain_alignment} "
        f"rainfall_total={float(np.sum(rainfall_vector)):.3f} "
        f"dem_min={float(dem_out[mask].min()):.3f} dem_max={float(dem_out[mask].max()):.3f} "
        f"barrier={float(barrier_elevation):.3f} nans={channel_nan_counts} "
        f"size_mib={output_tensor.numel() * output_tensor.element_size() / (1024**2):.3f}"
    )
    print(summary)


if __name__ == "__main__":
    main()
