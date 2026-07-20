#!/usr/bin/env python3
"""Convert cropped CityCAT RSL outputs to a PyTorch tensor for DNO training pipelines."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CityCAT RSL files to a PT tensor")
    parser.add_argument("--rsl-dir", required=True, help="Directory containing CityCAT .rsl files")
    parser.add_argument("--dem-path", required=True, help="Path to the cropped DEM GeoTIFF")
    parser.add_argument("--rainfall-path", required=True, help="Path to the CityCAT rainfall TXT file")
    parser.add_argument("--output-path", required=True, help="Output .pt path")
    parser.add_argument("--start-index", type=int, default=0, help="First CityCAT timestep index, inclusive")
    parser.add_argument("--end-index", type=int, default=24, help="Last CityCAT timestep index, inclusive")
    parser.add_argument(
        "--rain-alignment",
        choices=["future", "legacy"],
        default="future",
        help="Rainfall alignment mode",
    )
    parser.add_argument("--barrier-offset", type=float, default=30.0, help="Barrier elevation offset")
    parser.add_argument(
        "--expected-step-seconds",
        type=int,
        default=300,
        help="Expected timestep interval in seconds",
    )
    return parser.parse_args()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _parse_rsl_filename(path: Path) -> Dict[str, object]:
    match = re.search(r"T(?P<index>\d+)_(?P<minutes>\d+)min\.rsl$", path.name)
    if match is None:
        raise ValueError(f"Could not parse CityCAT timestep from filename: {path.name}")
    index = int(match.group("index"))
    minutes = int(match.group("minutes"))
    return {"path": path, "index": index, "minutes": minutes, "seconds": minutes * 60}


def _load_rsl_timestep(
    path: Path,
    bounds: Tuple[float, float, float, float],
    expected_step_seconds: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    xmin, ymin, xmax, ymax = bounds
    x_values: List[float] = []
    y_values: List[float] = []
    depth_values: List[float] = []
    u_values: List[float] = []
    v_values: List[float] = []

    try:
        chunks = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=1,
            header=None,
            names=["XCen", "YCen", "Depth", "Vx", "Vy"],
            usecols=[0, 1, 2, 3, 4],
            chunksize=250000,
            engine="python",
            dtype=np.float64,
            on_bad_lines="error",
        )
        for chunk in chunks:
            if chunk.shape[1] != 5:
                raise ValueError(f"Unexpected column count in {path}: {chunk.shape[1]}")
            chunk = chunk.astype(np.float64)
            if chunk.isna().any().any():
                raise ValueError(f"Non-finite values encountered in {path}")

            chunk_x = chunk.iloc[:, 0].to_numpy(dtype=np.float64)
            chunk_y = chunk.iloc[:, 1].to_numpy(dtype=np.float64)
            valid_mask = (chunk_x >= xmin) & (chunk_x < xmax) & (chunk_y >= ymin) & (chunk_y < ymax)
            if not np.any(valid_mask):
                continue

            x_values.extend(chunk_x[valid_mask].tolist())
            y_values.extend(chunk_y[valid_mask].tolist())
            depth_values.extend(chunk.iloc[:, 2].to_numpy(dtype=np.float64)[valid_mask].tolist())
            u_values.extend(chunk.iloc[:, 3].to_numpy(dtype=np.float64)[valid_mask].tolist())
            v_values.extend(chunk.iloc[:, 4].to_numpy(dtype=np.float64)[valid_mask].tolist())
    except Exception as exc:  # pragma: no cover - exercised via runtime validation
        raise ValueError(f"Malformed RSL numeric data in {path}: {exc}") from exc

    if not x_values:
        raise ValueError(f"No cropped numeric data found in {path}")

    if expected_step_seconds <= 0:
        raise ValueError("Expected step seconds must be positive")

    x_array = np.asarray(x_values, dtype=np.float32)
    y_array = np.asarray(y_values, dtype=np.float32)
    depth_array = np.asarray(depth_values, dtype=np.float32)
    u_array = np.asarray(u_values, dtype=np.float32)
    v_array = np.asarray(v_values, dtype=np.float32)

    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError(f"Non-finite coordinate values encountered in {path}")
    if not np.isfinite(depth_array).all() or not np.isfinite(u_array).all() or not np.isfinite(v_array).all():
        raise ValueError(f"Non-finite hydraulic values encountered in {path}")
    if np.any(depth_array < 0.0):
        raise ValueError(f"Negative depth values encountered in {path}")

    return x_array, y_array, depth_array, u_array, v_array


def _validate_dem(dem_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    import rasterio
    from rasterio.transform import array_bounds

    with rasterio.open(dem_path) as src:
        if src.count != 1:
            raise ValueError(f"Expected a single band in DEM {dem_path}, found {src.count}")
        dem = src.read(1, out_dtype=np.float32)
        if dem.ndim != 2:
            raise ValueError(f"DEM raster is not 2D: {dem.shape}")
        transform = src.transform
        bounds = tuple(float(val) for val in src.bounds)
        height, width = dem.shape
        resolution = tuple(float(val) for val in src.res)
        crs = src.crs
        nodata = src.nodata

        if crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        if width <= 0 or height <= 0:
            raise ValueError(f"DEM has invalid shape: {dem.shape}")
        if transform.a == 0.0 or transform.e == 0.0:
            raise ValueError(f"DEM transform is invalid: {transform}")
        expected_bounds = tuple(float(val) for val in array_bounds(height, width, transform))
        if not np.allclose(bounds, expected_bounds, rtol=0.0, atol=1e-6):
            raise ValueError(f"DEM bounds are inconsistent with the transform: bounds={bounds}, expected={expected_bounds}")
        if not math.isclose(abs(transform.a), abs(resolution[0]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"DEM x resolution is inconsistent with the transform: transform.a={transform.a}, res={resolution[0]}")
        if not math.isclose(abs(transform.e), abs(resolution[1]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"DEM y resolution is inconsistent with the transform: transform.e={transform.e}, res={resolution[1]}")

        if nodata is not None:
            dem = dem.astype(np.float32, copy=True)
            dem[dem == nodata] = np.nan
        else:
            dem = dem.astype(np.float32, copy=True)

        if not np.isfinite(dem).all() and np.isnan(dem).any():
            # Keep the array finite for downstream use by masking invalid source nodata to NaN.
            pass

    metadata = {
        "height": int(height),
        "width": int(width),
        "transform": [float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)],
        "bounds": bounds,
        "resolution": [float(resolution[0]), float(resolution[1])],
        "crs": str(crs),
        "nodata": None if nodata is None else float(nodata),
    }
    return dem, metadata


def _select_rsl_files(rsl_dir: Path, start_index: int, end_index: int, expected_step_seconds: int) -> List[Dict[str, object]]:
    if start_index > end_index:
        raise ValueError(f"start-index {start_index} cannot be greater than end-index {end_index}")

    files = sorted([path for path in rsl_dir.glob("*.rsl") if path.is_file()], key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .rsl files found in {rsl_dir}")

    parsed: List[Dict[str, object]] = []
    seen_indices: set[int] = set()
    for path in files:
        item = _parse_rsl_filename(path)
        index = int(item["index"])
        if index in seen_indices:
            raise ValueError(f"Duplicate timestep index {index} found in {path}")
        seen_indices.add(index)
        minutes = int(item["minutes"])
        seconds = int(item["seconds"])
        expected_seconds = index * expected_step_seconds
        if abs(seconds - expected_seconds) > 1e-9:
            raise ValueError(
                f"Inconsistent filename time for index {index}: expected {expected_seconds}s from the timestep index, got {seconds}s"
            )
        if minutes * 60 != expected_seconds:
            raise ValueError(
                f"Inconsistent filename time for index {index}: expected {expected_seconds}s from the timestep index, got {minutes * 60}s"
            )
        parsed.append(item)

    parsed.sort(key=lambda item: int(item["index"]))
    requested_indices = list(range(start_index, end_index + 1))
    available = {int(item["index"]) for item in parsed}
    missing = [idx for idx in requested_indices if idx not in available]
    if missing:
        raise ValueError(f"Requested indices not found: {missing}")

    selected = [item for item in parsed if int(item["index"]) in requested_indices]
    if len(selected) != len(requested_indices):
        raise ValueError(f"Failed to select the full requested index range: requested={requested_indices}, selected={[int(item['index']) for item in selected]}")
    return selected


def _map_points_to_raster(
    x_values: np.ndarray,
    y_values: np.ndarray,
    dem: np.ndarray,
    transform,
    bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, Set[Tuple[int, int]]]:
    import rasterio

    height, width = dem.shape
    xmin, ymin, xmax, ymax = bounds
    rows = np.empty(len(x_values), dtype=np.int64)
    cols = np.empty(len(x_values), dtype=np.int64)
    valid_count = 0
    cells: Set[Tuple[int, int]] = set()
    for idx, (x, y) in enumerate(zip(x_values, y_values)):
        if not (xmin <= x < xmax and ymin <= y < ymax):
            continue
        row, col = rasterio.transform.rowcol(transform, x, y)
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"Off-grid point coordinates: x={x}, y={y} -> row={row}, col={col}")
        center_x, center_y = rasterio.transform.xy(transform, row, col)
        if not (abs(center_x - x) <= 1e-6 and abs(center_y - y) <= 1e-6):
            raise ValueError(f"Point coordinates do not align to pixel centres: x={x}, y={y}, centre_x={center_x}, centre_y={center_y}")
        cell = (int(row), int(col))
        if cell in cells:
            raise ValueError(f"Duplicate raster cell detected in the RSL timestep: {cell}")
        rows[valid_count] = row
        cols[valid_count] = col
        cells.add(cell)
        valid_count += 1

    if valid_count == 0:
        raise ValueError("No RSL points fell within the DEM extent")

    return rows[:valid_count], cols[:valid_count], cells


def _parse_rainfall(
    path: Path,
    selected_times_seconds: Sequence[int],
    expected_step_seconds: int,
    rain_alignment: str,
) -> Tuple[np.ndarray, List[float], Dict[int, float], List[Optional[int]]]:
    if expected_step_seconds <= 0:
        raise ValueError("Expected step seconds must be positive")
    rainfall_times: List[float] = []
    rainfall_values: List[float] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                time_seconds = float(parts[0])
                rainfall_value = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Malformed rainfall numeric line in {path}: {raw_line.rstrip()}") from exc
            if not np.isfinite(time_seconds) or not np.isfinite(rainfall_value):
                raise ValueError(f"Non-finite rainfall data in {path}: {raw_line.rstrip()}")
            if rainfall_value < 0.0:
                raise ValueError(f"Negative rainfall value in {path}: {raw_line.rstrip()}")
            rainfall_times.append(float(time_seconds))
            rainfall_values.append(float(rainfall_value))

    if not rainfall_times:
        raise ValueError(f"No numeric rainfall data found in {path}")
    rainfall_times_arr = np.asarray(rainfall_times, dtype=np.float64)
    rainfall_values_arr = np.asarray(rainfall_values, dtype=np.float32)
    if np.any(np.diff(np.sort(rainfall_times_arr)) <= 0.0):
        raise ValueError(f"Duplicate rainfall times found in {path}")
    sorted_times = np.sort(rainfall_times_arr)
    if len(sorted_times) > 1:
        diffs = np.diff(sorted_times)
        if not np.allclose(diffs, expected_step_seconds, rtol=0.0, atol=1e-9):
            raise ValueError("Rainfall times do not follow the expected interval")

    rainfall_lookup = {int(round(time_seconds)): float(value) for time_seconds, value in zip(rainfall_times_arr, rainfall_values_arr)}
    if len(rainfall_lookup) != len(rainfall_times_arr):
        raise ValueError(f"Duplicate rainfall times found in {path}")

    vector = np.zeros(len(selected_times_seconds), dtype=np.float32)
    rainfall_source_times_seconds: List[Optional[int]] = []
    for idx, selected_time_seconds in enumerate(selected_times_seconds):
        if rain_alignment == "legacy":
            required_time = int(round(selected_time_seconds))
        elif selected_time_seconds == 0:
            rainfall_source_times_seconds.append(None)
            continue
        else:
            required_time = int(round(selected_time_seconds - expected_step_seconds))
        if required_time not in rainfall_lookup:
            raise ValueError(f"Missing rainfall value for required time {required_time}s (alignment={rain_alignment})")
        vector[idx] = rainfall_lookup[required_time]
        rainfall_source_times_seconds.append(required_time)
    return vector, [float(value) for value in vector], rainfall_lookup, rainfall_source_times_seconds


def _assemble_tensor(
    h_array: np.ndarray,
    u_array: np.ndarray,
    v_array: np.ndarray,
    dem: np.ndarray,
    rainfall_vector: np.ndarray,
    hydraulic_mask: np.ndarray,
    barrier_offset: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    height, width, num_times = h_array.shape
    rainfall_spatial = np.broadcast_to(rainfall_vector.reshape(1, 1, -1), (height, width, num_times)).astype(np.float32)
    finite_dem = dem.astype(np.float32, copy=True)
    finite_dem[np.isnan(finite_dem)] = np.nan
    active_dem = finite_dem[hydraulic_mask]
    if active_dem.size == 0:
        raise ValueError("Hydraulic mask does not contain any active cells")
    if not np.isfinite(active_dem).all():
        raise ValueError("DEM contains non-finite values inside the hydraulic mask")
    barrier_elevation = float(np.max(active_dem)) + barrier_offset
    dem_out = np.where(hydraulic_mask, finite_dem, barrier_elevation).astype(np.float32)
    if not np.isfinite(dem_out).all():
        raise ValueError("DEM output contains non-finite values")

    output = np.stack(
        [
            h_array.astype(np.float32),
            u_array.astype(np.float32),
            v_array.astype(np.float32),
            rainfall_spatial,
            np.repeat(dem_out[..., None], num_times, axis=-1),
        ],
        axis=-1,
    )
    if output.shape[-1] != 5:
        raise ValueError(f"Unexpected number of output channels: {output.shape[-1]}")
    if not np.isfinite(output).all():
        raise ValueError("Output tensor contains non-finite values")
    return output, dem_out, barrier_elevation


def main() -> None:
    args = _parse_args()
    rsl_dir = Path(args.rsl_dir).expanduser().resolve()
    dem_path = Path(args.dem_path).expanduser().resolve()
    rainfall_path = Path(args.rainfall_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not rsl_dir.is_dir():
        raise FileNotFoundError(f"RSL directory not found: {rsl_dir}")
    dem_path = _require_file(dem_path)
    rainfall_path = _require_file(rainfall_path)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_items = _select_rsl_files(rsl_dir, args.start_index, args.end_index, args.expected_step_seconds)
    dem, dem_metadata = _validate_dem(dem_path)
    height, width = dem.shape
    transform = dem_metadata["transform"]
    bounds = tuple(dem_metadata["bounds"])

    import rasterio

    transform_obj = rasterio.transform.Affine(*transform)

    selected_times_seconds = [int(item["index"]) * args.expected_step_seconds for item in selected_items]
    selected_times_minutes = [int(sec // 60) for sec in selected_times_seconds]
    selected_filenames = [Path(item["path"]).name for item in selected_items]

    h_tensor = np.zeros((height, width, len(selected_items)), dtype=np.float32)
    u_tensor = np.zeros((height, width, len(selected_items)), dtype=np.float32)
    v_tensor = np.zeros((height, width, len(selected_items)), dtype=np.float32)
    hydraulic_mask = np.zeros((height, width), dtype=bool)
    reference_cells: Set[Tuple[int, int]] | None = None

    for t_idx, item in enumerate(selected_items):
        path = Path(item["path"])
        print(f"Processing timestep {int(item['index'])} ({path.name})")
        x_vals, y_vals, depth_vals, u_vals, v_vals = _load_rsl_timestep(path, bounds, args.expected_step_seconds)
        valid_rows, valid_cols, current_cells = _map_points_to_raster(
            x_vals,
            y_vals,
            dem,
            transform_obj,
            bounds,
        )

        if reference_cells is None:
            reference_cells = current_cells
            hydraulic_mask[valid_rows, valid_cols] = True
        else:
            if current_cells != reference_cells:
                missing = sorted(reference_cells - current_cells)
                extra = sorted(current_cells - reference_cells)
                raise ValueError(
                    f"Hydraulic mask changed at timestep {int(item['index'])}: "
                    f"missing_cells={len(missing)}, extra_cells={len(extra)}"
                )

        h_slice = np.zeros((height, width), dtype=np.float32)
        u_slice = np.zeros((height, width), dtype=np.float32)
        v_slice = np.zeros((height, width), dtype=np.float32)
        for row, col, depth_value, u_value, v_value in zip(valid_rows, valid_cols, depth_vals, u_vals, v_vals):
            if not np.isfinite(depth_value) or not np.isfinite(u_value) or not np.isfinite(v_value):
                raise ValueError(f"Non-finite hydraulic values encountered in {path}")
            h_slice[int(row), int(col)] = float(depth_value)
            u_slice[int(row), int(col)] = float(u_value)
            v_slice[int(row), int(col)] = float(v_value)

        h_tensor[..., t_idx] = h_slice
        u_tensor[..., t_idx] = u_slice
        v_tensor[..., t_idx] = v_slice

    if reference_cells is None:
        raise ValueError("No reference hydraulic cells were found")
    hydraulic_mask = np.zeros((height, width), dtype=bool)
    hydraulic_mask[np.array([row for row, _ in reference_cells]), np.array([col for _, col in reference_cells])] = True
    if not np.isfinite(h_tensor).all() or not np.isfinite(u_tensor).all() or not np.isfinite(v_tensor).all():
        raise ValueError("Hydraulic tensors contain non-finite values")

    rainfall_vector, rainfall_values, _, rainfall_source_times_seconds = _parse_rainfall(
        rainfall_path,
        selected_times_seconds,
        args.expected_step_seconds,
        args.rain_alignment,
    )

    output, dem_out, barrier_elevation = _assemble_tensor(
        h_tensor,
        u_tensor,
        v_tensor,
        dem,
        rainfall_vector,
        hydraulic_mask,
        args.barrier_offset,
    )

    output_tensor = torch.from_numpy(np.ascontiguousarray(output, dtype=np.float32))
    if output_tensor.shape != (height, width, len(selected_items), 5):
        raise ValueError(f"Unexpected output tensor shape: {tuple(output_tensor.shape)}")
    if output_tensor.dtype != torch.float32:
        raise ValueError(f"Unexpected output dtype: {output_tensor.dtype}")

    torch.save(output_tensor, output_path)

    metadata = {
        "source_rsl_dir": str(rsl_dir),
        "source_dem_path": str(dem_path),
        "source_rainfall_path": str(rainfall_path),
        "selected_rsl_filenames": selected_filenames,
        "selected_timestep_indices": [int(item["index"]) for item in selected_items],
        "selected_times_minutes": selected_times_minutes,
        "selected_times_seconds": selected_times_seconds,
        "tensor_shape": list(output_tensor.shape),
        "dtype": str(output_tensor.dtype),
        "channel_order": ["H", "U", "V", "rainfall", "DEM"],
        "rainfall_units": "m/s",
        "rainfall_alignment": args.rain_alignment,
        "rainfall_vector": rainfall_values,
        "rainfall_source_times_seconds": rainfall_source_times_seconds,
        "rainfall_intensity_sum": float(np.sum(rainfall_vector)),
        "rainfall_depth_m": float(np.sum(rainfall_vector) * args.expected_step_seconds),
        "rainfall_depth_mm": float(np.sum(rainfall_vector) * args.expected_step_seconds * 1000.0),
        "raster_width": int(width),
        "raster_height": int(height),
        "raster_bounds": list(bounds),
        "raster_resolution": list(dem_metadata["resolution"]),
        "affine_transform": dem_metadata["transform"],
        "crs": dem_metadata["crs"],
        "hydraulic_active_cell_count": int(np.count_nonzero(hydraulic_mask)),
        "hydraulic_inactive_cell_count": int(np.count_nonzero(~hydraulic_mask)),
        "hydraulic_total_cell_count": int(hydraulic_mask.size),
        "barrier_offset": float(args.barrier_offset),
        "barrier_elevation": float(barrier_elevation),
        "dem_min_inside_mask": float(np.nanmin(dem[hydraulic_mask])),
        "dem_max_inside_mask": float(np.nanmax(dem[hydraulic_mask])),
        "h_min": float(np.nanmin(h_tensor[hydraulic_mask])),
        "h_max": float(np.nanmax(h_tensor[hydraulic_mask])),
        "u_min": float(np.nanmin(u_tensor[hydraulic_mask])),
        "u_max": float(np.nanmax(u_tensor[hydraulic_mask])),
        "v_min": float(np.nanmin(v_tensor[hydraulic_mask])),
        "v_max": float(np.nanmax(v_tensor[hydraulic_mask])),
        "nan_count_per_channel": [int(np.isnan(output[..., channel_index]).sum()) for channel_index in range(5)],
        "output_tensor_size_mib": float(output_tensor.numel() * output_tensor.element_size() / (1024**2)),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Completed conversion: output={output_path} shape={tuple(output_tensor.shape)} "
        f"active_cells={int(np.count_nonzero(hydraulic_mask))} barrier={barrier_elevation:.3f}"
    )


if __name__ == "__main__":
    import torch

    main()
