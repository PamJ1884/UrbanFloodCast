#!/usr/bin/env python3
"""Independent, read-only QC for the three Stockbridge Phase 3 DNO smoke tensors."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import torch


DATA = Path.home() / "flood/02_data/CityCAT_Winchester"
REFERENCE_PT = DATA / "pt/stockbridge_phase2_core_v01/train/R001_stockbridge_T000_T024.pt"
SMOKE_PT = DATA / "smoke/phase3_dno_smoke_v01/pt"
RAINFALL = DATA / "rainfall/stockbridge_phase3_v01/rainfall_data_phase3"
EVENTS = {"R101": ("triangular", "triangular"),
          "R142": ("gaussian", "central"),
          "R221": ("fsr_feh", "summer")}
SHAPE = (305, 326, 25, 5)
TIMES = list(range(0, 7201, 300))
CHANNELS = ["H", "U", "V", "rainfall", "DEM"]
BARRIER = 152.61531829833984
ACTIVE, INACTIVE, TOTAL = 32249, 67181, 99430
THRESHOLDS = (0.00, 0.01, 0.05, 0.10)


class QCError(ValueError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise QCError(message)


def load_tensor(path: Path) -> torch.Tensor:
    check(path.is_file(), f"missing tensor: {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    check(isinstance(tensor, torch.Tensor), f"not a torch.Tensor: {path}")
    check(tuple(tensor.shape) == SHAPE, f"wrong shape in {path}: {tuple(tensor.shape)}")
    check(tensor.dtype == torch.float32, f"wrong dtype in {path}: {tensor.dtype}")
    check(bool(torch.isfinite(tensor).all()), f"non-finite tensor values in {path}")
    return tensor


def load_reference() -> tuple[torch.Tensor, torch.Tensor]:
    sidecar_path = REFERENCE_PT.with_suffix(".pt.json")
    check(sidecar_path.is_file(), f"missing Phase 2 sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    check(sidecar.get("barrier_elevation") == BARRIER,
          f"Phase 2 barrier elevation differs: {sidecar.get('barrier_elevation')!r}")
    tensor = load_tensor(REFERENCE_PT)
    dem = tensor[..., 4].clone()
    check(torch.equal(dem, dem[..., :1].expand_as(dem)),
          "Phase 2 reference DEM changes over time")
    barrier_float32 = torch.tensor(BARRIER, dtype=torch.float32).item()
    inactive = dem[..., 0] == barrier_float32
    inactive_count = int(inactive.sum())
    active_count = inactive.numel() - inactive_count
    check((active_count, inactive_count, inactive.numel()) == (ACTIVE, INACTIVE, TOTAL),
          f"Phase 2 spatial mask counts differ: {active_count}/{inactive_count}/{inactive.numel()}")
    return dem, inactive


def parse_rainfall(path: Path, event_id: str, family: str, subtype: str):
    check(path.is_file(), f"{event_id}: missing rainfall file: {path}")
    header: dict[str, str] = {}
    declared_count = None
    records: list[tuple[float, float]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if parts[0] == "*":
            for token in parts:
                if "=" in token:
                    key, value = token.split("=", 1)
                    header[key] = value
            continue
        if len(parts) == 1 and not records and declared_count is None:
            try:
                declared_count = int(parts[0])
            except ValueError as exc:
                raise QCError(f"{event_id}: invalid rainfall count at line {line_no}") from exc
            continue
        check(len(parts) == 2, f"{event_id}: malformed rainfall line {line_no}")
        try:
            time, value = map(float, parts)
        except ValueError as exc:
            raise QCError(f"{event_id}: non-numeric rainfall line {line_no}") from exc
        check(math.isfinite(time) and math.isfinite(value) and value >= 0,
              f"{event_id}: non-finite or negative rainfall at line {line_no}")
        records.append((time, value))
    check(declared_count == 25 and len(records) == 25,
          f"{event_id}: expected 25 declared and numeric rainfall records; got {declared_count}/{len(records)}")
    check([time for time, _ in records] == TIMES,
          f"{event_id}: rainfall times are not 0,300,...,7200 seconds")
    for key, expected in (("backbone", "B01"), ("family", family), ("subtype", subtype)):
        check(header.get(key) == expected,
              f"{event_id}: rainfall header {key}={header.get(key)!r}, expected {expected!r}")
    for key, expected in (("D_min", 15), ("T_years", 2), ("depth_mm", 8.3)):
        try:
            actual = float(header[key])
        except (KeyError, ValueError) as exc:
            raise QCError(f"{event_id}: invalid rainfall header {key}") from exc
        check(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9),
              f"{event_id}: rainfall header {key}={actual}, expected {expected}")
    values = [value for _, value in records]
    depth_mm = math.fsum(values) * 300 * 1000
    check(math.isclose(depth_mm, 8.3, rel_tol=0, abs_tol=1e-3),
          f"{event_id}: integrated source rainfall depth {depth_mm:.9f} mm is not approximately 8.3 mm")
    peak_index = max(range(25), key=values.__getitem__)
    expected_future = torch.tensor([0.0, *values[:-1]], dtype=torch.float32)
    return expected_future, depth_mm, values[peak_index] * 3_600_000, peak_index


def check_event(event_id: str, family: str, subtype: str,
                reference_dem: torch.Tensor, inactive: torch.Tensor) -> dict:
    pt_path = SMOKE_PT / f"{event_id}_stockbridge_T000_T024.pt"
    tensor = load_tensor(pt_path)
    h, vx, vy, rain, dem = (tensor[..., index] for index in range(5))
    check(bool(torch.all(h >= 0)), f"{event_id}: negative H values")
    check(torch.equal(dem, dem[..., :1].expand_as(dem)),
          f"{event_id}: DEM changes over time")
    check(torch.equal(dem, reference_dem),
          f"{event_id}: DEM differs pixel-by-pixel from Phase 2")
    check(not bool(torch.any(tensor[..., :3][inactive] != 0)),
          f"{event_id}: H/U/V nonzero on inactive cells")
    rainfall_path = RAINFALL / f"Rainfall_Data_{int(event_id[1:])}.txt"
    expected_rain, depth_mm, peak_mmh, peak_index = parse_rainfall(
        rainfall_path, event_id, family, subtype)
    check(torch.equal(rain, rain[:1, :1, :].expand_as(rain)),
          f"{event_id}: rainfall is not spatially uniform")
    check(torch.equal(rain, expected_rain.view(1, 1, 25).expand_as(rain)),
          f"{event_id}: rainfall channel differs from independently aligned source rainfall")
    active = ~inactive
    flooded = {}
    for threshold in THRESHOLDS:
        counts = (h[active] > threshold).sum(dim=0)
        flooded[threshold] = (int(counts.max()), int(counts[-1]))
    def bounds(channel: torch.Tensor) -> tuple[float, float]:
        cells = channel[active]
        return float(cells.min()), float(cells.max())
    return {"family": family, "h": h.clone(), "rain": expected_rain,
            "h_bounds": bounds(h), "vx_bounds": bounds(vx), "vy_bounds": bounds(vy),
            "flooded": flooded, "depth_mm": depth_mm, "peak_mmh": peak_mmh,
            "peak_index": peak_index}


def check_sidecar(event_id: str) -> None:
    path = SMOKE_PT / f"{event_id}_stockbridge_T000_T024.pt.json"
    check(path.is_file(), f"{event_id}: missing sidecar: {path}")
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    expected = {"channel_order": CHANNELS, "rainfall_units": "m/s",
                "rainfall_alignment": "future", "selected_timestep_indices": list(range(25)),
                "selected_times_seconds": TIMES, "hydraulic_active_cell_count": ACTIVE,
                "hydraulic_inactive_cell_count": INACTIVE,
                "hydraulic_total_cell_count": TOTAL, "barrier_elevation": BARRIER}
    for key, value in expected.items():
        check(sidecar.get(key) == value,
              f"{event_id}: sidecar {key}={sidecar.get(key)!r}, expected {value!r}")


def print_event_tables(results: dict[str, dict]) -> None:
    print("Event | Family | H[min,max] m | Vx[min,max] m/s | Vy[min,max] m/s | "
          "Max >1cm | T24 >1cm | Rain mm | Peak mm/h | Peak at")
    for event_id, result in results.items():
        fmt = lambda pair: f"[{pair[0]:.4f},{pair[1]:.4f}]"
        max_flooded, t24_flooded = result["flooded"][0.01]
        peak = result["peak_index"]
        print(f"{event_id} | {result['family']} | {fmt(result['h_bounds'])} | "
              f"{fmt(result['vx_bounds'])} | {fmt(result['vy_bounds'])} | "
              f"{max_flooded} | {t24_flooded} | {result['depth_mm']:.4f} | "
              f"{result['peak_mmh']:.4f} | T{peak} ({TIMES[peak]}s)")
    print("\nFlooded active cells (strict H > threshold):")
    print("Event | Threshold m | Max across T0-T24 | At T24")
    for event_id, result in results.items():
        for threshold in THRESHOLDS:
            maximum, final = result["flooded"][threshold]
            print(f"{event_id} | {threshold:.2f} | {maximum} | {final}")


def compare_pairs(results: dict[str, dict], failures: list[str]) -> None:
    print("\nPair | Equal H? | Max |dH| m | Mean |dH| m | Equal rainfall?")
    for first, second in (("R101", "R142"), ("R101", "R221"), ("R142", "R221")):
        if first not in results or second not in results:
            print(f"{first}-{second} | unavailable")
            continue
        h_first, h_second = results[first]["h"], results[second]["h"]
        equal = torch.equal(h_first, h_second)
        difference = torch.abs(h_first - h_second)
        maximum, mean = float(difference.max()), float(difference.mean())
        rain_equal = torch.equal(results[first]["rain"], results[second]["rain"])
        print(f"{first}-{second} | {equal} | {maximum:.9g} | {mean:.9g} | {rain_equal}")
        if equal or maximum <= 0:
            failures.append(f"{first}-{second}: hydraulic H tensors are identical")


def main() -> int:
    failures: list[str] = []
    try:
        reference_dem, inactive = load_reference()
    except Exception as exc:
        print(f"QC FAIL: Phase 2 reference: {exc}", file=sys.stderr)
        return 1
    results: dict[str, dict] = {}
    for event_id, (family, subtype) in EVENTS.items():
        try:
            results[event_id] = check_event(event_id, family, subtype,
                                            reference_dem, inactive)
        except Exception as exc:
            failures.append(f"{event_id}: {exc}")
        try:
            check_sidecar(event_id)
        except Exception as exc:
            failures.append(f"{event_id}: {exc}")
    if results:
        print_event_tables(results)
    compare_pairs(results, failures)
    if len(results) != 3:
        failures.append("pairwise hydraulic comparisons incomplete")
    if failures:
        print("\nQC FAIL:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    for message in ("canonical tensor contract",
                    "Phase 3 DEM equals Phase 2 pixel-by-pixel",
                    "active/inactive spatial mask matches Phase 2",
                    "rainfall future alignment independently verified",
                    "rainfall depths independently verified",
                    "hydraulic responses differ across rainfall profile families"):
        print(f"PASS: {message}")
    print("QC PASS: 3/3 events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
