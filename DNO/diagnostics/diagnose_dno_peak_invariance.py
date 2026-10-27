#!/usr/bin/env python3
"""Diagnose why DNO peak depths are near-invariant across Stockbridge test events.

Answers three questions with an already-trained checkpoint (no retraining):

1. Inputs  - what actually reaches the network? Is T0 dry in every event, and
             how large is the rainfall channel after the loader's transform?
2. Outputs - does the predicted peak sit in the same cell for every event, and
             how much does the prediction vary between events compared with
             CityCAT (across-event variance of max-depth maps)?
3. Sensitivity - how much does the prediction change when the rainfall input
             is zeroed or swapped with another event's rainfall?

Run from the repository root, for example:

    python DNO/diagnostics/diagnose_dno_peak_invariance.py \
        --test-dir ~/flood/02_data/CityCAT_Winchester/pt/stockbridge_phase2plus3_300_v01/test \
        --checkpoint ~/flood/03_outputs/UrbanFloodCast/report_figures/dno_stockbridge_300_v01/model_best.pt \
        --output-dir ~/flood/03_outputs/UrbanFloodCast/report_figures/dno_stockbridge_300_v01/peak_invariance_v01 \
        --label DNO-300

No rasterio needed: the DEM is read from channel 4 of the PT tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DNO.models.DNO import DNO  # noqa: E402
from DNO.utils25 import flood_data  # noqa: E402

T_IN = 1
T_OUT = 24
N_OUT = 3
CELL_SIZE_M = 5.0
RAIN_EPS = 1e-2  # must match flood_data.log_transform(eps=1e-2)
RAIN_SCALE = 10.0  # must match "/ 10.0" in flood_data.__getitem__
CH_H, CH_RAIN, CH_DEM = 0, 3, 4
WET_MAP_M = 0.01  # a cell counts as "flood-prone" if its mean CityCAT max depth exceeds this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", default="DNO", help="Label used in figure titles and file names")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--rain_input_scale", "--rain-input-scale", dest="rain_input_scale", type=float, default=None,
                   help="Rainfall multiplier used in training (read from the checkpoint when stored; legacy 1.0)")
    p.add_argument("--width", type=int, default=10, help="DNO width (Stockbridge baselines used 10)")
    p.add_argument(
        "--sensitivity-events", nargs="*", default=None,
        help="Event ids for the rainfall sensitivity test (default: lowest- and highest-rainfall events)",
    )
    return p.parse_args()


def resolve_rain_input_scale(checkpoint, cli_value):
    """Rain scaling must match training. New checkpoints store it; legacy ones imply 1.0."""
    stored = checkpoint.get("rain_input_scale") if isinstance(checkpoint, dict) else None
    if stored is None:
        stored = 1.0
    if cli_value is not None and float(cli_value) != float(stored):
        raise ValueError(
            f"--rain-input-scale {cli_value} differs from the value used in training ({stored})"
        )
    return float(stored)


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(choice)


def load_model(path: Path, width: int, device: torch.device) -> tuple[DNO, Any, Any]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    model = DNO(num_channels=5, width=width, initial_step=T_IN, pad=False, factor=1).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, epoch, ckpt


def rain_mmh_from_input(x2: np.ndarray, rain_scale: float) -> np.ndarray:
    """Invert x2 = log(1 + scale*p/eps)/10 and convert m/s -> mm/h (assumes tensors store m/s)."""
    p_ms = RAIN_EPS * np.expm1(RAIN_SCALE * x2.astype(np.float64)) / rain_scale
    return p_ms * 1000.0 * 3600.0


@torch.no_grad()
def predict_depth(model: DNO, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [1,H,W,24,1,5]; returns masked depth [H,W,24] on the model device."""
    h, w = x.shape[1], x.shape[2]
    out = model(x).reshape(1, h, w, T_OUT, N_OUT) * mask
    return out[0, ..., 0]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_pairwise_corr(stack: np.ndarray) -> float:
    """Mean off-diagonal Pearson correlation between rows of stack [n_events, n_cells]."""
    c = np.corrcoef(stack)
    n = c.shape[0]
    return float((c.sum() - np.trace(c)) / (n * (n - 1)))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    tag = args.label.replace(" ", "_")

    model, epoch, ckpt = load_model(args.checkpoint, args.width, device)
    rain_scale = resolve_rain_input_scale(ckpt, args.rain_input_scale)
    print(f"rain_input_scale: {rain_scale:g}")
    dataset = flood_data(path_root=str(args.test_dir), T_in=T_IN, T_out=T_OUT, train=False,
                         strategy="oneshot", rain_input_scale=rain_scale)
    n_events = len(dataset.data)
    if n_events < 2:
        raise ValueError("Need at least two test events to compare between events")

    raw_dem = torch.load(dataset.data[0], map_location="cpu")[..., 0, CH_DEM].numpy().astype(np.float64)

    records: list[dict[str, Any]] = []
    pred_max_maps, obs_max_maps, rain_vectors = [], [], {}
    active = None
    input_ranges = {"rain_input_max": 0.0, "dem_input_min": np.inf, "dem_input_max": -np.inf}

    for i in range(n_events):
        event_id = Path(dataset.data[i]).stem
        x, y, mask = dataset[i]
        x, y, mask = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)

        # ---- inputs actually seen by the network ----
        t0_depth = x[0, ..., 0, 0, CH_H].cpu().numpy()  # T0 depth (identical across the 24 repeats)
        rain_in = x[0, ..., 0, CH_RAIN].cpu().numpy()  # [H,W,24]
        rain_vec = rain_in.reshape(-1, T_OUT).mean(axis=0)
        rain_spatial_spread = float(np.abs(rain_in - rain_vec).max())
        rain_vectors[event_id] = rain_vec
        rain_mmh = rain_mmh_from_input(rain_vec, rain_scale)
        dem_in = x[0, ..., 0, CH_DEM].cpu().numpy()
        input_ranges["rain_input_max"] = max(input_ranges["rain_input_max"], float(rain_in.max()))
        input_ranges["dem_input_min"] = min(input_ranges["dem_input_min"], float(dem_in.min()))
        input_ranges["dem_input_max"] = max(input_ranges["dem_input_max"], float(dem_in.max()))

        # ---- outputs ----
        valid = mask[0, ..., 0].bool()  # [H,W,24]
        pred = predict_depth(model, x, mask)
        obs = (y * mask)[0, ..., 0]
        pred_max = pred.amax(dim=-1).cpu().numpy()
        obs_max = obs.amax(dim=-1).cpu().numpy()
        valid_map = valid.any(dim=-1).cpu().numpy()
        active = valid_map if active is None else (active | valid_map)
        pred_max_maps.append(pred_max.astype(np.float32))
        obs_max_maps.append(obs_max.astype(np.float32))

        pr, pc = np.unravel_index(np.argmax(np.where(valid_map, pred_max, -np.inf)), pred_max.shape)
        orow, ocol = np.unravel_index(np.argmax(np.where(valid_map, obs_max, -np.inf)), obs_max.shape)
        pmax, omax = float(pred_max[pr, pc]), float(obs_max[orow, ocol])
        final_pred = pred[..., -1].cpu().numpy()
        final_obs = obs[..., -1].cpu().numpy()

        records.append({
            "event_id": event_id,
            "t0_depth_max_m": float(np.nanmax(t0_depth)),
            "t0_wet_cells_gt_1cm": int((t0_depth > 0.01).sum()),
            "rain_input_max": float(rain_in.max()),
            "rain_input_spatial_spread": rain_spatial_spread,
            "rain_peak_mm_h": float(rain_mmh.max()),
            "rain_total_mm": float(rain_mmh.sum() * 5.0 / 60.0),
            "obs_max_m": omax,
            "pred_max_m": pmax,
            "obs_peak_row": int(orow), "obs_peak_col": int(ocol),
            "pred_peak_row": int(pr), "pred_peak_col": int(pc),
            "peak_distance_m": float(np.hypot(pr - orow, pc - ocol) * CELL_SIZE_M),
            "dem_at_pred_peak_m": float(raw_dem[pr, pc]),
            "dem_at_obs_peak_m": float(raw_dem[orow, ocol]),
            "pred_cells_within_1cm_of_pred_max": int((pred_max >= pmax - 0.01).sum()),
            "obs_final_volume_m3": float(final_obs.sum() * CELL_SIZE_M**2),
            "pred_final_volume_m3": float(final_pred.sum() * CELL_SIZE_M**2),
            "obs_area_max_gt_10cm_m2": float((obs_max > 0.10).sum() * CELL_SIZE_M**2),
            "pred_area_max_gt_10cm_m2": float((pred_max > 0.10).sum() * CELL_SIZE_M**2),
        })
        print(f"[{i + 1}/{n_events}] {event_id}: obs max {omax:.3f} m, pred max {pmax:.4f} m "
              f"at ({pr},{pc}); rain peak {rain_mmh.max():.1f} mm/h -> input {rain_in.max():.2e}")

    pred_stack = np.stack(pred_max_maps)  # [E,H,W]
    obs_stack = np.stack(obs_max_maps)
    act = active.astype(bool)
    floodprone = act & (obs_stack.mean(axis=0) > WET_MAP_M)
    pred_std = pred_stack.std(axis=0)
    obs_std = obs_stack.std(axis=0)

    # ---- sensitivity: zero / swap rainfall, zero T0 ----
    by_total = sorted(records, key=lambda r: r["rain_total_mm"])
    sens_ids = args.sensitivity_events or [by_total[0]["event_id"], by_total[-1]["event_id"]]
    id_to_idx = {Path(p).stem: k for k, p in enumerate(dataset.data)}
    sens_rows = []
    for event_id in sens_ids:
        if event_id not in id_to_idx:
            print(f"WARNING: sensitivity event {event_id} not in test set, skipped")
            continue
        k = id_to_idx[event_id]
        other_id = by_total[-1]["event_id"] if event_id != by_total[-1]["event_id"] else by_total[0]["event_id"]
        x, y, mask = dataset[k]
        x, mask = x.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
        base = predict_depth(model, x, mask)
        obs_self = obs_stack[k]
        obs_other = obs_stack[id_to_idx[other_id]]
        obs_change = float(np.linalg.norm(obs_other - obs_self) / max(np.linalg.norm(obs_self), 1e-12))

        variants = {}
        xz = x.clone(); xz[..., CH_RAIN] = 0.0
        variants["rain_zeroed"] = xz
        xs = x.clone()
        xs[..., CH_RAIN] = torch.as_tensor(rain_vectors[other_id], device=device).view(1, 1, 1, T_OUT, 1)
        variants[f"rain_swapped_with_{other_id}"] = xs
        xt = x.clone(); xt[..., 0:3] = 0.0
        variants["t0_state_zeroed"] = xt

        for name, xv in variants.items():
            out = predict_depth(model, xv, mask)
            rel = float(torch.linalg.norm(out - base) / torch.linalg.norm(base).clamp_min(1e-12))
            sens_rows.append({
                "event_id": event_id, "perturbation": name,
                "relative_change_pred_depth": rel,
                "pred_max_before_m": float(base.max()), "pred_max_after_m": float(out.max()),
                "citycat_relative_change_max_map_between_these_events": obs_change if "swapped" in name else "",
            })

    # ---- summary statistics ----
    obs_peaks = np.array([r["obs_max_m"] for r in records])
    pred_peaks = np.array([r["pred_max_m"] for r in records])
    rain_tot = np.array([r["rain_total_mm"] for r in records])
    pred_vol = np.array([r["pred_final_volume_m3"] for r in records])
    obs_vol = np.array([r["obs_final_volume_m3"] for r in records])
    peak_cells = {(r["pred_peak_row"], r["pred_peak_col"]) for r in records}
    flat_pred = pred_stack[:, floodprone]
    flat_obs = obs_stack[:, floodprone]

    summary = {
        "label": args.label, "checkpoint": str(args.checkpoint), "checkpoint_epoch": epoch,
        "rain_input_scale": rain_scale,
        "n_events": n_events,
        "inputs": {
            "events_with_dry_T0 (no cell > 1 cm)": int(sum(r["t0_wet_cells_gt_1cm"] == 0 for r in records)),
            "max_T0_depth_any_event_m": float(max(r["t0_depth_max_m"] for r in records)),
            "rain_input_max_after_transform": input_ranges["rain_input_max"],
            "rain_peak_mm_h_range": [float(min(r["rain_peak_mm_h"] for r in records)),
                                     float(max(r["rain_peak_mm_h"] for r in records))],
            "dem_input_range_after_normalize": [input_ranges["dem_input_min"], input_ranges["dem_input_max"]],
        },
        "peaks": {
            "pred_max_mean_m": float(pred_peaks.mean()), "pred_max_std_m": float(pred_peaks.std()),
            "pred_max_range_m": [float(pred_peaks.min()), float(pred_peaks.max())],
            "obs_max_mean_m": float(obs_peaks.mean()), "obs_max_std_m": float(obs_peaks.std()),
            "obs_max_range_m": [float(obs_peaks.min()), float(obs_peaks.max())],
            "distinct_pred_peak_cells": len(peak_cells),
            "distinct_obs_peak_cells": len({(r["obs_peak_row"], r["obs_peak_col"]) for r in records}),
            "corr_obs_peak_vs_pred_peak": pearson(obs_peaks, pred_peaks),
        },
        "between_event_variability": {
            "floodprone_cells": int(floodprone.sum()),
            "total_variance_ratio_pred_over_obs": float(flat_pred.var(axis=0).sum() / max(flat_obs.var(axis=0).sum(), 1e-12)),
            "median_cell_std_ratio_pred_over_obs": float(np.median(pred_std[floodprone] / np.maximum(obs_std[floodprone], 1e-6))),
            "mean_pairwise_corr_pred_max_maps": mean_pairwise_corr(flat_pred),
            "mean_pairwise_corr_obs_max_maps": mean_pairwise_corr(flat_obs),
        },
        "rainfall_response": {
            "corr_rain_total_vs_obs_final_volume": pearson(rain_tot, obs_vol),
            "corr_rain_total_vs_pred_final_volume": pearson(rain_tot, pred_vol),
            "pred_volume_cv": float(pred_vol.std() / max(pred_vol.mean(), 1e-12)),
            "obs_volume_cv": float(obs_vol.std() / max(obs_vol.mean(), 1e-12)),
        },
        "sensitivity": sens_rows,
    }

    # ---- write files ----
    with (args.output_dir / f"{tag}_peak_invariance_per_event.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    with (args.output_dir / f"{tag}_rainfall_sensitivity.csv").open("w", newline="") as fh:
        if sens_rows:
            w = csv.DictWriter(fh, fieldnames=list(sens_rows[0].keys()))
            w.writeheader(); w.writerows(sens_rows)
    (args.output_dir / f"{tag}_peak_invariance_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output_dir / f"{tag}_max_depth_stacks.npz",
        event_ids=np.array([r["event_id"] for r in records]),
        pred_max=pred_stack, obs_max=obs_stack, active=act, floodprone=floodprone,
    )

    # ---- figure ----
    dem_show = np.ma.masked_invalid(np.where(act, raw_dem, np.nan))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    ax = axes[0, 0]
    ax.scatter(obs_peaks, pred_peaks, s=22, color="#2b6cb0")
    lim = max(obs_peaks.max(), pred_peaks.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="CityCAT event max depth (m)",
           ylabel=f"{args.label} event max depth (m)", title="Event peak: CityCAT vs DNO")

    ax = axes[0, 1]
    ax.imshow(dem_show, cmap="Greys", origin="upper")
    ax.scatter([r["obs_peak_col"] for r in records], [r["obs_peak_row"] for r in records],
               s=40, facecolors="none", edgecolors="#c53030", label="CityCAT peak cell")
    ax.scatter([r["pred_peak_col"] for r in records], [r["pred_peak_row"] for r in records],
               s=14, marker="x", color="#2b6cb0", label=f"{args.label} peak cell")
    ax.set_title(f"Peak locations ({len(peak_cells)} distinct DNO cells, {n_events} events)")
    ax.legend(loc="lower left", fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 2]
    ax.scatter(rain_tot, obs_vol, s=22, color="#c53030", label="CityCAT")
    ax.scatter(rain_tot, pred_vol, s=22, color="#2b6cb0", label=args.label)
    ax.set(xlabel="Event rainfall total (mm, 0-120 min)", ylabel="Stored water at T24 (m³)",
           title="Response to rainfall amount")
    ax.legend(fontsize=8)

    vmax = float(np.nanpercentile(obs_std[floodprone], 99)) if floodprone.any() else 1.0
    for ax, arr, name in ((axes[1, 0], obs_std, "CityCAT"), (axes[1, 1], pred_std, args.label)):
        im = ax.imshow(np.ma.masked_where(~act, arr), cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(f"{name}: std of max depth across events (m)"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1, 2]
    if sens_rows:
        labels = [f"{s['event_id']}\n{s['perturbation'].replace('_with_', ' ↔ ').replace('_', ' ')}" for s in sens_rows]
        floor = 1e-8  # log axis: show exact zeros at the floor
        vals = [max(s["relative_change_pred_depth"], floor) for s in sens_rows]
        ax.barh(range(len(sens_rows)), vals, left=floor, color="#2b6cb0")
        ax.set_xscale("log")
        for k, s in enumerate(sens_rows):
            ax.text(vals[k], k, f" {s['relative_change_pred_depth']:.1e}", va="center", fontsize=7)
        ax.set_yticks(range(len(sens_rows))); ax.set_yticklabels(labels, fontsize=7)
        swaps = [s for s in sens_rows if s["citycat_relative_change_max_map_between_these_events"] != ""]
        if swaps:
            ax.axvline(swaps[0]["citycat_relative_change_max_map_between_these_events"], color="#c53030",
                       ls="--", lw=1, label="CityCAT difference between\nthe two swapped events")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel("Relative change in predicted depth (L2)")
        ax.set_title("Input sensitivity")
    fig.suptitle(f"{args.label} peak-invariance diagnostics (epoch {epoch}, {n_events} test events)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{tag}_peak_invariance.png", dpi=150)

    b, s, i_ = summary["between_event_variability"], summary["peaks"], summary["inputs"]
    print("\n=== Summary ===")
    print(f"Dry T0 in {i_['events_with_dry_T0 (no cell > 1 cm)']}/{n_events} events; "
          f"max rainfall input after transform = {i_['rain_input_max_after_transform']:.2e}")
    print(f"Pred peak {s['pred_max_mean_m']:.4f} ± {s['pred_max_std_m']:.4f} m in "
          f"{s['distinct_pred_peak_cells']} distinct cells; CityCAT {s['obs_max_mean_m']:.3f} ± {s['obs_max_std_m']:.3f} m")
    print(f"Between-event variance ratio (pred/obs) = {b['total_variance_ratio_pred_over_obs']:.3g}; "
          f"pairwise corr of max maps: pred {b['mean_pairwise_corr_pred_max_maps']:.3f} vs obs {b['mean_pairwise_corr_obs_max_maps']:.3f}")
    for row in sens_rows:
        print(f"  {row['event_id']} {row['perturbation']}: relative change {row['relative_change_pred_depth']:.2e}")
    print(f"Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
