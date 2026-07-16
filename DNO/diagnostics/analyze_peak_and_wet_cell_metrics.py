#!/usr/bin/env python3
"""Compare DNO training outputs by analyzing peak depth and wet-cell metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


CHANNEL_NAMES = ["H", "U", "V"]
CHANNEL_INDEX = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze peak depth and wet-cell metrics for DNO runs")
    parser.add_argument("--runs", nargs="+", required=True, help="One or more training output directories")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for the runs, in the same order")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic outputs")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10],
        help="Wet-cell thresholds to evaluate",
    )
    parser.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.05],
        help="Top-depth fractions for evaluation",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_tensor_layout(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} has a leading dimension {tensor.shape[0]}, expected 1")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be 3D after normalization, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] != 3:
        raise ValueError(f"{name} must end in 3 channels, got shape {tuple(tensor.shape)}")
    return tensor


def reduce_spatial_mask(spatial_mask: torch.Tensor) -> torch.Tensor:
    mask = spatial_mask.detach().cpu().bool()
    while mask.ndim > 2:
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.shape[0] == 1:
            mask = mask.squeeze(0)
        else:
            mask = mask.any(dim=0)
    if mask.ndim != 2:
        raise ValueError(f"Spatial mask must reduce to 2D, got shape {tuple(mask.shape)}")
    return mask


def load_run_data(run_dir: Path) -> Dict[str, Any]:
    predictions_path = run_dir / "test_predictions.pt"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing test_predictions.pt in {run_dir}")
    saved = torch.load(predictions_path, map_location="cpu")
    if not isinstance(saved, dict):
        raise TypeError(f"Expected a dict in {predictions_path}, found {type(saved).__name__}")

    required_keys = ["prediction_valid", "target_valid", "spatial_mask"]
    for key in required_keys:
        if key not in saved:
            raise KeyError(f"Missing key '{key}' in {predictions_path}")

    prediction = normalize_tensor_layout(saved["prediction_valid"], "prediction_valid")
    target = normalize_tensor_layout(saved["target_valid"], "target_valid")
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")

    mask = reduce_spatial_mask(saved["spatial_mask"])
    valid_rows, valid_cols = torch.where(mask)
    if valid_rows.numel() != prediction.shape[0]:
        raise ValueError(
            f"Number of valid cells from spatial mask ({valid_rows.numel()}) does not match prediction rows ({prediction.shape[0]})"
        )

    return {
        "prediction_valid": prediction,
        "target_valid": target,
        "spatial_mask": mask,
        "valid_rows": valid_rows,
        "valid_cols": valid_cols,
    }


def compute_full_domain_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    flat_pred = prediction.reshape(-1, 3)
    flat_target = target.reshape(-1, 3)
    diff = flat_pred - flat_target
    count = float(diff.numel())
    if count <= 0:
        raise ValueError("No values available for full-domain metrics")

    metrics: Dict[str, float] = {
        "mse": (diff.square().sum() / count).item(),
        "rmse": math.sqrt((diff.square().sum() / count).item()),
        "mae": (diff.abs().sum() / count).item(),
        "bias": (diff.sum() / count).item(),
    }
    for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
        channel_diff = diff[:, channel_idx]
        channel_count = float(channel_diff.numel())
        metrics[f"{channel_name.lower()}_mse"] = (channel_diff.square().sum() / channel_count).item()
        metrics[f"{channel_name.lower()}_rmse"] = math.sqrt((channel_diff.square().sum() / channel_count).item())
        metrics[f"{channel_name.lower()}_mae"] = (channel_diff.abs().sum() / channel_count).item()
        metrics[f"{channel_name.lower()}_bias"] = (channel_diff.sum() / channel_count).item()
    return metrics


def compute_wet_cell_metrics(prediction: torch.Tensor, target: torch.Tensor, thresholds: Sequence[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    flat_pred = prediction.reshape(-1, 3)
    flat_target = target.reshape(-1, 3)
    target_h = flat_target[:, CHANNEL_INDEX["H"]]

    for threshold in thresholds:
        selected = target_h > threshold
        if not selected.any():
            row = {"threshold": threshold, "wet_fraction": 0.0, "n_values": 0}
            for channel_name in CHANNEL_NAMES:
                row[f"{channel_name.lower()}_mse"] = math.nan
                row[f"{channel_name.lower()}_rmse"] = math.nan
                row[f"{channel_name.lower()}_mae"] = math.nan
                row[f"{channel_name.lower()}_bias"] = math.nan
            rows.append(row)
            continue

        selected_pred = flat_pred[selected]
        selected_target = flat_target[selected]
        diff = selected_pred - selected_target
        wet_fraction = float(selected.sum().item() / selected.numel())
        row = {"threshold": threshold, "wet_fraction": wet_fraction, "n_values": int(selected.sum().item())}
        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            channel_diff = diff[:, channel_idx]
            channel_count = float(channel_diff.numel())
            row[f"{channel_name.lower()}_mse"] = (channel_diff.square().sum() / channel_count).item()
            row[f"{channel_name.lower()}_rmse"] = math.sqrt((channel_diff.square().sum() / channel_count).item())
            row[f"{channel_name.lower()}_mae"] = (channel_diff.abs().sum() / channel_count).item()
            row[f"{channel_name.lower()}_bias"] = (channel_diff.sum() / channel_count).item()
        rows.append(row)
    return rows


def compute_peak_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_rows: torch.Tensor,
    valid_cols: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for time_idx in range(target.shape[1]):
        target_h = target[:, time_idx, CHANNEL_INDEX["H"]]
        pred_h = prediction[:, time_idx, CHANNEL_INDEX["H"]]

        target_max = target_h.max().item()
        pred_max = pred_h.max().item()
        target_idx = int(target_h.argmax().item())
        pred_idx = int(pred_h.argmax().item())

        target_row = int(valid_rows[target_idx].item())
        target_col = int(valid_cols[target_idx].item())
        pred_row = int(valid_rows[pred_idx].item())
        pred_col = int(valid_cols[pred_idx].item())

        signed_error = pred_max - target_max
        abs_error = abs(signed_error)
        peak_ratio = pred_max / target_max if target_max > 0 else math.nan
        rows.append({
            "time_index": time_idx,
            "target_max_h": target_max,
            "prediction_max_h": pred_max,
            "signed_peak_error": signed_error,
            "absolute_peak_error": abs_error,
            "peak_ratio": peak_ratio,
            "target_peak_row": target_row,
            "target_peak_col": target_col,
            "prediction_peak_row": pred_row,
            "prediction_peak_col": pred_col,
            "same_peak_cell": (target_row, target_col) == (pred_row, pred_col),
        })
    return rows


def compute_top_depth_metrics(prediction: torch.Tensor, target: torch.Tensor, top_fraction: float) -> Dict[str, Any]:
    target_h_flat = target[..., CHANNEL_INDEX["H"]].reshape(-1)
    pred_h_flat = prediction[..., CHANNEL_INDEX["H"]].reshape(-1)
    k = max(1, int(round(target_h_flat.numel() * top_fraction)))
    _, top_indices = torch.topk(target_h_flat, k=k)
    target_selected = target_h_flat[top_indices]
    pred_selected = pred_h_flat[top_indices]
    diff = pred_selected - target_selected
    count = float(diff.numel())
    return {
        "top_fraction": top_fraction,
        "n_values": int(count),
        "mse": (diff.square().sum() / count).item(),
        "rmse": math.sqrt((diff.square().sum() / count).item()),
        "mae": (diff.abs().sum() / count).item(),
        "bias": (diff.sum() / count).item(),
    }


def compute_flood_extent_metrics(prediction: torch.Tensor, target: torch.Tensor, thresholds: Sequence[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target_h = target[..., CHANNEL_INDEX["H"]].reshape(-1)
    pred_h = prediction[..., CHANNEL_INDEX["H"]].reshape(-1)
    for threshold in thresholds:
        target_binary = target_h > threshold
        pred_binary = pred_h > threshold
        true_positive = int(((target_binary) & (pred_binary)).sum().item())
        false_positive = int((~target_binary & pred_binary).sum().item())
        false_negative = int((target_binary & ~pred_binary).sum().item())
        true_negative = int((~target_binary & ~pred_binary).sum().item())

        total_positive = true_positive + false_negative
        total_pred_positive = true_positive + false_positive
        precision = true_positive / total_pred_positive if total_pred_positive else math.nan
        recall = true_positive / total_positive if total_positive else math.nan
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else math.nan
        csi = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) else math.nan
        false_alarm_ratio = false_positive / total_pred_positive if total_pred_positive else math.nan
        miss_rate = false_negative / total_positive if total_positive else math.nan
        target_wet_fraction = float(target_binary.sum().item() / target_binary.numel())
        prediction_wet_fraction = float(pred_binary.sum().item() / pred_binary.numel())
        rows.append({
            "threshold": threshold,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "true_negatives": true_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "csi": csi,
            "false_alarm_ratio": false_alarm_ratio,
            "miss_rate": miss_rate,
            "target_wet_fraction": target_wet_fraction,
            "prediction_wet_fraction": prediction_wet_fraction,
        })
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_diagnostic_metadata(
    path: Path,
    runs: Sequence[Path],
    labels: Sequence[str],
    thresholds: Sequence[float],
    top_fractions: Sequence[float],
    output_files: Sequence[Path],
) -> None:
    payload = {
        "runs": [str(run) for run in runs],
        "labels": list(labels),
        "thresholds": list(thresholds),
        "top_fractions": list(top_fractions),
        "output_files": [str(path_item) for path_item in output_files],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def plot_peak_depth(output_dir: Path, labels: Sequence[str], series: Sequence[List[Dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, rows in zip(labels, series):
        time_indices = [row["time_index"] for row in rows]
        target_vals = [row["target_max_h"] for row in rows]
        pred_vals = [row["prediction_max_h"] for row in rows]
        ax.plot(time_indices, target_vals, marker="o", linewidth=1.0, label=f"{label} target")
        ax.plot(time_indices, pred_vals, marker="x", linewidth=1.0, label=f"{label} prediction")
    ax.set_xlabel("time index")
    ax.set_ylabel("max H")
    ax.set_title("Peak depth by time")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "peak_depth_by_time.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    if len(args.runs) != len(args.labels):
        raise ValueError("The number of --runs must match the number of --labels")

    run_paths = [Path(run).expanduser().resolve() for run in args.runs]
    peak_series: List[List[Dict[str, Any]]] = []
    full_domain_rows: List[Dict[str, Any]] = []
    wet_cell_rows: List[Dict[str, Any]] = []
    top_depth_rows: List[Dict[str, Any]] = []
    flood_extent_rows: List[Dict[str, Any]] = []

    for run_path, label in zip(run_paths, args.labels):
        saved = load_run_data(run_path)
        prediction = saved["prediction_valid"]
        target = saved["target_valid"]
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch in {run_path}: {tuple(prediction.shape)} vs {tuple(target.shape)}")

        full_domain_metrics = compute_full_domain_metrics(prediction, target)
        full_domain_rows.append({"run": label, **full_domain_metrics})

        wet_cell_metrics = compute_wet_cell_metrics(prediction, target, args.thresholds)
        for row in wet_cell_metrics:
            wet_cell_rows.append({"run": label, **row})

        peak_series.append(
            compute_peak_depth_metrics(prediction, target, saved["valid_rows"], saved["valid_cols"])
        )

        for top_fraction in args.top_fractions:
            top_depth_rows.append({"run": label, **compute_top_depth_metrics(prediction, target, top_fraction)})

        flood_extent_metrics = compute_flood_extent_metrics(prediction, target, args.thresholds)
        for row in flood_extent_metrics:
            flood_extent_rows.append({"run": label, **row})

    full_domain_fieldnames = ["run", "mse", "rmse", "mae", "bias"] + [
        f"{channel.lower()}_{metric}"
        for channel in CHANNEL_NAMES
        for metric in ["mse", "rmse", "mae", "bias"]
    ]
    write_csv(output_dir / "full_domain_metrics.csv", full_domain_rows, full_domain_fieldnames)

    wet_cell_fieldnames = ["run", "threshold", "wet_fraction", "n_values"] + [
        f"{channel.lower()}_{metric}" for channel in CHANNEL_NAMES for metric in ["mse", "rmse", "mae", "bias"]
    ]
    write_csv(output_dir / "wet_cell_metrics.csv", wet_cell_rows, wet_cell_fieldnames)

    peak_depth_fieldnames = [
        "run",
        "time_index",
        "target_max_h",
        "prediction_max_h",
        "signed_peak_error",
        "absolute_peak_error",
        "peak_ratio",
        "target_peak_row",
        "target_peak_col",
        "prediction_peak_row",
        "prediction_peak_col",
        "same_peak_cell",
    ]
    peak_depth_rows: List[Dict[str, Any]] = []
    for label, rows in zip(args.labels, peak_series):
        for row in rows:
            peak_depth_rows.append({"run": label, **row})
    write_csv(output_dir / "peak_depth_by_time.csv", peak_depth_rows, peak_depth_fieldnames)

    top_depth_fieldnames = ["run", "top_fraction", "n_values", "mse", "rmse", "mae", "bias"]
    write_csv(output_dir / "top_depth_metrics.csv", top_depth_rows, top_depth_fieldnames)

    flood_extent_fieldnames = [
        "run",
        "threshold",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "precision",
        "recall",
        "f1",
        "csi",
        "false_alarm_ratio",
        "miss_rate",
        "target_wet_fraction",
        "prediction_wet_fraction",
    ]
    write_csv(output_dir / "flood_extent_metrics.csv", flood_extent_rows, flood_extent_fieldnames)

    plot_peak_depth(output_dir, args.labels, peak_series)

    output_files = [
        output_dir / "full_domain_metrics.csv",
        output_dir / "wet_cell_metrics.csv",
        output_dir / "peak_depth_by_time.csv",
        output_dir / "top_depth_metrics.csv",
        output_dir / "flood_extent_metrics.csv",
        output_dir / "peak_depth_by_time.png",
        output_dir / "diagnostic_metadata.json",
    ]
    save_diagnostic_metadata(
        output_dir / "diagnostic_metadata.json",
        run_paths,
        args.labels,
        args.thresholds,
        args.top_fractions,
        output_files,
    )


if __name__ == "__main__":
    main()
