"""Evaluate a trained Stockbridge DNO checkpoint and create report outputs."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader

    from DNO.models.DNO import DNO
    from DNO.utils25 import LpLoss, corr, critical_success_index, flood_data, nse
except ImportError as exc:
    torch = None
    DataLoader = DNO = LpLoss = corr = critical_success_index = flood_data = nse = None
    ML_IMPORT_ERROR: ImportError | None = exc
else:
    ML_IMPORT_ERROR = None


T_IN = 1
T_OUT = 24
N_CHANNELS = 3
THRESHOLDS = (0.01, 0.10, 0.50)
FORECAST_MINUTES = np.asarray([30, 60, 90, 120], dtype=np.int64)
FORECAST_INDICES = (5, 11, 17, 23)
METRIC_FIELDS = (
    "relative_lp_loss",
    "nse",
    "correlation",
    "csi_0.01m",
    "csi_0.10m",
    "csi_0.50m",
    "depth_rmse_m",
    "depth_mae_m",
    "max_observed_depth_m",
    "max_predicted_depth_m",
    "inference_time_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DNO checkpoint on 15 Stockbridge test events."
    )
    parser.add_argument("--test-dir", type=Path, required=True, help="Directory containing the test PT files")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint containing epoch and state_dict")
    parser.add_argument("--dem-path", type=Path, required=True, help="Georeferenced DEM raster")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation outputs")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Inference device (default: auto)",
    )
    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def finite_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def load_dem(path: Path) -> dict[str, Any]:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "Rasterio is required to read and validate the supplied DEM"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"DEM does not exist: {path}")
    with rasterio.open(path) as src:
        dem = src.read(1, masked=True)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    if crs is None:
        raise ValueError(f"DEM has no CRS: {path}")
    affine_values = np.asarray(tuple(transform)[:6], dtype=float)
    bound_values = np.asarray(tuple(bounds), dtype=float)
    if not np.isfinite(affine_values).all():
        raise ValueError(f"DEM affine transform contains non-finite values: {transform}")
    if not np.isfinite(bound_values).all():
        raise ValueError(f"DEM bounds contain non-finite values: {bounds}")
    return {
        "array": dem,
        "transform": transform,
        "affine_values": affine_values,
        "bounds": bounds,
        "bound_values": bound_values,
        "crs": crs,
        "extent": (bounds.left, bounds.right, bounds.bottom, bounds.top),
    }


def validate_batch_shapes(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    if x.ndim != 6 or tuple(x.shape[3:]) != (T_OUT, T_IN, 5):
        raise ValueError(
            "Expected input dimensions [batch, height, width, 24, 1, 5], "
            f"found {tuple(x.shape)}"
        )
    if y.ndim != 5 or tuple(y.shape[3:]) != (T_OUT, N_CHANNELS):
        raise ValueError(
            "Expected target dimensions [batch, height, width, 24, 3], "
            f"found {tuple(y.shape)}"
        )
    if mask.shape != y.shape:
        raise ValueError(f"Validity mask shape {tuple(mask.shape)} differs from target {tuple(y.shape)}")
    if x.shape[0] != 1 or y.shape[0] != 1:
        raise ValueError("Evaluation requires batch_size=1")
    return int(y.shape[1]), int(y.shape[2])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_events(
    model: DNO, dataset: flood_data, device: torch.device, dem_shape: tuple[int, int]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = LpLoss(size_average=False)
    records: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}

    with torch.no_grad():
        for event_index, (x, target, mask) in enumerate(loader):
            event_path = Path(dataset.data[event_index])
            event_id = event_path.stem
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            height, width = validate_batch_shapes(x, target, mask)
            if (height, width) != dem_shape:
                raise ValueError(
                    f"DEM shape {dem_shape} does not match tensor spatial shape "
                    f"{(height, width)} for {event_path.name}"
                )

            # Match DNO_main.py: mask target, reshape model output, then mask prediction.
            target = target * mask
            synchronize(device)
            start = time.perf_counter()
            prediction = model(x)
            prediction = prediction.reshape(1, height, width, T_OUT, N_CHANNELS)
            synchronize(device)
            inference_time = time.perf_counter() - start
            prediction = prediction * mask

            pred_flat = prediction.reshape(1, -1, N_CHANNELS)
            target_flat = target.reshape(1, -1, N_CHANNELS)
            depth_valid = mask[0, ..., 0].bool()
            if not bool(depth_valid.any().item()):
                raise ValueError(f"No valid water-depth values in {event_path.name}")
            pred_depth_valid = prediction[0, ..., 0][depth_valid]
            target_depth_valid = target[0, ..., 0][depth_valid]
            difference = pred_depth_valid - target_depth_valid

            record: dict[str, Any] = {
                "event_id": event_id,
                "pt_filename": event_path.name,
                "relative_lp_loss": finite_float(loss_fn(pred_flat, target_flat)),
                "nse": finite_float(nse(pred_flat, target_flat)),
                "correlation": finite_float(corr(pred_flat, target_flat)),
                "csi_0.01m": finite_float(critical_success_index(
                    prediction[..., 0:1].reshape(1, -1, 1), target[..., 0:1].reshape(1, -1, 1), 0.01
                )),
                "csi_0.10m": finite_float(critical_success_index(
                    prediction[..., 0:1].reshape(1, -1, 1), target[..., 0:1].reshape(1, -1, 1), 0.10
                )),
                "csi_0.50m": finite_float(critical_success_index(
                    prediction[..., 0:1].reshape(1, -1, 1), target[..., 0:1].reshape(1, -1, 1), 0.50
                )),
                "depth_rmse_m": finite_float(torch.sqrt(torch.mean(difference.square()))),
                "depth_mae_m": finite_float(torch.mean(torch.abs(difference))),
                "max_observed_depth_m": finite_float(torch.max(target_depth_valid)),
                "max_predicted_depth_m": finite_float(torch.max(pred_depth_valid)),
                "inference_time_s": inference_time,
            }
            records.append(record)
            arrays[event_id] = {
                "target_depth": target[0, ..., 0].cpu().numpy(),
                "predicted_depth": prediction[0, ..., 0].cpu().numpy(),
                "validity_mask": depth_valid.cpu().numpy(),
            }
    return records, arrays


def write_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ("event_id", "pt_filename", *METRIC_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def metric_text(record: dict[str, Any]) -> str:
    values = [f"event_id: {record['event_id']}", f"pt_filename: {record['pt_filename']}"]
    values.extend(f"{name}: {record[name]:.10g}" for name in METRIC_FIELDS)
    return "\n".join(values)


def write_summary(path: Path, epoch: Any, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ranked = sorted(records, key=lambda row: row["relative_lp_loss"])
    selected = {"best": ranked[0], "representative (median)": ranked[len(ranked) // 2], "worst": ranked[-1]}
    lines = [f"Checkpoint epoch: {epoch}", f"Number of test events: {len(records)}"]
    for label, record in selected.items():
        lines.extend(("", f"{label.title()} event", metric_text(record)))
    for statistic, function in (("Mean", np.mean), ("Median", np.median)):
        lines.extend(("", f"{statistic} metrics across all events"))
        for name in METRIC_FIELDS:
            lines.append(f"{name}: {function([row[name] for row in records]):.10g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected


def map_axis(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlabel("British National Grid easting (m)")
    ax.set_ylabel("British National Grid northing (m)")
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def masked_depth_arrays(event: dict[str, np.ndarray], dem: np.ma.MaskedArray) -> tuple[np.ma.MaskedArray, ...]:
    hydraulic_valid = event["validity_mask"].all(axis=-1)
    valid = hydraulic_valid & ~np.ma.getmaskarray(dem)
    target_max = np.max(event["target_depth"], axis=-1)
    predicted_max = np.max(event["predicted_depth"], axis=-1)
    return (
        np.ma.masked_where(~valid, target_max),
        np.ma.masked_where(~valid, predicted_max),
        np.ma.masked_where(~valid, np.abs(predicted_max - target_max)),
        valid,
    )


def plot_max_depth(
    path: Path, event_id: str, target: np.ma.MaskedArray, predicted: np.ma.MaskedArray,
    error: np.ma.MaskedArray, extent: tuple[float, float, float, float]
) -> None:
    shared_max = max(float(target.max()), float(predicted.max()), np.finfo(float).eps)
    error_max = max(float(error.max()), np.finfo(float).eps)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    images = []
    for ax, values, title in zip(
        axes, (target, predicted, error),
        ("CityCAT maximum water depth", "DNO maximum water depth", "Absolute maximum-depth error"),
    ):
        vmax = error_max if values is error else shared_max
        image = ax.imshow(values, origin="upper", extent=extent, vmin=0, vmax=vmax)
        images.append(image)
        ax.set_title(title)
        map_axis(ax, extent)
    fig.colorbar(images[0], ax=axes[:2], label="Water depth (m)", shrink=0.82)
    fig.colorbar(images[2], ax=axes[2], label="Absolute error (m)", shrink=0.82)
    fig.suptitle(f"Maximum depth comparison — event {event_id}")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_extent(
    path: Path, event_id: str, threshold: float, event: dict[str, np.ndarray],
    valid: np.ndarray, extent: tuple[float, float, float, float]
) -> None:
    observed = np.max(event["target_depth"], axis=-1) > threshold
    predicted = np.max(event["predicted_depth"], axis=-1) > threshold
    classes = np.zeros(valid.shape, dtype=np.int8)
    classes[valid & ~observed & ~predicted] = 1
    classes[valid & observed & predicted] = 2
    classes[valid & ~observed & predicted] = 3
    classes[valid & observed & ~predicted] = 4
    shown = np.ma.masked_where(classes == 0, classes)
    colors = ["#eeeeee", "#2ca02c", "#ff7f0e", "#d62728"]
    labels = ["True negative", "True positive", "False positive", "False negative"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(shown, origin="upper", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"Flood extent at {threshold:.2f} m — event {event_id}")
    map_axis(ax, extent)
    ax.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, labels)], loc="best")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_temporal(
    path: Path, event_id: str, event: dict[str, np.ndarray], dem: np.ma.MaskedArray,
    extent: tuple[float, float, float, float]
) -> None:
    spatial_valid = event["validity_mask"].all(axis=-1) & ~np.ma.getmaskarray(dem)
    vmax = max(
        float(np.max(event["target_depth"][spatial_valid])),
        float(np.max(event["predicted_depth"][spatial_valid])),
        np.finfo(float).eps,
    )
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    image = None
    for column, (index, minutes) in enumerate(zip(FORECAST_INDICES, FORECAST_MINUTES)):
        for row, (name, key) in enumerate((("CityCAT", "target_depth"), ("DNO", "predicted_depth"))):
            values = np.ma.masked_where(~spatial_valid, event[key][..., index])
            image = axes[row, column].imshow(values, origin="upper", extent=extent, vmin=0, vmax=vmax)
            axes[row, column].set_title(f"{name}: {minutes} minutes")
            map_axis(axes[row, column], extent)
    fig.colorbar(image, ax=axes, label="Water depth (m)", shrink=0.82)
    fig.suptitle(f"Temporal evolution of water depth — event {event_id}")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if ML_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch and the repository DNO modules are required for evaluation"
        ) from ML_IMPORT_ERROR
    for label, path in (("test directory", args.test_dir), ("checkpoint", args.checkpoint)):
        exists = path.is_dir() if label == "test directory" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{label.title()} does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    dem_info = load_dem(args.dem_path)

    dataset = flood_data(
        path_root=args.test_dir, T_in=T_IN, T_out=T_OUT, train=False, strategy="oneshot"
    )
    if len(dataset.data) != 15:
        raise ValueError(f"Expected exactly 15 test PT events in {args.test_dir}, found {len(dataset.data)}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict) or "epoch" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dictionary containing 'epoch' and 'state_dict'")
    if not isinstance(checkpoint["state_dict"], dict):
        raise TypeError("Checkpoint 'state_dict' must be a state dictionary")
    model = DNO(num_channels=5, width=10, initial_step=1, pad=False, factor=1).to(device)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint state_dict is incompatible with the required DNO configuration: {exc}") from exc
    model.eval()

    records, event_arrays = evaluate_events(model, dataset, device, dem_info["array"].shape)
    write_metrics(args.output_dir / "dno_test_event_metrics.csv", records)
    selected = write_summary(
        args.output_dir / "dno_representative_event_summary.txt", checkpoint["epoch"], records
    )
    representative_id = selected["representative (median)"]["event_id"]
    representative = event_arrays[representative_id]
    target_max, predicted_max, absolute_error, spatial_valid = masked_depth_arrays(
        representative, dem_info["array"]
    )

    plot_max_depth(
        args.output_dir / "dno_max_depth_comparison.png", representative_id,
        target_max, predicted_max, absolute_error, dem_info["extent"],
    )
    for threshold, filename in zip(
        THRESHOLDS,
        ("dno_flood_extent_001m.png", "dno_flood_extent_010m.png", "dno_flood_extent_050m.png"),
    ):
        plot_extent(
            args.output_dir / filename, representative_id, threshold,
            representative, spatial_valid, dem_info["extent"],
        )
    plot_temporal(
        args.output_dir / "dno_temporal_evolution.png", representative_id,
        representative, dem_info["array"], dem_info["extent"],
    )

    np.savez_compressed(
        args.output_dir / "dno_representative_prediction.npz",
        event_id=np.asarray(representative_id),
        target_depth=representative["target_depth"],
        predicted_depth=representative["predicted_depth"],
        validity_mask=representative["validity_mask"],
        target_max_depth=target_max.filled(np.nan),
        predicted_max_depth=predicted_max.filled(np.nan),
        absolute_max_depth_error=absolute_error.filled(np.nan),
        affine_transform=dem_info["affine_values"],
        raster_bounds=dem_info["bound_values"],
        crs=np.asarray(str(dem_info["crs"])),
        forecast_minutes=FORECAST_MINUTES,
    )
    print(f"Evaluated {len(records)} events on {device}; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
