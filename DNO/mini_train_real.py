#!/usr/bin/env python3
"""Small full-domain DNO mini experiment using real Berlin II PT tensors."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DNO.models.DNO import DNO
from DNO.utils25 import flood_data


EXPECTED_X_SHAPE: Tuple[int, ...] = (1, 433, 692, 24, 1, 5)
EXPECTED_Y_SHAPE: Tuple[int, ...] = (1, 433, 692, 24, 3)
EXPECTED_MASK_SHAPE: Tuple[int, ...] = (1, 433, 692, 24, 3)
CHANNEL_ORDER: List[str] = ["H", "U", "V"]
GENERATED_FILES = {
    "config.json",
    "metrics.csv",
    "best_model.pt",
    "last_checkpoint.pt",
    "summary.json",
    "test_predictions.pt",
}
CHECKPOINT_KEYS = {
    "epoch",
    "train_mse",
    "valid_mse",
    "best_epoch",
    "best_valid_mse",
    "width",
    "learning_rate",
    "seed",
    "activation_offload",
    "model_state_dict",
    "optimizer_state_dict",
    "torch_rng_state",
    "train_generator_state",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini full-domain DNO training on real Berlin II PT tensors")
    parser.add_argument("--data-root", required=True, help="Root directory containing train/valid/test PT folders")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--width", type=int, default=6, help="DNO width")
    parser.add_argument("--epochs", type=int, default=3, help="Total target number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument(
        "--loss-mode",
        choices=["mse", "wet_weighted_mse"],
        default="mse",
        help="Training loss mode",
    )
    parser.add_argument("--wet-threshold", type=float, default=0.05, help="Water-depth threshold for wet weighted loss")
    parser.add_argument("--wet-weight", type=float, default=3.0, help="Loss weight multiplier for wet cells")
    parser.add_argument("--h-channel-weight", type=float, default=1.0, help="Additional loss weight for the H channel")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs")
    parser.add_argument(
        "--activation-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload saved CUDA activations to pinned CPU memory",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume epoch-level training from a checkpoint",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        contents = {item.name for item in output_dir.iterdir()}
        if contents and not overwrite:
            raise FileExistsError(f"Output directory is not empty; use --overwrite for a fresh run: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for file_name in list(GENERATED_FILES):
            path = output_dir / file_name
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    raise IsADirectoryError(f"Unexpected directory in output path: {path}")


def validate_data_root(data_root: Path) -> Dict[str, Path]:
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    split_paths = {
        "train": data_root / "train",
        "valid": data_root / "valid",
        "test": data_root / "test",
    }
    for split_name, split_path in split_paths.items():
        if not split_path.exists() or not split_path.is_dir():
            raise FileNotFoundError(f"Missing split directory '{split_name}': {split_path}")
    return split_paths


def build_datasets(data_root: Path, seed: int, num_workers: int) -> Dict[str, Any]:
    split_paths = validate_data_root(data_root)
    datasets: Dict[str, Any] = {}
    for split_name, split_path in split_paths.items():
        dataset = flood_data(
            path_root=split_path,
            T_in=1,
            T_out=24,
            train=False,
            strategy="oneshot",
        )
        if len(dataset) < 1:
            raise ValueError(f"Split '{split_name}' must contain at least one tensor")
        datasets[split_name] = dataset

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: Dict[str, DataLoader] = {}
    for split_name, dataset in datasets.items():
        shuffle = split_name == "train"
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator if shuffle else None,
        )
        loaders[split_name] = loader

    return {"split_paths": split_paths, "datasets": datasets, "loaders": loaders, "generator": generator}


def validate_batch_shapes(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
    if tuple(x.shape) != EXPECTED_X_SHAPE:
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")
    if tuple(y.shape) != EXPECTED_Y_SHAPE:
        raise ValueError(f"Unexpected y shape: {tuple(y.shape)}")
    if tuple(mask.shape) != EXPECTED_MASK_SHAPE:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")


def validate_finite(x: torch.Tensor, y: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    if not torch.isfinite(y).all():
        raise ValueError("y contains non-finite values")


def validate_spatial_mask(mask: torch.Tensor) -> None:
    mask_tensor = mask[0]
    spatial_mask = mask_tensor[:, :, 0, 0]
    for time_idx in range(mask_tensor.shape[2]):
        for channel_idx in range(mask_tensor.shape[3]):
            if not torch.equal(mask_tensor[:, :, time_idx, channel_idx], spatial_mask):
                raise ValueError(
                    f"Hydraulic spatial mask changed at time {time_idx} channel {channel_idx}"
                )


def compute_metrics_from_reduced(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")

    diff = pred - target
    total_elements = float(diff.numel())
    if total_elements <= 0.0:
        raise ValueError("No valid cells available for metric computation")

    mse = (diff.square().sum() / total_elements).item()
    rmse = (diff.square().sum().sqrt() / total_elements**0.5).item()
    mae = (diff.abs().sum() / total_elements).item()

    channel_metrics: Dict[str, float] = {}
    for channel_idx, channel_name in enumerate(CHANNEL_ORDER):
        channel_diff = diff[..., channel_idx]
        channel_count = float(channel_diff.numel())
        channel_metrics[f"{channel_name.lower()}_mse"] = (channel_diff.square().sum() / channel_count).item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        **channel_metrics,
    }


def weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor, wet_threshold: float, wet_weight: float, h_channel_weight: float) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")

    diff = prediction - target
    squared_error = diff.square()

    loss_weights = torch.ones_like(squared_error)
    wet_mask = target[..., 0] > wet_threshold
    if wet_mask.any():
        loss_weights[wet_mask] = loss_weights[wet_mask] * wet_weight

    if h_channel_weight != 1.0:
        loss_weights[..., 0] = loss_weights[..., 0] * h_channel_weight

    weighted_error = squared_error * loss_weights
    total_weight = loss_weights.sum()
    if total_weight <= 0.0:
        raise ValueError("Weighted loss has no positive weight")
    return weighted_error.sum() / total_weight


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_sse = 0.0
    total_abs = 0.0
    total_count = 0
    channel_sse: Dict[str, float] = {name: 0.0 for name in CHANNEL_ORDER}
    channel_count: Dict[str, int] = {name: 0 for name in CHANNEL_ORDER}

    with torch.inference_mode():
        for batch_idx, (x_cpu, y_cpu, mask_cpu) in enumerate(loader):
            validate_batch_shapes(x_cpu, y_cpu, mask_cpu)
            validate_finite(x_cpu, y_cpu)
            validate_spatial_mask(mask_cpu)

            spatial_mask_cpu = mask_cpu[0, :, :, 0, 0].contiguous()
            reduced_target_cpu = y_cpu[:, spatial_mask_cpu, :, :].contiguous()
            x = x_cpu.to(device)
            spatial_mask = spatial_mask_cpu.to(device)
            reduced_target = reduced_target_cpu.to(device)

            with contextlib.nullcontext():
                pred = model(x)
                if tuple(pred.shape) != EXPECTED_Y_SHAPE:
                    raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")
                reduced_pred = pred[:, spatial_mask, :, :]

            diff = reduced_pred - reduced_target
            total_sse += diff.square().sum().item()
            total_abs += diff.abs().sum().item()
            total_count += diff.numel()

            for channel_idx, channel_name in enumerate(CHANNEL_ORDER):
                channel_diff = diff[..., channel_idx]
                channel_sse[channel_name] += channel_diff.square().sum().item()
                channel_count[channel_name] += channel_diff.numel()

    if total_count <= 0:
        raise ValueError("No valid cells were evaluated")

    metrics = {
        "mse": total_sse / total_count,
        "rmse": (total_sse / total_count) ** 0.5,
        "mae": total_abs / total_count,
    }
    for channel_name in CHANNEL_ORDER:
        metrics[f"{channel_name.lower()}_mse"] = channel_sse[channel_name] / max(channel_count[channel_name], 1)
    return metrics


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_metrics(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "train_mse",
        "valid_mse",
        "valid_rmse",
        "valid_mae",
        "valid_h_mse",
        "valid_u_mse",
        "valid_v_mse",
        "epoch_seconds",
        "peak_gpu_allocated_mib",
        "peak_gpu_reserved_mib",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(path: Path, completed_epoch: int) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume requires metrics.csv: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Resume metrics file has no rows: {path}")

    previous_epoch = 0
    for row_number, row in enumerate(rows, start=2):
        try:
            epoch = int(row["epoch"])
            float(row["epoch_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid metrics.csv row {row_number}: {exc}") from exc
        if epoch <= previous_epoch:
            raise ValueError("metrics.csv epoch numbers must be strictly increasing without duplicates")
        previous_epoch = epoch
    if previous_epoch != completed_epoch:
        raise ValueError(
            f"metrics.csv final epoch {previous_epoch} does not match checkpoint epoch {completed_epoch}"
        )
    return rows


def activation_save_context(device: torch.device, enabled: bool) -> Any:
    if device.type == "cuda" and enabled:
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    return contextlib.nullcontext()


def tensors_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensors_to_cpu(item) for item in value)
    return value


def value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(value_to_device(item, device) for item in value)
    return value


def optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = value_to_device(state, device)


def validate_checkpoint(checkpoint: Dict[str, Any], args: argparse.Namespace, using_cuda: bool) -> None:
    missing = sorted(CHECKPOINT_KEYS.difference(checkpoint))
    if using_cuda and "cuda_rng_state_all" not in checkpoint:
        missing.append("cuda_rng_state_all")
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {', '.join(missing)}")
    if checkpoint["width"] != args.width:
        raise ValueError(f"Checkpoint width {checkpoint['width']} does not match --width {args.width}")
    if not abs(float(checkpoint["learning_rate"]) - args.learning_rate) <= max(1e-12, abs(args.learning_rate) * 1e-9):
        raise ValueError("Checkpoint learning_rate does not match --learning-rate")
    if checkpoint["seed"] != args.seed:
        raise ValueError(f"Checkpoint seed {checkpoint['seed']} does not match --seed {args.seed}")


def make_checkpoint(
    epoch: int,
    train_mse: float,
    valid_mse: float,
    best_epoch: int,
    best_valid_mse: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    args: argparse.Namespace,
    using_cuda: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "train_mse": train_mse,
        "valid_mse": valid_mse,
        "best_epoch": best_epoch,
        "best_valid_mse": best_valid_mse,
        "width": args.width,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "activation_offload": args.activation_offload,
        "model_state_dict": tensors_to_cpu(model.state_dict()),
        "optimizer_state_dict": tensors_to_cpu(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "train_generator_state": train_generator.get_state().clone(),
    }
    if using_cuda:
        payload["cuda_rng_state_all"] = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return payload


def save_test_predictions(
    path: Path,
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial_mask: torch.Tensor,
    source_file: str,
) -> None:
    payload = {
        "prediction_valid": prediction.detach().cpu(),
        "target_valid": target.detach().cpu(),
        "spatial_mask": spatial_mask.detach().cpu(),
        "source_file": source_file,
        "channel_order": CHANNEL_ORDER,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    resume_path = args.resume_from_checkpoint.expanduser().resolve() if args.resume_from_checkpoint else None
    if resume_path is not None:
        if args.overwrite:
            raise ValueError("--resume-from-checkpoint cannot be combined with --overwrite")
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {output_dir}")
        if resume_path.parent.resolve() != output_dir:
            raise ValueError("Resume checkpoint must be directly inside --output-dir")
    else:
        ensure_output_dir(output_dir, args.overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    print(f"device: {device}", flush=True)
    print(f"activation_offload: {args.activation_offload}", flush=True)
    print(f"loss_mode: {args.loss_mode}", flush=True)
    print(f"wet_threshold: {args.wet_threshold}", flush=True)
    print(f"wet_weight: {args.wet_weight}", flush=True)
    print(f"h_channel_weight: {args.h_channel_weight}", flush=True)
    print(f"target_epochs: {args.epochs}", flush=True)

    split_info = build_datasets(data_root, seed=args.seed, num_workers=args.num_workers)
    loaders = split_info["loaders"]
    datasets = split_info["datasets"]
    train_generator: torch.Generator = split_info["generator"]

    model = DNO(
        num_channels=5,
        width=args.width,
        initial_step=1,
        pad=False,
        factor=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(p.numel() for p in model.parameters())

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.csv"
    checkpoint: Optional[Dict[str, Any]] = None
    if resume_path is None:
        serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config = {
            "args": serializable_args,
            "activation_offload": args.activation_offload,
            "loss_mode": args.loss_mode,
            "wet_threshold": args.wet_threshold,
            "wet_weight": args.wet_weight,
            "h_channel_weight": args.h_channel_weight,
            "resolved_paths": {"data_root": str(data_root), "output_dir": str(output_dir)},
            "split_sizes": {name: len(datasets[name]) for name in ["train", "valid", "test"]},
            "model_parameter_count": parameter_count,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "created_utc": timestamp,
            "resume_history": [],
        }
        metrics_rows: List[Dict[str, Any]] = []
        start_epoch = 1
        best_epoch = 0
        best_valid_mse = float("inf")
    else:
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume requires config.json: {config_path}")
        if not (output_dir / "best_model.pt").is_file():
            raise FileNotFoundError(f"Resume requires best_model.pt in output directory: {output_dir}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint payload must be a dictionary: {resume_path}")
        validate_checkpoint(checkpoint, args, device.type == "cuda")
        completed_epoch = int(checkpoint["epoch"])
        if args.epochs <= completed_epoch:
            raise ValueError(
                f"--epochs ({args.epochs}) must exceed completed checkpoint epoch ({completed_epoch})"
            )
        metrics_rows = load_metrics(metrics_path, completed_epoch)
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict) or "created_utc" not in config:
            raise ValueError("Existing config.json must be an object containing created_utc")
        resume_history = config.setdefault("resume_history", [])
        if not isinstance(resume_history, list):
            raise ValueError("Existing config.json resume_history must be a list")
        config["activation_offload"] = args.activation_offload
        start_epoch = completed_epoch + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_valid_mse = float(checkpoint["best_valid_mse"])
        resume_history.append(
            {
                "resumed_utc": timestamp,
                "checkpoint_path": str(resume_path),
                "completed_epoch": completed_epoch,
                "next_epoch": start_epoch,
                "target_epochs": args.epochs,
                "activation_offload": args.activation_offload,
                "device": str(device),
            }
        )

    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    total_training_seconds = sum(float(row["epoch_seconds"]) for row in metrics_rows)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_state_to_device(optimizer, device)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        train_generator.set_state(checkpoint["train_generator_state"])
        best_epoch = int(checkpoint["best_epoch"])
        best_valid_mse = float(checkpoint["best_valid_mse"])
        print(
            f"resume_checkpoint: {resume_path}\n"
            f"completed_epoch: {checkpoint['epoch']}\n"
            f"next_epoch: {start_epoch}\n"
            f"target_epoch: {args.epochs}\n"
            f"previous_best_epoch: {best_epoch}\n"
            f"previous_best_validation_mse: {best_valid_mse:.12g}",
            flush=True,
        )
    save_json(config_path, config)

    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            model.train()
            epoch_start = time.perf_counter()
            train_losses: List[float] = []

            for x_cpu, y_cpu, mask_cpu in loaders["train"]:
                validate_batch_shapes(x_cpu, y_cpu, mask_cpu)
                validate_finite(x_cpu, y_cpu)
                validate_spatial_mask(mask_cpu)

                spatial_mask_cpu = mask_cpu[0, :, :, 0, 0].contiguous()
                reduced_target_cpu = y_cpu[:, spatial_mask_cpu, :, :].contiguous()

                x = x_cpu.to(device)
                spatial_mask = spatial_mask_cpu.to(device)
                reduced_target = reduced_target_cpu.to(device)

                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                with activation_save_context(device, args.activation_offload):
                    pred = model(x)
                    if tuple(pred.shape) != EXPECTED_Y_SHAPE:
                        raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")
                    reduced_pred = pred[:, spatial_mask, :, :]
                    if args.loss_mode == "mse":
                        loss = F.mse_loss(reduced_pred, reduced_target)
                    elif args.loss_mode == "wet_weighted_mse":
                        loss = weighted_mse_loss(
                            reduced_pred,
                            reduced_target,
                            wet_threshold=args.wet_threshold,
                            wet_weight=args.wet_weight,
                            h_channel_weight=args.h_channel_weight,
                        )
                    else:
                        raise ValueError(f"Unsupported loss mode: {args.loss_mode}")

                if not torch.isfinite(loss):
                    raise RuntimeError("Loss is not finite")

                loss.backward()

                grad_params = [parameter for parameter in model.parameters() if parameter.grad is not None]
                if not grad_params:
                    raise RuntimeError("No gradients were produced")
                for parameter in grad_params:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Gradient contains non-finite values")
                if not any(parameter.grad is not None and parameter.grad.detach().abs().sum().item() > 0.0 for parameter in model.parameters()):
                    raise RuntimeError("No non-zero gradients were produced")

                gradient_norm = 0.0
                for parameter in grad_params:
                    grad = parameter.grad.detach()
                    gradient_norm += grad.abs().square().sum().item()
                gradient_norm = float(gradient_norm) ** 0.5

                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)

                train_losses.append(loss.item())
                del x, spatial_mask, reduced_target, pred, reduced_pred, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            train_mse = sum(train_losses) / max(len(train_losses), 1)

            valid_metrics = evaluate_split(model, loaders["valid"], device)
            epoch_seconds = time.perf_counter() - epoch_start
            total_training_seconds += epoch_seconds

            peak_gpu_allocated_mib = 0.0
            peak_gpu_reserved_mib = 0.0
            if device.type == "cuda":
                peak_gpu_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
                peak_gpu_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)

            metrics_rows.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "valid_mse": valid_metrics["mse"],
                    "valid_rmse": valid_metrics["rmse"],
                    "valid_mae": valid_metrics["mae"],
                    "valid_h_mse": valid_metrics["h_mse"],
                    "valid_u_mse": valid_metrics["u_mse"],
                    "valid_v_mse": valid_metrics["v_mse"],
                    "epoch_seconds": epoch_seconds,
                    "peak_gpu_allocated_mib": peak_gpu_allocated_mib,
                    "peak_gpu_reserved_mib": peak_gpu_reserved_mib,
                }
            )
            save_metrics(metrics_path, metrics_rows)

            if valid_metrics["mse"] < best_valid_mse:
                best_valid_mse = valid_metrics["mse"]
                best_epoch = epoch
                improved = True
            else:
                improved = False
            epoch_checkpoint = make_checkpoint(
                epoch, train_mse, valid_metrics["mse"], best_epoch, best_valid_mse,
                model, optimizer, train_generator, args, device.type == "cuda",
            )
            torch.save(epoch_checkpoint, last_checkpoint_path)
            if improved:
                torch.save(epoch_checkpoint, best_checkpoint_path)

            print(
                f"epoch={epoch} train_mse={train_mse:.6f} valid_mse={valid_metrics['mse']:.6f} "
                f"valid_rmse={valid_metrics['rmse']:.6f} valid_mae={valid_metrics['mae']:.6f} "
                f"epoch_seconds={epoch_seconds:.3f}s",
                flush=True,
            )

        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        test_metrics = evaluate_split(model, loaders["test"], device)
        test_batch = next(iter(loaders["test"]))
        test_x_cpu, test_y_cpu, test_mask_cpu = test_batch
        validate_batch_shapes(test_x_cpu, test_y_cpu, test_mask_cpu)
        validate_finite(test_x_cpu, test_y_cpu)
        validate_spatial_mask(test_mask_cpu)
        spatial_mask_cpu = test_mask_cpu[0, :, :, 0, 0].contiguous()
        reduced_target_cpu = test_y_cpu[:, spatial_mask_cpu, :, :].contiguous()
        test_x = test_x_cpu.to(device)
        test_spatial_mask = spatial_mask_cpu.to(device)
        test_target = reduced_target_cpu.to(device)
        with torch.inference_mode():
            test_pred = model(test_x)
            if tuple(test_pred.shape) != EXPECTED_Y_SHAPE:
                raise ValueError(f"Unexpected prediction shape: {tuple(test_pred.shape)}")
            test_reduced_pred = test_pred[:, test_spatial_mask, :, :]
        save_test_predictions(
            output_dir / "test_predictions.pt",
            test_reduced_pred.detach().cpu(),
            test_target.detach().cpu(),
            spatial_mask_cpu.detach().cpu(),
            str(datasets["test"].data[0]),
        )

        summary = {
            "best_epoch": checkpoint["best_epoch"],
            "best_valid_metrics": {
                "mse": checkpoint["valid_mse"],
            },
            "final_test_metrics": test_metrics,
            "total_training_seconds": total_training_seconds,
            "resumed_from_checkpoint": str(resume_path) if resume_path is not None else None,
            "start_epoch": start_epoch,
            "final_epoch": args.epochs,
            "activation_offload": args.activation_offload,
            "loss_mode": args.loss_mode,
            "wet_threshold": args.wet_threshold,
            "wet_weight": args.wet_weight,
            "h_channel_weight": args.h_channel_weight,
            "output_file_paths": {
                "config": str(output_dir / "config.json"),
                "metrics": str(output_dir / "metrics.csv"),
                "best_model": str(output_dir / "best_model.pt"),
                "last_checkpoint": str(last_checkpoint_path),
                "summary": str(output_dir / "summary.json"),
                "test_predictions": str(output_dir / "test_predictions.pt"),
            },
        }
        save_json(output_dir / "summary.json", summary)

        print(
            f"final_test_mse={test_metrics['mse']:.6f} "
            f"final_test_rmse={test_metrics['rmse']:.6f} "
            f"final_test_mae={test_metrics['mae']:.6f}",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0
        print(
            f"CUDA OOM: epoch={current_epoch} peak_allocated_mib={peak_allocated_mib:.3f} "
            f"peak_reserved_mib={peak_reserved_mib:.3f} activation_offload={args.activation_offload}",
            flush=True,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
