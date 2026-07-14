#!/usr/bin/env python3
"""One-step full-domain DNO training smoke test using a real Berlin II PT tensor."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DNO.models.DNO import DNO
from DNO.utils25 import flood_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-step DNO training smoke test")
    parser.add_argument("--data-path", required=True, help="Path to the PT dataset directory")
    parser.add_argument("--width", type=int, default=6, help="DNO channel width")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--steps", type=int, default=1, help="Number of optimizer steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _validate_shapes(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
    if tuple(x.shape) != (1, 433, 692, 24, 1, 5):
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")
    if tuple(y.shape) != (1, 433, 692, 24, 3):
        raise ValueError(f"Unexpected y shape: {tuple(y.shape)}")
    if tuple(mask.shape) != (1, 433, 692, 24, 3):
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")


def _validate_finite(x: torch.Tensor, y: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    if not torch.isfinite(y).all():
        raise ValueError("y contains non-finite values")


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    if not parameters:
        raise ValueError("Expected at least one parameter")

    total_norm = torch.zeros((), dtype=torch.float32, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_norm = total_norm + grad.abs().square().sum()

    return total_norm.sqrt()


def _parameter_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return any(not torch.equal(b, a) for b, a in zip(before, after))


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    dataset = flood_data(
        path_root=data_path,
        T_in=1,
        T_out=24,
        train=False,
        strategy="oneshot",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    x_cpu, y_cpu, mask_cpu = next(iter(loader))
    _validate_shapes(x_cpu, y_cpu, mask_cpu)
    _validate_finite(x_cpu, y_cpu)

    spatial_mask_cpu = mask_cpu[0, :, :, 0, 0].contiguous()
    valid_spatial_count = int(spatial_mask_cpu.sum().item())
    reduced_target_cpu = y_cpu[:, spatial_mask_cpu, :, :].contiguous()
    y_shape = tuple(y_cpu.shape)
    mask_shape = tuple(mask_cpu.shape)
    del y_cpu
    del mask_cpu

    device = torch.device(args.device)
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA requested but not available")

    x = x_cpu.to(device)
    spatial_mask = spatial_mask_cpu.to(device)
    reduced_target = reduced_target_cpu.to(device)

    model = DNO(
        num_channels=5,
        width=args.width,
        initial_step=1,
        pad=False,
        factor=1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    parameter_count = sum(p.numel() for p in model.parameters())
    reduced_target_size_mib = reduced_target.numel() * reduced_target.element_size() / (1024**2)

    print(f"GPU: {torch.cuda.get_device_name(device.index) if device.type == 'cuda' else device}")
    print(f"width: {args.width}")
    print(f"parameter_count: {parameter_count}")
    print(f"x_shape: {tuple(x.shape)}")
    print(f"y_shape: {y_shape}")
    print(f"mask_shape: {mask_shape}")
    print(f"valid_spatial_cells: {valid_spatial_count}")
    print(f"reduced_target_size_mib: {reduced_target_size_mib:.3f}")

    try:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            before_params = [
                parameter.detach().cpu().clone()
                for parameter in model.parameters()
            ]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                pred = model(x)
                if tuple(pred.shape) != (1, 433, 692, 24, 3):
                    raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")
                reduced_pred = pred[:, spatial_mask, :, :]
                if tuple(reduced_pred.shape) != tuple(reduced_target.shape):
                    raise ValueError(f"Unexpected reduced prediction shape: {tuple(reduced_pred.shape)}")
                loss = F.mse_loss(reduced_pred, reduced_target)
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

            grad_norm = _gradient_norm(grad_params)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Gradient norm is not finite")

            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time

            after_params = [
                parameter.detach().cpu().clone()
                for parameter in model.parameters()
            ]
            changed = _parameter_changed(before_params, after_params)
            if not changed:
                raise RuntimeError("No parameter changed after optimizer step")

            if device.type == "cuda":
                print(
                    f"step={step + 1} loss={loss.item():.6f} "
                    f"grad_params={len(grad_params)} grad_norm={grad_norm.item():.6f} "
                    f"changed={changed} step_time={elapsed:.3f}s "
                    f"gpu_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.3f} "
                    f"gpu_reserved_mb={torch.cuda.max_memory_reserved(device) / 1024**2:.3f}"
                )
            else:
                print(
                    f"step={step + 1} loss={loss.item():.6f} "
                    f"grad_params={len(grad_params)} grad_norm={grad_norm.item():.6f} "
                    f"changed={changed} step_time={elapsed:.3f}s"
                )
    except torch.cuda.OutOfMemoryError as exc:
        print("CUDA OOM failure")
        print(f"error={exc}")
        if device.type == "cuda":
            print(
                f"peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.3f} "
                f"peak_reserved_mb={torch.cuda.max_memory_reserved(device) / 1024**2:.3f}"
            )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
