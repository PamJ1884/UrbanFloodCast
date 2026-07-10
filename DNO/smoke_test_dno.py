"""Synthetic forward-pass smoke test for the UrbanFloodCast DNO model."""

from __future__ import annotations

import argparse
import time

import torch

from models.DNO import DNO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic DNO forward-pass smoke test."
    )
    parser.add_argument("--size-x", type=int, default=64)
    parser.add_argument("--size-y", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda:0")

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    torch.manual_seed(0)

    model = DNO(
        num_channels=args.channels,
        width=args.width,
        initial_step=1,
        pad=args.pad,
        factor=1,
    ).to(device)

    model.eval()

    input_tensor = torch.randn(
        1,
        args.size_x,
        args.size_y,
        args.timesteps,
        args.channels,
        device=device,
    )

    expected_shape = (
        1,
        args.size_x,
        args.size_y,
        args.timesteps,
        3,
    )

    print("===== DNO SMOKE TEST =====")
    print("PyTorch:", torch.__version__)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("Input shape:", tuple(input_tensor.shape))
    print("Expected output:", expected_shape)
    print("Initial padding:", model.padding)
    print(
        "Parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )

    start = time.perf_counter()

    with torch.inference_mode():
        for run_number in range(1, args.runs + 1):
            output = model(input_tensor)

            output_shape = tuple(output.shape)
            finite = bool(torch.isfinite(output).all().item())

            print(f"Run {run_number} output:", output_shape)
            print(f"Run {run_number} finite:", finite)

            assert output_shape == expected_shape
            assert finite

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    assert model.padding == args.pad, (
        f"model.padding changed from {args.pad} to {model.padding}"
    )

    print("Final padding:", model.padding)
    print("Elapsed time:", round(elapsed, 3), "seconds")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print("Peak GPU memory:", round(peak_memory, 2), "MiB")

    print("DNO smoke test completed successfully.")


if __name__ == "__main__":
    main()
    