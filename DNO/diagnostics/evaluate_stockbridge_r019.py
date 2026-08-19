#!/usr/bin/env python3
"""Evaluate the trained Stockbridge DNO checkpoint for event R019 only.

This script:
- loads the existing trained DNO checkpoint;
- uses the same flood_data preprocessing as DNO_main.py;
- performs inference for R019 only;
- extracts water depth H over T1-T24;
- computes temporal maximum depth;
- saves NPZ outputs;
- generates the standard georeferenced comparison figure.

No training, preprocessing, architecture, or metric definitions are changed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DNO_ROOT = REPO_ROOT / "DNO"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(DNO_ROOT) not in sys.path:
    sys.path.insert(0, str(DNO_ROOT))


from models.DNO import DNO
from utils25 import flood_data
from common.stockbridge_plotting import plot_max_depth_comparison


T_IN = 1
T_OUT = 24
NUM_INPUT_CHANNELS = 5
NUM_OUTPUT_CHANNELS = 3

EXPECTED_SPATIAL_SHAPE = (305, 326)


def run_r019_inference(
    event_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """Run DNO inference for R019 using the original dataset preprocessing."""

    dataset = flood_data(
        path_root=event_path.parent,
        train=False,
        strategy="oneshot",
        T_in=T_IN,
        T_out=T_OUT,
    )

    matches = [
        i
        for i, path in enumerate(dataset.data)
        if Path(path).name == event_path.name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one dataset match for {event_path.name}, "
            f"found {len(matches)}"
        )

    event_index = matches[0]

    x, y, mask = dataset[event_index]

    if tuple(x.shape[:2]) != EXPECTED_SPATIAL_SHAPE:
        raise ValueError(
            f"Unexpected input spatial shape: {tuple(x.shape[:2])}"
        )

    if tuple(y.shape[:2]) != EXPECTED_SPATIAL_SHAPE:
        raise ValueError(
            f"Unexpected target spatial shape: {tuple(y.shape[:2])}"
        )

    print("Dataset event index:", event_index)
    print("Input shape:", tuple(x.shape))
    print("Target shape:", tuple(y.shape))
    print("Mask shape:", tuple(mask.shape))

    model = DNO(
        num_channels=NUM_INPUT_CHANNELS,
        width=10,
        initial_step=T_IN,
        pad=False,
        factor=1,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain 'state_dict': {checkpoint_path}"
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x_batch = x.unsqueeze(0).to(device)
    mask_batch = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_batch)

        # Same oneshot handling as DNO_main.py.
        pred = pred.squeeze(-1)

        pred = pred.view(
            1,
            EXPECTED_SPATIAL_SHAPE[0],
            EXPECTED_SPATIAL_SHAPE[1],
            T_OUT,
            NUM_OUTPUT_CHANNELS,
        )

        pred = pred * mask_batch

    pred = pred[0].detach().cpu()
    target = (y * mask).detach().cpu()

    print("Prediction shape:", tuple(pred.shape))
    print("Masked target shape:", tuple(target.shape))

    return pred, target


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--event",
        type=Path,
        default=(
            Path.home()
            / "flood/02_data/CityCAT_Winchester/pt/"
            "stockbridge_phase2_core_v01/test/"
            "R019_stockbridge_T000_T024.pt"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path.home()
            / "flood/03_outputs/UrbanFloodCast/comet/"
            "stockbridge_phase2_dno_main_100epochs_v01/"
            "2026-07-28_15:02:46.876866_stockbridge_all_events_100epochs/"
            "model.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path.home()
            / "flood/03_outputs/UrbanFloodCast/comet/"
            "stockbridge_phase2_dno_main_100epochs_v01/"
            "r019_comparison"
        ),
    )

    parser.add_argument(
        "--depth-vmax",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--error-vmax",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--dry-threshold",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    args.event = args.event.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.event.is_file():
        raise FileNotFoundError(args.event)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Event:", args.event)
    print("Checkpoint:", args.checkpoint)
    print("Output dir:", args.output_dir)

    pred, target = run_r019_inference(
        event_path=args.event,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Channel 0 = water depth H.
    pred_h = pred[..., 0].numpy()
    target_h = target[..., 0].numpy()

    pred_max = np.nanmax(pred_h, axis=2)
    target_max = np.nanmax(target_h, axis=2)

    absolute_error = np.abs(
        pred_max - target_max
    )

    print()
    print("DNO R019")
    print("Prediction maximum:", float(np.nanmax(pred_max)), "m")
    print("Target maximum:", float(np.nanmax(target_max)), "m")

    if args.depth_vmax is None:
        depth_vmax = float(
            np.nanpercentile(
                np.concatenate(
                    [
                        pred_max.reshape(-1),
                        target_max.reshape(-1),
                    ]
                ),
                99.5,
            )
        )
        depth_vmax = max(depth_vmax, 0.1)
    else:
        depth_vmax = args.depth_vmax

    if args.error_vmax is None:
        error_vmax = float(
            np.nanpercentile(
                absolute_error.reshape(-1),
                99.5,
            )
        )
        error_vmax = max(error_vmax, 0.02)
    else:
        error_vmax = args.error_vmax

    print("Depth colour maximum:", depth_vmax, "m")
    print("Error colour maximum:", error_vmax, "m")

    npz_path = (
        args.output_dir
        / "R019_dno_max_depth_maps.npz"
    )

    np.savez_compressed(
        npz_path,
        pred_max_m=pred_max.astype(np.float32),
        target_max_m=target_max.astype(np.float32),
        absolute_error_m=absolute_error.astype(np.float32),
        event_name="R019",
    )

    print("Saved:", npz_path)

    figure_path = (
        args.output_dir
        / "R019_max_depth_comparison.png"
    )

    plot_max_depth_comparison(
        target_max=target_max,
        prediction_max=pred_max,
        output_path=figure_path,
        model_name="DNO",
        event_label="R019",
        depth_vmax=depth_vmax,
        error_vmax=error_vmax,
        dry_threshold=args.dry_threshold,
        save_pdf=True,
    )


if __name__ == "__main__":
    main()
