import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


run = Path(os.environ["RUN"])
output_dir = run / "diagnostics" / "comparison_panels"
output_dir.mkdir(parents=True, exist_ok=True)

saved = torch.load(
    run / "test_predictions.pt",
    map_location="cpu",
)

prediction = saved["prediction_valid"].float()[0]
target = saved["target_valid"].float()[0]
spatial_mask = saved["spatial_mask"].bool()

channels = ["H", "U", "V"]
selected_minutes = [45, 60, 120]


def make_grid(values: torch.Tensor) -> np.ndarray:
    grid = torch.full(
        spatial_mask.shape,
        float("nan"),
        dtype=torch.float32,
    )
    grid[spatial_mask] = values
    return grid.numpy()


for channel_index, channel in enumerate(channels):
    figure, axes = plt.subplots(
        nrows=len(selected_minutes),
        ncols=3,
        figsize=(15, 15),
        constrained_layout=True,
    )

    for row, minutes in enumerate(selected_minutes):
        time_index = minutes // 5 - 1

        truth = make_grid(
            target[:, time_index, channel_index]
        )

        pred = make_grid(
            prediction[:, time_index, channel_index]
        )

        error = pred - truth

        truth_values = truth[np.isfinite(truth)]
        pred_values = pred[np.isfinite(pred)]

        combined_values = np.concatenate(
            [truth_values, pred_values]
        )

        if channel == "H":
            value_min = 0.0
            value_max = max(
                float(combined_values.max()),
                1e-12,
            )
        else:
            value_limit = max(
                abs(float(combined_values.min())),
                abs(float(combined_values.max())),
                1e-12,
            )
            value_min = -value_limit
            value_max = value_limit

        error_values = error[np.isfinite(error)]

        error_limit = max(
            float(np.abs(error_values).max()),
            1e-12,
        )

        target_image = axes[row, 0].imshow(
            truth,
            vmin=value_min,
            vmax=value_max,
        )

        axes[row, 1].imshow(
            pred,
            vmin=value_min,
            vmax=value_max,
        )

        error_image = axes[row, 2].imshow(
            error,
            vmin=-error_limit,
            vmax=error_limit,
        )

        axes[row, 0].set_title(
            f"Target — {minutes} min"
        )

        axes[row, 1].set_title(
            f"Prediction — {minutes} min"
        )

        axes[row, 2].set_title(
            f"Error — {minutes} min"
        )

        figure.colorbar(
            target_image,
            ax=[axes[row, 0], axes[row, 1]],
            shrink=0.75,
            label=channel,
        )

        figure.colorbar(
            error_image,
            ax=axes[row, 2],
            shrink=0.75,
            label=f"{channel} prediction − target",
        )

        for axis in axes[row]:
            axis.set_xlabel("Grid column")
            axis.set_ylabel("Grid row")

    figure.suptitle(
        f"{channel}: target, prediction and error",
        fontsize=16,
    )

    output_path = (
        output_dir
        / f"{channel}_comparison_045_060_120min.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)

print()
print("Comparison panels saved in:", output_dir)
