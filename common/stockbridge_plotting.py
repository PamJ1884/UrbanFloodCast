"""Shared Stockbridge plotting utilities.

Provides a common plotting standard for surrogate flood models
(DNO, U-RNN, SWE-GNN) on the Stockbridge domain.

This module handles visualisation only. It does not modify model
outputs, preprocessing, metrics, checkpoints, or predictions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


# ---------------------------------------------------------------------
# Stockbridge spatial reference
# ---------------------------------------------------------------------

CRS = "EPSG:27700"

NROWS = 305
NCOLS = 326
CELL_SIZE_M = 5.0

XMIN = 446340.0
XMAX = 447970.0
YMIN = 129215.0
YMAX = 130740.0

EXTENT = (XMIN, XMAX, YMIN, YMAX)

EXPECTED_SHAPE = (NROWS, NCOLS)

X_LABEL = "British National Grid easting (m)"
Y_LABEL = "British National Grid northing (m)"


def validate_stockbridge_grid(array: np.ndarray, name: str = "array") -> None:
    """Confirm that an array matches the Stockbridge spatial grid."""

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D, got shape {array.shape}"
        )

    if array.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{name} shape {array.shape} does not match "
            f"Stockbridge grid {EXPECTED_SHAPE}"
        )


def _configure_map_axis(axis) -> None:
    """Apply the common Stockbridge map-axis configuration."""

    axis.set_aspect("equal")
    axis.set_xlabel(X_LABEL, fontsize=9)
    axis.set_ylabel(Y_LABEL, fontsize=9)

    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))

    thousands = FuncFormatter(
        lambda value, position: f"{value:,.0f}"
    )

    axis.xaxis.set_major_formatter(thousands)
    axis.yaxis.set_major_formatter(thousands)

    axis.tick_params(
        axis="both",
        labelsize=8,
    )


def plot_max_depth_comparison(
    target_max: np.ndarray,
    prediction_max: np.ndarray,
    output_path: Path,
    *,
    model_name: str,
    event_label: str,
    depth_vmax: float,
    error_vmax: float,
    dry_threshold: float = 0.01,
    save_pdf: bool = True,
) -> None:
    """Plot CityCAT, surrogate prediction, and absolute error.

    Parameters
    ----------
    target_max
        CityCAT maximum water depth raster [m].
    prediction_max
        Model maximum water depth raster [m].
    output_path
        PNG output path.
    model_name
        Display name, e.g. ``"U-RNN"`` or ``"DNO"``.
    event_label
        Event identifier, e.g. ``"R019"``.
    depth_vmax
        Shared upper colour limit for CityCAT and prediction.
    error_vmax
        Upper colour limit for absolute error.
    dry_threshold
        Cells below this depth are visually masked.
    save_pdf
        Also save a PDF next to the PNG.
    """

    target_max = np.asarray(target_max)
    prediction_max = np.asarray(prediction_max)

    validate_stockbridge_grid(target_max, "target_max")
    validate_stockbridge_grid(prediction_max, "prediction_max")

    if not np.isfinite(depth_vmax) or depth_vmax <= 0:
        raise ValueError("depth_vmax must be finite and > 0")

    if not np.isfinite(error_vmax) or error_vmax <= 0:
        raise ValueError("error_vmax must be finite and > 0")

    if dry_threshold < 0:
        raise ValueError("dry_threshold must be >= 0")

    absolute_error = np.abs(prediction_max - target_max)

    target_masked = np.where(
        target_max >= dry_threshold,
        target_max,
        np.nan,
    )

    prediction_masked = np.where(
        prediction_max >= dry_threshold,
        prediction_max,
        np.nan,
    )

    error_masked = np.where(
        (prediction_max >= dry_threshold)
        | (target_max >= dry_threshold),
        absolute_error,
        np.nan,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 5.0),
        constrained_layout=True,
    )

    target_image = axes[0].imshow(
        target_masked,
        origin="upper",
        extent=EXTENT,
        aspect="equal",
        vmin=0.0,
        vmax=depth_vmax,
    )

    axes[0].set_title(
        f"CityCAT maximum water depth\n{event_label}"
    )

    prediction_image = axes[1].imshow(
        prediction_masked,
        origin="upper",
        extent=EXTENT,
        aspect="equal",
        vmin=0.0,
        vmax=depth_vmax,
    )

    axes[1].set_title(
        f"{model_name} maximum water depth"
    )

    error_image = axes[2].imshow(
        error_masked,
        origin="upper",
        extent=EXTENT,
        aspect="equal",
        vmin=0.0,
        vmax=error_vmax,
    )

    axes[2].set_title(
        f"|{model_name} - CityCAT|"
    )

    for axis in axes:
        _configure_map_axis(axis)

    depth_bar = figure.colorbar(
        target_image,
        ax=[axes[0], axes[1]],
        fraction=0.046,
        pad=0.04,
    )
    depth_bar.set_label("Maximum water depth [m]")

    error_bar = figure.colorbar(
        error_image,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )
    error_bar.set_label("Absolute error [m]")

    figure.suptitle(
        "Stockbridge test event: maximum water depth over T1–T24",
        fontsize=13,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print("Saved:", output_path)

    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )
        print("Saved:", pdf_path)

    plt.close(figure)
