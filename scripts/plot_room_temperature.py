"""Create PNG figures from room-temperature Parquet artifacts."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

COLORS = {
    "gray_prefix": "#4477AA",
    "margin_nearest": "#228833",
    "balanced_margin": "#AA3377",
    "random": "#CC6677",
}

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)


def plot_spatial_vs_temporal(data: pl.DataFrame, output: Path, dpi: int) -> None:
    max_k = data["k"].cast(pl.Int64).max()
    if not isinstance(max_k, int):
        raise TypeError("selector_metrics.parquet contains no integer k values")
    selected = data.filter(pl.col("k") == max_k)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for ax, selector in zip(axes, COLORS, strict=True):
        group = selected.filter(pl.col("selector") == selector)
        ax.scatter(group["p_temporal"], group["p_spatial"], s=8, alpha=0.35)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
        ax.set_title(selector.replace("_", " "))
        ax.set_xlabel("Temporal response probability")
    axes[0].set_ylabel("Spatial response probability")
    fig.suptitle(f"Spatial versus temporal estimates, k={max_k}")
    save(fig, output / "spatial_vs_temporal.png", dpi)


def plot_temporal_vs_spatial_curve(
    temporal: pl.DataFrame, spatial: pl.DataFrame, output: Path, dpi: int
) -> None:
    instances = (
        temporal.select("device_id", "slice").unique().sort("device_id", "slice")
    )
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (device, slice_name) in zip(axes.flat, instances.iter_rows()):
        ax.set_visible(True)
        temporal_group = temporal.filter(
            (pl.col("device_id") == device) & (pl.col("slice") == slice_name)
        ).sort("mean_margin")
        spatial_group = spatial.filter(
            (pl.col("device_id") == device) & (pl.col("slice") == slice_name)
        ).sort("mean_margin")
        ax.plot(
            temporal_group["mean_margin"],
            temporal_group["mean_response"],
            color="#4477AA",
            label="temporal",
        )
        ax.plot(
            spatial_group["mean_margin"],
            spatial_group["mean_response"],
            color="#EE6677",
            linestyle="--",
            label="spatial",
        )
        ax.set_title(f"{device.removeprefix('CMOD_S7-')} {slice_name}", fontsize=8)
        ax.grid(alpha=0.15)
    axes[0, 0].legend()
    fig.supxlabel("Logistic-regression margin")
    fig.supylabel("Mean response")
    fig.suptitle("Temporal repeats versus single-shot spatial ensemble")
    save(fig, output / "temporal_vs_spatial_curve.png", dpi)


def plot_error_curves(data: pl.DataFrame, output: Path, dpi: int) -> None:
    summary = (
        data.group_by("selector", "k", "n_spatial")
        .agg(
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("absolute_error").std().alias("sd"),
            pl.len().alias("n"),
        )
        .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("se"))
        .sort("selector", "k")
    )
    for x_name, filename, label in (
        ("k", "error_vs_k.png", "Dither bits k"),
        ("n_spatial", "error_vs_reads.png", "Spatial reads"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for selector, color in COLORS.items():
            group = summary.filter(pl.col("selector") == selector)
            ax.errorbar(
                group[x_name].to_numpy(),
                group["mae"].to_numpy(),
                yerr=1.96 * group["se"].to_numpy(),
                marker="o",
                markersize=4,
                color=color,
                label=selector.replace("_", " "),
            )
        if x_name == "n_spatial":
            ax.set_xscale("log", base=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean absolute probability error")
        ax.legend()
        ax.grid(alpha=0.2)
        save(fig, output / filename, dpi)


def plot_margin_mismatch(data: pl.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for selector, color in COLORS.items():
        group = data.filter(pl.col("selector") == selector)
        ax.scatter(
            group["margin_mae"],
            group["absolute_error"],
            s=5,
            alpha=0.12,
            color=color,
            label=selector.replace("_", " "),
        )
    ax.set_xlabel("Mean absolute modeled-margin mismatch")
    ax.set_ylabel("Absolute probability error")
    ax.legend(markerscale=3)
    save(fig, output / "error_vs_margin_mismatch.png", dpi)


def plot_reliability_cdf(
    data: pl.DataFrame, output: Path, dpi: int, k: int = 10
) -> None:
    temporal = data.select("device_id", "slice", "baseline_id", "p_temporal").unique()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    def draw(
        probabilities: np.ndarray, label: str, color: str, linestyle: str = "-"
    ) -> None:
        ber = np.sort(np.minimum(probabilities, 1 - probabilities))
        cdf = np.arange(1, len(ber) + 1) / len(ber)
        ax.step(ber, cdf, where="post", label=label, color=color, linestyle=linestyle)

    draw(temporal["p_temporal"].to_numpy(), "temporal", "black", "--")
    for selector, color in COLORS.items():
        group = data.filter((pl.col("selector") == selector) & (pl.col("k") == k))
        draw(group["p_spatial"].to_numpy(), selector.replace("_", " "), color)
    ax.set_xlabel("Majority-reference BER")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Temporal and spatial reliability distributions, k={k}")
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, output / "reliability_cdf.png", dpi)


def plot_reliability_vs_k(data: pl.DataFrame, output: Path, dpi: int) -> None:
    temporal = data.select("device_id", "slice", "baseline_id", "p_temporal").unique()
    temporal_p = temporal["p_temporal"].to_numpy()
    temporal_ber = float(np.minimum(temporal_p, 1 - temporal_p).mean())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for selector, color in COLORS.items():
        group = (
            data.filter(pl.col("selector") == selector)
            .with_columns(
                pl.min_horizontal("p_spatial", 1 - pl.col("p_spatial")).alias("ber")
            )
            .group_by("k")
            .agg(pl.col("ber").mean())
            .sort("k")
        )
        ax.plot(group["k"], group["ber"], marker="o", color=color, label=selector)
    ax.axhline(temporal_ber, color="black", linestyle="--", label="temporal")
    ax.set_xlabel("Dither bits k")
    ax.set_ylabel("Mean majority-reference BER")
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, output / "reliability_vs_k.png", dpi)


def plot_signed_error_vs_margin_bias(
    data: pl.DataFrame, output: Path, dpi: int, k: int = 10
) -> None:
    selected = data.filter(pl.col("k") == k)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, (selector, color) in zip(axes.flat, COLORS.items(), strict=True):
        group = selected.filter(pl.col("selector") == selector)
        ax.scatter(
            group["margin_bias"], group["signed_error"], s=7, alpha=0.25, color=color
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(selector.replace("_", " "))
    fig.supxlabel("Signed mean margin offset")
    fig.supylabel("Spatial minus temporal probability")
    fig.suptitle(f"Neighborhood centering and estimation bias, k={k}")
    save(fig, output / "signed_error_vs_margin_bias.png", dpi)


def plot_base_margin_vs_error(
    data: pl.DataFrame, output: Path, dpi: int, k: int = 10
) -> None:
    selected = data.filter(pl.col("k") == k)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for selector, color in COLORS.items():
        group = selected.filter(pl.col("selector") == selector)
        ax.scatter(
            group["base_margin"],
            group["absolute_error"],
            s=6,
            alpha=0.2,
            color=color,
            label=selector.replace("_", " "),
        )
    ax.set_xlabel("Base-challenge margin")
    ax.set_ylabel("Absolute probability error")
    ax.legend(markerscale=3)
    save(fig, output / "base_margin_vs_error.png", dpi)


def plot_balanced_selector_comparison(
    data: pl.DataFrame, output: Path, dpi: int
) -> None:
    selected = data.filter(
        pl.col("selector").is_in(["margin_nearest", "balanced_margin"])
    )
    summary = (
        selected.group_by("selector", "k")
        .agg(
            pl.col("absolute_error").mean().alias("probability_mae"),
            pl.col("margin_bias").abs().mean().alias("absolute_margin_bias"),
        )
        .sort("selector", "k")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for selector in ("margin_nearest", "balanced_margin"):
        group = summary.filter(pl.col("selector") == selector)
        color = COLORS[selector]
        axes[0].plot(
            group["k"],
            group["probability_mae"],
            marker="o",
            color=color,
            label=selector,
        )
        axes[1].plot(
            group["k"],
            group["absolute_margin_bias"],
            marker="o",
            color=color,
            label=selector,
        )
    axes[0].set_ylabel("Probability MAE")
    axes[1].set_ylabel("Mean absolute margin bias")
    for ax in axes:
        ax.set_xlabel("Dither bits k")
        ax.grid(alpha=0.2)
        ax.legend()
    save(fig, output / "balanced_selector_comparison.png", dpi)


def plot_margin_calibration(data: pl.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (device, slice_name), group in data.group_by("device_id", "slice"):
        group = group.sort("mean_margin")
        ax.plot(
            group["mean_margin"],
            group["mean_response"],
            linewidth=1,
            alpha=0.65,
            label=f"{device} {slice_name}",
        )
    x = np.linspace(-8, 8, 300)
    ax.plot(x, 1 / (1 + np.exp(-x)), color="black", linestyle="--", label="logistic")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Logistic-regression margin")
    ax.set_ylabel("Mean response")
    ax.legend(fontsize=6, ncol=2)
    save(fig, output / "margin_calibration.png", dpi)


def plot_gray_diagnostics(data: pl.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data["gray_margin_spearman"], bins=30, color="#4477AA")
    axes[0].set_xlabel("Spearman correlation: Gray index vs margin")
    axes[0].set_ylabel("Clusters")
    axes[1].scatter(
        data["gray_margin_step_mae"], data["absolute_error_full"], s=8, alpha=0.35
    )
    axes[1].set_xlabel("Mean absolute Gray-step margin change")
    axes[1].set_ylabel("Full-neighborhood absolute error")
    save(fig, output / "gray_order_diagnostics.png", dpi)


def plot_model_validation(data: pl.DataFrame, output: Path, dpi: int) -> None:
    labels = [
        f"{row[0]}\n{row[1]}" for row in data.select("device_id", "slice").iter_rows()
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.65), 4.5))
    ax.bar(x, data["test_accuracy"], color="#4477AA", label="accuracy")
    ax.scatter(
        x, data["test_balanced_accuracy"], color="#EE6677", label="balanced accuracy"
    )
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Held-out score")
    ax.legend()
    save(fig, output / "model_validation.png", dpi)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Reading artifacts from %s", args.artifacts)
    selectors = pl.read_parquet(args.artifacts / "selector_metrics.parquet")
    clusters = pl.read_parquet(args.artifacts / "cluster_metrics.parquet")
    margin_bins = pl.read_parquet(args.artifacts / "margin_bins.parquet")
    temporal_margin_bins = pl.read_parquet(
        args.artifacts / "temporal_margin_bins.parquet"
    )
    models = pl.read_parquet(args.artifacts / "models.parquet")
    plot_spatial_vs_temporal(selectors, args.output, args.dpi)
    plot_temporal_vs_spatial_curve(
        temporal_margin_bins, margin_bins, args.output, args.dpi
    )
    plot_error_curves(selectors, args.output, args.dpi)
    plot_margin_mismatch(selectors, args.output, args.dpi)
    plot_reliability_cdf(selectors, args.output, args.dpi)
    plot_reliability_vs_k(selectors, args.output, args.dpi)
    plot_signed_error_vs_margin_bias(selectors, args.output, args.dpi)
    plot_base_margin_vs_error(selectors, args.output, args.dpi)
    plot_balanced_selector_comparison(selectors, args.output, args.dpi)
    plot_margin_calibration(margin_bins, args.output, args.dpi)
    plot_gray_diagnostics(clusters, args.output, args.dpi)
    plot_model_validation(models, args.output, args.dpi)
    LOGGER.info("Plotting complete")


if __name__ == "__main__":
    main()
