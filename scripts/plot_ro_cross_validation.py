"""Plot fixed-bank versus LOO RO cross-validation results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from summarize_ro_cross_validation import (
    device_curve_summary,
    device_metric_summary,
    load_metrics,
    paired_mode_summary,
    population_curve_summary,
    population_metric_summary,
)

MODE_STYLE = {"fixed": ("-", "fixed bank"), "loo": ("--", "leave one out")}
METRIC_COLORS = {
    "curve_mae": "#4477AA",
    "x50_error": "#CC6677",
    "width_error": "#228833",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--loo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _title(dataset: str) -> str:
    return dataset.removeprefix("RO_").replace("_", " ")


def _dataset_axes(datasets: list[str]) -> tuple[plt.Figure, np.ndarray]:
    columns = min(2, len(datasets))
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 4.1 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax in flat[len(datasets) :]:
        ax.set_visible(False)
    return fig, flat


def _plot_mode_metric(
    summary: pl.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
    output: Path,
    dpi: int,
    *,
    zero_line: bool = False,
) -> None:
    data = summary.filter(
        (pl.col("selector") == "latent_nearest") & (pl.col("stratum_type") == "all")
    )
    datasets = data["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        for mode, (linestyle, label) in MODE_STYLE.items():
            group = data.filter(
                (pl.col("dataset") == dataset) & (pl.col("mode") == mode)
            ).sort("neighbors")
            if group.is_empty():
                continue
            x = group["neighbors"].to_numpy()
            median = group[f"{metric}_median"].to_numpy()
            ax.fill_between(
                x,
                group[f"{metric}_q25"].to_numpy(),
                group[f"{metric}_q75"].to_numpy(),
                alpha=0.13,
            )
            ax.plot(
                x, median, linestyle=linestyle, marker="o", linewidth=1.8, label=label
            )
        if zero_line:
            ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xscale("log", base=2)
        ax.set(title=_title(dataset), xlabel="Spatial neighbors $k$", ylabel=ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, output / filename, dpi)


def plot_boundary_brier(summary: pl.DataFrame, output: Path, dpi: int) -> None:
    labels = ("(0.1,0.25]", "(0.25,0.5]")
    data = summary.filter(
        (pl.col("selector") == "latent_nearest")
        & (pl.col("stratum_type") == "ber")
        & pl.col("stratum").is_in(labels)
    )
    datasets = data["dataset"].unique(maintain_order=True).to_list()
    fig, axes = plt.subplots(
        2, len(datasets), figsize=(4.1 * len(datasets), 7.2), squeeze=False
    )
    for row, stratum in enumerate(labels):
        for column, dataset in enumerate(datasets):
            ax = axes[row, column]
            for mode, (linestyle, label) in MODE_STYLE.items():
                group = data.filter(
                    (pl.col("dataset") == dataset)
                    & (pl.col("stratum") == stratum)
                    & (pl.col("mode") == mode)
                ).sort("neighbors")
                ax.plot(
                    group["neighbors"],
                    group["excess_brier_median"],
                    linestyle=linestyle,
                    marker="o",
                    linewidth=1.6,
                    label=label,
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_title(_title(dataset), fontsize=10)
            ax.set_xlabel("$k$")
            if column == 0:
                ax.set_ylabel(f"BER {stratum}\nExcess Brier")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    save(fig, output / "boundary_brier_vs_k.png", dpi)


def plot_curve_features(summary: pl.DataFrame, output: Path, dpi: int) -> None:
    data = summary.filter(pl.col("selector") == "latent_nearest")
    datasets = data["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        for metric, color in METRIC_COLORS.items():
            for mode, (linestyle, label) in MODE_STYLE.items():
                group = data.filter(
                    (pl.col("dataset") == dataset) & (pl.col("mode") == mode)
                ).sort("neighbors")
                ax.plot(
                    group["neighbors"],
                    group[f"{metric}_median"],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    linewidth=1.5,
                    label=f"{metric.replace('_', ' ')} — {label}",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set(
            title=_title(dataset),
            xlabel="Spatial neighbors $k$",
            ylabel="Absolute error",
        )
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=6)
    save(fig, output / "curve_features_vs_k.png", dpi)


def plot_paired(paired: pl.DataFrame, output: Path, dpi: int) -> None:
    data = paired.filter(pl.col("neighbors") == 512)
    datasets = data["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        group = data.filter(pl.col("dataset") == dataset)
        x = group["loo_reconstruction_mae"].to_numpy()
        y = group["fixed_reconstruction_mae"].to_numpy()
        bounds = (min(x.min(), y.min()), max(x.max(), y.max()))
        ax.scatter(x, y, s=18, alpha=0.65, color="#4477AA")
        ax.plot(bounds, bounds, color="black", linestyle=":", linewidth=1)
        ax.set(
            title=_title(dataset),
            xlabel="LOO reconstruction MAE",
            ylabel="Fixed-bank reconstruction MAE",
        )
        ax.grid(alpha=0.25)
    save(fig, output / "fixed_vs_loo_paired.png", dpi)


def plot_error_vs_radius(summary: pl.DataFrame, output: Path, dpi: int) -> None:
    data = summary.filter(
        (pl.col("selector") == "latent_nearest") & (pl.col("stratum_type") == "all")
    )
    datasets = data["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        for mode, (linestyle, label) in MODE_STYLE.items():
            group = data.filter(
                (pl.col("dataset") == dataset) & (pl.col("mode") == mode)
            ).sort("normalized_radius_median")
            ax.plot(
                group["normalized_radius_median"],
                group["reconstruction_mae_median"],
                linestyle=linestyle,
                marker="o",
                linewidth=1.8,
                label=label,
            )
        ax.set_xscale("log")
        ax.set(
            title=_title(dataset),
            xlabel=r"Normalized latent radius $\widetilde\rho_k$",
            ylabel="Reconstruction MAE",
        )
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, output / "error_vs_latent_radius.png", dpi)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.fixed, args.loo, "cv_metrics.parquet")
    curves = load_metrics(args.fixed, args.loo, "cv_curve_metrics.parquet")
    device_metrics = device_metric_summary(metrics)
    metric_summary = population_metric_summary(device_metrics)
    curve_summary = population_curve_summary(device_curve_summary(curves))
    paired = paired_mode_summary(device_metrics)
    _plot_mode_metric(
        metric_summary,
        "reconstruction_mae",
        "Reconstruction MAE",
        "reconstruction_vs_k.png",
        args.output,
        args.dpi,
    )
    _plot_mode_metric(
        metric_summary,
        "excess_brier",
        "Excess Brier score",
        "excess_brier_vs_k.png",
        args.output,
        args.dpi,
        zero_line=True,
    )
    plot_boundary_brier(metric_summary, args.output, args.dpi)
    plot_curve_features(curve_summary, args.output, args.dpi)
    plot_paired(paired, args.output, args.dpi)
    plot_error_vs_radius(metric_summary, args.output, args.dpi)


if __name__ == "__main__":
    main()
