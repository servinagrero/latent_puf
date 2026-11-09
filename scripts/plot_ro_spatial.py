"""Plot the core temporal--spatial RO PUF results as 300-DPI PNG figures."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SELECTOR_COLORS = {
    "latent_nearest": "#4477AA",
    "latent_nearest_disjoint": "#228833",
    "random": "#CC6677",
}
SELECTOR_LABELS = {
    "latent_nearest": "latent nearest",
    "latent_nearest_disjoint": "latent nearest, target-disjoint",
    "random": "random control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _dataset_axes(datasets: list[str]) -> tuple[plt.Figure, np.ndarray]:
    columns = min(2, len(datasets))
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.1 * columns, 4.0 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax in flat[len(datasets) :]:
        ax.set_visible(False)
    return fig, flat


def _title(dataset: str) -> str:
    return dataset.removeprefix("RO_").replace("_", " ")


def normalized_curves(curves: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Scale each device's latent coordinate while preserving the zero boundary."""

    scales = devices.select(
        "dataset",
        "device_id",
        pl.col("latent_std").alias("device_latent_std"),
    )
    return curves.join(scales, on=["dataset", "device_id"], how="inner").with_columns(
        (pl.col("latent_mean") / pl.col("device_latent_std")).alias("normalized_latent")
    )


def _curve_summary(curves: pl.DataFrame, value: str) -> pl.DataFrame:
    return (
        curves.group_by("dataset", "bin")
        .agg(
            pl.col("normalized_latent").median(),
            pl.col(value).median().alias("median"),
            pl.col(value).quantile(0.25).alias("q25"),
            pl.col(value).quantile(0.75).alias("q75"),
        )
        .sort("dataset", "bin")
    )


def plot_response_sigmoid(curves: pl.DataFrame, output: Path, dpi: int) -> None:
    summary = _curve_summary(curves, "response_probability")
    datasets = summary["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        group = summary.filter(pl.col("dataset") == dataset)
        x = group["normalized_latent"].to_numpy()
        ax.fill_between(x, group["q25"], group["q75"], color="#4477AA", alpha=0.2)
        ax.plot(x, group["median"], color="#4477AA", linewidth=2)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set(
            title=_title(dataset),
            xlabel=r"Normalized frequency difference $\widetilde z$",
            ylabel=r"$P(R=1\mid \widetilde z)$",
            ylim=(-0.03, 1.03),
        )
        ax.grid(alpha=0.25)
    save(fig, output / "ro_response_sigmoid.png", dpi)


def plot_reliability_entropy(curves: pl.DataFrame, output: Path, dpi: int) -> None:
    reliability = _curve_summary(curves, "reference_reliability")
    entropy = _curve_summary(curves, "binary_entropy")
    datasets = reliability["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        rel = reliability.filter(pl.col("dataset") == dataset)
        ent = entropy.filter(pl.col("dataset") == dataset)
        ax.plot(
            rel["normalized_latent"],
            rel["median"],
            color="#4477AA",
            linewidth=2,
            label="reference reliability",
        )
        ax.plot(
            ent["normalized_latent"],
            ent["median"],
            color="#CC6677",
            linewidth=2,
            label="binary entropy",
        )
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set(
            title=_title(dataset),
            xlabel=r"Normalized frequency difference $\widetilde z$",
            ylabel="Probability / bits",
            ylim=(-0.03, 1.03),
        )
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, output / "ro_reliability_entropy.png", dpi)


def _metric_summary(spatial: pl.DataFrame, metric: str) -> pl.DataFrame:
    return (
        spatial.group_by("dataset", "selector", "neighbors")
        .agg(
            pl.col(metric).median().alias("median"),
            pl.col(metric).quantile(0.25).alias("q25"),
            pl.col(metric).quantile(0.75).alias("q75"),
        )
        .sort("dataset", "selector", "neighbors")
    )


def plot_spatial_error(spatial: pl.DataFrame, output: Path, dpi: int) -> None:
    summary = _metric_summary(spatial, "snapshot_total_mae")
    datasets = summary["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        data = summary.filter(pl.col("dataset") == dataset)
        for selector, color in SELECTOR_COLORS.items():
            group = data.filter(pl.col("selector") == selector)
            if group.is_empty():
                continue
            x = group["neighbors"].to_numpy()
            ax.fill_between(x, group["q25"], group["q75"], color=color, alpha=0.15)
            ax.plot(
                x,
                group["median"],
                color=color,
                marker="o",
                linewidth=1.8,
                label=SELECTOR_LABELS[selector],
            )
        ax.set_yscale("log")
        ax.set(
            title=_title(dataset),
            xlabel="Spatial neighbors $k$",
            ylabel="Snapshot total MAE",
        )
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7)
    save(fig, output / "ro_spatial_error.png", dpi)


def plot_error_decomposition(spatial: pl.DataFrame, output: Path, dpi: int) -> None:
    selected = spatial.filter(pl.col("selector") == "latent_nearest")
    metrics = (
        ("spatial_bias_mae", "neighborhood bias", "#4477AA"),
        ("snapshot_precision_mae", "snapshot precision", "#CC6677"),
        ("snapshot_total_mae", "total error", "#228833"),
    )
    datasets = selected["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        for metric, label, color in metrics:
            group = _metric_summary(selected, metric).filter(
                pl.col("dataset") == dataset
            )
            x = group["neighbors"].to_numpy()
            ax.fill_between(x, group["q25"], group["q75"], color=color, alpha=0.13)
            ax.plot(
                x,
                group["median"],
                color=color,
                marker="o",
                linewidth=1.8,
                label=label,
            )
        ax.set_yscale("log")
        ax.set(
            title=_title(dataset),
            xlabel="Spatial neighbors $k$",
            ylabel="Mean absolute error",
        )
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7)
    save(fig, output / "ro_error_decomposition.png", dpi)


def plot_shared_oscillator_effect(
    spatial: pl.DataFrame, output: Path, dpi: int
) -> None:
    paired = (
        spatial.filter(
            pl.col("selector").is_in(["latent_nearest", "latent_nearest_disjoint"])
        )
        .select("dataset", "device_id", "neighbors", "selector", "snapshot_total_mae")
        .pivot(
            on="selector",
            index=["dataset", "device_id", "neighbors"],
            values="snapshot_total_mae",
        )
        .with_columns(
            (pl.col("latent_nearest_disjoint") - pl.col("latent_nearest")).alias(
                "disjoint_minus_unrestricted"
            )
        )
    )
    summary = (
        paired.group_by("dataset", "neighbors")
        .agg(
            pl.col("disjoint_minus_unrestricted").median().alias("median"),
            pl.col("disjoint_minus_unrestricted").quantile(0.25).alias("q25"),
            pl.col("disjoint_minus_unrestricted").quantile(0.75).alias("q75"),
        )
        .sort("dataset", "neighbors")
    )
    datasets = summary["dataset"].unique(maintain_order=True).to_list()
    fig, axes = _dataset_axes(datasets)
    for ax, dataset in zip(axes, datasets, strict=False):
        group = summary.filter(pl.col("dataset") == dataset)
        x = group["neighbors"].to_numpy()
        ax.fill_between(x, group["q25"], group["q75"], color="#AA3377", alpha=0.2)
        ax.plot(x, group["median"], color="#AA3377", marker="o", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=1)
        ax.set(
            title=_title(dataset),
            xlabel="Spatial neighbors $k$",
            ylabel="Disjoint MAE $-$ unrestricted MAE",
        )
        ax.grid(alpha=0.25)
    save(fig, output / "ro_shared_oscillator_effect.png", dpi)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = pl.read_parquet(args.artifacts / "device_metrics.parquet")
    curves = pl.read_parquet(args.artifacts / "latent_curves.parquet")
    spatial = pl.read_parquet(args.artifacts / "spatial_metrics.parquet")
    if any(frame.is_empty() for frame in (devices, curves, spatial)):
        raise ValueError("RO spatial artifacts are empty")
    normalized = normalized_curves(curves, devices)
    plot_response_sigmoid(normalized, args.output, args.dpi)
    plot_reliability_entropy(normalized, args.output, args.dpi)
    plot_spatial_error(spatial, args.output, args.dpi)
    plot_error_decomposition(spatial, args.output, args.dpi)
    plot_shared_oscillator_effect(spatial, args.output, args.dpi)


if __name__ == "__main__":
    main()
