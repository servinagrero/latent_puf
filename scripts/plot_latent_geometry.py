"""Plot exploratory latent-geometry artifacts as 300-DPI PNG figures."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INSTANCE_COLUMNS = [
    "device_id",
    "slice",
    "stages",
    "placement_variant",
    "placement_gap",
    "placement_offset",
]
COLORS = {
    "raw_hamming": "#4477AA",
    "atf_hamming": "#EE6677",
    "raw_umap": "#228833",
    "atf_umap": "#CCBB44",
    "response_supervised_umap": "#66CCEE",
    "margin_supervised_umap": "#AA3377",
    "modeled_margin_1d": "#000000",
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


def _scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    title: str,
    *,
    categorical: bool = False,
) -> None:
    image = ax.scatter(
        x,
        y,
        c=color,
        cmap="coolwarm",
        s=5,
        alpha=0.65,
        linewidths=0,
        vmin=0 if categorical else None,
        vmax=1 if categorical else None,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def plot_instance_embeddings(
    shared: pl.DataFrame,
    instances: pl.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    for identity, group in instances.group_by(INSTANCE_COLUMNS, maintain_order=True):
        data = group.join(shared, on="challenge_id", how="inner").sort("challenge_id")
        margin = data["modeled_margin"].to_numpy()
        probability = data["response_probability"].to_numpy()
        label = data["majority_response"].to_numpy()
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
        _scatter(
            axes[0, 0],
            data["raw_umap_x"].to_numpy(),
            data["raw_umap_y"].to_numpy(),
            label,
            "Raw bits: unsupervised UMAP, response color",
            categorical=True,
        )
        _scatter(
            axes[0, 1],
            data["raw_umap_x"].to_numpy(),
            data["raw_umap_y"].to_numpy(),
            margin,
            "Raw bits: unsupervised UMAP, margin color",
        )
        _scatter(
            axes[0, 2],
            data["atf_umap_x"].to_numpy(),
            data["atf_umap_y"].to_numpy(),
            margin,
            "Suffix parity: unsupervised UMAP, margin color",
        )
        _scatter(
            axes[1, 0],
            data["response_supervised_x"].to_numpy(),
            data["response_supervised_y"].to_numpy(),
            margin,
            "Raw bits: response-supervised UMAP, margin color",
        )
        _scatter(
            axes[1, 1],
            data["margin_supervised_x"].to_numpy(),
            data["margin_supervised_y"].to_numpy(),
            margin,
            "Raw bits: margin-supervised UMAP, margin color",
        )
        order = np.argsort(margin, kind="stable")
        axes[1, 2].scatter(
            margin[order], probability[order], s=5, alpha=0.22, color="#4477AA"
        )
        bins = np.array_split(order, 50)
        axes[1, 2].plot(
            [float(margin[ids].mean()) for ids in bins],
            [float(probability[ids].mean()) for ids in bins],
            color="black",
            linewidth=2,
            label="100-challenge bin mean",
        )
        axes[1, 2].set(
            xlabel="Modeled margin",
            ylabel="Temporal response probability",
            title="Direct modeled-margin ordering",
        )
        axes[1, 2].grid(alpha=0.25)
        axes[1, 2].legend(fontsize=8)
        device_id, slice_name, *_ = identity
        fig.suptitle(f"{device_id} — {slice_name}", fontsize=13)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{device_id}_{slice_name}")
        save(fig, output / f"embedding_{safe_name}.png", dpi)


def plot_neighborhood_quality(metrics: pl.DataFrame, output: Path, dpi: int) -> None:
    summary = (
        metrics.group_by("representation", "neighbors")
        .agg(
            pl.col("mean_abs_neighbor_margin_difference").median(),
            pl.col("mean_abs_neighbor_probability_difference").median(),
            pl.col("response_label_agreement").median(),
        )
        .sort("representation", "neighbors")
    )
    panels = (
        ("mean_abs_neighbor_margin_difference", "Neighbor margin MAE"),
        ("mean_abs_neighbor_probability_difference", "Neighbor probability MAE"),
        ("response_label_agreement", "Neighbor response agreement"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for representation, color in COLORS.items():
        group = summary.filter(pl.col("representation") == representation)
        if group.is_empty():
            continue
        for ax, (column, ylabel) in zip(axes, panels, strict=True):
            ax.plot(
                group["neighbors"],
                group[column],
                color=color,
                marker="o",
                linewidth=1.6,
                label=representation.replace("_", " "),
            )
            ax.set(xlabel="Nearest neighbors", ylabel=ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    save(fig, output / "geometry_neighborhood_quality.png", dpi)


def plot_hamming_distance(data: pl.DataFrame, output: Path, dpi: int) -> None:
    summary = (
        data.group_by("representation", "distance")
        .agg(
            pl.col("pairs").first(),
            pl.col("mean_abs_margin_difference").median(),
            pl.col("mean_abs_probability_difference").median(),
            pl.col("response_label_agreement").median(),
        )
        .filter(pl.col("pairs") >= 100)
        .sort("representation", "distance")
    )
    panels = (
        ("mean_abs_margin_difference", "Mean absolute margin difference"),
        ("mean_abs_probability_difference", "Mean probability difference"),
        ("response_label_agreement", "Response-label agreement"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for representation in ("raw_hamming", "atf_hamming"):
        group = summary.filter(pl.col("representation") == representation)
        for ax, (column, ylabel) in zip(axes, panels, strict=True):
            ax.plot(
                group["distance"],
                group[column],
                color=COLORS[representation],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=representation.replace("_", " "),
            )
            ax.set(xlabel="Hamming distance", ylabel=ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, output / "hamming_vs_modeled_margin.png", dpi)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shared = pl.read_parquet(args.artifacts / "challenge_embeddings.parquet")
    instances = pl.read_parquet(args.artifacts / "instance_embeddings.parquet")
    metrics = pl.read_parquet(args.artifacts / "neighborhood_metrics.parquet")
    pairs = pl.read_parquet(args.artifacts / "pair_distance_bins.parquet")
    if any(data.is_empty() for data in (shared, instances, metrics, pairs)):
        raise ValueError("Latent-geometry artifacts are empty")
    plot_instance_embeddings(shared, instances, args.output, args.dpi)
    plot_neighborhood_quality(metrics, args.output, args.dpi)
    plot_hamming_distance(pairs, args.output, args.dpi)


if __name__ == "__main__":
    main()
