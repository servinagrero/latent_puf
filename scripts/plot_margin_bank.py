"""Plot fixed modeled-margin bank tradeoffs as 300-DPI PNG figures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

INSTANCE_COLUMNS = [
    "device_id",
    "slice",
    "stages",
    "placement_variant",
    "placement_gap",
    "placement_offset",
]


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


def plot_curve_examples(
    curves: pl.DataFrame, reference: pl.DataFrame, output: Path, dpi: int
) -> None:
    first = curves.select(INSTANCE_COLUMNS).unique(maintain_order=True).row(0)
    selected = curves
    for column, value in zip(INSTANCE_COLUMNS, first, strict=True):
        selected = selected.filter(pl.col(column) == value)
    counts = selected["challenges_per_bin"].unique().sort().to_list()
    count = counts[len(counts) // 2]
    selected = selected.filter(pl.col("challenges_per_bin") == count)
    bank = (
        selected.select("bin_id", "bin_center", "p_temporal_bank")
        .unique()
        .sort("bin_center")
    )
    full = reference
    for column, value in zip(INSTANCE_COLUMNS, first, strict=True):
        full = full.filter(pl.col(column) == value)
    full = full.sort("bin_center")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        full["bin_center"],
        full["p_temporal_full"],
        color="black",
        linewidth=2.2,
        label="full temporal reference",
    )
    ax.plot(
        bank["bin_center"],
        bank["p_temporal_bank"],
        color="#555555",
        linestyle="--",
        linewidth=1.8,
        label="enrolled-bank temporal reference",
    )
    for sample_id in (0, 1, 2):
        snapshot = selected.filter(pl.col("sample_id") == sample_id).sort("bin_center")
        ax.plot(
            snapshot["bin_center"],
            snapshot["p_single_shot"],
            marker="o",
            markersize=2.5,
            linewidth=0.9,
            alpha=0.75,
            label=f"single-shot snapshot {sample_id}",
        )
    ax.set(xlabel="Room-temperature modeled margin", ylabel="Response probability")
    ax.set_title(f"Fixed margin bank: {count} challenges per bin")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, output / "margin_bank_curve_examples.png", dpi)


def _instance_summary(features: pl.DataFrame) -> pl.DataFrame:
    return (
        features.group_by(*INSTANCE_COLUMNS, "challenges_per_bin")
        .agg(
            pl.col("total_challenges").first(),
            pl.col("precision_curve_mae").median(),
            pl.col("enrollment_curve_mae").first(),
            pl.col("total_curve_mae").median(),
            pl.col("x50_precision_error").abs().median().alias("x50_precision"),
            pl.col("x50_enrollment_error").abs().first().alias("x50_enrollment"),
            pl.col("x50_total_error").abs().median().alias("x50_total"),
            pl.col("width_precision_error")
            .abs()
            .median()
            .alias("width_precision"),
            pl.col("width_enrollment_error")
            .abs()
            .first()
            .alias("width_enrollment"),
            pl.col("width_total_error").abs().median().alias("width_total"),
        )
        .sort("challenges_per_bin")
    )


def plot_tradeoff(features: pl.DataFrame, output: Path, dpi: int) -> None:
    summary = _instance_summary(features)
    panels = (
        ("total_curve_mae", "Total curve MAE"),
        ("x50_total", r"Median total $|x_{50}|$ error"),
        ("width_total", r"Median total $|W_{10-90}|$ error"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (column, label) in zip(axes, panels, strict=True):
        for group in summary.partition_by(INSTANCE_COLUMNS, maintain_order=True):
            ax.plot(
                group["total_challenges"],
                group[column],
                color="#4477AA",
                alpha=0.28,
                linewidth=0.9,
            )
        population = (
            summary.group_by("challenges_per_bin")
            .agg(
                pl.col("total_challenges").mean(),
                pl.col(column).median().alias(column),
            )
            .sort("challenges_per_bin")
        )
        ax.plot(
            population["total_challenges"],
            population[column],
            color="black",
            marker="o",
            linewidth=2,
            label="population median",
        )
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Actual challenge evaluations", ylabel=label)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, output / "margin_bank_curve_error.png", dpi)


def plot_feature_precision(features: pl.DataFrame, output: Path, dpi: int) -> None:
    counts = features["challenges_per_bin"].unique().sort().to_list()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, column, label in (
        (axes[0], "x50_precision_error", r"$x_{50}$ precision error"),
        (axes[1], "width_precision_error", r"$W_{10-90}$ precision error"),
    ):
        values = [
            features.filter(pl.col("challenges_per_bin") == count)[column]
            .drop_nulls()
            .to_numpy()
            for count in counts
        ]
        ax.boxplot(values, tick_labels=[str(count) for count in counts], showfliers=False)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Challenges per margin bin", ylabel=label)
        ax.grid(axis="y", alpha=0.25)
    save(fig, output / "margin_bank_feature_precision.png", dpi)


def plot_error_decomposition(features: pl.DataFrame, output: Path, dpi: int) -> None:
    summary = _instance_summary(features)
    panels = (
        (
            ("precision_curve_mae", "enrollment_curve_mae", "total_curve_mae"),
            "Curve MAE",
        ),
        (("x50_precision", "x50_enrollment", "x50_total"), r"$|x_{50}|$ error"),
        (
            ("width_precision", "width_enrollment", "width_total"),
            r"$|W_{10-90}|$ error",
        ),
    )
    colors = ("#4477AA", "#CC6677", "#228833")
    labels = ("precision", "enrollment", "total")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (columns, ylabel) in zip(axes, panels, strict=True):
        for column, color, label in zip(columns, colors, labels, strict=True):
            population = (
                summary.group_by("challenges_per_bin")
                .agg(
                    pl.col("total_challenges").mean(),
                    pl.col(column).median().alias(column),
                )
                .sort("challenges_per_bin")
            )
            ax.plot(
                population["total_challenges"],
                population[column],
                color=color,
                marker="o",
                linewidth=1.8,
                label=label,
            )
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Actual challenge evaluations", ylabel=ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(fig, output / "margin_bank_error_decomposition.png", dpi)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curves = pl.read_parquet(args.artifacts / "bank_curves.parquet")
    reference = pl.read_parquet(args.artifacts / "reference_curves.parquet")
    features = pl.read_parquet(args.artifacts / "bank_features.parquet")
    if features.is_empty() or curves.is_empty() or reference.is_empty():
        raise ValueError("Margin-bank artifacts are empty")
    plot_curve_examples(curves, reference, args.output, args.dpi)
    plot_tradeoff(features, args.output, args.dpi)
    plot_feature_precision(features, args.output, args.dpi)
    plot_error_decomposition(features, args.output, args.dpi)


if __name__ == "__main__":
    main()
