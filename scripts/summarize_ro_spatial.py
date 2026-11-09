"""Create CSV and LaTeX summaries from RO temporal--spatial artifacts."""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def dataset_summary(datasets: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Combine HDF5 structure and device-level PUF statistics."""

    device_summary = devices.group_by("dataset").agg(
        pl.len().alias("analyzed_devices"),
        pl.col("valid_pairs").median().alias("valid_pairs_median"),
        pl.col("response_balance").median().alias("response_balance_median"),
        pl.col("reference_reliability").median().alias("reference_reliability_median"),
        pl.col("pairwise_repeatability")
        .median()
        .alias("pairwise_repeatability_median"),
        pl.col("noise_std_mean").median().alias("noise_std_median"),
    )
    return datasets.join(device_summary, on="dataset", how="left").sort("dataset")


def spatial_summary(spatial: pl.DataFrame) -> pl.DataFrame:
    """Aggregate spatial-estimator metrics across devices."""

    return (
        spatial.group_by("dataset", "selector", "neighbors")
        .agg(
            pl.col("device_id").n_unique().alias("devices"),
            pl.col("mean_latent_radius").median().alias("latent_radius_median"),
            pl.col("spatial_bias_mae").median().alias("spatial_bias_mae_median"),
            pl.col("snapshot_precision_mae")
            .median()
            .alias("snapshot_precision_mae_median"),
            pl.col("snapshot_total_mae").median().alias("total_mae_median"),
            pl.col("snapshot_total_mae").quantile(0.25).alias("total_mae_q25"),
            pl.col("snapshot_total_mae").quantile(0.75).alias("total_mae_q75"),
            pl.col("shared_oscillator_fraction")
            .median()
            .alias("shared_fraction_median"),
        )
        .sort("dataset", "selector", "neighbors")
    )


def latex_escape(value: object) -> str:
    """Escape a scalar for use in a LaTeX table cell."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _integer(value: Any) -> str:
    if value is None:
        return "--"
    return str(int(value))


def _write_latex(
    path: Path,
    frame: pl.DataFrame,
    *,
    caption: str,
    label: str,
    headers: tuple[str, ...],
    columns: tuple[str, ...],
    formatters: tuple[Callable[[object], str], ...],
) -> None:
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in frame.select(columns).iter_rows():
        lines.append(
            " & ".join(formatter(value) for formatter, value in zip(formatters, row))
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote %s", path)


def write_dataset_latex(frame: pl.DataFrame, path: Path) -> None:
    columns = (
        "dataset",
        "analyzed_devices",
        "oscillators",
        "pairs",
        "samples",
        "graph_components",
        "valid_pairs_median",
        "response_balance_median",
        "reference_reliability_median",
        "pairwise_repeatability_median",
    )
    _write_latex(
        path,
        frame,
        caption="RO PUF dataset structure and median device-level statistics.",
        label="tab:ro-datasets",
        headers=(
            "Dataset",
            "$D$",
            "ROs",
            "$B$",
            "$S$",
            "Components",
            "Valid pairs",
            "Balance",
            "$R_{\\mathrm{ref}}$",
            "$R_{\\mathrm{pair}}$",
        ),
        columns=columns,
        formatters=(
            latex_escape,
            _integer,
            _integer,
            _integer,
            _integer,
            _integer,
            _integer,
            _number,
            _number,
            _number,
        ),
    )


def write_spatial_latex(frame: pl.DataFrame, path: Path) -> None:
    columns = (
        "dataset",
        "selector",
        "neighbors",
        "devices",
        "latent_radius_median",
        "spatial_bias_mae_median",
        "snapshot_precision_mae_median",
        "total_mae_median",
        "total_mae_q25",
        "total_mae_q75",
        "shared_fraction_median",
    )
    _write_latex(
        path,
        frame,
        caption="Temporal--spatial estimation error across RO PUF devices.",
        label="tab:ro-spatial",
        headers=(
            "Dataset",
            "Selector",
            "$k$",
            "$D$",
            "$\\rho_k$",
            "Bias MAE",
            "Precision MAE",
            "Total MAE",
            "$Q_{25}$",
            "$Q_{75}$",
            "Shared",
        ),
        columns=columns,
        formatters=(
            latex_escape,
            latex_escape,
            _integer,
            _integer,
            _number,
            _number,
            _number,
            _number,
            _number,
            _number,
            _number,
        ),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = pl.read_parquet(args.artifacts / "dataset_summary.parquet")
    devices = pl.read_parquet(args.artifacts / "device_metrics.parquet")
    spatial = pl.read_parquet(args.artifacts / "spatial_metrics.parquet")
    if any(frame.is_empty() for frame in (datasets, devices, spatial)):
        raise ValueError("RO spatial artifacts are empty")

    dataset_table = dataset_summary(datasets, devices)
    spatial_table = spatial_summary(spatial)
    dataset_table.write_csv(args.output / "dataset_summary.csv")
    spatial_table.write_csv(args.output / "spatial_summary.csv")
    write_dataset_latex(dataset_table, args.output / "dataset_summary.tex")
    write_spatial_latex(spatial_table, args.output / "spatial_summary.tex")
    LOGGER.info("Wrote CSV summaries to %s", args.output)


if __name__ == "__main__":
    main()
