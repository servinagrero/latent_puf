"""Summarize fixed-bank and LOO RO cross-validation artifacts."""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

LOGGER = logging.getLogger(__name__)
METRIC_COLUMNS = (
    "reconstruction_mae",
    "spatial_brier",
    "temporal_brier",
    "excess_brier",
    "normalized_radius",
    "shared_oscillator_fraction",
)
CURVE_COLUMNS = (
    "curve_mae",
    "curve_max_error",
    "x50_error",
    "width_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--loo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def load_metrics(fixed: Path, loo: Path, filename: str) -> pl.DataFrame:
    """Load both validation modes with an explicit mode column."""

    return pl.concat(
        [
            pl.read_parquet(fixed / filename).with_columns(
                pl.lit("fixed").alias("mode")
            ),
            pl.read_parquet(loo / filename).with_columns(pl.lit("loo").alias("mode")),
        ],
        how="diagonal_relaxed",
    )


def device_metric_summary(metrics: pl.DataFrame) -> pl.DataFrame:
    """Aggregate correlated folds before devices become population observations."""

    keys = (
        "mode",
        "dataset",
        "device_id",
        "selector",
        "neighbors",
        "stratum_type",
        "stratum",
    )
    return metrics.group_by(*keys).agg(
        pl.col("targets").median(),
        *(pl.col(column).median() for column in METRIC_COLUMNS),
    )


def population_metric_summary(devices: pl.DataFrame) -> pl.DataFrame:
    """Compute device-level medians and interquartile ranges."""

    keys = ("mode", "dataset", "selector", "neighbors", "stratum_type", "stratum")
    expressions: list[pl.Expr] = [
        pl.col("device_id").n_unique().alias("devices"),
        pl.col("targets").median().alias("targets_median"),
    ]
    for column in METRIC_COLUMNS:
        expressions.extend(
            (
                pl.col(column).median().alias(f"{column}_median"),
                pl.col(column).quantile(0.25).alias(f"{column}_q25"),
                pl.col(column).quantile(0.75).alias(f"{column}_q75"),
            )
        )
    return devices.group_by(*keys).agg(*expressions).sort(*keys)


def device_curve_summary(curves: pl.DataFrame) -> pl.DataFrame:
    keys = ("mode", "dataset", "device_id", "selector", "neighbors")
    return curves.group_by(*keys).agg(
        *(pl.col(column).median() for column in CURVE_COLUMNS)
    )


def population_curve_summary(devices: pl.DataFrame) -> pl.DataFrame:
    keys = ("mode", "dataset", "selector", "neighbors")
    expressions: list[pl.Expr] = [pl.col("device_id").n_unique().alias("devices")]
    for column in CURVE_COLUMNS:
        values = (
            pl.col(column).abs()
            if column in {"x50_error", "width_error"}
            else pl.col(column)
        )
        expressions.extend(
            (
                values.median().alias(f"{column}_median"),
                values.quantile(0.25).alias(f"{column}_q25"),
                values.quantile(0.75).alias(f"{column}_q75"),
            )
        )
    return devices.group_by(*keys).agg(*expressions).sort(*keys)


def paired_mode_summary(devices: pl.DataFrame) -> pl.DataFrame:
    """Pair fixed and LOO metrics only where dataset and device coincide."""

    selected = devices.filter(
        (pl.col("selector") == "latent_nearest") & (pl.col("stratum_type") == "all")
    ).select(
        "mode",
        "dataset",
        "device_id",
        "neighbors",
        "reconstruction_mae",
        "excess_brier",
    )
    fixed = selected.filter(pl.col("mode") == "fixed").drop("mode")
    loo = selected.filter(pl.col("mode") == "loo").drop("mode")
    return (
        fixed.join(
            loo,
            on=["dataset", "device_id", "neighbors"],
            how="inner",
            suffix="_loo",
        )
        .rename(
            {
                "reconstruction_mae": "fixed_reconstruction_mae",
                "excess_brier": "fixed_excess_brier",
                "reconstruction_mae_loo": "loo_reconstruction_mae",
                "excess_brier_loo": "loo_excess_brier",
            }
        )
        .with_columns(
            (
                pl.col("fixed_reconstruction_mae") - pl.col("loo_reconstruction_mae")
            ).alias("reconstruction_mae_delta"),
            (pl.col("fixed_excess_brier") - pl.col("loo_excess_brier")).alias(
                "excess_brier_delta"
            ),
        )
        .sort("dataset", "device_id", "neighbors")
    )


def latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _number(value: Any, digits: int = 4) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _integer(value: Any) -> str:
    return "--" if value is None else str(int(value))


def _write_latex(
    path: Path,
    frame: pl.DataFrame,
    *,
    caption: str,
    label: str,
    headers: tuple[str, ...],
    columns: tuple[str, ...],
    formatters: tuple[Callable[[Any], str], ...],
) -> None:
    alignment = "ll" + "r" * (len(columns) - 2)
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


def _write_tables(
    output: Path,
    overall: pl.DataFrame,
    boundary: pl.DataFrame,
    curves: pl.DataFrame,
) -> None:
    overall_tex = overall.filter(pl.col("selector") == "latent_nearest")
    _write_latex(
        output / "overall_by_mode.tex",
        overall_tex,
        caption="Fixed-bank and leave-one-out RO spatial validation.",
        label="tab:ro-cv-overall",
        headers=(
            "Mode",
            "Dataset",
            "$k$",
            "$D$",
            "MAE",
            "Spatial BS",
            "Temporal BS",
            "$\\Delta$BS",
        ),
        columns=(
            "mode",
            "dataset",
            "neighbors",
            "devices",
            "reconstruction_mae_median",
            "spatial_brier_median",
            "temporal_brier_median",
            "excess_brier_median",
        ),
        formatters=(
            latex_escape,
            latex_escape,
            _integer,
            _integer,
            _number,
            _number,
            _number,
            _number,
        ),
    )
    boundary_tex = boundary.filter(
        (pl.col("selector") == "latent_nearest")
        & (pl.col("stratum").is_in(["(0.1,0.25]", "(0.25,0.5]"]))
    )
    _write_latex(
        output / "boundary_by_mode.tex",
        boundary_tex,
        caption="Cross-validation within temporally unstable RO-pair strata.",
        label="tab:ro-cv-boundary",
        headers=(
            "Mode",
            "Dataset",
            "BER stratum",
            "$k$",
            "$D$",
            "MAE",
            "Spatial BS",
            "$\\Delta$BS",
        ),
        columns=(
            "mode",
            "dataset",
            "stratum",
            "neighbors",
            "devices",
            "reconstruction_mae_median",
            "spatial_brier_median",
            "excess_brier_median",
        ),
        formatters=(
            latex_escape,
            latex_escape,
            latex_escape,
            _integer,
            _integer,
            _number,
            _number,
            _number,
        ),
    )
    curve_tex = curves.filter(pl.col("selector") == "latent_nearest")
    _write_latex(
        output / "curve_features_by_mode.tex",
        curve_tex,
        caption="One-shot spatial curve reconstruction errors.",
        label="tab:ro-cv-curves",
        headers=("Mode", "Dataset", "$k$", "$D$", "Curve MAE", "$|x_{50}|$", "$|W|$"),
        columns=(
            "mode",
            "dataset",
            "neighbors",
            "devices",
            "curve_mae_median",
            "x50_error_median",
            "width_error_median",
        ),
        formatters=(
            latex_escape,
            latex_escape,
            _integer,
            _integer,
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
    metrics = load_metrics(args.fixed, args.loo, "cv_metrics.parquet")
    curves = load_metrics(args.fixed, args.loo, "cv_curve_metrics.parquet")
    device_metrics = device_metric_summary(metrics)
    device_curves = device_curve_summary(curves)
    population = population_metric_summary(device_metrics)
    curve_population = population_curve_summary(device_curves)
    overall = population.filter(pl.col("stratum_type") == "all")
    boundary = population.filter(pl.col("stratum_type").is_in(["ber", "latent"]))
    paired = paired_mode_summary(device_metrics)
    outputs = {
        "overall_by_mode.csv": overall,
        "boundary_by_mode.csv": boundary,
        "curve_features_by_mode.csv": curve_population,
        "paired_fixed_vs_loo.csv": paired,
    }
    for filename, frame in outputs.items():
        frame.write_csv(args.output / filename)
        LOGGER.info("Wrote %s", args.output / filename)
    _write_tables(args.output, overall, boundary, curve_population)


if __name__ == "__main__":
    main()
