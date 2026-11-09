"""Create CSV result tables from room-temperature Parquet artifacts."""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp, wasserstein_distance

LOGGER = logging.getLogger(__name__)
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
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def _summary_row(
    group: pl.DataFrame, selector: str, k: int, prefix: dict[str, object]
) -> dict[str, object]:
    temporal = group["p_temporal"].to_numpy()
    spatial = group["p_spatial"].to_numpy()
    error = spatial - temporal
    temporal_ber = np.minimum(temporal, 1 - temporal)
    spatial_ber = np.minimum(spatial, 1 - spatial)
    temporal_pairwise_ber = 2 * temporal * (1 - temporal)
    spatial_pairwise_ber = 2 * spatial * (1 - spatial)
    return {
        **prefix,
        "selector": selector,
        "k": k,
        "n_spatial": int(group["n_spatial"][0]),
        "references": len(group),
        "probability_bias": float(error.mean()),
        "probability_mae": float(np.abs(error).mean()),
        "probability_rmse": float(np.sqrt(np.mean(error**2))),
        "within_0_05": float(np.mean(np.abs(error) <= 0.05)),
        "temporal_mean_ber": float(temporal_ber.mean()),
        "spatial_mean_ber": float(spatial_ber.mean()),
        "ber_bias": float(spatial_ber.mean() - temporal_ber.mean()),
        "temporal_pairwise_ber": float(temporal_pairwise_ber.mean()),
        "spatial_pairwise_ber": float(spatial_pairwise_ber.mean()),
        "pairwise_ber_bias": float(
            spatial_pairwise_ber.mean() - temporal_pairwise_ber.mean()
        ),
        "wasserstein_probability": float(wasserstein_distance(temporal, spatial)),
        "ks_probability": float(ks_2samp(temporal, spatial).statistic),
        "mean_margin_bias": float(group["margin_bias"].to_numpy().mean()),
        "mean_absolute_margin_bias": float(
            np.abs(group["margin_bias"].to_numpy()).mean()
        ),
        "mean_margin_mae": float(group["margin_mae"].to_numpy().mean()),
        "mean_margin_radius": float(group["margin_radius"].to_numpy().mean()),
        "mean_below_fraction": float(group["below_fraction"].to_numpy().mean()),
        "mean_balance_shortfall": float(group["balance_shortfall"].to_numpy().mean()),
    }


def _selector_summaries(data: pl.DataFrame, group_by_instance: bool) -> pl.DataFrame:
    columns = (
        [*INSTANCE_COLUMNS, "selector", "k"] if group_by_instance else ["selector", "k"]
    )
    rows: list[dict[str, object]] = []
    for key, group in data.group_by(columns, maintain_order=True):
        values = key if isinstance(key, tuple) else (key,)
        mapping = dict(zip(columns, values, strict=True))
        selector = str(mapping.pop("selector"))
        k = int(mapping.pop("k"))
        rows.append(_summary_row(group, selector, k, mapping))
    return pl.DataFrame(rows).sort(columns)


def _reliability_distribution(data: pl.DataFrame) -> pl.DataFrame:
    temporal = data.select("device_id", "slice", "baseline_id", "p_temporal").unique()
    rows: list[dict[str, object]] = []

    def append(
        source: str, selector: str, k: int | None, probabilities: np.ndarray
    ) -> None:
        ber = np.minimum(probabilities, 1 - probabilities)
        pairwise = 2 * probabilities * (1 - probabilities)
        rows.append(
            {
                "source": source,
                "selector": selector,
                "k": k,
                "observations": len(probabilities),
                "mean_ber": float(ber.mean()),
                "mean_majority_reliability": float(1 - ber.mean()),
                "mean_pairwise_ber": float(pairwise.mean()),
                "mean_pairwise_reliability": float(1 - pairwise.mean()),
                "ber_q50": float(np.quantile(ber, 0.50)),
                "ber_q90": float(np.quantile(ber, 0.90)),
                "ber_q95": float(np.quantile(ber, 0.95)),
                "ber_q99": float(np.quantile(ber, 0.99)),
                "ber_max": float(ber.max()),
            }
        )

    append("temporal", "temporal", None, temporal["p_temporal"].to_numpy())
    for (selector, k), group in data.group_by("selector", "k", maintain_order=True):
        append("spatial", str(selector), int(k), group["p_spatial"].to_numpy())
    return pl.DataFrame(rows).sort("source", "selector", "k", nulls_last=False)


def _curve_agreement(artifacts: Path) -> pl.DataFrame:
    temporal = pl.read_parquet(artifacts / "temporal_margin_bins.parquet")
    spatial = pl.read_parquet(artifacts / "margin_bins.parquet")
    joined = temporal.join(
        spatial,
        on=[*INSTANCE_COLUMNS, "bin_id"],
        how="inner",
        suffix="_spatial",
    ).filter(pl.col("n_challenges") >= 5)
    rows: list[dict[str, object]] = []
    for key, group in joined.group_by(INSTANCE_COLUMNS, maintain_order=True):
        prefix = dict(zip(INSTANCE_COLUMNS, key, strict=True))
        temporal_p = group["mean_response"].to_numpy()
        spatial_p = group["mean_response_spatial"].to_numpy()
        for region, selected in (
            ("all", np.ones(len(group), dtype=bool)),
            ("transition", (temporal_p >= 0.05) & (temporal_p <= 0.95)),
        ):
            if selected.sum() < 2:
                continue
            error = spatial_p[selected] - temporal_p[selected]
            rows.append(
                {
                    **prefix,
                    "region": region,
                    "bins": int(selected.sum()),
                    "bias": float(error.mean()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "max_absolute_error": float(np.abs(error).max()),
                    "correlation": float(
                        np.corrcoef(temporal_p[selected], spatial_p[selected])[0, 1]
                    ),
                }
            )
    return pl.DataFrame(rows).sort([*INSTANCE_COLUMNS, "region"])


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    data = pl.read_parquet(args.artifacts / "selector_metrics.parquet")
    outputs = {
        "selector_by_k.csv": _selector_summaries(data, False),
        "selector_by_instance_and_k.csv": _selector_summaries(data, True),
        "reliability_distribution.csv": _reliability_distribution(data),
        "curve_agreement.csv": _curve_agreement(args.artifacts),
        "base_challenge_bias.csv": data.select(
            *INSTANCE_COLUMNS,
            "baseline_id",
            "selector",
            "k",
            "n_spatial",
            "p_temporal",
            "p_spatial",
            "base_margin",
            "signed_error",
            "absolute_error",
            "margin_bias",
            "margin_mae",
            "margin_radius",
            "below_fraction",
            "balance_shortfall",
        ).sort(*INSTANCE_COLUMNS, "baseline_id", "selector", "k"),
    }
    for filename, frame in outputs.items():
        path = args.output / filename
        frame.write_csv(path)
        LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
