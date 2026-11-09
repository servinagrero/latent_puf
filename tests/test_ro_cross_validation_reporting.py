import runpy
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parents[1]
REPORT = runpy.run_path(ROOT / "scripts/summarize_ro_cross_validation.py")
device_metric_summary = REPORT["device_metric_summary"]
population_metric_summary = REPORT["population_metric_summary"]
paired_mode_summary = REPORT["paired_mode_summary"]
latex_escape = REPORT["latex_escape"]


def _metric_rows() -> pl.DataFrame:
    rows = []
    for mode, device, folds, values in (
        ("fixed", "a", (0, 1), (0.0, 100.0)),
        ("fixed", "b", (0,), (10.0,)),
        ("loo", "a", (0,), (5.0,)),
    ):
        for fold, value in zip(folds, values, strict=True):
            rows.append(
                {
                    "mode": mode,
                    "dataset": "RO_test",
                    "device_id": device,
                    "fold": fold,
                    "selector": "latent_nearest",
                    "neighbors": 4,
                    "stratum_type": "all",
                    "stratum": "all",
                    "targets": 10,
                    "reconstruction_mae": value,
                    "spatial_brier": value,
                    "temporal_brier": 1.0,
                    "excess_brier": value - 1,
                    "normalized_radius": 0.01,
                    "shared_oscillator_fraction": 0.0,
                }
            )
    return pl.DataFrame(rows)


def test_folds_are_aggregated_before_devices():
    devices = device_metric_summary(_metric_rows())
    population = population_metric_summary(devices)
    fixed = population.filter(pl.col("mode") == "fixed")
    assert fixed["reconstruction_mae_median"][0] == pytest.approx(30.0)
    assert fixed["devices"][0] == 2


def test_pairing_uses_only_common_devices():
    paired = paired_mode_summary(device_metric_summary(_metric_rows()))
    assert paired["device_id"].to_list() == ["a"]
    assert paired["fixed_reconstruction_mae"][0] == pytest.approx(50.0)
    assert paired["loo_reconstruction_mae"][0] == pytest.approx(5.0)


def test_strata_and_latex_escaping_remain_distinct():
    rows = pl.concat(
        [
            _metric_rows(),
            _metric_rows().with_columns(
                pl.lit("ber").alias("stratum_type"),
                pl.lit("(0.25,0.5]").alias("stratum"),
            ),
        ]
    )
    population = population_metric_summary(device_metric_summary(rows))
    assert set(population["stratum_type"]) == {"all", "ber"}
    assert latex_escape("RO_test&set") == r"RO\_test\&set"
