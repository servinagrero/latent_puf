import runpy
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parents[1]
PLOT = runpy.run_path(ROOT / "scripts/plot_ro_spatial.py")
SUMMARY = runpy.run_path(ROOT / "scripts/summarize_ro_spatial.py")
normalized_curves = PLOT["normalized_curves"]
dataset_summary = SUMMARY["dataset_summary"]
latex_escape = SUMMARY["latex_escape"]
spatial_summary = SUMMARY["spatial_summary"]


def test_reporting_aggregates_devices_and_normalizes_latent_coordinates():
    datasets = pl.DataFrame(
        {
            "dataset": ["RO_test_set"],
            "oscillators": [4],
            "pairs": [4],
            "samples": [6],
            "graph_components": [1],
        }
    )
    devices = pl.DataFrame(
        {
            "dataset": ["RO_test_set", "RO_test_set"],
            "device_id": ["a", "b"],
            "valid_pairs": [4, 2],
            "response_balance": [0.4, 0.6],
            "reference_reliability": [0.8, 1.0],
            "pairwise_repeatability": [0.7, 0.9],
            "noise_std_mean": [2.0, 4.0],
            "latent_mean": [10.0, 20.0],
            "latent_std": [2.0, 4.0],
        }
    )
    summary = dataset_summary(datasets, devices)
    assert summary["valid_pairs_median"][0] == pytest.approx(3.0)
    assert summary["response_balance_median"][0] == pytest.approx(0.5)

    curves = pl.DataFrame(
        {
            "dataset": ["RO_test_set", "RO_test_set"],
            "device_id": ["a", "b"],
            "bin": [0, 0],
            "latent_mean": [12.0, 16.0],
        }
    )
    normalized = normalized_curves(curves, devices)
    assert normalized["normalized_latent"].to_list() == pytest.approx([6.0, 4.0])


def test_spatial_summary_preserves_selectors_and_quartiles():
    spatial = pl.DataFrame(
        {
            "dataset": ["x"] * 3,
            "device_id": [1, 2, 3],
            "selector": ["latent_nearest"] * 3,
            "neighbors": [4] * 3,
            "mean_latent_radius": [1.0, 2.0, 3.0],
            "spatial_bias_mae": [0.1, 0.2, 0.3],
            "snapshot_precision_mae": [0.2, 0.3, 0.4],
            "snapshot_total_mae": [0.1, 0.2, 0.3],
            "shared_oscillator_fraction": [0.0, 0.1, 0.2],
        }
    )
    summary = spatial_summary(spatial)
    assert summary["selector"][0] == "latent_nearest"
    assert summary["total_mae_median"][0] == pytest.approx(0.2)
    assert summary["total_mae_q25"][0] <= summary["total_mae_median"][0]
    assert summary["total_mae_q75"][0] >= summary["total_mae_median"][0]


def test_latex_escape_handles_dataset_names():
    assert latex_escape("RO_test&set") == r"RO\_test\&set"
