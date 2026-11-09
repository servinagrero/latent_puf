import h5py
import numpy as np
import polars as pl
import pytest

from latent_puf.ro_cross_validation import (
    ROCrossValidationConfig,
    one_shot_probability,
    run_ro_cross_validation,
    temporal_leave_one_out,
)


def test_leave_one_out_excludes_the_held_out_response():
    responses = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    np.testing.assert_allclose(temporal_leave_one_out(responses, 0), [1.0, 0.5])
    np.testing.assert_allclose(temporal_leave_one_out(responses, 2), [0.5, 1.0])


def test_one_shot_probability_uses_only_held_out_neighbors():
    held_out = np.array([0.0, 1.0, 1.0, 0.0])
    selected = np.array([[1, 2], [0, 3]])
    np.testing.assert_allclose(one_shot_probability(held_out, selected, 1), [1.0, 0.0])
    np.testing.assert_allclose(one_shot_probability(held_out, selected, 2), [1.0, 0.0])


@pytest.mark.parametrize("mode", ["fixed", "loo"])
def test_cross_validation_writes_metrics_and_checkpoints(tmp_path, mode):
    input_path = tmp_path / "tiny.h5"
    output_path = tmp_path / mode
    pairs = np.array(
        [[0, 10], [0, 11], [1, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15]]
    )
    base = np.array([-2.0, -1.0, -0.2, -0.05, 0.05, 0.2, 1.0, 2.0])
    shifts = np.array([-0.1, 0.1, -0.05, 0.05, -0.02, 0.02])
    values = (base[:, None] + shifts[None, :])[None, :, :].astype(np.float32)
    with h5py.File(input_path, "w") as handle:
        group = handle.create_group("tiny")
        group.create_dataset("frequency_differences", data=values, chunks=(1, 4, 6))
        group.create_dataset("frequency_pairs", data=pairs)
        group.create_dataset("device_ids", data=np.array([b"device_1"]))
        group.create_dataset("sample_ids", data=np.arange(6))

    run_ro_cross_validation(
        input_path=input_path,
        output_dir=output_path,
        config=ROCrossValidationConfig(
            mode=mode,
            neighbors=(1, 2),
            targets_per_stratum=1,
            latent_edges=(0.0, 0.1, 0.5, float("inf")),
            pair_block_size=4,
            curve_bins=4,
            max_folds=2,
            seed=1,
        ),
    )

    metrics = pl.read_parquet(output_path / "cv_metrics.parquet")
    curves = pl.read_parquet(output_path / "cv_curve_metrics.parquet")
    summary = pl.read_parquet(output_path / "cv_dataset_summary.parquet")
    assert not metrics.is_empty()
    assert not curves.is_empty()
    assert summary["mode"].to_list() == [mode]
    assert metrics["device_id"].unique().to_list() == ["device_1"]
    assert set(metrics["stratum_type"]) >= {"all", "latent", "ber"}
