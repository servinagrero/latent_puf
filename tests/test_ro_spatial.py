import h5py
import numpy as np
import polars as pl

from latent_puf.ro_data import split_samples
from latent_puf.ro_spatial import (
    ROSpatialConfig,
    analyze_ro_spatial,
    nearest_latent_indices,
)


def test_sample_split_is_disjoint_and_complete():
    enrollment, evaluation = split_samples(7, "alternating")
    assert set(enrollment).isdisjoint(evaluation)
    np.testing.assert_array_equal(
        np.sort(np.concatenate((enrollment, evaluation))), range(7)
    )


def test_nearest_latent_can_exclude_shared_oscillators():
    latent = np.array([-2.0, -1.9, -1.8, -1.7, 0.0])
    pairs = np.array([[0, 10], [0, 11], [1, 10], [1, 11], [2, 12]])
    unrestricted = nearest_latent_indices(latent, np.array([0]), 1)
    disjoint = nearest_latent_indices(
        latent,
        np.array([0]),
        1,
        pairs=pairs,
        disjoint_from_target=True,
    )
    assert unrestricted[0, 0] == 1
    assert disjoint[0, 0] == 3


def test_streaming_analysis_writes_expected_artifacts(tmp_path):
    input_path = tmp_path / "tiny.h5"
    output_path = tmp_path / "artifacts"
    pairs = np.array([[0, 2], [0, 3], [1, 2], [1, 3], [4, 5], [6, 7]])
    base = np.array([-2.0, -1.0, -0.2, 0.2, 1.0, 2.0])
    values = np.stack(
        [base + shift for shift in (-0.2, 0.1, -0.1, 0.2, 0.0, 0.1)], axis=1
    )[None, :, :]
    with h5py.File(input_path, "w") as handle:
        group = handle.create_group("tiny")
        group.create_dataset("frequency_differences", data=values, chunks=(1, 3, 6))
        group.create_dataset("frequency_pairs", data=pairs)
        group.create_dataset("device_ids", data=np.array([7]))
        group.create_dataset("sample_ids", data=np.arange(6))
        string_group = handle.create_group("tiny_string_ids")
        string_group.create_dataset(
            "frequency_differences", data=values, chunks=(1, 3, 6)
        )
        string_group.create_dataset("frequency_pairs", data=pairs)
        string_group.create_dataset("device_ids", data=np.array([b"device_8"]))
        string_group.create_dataset("sample_ids", data=np.arange(6))

    analyze_ro_spatial(
        input_path=input_path,
        output_dir=output_path,
        config=ROSpatialConfig(
            neighbors=(1,),
            pair_block_size=3,
            targets_per_device=3,
            curve_bins=3,
            seed=1,
        ),
    )

    assert (output_path / "manifest.json").is_file()
    summary = pl.read_parquet(output_path / "dataset_summary.parquet")
    spatial = pl.read_parquet(output_path / "spatial_metrics.parquet")
    devices = pl.read_parquet(output_path / "device_metrics.parquet")
    assert summary["pairs"].to_list() == [6, 6]
    assert devices.schema["device_id"] == pl.String
    assert devices["device_id"].to_list() == ["7", "device_8"]
    assert set(spatial["selector"]) == {
        "latent_nearest",
        "latent_nearest_disjoint",
        "random",
    }
