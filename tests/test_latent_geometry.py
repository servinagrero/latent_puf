import numpy as np
import pytest

from latent_puf.latent_geometry import (
    aggregate_pair_distances,
    hamming_distances,
    nearest_indices,
    neighborhood_metric_rows,
    sample_pairs,
)


def test_sample_pairs_are_distinct_and_deterministic():
    first = sample_pairs(10, 100, np.random.default_rng(3))
    second = sample_pairs(10, 100, np.random.default_rng(3))
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:, 0] != first[:, 1])
    assert first.min() >= 0
    assert first.max() < 10


def test_hamming_distances_for_selected_pairs():
    values = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    np.testing.assert_array_equal(hamming_distances(values, pairs), [1, 3, 2])


def test_pair_aggregation_preserves_counts_and_means():
    rows = aggregate_pair_distances(
        distances=np.array([1, 1, 2]),
        margin_differences=np.array([0.2, 0.4, 1.0]),
        probability_differences=np.array([0.1, 0.3, 0.8]),
        label_agreement=np.array([True, False, False]),
    )
    assert sum(int(row["pairs"]) for row in rows) == 3
    assert rows[0]["distance"] == 1
    assert rows[0]["mean_abs_margin_difference"] == pytest.approx(0.3)
    assert rows[0]["response_label_agreement"] == pytest.approx(0.5)


def test_neighborhood_metrics_recover_known_ordering():
    margins = np.array([-2.0, -1.0, 1.0, 2.0])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    labels = probabilities.astype(np.int8)
    neighbors = np.array([[1, 2], [0, 2], [3, 1], [2, 1]])
    rows = neighborhood_metric_rows(
        neighbor_indices=neighbors,
        margins=margins,
        probabilities=probabilities,
        labels=labels,
        neighbor_counts=(1, 2),
    )
    assert rows[0]["mean_abs_neighbor_margin_difference"] == pytest.approx(1.0)
    assert rows[0]["mean_abs_neighbor_probability_difference"] == pytest.approx(0.0)
    assert rows[0]["response_label_agreement"] == pytest.approx(1.0)
    assert rows[1]["mean_abs_neighbor_margin_difference"] > 1.0


def test_neighborhood_metrics_reject_misaligned_inputs():
    with pytest.raises(ValueError, match="lengths differ"):
        neighborhood_metric_rows(
            neighbor_indices=np.zeros((2, 1), dtype=np.int64),
            margins=np.zeros(2),
            probabilities=np.zeros(1),
            labels=np.zeros(2),
            neighbor_counts=(1,),
        )


def test_nearest_indices_exclude_self_but_keep_closest_row():
    values = np.array([[0.0], [1.0], [3.0]])
    indices = nearest_indices(values, 1, "euclidean")
    np.testing.assert_array_equal(indices[:, 0], [1, 0, 1])
