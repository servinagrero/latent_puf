import sqlite3

import numpy as np
import pytest

from latent_puf.margin_bank import (
    curve_error_metrics,
    monotone_curve_features,
    read_response_matrix,
    select_margin_bank,
)


def test_read_response_matrix_orders_samples_and_challenges():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE crps "
        "(acquisition_id INTEGER, sample_id INTEGER, challenge_id INTEGER, "
        "response INTEGER)"
    )
    rows = [
        (7, sample, challenge, (sample + challenge) % 2)
        for sample in range(2)
        for challenge in range(3)
    ]
    connection.executemany("INSERT INTO crps VALUES (?, ?, ?, ?)", rows[::-1])
    responses = read_response_matrix(connection, 7, 3)
    np.testing.assert_array_equal(responses, [[0, 1, 0], [1, 0, 1]])


def test_margin_bank_is_deterministic_nested_and_binned():
    margins = np.array([-0.9, -0.6, -0.2, 0.1, 0.4, 0.8])
    edges = np.array([-1.0, 0.0, 1.0])
    one = select_margin_bank(margins, edges, 1)
    two = select_margin_bank(margins, edges, 2)
    assert [values.tolist() for values in one] == [[1], [4]]
    assert all(set(small) <= set(large) for small, large in zip(one, two, strict=True))
    assert set(two[0]).isdisjoint(two[1])
    assert np.all(margins[two[0]] < 0)
    assert np.all(margins[two[1]] >= 0)


def test_monotone_curve_features_recover_linear_crossings():
    margins = np.linspace(-1, 1, 11)
    probabilities = (margins + 1) / 2
    result = monotone_curve_features(margins, probabilities, np.ones(11))
    assert result["x10"] == pytest.approx(-0.8)
    assert result["x50"] == pytest.approx(0.0)
    assert result["x90"] == pytest.approx(0.8)
    assert result["width_10_90"] == pytest.approx(1.6)


def test_monotone_curve_features_report_missing_crossings():
    result = monotone_curve_features(
        np.array([-1.0, 0.0, 1.0]),
        np.array([0.6, 0.7, 0.8]),
        np.ones(3),
    )
    assert result["x10"] is None
    assert result["x50"] is None
    assert result["x90"] is None
    assert result["width_10_90"] is None


def test_curve_error_decomposition_uses_one_common_reference():
    snapshot = np.array([0.0, 0.6, 1.0])
    bank = np.array([0.0, 0.5, 1.0])
    full = np.array([0.1, 0.4, 0.9])
    result = curve_error_metrics(snapshot=snapshot, bank=bank, full=full)
    precision = snapshot - bank
    enrollment = bank - full
    total = snapshot - full
    np.testing.assert_allclose(total, precision + enrollment)
    assert result["precision_curve_bias"] == pytest.approx(precision.mean())
    assert result["enrollment_curve_bias"] == pytest.approx(enrollment.mean())
    assert result["total_curve_bias"] == pytest.approx(total.mean())
    assert result["total_curve_mae"] == pytest.approx(np.abs(total).mean())


def test_curve_error_decomposition_rejects_empty_common_bins():
    with pytest.raises(ValueError, match="nonempty"):
        curve_error_metrics(
            snapshot=np.array([]), bank=np.array([]), full=np.array([])
        )


def test_curve_crossings_use_the_supplied_bin_centers():
    bin_centers = np.array([-0.75, -0.25, 0.25, 0.75])
    probabilities = np.array([0.0, 0.25, 0.75, 1.0])
    result = monotone_curve_features(bin_centers, probabilities, np.ones(4))
    assert result["x50"] == pytest.approx(0.0)
