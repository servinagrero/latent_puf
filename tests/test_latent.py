import numpy as np
import pytest

from latent_puf.latent import BinaryLatentBank


def test_margin_bank_preserves_an_existing_signed_margin():
    margin = np.array([-1.0, 0.0, 2.0, np.nan])
    bank = BinaryLatentBank.from_margin(margin)

    np.testing.assert_array_equal(bank.z, margin)
    np.testing.assert_array_equal(bank.margin, margin)
    np.testing.assert_array_equal(bank.valid, [True, True, True, False])
    np.testing.assert_allclose(
        bank.nominal_response,
        [0.0, 0.0, 1.0, np.nan],
        equal_nan=True,
    )


def test_threshold_bank_supports_both_decision_orientations():
    z = np.array([0.5, 1.0, 1.5])

    above = BinaryLatentBank.from_threshold(z, threshold=1.0)
    below = BinaryLatentBank.from_threshold(
        z,
        threshold=1.0,
        one_above_threshold=False,
    )

    np.testing.assert_allclose(above.margin, [-0.5, 0.0, 0.5])
    np.testing.assert_allclose(below.margin, [0.5, 0.0, -0.5])
    np.testing.assert_allclose(above.nominal_response, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(below.nominal_response, [1.0, 0.0, 0.0])


def test_threshold_can_vary_per_response():
    bank = BinaryLatentBank.from_threshold(
        np.array([1.0, 2.0, 3.0]),
        threshold=np.array([0.5, 2.5, 2.0]),
    )

    np.testing.assert_allclose(bank.margin, [0.5, -0.5, 1.0])


def test_vector_latent_state_accepts_a_separate_scalar_margin():
    z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bank = BinaryLatentBank(z=z, margin=np.array([-1.0, 0.0, 1.0]))

    assert bank.z.shape == (3, 2)
    np.testing.assert_allclose(bank.margin, [-1.0, 0.0, 1.0])


def test_bank_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="same responses"):
        BinaryLatentBank(
            z=np.array([1.0, 2.0]),
            margin=np.array([1.0]),
        )

    with pytest.raises(ValueError, match="one-dimensional z"):
        BinaryLatentBank.from_threshold(np.ones((2, 2)))

    with pytest.raises(ValueError, match="not compatible"):
        BinaryLatentBank.from_threshold(
            np.ones(3),
            threshold=np.ones(2),
        )
