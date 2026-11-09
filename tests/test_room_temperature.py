import numpy as np

from latent_puf.room_temperature import atf, selector_metrics, validate_gray_nesting


def test_atf_matches_suffix_parity():
    challenges = np.array([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.uint8)
    polar = np.where(challenges == 0, 1, -1)
    expected = np.cumprod(polar[:, ::-1], axis=1)[:, ::-1]
    np.testing.assert_array_equal(atf(challenges), expected)


def test_nested_gray_prefixes():
    values = np.arange(16, dtype=np.uint64)
    gray = values ^ (values >> 1)
    shifts = np.arange(4, dtype=np.uint64)
    mirrored = ((gray[:, None] >> shifts) & 1).astype(np.uint8)
    suffix = np.tile(np.array([[1, 0]], dtype=np.uint8), (16, 1))
    validate_gray_nesting(np.column_stack([mirrored, suffix]), 4, (1, 2, 3, 4))


def test_selector_metrics_exclude_base_and_are_deterministic():
    responses = np.array([1, 0, 0, 1, 1, 0, 1, 0], dtype=np.int8)
    margins = np.array([0.0, -0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    first = selector_metrics(
        responses, margins, 0.75, (1, 2, 3), np.random.default_rng(7)
    )
    second = selector_metrics(
        responses, margins, 0.75, (1, 2, 3), np.random.default_rng(7)
    )
    assert first == second
    gray_k1 = next(
        row for row in first if row["selector"] == "gray_prefix" and row["k"] == 1
    )
    assert gray_k1["n_spatial"] == 1
    assert gray_k1["p_spatial"] == 0.0
    assert gray_k1["signed_error"] == -0.75


def test_balanced_selector_uses_both_sides_and_minimizes_odd_offset():
    responses = np.array([0, 0, 1, 0, 1], dtype=np.int8)
    margins = np.array([0.0, -0.1, 0.12, -0.4, 0.5])
    rows = selector_metrics(responses, margins, 0.5, (2,), np.random.default_rng(3))
    balanced = next(row for row in rows if row["selector"] == "balanced_margin")
    assert balanced["n_spatial"] == 3
    assert balanced["below_fraction"] == 2 / 3
    assert balanced["balance_shortfall"] == 1 / 3
    margin_bias = balanced["margin_bias"]
    assert isinstance(margin_bias, float)
    assert np.isclose(margin_bias, (-0.1 + 0.12 - 0.4) / 3)
    assert balanced["margin_radius"] == 0.4


def test_balanced_selector_reports_one_sided_shortfall():
    responses = np.array([0, 1, 1, 1], dtype=np.int8)
    margins = np.array([0.0, 0.1, 0.2, 0.3])
    rows = selector_metrics(responses, margins, 0.5, (2,), np.random.default_rng(3))
    balanced = next(row for row in rows if row["selector"] == "balanced_margin")
    assert balanced["below_fraction"] == 0.0
    assert balanced["balance_shortfall"] == 1.0
