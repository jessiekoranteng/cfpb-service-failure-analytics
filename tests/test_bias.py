import pandas as pd
import pytest

from src.pipeline.bias import assert_publishable, bias_suite


def test_bias_suite_structure_on_sample(curated):
    rep = bias_suite(curated, min_block=1, min_warn=30)
    assert set(rep["checks"]) == {"coverage", "representation_drift", "differential_missingness", "outcome_disparity", "concentration"}
    assert rep["overall"] in {"PASS", "WARN"}
    labs = rep["checks"]["outcome_disparity"]["labels"]
    assert set(labs) == {"monetary_relief", "untimely"}
    assert rep["n_records"] == len(curated)


def test_blocks_when_a_group_is_too_small(curated):
    rep = bias_suite(curated, min_block=10_000, min_warn=20_000)
    assert rep["overall"] == "BLOCK"
    with pytest.raises(ValueError):
        assert_publishable(rep)


def test_detects_planted_representation_drift_and_missingness():
    n = 400
    d = pd.DataFrame({
        "tag_group": ["Older American"] * 200 + ["None recorded"] * 200,
        "region": ["West"] * n, "Submitted via": ["Web"] * n, "Product": ["P"] * n,
        "year": [2023] * 100 + [2024] * 100 + [2023] * 100 + [2024] * 100,
        "tag_recorded": [1] * 200 + [0] * 200,
        "company_key": ["A"] * 300 + ["B"] * 100,
        "monetary_relief": pd.array([1] * 100 + [0] * 100 + [0] * 200, dtype="Int8"),
        "untimely": [0] * n,
    })
    # shift Older American toward 2024 only
    d.loc[:199, "year"] = [2024] * 160 + [2023] * 40
    rep = bias_suite(d, min_block=1, min_warn=1)
    assert rep["checks"]["representation_drift"]["status"] == "WARN"
    assert rep["checks"]["concentration"]["status"] == "WARN"          # top company holds 75%
    assert rep["checks"]["outcome_disparity"]["status"] == "WARN"      # relief 50% vs 0% across tag groups


def test_tiny_channel_warns_but_does_not_block(curated):
    from src.pipeline.bias import _coverage
    d = curated.copy()
    d.loc[d.index[:3], "Submitted via"] = "Email"
    rep = _coverage(d, min_block=10, min_warn=20)
    email = next(g for g in rep["groups"] if g["group"] == "Submitted via" and g["value"] == "Email")
    assert email["severity"] == "WARN"
