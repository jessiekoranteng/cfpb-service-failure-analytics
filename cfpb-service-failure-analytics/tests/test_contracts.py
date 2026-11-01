from pathlib import Path

import pandas as pd
import pytest

from src.quality.contracts import assert_contract, check_contract

SAMPLE = Path(__file__).parent / "fixtures" / "ccdb_sample.csv"


@pytest.fixture
def sample():
    return pd.read_csv(SAMPLE, dtype=str)


def test_real_sample_passes_gate(sample):
    assert check_contract(sample) == []


def test_prohibited_zip_column_fails(sample):
    sample["ZIP code"] = "12345"
    assert any("prohibited" in v for v in check_contract(sample))


def test_duplicate_ids_fail(sample):
    dup = pd.concat([sample, sample.head(1)], ignore_index=True)
    assert any("duplicates" in v for v in check_contract(dup))


def test_unexpected_timely_value_fails(sample):
    sample.loc[sample.index[0], "Timely response?"] = "Maybe"
    with pytest.raises(ValueError):
        assert_contract(sample)


def test_out_of_scope_product_fails(sample):
    sample.loc[sample.index[0], "Product"] = "Mortgage"
    assert any("out of scope" in v for v in check_contract(sample))


def test_missing_column_reported(sample):
    assert check_contract(sample.drop(columns=["Issue"])) == ["missing columns: ['Issue']"]
