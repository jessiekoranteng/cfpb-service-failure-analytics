import pandas as pd

from src.modeling import config as C
from src.modeling.ablation import VARIANTS, neutralise


def frame():
    return pd.DataFrame({"company_key": ["A", "A", "B"], "company_volume_lag4": [1.0, 2.0, 3.0], "Sub-issue": ["x", "y", "y"], "region": ["S", "W", "S"]})


def test_neutralise_removes_all_information_from_the_named_inputs_only():
    d = frame()
    out = neutralise(d, ["company_key", "company_volume_lag4"], d)
    assert out["company_key"].nunique() == 1 and out["company_volume_lag4"].isna().all()
    assert list(out["Sub-issue"]) == list(d["Sub-issue"]) and list(out["region"]) == list(d["region"])
    assert d["company_key"].nunique() == 2          # the input frame is not modified


def test_variants_only_touch_model_inputs_and_are_nested():
    assert set().union(*VARIANTS.values()) <= set(C.RAW_INPUT)
    assert VARIANTS["full"] == [] and set(VARIANTS["no_company"]) < set(VARIANTS["no_company_or_subissue"])
    assert not {"tag_older", "tag_servicemember"} & {c for cols in VARIANTS.values() for c in cols}
