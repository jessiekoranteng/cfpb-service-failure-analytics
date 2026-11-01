import pandas as pd

from src.pipeline.clean import clean
from src.pipeline.integrate import MODEL_FEATURES, add_region, company_lag4, dashboard_feed, load_state_region, split_outputs
from src.pipeline.transform import company_key, transform


def test_labels_follow_dictionary(raw):
    d = transform(clean(raw)[0])
    open_ = d["Company response to consumer"] == "In progress"
    assert d.loc[open_, "monetary_relief"].isna().all()
    relief = d["Company response to consumer"] == "Closed with monetary relief"
    assert (d.loc[relief, "monetary_relief"] == 1).all()
    assert d.loc[~open_ & ~relief, "monetary_relief"].eq(0).all()
    assert (d["untimely"] == (d["Timely response?"] == "No")).all()


def test_week_start_is_monday_and_tags(raw):
    d = transform(clean(raw)[0])
    assert (d["week_start"].dt.dayofweek == 0).all()
    both = d["Tags"] == "Older American, Servicemember"
    assert (d.loc[both, "tag_group"] == "Both").all()
    assert (d.loc[d["Tags"].isna(), "tag_group"] == "None recorded").all()
    assert (d.loc[d["Tags"].isna(), "tag_recorded"] == 0).all()


def test_company_key_normalises():
    s = pd.Series(["Acme, Inc.", "ACME INC", " acme  inc "])
    assert company_key(s).nunique() == 1


def test_region_join_and_unmapped_become_other(raw, ref):
    d = transform(clean(raw)[0])
    d.loc[0, "State"] = "ZZ"
    out = add_region(d, ref)
    assert out.loc[0, "region"] == "Other" and len(out) == len(d)
    assert set(out["region"]) <= {"Northeast", "Midwest", "South", "West", "Other"}


def test_reference_table_has_no_duplicates(ref):
    assert ref["state"].is_unique and len(ref) >= 55


def test_lag4_never_includes_current_week():
    d = pd.DataFrame({"company_key": ["A"] * 6, "week_start": pd.to_datetime(
        ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-02-05"])})
    lag = company_lag4(d)
    assert lag.iloc[0] == 0 and lag.iloc[1] == 0          # first week: no history
    assert lag.iloc[2] == 2                               # prior week had 2
    assert lag.iloc[5] == 3                               # 01-08, 01-15, 01-22 each had 1; 01-29 had none


def test_split_outputs_keeps_protected_fields_out_of_features(curated):
    curated = curated.assign(complaint_key=curated["Complaint ID"])
    analysis, audit = split_outputs(curated)
    assert not {"tag_group", "tag_older", "tag_servicemember", "State", "Complaint ID"} & set(analysis.columns)
    assert {"tag_group", "tag_older"} <= set(audit.columns)
    assert set(MODEL_FEATURES) <= set(analysis.columns)


def test_dashboard_feed_runs_on_small_data(curated):
    feed = dashboard_feed(curated, min_total=1, last_weeks=520)
    assert {"week_start", "n", "expected", "z", "surge"} <= set(feed.columns) or feed.empty


def test_state_region_loader_rejects_duplicates(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("state,region\nNY,Northeast\nNY,South\n")
    try:
        load_state_region(p)
    except ValueError:
        return
    raise AssertionError("duplicate state codes must be rejected")
