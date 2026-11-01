import pandas as pd

from src.pipeline.clean import clean


def test_clean_counts_and_fills(raw):
    d = raw.copy()
    d.loc[0, "Issue"] = "  Fraud or scam  "
    d.loc[1, "Sub-issue"] = ""
    d = pd.concat([d, d.head(2)], ignore_index=True)              # 2 exact duplicates
    dup_id = d.head(1).copy()
    dup_id["Issue"] = "Other"                                      # same Complaint ID, different row
    d = pd.concat([d, dup_id], ignore_index=True)
    out, rep = clean(d)
    assert rep["exact_duplicates_dropped"] == 2
    assert rep["id_duplicates_dropped"] == 1
    assert out["Complaint ID"].is_unique
    assert out.loc[0, "Issue"] == "Fraud or scam"
    assert out["Sub-issue"].isna().sum() == 0 and out["State"].isna().sum() == 0
    assert rep["rows_out"] == len(out) == len(raw)


def test_clean_drops_bad_and_future_dates(raw):
    d = raw.head(5).copy()
    d.loc[0, "Date received"] = "not-a-date"
    d.loc[1, "Date received"] = "2031-01-01"
    out, rep = clean(d, snapshot_date="2026-11-01")
    assert rep["bad_or_future_dates_dropped"] == 2
    assert len(out) == 3


def test_tags_stay_missing(raw):
    out, rep = clean(raw)
    assert out["Tags"].isna().sum() == rep["tags_missing"] > 0


def test_clean_does_not_mutate_input(raw):
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_blank_issue_is_filled_unknown(raw):
    d = raw.copy()
    d.loc[d.index[0], "Issue"] = None
    out, rep = clean(d)
    assert rep["nulls_filled_unknown"]["Issue"] == 1
    assert out["Issue"].notna().all()
