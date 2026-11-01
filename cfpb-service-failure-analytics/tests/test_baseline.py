import pandas as pd

from src.baseline.weekly import surge_flags, weekly_counts


def _frame(weekly_values, company="A", issue="X"):
    rows = []
    start = pd.Timestamp("2024-01-01")  # a Monday
    for i, n in enumerate(weekly_values):
        rows += [{"Company": company, "Issue": issue, "Date received": start + pd.Timedelta(weeks=i, days=2)}] * n
    return pd.DataFrame(rows)


def test_spike_after_stable_period_is_flagged():
    vals = [20, 22, 19, 21, 20, 23, 18, 21, 20, 22, 19, 21, 90]
    flags = surge_flags(weekly_counts(_frame(vals)), window=12)
    last = flags.sort_values("week_start").iloc[-1]
    assert bool(last["surge"]) and last["z"] > 3


def test_stable_series_has_no_surge():
    vals = [20, 22, 19, 21, 20, 23, 18, 21, 20, 22, 19, 21, 22, 20]
    assert not surge_flags(weekly_counts(_frame(vals)), window=12)["surge"].any()


def test_current_week_not_in_its_own_baseline():
    vals = [10] * 12 + [50]
    last = surge_flags(weekly_counts(_frame(vals)), window=12, min_count=5).sort_values("week_start").iloc[-1]
    assert last["expected"] == 10


def test_groups_are_independent():
    a = _frame([20, 21, 19, 20, 22, 20, 21, 19, 20, 21, 20, 22, 95], company="A")
    b = _frame([20, 21, 19, 20, 22, 20, 21, 19, 20, 21, 20, 22, 21], company="B")
    flags = surge_flags(weekly_counts(pd.concat([a, b])), window=12)
    assert set(flags[flags["surge"]]["Company"]) == {"A"}
