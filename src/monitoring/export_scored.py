"""Stage 1 of monitoring (model environment): score the reference and replay windows with the DEPLOYED model.

Writes one parquet with the intake fields, the calibrated score, the deployed review flag (region thresholds), the label
and the restricted audit tags. It is complaint-level (pseudonymous), so it stays in data/processed (git-ignored).
Stage 2 (src/monitoring/job.py) reads it in a separate environment, because Evidently needs pandas 2.

Reference window = the validation window (2025-04 .. 2025-09), the last data the calibration was fitted on.
Replay window = the test window (2025-10 .. 2026-06): complaints the model never saw, replayed month by month as if live.
This is a replay, not live traffic; the report says so wherever these numbers appear.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import config as C
from src.modeling import data as D
from src.modeling.charter import deployed_flags
from src.monitoring.metrics import size_tiers


def build(processed=C.PROCESSED, model_dir=C.MODELS) -> pd.DataFrame:
    a, g = D.load(processed)
    parts = {k: D.with_groups(v, g) for k, v in D.split(D.prepare(a), C.PRIMARY).items()}
    bundle = joblib.load(Path(model_dir) / f"risk_{C.PRIMARY}.joblib")
    seen = set(bundle["base"]["coder"].categories_["company_key"])
    tiers_src = parts["train"]["company_key"]
    out = []
    for name in ("valid", "test"):
        d = parts[name].copy()
        d["relief_probability"] = bundle["calibrated"].predict_proba(d[C.RAW_INPUT])[:, 1]
        d["flag_for_review"] = deployed_flags(bundle, d, d["relief_probability"])
        d["y"] = d[C.PRIMARY].astype(int)
        d["window"] = "reference" if name == "valid" else "replay"
        d["month_received"] = d["week_start"].dt.strftime("%Y-%m")
        d["company_seen"] = d["company_key"].isin(seen)
        d["size_tier"] = size_tiers(tiers_src, d["company_key"]).to_numpy()
        out.append(d)
    keep = C.RAW_INPUT + ["relief_probability", "flag_for_review", "y", "window", "month_received", "company_seen", "size_tier",
                          "tag_older", "tag_servicemember", "tag_group", "week_start"]
    return pd.concat(out, ignore_index=True)[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.PROCESSED / "monitoring_scored.parquet"))
    args = ap.parse_args()
    df = build()
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df):,} rows -> {args.out}")
    print(df.groupby("window").agg(n=("y", "size"), relief=("y", "mean"), flagged=("flag_for_review", "mean")).round(4))


if __name__ == "__main__":
    main()
