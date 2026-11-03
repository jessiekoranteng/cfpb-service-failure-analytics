"""One scoring implementation shared by the FastAPI service and the Streamlit dashboard.

The API and the dashboard must give the same answer for the same complaint, so the logic lives here once:
build the input frame, score with both calibrated models, apply the region's review threshold, and list the three
strongest SHAP reasons in plain language. Nothing here reads data files; it only needs the two model bundles.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling import config as C

PLAIN = {"company_key": "Company", "Issue": "Issue", "Sub-issue": "Sub-issue", "Sub-product": "Sub-product", "Product": "Product",
         "Submitted via": "Channel", "region": "Region", "company_volume_lag4": "Company volume (prior 4 wks)",
         "log_company_volume_lag4": "Company volume, log", "month": "Month received", "subissue_unknown": "Sub-issue not given",
         "company_volume_lag4_missing": "Volume history missing"}


def load_bundle(target, model_dir=None):
    p = Path(model_dir or os.environ.get("MODEL_DIR", C.MODELS)) / f"risk_{target}.joblib"
    if not p.exists():
        raise FileNotFoundError(p)
    return joblib.load(p)


class Scorer:
    """Holds the two bundles and the SHAP explainer; `score` takes a plain dict of the nine intake fields."""

    def __init__(self, relief, untimely, explain=True):
        self.relief, self.untimely = relief, untimely
        self.explainer = None
        if explain and relief["algorithm"] != "logreg":
            import shap  # imported lazily: scoring without reasons does not need it
            self.explainer = shap.TreeExplainer(relief["base"][-1])

    @classmethod
    def from_dir(cls, model_dir=None, explain=True):
        return cls(load_bundle(C.PRIMARY, model_dir), load_bundle(C.SECONDARY, model_dir), explain=explain)

    @staticmethod
    def frame(c: dict) -> pd.DataFrame:
        vol = c.get("company_volume_lag4")
        return pd.DataFrame([{"Product": c["product"], "Sub-product": c.get("sub_product", "Unknown"), "Issue": c["issue"],
                              "Sub-issue": c.get("sub_issue", "Unknown"), "Submitted via": c["channel"], "region": c["region"],
                              "company_key": c["company"].upper(), "company_volume_lag4": np.nan if vol is None else vol,
                              "month": c["month"]}])[C.RAW_INPUT]

    def relief_many(self, cs: list) -> pd.DataFrame:
        """Relief score and review flag for many complaints at once (no reasons): used for what-if sweeps."""
        X = pd.concat([self.frame(c) for c in cs], ignore_index=True)
        p = self.relief["calibrated"].predict_proba(X)[:, 1]
        cut = np.array([self.cut_for(c["region"]) for c in cs])
        return pd.DataFrame({"relief_probability": p, "review_threshold": cut, "flag_for_review": p >= cut})

    def cut_for(self, region):
        return self.relief.get("region_thresholds", {}).get(region, self.relief["review_threshold"])

    def unseen(self, X):
        return [f"{col}={X[col].iloc[0]}" for col in C.CATEGORICAL
                if str(X[col].iloc[0]) not in set(self.relief["base"]["coder"].categories_[col])]

    def score(self, c: dict, reasons=True) -> dict:
        X = self.frame(c)
        p_relief = float(self.relief["calibrated"].predict_proba(X)[0, 1])
        p_late = float(self.untimely["calibrated"].predict_proba(X)[0, 1])
        cut = self.cut_for(c["region"])
        unseen = self.unseen(X)
        out = {"relief_probability": round(p_relief, 4), "untimely_probability": round(p_late, 4), "flag_for_review": p_relief >= cut,
               "review_threshold": round(cut, 4), "reasons": [], "unseen_values": unseen}
        if reasons and self.explainer is not None:
            Xt = self.relief["base"][:-1].transform(X)
            sv = self.explainer.shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) else sv
            sv = np.asarray(sv)
            sv = sv[0] if sv.ndim == 2 else sv[0, :, 1]
            for i in np.argsort(-np.abs(sv))[:3]:
                f = C.FEATURES[i]
                raw = X[f].iloc[0] if f in X.columns else Xt[f].iloc[0]  # show what the caller sent, not the coded value
                if pd.isna(raw):
                    shown = "not provided"
                elif f in C.CATEGORICAL and any(u.startswith(f + "=") for u in unseen):
                    shown = f"{raw} (not seen in training)"
                else:
                    shown = str(raw)
                out["reasons"].append({"feature": PLAIN.get(f, f), "value": shown, "effect": round(float(sv[i]), 4)})
        return out

    def model_versions(self):
        return {"relief": self.relief["algorithm"] + "@" + self.relief["trained_at"],
                "untimely": self.untimely["algorithm"] + "@" + self.untimely["trained_at"]}
