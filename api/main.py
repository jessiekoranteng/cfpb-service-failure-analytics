"""FastAPI service for the complaint risk models.

    uvicorn api.main:app --reload          # then open http://127.0.0.1:8000/docs

POST /predict scores one complaint at intake. The service never receives names, account numbers, narratives or
protected attributes: only the eight categorical/numeric intake fields below.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.modeling import config as C
from src.modeling.explain import PLAIN

STATE: dict = {}


class Complaint(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "product": "Checking or savings account", "sub_product": "Checking account", "issue": "Managing an account",
        "sub_issue": "Fee problem", "channel": "Web", "region": "South", "company": "BANK OF AMERICA NATIONAL ASSOCIATION",
        "company_volume_lag4": 900, "month": 5}})
    product: str = Field(min_length=1, max_length=120)
    sub_product: str = Field("Unknown", max_length=120)
    issue: str = Field(min_length=1, max_length=120)
    sub_issue: str = Field("Unknown", max_length=160)
    channel: str = Field(min_length=1, max_length=40, description="Submitted via: Web, Phone, Referral, Postal mail, Fax, Email")
    region: str = Field(min_length=1, max_length=20, description="Northeast, Midwest, South, West or Other")
    company: str = Field(min_length=1, max_length=120, description="Company key as in the analysis table (upper case)")
    company_volume_lag4: float | None = Field(None, ge=0, description="Complaints about this company in the previous 4 weeks")
    month: int = Field(ge=1, le=12)


class Reason(BaseModel):
    feature: str
    value: str
    effect: float


class Prediction(BaseModel):
    relief_probability: float
    untimely_probability: float
    flag_for_review: bool
    review_threshold: float
    reasons: list[Reason]
    unseen_values: list[str]
    model_versions: dict[str, str]
    note: str = "Decision support only. Scores are associations in historical complaints, not causes; a person decides."


def _load(target):
    p = Path(os.environ.get("MODEL_DIR", C.MODELS)) / f"risk_{target}.joblib"
    if not p.exists():
        raise FileNotFoundError(p)
    return joblib.load(p)


@asynccontextmanager
async def lifespan(app):
    STATE["relief"] = _load(C.PRIMARY)
    STATE["untimely"] = _load(C.SECONDARY)
    STATE["explainer"] = shap.TreeExplainer(STATE["relief"]["base"][-1]) if STATE["relief"]["algorithm"] != "logreg" else None
    yield
    STATE.clear()


app = FastAPI(title="CFPB complaint risk API", version="1.0.0", lifespan=lifespan,
              description="Scores a complaint at intake: probability of monetary relief and of an untimely company response.")


def _frame(c: Complaint) -> pd.DataFrame:
    return pd.DataFrame([{"Product": c.product, "Sub-product": c.sub_product, "Issue": c.issue, "Sub-issue": c.sub_issue,
                          "Submitted via": c.channel, "region": c.region, "company_key": c.company.upper(),
                          "company_volume_lag4": np.nan if c.company_volume_lag4 is None else c.company_volume_lag4, "month": c.month}])[C.RAW_INPUT]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(k for k in STATE if k in ("relief", "untimely"))}


@app.get("/model-info")
def model_info():
    return {k: {"algorithm": STATE[k]["algorithm"], "trained_at": STATE[k]["trained_at"], "review_threshold": STATE[k]["review_threshold"]}
            for k in ("relief", "untimely")} | {"features": C.RAW_INPUT, "review_capacity": C.REVIEW_CAPACITY}


@app.post("/predict", response_model=Prediction)
def predict(c: Complaint):
    if "relief" not in STATE:
        raise HTTPException(503, "models not loaded")
    X = _frame(c)
    relief, late = STATE["relief"], STATE["untimely"]
    p_relief = float(relief["calibrated"].predict_proba(X)[0, 1])
    p_late = float(late["calibrated"].predict_proba(X)[0, 1])
    cut = relief.get("region_thresholds", {}).get(c.region, relief["review_threshold"])  # equal-opportunity cut-off per region
    coder = relief["base"][:-1]
    Xt = coder.transform(X)
    unseen = [f"{col}={X[col].iloc[0]}" for col in C.CATEGORICAL if str(X[col].iloc[0]) not in set(relief["base"]["coder"].categories_[col])]
    reasons = []
    if STATE["explainer"] is not None:
        sv = STATE["explainer"].shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.asarray(sv)[0] if np.asarray(sv).ndim == 2 else np.asarray(sv)[0, :, 1]
        for i in np.argsort(-np.abs(sv))[:3]:
            f = C.FEATURES[i]
            raw = X[f].iloc[0] if f in X.columns else Xt[f].iloc[0]  # show what the caller sent, not the coded value
            if pd.isna(raw):
                shown = "not provided"
            elif f in C.CATEGORICAL and any(u.startswith(f + "=") for u in unseen):
                shown = f"{raw} (not seen in training)"
            else:
                shown = str(raw)
            reasons.append(Reason(feature=PLAIN.get(f, f), value=shown, effect=round(float(sv[i]), 4)))
    return Prediction(relief_probability=round(p_relief, 4), untimely_probability=round(p_late, 4),
                      flag_for_review=p_relief >= cut, review_threshold=round(cut, 4),
                      reasons=reasons, unseen_values=unseen,
                      model_versions={"relief": relief["algorithm"] + "@" + relief["trained_at"], "untimely": late["algorithm"] + "@" + late["trained_at"]})
