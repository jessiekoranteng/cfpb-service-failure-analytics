"""FastAPI service for the complaint risk models.

    uvicorn api.main:app --reload          # then open http://127.0.0.1:8000/docs

POST /predict scores one complaint at intake. The service never receives names, account numbers, narratives or
protected attributes: only the eight categorical/numeric intake fields below.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.modeling import config as C
from src.modeling.scoring import Scorer

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


@asynccontextmanager
async def lifespan(app):
    STATE["scorer"] = Scorer.from_dir()
    yield
    STATE.clear()


app = FastAPI(title="CFPB complaint risk API", version="1.0.0", lifespan=lifespan,
              description="Scores a complaint at intake: probability of monetary relief and of an untimely company response.")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": ["relief", "untimely"] if "scorer" in STATE else []}


@app.get("/model-info")
def model_info():
    sc = STATE["scorer"]
    return {k: {"algorithm": b["algorithm"], "trained_at": b["trained_at"], "review_threshold": b["review_threshold"]}
            for k, b in (("relief", sc.relief), ("untimely", sc.untimely))} | {"features": C.RAW_INPUT, "review_capacity": C.REVIEW_CAPACITY}


@app.post("/predict", response_model=Prediction)
def predict(c: Complaint):
    if "scorer" not in STATE:
        raise HTTPException(503, "models not loaded")
    sc = STATE["scorer"]
    r = sc.score(c.model_dump())
    return Prediction(**r, model_versions=sc.model_versions())
