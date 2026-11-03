import sys
from pathlib import Path

import pandas as pd
import pytest

APP = str(Path(__file__).resolve().parents[1] / "app")
if APP not in sys.path:  # dashboard views import their shared helpers as `from common import ...`
    sys.path.insert(0, APP)

SAMPLE = Path(__file__).parent / "fixtures" / "ccdb_sample.csv"


@pytest.fixture(scope="session")
def raw():
    """300 real CFPB rows (stratified sample), strings as the ingest stage emits them."""
    return pd.read_csv(SAMPLE, dtype=str)


@pytest.fixture()
def ref():
    from src.pipeline.integrate import load_state_region
    return load_state_region(Path(__file__).resolve().parents[1] / "reference" / "state_region.csv")


@pytest.fixture()
def curated(raw, ref):
    from src.pipeline.clean import clean
    from src.pipeline.integrate import integrate
    from src.pipeline.transform import transform
    d, _ = clean(raw)
    return integrate(transform(d), ref)


@pytest.fixture(autouse=True)
def _pseudonym_key(monkeypatch):
    monkeypatch.setenv("PIPELINE_PSEUDONYM_KEY", "test-key")
