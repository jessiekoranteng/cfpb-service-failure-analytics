"""Shared helpers for the dashboard: data loading, the scorer, plain-language text and a consistent chart style."""
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run app/dashboard.py` puts app/ on the path, not the repo root
    sys.path.insert(0, str(ROOT))

from src.modeling import config as C
from src.modeling.scoring import Scorer

ASSETS = Path(__file__).parent / "assets" / "dashboard_data.json"
CYAN, VIOLET, AMBER, GREY, INK = "#06B6D4", "#8B5CF6", "#F59E0B", "#94A3B8", "#E6EAF7"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LIMITS = {**C.CRITERIA["fairness"], "charter_f2": 0.05}


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    return json.loads(ASSETS.read_text())


@st.cache_resource(show_spinner="Loading the two Module 4 models...")
def load_scorer() -> Scorer:
    return Scorer.from_dir(C.MODELS)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def style(fig: go.Figure, title: str, height=360, yfmt=None, xtitle=None, ytitle=None, legend=True) -> go.Figure:
    """One look for every chart: dark transparent canvas, direct titles, zero-based value axes are set by callers."""
    fig.update_layout(title={"text": title, "x": 0, "font": {"size": 15, "color": INK}}, height=height, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin={"l": 10, "r": 10, "t": 50, "b": 10},
                      font={"color": INK}, showlegend=legend, legend={"orientation": "h", "y": -0.2})
    fig.update_xaxes(title=xtitle, gridcolor="rgba(148,163,184,.15)")
    fig.update_yaxes(title=ytitle, tickformat=yfmt, gridcolor="rgba(148,163,184,.15)")
    return fig


def caption(text: str):
    st.caption(text)


def banner():
    st.info("**Decision support only.** Scores describe patterns in past public complaints. They are not causes, not findings about any "
            "person, and never a decision: a trained reviewer decides every case.", icon="ℹ️")
