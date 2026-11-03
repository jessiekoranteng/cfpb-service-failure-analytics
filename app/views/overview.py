"""Overview: the business problem, what the model buys, and how stable it is."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import AMBER, CYAN, GREY, VIOLET, banner, load_data, pct, style

d = load_data()
perf, cap, dep = d["performance"], d["performance"]["capacity"], d["performance"]["deployed"]

st.title("Which complaints deserve a closer look first?")
st.markdown(
    "Banks and payment companies receive far more service-failure complaints than reviewers can read closely. This dashboard shows what a "
    "risk model built on **535,601 public CFPB complaints** (checking/savings and money-transfer products, 2020 to mid-2026) can and cannot do "
    "to help a reviewer decide *where to look first*. It supports goals G1 (relief risk), G2 (missed deadlines), G4 (fair prioritisation) and G5 "
    "(better triage) of the project Vision Document.")
banner()

c = st.columns(4)
c[0].metric("Cases ending in relief", pct(perf["relief"]["prevalence"]), help="Share of complaints in the untouched test window (Oct 2025 to Jun 2026) where the company paid or refunded.")
c[1].metric("Relief found in top 20%", pct(cap["recall_top20"]), delta=f"{cap['recall_top20'] / 0.20:.1f}x random reading", help="If reviewers read only the 20% of complaints the model scores highest.")
c[2].metric("Ranking ability (ROC-AUC)", f"{perf['relief']['roc_auc']:.2f}", help="0.50 is a coin flip, 1.00 is perfect. 95% interval 0.786 to 0.795.")
c[3].metric("Flags that were right", pct(dep["precision"]), help=f"Of the {pct(dep['flag_share'])} of complaints flagged under the regional review lines, this share ended in relief (the base rate is {pct(perf['relief']['prevalence'])}).")

left, right = st.columns(2)
with left:
    g = pd.DataFrame(d["gains"])
    x = [0] + [r * 10 for r in g["decile"]]
    y = [0] + [r * 100 for r in g["cumulative_recall"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=x, name="Reading at random", mode="lines", line={"color": GREY, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=x, y=y, name="Reading in model order", mode="lines+markers", line={"color": CYAN, "width": 3},
                             hovertemplate="Read top %{x:.0f}% -> find %{y:.1f}% of relief cases<extra></extra>"))
    style(fig, "What reading the highest-scored complaints first buys", xtitle="Share of complaints read (highest score first)", ytitle="Share of relief cases found")
    ticks = list(range(0, 101, 20))
    fig.update_xaxes(range=[0, 100], tickvals=ticks, ticktext=[f"{t}%" for t in ticks])
    fig.update_yaxes(range=[0, 100], tickvals=ticks, ticktext=[f"{t}%" for t in ticks])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Test window, 94,305 complaints. At 20% of complaints read, the model finds about half of the relief cases; random reading would find 20%. About half are still missed.")
with right:
    m = pd.DataFrame(d["monthly"])
    fig = go.Figure(go.Bar(x=m["month"], y=m["complaints"], marker_color=VIOLET, hovertemplate="%{x}: %{y:,} complaints<extra></extra>"))
    style(fig, "Complaints received per month (volume surges are hard to predict)", ytitle="Complaints", legend=False)
    fig.update_yaxes(rangemode="tozero", tickformat=",")
    fig.add_annotation(x="2025-01", y=float(m.set_index("month").loc["2025-01", "complaints"]), text="Jan 2025 surge", showarrow=True, ay=-30, font={"color": AMBER})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("One company (Block Inc) accounted for 46% of that month's complaints. The model was trained with the spike on purpose, and retraining without it changed ROC-AUC from 0.791 to 0.792.")

left, right = st.columns(2)
with left:
    q = pd.DataFrame(d["quarterly"])
    fig = go.Figure(go.Bar(x=q["quarter"], y=q["roc_auc"], marker_color=CYAN, text=[f"{v:.3f}" for v in q["roc_auc"]], textposition="outside",
                           hovertemplate="%{x}: ROC-AUC %{y:.3f}<extra></extra>"))
    style(fig, "Ranking quality by quarter in the test window", ytitle="ROC-AUC (0.5 = coin flip)", legend=False)
    fig.update_yaxes(range=[0.5, 1.0])
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Axis starts at 0.5 because 0.5 is chance, not zero. The slow decline (0.802 to 0.778) is the early sign of drift the monitoring plan watches.")
with right:
    cal = pd.DataFrame(d["calibration_bins"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], name="Perfect calibration", mode="lines", line={"color": GREY, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=cal["mean_score"], y=cal["observed_rate"], name="Model", mode="lines+markers", line={"color": AMBER, "width": 3},
                             hovertemplate="Model said %{x:.1%}, actual %{y:.1%}<extra></extra>"))
    style(fig, "When the model says 'x% chance of relief', how often is it right?", xtitle="Score (mean per tenth of complaints)", ytitle="Share that actually ended in relief")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Scores track reality closely (average calibration error {perf['calibration']['ece_calibrated']:.3f}), so a score near 20% really means about one in five.")

v = perf["vs_logreg"]
st.warning(
    f"**An honest limit.** A simple logistic-regression baseline scores slightly higher than the chosen model on average precision "
    f"(difference {v['difference']:+.3f}, 95% interval {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f}). The gradient-boosted model was chosen before the test "
    "window was opened, and the result is reported rather than hidden. See *Limits and transparency* for all 17 pre-registered checks (13 met).")

st.subheader("Progress against the Module 1 goals")
st.caption("Targets are quoted from the Vision Document. Misses are shown as misses.")
gl = pd.DataFrame(d["goals"]).rename(columns={"id": "Goal", "goal": "What", "target": "Target", "observed": "Observed", "status": "Status", "note": "Note"})
st.dataframe(gl.drop(columns=["Note"]), hide_index=True, use_container_width=True,
             column_config={"What": st.column_config.TextColumn(width="medium"), "Target": st.column_config.TextColumn(width="large"), "Observed": st.column_config.TextColumn(width="large")})
for g in d["goals"]:
    if g["note"]:
        st.caption(f"**{g['id']}**: {g['note']}")
