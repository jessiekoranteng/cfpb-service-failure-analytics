"""Limits and transparency: what the model can and cannot do, how fragile it is, and every pre-registered check."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import AMBER, CYAN, banner, load_data, style

d = load_data()
st.title("Limits and transparency")
banner()

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("#### What the model can do")
        st.markdown("- Rank new complaints so the top 20% hold about half of the eventual monetary-relief cases (about 2.6x a random pick).\n"
                    "- Give a calibrated chance of relief, and a second, advisory chance of a late company response.\n"
                    "- Name the three facts that moved each score, in plain words.\n"
                    "- Flag using a review line per region so that no region is systematically under-reviewed.")
with c2:
    with st.container(border=True):
        st.markdown("#### What the model cannot do")
        st.markdown("- **Decide anything.** It issues no verdict, deadline or denial; a trained reviewer does.\n"
                    "- **Explain causes.** Scores are associations in past complaints.\n"
                    "- **Judge a new company.** An unseen company drops ROC-AUC from 0.79 to 0.63.\n"
                    "- **Transfer to Ghana unchanged.** It learned from US consumer complaints, so local validation is required (risk R6).\n"
                    "- **Beat a simple baseline.** Logistic regression scores slightly higher on average precision.")

st.subheader("How fragile is it?")
p = pd.DataFrame(d["perturbations"]).iloc[1:]
fig = go.Figure(go.Bar(x=p["flag_flip_rate"], y=p["scenario"], orientation="h", marker_color=[AMBER if v > 0.05 else CYAN for v in p["flag_flip_rate"]],
                       text=[f"{v:.1%}" for v in p["flag_flip_rate"]], textposition="outside", hovertemplate="%{y}: %{x:.1%} of flags change<extra></extra>"))
style(fig, "Share of review flags that change when an input is altered or missing", height=360, xtitle="Flags that flip", legend=False)
fig.update_xaxes(tickformat=".0%", range=[0, 0.22])
st.plotly_chart(fig, use_container_width=True)
st.caption("Small numeric nudges barely matter. Missing company volume (11.8%), a missing sub-issue (8.2%) and an unseen company (18.0%) change many flags: data completeness at intake is part of the model's safety.")

st.subheader("Model drift to watch")
q = pd.DataFrame(d["quarterly"])
st.markdown(f"Quarterly ranking quality slipped from {q['roc_auc'].iloc[0]:.3f} to {q['roc_auc'].iloc[-1]:.3f} across the test window while the relief rate rose from {q['prevalence'].iloc[0]:.1%} to {q['prevalence'].iloc[-1]:.1%}. "
            "The late-response model is less stable still (ROC-AUC 0.892 on test against 0.759 on validation), which is why it is advisory.")

st.subheader("Transparency statement")
st.markdown(
    "- **Data:** public CFPB Consumer Complaint Database, 535,601 in-scope records. No names, account numbers or narratives enter the model or this page.\n"
    "- **Models:** gradient-boosted trees (LightGBM), probability-calibrated; selected on validation, scored once on the final window.\n"
    "- **Protected tags** (Older American, Servicemember) are never model inputs. They are used only to audit.\n"
    "- **Human in the loop:** the API returns a score, a flag and reasons. It never returns a decision or touches a statutory deadline (Charter F4).\n"
    "- **Everything is logged:** 51 MLflow runs, a registered champion, and a Model Card and Fairness Metrics Report in the repository.")

st.subheader("All 17 pre-registered checks")
crit = pd.DataFrame(d["criteria"])
st.caption(f"{int(crit['met'].sum())} of {len(crit)} met. The misses are shown, not removed.")
crit["Status"] = crit["met"].map({True: "✅ met", False: "❌ not met"})
crit["Observed"] = crit["observed"].map(lambda v: f"{v:.3f}")
st.dataframe(crit[["criterion", "target", "Observed", "Status", "note"]].rename(columns={"criterion": "Check", "target": "Target", "note": "Note"}), hide_index=True, use_container_width=True)
