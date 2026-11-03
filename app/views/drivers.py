"""What drives predictions: SHAP in business language, with the four worked complaints."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import AMBER, CYAN, banner, load_data, style

d = load_data()
st.title("What drives a score?")
st.markdown("The model looks at nine facts a handler records when a complaint arrives. **SHAP** measures how much each fact moved the score "
            "across 6,000 test complaints. Bigger bar = that fact matters more to the model's ranking.")
banner()

s = pd.DataFrame(d["shap_global"]).query("share >= 0.05").sort_values("share")
fig = go.Figure(go.Bar(x=s["share"], y=s["label"], orientation="h", marker_color=[AMBER if lbl == "Company" else CYAN for lbl in s["label"]],
                       text=[f"{v:.1f}%" for v in s["share"]], textposition="outside", hovertemplate="%{y}: %{x:.1f}% of total influence<extra></extra>"))
style(fig, "Share of the model's influence by fact (all complaints, test window)", height=420, xtitle="Share of total influence (%)", legend=False)
fig.update_xaxes(range=[0, 75], ticksuffix="%")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### In plain language")
    st.markdown("- **Who the complaint is about** carries about two-thirds of the influence. Some companies resolve a larger share of complaints with money back than others.\n"
                "- **What went wrong** (sub-issue and issue) is next: a fee problem behaves differently from a card problem.\n"
                "- **How busy the company has been** and **the month** add a little.\n"
                "- **Region and channel** barely matter to the score. Region is used only to set the review line (see *Ethical Compliance*).")
with c2:
    st.markdown("#### What this does *not* say")
    st.markdown("- It does not say a company *caused* the outcome. It says complaints about that company have ended in relief more often in the past.\n"
                "- Because company dominates, the model is better at *ranking known companies* than judging a new one. An unseen company drops ROC-AUC from 0.79 to 0.63.\n"
                "- A high score is a reason to look, never a reason to decide.")

st.subheader("Four worked complaints")
st.caption("Each card shows the three facts that moved that complaint's score most. LIME, a second explanation method, agreed with SHAP on the top two drivers for complaint A. For complaint B it ranked company volume first, so treat the order of smaller drivers as indicative.")
cols = st.columns(2)
for i, e in enumerate(d["examples"]):
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"**{e['id']}. {e['title']}**")
            st.markdown(f"Score **{e['score']:.1%}** · review line for {e['inputs']['region']}: {e['region_cut']:.1%} · "
                        + ("**flagged**" if e["flagged_regional"] else "not flagged") + f" · actual outcome: {'relief' if e['relief_happened'] else 'no relief'}")
            rows = e["top_contributions"] or []
            for r in rows:
                arrow = "pushed the score **up**" if r["shap"] > 0 else "pushed the score **down**"
                st.markdown(f"- {r['feature']} = *{str(r['value'])[:48]}* {arrow}")
            if not rows:
                st.markdown(f"- {e['inputs']['company']} · {e['inputs']['sub_issue']} · {e['inputs']['channel']}")
st.caption("B and D sit exactly on the South review line (16.3%): the same complaint is flagged at the regional line and not at the single 17.5% line. Scores that land on a boundary are a coin toss, which is why a person reviews the near-misses.")
