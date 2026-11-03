"""Ethical Compliance Dashboard: fairness metrics (Fairlearn), the bias that was found, how it was addressed, and what is still open."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import AMBER, CYAN, GREY, LIMITS, VIOLET, banner, load_data, pct, style

from src.dashboard import fair

d = load_data()
F, meta = d["fairness"], d["meta"]
st.title("Ethical Compliance")
st.markdown("Does the model treat groups of complainants evenly? This page measures that with **Fairlearn** on the untouched test window "
            "(94,305 complaints) and reports misses as plainly as passes. The model never sees region, age or military tags as inputs; "
            "the tags are used only afterwards, to audit.")
banner()

crit = pd.DataFrame(d["criteria"])
fair_rows = crit[crit["criterion"].str.contains("Older American|Servicemember|Charter")]
n_met, n_all = int(fair_rows["met"].sum()), len(fair_rows)
pre = fair_rows[~fair_rows["criterion"].str.contains("Charter")]
chart = fair_rows[fair_rows["criterion"].str.contains("Charter")]
confirmed = int((chart["met"] & ~chart["note"].str.contains("not confirmed")).sum())
t1, t2, t3, t4 = st.tabs(["Scorecard", "Explore groups", "Bias found and how it was addressed", "Can protected tags be guessed?"])

with t1:
    c = st.columns(3)
    c[0].metric("Fairness checks met", f"{n_met} of {n_all}", help="Counts a check as met when the observed value is inside its limit, before looking at confidence intervals.")
    c[1].metric("Pre-registered limits", f"{int(pre['met'].sum())} of {len(pre)}", help="Disparate impact >= 0.80, equalized-odds and demographic-parity differences <= 0.10, for Older American and Servicemember complainants. Set before any model was trained.")
    c[2].metric("Stricter Charter checks, confirmed", f"{confirmed} of {len(chart)}", help="F2 gaps <= 0.05 and F3 proxy test. Only Servicemember F2 is confirmed with its confidence interval.")
    show = fair_rows.assign(Status=lambda t: t["met"].map({True: "✅ met", False: "❌ not met"}), Observed=lambda t: t["observed"].map(lambda v: f"{v:.3f}"))
    st.dataframe(show[["criterion", "target", "Observed", "Status", "note"]].rename(columns={"criterion": "Check", "target": "Limit", "note": "Note"}),
                 hide_index=True, use_container_width=True,
                 column_config={"Check": st.column_config.TextColumn(width="large"), "Note": st.column_config.TextColumn(width="medium")})
    st.markdown("**Plain reading.** The model clears every limit we promised before testing. Against the stricter Charter we set ourselves, "
                "the Older American gap (0.047) is under 0.05 only on the point estimate, with its interval reaching 0.075; the four-group comparison "
                "fails (0.061); and the model's inputs let a classifier guess the Older American or Servicemember tag better than chance (see the last tab). "
                "These are open items, not footnotes.")

with t2:
    dim = st.radio("Compare groups by", list(F["dims"]), format_func=lambda k: F["dims"][k], horizontal=True)
    rule = st.radio("Review rule", ["Deployed rule: a review line per region", "One review line for everyone"], horizontal=True)
    if rule.startswith("One"):
        thr = st.slider("Review line (flag when the relief score reaches this)", 0.05, 0.40, float(meta["global_threshold"]), step=0.0025, format="%.4f",
                        help="Moves in steps of 0.0025 so every figure is exact.")
        counts = fair.counts_from_hist(F["hist"][dim], thr, meta["bin_width"])
        st.caption(f"Flagging every complaint scoring at least {thr:.4f}.")
    else:
        counts = F["deployed_counts"][dim]
        st.caption("Review lines: " + ", ".join(f"{k} {v:.3f}" for k, v in meta["region_thresholds"].items()) + ". Complaints are flagged against their own region's line.")
    s = fair.summarise(counts)
    c = st.columns(5)
    for col, (name, val, lim, ok) in zip(c, fair.verdicts(s, LIMITS)):
        col.metric(name, f"{val:.3f}", delta=("within " if ok else "outside ") + lim, delta_color="normal" if ok else "inverse")
    st.caption(f"Flagged overall: {pct(s['flagged_share'])} of complaints (capacity target 20%). Disparate impact ratio = lowest group flag rate ÷ highest.")
    t = s["table"]
    fig = go.Figure()
    for col_name, label, color in (("flagged", "Flagged", VIOLET), ("tpr", "Relief cases caught (recall)", CYAN), ("fpr", "False alarms", AMBER)):
        fig.add_trace(go.Bar(x=t["Group"], y=t[col_name], name=label, marker_color=color, hovertemplate="%{x}: %{y:.1%}<extra>" + label + "</extra>"))
    style(fig, "Same rule, three rates per group", height=380, ytitle="Share", )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, max(0.7, float(t[["flagged", "tpr", "fpr"]].max().max()) * 1.1)], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    out = t.assign(complaints=t["complaints"].map("{:,}".format), actual_relief_rate=t["actual_relief_rate"].map(pct), flagged=t["flagged"].map(pct),
                   tpr=t["tpr"].map(pct), fpr=t["fpr"].map(pct)).rename(columns={"complaints": "Complaints (n)", "actual_relief_rate": "Real relief rate", "flagged": "Flagged",
                                                                                 "tpr": "Relief cases caught", "fpr": "False alarms"})
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.caption("Group sizes matter: 'Both' (Older American and Servicemember) has only 1,349 complaints, so its rates are noisy. Real relief rates differ between groups, "
               "so equal flag rates would not be fair on their own; the page therefore also shows who gets caught and who gets a false alarm.")

with t3:
    rb, ra, to = F["region_before"], F["region_after"], F["threshold_optimizer"]
    st.markdown("**Found.** With one review line for everyone, Northeast complaints were flagged far more often than Other-region complaints, and recall differed by region.")
    reg = pd.DataFrame(rb["table"]).merge(pd.DataFrame(ra["table"]), on="region", suffixes=("_before", "_after"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=reg["region"], y=reg["tpr_before"], name="Before: one line", marker_color=GREY))
    fig.add_trace(go.Bar(x=reg["region"], y=reg["tpr_after"], name="After: a line per region", marker_color=CYAN))
    style(fig, "Share of real relief cases the model catches, by region", height=360, ytitle="Recall")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 0.7], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    m = st.columns(4)
    m[0].metric("Recall gap across regions", f"{ra['tpr_gap']:.3f}", delta=f"{ra['tpr_gap'] - rb['tpr_gap']:+.3f} vs before", delta_color="inverse")
    m[1].metric("False-alarm gap", f"{ra['fpr_gap']:.3f}", delta=f"{ra['fpr_gap'] - rb['fpr_gap']:+.3f} vs before", delta_color="inverse")
    m[2].metric("Flag-rate ratio (low ÷ high)", f"{ra['selection_ratio']:.2f}", delta=f"{ra['selection_ratio'] - rb['selection_ratio_min_max']:+.2f} vs before")
    m[3].metric("Flagged share", pct(ra["flag_share"]), delta=f"{100 * (ra['flag_share'] - to['before']['flag_share']):+.1f} pts vs before", delta_color="off")
    st.markdown(f"**Addressed.** Fairlearn's ThresholdOptimizer was tried first and rejected: it equalised error rates by flagging {pct(to['after']['flag_share'])} of complaints "
                f"(up from {pct(to['before']['flag_share'])}), far over the 20% review capacity. Instead, a separate review line per region was set on the validation window so that each region's recall is "
                f"close to the same target. Cost: precision fell from {pct(to['before']['precision'])} to {pct(ra['precision'])}. "
                "**Still open:** a recall gap of 0.093 remains, just under the 0.10 limit, and part of the original gap reflects real base-rate differences (Northeast 13.1% relief vs Other 7.6%).")

with t4:
    f3 = F["charter"]["f3"]
    st.markdown("**The proxy test.** If a second model can guess a complainant's tag (Older American or Servicemember) from the same nine inputs the risk model uses, "
                "those inputs may carry the tag indirectly. A score of 0.50 means no better than chance.")
    rows = [("Older American", f3["older_american"]), ("Servicemember", f3["servicemember"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[1]["auc_recover_tag_from_features"] for r in rows], name="Guess the tag from all nine inputs", marker_color=AMBER,
                         error_y={"type": "data", "symmetric": False, "array": [r[1]["ci95"][1] - r[1]["auc_recover_tag_from_features"] for r in rows],
                                  "arrayminus": [r[1]["auc_recover_tag_from_features"] - r[1]["ci95"][0] for r in rows]}))
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[1]["auc_relief_score_vs_tag"] for r in rows], name="Guess the tag from the relief score alone", marker_color=CYAN))
    style(fig, "How well can the tag be guessed? (0.50 = chance)", height=360, ytitle="AUC")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0.4, 0.8])
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY, annotation_text="chance")
    st.plotly_chart(fig, use_container_width=True)
    drv = F["charter"]["f3_drivers"]
    only = pd.DataFrame({"Input": list(drv["older_american"]), "Older American": [v["only"] for v in drv["older_american"].values()],
                         "Servicemember": [v["only"] for v in drv["servicemember"].values()]}).sort_values("Older American", ascending=False)
    only["Input"] = only["Input"].map(d["plain_labels"]).fillna(only["Input"])
    st.markdown("**Where the signal sits.** Guessing the tag from one input alone (AUC; 0.50 = chance):")
    st.dataframe(only.assign(**{c: only[c].map("{:.3f}".format) for c in ("Older American", "Servicemember")}), hide_index=True, use_container_width=True)
    st.markdown("**Reading.** Company is the main carrier: some companies receive proportionally more complaints from older or servicemember customers. "
                "The relief *score* itself separates tagged from untagged only slightly (0.56 and 0.52). The Charter objective F3 is therefore **not met**, "
                "and the model is documented as such rather than adjusted away.")
