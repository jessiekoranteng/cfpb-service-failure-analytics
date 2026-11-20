"""Model monitoring dashboard (Final Project).

    streamlit run monitoring_app/dashboard.py

Reads one generated file, monitoring_app/assets/monitoring_data.json (written by `python -m src.monitoring.job`).
It holds aggregates only: no complaint-level data. The numbers are a REPLAY of the 2025-10 to 2026-06 test window scored
month by month with the deployed model, not live traffic; the live API exposes /metrics for Prometheus (see docker-compose.yml).
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
DATA = HERE / "assets" / "monitoring_data.json"
HTML = HERE / "assets" / "evidently_drift_report.html"
RETRAIN = HERE / "assets" / "retrain_decision.json"
CYAN, VIOLET, AMBER, RED, GREY, INK = "#06B6D4", "#8B5CF6", "#F59E0B", "#EF4444", "#94A3B8", "#E6EAF7"
BADGE = {"ok": "🟢 ok", "warn": "🟡 warn", "hard": "🔴 hard", "n/a": "⚪ n/a"}
NAMES = {"roc_auc": "ROC-AUC", "pr_auc_lift": "PR-AUC lift", "recall_at_flag": "Recall at the review line", "ece": "Calibration error (ECE)",
         "flag_share": "Share flagged for review", "unseen_company_share": "Unseen-company share", "psi_company": "Company-mix PSI",
         "psi_score": "Score PSI", "region_selection_ratio": "Regional selection ratio", "f2_tpr_gap": "F2 TPR gap", "f2_fpr_gap": "F2 FPR gap"}
PLAYBOOK = {
    "roc_auc": "Hard: open an incident, freeze threshold changes, start the retraining review. Warn: ticket and re-check next month.",
    "pr_auc_lift": "Same as ROC-AUC. Check whether the base rate moved before blaming the model.",
    "recall_at_flag": "Check capacity and thresholds; a drop with stable AUC means the review line drifted, not the ranking.",
    "ece": "Recalibrate on the latest closed window (isotonic) through the retraining gates; never hand-edit scores.",
    "flag_share": "Reviewers are over- or under-loaded. Check the score PSI and the region thresholds.",
    "unseen_company_share": "New companies are scored on weak signal (ROC-AUC 0.626). Route to manual review and add them at the next retrain.",
    "psi_company": "A different set of companies is filing. Compare with the Evidently report, then decide on retraining.",
    "psi_score": "Score distribution moved. Look for an input change before assuming the model changed.",
    "region_selection_ratio": "Check the smallest region's volume first (ratio over under 300 complaints is noisy), then the regional thresholds.",
    "f2_tpr_gap": "Charter F2. A confirmed breach (interval wholly above 0.05) goes to the AI Ethics and Data Governance Review Panel.",
    "f2_fpr_gap": "Charter F2. Same escalation as the TPR gap.",
}

st.set_page_config(page_title="Model monitoring", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load() -> dict:
    return json.loads(DATA.read_text())


def style(fig, title, height=330, yfmt=None):
    fig.update_layout(title={"text": title, "x": 0, "font": {"size": 15, "color": INK}}, height=height, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin={"l": 10, "r": 10, "t": 50, "b": 10},
                      font={"color": INK}, legend={"orientation": "h", "y": -0.25})
    fig.update_yaxes(tickformat=yfmt, gridcolor="rgba(148,163,184,.15)")
    fig.update_xaxes(gridcolor="rgba(148,163,184,.15)")
    return fig


def trend(m: pd.DataFrame, col: str, rules: dict, title: str, yfmt=None, fixed=None):
    fig = go.Figure(go.Scatter(x=m["month"], y=m[col], mode="lines+markers", line={"color": CYAN, "width": 3}, name=NAMES[col]))
    r = rules[col]
    for val, nm, colr in ((r["warn"], "warn", AMBER), (r["hard"], "hard", RED)):
        if isinstance(val, (int, float)):
            fig.add_hline(y=val, line={"color": colr, "dash": "dash", "width": 1.5}, annotation_text=f"{nm} {val}", annotation_position="top left")
    if fixed:
        fig.update_yaxes(range=fixed)
    st.plotly_chart(style(fig, title, yfmt=yfmt), width="stretch")


if not DATA.exists():
    st.error("monitoring_data.json is missing. Run `python -m src.monitoring.job` first.")
    st.stop()

d = load()
m = pd.DataFrame(d["monthly"])
rules = d["rules"]
worst = {"ok": 0, "warn": 1, "hard": 2}
open_alerts = [a for a in d["alerts"] if not a["noisy"]]
noisy_alerts = [a for a in d["alerts"] if a["noisy"]]
tags = d["fairness_tags"]["replay_window"]

st.title("Model monitoring: complaint service-failure risk")
st.warning(f"**This is a replay, not live traffic.** {d['basis']}. Generated {d['generated_at'][:10]}. No human review or analyst override is "
           f"observed here, so the override-rate signal is not computed. Decision support only: a trained reviewer decides every case.", icon="⚠️")

c = st.columns(5)
c[0].metric("ROC-AUC (replay)", f"{d['crosscheck']['replay_roc_auc']:.3f}", help="Module 4 limit: 0.70")
c[1].metric("Calibration error", f"{d['crosscheck']['replay_ece']:.4f}", help="Warn above 0.03, hard above 0.05")
c[2].metric("Flagged for review", f"{100 * d['replay']['flag_share']:.1f}%", help="Expected band 15% to 25%")
dc = d["drift"]["drifted_columns"]
c[3].metric("Drifted columns (Evidently)", f"{int(dc['count'])} of {len(d['drift']['columns'])}")
c[4].metric("Open alerts", f"{len(open_alerts)}", delta=f"{len(noisy_alerts)} on thin regional cells", delta_color="off")

t_status, t_perf, t_drift, t_fair, t_alerts, t_live = st.tabs(["Status", "Performance", "Drift", "Fairness", "Alerts and actions", "Live API"])

with t_status:
    st.subheader("Latest month and trend")
    last = d["monthly"][-1]
    rows = [{"Signal": NAMES[k], "Latest": f"{last[k]:.4f}" if last.get(k) is not None else "n/a", "Status": BADGE[lv],
             "Months in breach": sum(1 for r in d["monthly"] if r["levels"].get(k) in ("warn", "hard"))} for k, lv in last["levels"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if RETRAIN.exists():
        rd = json.loads(RETRAIN.read_text())
        st.subheader("Retraining decision")
        st.write(f"**{rd['decision']}**. {rd['summary']}")
    st.caption(f"Reference window {d['reference']['window']} (n={d['reference']['n']:,}, relief rate {100 * d['reference']['relief_rate']:.1f}%). "
               f"Replay window {d['replay']['window']} (n={d['replay']['n']:,}, relief rate {100 * d['replay']['relief_rate']:.1f}%).")

with t_perf:
    a, b = st.columns(2)
    with a:
        trend(m, "roc_auc", rules, "ROC-AUC by month (limit 0.70)", fixed=[0.6, 0.9])
        trend(m, "recall_at_flag", rules, "Recall at the review line (limit 0.45)", yfmt=".0%", fixed=[0.3, 0.7])
    with b:
        trend(m, "ece", rules, "Calibration error by month (warn 0.03, hard 0.05)", fixed=[0, 0.06])
        trend(m, "pr_auc_lift", rules, "PR-AUC lift by month (limit 2.0)", fixed=[1.5, 4])
    st.caption("Each month is scored on complaints that were closed by the data snapshot; the last months have fewer closed cases, so they are less certain.")

with t_drift:
    a, b = st.columns(2)
    with a:
        trend(m, "psi_company", rules, "Company-mix PSI vs the reference window", fixed=[0, 0.35])
    with b:
        trend(m, "psi_score", rules, "Score PSI vs the reference window", fixed=[0, 0.35])
    st.subheader("Evidently data-drift report")
    ev = pd.DataFrame(d["drift"]["columns"])
    ev["Status"] = ev["drifted"].map({True: "🔴 drifted", False: "🟢 stable"})
    st.dataframe(ev[["column", "method", "value", "threshold", "Status"]].rename(columns={"value": "distance"}), hide_index=True, width="stretch")
    st.caption(f"Evidently {d['drift']['evidently_version']}, {d['drift']['sample_per_window']:,} complaints sampled from each window. "
               "Company-volume history is expected to drift (it counts recent filings); company identity drifting is the signal that matters.")
    if HTML.exists():
        st.download_button("Download the full Evidently report (HTML)", HTML.read_bytes(), file_name="evidently_drift_report.html", mime="text/html")
        if st.toggle("Show the full report here (about 4 MB, slow to load)"):
            st.components.v1.html(HTML.read_text(encoding="utf-8"), height=900, scrolling=True)

with t_fair:
    st.subheader("Charter F2: true- and false-positive-rate gaps by protected tag (limit 0.05)")
    fr = []
    for k, label in (("older_american", "Older American"), ("servicemember", "Servicemember")):
        g = tags[k]
        fr.append({"Group": label, "TPR gap": g["tpr_gap"], "TPR 95% interval": f"{g['tpr_ci95'][0]:.3f} to {g['tpr_ci95'][1]:.3f}",
                   "FPR gap": g["fpr_gap"], "FPR 95% interval": f"{g['fpr_ci95'][0]:.3f} to {g['fpr_ci95'][1]:.3f}",
                   "Group size": g["n_group"], "Confirmed breach": "yes" if g["confirmed_breach"] else "no"})
    st.dataframe(pd.DataFrame(fr).style.format({"TPR gap": "{:.3f}", "FPR gap": "{:.3f}", "Group size": "{:,}"}), hide_index=True, width="stretch")
    st.caption("A gap is a confirmed breach only when its whole 95% interval is above 0.05. The Older American TPR interval reaches 0.075, so the "
               "limit is not confirmed as met either way (same finding as Module 4). Tags cover about 15% of complaints; 'rest' includes untagged.")
    q = []
    for qk, qv in d["fairness_tags"]["by_quarter"].items():
        for k in ("older_american", "servicemember"):
            q.append({"Quarter": qk, "Group": k.replace("_", " ").title(), "TPR gap": qv[k]["tpr_gap"], "FPR gap": qv[k]["fpr_gap"]})
    qd = pd.DataFrame(q)
    fig = go.Figure()
    for grp, colr in (("Older American", VIOLET), ("Servicemember", CYAN)):
        s = qd[qd["Group"] == grp]
        fig.add_bar(x=s["Quarter"], y=s["TPR gap"], name=f"{grp} TPR gap", marker_color=colr)
    fig.add_hline(y=0.05, line={"color": AMBER, "dash": "dash"}, annotation_text="charter limit 0.05")
    st.plotly_chart(style(fig, "TPR gap by quarter"), width="stretch")
    st.subheader("By product, company size and region")
    st.caption("Module 1 asked for these cuts; they were never produced before this job. Cells with fewer than 10 positives or negatives are suppressed. "
               "Spread is the highest minus the lowest rate across groups. No limit was pre-registered for these cuts beyond Module 1's TPR 0.05, so they are reported, not graded.")
    for key, label in (("product", "Product"), ("company_size_tier", "Company size tier (training-window complaint counts)"), ("region", "Region")):
        cut = d["fairness_cuts"][key]
        df = pd.DataFrame(cut["groups"])[["group", "n", "positives", "flag_rate", "tpr", "fpr", "suppressed"]]
        st.markdown(f"**{label}**: TPR spread {cut['tpr_spread']:.3f}, FPR spread {cut['fpr_spread']:.3f}, selection ratio {cut['selection_ratio']:.3f}")
        st.dataframe(df.style.format({"n": "{:,}", "flag_rate": "{:.1%}", "tpr": "{:.1%}", "fpr": "{:.1%}"}, na_rep="suppressed"), hide_index=True, width="stretch")

with t_alerts:
    st.subheader("Alerts raised on the replay window")
    if d["alerts"]:
        ad = pd.DataFrame(d["alerts"])
        ad["Signal"] = ad["signal"].map(NAMES)
        ad["Level"] = ad["level"].map(BADGE)
        ad["Note"] = ad["noisy"].map({True: "smallest region under 300 complaints that month: noisy", False: ""})
        st.dataframe(ad[["month", "Signal", "value", "Level", "Note", "rule"]].rename(columns={"month": "Month", "value": "Value", "rule": "Rule and source"}),
                     hide_index=True, width="stretch")
    else:
        st.success("No alerts raised.")
    st.subheader("What happens when a signal fires")
    st.dataframe(pd.DataFrame([{"Signal": NAMES[k], "Warn": str(rules[k]["warn"]), "Hard": str(rules[k]["hard"]), "Action": v} for k, v in PLAYBOOK.items()]),
                 hide_index=True, width="stretch")
    st.caption("Warn raises a ticket. Hard pages the model owner and opens an incident (docs/final/incident_response_plan.md).")

with t_live:
    lv = d["live_api"]
    if lv["available"]:
        st.metric("Requests scored", f"{lv['n']:,}")
        st.write(f"From {lv['first']} to {lv['last']}: {100 * lv['flag_share']:.1f}% flagged, {100 * lv['unseen_share']:.1f}% with a value not seen in training.")
    else:
        st.info("This public snapshot is not connected to the live API. Run `docker compose up` to start the API, Prometheus and Grafana: "
                "Prometheus scrapes `/metrics` (request counts, flagged share, unseen-value share, latency) and the API's explanation store "
                "feeds this job via `--store`.")
st.caption("Source: CFPB Consumer Complaint Database (public). Aggregates only. BAN6800 Final Project, Jessie Kojo Koranteng.")
