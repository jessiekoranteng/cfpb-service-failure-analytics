"""Try a complaint: score a complaint, change the inputs, and see how the score and the review flag respond."""
import plotly.graph_objects as go
import streamlit as st
from common import AMBER, CYAN, GREY, MONTHS, banner, load_data, load_scorer, pct, style

d = load_data()
V = d["vocab"]
sc = load_scorer()
ex = {e["id"]: e for e in d["examples"]}
COMP = {c["name"]: c for c in V["companies"]}
DEFAULT = {"product": "Checking or savings account", "sub_product": "Checking account", "issue": "Managing an account", "sub_issue": "Fee problem",
           "channel": "Web", "region": "South", "company": "BANK OF AMERICA NATIONAL ASSOCIATION", "month": 5}


def _opts(prefix, k):
    p = st.session_state[f"{prefix}product"]
    if k == "sub_product":
        return V["sub_product"][p]
    if k == "issue":
        return V["issue"][p]
    return V["sub_issue"].get(f"{p}||{st.session_state[f'{prefix}issue']}", ["Unknown"])


def _init(prefix, base):
    for k, v in base.items():
        st.session_state.setdefault(f"{prefix}{k}", v)
    st.session_state.setdefault(f"{prefix}vol_missing", False)
    st.session_state.setdefault(f"{prefix}volume", float(COMP[base["company"]]["typical_volume_4wk"] or 0))


def _preset(k):
    e = ex[k]["inputs"]
    for pre in ("a_", "b_"):
        for f in DEFAULT:
            st.session_state[f"{pre}{f}"] = e[f]
        st.session_state[f"{pre}vol_missing"] = e["company_volume_lag4"] is None
        st.session_state[f"{pre}volume"] = float(e["company_volume_lag4"] or 0)
    st.session_state["sweep_note"] = ""


def _fix(prefix):
    """Keep dependent drop-downs valid after the parent changed."""
    for k in ("sub_product", "issue"):
        o = _opts(prefix, k)
        if st.session_state[f"{prefix}{k}"] not in o:
            st.session_state[f"{prefix}{k}"] = o[0]
    o = _opts(prefix, "sub_issue")
    if st.session_state[f"{prefix}sub_issue"] not in o:
        st.session_state[f"{prefix}sub_issue"] = o[0]


def form(prefix, label):
    _init(prefix, DEFAULT)
    _fix(prefix)
    st.markdown(f"**{label}**")
    st.selectbox("Product", list(V["issue"]), key=f"{prefix}product")
    st.selectbox("Sub-product", _opts(prefix, "sub_product"), key=f"{prefix}sub_product")
    st.selectbox("Issue", _opts(prefix, "issue"), key=f"{prefix}issue")
    _fix(prefix)
    st.selectbox("Sub-issue", _opts(prefix, "sub_issue"), key=f"{prefix}sub_issue")
    c1, c2 = st.columns(2)
    c1.selectbox("Channel", V["channel"], key=f"{prefix}channel")
    c2.selectbox("Region", V["region"], key=f"{prefix}region")
    st.selectbox("Company", list(COMP), key=f"{prefix}company", help="Type to search. Companies are listed from most to fewest complaints.")
    c3, c4 = st.columns(2)
    c3.selectbox("Month received", range(1, 13), format_func=lambda m: MONTHS[m - 1], key=f"{prefix}month")
    typical = COMP[st.session_state[f"{prefix}company"]]["typical_volume_4wk"]
    c4.number_input("Company complaints, prior 4 weeks", min_value=0.0, step=10.0, key=f"{prefix}volume", disabled=st.session_state[f"{prefix}vol_missing"],
                    help=f"Typical for this company in the test window: {typical:,.0f}" if typical is not None else "No typical value for this company.")
    st.checkbox("Company volume not available", key=f"{prefix}vol_missing")
    return collect(prefix)


def collect(prefix):
    g = lambda k: st.session_state[f"{prefix}{k}"]
    return {"product": g("product"), "sub_product": g("sub_product"), "issue": g("issue"), "sub_issue": g("sub_issue"), "channel": g("channel"),
            "region": g("region"), "company": g("company"), "month": int(g("month")),
            "company_volume_lag4": None if g("vol_missing") else float(g("volume"))}


st.title("Try a complaint")
st.markdown("Pick the facts a handler would record at intake and watch the score move. The scoring code is the **same module the Module 4 API uses**, "
            "so this page and the API return identical numbers.")
banner()
st.caption("Start from a worked complaint:")
pc = st.columns(5)
for col, k in zip(pc, "ABCD"):
    col.button(f"{k} · {ex[k]['short']}", on_click=_preset, args=(k,), use_container_width=True)
pc[4].button("Reset", on_click=lambda: [st.session_state.update({f"{p}{f}": v for p in ("a_", "b_") for f, v in DEFAULT.items()})], use_container_width=True)

st.divider()
colA, colB = st.columns(2)
with colA:
    A = form("a_", "Scenario A: the complaint as received")
with colB:
    st.markdown("**Scenario B: what if one or more facts were different?**")
    st.caption("Scenario B starts as a copy of A. Change any field below. Everything else stays equal to A.")
    for k in (*DEFAULT, "vol_missing", "volume"):  # B starts equal to A; the user then overrides what they want to test
        st.session_state.setdefault(f"b_{k}", st.session_state[f"a_{k}"])
    if st.button("Copy A into B", use_container_width=True):
        for k in DEFAULT:
            st.session_state[f"b_{k}"] = st.session_state[f"a_{k}"]
        st.session_state["b_vol_missing"] = st.session_state["a_vol_missing"]
        st.session_state["b_volume"] = st.session_state["a_volume"]
        st.rerun()
    B = form("b_", "Scenario B")

ra, rb = sc.score(A), sc.score(B)
st.divider()


def card(col, name, r):
    with col:
        with st.container(border=True):
            st.markdown(f"##### {name}")
            m = st.columns(3)
            m[0].metric("Relief chance", pct(r["relief_probability"]))
            m[1].metric("Late-response chance", pct(r["untimely_probability"]), help="Advisory only: this second model is less stable over time.")
            m[2].metric("Review line", pct(r["review_threshold"]))
            if r["flag_for_review"]:
                st.success("**Flag for closer review.** The score reaches this region's review line.")
            else:
                st.info("**Not flagged.** The score is under this region's review line. A person can still pick it up.")
            if r["unseen_values"]:
                st.warning("The model never saw: " + ", ".join(r["unseen_values"]) + ". Treat this score with extra caution.")
            for x in r["reasons"]:
                st.markdown(f"- {x['feature']}: *{x['value']}* {'pushed the score **up**' if x['effect'] > 0 else 'pushed the score **down**'}")


c1, c2 = st.columns(2)
card(c1, "Scenario A", ra)
card(c2, "Scenario B", rb)
delta = rb["relief_probability"] - ra["relief_probability"]
changed = [k for k in DEFAULT if A[k] != B[k]] + (["company_volume_lag4"] if A["company_volume_lag4"] != B["company_volume_lag4"] else [])
if changed:
    st.markdown(f"**Effect of the change** ({', '.join(changed)}): relief score {pct(ra['relief_probability'])} → {pct(rb['relief_probability'])} "
                f"(**{100 * delta:+.1f} points**); review flag {'unchanged' if ra['flag_for_review'] == rb['flag_for_review'] else 'CHANGED'}.")
    st.caption("This is an association learned from past complaints. It does not mean that changing the field would change what the company decides.")
else:
    st.caption("A and B are identical. Change a field in B to see its effect.")

st.subheader("Sweep one fact at a time")
st.caption("Scenario A with a single field varied across every value the model knows for it. The dashed line is the review line for A's region.")
which = st.radio("Vary", ["Sub-issue", "Channel", "Month", "Region", "Company volume"], horizontal=True)
if which == "Sub-issue":
    vals = _opts("a_", "sub_issue"); mk = lambda v: {**A, "sub_issue": v}; labels = vals
elif which == "Channel":
    vals = V["channel"]; mk = lambda v: {**A, "channel": v}; labels = vals
elif which == "Month":
    vals = list(range(1, 13)); mk = lambda v: {**A, "month": v}; labels = MONTHS
elif which == "Region":
    vals = V["region"]; mk = lambda v: {**A, "region": v}; labels = vals
else:
    vals = [0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]; mk = lambda v: {**A, "company_volume_lag4": float(v)}; labels = [f"{v:,}" for v in vals]
sw = sc.relief_many([mk(v) for v in vals])
sw["label"] = labels
fig = go.Figure(go.Bar(x=sw["label"], y=sw["relief_probability"], marker_color=[CYAN if f else GREY for f in sw["flag_for_review"]],
                       hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
fig.add_hline(y=ra["review_threshold"], line_dash="dash", line_color=AMBER, annotation_text=f"review line {ra['review_threshold']:.1%}")
style(fig, f"Relief score for complaint A as {which.lower()} varies", height=380, ytitle="Chance of monetary relief", legend=False)
fig.update_yaxes(rangemode="tozero", tickformat=".0%")
if which == "Sub-issue":
    fig.update_xaxes(tickangle=-35)
st.plotly_chart(fig, use_container_width=True)
st.caption("Bars in cyan are at or above the review line; grey are below. Bars start at zero.")
