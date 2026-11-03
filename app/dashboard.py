"""Service-failure analytics: stakeholder dashboard (Module 5).

    streamlit run app/dashboard.py

Pages: Overview, What drives predictions, Try a complaint (what-if), Ethical Compliance, Limits and transparency.
The dashboard reads one aggregated file (app/assets/dashboard_data.json) and the two Module 4 model files. It holds no
complaint-level data, no names and no narratives.
"""
import streamlit as st

st.set_page_config(page_title="Complaint service-failure analytics", page_icon="🛡️", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon="📊", default=True, url_path="overview"),
    st.Page("views/drivers.py", title="What drives predictions", icon="🔎", url_path="drivers"),
    st.Page("views/whatif.py", title="Try a complaint (what-if)", icon="🧪", url_path="what-if"),
    st.Page("views/ethics.py", title="Ethical Compliance", icon="⚖️", url_path="ethics"),
    st.Page("views/limits.py", title="Limits and transparency", icon="🧭", url_path="limits"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.caption("BAN6800 capstone · Module 5\n\nJessie Kojo Koranteng\n\nData: CFPB Consumer Complaint Database (public).")
nav.run()
