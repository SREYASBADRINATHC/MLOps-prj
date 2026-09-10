"""
CartSense — Enterprise Streamlit Dashboard
Drift-Resilient Hybrid Recommendation Engine for Electronics

7 Pages:
  A. Home / Overview
  B. Product Catalog
  C. Recommendations
  D. Cold-Start Demo
  E. MLOps Dashboard
  F. Drift Simulation
  G. Model / Experiment Info

All recommendation results come from the FastAPI backend.
No recommendation logic lives in this file.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 20

st.set_page_config(
    page_title="CartSense | AI Recommendation Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Premium dark theme */
.stApp { background-color: #0f1117; }
[data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2d3148; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2235 0%, #252a3d 100%);
    border: 1px solid #3d4266;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2rem; font-weight: 700; color: #7c8cf8; }
.metric-label { font-size: 0.85rem; color: #8890a8; margin-top: 0.3rem; }

/* Status badges */
.badge-healthy { background: #1a3a2a; color: #4ade80; border: 1px solid #22c55e;
    padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.8rem; }
.badge-error { background: #3a1a1a; color: #f87171; border: 1px solid #ef4444;
    padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.8rem; }
.badge-warn { background: #3a2e1a; color: #fbbf24; border: 1px solid #f59e0b;
    padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.8rem; }

/* Score bars */
.score-bar-container { margin: 0.3rem 0; }
.score-bar-label { font-size: 0.75rem; color: #8890a8; }
.score-bar-bg { background: #1e2235; border-radius: 4px; height: 8px; margin-top: 2px; }
.score-bar-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #7c8cf8, #a78bfa); }

/* Recommendation cards */
.rec-card {
    background: linear-gradient(135deg, #1e2235 0%, #1a2040 100%);
    border: 1px solid #3d4266; border-radius: 12px; padding: 1.2rem;
    margin-bottom: 1rem; transition: border-color 0.2s;
}
.rec-card:hover { border-color: #7c8cf8; }
.rec-title { font-size: 1.05rem; font-weight: 600; color: #e2e8f0; }
.rec-brand { font-size: 0.8rem; color: #8890a8; }

/* Architecture diagram */
.arch-box {
    background: #1e2235; border: 1px solid #3d4266; border-radius: 8px;
    padding: 0.6rem 1rem; text-align: center; color: #c4c9e2; font-size: 0.85rem;
}

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #7c8cf8;
    border-bottom: 2px solid #3d4266; padding-bottom: 0.5rem; margin-bottom: 1rem;
}

/* Cold-start highlight */
.cold-start-box {
    background: linear-gradient(135deg, #1a2f3a 0%, #152030 100%);
    border: 2px solid #0ea5e9; border-radius: 12px; padding: 1.2rem; margin: 1rem 0;
}
.cold-start-title { color: #38bdf8; font-size: 1.1rem; font-weight: 700; }

/* Drift alert */
.drift-alert {
    background: linear-gradient(135deg, #3a1a1a 0%, #2d1515 100%);
    border: 2px solid #ef4444; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
}
.drift-ok {
    background: linear-gradient(135deg, #1a3a1a 0%, #152d15 100%);
    border: 2px solid #22c55e; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── API Client ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def api_get(endpoint: str) -> Any:
    """GET request to FastAPI backend with caching."""
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"_error": f"Cannot connect to backend at {BACKEND_URL}"}
    except requests.HTTPError as e:
        return {"_error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"_error": str(e)}


def api_post(endpoint: str, payload: dict) -> Any:
    """POST request to FastAPI backend (no cache)."""
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"_error": f"Cannot connect to backend at {BACKEND_URL}"}
    except requests.HTTPError as e:
        return {"_error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"_error": str(e)}


def show_error(data: Any) -> bool:
    """If data contains an _error key, display it and return True."""
    if isinstance(data, dict) and "_error" in data:
        st.error(f"⚠️ {data['_error']}")
        return True
    return False


def score_bar_html(label: str, score: float, color: str = "#7c8cf8") -> str:
    pct = round(score * 100, 1)
    return f"""
    <div class="score-bar-container">
        <div class="score-bar-label">{label}: {pct:.1f}%</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background: {color};"></div>
        </div>
    </div>"""


# ── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛒 CartSense")
    st.markdown("*Hybrid Recommendation Engine*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Home / Overview",
            "📦 Product Catalog",
            "🎯 Recommendations",
            "❄️ Cold-Start Demo",
            "📊 MLOps Dashboard",
            "🌊 Drift Simulation",
            "🔬 Model / Experiments",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Quick health check in sidebar
    health = api_get("/api/health")
    if not show_error(health):
        db_ok = health.get("database", False)
        api_ok = True
        tfidf_ok = health.get("tfidf_loaded", False)
        als_ok = health.get("als_loaded", False)

        def badge(ok: bool, label: str) -> str:
            cls = "badge-healthy" if ok else "badge-error"
            icon = "✓" if ok else "✗"
            return f'<span class="{cls}">{icon} {label}</span>'

        st.markdown("**System Health**")
        st.markdown(badge(api_ok, "API"), unsafe_allow_html=True)
        st.markdown(badge(db_ok, "Database"), unsafe_allow_html=True)
        st.markdown(badge(tfidf_ok, "TF-IDF"), unsafe_allow_html=True)
        st.markdown(badge(als_ok, "ALS"), unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-error">✗ API Offline</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"Backend: `{BACKEND_URL}`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE A — HOME / OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home / Overview":
    st.markdown("# 🛒 CartSense")
    st.markdown(
        "### Drift-Resilient Hybrid Recommendation Engine for Electronics\n"
        "Combining **TF-IDF content-based** and **PySpark ALS collaborative filtering** "
        "to deliver recommendations resilient to item cold-start and concept drift."
    )

    # ── Health Status ─────────────────────────────────────────────────────────
    health = api_get("/api/health")
    model_status = api_get("/api/model/status")
    stats = api_get("/api/stats")

    col1, col2, col3, col4 = st.columns(4)
    if not show_error(health):
        with col1:
            st.metric("🗄️ Database", "✓ Healthy" if health.get("database") else "✗ Down")
        with col2:
            st.metric("🤖 TF-IDF Model", "✓ Loaded" if health.get("tfidf_loaded") else "✗ Missing")
        with col3:
            st.metric("⚡ ALS Model", "✓ Loaded" if health.get("als_loaded") else "✗ Missing")
        with col4:
            st.metric("📡 MLflow", "✓ Connected" if health.get("mlflow") else "⚠ Offline")

    st.markdown("---")

    # ── KPI Grid ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 System KPIs")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    if not show_error(stats):
        c1.metric("🖥️ Products", f"{stats.get('n_products', 0):,}")
        c2.metric("👤 Users", f"{stats.get('n_users', 0):,}")
        c3.metric("🔗 Interactions", f"{stats.get('n_interactions', 0):,}")
        c4.metric("❄️ Cold-Start Items", f"{stats.get('n_cold_start_products', 0):,}")
        c5.metric("💻 Laptops", f"{stats.get('n_laptops', 0):,}")
        c6.metric("📱 Smartphones", f"{stats.get('n_smartphones', 0):,}")

    if not show_error(model_status):
        st.markdown("### 🧠 Model Status")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model Version", model_status.get("model_version", "N/A"))
        mc2.metric("TF-IDF Vocabulary", f"{model_status.get('tfidf_vocabulary_size', 0):,}")
        mc3.metric("ALS Users", f"{model_status.get('als_n_users', 0):,}")
        mc4.metric("ALS Rank", model_status.get("als_rank", "N/A"))

        ts = model_status.get("training_timestamp", "N/A")
        if ts and ts != "N/A":
            st.caption(f"Last trained: {ts[:19].replace('T', ' ')} UTC")

    # ── Architecture Diagram ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏗️ Architecture")

    st.markdown("""
```
USER REQUEST
    │
    ▼
STREAMLIT UI  ──HTTP──▶  FASTAPI BACKEND
                                │
                         HYBRID ROUTER
                        ┌────────┴────────┐
                        │                 │
                  NEW ITEM            EXISTING ITEM
              interaction=0          interaction>0
                        │                 │
                   TF-IDF           ALS + TF-IDF
                   Cosine           Latent Factors
                   Spec Match       Spec Boost
                        │                 │
                        └────────┬────────┘
                                 │
                          HYBRID RANKER
                     FinalScore = α·ALS + (1-α)·TF-IDF + β·Spec
                                 │
                              TOP-K  ──▶  USER
                                 │
                USER INTERACTIONS  ──▶  POSTGRESQL
                                 │
                           EVIDENTLY AI
                                 │
                           DRIFT DETECTED?
                              YES  →  RETRAIN  →  MLFLOW  →  RELOAD
```
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE B — PRODUCT CATALOG
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Product Catalog":
    st.markdown("# 📦 Product Catalog")
    st.markdown("Browse the electronics catalog. **Cold-start products** (0 interactions) are highlighted.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        cat_filter = st.selectbox("Category", ["All", "laptop", "smartphone"])
    with col2:
        limit = st.selectbox("Items per page", [25, 50, 100, 200], index=1)
    with col3:
        search_term = st.text_input("🔍 Search (name, specs, brand)", placeholder="e.g. OLED 32GB Dell")

    # Fetch
    params = f"?limit={limit}"
    if cat_filter != "All":
        params += f"&category={cat_filter}"
    if search_term:
        params += f"&search={search_term}"

    products = api_get(f"/api/products{params}")
    if show_error(products):
        st.stop()

    df = pd.DataFrame(products)
    if df.empty:
        st.warning("No products found. Run `training/generate_data.py` to seed the catalog.")
        st.stop()

    st.markdown(f"**Showing {len(df)} products**")

    # Highlight cold-start rows
    cold_start_count = int((df["interaction_count"] == 0).sum())
    st.info(f"❄️ **{cold_start_count} cold-start products** (interaction_count = 0) in this view — these use TF-IDF only for recommendations.")

    display_cols = ["product_id", "name", "category", "brand", "price_usd",
                    "ram_gb", "storage_gb", "display_type", "refresh_rate_hz", "interaction_count"]
    available_cols = [c for c in display_cols if c in df.columns]

    def highlight_cold(row):
        if row.get("interaction_count", 1) == 0:
            return ["background-color: #1a2f3a"] * len(row)
        return [""] * len(row)

    styled = df[available_cols].style.apply(highlight_cold, axis=1).format({
        "price_usd": "${:.0f}",
        "ram_gb": "{:.0f}GB",
        "storage_gb": "{:.0f}GB",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Detail view
    st.markdown("---")
    st.markdown("### 🔍 Product Detail")
    selected_pid = st.selectbox("Select a product to view full specs", df["product_id"].tolist())
    if selected_pid:
        detail = api_get(f"/api/products/{selected_pid}")
        if not show_error(detail):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**{detail['name']}**")
                st.markdown(f"*{detail['brand']} | {detail['category'].title()}*")
                st.markdown(f"💰 **${detail['price_usd']:.0f}**")
                if detail.get("interaction_count", 0) == 0:
                    st.markdown("❄️ **COLD-START PRODUCT** (zero interactions)")
                else:
                    st.markdown(f"📊 **{detail['interaction_count']}** interactions")
            with dc2:
                specs = {}
                for k in ["processor", "chipset", "ram_gb", "storage_gb", "display_type",
                          "refresh_rate_hz", "gpu", "battery_mah", "camera_mp", "os"]:
                    v = detail.get(k)
                    if v:
                        label = k.replace("_", " ").title()
                        if k == "ram_gb": v = f"{int(v)}GB"
                        elif k == "storage_gb": v = f"{int(v)}GB"
                        elif k == "refresh_rate_hz": v = f"{v}Hz"
                        elif k == "battery_mah": v = f"{v}mAh"
                        elif k == "camera_mp": v = f"{v}MP"
                        specs[label] = v
                for k, v in specs.items():
                    st.text(f"{k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE C — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Recommendations":
    st.markdown("# 🎯 Recommendations")
    st.markdown("Configure a recommendation request. Scores come directly from the FastAPI backend — no frontend computation.")

    # ── Load users and products for selectors ─────────────────────────────────
    products = api_get("/api/products?limit=500")
    users = api_get("/api/users?limit=100")

    col1, col2 = st.columns(2)
    with col1:
        if not show_error(users) and users:
            user_df = pd.DataFrame(users)
            user_id = st.selectbox("👤 Select User", user_df["user_id"].tolist())
        else:
            user_id = st.text_input("👤 User ID", value="U0001")

    with col2:
        cat_choice = st.selectbox("🏷️ Category Filter", ["All", "laptop", "smartphone"])

    if not show_error(products):
        prod_df = pd.DataFrame(products)
        if cat_choice != "All":
            prod_df = prod_df[prod_df["category"] == cat_choice]

        prod_options = prod_df.apply(
            lambda r: f"{r['product_id']} | {r['name'][:50]} | ${r['price_usd']:.0f} | {int(r.get('interaction_count', 0))} interactions",
            axis=1,
        ).tolist()
        selected_option = st.selectbox("📦 Select Anchor Product", prod_options)
        selected_pid = selected_option.split(" | ")[0].strip()

        col3, col4 = st.columns(2)
        with col3:
            required_specs = st.text_input(
                "🔧 Hardware Requirements (optional)",
                placeholder="e.g. 32GB RAM OLED 144Hz under $2000",
            )
        with col4:
            top_k = st.slider("🔢 Top-K Recommendations", 1, 10, 5)

        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Querying recommendation engine..."):
                payload = {
                    "user_id": str(user_id),
                    "product_id": selected_pid,
                    "required_specs": required_specs,
                    "top_k": top_k,
                }
                result = api_post("/api/recommend", payload)

            if show_error(result):
                pass
            else:
                # ── Route indicator ───────────────────────────────────────────
                mode = result.get("recommendation_mode", "unknown")
                mode_colors = {
                    "hybrid": ("#4ade80", "🔀 HYBRID"),
                    "cold_start": ("#38bdf8", "❄️ COLD-START"),
                    "new_user_fallback": ("#fbbf24", "👤 NEW USER FALLBACK"),
                }
                mode_color, mode_label = mode_colors.get(mode, ("#8890a8", mode.upper()))

                col_mode, col_lat, col_mv = st.columns(3)
                with col_mode:
                    st.markdown(
                        f'<div style="color:{mode_color};font-size:1.1rem;font-weight:700;">'
                        f'{mode_label}</div>',
                        unsafe_allow_html=True,
                    )
                with col_lat:
                    st.metric("⏱ Latency", f"{result.get('latency_ms', 0):.1f} ms")
                with col_mv:
                    st.metric("📌 Model Version", result.get("model_version", "N/A"))

                # ── Explanation box ───────────────────────────────────────────
                exp = result.get("explanation", {})
                if exp:
                    st.info(
                        f"**Formula:** {exp.get('formula', 'N/A')}  \n"
                        + (f"**ALS skipped:** {exp.get('als_skipped_reason')}" if exp.get("als_skipped_reason") else "")
                    )

                st.markdown("---")
                st.markdown(f"### Top {len(result.get('recommendations', []))} Recommendations")

                for i, rec in enumerate(result.get("recommendations", []), 1):
                    with st.container():
                        st.markdown(f"""
<div class="rec-card">
  <div class="rec-title">#{i} {rec['name']}</div>
  <div class="rec-brand">{rec['brand']} · {rec['category'].title()} · ${rec['price_usd']:.0f}</div>
  <hr style="border-color:#3d4266;margin:0.6rem 0;">
  {score_bar_html("ALS Score", rec['als_score'], "#60a5fa")}
  {score_bar_html("TF-IDF Score", rec['tfidf_score'], "#a78bfa")}
  {score_bar_html("Spec Match", rec['spec_match_score'], "#34d399")}
  {score_bar_html("Final Score", rec['final_score'], "#f472b6")}
  <div style="margin-top:0.7rem;font-size:0.8rem;color:#8890a8;">💡 {rec['reason']}</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE D — COLD-START DEMO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "❄️ Cold-Start Demo":
    st.markdown("# ❄️ Item Cold-Start Demonstration")
    st.markdown(
        "Select a **new product** (interaction_count = 0) to see how CartSense handles "
        "the cold-start problem without ALS behavioral data."
    )

    products = api_get("/api/products?limit=500")
    users = api_get("/api/users?limit=50")

    if show_error(products):
        st.stop()

    prod_df = pd.DataFrame(products)
    cold_df = prod_df[prod_df["interaction_count"] == 0]
    warm_df = prod_df[prod_df["interaction_count"] > 0]

    cs1, cs2 = st.columns(2)
    with cs1:
        st.metric("❄️ Cold-Start Products", len(cold_df))
    with cs2:
        st.metric("🔥 Warm Products", len(warm_df))

    if cold_df.empty:
        st.warning("All products have interactions. No cold-start products available.")
        st.info("Tip: Products with 0 interactions are generated when the catalog has new items.")
        st.stop()

    cold_options = cold_df.apply(
        lambda r: f"{r['product_id']} | {r['name'][:50]} | {r['category']}",
        axis=1,
    ).tolist()
    selected_cold = st.selectbox("🆕 Select Cold-Start Product", cold_options)
    cold_pid = selected_cold.split(" | ")[0].strip()

    user_id = "U0001"
    if not show_error(users) and users:
        user_df = pd.DataFrame(users)
        user_id = st.selectbox("👤 Select User", user_df["user_id"].tolist())

    required_specs = st.text_input("🔧 Hardware Requirements (optional)", placeholder="e.g. OLED 120Hz 16GB")

    if st.button("🚀 Run Cold-Start Recommendation", type="primary"):
        with st.spinner("Running cold-start recommendation..."):
            payload = {"user_id": user_id, "product_id": cold_pid, "required_specs": required_specs, "top_k": 5}
            result = api_post("/api/recommend", payload)

        if show_error(result):
            st.stop()

        mode = result.get("recommendation_mode", "")
        st.markdown("---")

        # ── Cold-start explanation panel ──────────────────────────────────────
        st.markdown("""
<div class="cold-start-box">
  <div class="cold-start-title">🧊 ITEM COLD-START DETECTED</div>
  <br>
  <table style="color:#c4c9e2;font-size:0.9rem;width:100%">
    <tr><td style="width:40%"><b>Anchor Product</b></td><td>New product with 0 interactions</td></tr>
    <tr><td><b>ALS Status</b></td><td><span style="color:#ef4444">⛔ SKIPPED</span> — No behavioral interaction history</td></tr>
    <tr><td><b>TF-IDF Status</b></td><td><span style="color:#4ade80">✅ ACTIVE</span> — Content similarity computed</td></tr>
    <tr><td><b>Cosine Similarity</b></td><td><span style="color:#4ade80">✅ ACTIVE</span> — Product spec vectors compared</td></tr>
    <tr><td><b>Spec Matching</b></td><td><span style="color:#4ade80">✅ ACTIVE</span> — Hardware requirements matched</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

        if mode != "cold_start":
            st.warning(f"⚠️ Engine returned mode: {mode}. This product may have some interactions.")

        exp = result.get("explanation", {})
        if exp.get("als_skipped_reason"):
            st.markdown(f"> **ALS skipped reason:** {exp['als_skipped_reason']}")

        st.markdown("---")
        st.markdown(f"### Content-Based Recommendations for `{cold_pid}`")
        st.markdown(f"*Latency: {result.get('latency_ms', 0):.1f}ms | Model: {result.get('model_version', 'N/A')}*")

        for i, rec in enumerate(result.get("recommendations", []), 1):
            cols = st.columns([3, 1, 1, 1, 1])
            with cols[0]:
                st.markdown(f"**#{i} {rec['name']}**")
                st.caption(f"{rec['brand']} · ${rec['price_usd']:.0f}")
            with cols[1]:
                st.metric("TF-IDF", f"{rec['tfidf_score']:.3f}")
            with cols[2]:
                st.metric("Spec Match", f"{rec['spec_match_score']:.3f}")
            with cols[3]:
                st.metric("ALS", "N/A")
            with cols[4]:
                st.metric("Final", f"{rec['final_score']:.3f}")
            st.caption(f"💡 {rec['reason']}")
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE E — MLOPS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 MLOps Dashboard":
    st.markdown("# 📊 MLOps Dashboard")

    model_status = api_get("/api/model/status")
    training_history = api_get("/api/training/history?limit=20")
    drift_status = api_get("/api/drift")

    # ── Model Metrics ─────────────────────────────────────────────────────────
    st.markdown("### 🧠 Current Model Metrics")
    if not show_error(model_status):
        m1, m2, m3, m4 = st.columns(4)
        tfidf_metrics = model_status.get("tfidf_metrics", {})
        als_metrics = model_status.get("als_metrics", {})
        combined_metrics = {**tfidf_metrics, **als_metrics}

        m1.metric("RMSE", f"{combined_metrics.get('rmse', 'N/A')}")
        m2.metric("Precision@5", f"{combined_metrics.get('precision_at_5', 'N/A')}")
        m3.metric("Recall@5", f"{combined_metrics.get('recall_at_5', 'N/A')}")
        m4.metric("NDCG@5", f"{combined_metrics.get('ndcg_at_5', 'N/A')}")

        st.markdown(f"""
| Component | Details |
|---|---|
| Model Version | `{model_status.get('model_version', 'N/A')}` |
| TF-IDF Vocabulary | {model_status.get('tfidf_vocabulary_size', 0):,} terms |
| TF-IDF Documents | {model_status.get('tfidf_n_documents', 0):,} products |
| ALS Rank | {model_status.get('als_rank', 'N/A')} |
| ALS Users | {model_status.get('als_n_users', 0):,} |
| ALS Items | {model_status.get('als_n_items', 0):,} |
| Training Time | {model_status.get('training_timestamp', 'N/A')} |
""")

    # ── Drift Status ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌊 Drift Status")
    if not show_error(drift_status):
        if "dataset_drift_detected" in drift_status:
            drift_detected = drift_status.get("dataset_drift_detected", False)
            share = drift_status.get("share_drifted_columns", 0.0)
            threshold = drift_status.get("drift_threshold", 0.5)

            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Drift Detected", "YES ⚠️" if drift_detected else "NO ✓")
            dc2.metric("Share Drifted", f"{share*100:.1f}%")
            dc3.metric("Threshold", f"{threshold*100:.0f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=share * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Drift Score (% features drifted)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if drift_detected else "#4ade80"},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#1e2235"},
                        {"range": [threshold * 100, 100], "color": "#3a1a1a"},
                    ],
                    "threshold": {"line": {"color": "orange", "width": 3}, "value": threshold * 100},
                },
            ))
            fig.update_layout(height=250, paper_bgcolor="#0f1117", font_color="#c4c9e2")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drift checks run yet. Use the **Drift Simulation** page to run one.")

    # ── Training History ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Training History")
    if not show_error(training_history) and training_history:
        hist_df = pd.DataFrame(training_history)
        if not hist_df.empty:
            st.dataframe(
                hist_df[["id", "model_type", "model_version", "rmse", "status", "promoted", "timestamp"]],
                use_container_width=True,
                hide_index=True,
            )

            # RMSE trend
            numeric_df = hist_df.dropna(subset=["rmse"])
            if len(numeric_df) > 1:
                fig = px.line(
                    numeric_df, x="timestamp", y="rmse",
                    color="model_type", markers=True,
                    title="RMSE Over Training Runs",
                    labels={"rmse": "RMSE", "timestamp": "Training Time"},
                )
                fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235", font_color="#c4c9e2")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training runs recorded yet. Trigger retraining from the **Drift Simulation** page.")

    # ── Manual Retraining ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Manual Retraining")
    reason = st.text_input("Reason", value="manual_trigger_dashboard")
    if st.button("🔄 Trigger Retraining", type="secondary"):
        with st.spinner("Starting retraining pipeline..."):
            result = api_post("/api/retrain", {"force": True, "reason": reason})
        if not show_error(result):
            st.success(f"✅ {result.get('message', 'Retraining started.')}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE F — DRIFT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌊 Drift Simulation":
    st.markdown("# 🌊 Concept Drift Simulation")
    st.markdown("""
This page demonstrates the **concept drift detection and retraining pipeline**.

**Baseline market (training data):** IPS displays, 60Hz, 8-16GB RAM, traditional processors  
**Drifted market (simulated):** OLED, 120-165Hz, 24-64GB RAM, AI/NPU processors
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Baseline Distribution")
        st.markdown("""
| Feature | Baseline |
|---|---|
| RAM | 8-16GB typical |
| Display | IPS / LCD dominant |
| Refresh Rate | 60Hz typical |
| Price | $800-1200 mid-range |
| Battery | 4000mAh average |
""")
    with col2:
        st.markdown("### 📈 Drifted Distribution (Simulated)")
        st.markdown("""
| Feature | Drifted Market |
|---|---|
| RAM | 24-64GB common |
| Display | OLED / AMOLED dominant |
| Refresh Rate | 120-165Hz standard |
| Price | $1500-2500 premium shift |
| Battery | 5000-6000mAh |
""")

    st.markdown("---")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        simulate = st.checkbox("🎭 Inject simulated market drift data (demo mode)", value=True)
    with col_b:
        threshold_display = st.number_input("Drift threshold (share of features)", 0.1, 1.0, 0.5, 0.05)

    if st.button("🔍 Run Drift Detection", type="primary", use_container_width=True):
        with st.spinner("Running Evidently AI drift analysis..."):
            payload = {
                "simulate_drift": simulate,
                "drift_type": "market_shift",
                "n_reference_days": 90,
                "n_current_days": 30,
            }
            result = api_post("/api/drift/check", payload)

        if show_error(result):
            st.stop()

        drift_detected = result.get("dataset_drift_detected", False)
        share = result.get("share_drifted_columns", 0.0)
        n_drifted = result.get("n_drifted_columns", 0)
        n_total = result.get("n_total_columns", 0)

        st.markdown("---")
        st.markdown("### 📋 Drift Analysis Results")

        if drift_detected:
            st.markdown(f"""
<div class="drift-alert">
  <h3 style="color:#ef4444;">🚨 CONCEPT DRIFT DETECTED</h3>
  <p><b>{n_drifted} of {n_total} features</b> show statistically significant distributional shift (KS test)</p>
  <p>Share drifted: <b>{share*100:.1f}%</b> | Threshold: <b>{threshold_display*100:.0f}%</b></p>
  <p>→ <b>Retraining is recommended</b></p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="drift-ok">
  <h3 style="color:#4ade80;">✅ NO SIGNIFICANT DRIFT</h3>
  <p>Share drifted: <b>{share*100:.1f}%</b> — below threshold of {threshold_display*100:.0f}%</p>
</div>
""", unsafe_allow_html=True)

        # Column details
        col_results = result.get("column_results", [])
        if col_results:
            st.markdown("### 📊 Feature-Level Drift Results")
            cr_df = pd.DataFrame(col_results)
            cr_df["drift_detected"] = cr_df["drift_detected"].map({True: "⚠️ DRIFTED", False: "✓ Stable"})
            st.dataframe(cr_df, use_container_width=True, hide_index=True)

            # Bar chart
            chart_df = pd.DataFrame(col_results)
            fig = px.bar(
                chart_df, x="column_name", y="statistic_value",
                color="drift_detected",
                color_discrete_map={True: "#ef4444", False: "#4ade80"},
                title="KS Statistic per Feature (higher = more drift)",
                labels={"statistic_value": "KS Statistic", "column_name": "Feature"},
            )
            fig.add_hline(y=threshold_display, line_dash="dash", line_color="orange",
                          annotation_text=f"Threshold ({threshold_display:.2f})")
            fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2235", font_color="#c4c9e2")
            st.plotly_chart(fig, use_container_width=True)

        # ── Retraining trigger ─────────────────────────────────────────────────
        if drift_detected:
            st.markdown("---")
            st.markdown("### 🔄 Trigger Retraining")
            st.warning("Drift exceeds threshold. Retrain the models?")
            if st.button("⚡ Retrain Models Now", type="primary"):
                with st.spinner("Starting retraining pipeline..."):
                    retrain_result = api_post("/api/retrain", {
                        "force": True,
                        "reason": "concept_drift_detected_via_simulation",
                    })
                if not show_error(retrain_result):
                    st.success(f"✅ {retrain_result.get('message', 'Retraining started!')}")
                    st.info("After retraining completes, the model version will update automatically via hot-reload.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE G — MODEL / EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Model / Experiments":
    st.markdown("# 🔬 Model & Experiment Details")

    model_status = api_get("/api/model/status")
    mlflow_exps = api_get("/api/mlflow/experiments")
    mlflow_runs = api_get("/api/mlflow/runs")

    # ── TF-IDF Info ───────────────────────────────────────────────────────────
    st.markdown("### 📝 Content-Based Model (TF-IDF)")
    if not show_error(model_status):
        col1, col2, col3 = st.columns(3)
        col1.metric("Vocabulary Size", f"{model_status.get('tfidf_vocabulary_size', 0):,}")
        col2.metric("Documents (Products)", model_status.get("tfidf_n_documents", 0))
        col3.metric("Model Version", model_status.get("model_version", "N/A"))

        tfidf_m = model_status.get("tfidf_metrics", {})
        if tfidf_m:
            st.json(tfidf_m)

        st.markdown("""
**How TF-IDF works in CartSense:**
- Fitted EXCLUSIVELY on our own 1000-product electronics catalog
- `ngram_range=(1,2)`: captures single words AND bigrams ("OLED display", "32GB RAM")  
- `min_df=2`: ignores terms unique to only one product  
- `sublinear_tf=True`: dampens effect of very frequent terms
- At inference: computes cosine similarity between anchor product vector and all same-category candidates
        """)

    # ── ALS Info ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤝 Collaborative Filtering Model (PySpark ALS)")
    if not show_error(model_status):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rank", model_status.get("als_rank", "N/A"))
        col2.metric("Users", f"{model_status.get('als_n_users', 0):,}")
        col3.metric("Items", f"{model_status.get('als_n_items', 0):,}")
        col4.metric("Status", "✓ Loaded" if model_status.get("als_loaded") else "✗ Not loaded")

        als_m = model_status.get("als_metrics", {})
        if als_m:
            st.json(als_m)

        st.markdown("""
**How ALS works in CartSense:**
- Trained FROM SCRATCH using PySpark MLlib ALS on our interaction matrix
- Interaction weights: view=1, click=2, wishlist=3, cart=4, purchase=5
- Temporal split: oldest 80% → training, newest 20% → evaluation
- Latent factors extracted after training and saved as Parquet files
- **At inference: numpy dot-product only** — no Spark session started per request
- Unknown users → fallback to TF-IDF content-based recommendations
        """)

    # ── MLflow ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📡 MLflow Experiment Tracking")

    if not show_error(mlflow_exps) and isinstance(mlflow_exps, list):
        st.markdown(f"**{len(mlflow_exps)} experiments** registered in MLflow")
        for exp in mlflow_exps:
            st.markdown(f"- `{exp.get('name', 'N/A')}` (ID: {exp.get('experiment_id', 'N/A')})")

    if not show_error(mlflow_runs):
        if isinstance(mlflow_runs, dict) and "runs" in mlflow_runs:
            runs_data = mlflow_runs["runs"]
        elif isinstance(mlflow_runs, list):
            runs_data = mlflow_runs
        else:
            runs_data = []

        if runs_data:
            runs_df = pd.DataFrame(runs_data)
            st.markdown("**Recent MLflow Runs:**")
            st.dataframe(runs_df, use_container_width=True, hide_index=True)

    # ── Baseline Comparison ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Comparison (Baseline vs CartSense Hybrid)")
    st.markdown("""
To run a real baseline comparison, trigger the evaluation via the retraining pipeline.
The table below shows the design target comparisons:
    """)

    comparison_data = {
        "Model": ["Popularity Baseline", "TF-IDF Only", "ALS Only", "CartSense Hybrid"],
        "Precision@5": ["~0.05", "~0.12", "~0.18", "~0.22"],
        "Recall@5": ["~0.08", "~0.15", "~0.20", "~0.25"],
        "NDCG@5": ["~0.06", "~0.14", "~0.19", "~0.24"],
        "Cold-Start": ["❌ No", "✅ Yes", "❌ No", "✅ Yes"],
        "Drift-Resilient": ["❌ No", "⚠️ Partial", "❌ No", "✅ Yes"],
    }
    st.table(pd.DataFrame(comparison_data).set_index("Model"))
    st.caption("*Note: Values above are architectural estimates. Actual metrics are computed from real test data after training.*")
