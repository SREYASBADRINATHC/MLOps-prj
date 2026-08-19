import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 12

st.set_page_config(page_title="CartSense", page_icon="🛒", layout="wide")
st.title("CartSense — Drift-Resilient Hybrid Recommender")


def call_backend_api(method: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
    """Centralized API client so UI keeps transport and timeout behavior consistent."""
    url = f"{BACKEND_URL}{endpoint}"
    if method == "GET":
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    elif method == "POST":
        response = requests.post(url, json=payload or {}, timeout=REQUEST_TIMEOUT_SECONDS)
    else:
        raise ValueError(f"Unsupported method={method}")
    response.raise_for_status()
    return response.json()


def render_storefront_tab() -> None:
    st.subheader("Storefront")
    try:
        products_payload = call_backend_api("GET", "/api/products")
    except requests.RequestException as exc:
        st.error(f"Unable to load products from FastAPI backend: {exc}")
        return

    products_df = pd.DataFrame(products_payload)
    if products_df.empty:
        st.warning("Catalog is empty. Run data ingestion before using the storefront.")
        return

    controls_col_1, controls_col_2, controls_col_3 = st.columns([1, 2, 2])
    with controls_col_1:
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    with controls_col_2:
        selected_product_name = st.selectbox("Select Product", options=products_df["product_name"].tolist())
    with controls_col_3:
        required_specs = st.text_input("Required Specifications", placeholder="e.g., OLED, 32GB RAM")

    selected_product_id = int(
        products_df.loc[products_df["product_name"] == selected_product_name, "product_id"].iloc[0]
    )

    st.dataframe(
        products_df[["product_id", "product_name", "category", "brand", "price_usd"]],
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Get Top 3 Recommendations", type="primary", use_container_width=True):
        payload = {"user_id": int(user_id), "product_id": selected_product_id, "required_specs": required_specs}
        try:
            recommendations_payload = call_backend_api("POST", "/api/recommend", payload=payload)
            recommendations_df = pd.DataFrame(recommendations_payload)
            st.success("Recommendations generated successfully.")
            st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
        except requests.HTTPError as exc:
            error_text = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Recommendation API returned an error: {error_text}")
        except requests.RequestException as exc:
            st.error(f"Failed to call recommendation API: {exc}")


def render_mlops_tab() -> None:
    st.subheader("MLOps Dashboard")
    try:
        drift_payload = call_backend_api("GET", "/api/drift")
    except requests.RequestException as exc:
        st.error(f"Unable to load drift metrics: {exc}")
        return

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Dataset Drift", "Detected" if drift_payload["dataset_drift_detected"] else "Not Detected")
    metric_col_2.metric("Drifted Columns", drift_payload["drifted_columns_count"])
    metric_col_3.metric("Drift Share", f'{drift_payload["share_of_drifted_columns"]}%')

    trend_df = pd.DataFrame(
        {
            "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "drift_score": [6.2, 7.1, 8.4, 9.3, 10.1, 11.2, drift_payload["share_of_drifted_columns"]],
        }
    )
    trend_fig = px.line(trend_df, x="day", y="drift_score", markers=True, title="Weekly Drift Trend (Mock)")
    st.plotly_chart(trend_fig, use_container_width=True)


tab_storefront, tab_mlops = st.tabs(["Storefront", "MLOps Dashboard"])
with tab_storefront:
    render_storefront_tab()
with tab_mlops:
    render_mlops_tab()
