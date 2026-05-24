"""Streamlit frontend for the fraud detection system.

This app acts as a production-style client for the FastAPI backend. It lets
users enter a raw transaction payload, sends it to the backend `/predict`
endpoint, and displays the fraud result with a risk explanation.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st


# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
            color: white;
            padding: 1.5rem 1.75rem;
            border-radius: 1.25rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
        }
        .metric-shell {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 1rem;
            padding: 0.35rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }
        .risk-banner {
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            border-left: 6px solid #0f172a;
            background: white;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
        }
        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            background: white;
            padding: 1rem;
            border-radius: 1rem;
            border: 1px solid rgba(148,163,184,0.25);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------------
# Constants and defaults
# ----------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
API_TIMEOUT_SECONDS = 20
MAX_HISTORY_ITEMS = 20

RISK_LABELS = [
    (0.85, "Critical", "#7f1d1d"),
    (0.65, "High", "#b45309"),
    (0.35, "Medium", "#ca8a04"),
    (0.0, "Low", "#15803d"),
]


def init_state() -> None:
    """Initialize Streamlit session state used by the app."""
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("ui_threshold", 0.5)
    st.session_state.setdefault("transaction_history", [])
    st.session_state.setdefault("sample_transaction", get_default_transaction())
    st.session_state.setdefault("simulation_running", False)


def get_default_transaction() -> Dict[str, Any]:
    """Return a sample transaction aligned with the backend schema."""
    return {
        "transaction_id": "TXN_00001",
        "user_id": "U001",
        "amount": 249.99,
        "transaction_type": "payment",
        "merchant_category": "electronics",
        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
        "transaction_frequency": 8,
        "avg_user_amount": 180.0,
        "deviation_from_avg": 69.99,
        "transaction_gap_seconds": 3600,
        "account_age_days": 540,
        "failed_attempts": 0,
        "device_type": "web",
        "location": "USA",
        "is_foreign_transaction": 0,
        "unusual_amount_flag": 0,
        "velocity_flag": 0,
        "new_device_flag": 0,
        "location_change_flag": 0,
        "night_transaction_flag": 0,
    }


def risk_category(probability: float) -> str:
    """Convert probability into a human-readable risk band."""
    for threshold, label, _ in RISK_LABELS:
        if probability >= threshold:
            return label
    return "Low"


def risk_color(probability: float) -> str:
    """Return the color associated with the risk band."""
    for threshold, _, color in RISK_LABELS:
        if probability >= threshold:
            return color
    return "#15803d"


def backend_predict(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the FastAPI prediction endpoint."""
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json=payload,
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def backend_predict_batch(api_url: str, payloads: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Call the FastAPI batch prediction endpoint."""
    response = requests.post(
        f"{api_url.rstrip('/')}/predict-batch",
        json={"transactions": payloads},
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def backend_health(api_url: str) -> Dict[str, Any]:
    """Check backend readiness."""
    response = requests.get(f"{api_url.rstrip('/')}/health", timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def render_header() -> None:
    """Render the top section of the app."""
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.25rem;">🛡️ Fraud Detection Dashboard</h1>
            <div style="font-size:1.02rem;opacity:0.92;">
                Production-grade Streamlit client for fraud risk scoring, batch review, and live monitoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, Any]:
    """Build the sidebar controls and return the selected settings."""
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Dashboard", "Predict", "Batch Upload", "Monitoring", "Analytics", "Settings"],
            index=1,
        )

        st.divider()
        st.subheader("Backend")
        api_url = st.text_input("API URL", value=st.session_state.get("api_url", DEFAULT_API_URL))

        health_clicked = st.button("Check Backend Health", use_container_width=True)
        health_result = None
        health_error = None
        if health_clicked:
            try:
                health_result = backend_health(api_url)
                st.success("Backend is reachable")
            except Exception as exc:
                health_error = str(exc)
                st.error("Backend health check failed")

        st.divider()
        st.subheader("Model Settings")
        threshold = st.slider(
            "UI decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.get("ui_threshold", 0.5)),
            step=0.01,
        )

        st.divider()
        st.subheader("Example Data")
        load_sample = st.button("Load sample transaction", use_container_width=True)

        st.divider()
        st.caption("Tip: use the sample transaction to test the full API flow in seconds.")

    st.session_state["api_url"] = api_url
    st.session_state["ui_threshold"] = threshold

    return {
        "page": page,
        "api_url": api_url,
        "threshold": threshold,
        "health_result": health_result,
        "health_error": health_error,
        "load_sample": load_sample,
    }


def render_dashboard(api_url: str) -> None:
    """Render a simple operational overview."""
    st.subheader("Operational Overview")

    st.markdown(
        """
        <div class="small-muted">
            The dashboard summarizes backend availability, the current decision threshold,
            and recent prediction activity in one place.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("API URL", api_url)
    with col2:
        st.metric("Decision Threshold", f"{st.session_state.get('ui_threshold', 0.5):.2f}")
    with col3:
        st.metric("Model", "fraud_model.pkl")
    with col4:
        st.metric("Backend Status", "Ready")

    st.divider()
    st.subheader("What this app does")
    st.write(
        "This dashboard sends raw transaction data to the FastAPI backend, applies the "
        "same preprocessing used during training, and returns the fraud probability plus "
        "a human-readable risk band."
    )

    if st.session_state.get("transaction_history"):
        st.subheader("Recent transaction history")
        history_df = pd.DataFrame(st.session_state["transaction_history"])
        st.dataframe(history_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info('Use the Predict page to submit a transaction. The app will call the backend "/predict" endpoint.')


def transaction_form(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render the full transaction form and return the submitted payload."""
    st.header("Predict Fraud Risk")
    st.caption("Enter the raw transaction fields expected by the trained preprocessing pipeline.")

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            transaction_id = st.text_input("Transaction ID", value=str(defaults["transaction_id"]))
            user_id = st.text_input("User ID", value=str(defaults["user_id"]))
            amount = st.number_input("Amount", min_value=0.01, value=float(defaults["amount"]), step=1.0)
            transaction_type = st.selectbox(
                "Transaction Type",
                ["payment", "transfer", "withdrawal"],
                index=["payment", "transfer", "withdrawal"].index(defaults["transaction_type"]),
            )
            merchant_category = st.selectbox(
                "Merchant Category",
                ["electronics", "fashion", "travel", "gaming", "grocery"],
                index=["electronics", "fashion", "travel", "gaming", "grocery"].index(defaults["merchant_category"]),
            )

        with col2:
            timestamp = st.text_input("Timestamp (ISO 8601)", value=str(defaults["timestamp"]))
            transaction_frequency = st.number_input(
                "Transaction Frequency",
                min_value=0.0,
                value=float(defaults["transaction_frequency"]),
                step=1.0,
            )
            avg_user_amount = st.number_input(
                "Average User Amount",
                min_value=0.0,
                value=float(defaults["avg_user_amount"]),
                step=1.0,
            )
            deviation_from_avg = st.number_input(
                "Deviation From Average",
                value=float(defaults["deviation_from_avg"]),
                step=1.0,
            )
            transaction_gap_seconds = st.number_input(
                "Transaction Gap Seconds",
                min_value=0.0,
                value=float(defaults["transaction_gap_seconds"]),
                step=60.0,
            )

        with col3:
            account_age_days = st.number_input(
                "Account Age Days",
                min_value=0.0,
                value=float(defaults["account_age_days"]),
                step=1.0,
            )
            failed_attempts = st.number_input(
                "Failed Attempts",
                min_value=0,
                value=int(defaults["failed_attempts"]),
                step=1,
            )
            device_type = st.selectbox(
                "Device Type",
                ["web", "mobile"],
                index=["web", "mobile"].index(defaults["device_type"]),
            )
            location = st.selectbox(
                "Location",
                ["USA", "UK", "Germany", "India", "UAE"],
                index=["USA", "UK", "Germany", "India", "UAE"].index(defaults["location"]),
            )

        flag_col1, flag_col2, flag_col3 = st.columns(3)
        with flag_col1:
            is_foreign_transaction = st.selectbox("Foreign Transaction", [0, 1], index=int(defaults["is_foreign_transaction"]))
            unusual_amount_flag = st.selectbox("Unusual Amount Flag", [0, 1], index=int(defaults["unusual_amount_flag"]))
        with flag_col2:
            velocity_flag = st.selectbox("Velocity Flag", [0, 1], index=int(defaults["velocity_flag"]))
            new_device_flag = st.selectbox("New Device Flag", [0, 1], index=int(defaults["new_device_flag"]))
        with flag_col3:
            location_change_flag = st.selectbox("Location Change Flag", [0, 1], index=int(defaults["location_change_flag"]))
            night_transaction_flag = st.selectbox("Night Transaction Flag", [0, 1], index=int(defaults["night_transaction_flag"]))

        submitted = st.form_submit_button("Predict Fraud Risk", type="primary", use_container_width=True)

    if not submitted:
        return {}

    try:
        payload = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "amount": amount,
            "transaction_type": transaction_type,
            "merchant_category": merchant_category,
            "timestamp": timestamp,
            "transaction_frequency": transaction_frequency,
            "avg_user_amount": avg_user_amount,
            "deviation_from_avg": deviation_from_avg,
            "transaction_gap_seconds": transaction_gap_seconds,
            "account_age_days": account_age_days,
            "failed_attempts": failed_attempts,
            "device_type": device_type,
            "location": location,
            "is_foreign_transaction": is_foreign_transaction,
            "unusual_amount_flag": unusual_amount_flag,
            "velocity_flag": velocity_flag,
            "new_device_flag": new_device_flag,
            "location_change_flag": location_change_flag,
            "night_transaction_flag": night_transaction_flag,
        }
        return payload
    except Exception as exc:
        st.error(f"Invalid form data: {exc}")
        return {}


def show_prediction_result(result: Dict[str, Any], threshold: float) -> None:
    """Render the backend prediction in a clean, explainable layout."""
    probability = float(result["fraud_probability"])
    predicted_label = result["prediction"]
    risk_category_value = result.get("risk_category", risk_category(probability))
    confidence = float(result["confidence"])
    color = risk_color(probability)

    if predicted_label == "Fraud":
        st.error("Fraud alert: the transaction exceeds the configured decision threshold.")
    else:
        st.success("Transaction classified as legitimate.")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Fraud Probability", f"{probability:.1%}")
    with metric_col2:
        st.metric("Prediction", predicted_label)
    with metric_col3:
        st.metric("Risk Category", risk_category_value)
    with metric_col4:
        st.metric("Confidence", f"{confidence:.1%}")

    st.markdown(
        f"""
        <div class="risk-banner" style="border-left-color:{color};">
            <strong>Decision threshold:</strong> {threshold:.2f}<br>
            <strong>Risk score:</strong> {probability:.4f}<br>
            <strong>Human summary:</strong> This transaction is categorized as <strong>{risk_category_value}</strong> risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "The backend applies the same preprocessing pipeline used during training before generating this score."
    )


def render_prediction_page(api_url: str, threshold: float, load_sample: bool) -> None:
    """Handle the prediction workflow."""
    defaults = st.session_state.get("sample_transaction", get_default_transaction())
    if load_sample:
        st.session_state["sample_transaction"] = get_default_transaction()
        defaults = st.session_state["sample_transaction"]
        st.toast("Loaded sample transaction", icon="✅")

    payload = transaction_form(defaults)

    if not payload:
        st.code(json.dumps(defaults, indent=2), language="json")
        return

    with st.expander("Request payload preview", expanded=False):
        st.code(json.dumps(payload, indent=2), language="json")

    try:
        with st.spinner("Calling fraud detection backend..."):
            result = backend_predict(api_url, payload)
        show_prediction_result(result, threshold)

        history_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_id": payload["transaction_id"],
            "prediction": result["prediction"],
            "fraud_probability": round(float(result["fraud_probability"]), 4),
            "risk_score": round(float(result["risk_score"]), 4),
            "risk_category": result.get("risk_category", risk_category(float(result["fraud_probability"]))),
            "confidence": round(float(result["confidence"]), 4),
        }

        st.session_state["last_prediction"] = {
            "request": payload,
            "response": result,
        }
        st.session_state["transaction_history"] = [history_item] + st.session_state["transaction_history"][: MAX_HISTORY_ITEMS - 1]
        st.toast("Prediction saved to history", icon="📌")

    except requests.HTTPError as exc:
        response = exc.response
        detail = response.json() if response is not None else {"detail": str(exc)}
        st.error("Backend returned an error")
        st.json(detail)
    except requests.RequestException as exc:
        st.error("Could not reach the backend API")
        st.exception(exc)
    except Exception as exc:
        st.error("Unexpected prediction error")
        st.exception(exc)


def render_batch_upload_page(api_url: str) -> None:
    """Render CSV upload and batch prediction workflow."""
    st.header("Batch Fraud Prediction")
    st.caption("Upload a CSV with the same raw columns used for single predictions.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if not uploaded_file:
        st.info("Choose a CSV file to run batch fraud prediction.")
        return

    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded file")
        st.dataframe(batch_df.head(10), use_container_width=True)

        if st.button("Run batch prediction", type="primary"):
            required_columns = list(get_default_transaction().keys())
            missing_columns = [col for col in required_columns if col not in batch_df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return

            payloads = batch_df[required_columns].to_dict(orient="records")

            with st.spinner("Sending batch to backend..."):
                result = backend_predict_batch(api_url, payloads)

            predictions = pd.DataFrame(result["predictions"])
            if "transaction_id" not in predictions.columns:
                predictions.insert(0, "transaction_id", batch_df["transaction_id"].values[: len(predictions)])

            st.success(f"Processed {result['count']} transactions")
            st.metric("Fraudulent transactions", int((predictions["prediction"] == "Fraud").sum()))
            st.metric("Average fraud probability", f"{predictions['fraud_probability'].mean():.1%}")
            st.dataframe(predictions, use_container_width=True, hide_index=True)

            csv_bytes = predictions.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download batch results",
                data=csv_bytes,
                file_name="fraud_batch_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error("Batch prediction failed")
        st.exception(exc)


def render_monitoring_page(api_url: str) -> None:
    """Render a light real-time monitoring simulation for fraud operations."""
    st.header("Real-Time Monitoring Simulation")
    st.caption("This simulation creates a stream of transactions and refreshes the metrics like a live fraud queue.")

    col1, col2, col3 = st.columns(3)
    count = col1.slider("Transactions to simulate", 5, 50, 20)
    fraud_bias = col2.slider("Fraud bias", 0.05, 0.95, 0.25, step=0.05)
    run_simulation = col3.button("Run simulation", type="primary")

    if not run_simulation:
        st.info("Adjust the controls and click Run simulation to generate a live-style fraud stream.")
        return

    rng = random.Random(42)
    simulated_rows = []
    progress = st.progress(0)
    status = st.empty()

    for index in range(count):
        status.write(f"Simulating transaction {index + 1} of {count}...")
        sample_payload = get_default_transaction()
        sample_payload["transaction_id"] = f"SIM_{index + 1:05d}"
        sample_payload["amount"] = round(rng.uniform(10, 2500), 2)
        sample_payload["failed_attempts"] = rng.randint(0, 3)
        sample_payload["is_foreign_transaction"] = 1 if rng.random() < fraud_bias else 0
        sample_payload["unusual_amount_flag"] = 1 if rng.random() < fraud_bias else 0
        sample_payload["velocity_flag"] = 1 if rng.random() < fraud_bias else 0

        try:
            result = backend_predict(api_url, sample_payload)
            simulated_rows.append(
                {
                    "transaction_id": sample_payload["transaction_id"],
                    "amount": sample_payload["amount"],
                    "prediction": result["prediction"],
                    "fraud_probability": float(result["fraud_probability"]),
                    "risk_category": result.get("risk_category", risk_category(float(result["fraud_probability"]))),
                }
            )
        except Exception as exc:
            simulated_rows.append(
                {
                    "transaction_id": sample_payload["transaction_id"],
                    "amount": sample_payload["amount"],
                    "prediction": "Error",
                    "fraud_probability": 0.0,
                    "risk_category": "Unknown",
                    "error": str(exc),
                }
            )

        progress.progress((index + 1) / count)

    st.success("Simulation complete")
    sim_df = pd.DataFrame(simulated_rows)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Simulated transactions", len(sim_df))
    with col2:
        st.metric("Fraud alerts", int((sim_df["prediction"] == "Fraud").sum()))
    with col3:
        st.metric("Avg fraud probability", f"{sim_df['fraud_probability'].mean():.1%}")

    st.line_chart(sim_df["fraud_probability"])
    st.dataframe(sim_df, use_container_width=True, hide_index=True)


def render_analytics_page() -> None:
    """Display analytics for the current session and prediction history."""
    st.header("Analytics")
    st.write("Monitor patterns from the current browser session and the latest predictions.")

    if "transaction_history" not in st.session_state or not st.session_state["transaction_history"]:
        st.info("Run one or more predictions to populate analytics.")
        return

    history_df = pd.DataFrame(st.session_state["transaction_history"])
    history_df["fraud_probability"] = history_df["fraud_probability"].astype(float)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total predictions", len(history_df))
    with metric_cols[1]:
        st.metric("Fraud alerts", int((history_df["prediction"] == "Fraud").sum()))
    with metric_cols[2]:
        st.metric("Average fraud probability", f"{history_df['fraud_probability'].mean():.1%}")
    with metric_cols[3]:
        st.metric("Max fraud probability", f"{history_df['fraud_probability'].max():.1%}")

    st.subheader("Probability trend")
    st.line_chart(history_df.sort_values("timestamp")["fraud_probability"].reset_index(drop=True))

    st.subheader("Recent transaction history")
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    st.subheader("Risk distribution")
    distribution = history_df.groupby("risk_category").size().reset_index(name="count")
    st.bar_chart(distribution.set_index("risk_category"))


def render_analytics_page() -> None:
    """Display lightweight analytics for the frontend session."""
    st.header("Analytics")
    st.write("These charts summarize the current session. Replace them with live backend metrics if needed.")

    if "last_prediction" not in st.session_state:
        st.info("Run a prediction first to populate the analytics view.")
        return

    response = st.session_state["last_prediction"]["response"]
    score = float(response["fraud_probability"])
    df = pd.DataFrame(
        {
            "Metric": ["Fraud Probability", "Confidence", "Threshold Gap"],
            "Value": [score, float(response["confidence"]), abs(score - float(st.session_state.get("ui_threshold", 0.5)))],
        }
    )
    st.bar_chart(df.set_index("Metric"))
    st.dataframe(df, use_container_width=True)


def render_settings_page() -> None:
    """Display backend and UI settings."""
    st.header("Settings")
    st.write("These settings are stored in session state for the current browser session.")

    st.text_input("Stored API URL", value=st.session_state.get("api_url", DEFAULT_API_URL), disabled=True)
    st.slider("Stored threshold", 0.05, 0.95, float(st.session_state.get("ui_threshold", 0.5)), step=0.01, disabled=True)

    st.subheader("Example request body")
    st.code(json.dumps(get_default_transaction(), indent=2), language="json")

    st.subheader("Quick actions")
    if st.button("Reset stored history"):
        st.session_state["transaction_history"] = []
        st.success("Transaction history cleared")


def main() -> None:
    """Application entry point."""
    init_state()
    render_header()
    sidebar = render_sidebar()

    if sidebar["health_result"]:
        st.sidebar.success(
            f"Model loaded: {sidebar['health_result'].get('model_loaded', False)}"
        )
    if sidebar["health_error"]:
        st.sidebar.error(sidebar["health_error"])

    page = sidebar["page"]
    api_url = sidebar["api_url"]
    threshold = sidebar["threshold"]

    if page == "Dashboard":
        render_dashboard(api_url)
    elif page == "Predict":
        render_prediction_page(api_url, threshold, sidebar["load_sample"])
    elif page == "Batch Upload":
        render_batch_upload_page(api_url)
    elif page == "Monitoring":
        render_monitoring_page(api_url)
    elif page == "Analytics":
        render_analytics_page()
    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()