import streamlit as st
import pickle
import pandas as pd
import os

import os
import pickle
import streamlit as st

# — Load Model & Preprocessor —
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "model", "fraud_xgb_model.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_preprocessor():
    with open(os.path.join(BASE_DIR, "model", "preprocessor.pkl"), "rb") as f:
        return pickle.load(f)

model = load_model()
preprocessor = load_preprocessor()

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection App", page_icon="🔍", layout="wide")

# ── Header ────────────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection System")
st.markdown("Fill in the transaction details and click **Predict** to check if the transaction is fraudulent.")
st.divider()

# ── Input Form ────────────────────────────────────────────────────
st.subheader("📋 Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    transaction_amount = st.number_input(
        "💰 Transaction Amount (₹)",
        min_value=0.0, max_value=500000.0,
        value=5000.0, step=100.0
    )
    time_of_transaction = st.slider(
        "🕐 Time of Transaction (Hour)",
        min_value=0, max_value=23, value=12
    )
    previous_fraudulent_transactions = st.number_input(
        "⚠️ Previous Fraudulent Transactions",
        min_value=0, max_value=20, value=0, step=1
    )

with col2:
    account_age = st.number_input(
        "📅 Account Age (months)",
        min_value=0, max_value=600, value=24, step=1
    )
    number_of_transactions_last_24h = st.number_input(
        "🔁 Transactions in Last 24 Hours",
        min_value=0, max_value=100, value=3, step=1
    )
    device_used = st.selectbox(
        "💻 Device Used",
        ["Mobile", "Desktop", "Unknown"]
    )

with col3:
    location = st.selectbox(
        "📍 Location",
        ["Chicago", "Seattle", "New York", "Los Angeles", "Houston", "Phoenix"]
    )
    payment_method = st.selectbox(
        "💳 Payment Method",
        ["UPI", "Net Banking", "Debit Card", "Credit Card", "Invalid"]
    )
    transaction_type = st.selectbox(
        "🔄 Transaction Type",
        ["Online Purchase", "POS Payment", "Bill Payment", "Fund Transfer"]
    )

st.divider()

# ── Predict ───────────────────────────────────────────────────────
if st.button("🔎 Predict Transaction", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        "device_used":                      device_used,
        "location":                         location,
        "payment_method":                   payment_method,
        "transaction_type":                 transaction_type,
        "transaction_amount":               transaction_amount,
        "time_of_transaction":              time_of_transaction,
        "previous_fraudulent_transactions": previous_fraudulent_transactions,
        "account_age":                      account_age,
        "number_of_transactions_last_24h":  number_of_transactions_last_24h,
    }])

    processed   = preprocessor.transform(input_data)
    prediction  = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    st.subheader("📊 Prediction Result")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if prediction == 1:
            st.error("🚨 FRAUDULENT")
        else:
            st.success("✅ GENUINE")

    with col_b:
        st.metric("Fraud Probability", f"{probability * 100:.2f}%")

    with col_c:
        st.metric("Genuine Probability", f"{(1 - probability) * 100:.2f}%")

    st.progress(float(probability), text=f"Fraud Risk: {probability*100:.2f}%")

    # ── Risk Level ────────────────────────────────────────────────
    st.subheader("🧭 Risk Level")
    if probability < 0.3:
        st.success("🟢 LOW RISK — Transaction looks safe.")
    elif probability < 0.6:
        st.warning("🟡 MEDIUM RISK — Transaction needs review.")
    else:
        st.error("🔴 HIGH RISK — Likely fraudulent. Block immediately.")

    # ── Input Summary ─────────────────────────────────────────────
    with st.expander("📄 View Input Summary"):
        st.dataframe(input_data, use_container_width=True)