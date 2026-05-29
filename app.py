import streamlit as st
import pandas as pd
import pickle
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

@st.cache_resource
def load_artifacts():
    with open(f"{BASE}/preprocessor.pkl",    "rb") as f: preprocessor = pickle.load(f)
    with open(f"{BASE}/fraud_xgb_model.pkl", "rb") as f: model        = pickle.load(f)
    with open(f"{BASE}/threshold.pkl",       "rb") as f: threshold    = pickle.load(f)
    with open(f"{BASE}/category_info.pkl",   "rb") as f: cats         = pickle.load(f)
    return preprocessor, model, threshold, cats

preprocessor, model, THRESHOLD, cats = load_artifacts()

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")
st.title("🔍 Fraud Detection System")
st.caption(f"Optimized detection threshold: {THRESHOLD:.2f} (tuned for maximum fraud recall)")

st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)
with col1:
    device_used      = st.selectbox("Device Used",       cats["device_used"])
    location         = st.selectbox("Location",          cats["location"])
    payment_method   = st.selectbox("Payment Method",    cats["payment_method"])
    transaction_type = st.selectbox("Transaction Type",  cats["transaction_type"])
with col2:
    transaction_amount  = st.number_input("Transaction Amount (₹)", min_value=0.0, value=5000.0, step=100.0)
    time_of_transaction = st.slider("Time of Transaction (Hour 0–23)", 0, 23, 12)
    previous_fraud      = st.slider("Previous Fraudulent Transactions", 0, 10, 0)
    account_age         = st.number_input("Account Age (years)", min_value=0, value=3, step=1)
    txns_24h            = st.number_input("Transactions in Last 24h", min_value=0, value=2, step=1)

if st.button("🔎 Check Transaction", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "device_used":                      device_used,
        "location":                         location,
        "payment_method":                   payment_method,
        "transaction_type":                 transaction_type,
        "transaction_amount":               transaction_amount,
        "time_of_transaction":              time_of_transaction,
        "previous_fraudulent_transactions": previous_fraud,
        "account_age":                      account_age,
        "number_of_transactions_last_24h":  txns_24h,
    }])

    processed  = preprocessor.transform(input_df)
    fraud_prob = model.predict_proba(processed)[0][1]
    is_fraud   = fraud_prob >= THRESHOLD

    st.divider()
    if is_fraud:
        st.error("🚨 **FRAUDULENT Transaction Detected!**")
    else:
        st.success("✅ **Genuine Transaction**")

    st.metric("Fraud Probability", f"{fraud_prob * 100:.1f}%")
    st.progress(float(fraud_prob), text=f"Risk Score: {fraud_prob:.3f}  |  Threshold: {THRESHOLD:.2f}")

    with st.expander("ℹ️ How is this calculated?"):
        st.write(
            f"The model assigns a fraud probability of **{fraud_prob*100:.1f}%** to this transaction. "
            f"Any score at or above **{THRESHOLD*100:.0f}%** is flagged as fraud. "
            f"This threshold was tuned on your dataset to maximize the detection of real fraud cases (Recall)."
        )