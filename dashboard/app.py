import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib

from src.config import X_TEST_PATH, MODELS_DIR
from llm_layer.generate_report import generate_explanation, inverse_transform_features, decode_category
from src.shap import get_top_pca_features
from src.mappings import get_label_map

# Dynamically loaded maps from encoders
GENDER_MAP = get_label_map("gender")
CITY_MAP = get_label_map("city")
MERCHANT_MAP = get_label_map("merchant")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection & Explanation")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv(X_TEST_PATH)

def load_model():
    return joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))

X_test = load_data()
model = load_model()

# Predict fraud probabilities
X_test['fraud_probability'] = model.predict_proba(X_test)[:, 1]

# Sidebar filters
st.sidebar.header("Filter Transactions")
fraud_thresh = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)
top_n = st.sidebar.slider("Top N Risky Transactions", 5, 100, 10)

filtered = X_test[X_test['fraud_probability'] >= fraud_thresh]

#Decoded Table
def decode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    decoded_rows = []
    for _, row in df.iterrows():
        amt, hour, age = inverse_transform_features(row)
        decoded_rows.append({
            "Amount ($)": round(amt, 2),
            "Transaction Hour": int(hour),
            "User Age": int(age),
            "Gender": GENDER_MAP.get(row["gender"], "Unknown"),
            "City": CITY_MAP.get(row["city"], f"City-{row['city']}"),
            "Category": decode_category(row["category"]),
            "Merchant": MERCHANT_MAP.get(row["merchant"], f"Merchant-{row['merchant']}"),
            "Fraud Probability": round(row["fraud_probability"], 4),
            "Index": row.name
        })
    return pd.DataFrame(decoded_rows).set_index("Index")

decoded_df = decode_dataframe(filtered.sort_values(by='fraud_probability', ascending=False).head(top_n))

st.subheader("Flagged Transactions")
st.dataframe(decoded_df, use_container_width=True)

#formatted UI view 
def format_transaction_for_ui(tx_dict: dict) -> dict:
    row = pd.Series(tx_dict)
    amt_orig, trans_hour_orig, age_orig = inverse_transform_features(row)
    category = decode_category(tx_dict.get("category", "unknown"))
    pca_summary = get_top_pca_features(model, row)

    return {
        "amount": round(amt_orig, 2),
        "transaction_hour": int(trans_hour_orig),
        "age": int(age_orig),
        "category": category,
        "gender": GENDER_MAP.get(tx_dict.get("gender"), "Unknown"),
        "city": CITY_MAP.get(tx_dict.get("city"), f"City-{tx_dict.get('city')}"),
        "merchant": MERCHANT_MAP.get(tx_dict.get("merchant"), f"Merchant-{tx_dict.get('merchant')}"),
        "pca_summary": pca_summary,
        "raw_features": tx_dict
    }

# Select a transaction to ask the LLM
st.subheader("AI-Powered Explanation")
tx_id = st.selectbox("Select a transaction index to explain", filtered.index.tolist())

if st.button("Generate Explanation"):
    selected_tx = filtered.loc[tx_id].to_dict()
    with st.spinner("Generating LLM explanation..."):
        explanation = generate_explanation(selected_tx)
        readable_tx = format_transaction_for_ui(selected_tx)

    st.success("Explanation ready:")
    st.markdown(f"> {explanation}")
