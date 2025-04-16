from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

from src.config import X_TEST_PATH, MODELS_DIR
from src.inference import run_inference
from src.shap import get_top_pca_features
from src.mappings import GENDER_MAP, CITY_MAP, MERCHANT_MAP 

load_dotenv()
model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
category_encoder = joblib.load(os.path.join(MODELS_DIR, 'encoder_category.pkl'))

# LLM prompt template
template = """
A user made a transaction with the following details:

- Category: {category}
- Amount: ${amount}
- Age of user: {age}
- Time of transaction: {transaction_hour}:00 hrs
- Merchant: {merchant}
- City: {city}
- Gender: {gender}
- PCA Summary (top contributing factors): {pca_summary}

Based on this information, write a clear and logical explanation on **why this transaction might be fraudulent**. Connect the information together like a story, explaining how the amount, time, user age, and category could indicate fraud. Avoid listing features separately — focus on how they interact to form a suspicious pattern.
Keep the explanation concise and to the point, ideally 1-2 paragraphs.
"""

prompt = PromptTemplate(
    input_variables=["category", "amount", "age", "transaction_hour", "merchant", "city", "gender", "pca_summary"],
    template=template
)

def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        temperature=0.2,
        model_name="llama3-8b-8192",
        groq_api_key=groq_api_key
    )

def inverse_transform_features(row: pd.Series):
    features = ['amt', 'transaction_hour', 'age']
    arr = row[features].values.reshape(1, -1)
    original = scaler.inverse_transform(arr)[0]
    return round(original[0], 2), int(round(original[1])), int(round(original[2]))

def decode_category(cat_val):
    try:
        if isinstance(cat_val, str):
            cat_val = int(float(cat_val))
        elif isinstance(cat_val, float):
            cat_val = int(cat_val)
        if 0 <= cat_val < len(category_encoder.classes_):
            return category_encoder.inverse_transform([cat_val])[0]
    except Exception as e:
        print(f"Failed to decode category: {cat_val} ({e})")
    return "unknown"

def generate_explanations(top_transactions: pd.DataFrame):
    llm = get_llm()
    for idx, row in top_transactions.iterrows():
        amt_orig, trans_hour_orig, age_orig = inverse_transform_features(row)
        category = decode_category(row.get("category", "unknown"))
        pca_summary = str(get_top_pca_features(model, row))

        gender = GENDER_MAP.get(row.get("gender"), "Unknown")
        city = CITY_MAP.get(row.get("city"), f"City-{row.get('city')}")
        merchant = MERCHANT_MAP.get(row.get("merchant"), f"Merchant-{row.get('merchant')}")

        user_input = prompt.format(
            category=category,
            amount=amt_orig,
            age=age_orig,
            transaction_hour=trans_hour_orig,
            merchant=merchant,
            city=city,
            gender=gender,
            pca_summary=pca_summary
        )
        explanation = llm([HumanMessage(content=user_input)])
        print(f"\nTransaction {idx} Explanation:\n{explanation.content}\n")

def generate_explanation(tx_dict: dict) -> str:
    llm = get_llm()
    row = pd.Series(tx_dict)
    amt_orig, trans_hour_orig, age_orig = inverse_transform_features(row)
    category = decode_category(tx_dict.get("category", "unknown"))
    pca_summary = str(get_top_pca_features(model, row))

    gender = GENDER_MAP.get(tx_dict.get("gender"), "Unknown")
    city = CITY_MAP.get(tx_dict.get("city"), f"City-{tx_dict.get('city')}")
    merchant = MERCHANT_MAP.get(tx_dict.get("merchant"), f"Merchant-{tx_dict.get('merchant')}")

    user_input = prompt.format(
        category=category,
        amount=amt_orig,
        age=age_orig,
        transaction_hour=trans_hour_orig,
        merchant=merchant,
        city=city,
        gender=gender,
        pca_summary=pca_summary
    )
    response = llm([HumanMessage(content=user_input)])
    return response.content

if __name__ == "__main__":
    top_transactions = run_inference(model)
    generate_explanations(top_transactions)
