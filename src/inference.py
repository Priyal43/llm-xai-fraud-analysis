import pandas as pd
import joblib
import os
from src.config import X_TEST_PATH

def run_inference(model):
    print("\nRunning inference...")
    X_test = pd.read_csv(X_TEST_PATH)

    y_scores = model.predict_proba(X_test)[:, 1]
    
    results = X_test.copy()
    results['fraud_probability'] = y_scores
    top_suspicious = results.sort_values(by='fraud_probability', ascending=False).head(5)
    
    print("Top 5 suspicious transactions:")
    print(top_suspicious[['fraud_probability']].head(5))
    
    return top_suspicious

if __name__ == "__main__":
    print("Use main.py to run the full pipeline.")
