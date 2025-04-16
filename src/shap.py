import shap
import pandas as pd
from src.config import X_TRAIN_PATH

def get_top_pca_features(model, transaction: pd.Series, top_n=3) -> str:
    """
    Returns a summary string of the top contributing PCA features for a given transaction.
    Includes a fallback if 'tree_path_dependent' perturbation fails.
    """
    df_train = pd.read_csv(X_TRAIN_PATH, nrows=1)
    expected_columns = df_train.columns.tolist()

    tx_subset = transaction.reindex(expected_columns).fillna(0)
    X = tx_subset.values.reshape(1, -1)
    
    try:
        explainer = shap.Explainer(model, df_train, feature_perturbation='tree_path_dependent')
        shap_values = explainer(X)
    except Exception as e:
        print(f"[SHAP Warning] Falling back to 'interventional' mode due to: {e}")
        explainer = shap.Explainer(model, df_train, feature_perturbation='interventional')
        shap_values = explainer(X)

    # Extract SHAP values
    shap_array = shap_values.values[0]
    shap_series = pd.Series(shap_array, index=expected_columns)

    # Return top contributing features
    top_features = shap_series.abs().nlargest(top_n).index.tolist()
    return "High SHAP values in " + ", ".join(top_features)
