import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import joblib
from src.config import X_TRAIN_PATH, Y_TRAIN_PATH, MODELS_DIR, RANDOM_STATE

def train_model():
    print("\n📦 Loading training data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()

    # Calculate class imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=10,
        reg_lambda=20,
        gamma=1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    
    # Feature importance plot
    print("Plotting top feature importances...")
    xgb.plot_importance(model, max_num_features=10)
    plt.tight_layout()
    plt.show()

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model

if __name__ == "__main__":
    train_model()
