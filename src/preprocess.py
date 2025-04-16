import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os
import joblib

from src.config import (
    TRAIN_RAW_PATH, X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
    PROCESSED_DIR, MODELS_DIR, TEST_SIZE, RANDOM_STATE, USE_SMOTE
)

def preprocess_and_save_data():
    df = pd.read_csv(TRAIN_RAW_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    df['trans_date_trans_time'] = df['trans_date_trans_time'].astype(str)
    df['dob'] = df['dob'].astype(str)

    # Parse datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d', errors='coerce')

    if df['trans_date_trans_time'].isna().sum() > 0 or df['dob'].isna().sum() > 0:
        print(f"Datetime parse issues — trans_time: {df['trans_date_trans_time'].isna().sum()}, dob: {df['dob'].isna().sum()}")

    #Feature Engineering
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour.fillna(df['trans_date_trans_time'].dt.hour.median())
    df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days / 365).round(2)
    df['age'] = df['age'].fillna(df['age'].median())

    #Encoding
    encoders = {}
    for col in ['gender', 'city', 'category', 'merchant']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        joblib.dump(le, os.path.join(MODELS_DIR, f'encoder_{col}.pkl'))
        print(f"Saved LabelEncoder for {col} → models/encoder_{col}.pkl")

    features = ['amt', 'transaction_hour', 'age', 'gender', 'city', 'category', 'merchant']
    X = df[features].copy()
    y = df['is_fraud']

    X.dropna(inplace=True)
    y = y.loc[X.index]

    #Scale numeric features
    scaler = StandardScaler()
    X[['amt', 'transaction_hour', 'age']] = scaler.fit_transform(X[['amt', 'transaction_hour', 'age']])
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("Saved StandardScaler → models/scaler.pkl")

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    #SMOTE
    if USE_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("SMOTE applied")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    print("Processed data saved successfully.")
    print(f"Final shapes — X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

def run_all():
    print("Running full preprocessing pipeline...")
    preprocess_and_save_data()
    print("Preprocessing complete.")

if __name__ == "__main__":
    run_all()
