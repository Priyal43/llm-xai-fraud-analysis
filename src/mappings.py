import joblib
import os
from src.config import MODELS_DIR

def load_encoder(col_name):
    return joblib.load(os.path.join(MODELS_DIR, f'encoder_{col_name}.pkl'))

def get_label_map(col_name):
    encoder = load_encoder(col_name)
    return {int(encoder.transform([label])[0]): label for label in encoder.classes_}

GENDER_MAP = get_label_map("gender")
CITY_MAP = get_label_map("city")
MERCHANT_MAP = get_label_map("merchant")
