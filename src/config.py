import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Raw files
TRAIN_RAW_PATH = os.path.join(RAW_DIR, 'fraudTrain.csv')
TEST_RAW_PATH = os.path.join(RAW_DIR, 'fraudTest.csv')

# Processed files
X_TRAIN_PATH = os.path.join(PROCESSED_DIR, 'X_train.csv')
X_TEST_PATH = os.path.join(PROCESSED_DIR, 'X_test.csv')
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, 'y_train.csv')
Y_TEST_PATH = os.path.join(PROCESSED_DIR, 'y_test.csv')

# Models
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Preprocessing
USE_SMOTE = False 
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model
MODEL_TYPE = 'xgboost'
