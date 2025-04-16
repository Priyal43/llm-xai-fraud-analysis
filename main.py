from src.preprocess import preprocess_and_save_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.inference import run_inference

def main():
    print("Starting Fraud Detection Pipeline...\n")
    preprocess_and_save_data()
    model = train_model()
    evaluate_model(model)
    run_inference(model)
    print("\n All steps completed successfully.")

if __name__ == "__main__":
    main()
