import pandas as pd
import os
import joblib

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(BASE_DIR, "models/flood_model_xgb.pkl")

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data/processed/realtime_dataset.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data/processed/realtime_predictions.csv"
)


def main():

    print("📥 Loading realtime data...")
    df = pd.read_csv(INPUT_FILE, on_bad_lines="skip")

    print("📦 Loading model...")
    model = joblib.load(MODEL_PATH)

    # select numeric features
    X = df.select_dtypes(include=["number"])

    print("🤖 Predicting flood...")
    df["flood_prediction"] = model.predict(X)

    # probability (VERY IMPORTANT)
    df["flood_probability"] = model.predict_proba(X)[:, 1]

    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Predictions saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
