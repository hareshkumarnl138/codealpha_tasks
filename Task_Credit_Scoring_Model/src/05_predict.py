"""Predict default risk for a CSV of customer records using the saved best model."""
import argparse
import pandas as pd
from joblib import load
from src.common import MODELS, TARGET, ID_COLUMN, setup_logging

def main():
    parser = argparse.ArgumentParser(description="Score customer records for likely next-month payment default.")
    parser.add_argument("--input", required=True, help="Path to input CSV containing the model feature columns.")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path (default: predictions.csv).")
    args = parser.parse_args()

    logger = setup_logging("prediction")
    model_path = MODELS / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Best model not found. Run training first: python -m src.03_train_models")

    data = pd.read_csv(args.input)
    X = data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    model = load(model_path)
    probabilities = model.predict_proba(X)[:, 1]
    labels = (probabilities >= 0.5).astype(int)
    output = data.copy()
    output["predicted_default"] = labels
    output["default_probability"] = probabilities
    output["risk_band"] = pd.cut(
        probabilities, bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Lower", "Moderate", "Higher"], include_lowest=True
    ).astype(str)
    output.to_csv(args.output, index=False)
    logger.info("Saved %s predictions to %s", len(output), args.output)
    print(output[["predicted_default", "default_probability", "risk_band"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
