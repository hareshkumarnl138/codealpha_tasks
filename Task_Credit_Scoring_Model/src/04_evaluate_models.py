"""Print a concise evaluation summary from the saved model comparison."""
from pathlib import Path
import pandas as pd
from src.common import TABLES, setup_logging

def main():
    logger = setup_logging("evaluation")
    path = TABLES / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError("Run step 3 first: python -m src.03_train_models")
    results = pd.read_csv(path)
    print("\nCREDIT DEFAULT MODEL COMPARISON (held-out test set)\n")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    logger.info("Evaluation summary displayed.")

if __name__ == "__main__":
    main()
