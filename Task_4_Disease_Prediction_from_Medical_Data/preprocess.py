
from pathlib import Path
import pandas as pd
import numpy as np

# Project folders
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_csv(file_path):
    print("\n" + "=" * 60)
    print(f"Processing: {file_path.name}")
    print("=" * 60)

    try:
        # Load CSV
        df = pd.read_csv(file_path)

        print("Original shape:", df.shape)
        print("\nFirst five rows:")
        print(df.head())

        # Standardize column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )

        # Remove duplicate rows
        duplicate_count = df.duplicated().sum()
        df = df.drop_duplicates()

        print("\nDuplicate rows removed:", duplicate_count)

        # Convert common missing-value markers to NaN
        df = df.replace(
            ["?", "NA", "N/A", "na", "n/a", "None", ""],
            np.nan
        )

        # Convert numeric-looking text columns to numbers
        for col in df.columns:
            if df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")

                # Convert only when most non-missing values are numeric
                non_missing = df[col].notna().sum()
                if non_missing and converted.notna().sum() / non_missing >= 0.90:
                    df[col] = converted

        print("\nMissing values by column:")
        missing = df.isna().sum()
        print(missing[missing > 0] if missing.any() else "None")

        # Save a cleaned copy while preserving missing values
        # Target-specific imputation and encoding will be handled
        # after the correct disease target column is identified.
        output_path = PROCESSED_DIR / f"{file_path.stem}_cleaned.csv"
        df.to_csv(output_path, index=False)

        print("\nCleaned shape:", df.shape)
        print("Saved to:", output_path)

    except Exception as error:
        print(f"Could not process {file_path.name}: {error}")


def main():
    if not DATA_DIR.exists():
        print("ERROR: data folder was not found.")
        return

    csv_files = [
        path for path in DATA_DIR.glob("*.csv")
        if path.is_file()
    ]

    if not csv_files:
        print("No CSV files found in:", DATA_DIR)
        print("Copy your CSV datasets into the data folder first.")
        return

    print(f"Found {len(csv_files)} CSV dataset(s).")

    for file_path in csv_files:
        preprocess_csv(file_path)

    print("\nPreprocessing inspection completed.")
    print("Review the cleaned files in data/processed/.")


if __name__ == "__main__":
    main()
