"""Validate the source dataset and write clean, reproducible CSV outputs."""
import json
from src.common import load_raw_data, PROCESSED_DATA, TABLES, setup_logging, TARGET, ID_COLUMN

def main():
    logger = setup_logging("preprocessing")
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    before_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    missing_by_column = df.isna().sum().astype(int).to_dict()

    # Do not silently impute values here: model pipelines perform train-fold imputation.
    # Remove exact duplicate records if present, preserving the first occurrence.
    df = df.drop_duplicates().copy()

    # Basic schema checks.
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target contains values other than 0 and 1.")

    output_path = PROCESSED_DATA / "credit_default_clean.csv"
    df.to_csv(output_path, index=False)

    summary = {
        "source_file": "data/raw/default of credit card clients.xls",
        "rows_before_deduplication": before_rows,
        "duplicate_rows_removed": duplicate_rows,
        "rows_after_deduplication": len(df),
        "columns": len(df.columns),
        "missing_values_before_deduplication": missing_by_column,
        "target_column": TARGET,
        "target_counts": {str(int(k)): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "identifier_excluded_from_model": ID_COLUMN,
        "cleaned_csv": "data/processed/credit_default_clean.csv",
    }
    with open(TABLES / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Loaded %s rows and %s columns.", before_rows, len(df.columns))
    logger.info("Removed %s exact duplicate rows.", duplicate_rows)
    logger.info("Target distribution: %s", summary["target_counts"])
    logger.info("Saved cleaned dataset to %s", output_path)
    logger.info("Saved data quality report to %s", TABLES / "data_quality_report.json")

if __name__ == "__main__":
    main()
