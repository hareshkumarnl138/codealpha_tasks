"""Exploratory data analysis charts and summary tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.common import load_raw_data, FIGURES, TABLES, TARGET, ID_COLUMN, setup_logging

sns.set_theme(style="whitegrid")

def main():
    logger = setup_logging("eda")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df = load_raw_data()
    model_df = df.drop(columns=[ID_COLUMN], errors="ignore")

    df.describe(include="all").transpose().to_csv(TABLES / "descriptive_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv(TABLES / "missing_values.csv")

    plt.figure(figsize=(7, 5))
    order = model_df[TARGET].value_counts().sort_index()
    sns.barplot(x=order.index.astype(str), y=order.values)
    plt.title("Target Distribution")
    plt.xlabel("Default next month (0 = No, 1 = Yes)")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(FIGURES / "target_distribution.png", dpi=160)
    plt.close()

    numeric_df = model_df.select_dtypes(include="number")
    plt.figure(figsize=(15, 12))
    corr = numeric_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Numeric Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png", dpi=160)
    plt.close()

    # Default rate by categorical customer profile fields.
    for col in [c for c in ["sex", "education", "marriage"] if c in df.columns]:
        rates = df.groupby(col, dropna=False)[TARGET].mean().reset_index()
        rates.columns = [col, "default_rate"]
        rates.to_csv(TABLES / f"default_rate_by_{col}.csv", index=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=rates, x=col, y="default_rate")
        plt.title(f"Default Rate by {col.title()}")
        plt.ylabel("Default rate")
        plt.tight_layout()
        plt.savefig(FIGURES / f"default_rate_by_{col}.png", dpi=160)
        plt.close()

    logger.info("EDA complete. Charts: %s", FIGURES)
    logger.info("EDA tables: %s", TABLES)

if __name__ == "__main__":
    main()
