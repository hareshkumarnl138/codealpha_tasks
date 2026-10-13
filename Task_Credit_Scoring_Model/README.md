# Credit Scoring Model — CodeAlpha Task 1

A reproducible machine-learning project that estimates whether a credit-card customer will default on payment next month. It compares Logistic Regression, Decision Tree, and Random Forest using a stratified hold-out test set.

> **Dataset note:** The supplied UCI dataset contains credit limit, customer profile, repayment status, bill amounts, payment amounts, and the target `default payment next month`. It does **not** contain an income field. Do not describe income as a model input.

## Project goals

- Validate and clean the source dataset.
- Explore class balance and relationships in the data.
- Train three classification baselines using the same train/test split.
- Compare Accuracy, Precision, Recall, F1-score, ROC-AUC, and Average Precision.
- Save the best model and reusable prediction script.

## Project structure

```text
Task_Credit_Scoring_Model/
├── data/
│   ├── raw/
│   │   └── default of credit card clients.xls
│   └── processed/
├── models/
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── common.py
│   ├── 01_data_preprocessing.py
│   ├── 02_eda.py
│   ├── 03_train_models.py
│   ├── 04_evaluate_models.py
│   └── 05_predict.py
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or newer (64-bit recommended)
- Internet access for installing Python packages
- The provided `.xls` dataset in `data/raw/`

## Run on Windows

Open a terminal in this project folder.

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run preprocessing

```bash
python -m src.01_data_preprocessing
```

Creates `data/processed/credit_default_clean.csv` and `results/tables/data_quality_report.json`.

### 4. Generate exploratory analysis

```bash
python -m src.02_eda
```

Creates charts in `results/figures/` and tables in `results/tables/`.

### 5. Train and compare models

```bash
python -m src.03_train_models
```

Creates fitted pipelines in `models/`, evaluation tables in `results/tables/`, and plots in `results/figures/`.

### 6. Display evaluation results

```bash
python -m src.04_evaluate_models
```

Use the actual values generated in `results/tables/model_comparison.csv` in your report. Metrics can differ slightly if you change the data or training setup.

### 7. Predict from a new CSV

The input CSV must contain the same feature columns used for training. The `ID` and target column are optional and ignored if supplied.

```bash
python -m src.05_predict --input path\\to\\new_customers.csv --output predictions.csv
```

## Methodology

1. **Target:** `default_payment` (source label: `default payment next month`); `1` means default and `0` means no default.
2. **Identifier:** `ID` is excluded because it identifies a row rather than describing credit risk.
3. **Categorical fields:** `SEX`, `EDUCATION`, and `MARRIAGE` are one-hot encoded.
4. **Numeric fields:** missing values are median-imputed and numeric values are standardized inside the training pipeline.
5. **Data split:** 80% train / 20% test, stratified, random seed 42.
6. **Models:** Logistic Regression, Decision Tree, Random Forest.
7. **Model selection:** highest ROC-AUC, with F1-score as a tie-breaker. The comparison table shows all metrics.
8. **Reproducibility:** the random seed is fixed; fitted preprocessing and estimator are saved together in a scikit-learn pipeline.

## Outputs

- `results/tables/data_quality_report.json`
- `results/tables/descriptive_statistics.csv`
- `results/tables/model_comparison.csv`
- `results/tables/*_classification_report.csv`
- `results/figures/target_distribution.png`
- `results/figures/correlation_heatmap.png`
- `results/figures/model_comparison.png`
- `results/figures/best_model_roc_curve.png`
- `results/figures/best_model_precision_recall_curve.png`
- `models/best_model.joblib`
- `models/model_metadata.json`

## Responsible-use note

This is an educational prototype, not a production lending decision system. Credit decisions are high impact. Before real use, assess fairness and disparate impact, validate against temporal/out-of-sample data, calibrate probabilities, document limitations, secure customer data, and require appropriate human/legal review. A probability estimate must not be treated as a definitive statement about a person.

## Dataset attribution

Default of Credit Card Clients dataset, commonly distributed through the UCI Machine Learning Repository. Retain the original dataset's license and attribution terms when redistributing the dataset.
