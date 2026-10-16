# Task 4 — Disease Prediction from Medical Data

**Project:** DLI HealthRisk AI  
**Internship task:** CodeAlpha — Machine Learning  
**Application:** Streamlit web app for exploring tabular medical datasets and training disease-risk classification models.

> **Important:** This project is an educational machine-learning prototype only. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions.

## Overview

DLI HealthRisk AI provides a workflow to inspect a selected dataset, preprocess the data, train a classification model, and review its evaluation results in a browser-based interface.

### Main features

- Select a supported dataset preset or upload a CSV, DATA, or TXT file.
- Review dataset dimensions, columns, missing values, and duplicate records.
- Choose a target column for classification.
- Remove duplicate records and rows without a target label.
- Treat empty strings and `?` values as missing values.
- Impute missing numeric values with the median and categorical values with the most frequent category.
- Standardize numeric features and one-hot encode categorical features in a scikit-learn pipeline.
- Split data into training and test sets, using stratification when class counts permit.
- Train a selected classifier and review metrics and a confusion matrix.
- Inspect and download the classification report and evaluation summary.
- Enter feature values to obtain a model prediction for an individual record.

## Algorithms

The app includes classifier choices such as:

- Random Forest
- Extra Trees
- Histogram-based Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)

Training time depends on dataset size, selected algorithm, and computer resources. SVM can be slow on large datasets; use a smaller dataset or a representative sample for experimentation when necessary.

## Tech stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- Matplotlib

## Project structure

Keep the task in the repository under:

```text
Task_4_Disease_Prediction_from_Medical_Data/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/       # Local datasets; do not commit large or sensitive files
└── results/    # Generated evaluation outputs; keep local unless intentionally sharing
```

Your actual folder may contain additional helper files. Keep any files required by `app.py` in the project folder.

## Setup on Windows

Open Command Prompt in this project directory.

### 1. Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\\Scripts\\activate
```

### 2. Install dependencies

If `requirements.txt` is present:

```cmd
python -m pip install -r requirements.txt
```

Otherwise install the core packages:

```cmd
python -m pip install streamlit pandas numpy matplotlib scikit-learn
```

### 3. Run the application

```cmd
streamlit run app.py
```

Streamlit normally opens the application in your browser at `http://localhost:8501`.

## Workflow

1. Choose a dataset preset or upload a tabular data file.
2. Verify that the correct target column is selected.
3. Review the preprocessing summary.
4. Select a classifier and test-set percentage.
5. Start training and wait for evaluation to finish.
6. Review accuracy, weighted precision, weighted recall, weighted F1, the confusion matrix, and the classification report.
7. Download the available evaluation outputs for your project report.

Do not compare models using accuracy alone, especially when the target classes are imbalanced. Review precision, recall, F1-score, and the class-level report as well.

## Dataset and privacy

- Place local datasets inside the project's `data/` directory if the app is configured to read from that folder.
- Do not upload personal, identifiable, confidential, or otherwise restricted health data to a public repository.
- Avoid committing large raw datasets, virtual environments, trained model binaries, caches, or generated files unless there is a clear reason and permission to share them.
- Dataset presets and their target columns should be verified in the app before training.

## Limitations

Model performance depends on the dataset, target definition, data quality, class balance, and chosen algorithm. A successful training run does not establish clinical validity, fairness, or suitability for real-world medical decisions.

## Author

Developed as a CodeAlpha machine-learning task under the DLI HealthRisk AI project.
