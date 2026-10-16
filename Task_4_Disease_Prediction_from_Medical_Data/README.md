# Task 4 — Disease Prediction from Medical Data

Streamlit project for inspecting structured medical datasets and training classification models.

## Included
- CSV and selected UCI `.data` file loading
- Dataset preview, missing-value counts, target-class distribution
- Preprocessing: median/mode imputation, numeric scaling, categorical one-hot encoding
- Random Forest, Logistic Regression, SVM, Extra Trees, Gradient Boosting
- Accuracy, weighted precision/recall/F1, classification report and confusion matrix
- Saves evaluation metrics to `results/`

## Windows setup
```cmd
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Copy datasets from your Downloads\Data folder into this project's `data` folder before launching.

## Important
Always verify the correct target column against the original dataset documentation. Several `.data` files are headerless and may need specific column mappings. This prototype is for educational research only and must not be used for diagnosis or treatment. Keep medical datasets out of GitHub.
