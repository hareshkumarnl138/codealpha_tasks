from pathlib import Path
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="DLI HealthRisk AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {background:#f4f7fb;color:#1f2937}
[data-testid="stAppViewContainer"] {background:#f4f7fb}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {color:#1f2937}
.hero {padding:1.4rem 1.6rem;border-radius:18px;color:white;
background:linear-gradient(115deg,#14213d,#0077b6);margin-bottom:1rem}
.hero h1 {color:white!important;margin:0}
.hero p {color:#eaf4ff!important;margin:.4rem 0 0}
div[data-testid="stMetric"] {background:white;border:1px solid #dbe3ef;
padding:.8rem;border-radius:12px}
div[data-testid="stMetric"] label,div[data-testid="stMetricValue"] {color:#1f2937!important}
[data-testid="stSidebar"] {background:#242630}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {color:white}
</style>
""", unsafe_allow_html=True)


def read_table(source, filename=""):
    """Read CSV, DATA, or TXT data with common delimiters."""
    if hasattr(source, "getvalue"):
        from io import BytesIO
        stream = BytesIO(source.getvalue())
        filename = getattr(source, "name", filename).lower()
    else:
        stream = source
        filename = str(source).lower()

    if filename.endswith((".data", ".txt")):
        df = pd.read_csv(stream, sep=None, engine="python", header=None)
    else:
        df = pd.read_csv(stream, sep=None, engine="python")

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace({"": np.nan, "?": np.nan, "NA": np.nan,
                                   "N/A": np.nan, "null": np.nan})
    return df


def norm(text):
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def guess_target(df, preset):
    hints = {
        "Diabetes (3-class)": ["Diabetes_012"],
        "Diabetes (binary)": ["Diabetes_binary"],
        "Heart failure": ["DEATH_EVENT"],
        "Maternal health risk": ["RiskLevel", "Risk Level"],
        "Indian liver patient": ["Dataset", "class", "target"],
        "Thyroid disease": ["Class", "target", "Diagnosis"],
        "Auto-detect": ["target", "label", "class", "outcome", "diagnosis"],
    }.get(preset, [])
    cols = {norm(c): c for c in df.columns}
    for hint in hints:
        if norm(hint) in cols:
            return cols[norm(hint)]
    for hint in ["target", "label", "class", "outcome", "diagnosis", "disease"]:
        if norm(hint) in cols:
            return cols[norm(hint)]
    return df.columns[-1]


def clean_data(df, target):
    df = df.copy().drop_duplicates()
    df = df.dropna(axis=1, how="all")
    if target not in df.columns:
        raise ValueError("The target column does not exist in the dataset.")
    df = df.dropna(subset=[target])
    if df.empty:
        raise ValueError("No rows remain after removing rows without target labels.")
    if df[target].nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")
    return df


def make_pipeline(algorithm, numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", cat_pipe, categorical_cols))
    prep = ColumnTransformer(transformers=transformers)

    if algorithm == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                     n_jobs=-1, class_weight="balanced")
    elif algorithm == "Extra Trees":
        clf = ExtraTreesClassifier(n_estimators=100, random_state=42,
                                   n_jobs=-1, class_weight="balanced")
    elif algorithm == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=42)
    else:
        clf = LinearSVC(max_iter=3000, class_weight="balanced",
                        random_state=42, dual="auto")
    return Pipeline([("preprocess", prep), ("model", clf)])


st.sidebar.title("Dataset & training")
preset = st.sidebar.selectbox("Disease preset", [
    "Diabetes (3-class)", "Diabetes (binary)", "Heart failure",
    "Maternal health risk", "Indian liver patient", "Thyroid disease",
    "Auto-detect"
])

files = sorted(p for p in DATA_DIR.iterdir()
               if p.is_file() and p.suffix.lower() in {".csv", ".data", ".txt"})
uploaded = st.sidebar.file_uploader("Or upload CSV / DATA / TXT", type=["csv", "data", "txt"])

if files:
    preferred = 0
    preset_hints = {
        "Diabetes (3-class)": "diabetes_012_health_indicators_br",
        "Diabetes (binary)": "diabetes_binary_health_indicators_br",
        "Heart failure": "heart_failure_clinical_records",
        "Maternal health risk": "maternal health risk",
        "Indian liver patient": "ilpd",
        "Thyroid disease": "thyroid_diff",
    }
    hint = preset_hints.get(preset, "")
    if hint:
        matches = [i for i, p in enumerate(files) if hint in p.name.lower()]
        if matches:
            preferred = matches[0]
    selected_file = st.sidebar.selectbox("Dataset file", [p.name for p in files],
                                         index=preferred)
else:
    selected_file = None
    st.sidebar.info(f"Place dataset files inside: {DATA_DIR}")

algorithm = st.sidebar.selectbox("Algorithm", [
    "Random Forest", "Extra Trees", "Logistic Regression", "Linear SVM"
])
test_size = st.sidebar.slider("Test set (%)", 10, 40, 20, 5)
row_limit = st.sidebar.selectbox("Training row limit", [
    "Use all rows", "50,000 rows", "100,000 rows"
])
st.sidebar.caption("Educational prototype only — not for clinical diagnosis.")

st.markdown("""
<div class="hero">
<h1>🩺 DLI HealthRisk AI</h1>
<p>Explore medical datasets, preprocess data, and train disease-risk classification models.</p>
</div>
""", unsafe_allow_html=True)

st.title("1. Dataset overview")
try:
    if uploaded is not None:
        raw_df = read_table(uploaded, uploaded.name)
        source_name = uploaded.name
    elif selected_file:
        raw_df = read_table(DATA_DIR / selected_file, selected_file)
        source_name = selected_file
    else:
        st.warning(f"Add a CSV/DATA/TXT file to {DATA_DIR}, or upload one in the sidebar.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the dataset: {type(exc).__name__}: {exc}")
    st.stop()

if raw_df.empty or len(raw_df.columns) < 2:
    st.error("The file must contain data rows and at least two columns.")
    st.stop()

st.caption(f"Dataset: **{source_name}**")
a, b, c, d = st.columns(4)
a.metric("Rows", f"{len(raw_df):,}")
b.metric("Columns", f"{len(raw_df.columns):,}")
c.metric("Missing cells", f"{int(raw_df.isna().sum().sum()):,}")
d.metric("Duplicate rows", f"{int(raw_df.duplicated().sum()):,}")

with st.expander("Preview raw data", expanded=True):
    st.dataframe(raw_df.head(15), use_container_width=True)

with st.expander("Column types and missing values"):
    st.dataframe(pd.DataFrame({
        "Column": raw_df.columns,
        "Type": [str(raw_df[col].dtype) for col in raw_df.columns],
        "Missing": [int(raw_df[col].isna().sum()) for col in raw_df.columns],
        "Unique": [int(raw_df[col].nunique(dropna=True)) for col in raw_df.columns],
    }), use_container_width=True)

target_guess = guess_target(raw_df, preset)
target_index = list(raw_df.columns).index(target_guess)
st.header("2. Preprocess the dataset")
target = st.selectbox("Target column (what the model predicts)",
                      list(raw_df.columns), index=target_index)
try:
    cleaned = clean_data(raw_df, target)
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()

x1, x2, x3, x4 = st.columns(4)
x1.metric("Rows after cleaning", f"{len(cleaned):,}")
x2.metric("Duplicates removed", f"{int(raw_df.duplicated().sum()):,}")
x3.metric("Target classes", f"{cleaned[target].nunique():,}")
x4.metric("Remaining missing cells", f"{int(cleaned.isna().sum().sum()):,}")

st.markdown("""
**Preprocessing applied**
- Removed duplicate rows and rows with missing target labels.
- Converted blank strings and `?` entries to missing values.
- Imputed numeric features with the median and categorical features with the most frequent value during training.
- Standardized numeric features and one-hot encoded categorical features inside the training pipeline.
- Used a stratified train/test split when possible.
""")
with st.expander("Preview cleaned data"):
    st.dataframe(cleaned.head(15), use_container_width=True)
with st.expander("Target class distribution"):
    st.dataframe(cleaned[target].value_counts().rename_axis("Class").reset_index(name="Rows"),
                 use_container_width=True)

st.download_button("Download cleaned dataset", cleaned.to_csv(index=False).encode("utf-8"),
                   "cleaned_dataset.csv", "text/csv")

st.header("3. Train and evaluate")
st.write(f"Algorithm: **{algorithm}** · Test set: **{test_size}%**")
if st.button("Train model", type="primary", use_container_width=True):
    try:
        data = cleaned.copy()
        if row_limit != "Use all rows":
            limit = 50000 if row_limit == "50,000 rows" else 100000
            if len(data) > limit:
                try:
                    data, _ = train_test_split(data, train_size=limit, random_state=42,
                                               stratify=data[target])
                except ValueError:
                    data = data.sample(n=limit, random_state=42)
        X = data.drop(columns=[target]).copy()
        y_text = data[target].astype(str)
        # Convert text columns that are almost entirely numeric.
        for col in X.select_dtypes(include=["object"]).columns:
            converted = pd.to_numeric(X[col], errors="coerce")
            non_missing = X[col].notna().sum()
            if non_missing and converted.notna().sum() / non_missing >= 0.9:
                X[col] = converted
        constant_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
        X = X.drop(columns=constant_cols)
        if X.empty:
            st.error("No usable feature columns remain.")
            st.stop()

        encoder = LabelEncoder()
        y = encoder.fit_transform(y_text)
        counts = pd.Series(y).value_counts()
        stratify = y if len(counts) > 1 and counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42, stratify=stratify
        )
        numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [col for col in X_train.columns if col not in numeric_cols]
        model = make_pipeline(algorithm, numeric_cols, categorical_cols)

        with st.spinner("Training model. Large datasets may take a while..."):
            started = time.time()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            elapsed = time.time() - started

        results = {
            "dataset": source_name,
            "algorithm": algorithm,
            "target": target,
            "rows_used": int(len(data)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "training_seconds": round(elapsed, 2),
        }
        st.session_state["dli_model"] = model
        st.session_state["dli_features"] = list(X.columns)
        st.session_state["dli_numeric"] = numeric_cols
        st.session_state["dli_labels"] = [str(v) for v in encoder.classes_]
        st.session_state["dli_results"] = results
        st.session_state["dli_y_test"] = y_test
        st.session_state["dli_pred"] = pred
        st.session_state["dli_class_names"] = [str(v) for v in encoder.classes_]
        st.success(f"Training completed in {elapsed:.1f} seconds.")
    except Exception as exc:
        st.error(f"Training failed: {type(exc).__name__}: {exc}")
        st.info("Verify the target column. For large datasets, choose a row limit or try Random Forest/Extra Trees.")

if "dli_results" in st.session_state:
    r = st.session_state["dli_results"]
    st.subheader("Evaluation metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{r['accuracy']:.4f}")
    m2.metric("Weighted precision", f"{r['precision_weighted']:.4f}")
    m3.metric("Weighted recall", f"{r['recall_weighted']:.4f}")
    m4.metric("Weighted F1", f"{r['f1_weighted']:.4f}")
    st.caption(f"Training time: {r['training_seconds']} seconds · Train rows: {r['train_rows']:,} · Test rows: {r['test_rows']:,}")

    y_test = st.session_state["dli_y_test"]
    pred = st.session_state["dli_pred"]
    class_names = st.session_state["dli_class_names"]
    cm = confusion_matrix(y_test, pred, labels=list(range(len(class_names))))
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * .8), max(4, len(class_names) * .65)))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    if len(class_names) <= 12:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(y_test, pred, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(4)
    st.subheader("Classification report")
    st.dataframe(report_df, use_container_width=True)
    st.download_button("Download classification report",
                       report_df.to_csv().encode("utf-8"),
                       "classification_report.csv", "text/csv")
    st.download_button("Download metrics JSON",
                       json.dumps(r, indent=2).encode("utf-8"),
                       "evaluation_summary.json", "application/json")

    st.subheader("Predict a new record")
    st.write("Values default to typical values from the dataset; change them before predicting.")
    source_features = cleaned.drop(columns=[target])
    features = st.session_state["dli_features"]
    with st.form("predict_form"):
        values = {}
        cols = st.columns(3)
        for i, col in enumerate(features):
            series = source_features[col]
            with cols[i % 3]:
                if col in st.session_state["dli_numeric"]:
                    med = pd.to_numeric(series, errors="coerce").median()
                    default = float(med) if pd.notna(med) else 0.0
                    values[col] = st.number_input(str(col), value=default, key=f"new_{col}")
                else:
                    opts = series.dropna().astype(str).value_counts().index.tolist() or ["Unknown"]
                    values[col] = st.selectbox(str(col), opts, key=f"new_{col}")
        submit = st.form_submit_button("Predict class")
    if submit:
        try:
            new_row = pd.DataFrame([values], columns=features)
            for col in st.session_state["dli_numeric"]:
                new_row[col] = pd.to_numeric(new_row[col], errors="coerce")
            prediction = int(st.session_state["dli_model"].predict(new_row)[0])
            label = st.session_state["dli_labels"][prediction]
            st.success(f"Predicted class: **{label}**")
            st.caption("Educational output only; not a medical diagnosis.")
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

st.divider()
st.caption("DLI HealthRisk AI is an educational machine-learning prototype. Do not use it to diagnose or treat medical conditions.")
