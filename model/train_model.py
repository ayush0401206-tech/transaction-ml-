"""Train and compare four fraud classifiers. Run from the project root:
python model/train_model.py
"""
from pathlib import Path
import json
import time
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "dataset" / "finance_fraud_data.csv"

# Load data and exclude the confusing IP risk feature if present.
data = pd.read_csv(DATA_PATH)
if "IP_Risk_Score" in data.columns:
    data = data.drop(columns=["IP_Risk_Score"])

ts = pd.to_datetime(data["Timestamp"], errors="coerce")
if ts.isna().any():
    raise ValueError("Timestamp contains values that could not be parsed.")

X = pd.DataFrame({
    "Transaction_Hour": ts.dt.hour,
    "Day_of_Week": ts.dt.dayofweek,
    "Month": ts.dt.month,
    "Amount": data["Amount"],
    "Merchant_Category": data["Merchant_Category"],
    "Distance_from_Home": data["Distance_from_Home"],
    "Device_Type": data["Device_Type"],
    "Avg_Spending_Habit": data["Avg_Spending_Habit"],
    "Is_Weekend": data["Is_Weekend"],
    "Is_Night_Transaction": data["Is_Night_Transaction"],
})
y = data["Is_Fraud"].astype(int)
num = ["Transaction_Hour", "Day_of_Week", "Month", "Amount", "Distance_from_Home", "Avg_Spending_Habit", "Is_Weekend", "Is_Night_Transaction"]
cat = ["Merchant_Category", "Device_Type"]

# All models use the same split for a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression benefits from scaled numeric features. Tree models use passthrough numeric features.
def make_preprocessor(scale_numeric=False):
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat),
        ("numeric", numeric_transform, num),
    ])

models = {
    "Logistic Regression": Pipeline([
        ("preprocessing", make_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("preprocessing", make_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocessing", make_preprocessor()),
        ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ]),
    "AdaBoost": Pipeline([
        ("preprocessing", make_preprocessor()),
        ("classifier", AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
    ]),
}

results = {}
for display_name, pipeline in models.items():
    start = time.perf_counter()
    if display_name in ("Gradient Boosting", "AdaBoost"):
        # Give rare fraud examples more weight during training.
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    else:
        pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    key = display_name.lower().replace(" ", "_")
    joblib.dump(pipeline, MODEL_DIR / f"{key}_model.pkl")
    results[display_name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
        "training_seconds": round(float(training_seconds), 4),
        "model_file": f"{key}_model.pkl",
    }
    print(f"{display_name:20s} | Accuracy {accuracy:.2%} | Precision {precision:.2%} | Recall {recall:.2%} | F1 {f1:.2%} | Train {training_seconds:.2f}s")

best_model = max(results, key=lambda name: results[name]["f1_score"])
# Preserve the filename used by the existing prediction page.
joblib.dump(models["Random Forest"], MODEL_DIR / "fraud_model.pkl")
comparison = {
    "dataset_rows": int(len(data)),
    "columns_after_removal": int(len(data.columns)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "test_size": 0.2,
    "random_state": 42,
    "best_model_by_f1": best_model,
    "models": results,
    "metric_note": "Metrics are calculated on the same stratified 80/20 held-out test split. They are dataset-specific and do not guarantee real-world performance.",
    "target_note": "Class 0 is treated as Normal and class 1 as Suspicious, assuming the dataset labels use this mapping.",
}
with open(MODEL_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

# Keep the existing dashboard's Random Forest metrics file compatible.
rf = results["Random Forest"]
rf_metrics = {
    "rows": int(len(data)),
    "columns_after_removal": int(len(data.columns)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "accuracy": rf["accuracy"],
    "precision": rf["precision"],
    "recall": rf["recall"],
    "f1_score": rf["f1_score"],
    "confusion_matrix": rf["confusion_matrix"],
    "classification_report": rf["classification_report"],
    "note": comparison["metric_note"],
}
with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(rf_metrics, f, indent=2)
print(f"\nBest model by F1 score: {best_model}")
print("Saved models and model_comparison.json")
