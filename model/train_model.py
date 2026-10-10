"""Train the basic Random Forest model without IP_Risk_Score.
Run from the project root: python model/train_model.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "dataset" / "finance_fraud_data.csv")
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
preprocessor = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), cat), ("numeric", "passthrough", num)])
model = Pipeline([("preprocessing", preprocessor), ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "rows": int(len(data)), "columns_after_removal": int(len(data.columns)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, pred)), "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)), "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    "note": "Metrics are from a stratified 80/20 held-out test split using this dataset with IP_Risk_Score removed."
}
joblib.dump(model, ROOT / "model" / "fraud_model.pkl")
with open(ROOT / "model" / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print("Model trained and saved.")
print(f"Accuracy : {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall   : {metrics['recall']:.2%}")
print(f"F1 score : {metrics['f1_score']:.2%}")
print("Confusion matrix:")
print(metrics["confusion_matrix"])
