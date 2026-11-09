    """Train, validate, select, and finally test fraud classifiers.
    Run from the project root: python model/train_model.py
    """
    from pathlib import Path
    import json
    import time
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, classification_report)
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    ROOT = Path(__file__).resolve().parents[1]
    MODEL_DIR = ROOT / "model"
    DATA_PATH = ROOT / "dataset" / "finance_fraud_data.csv"
    RANDOM_STATE = 42


    def build_features(data):
        required = {"Timestamp", "Amount", "Merchant_Category", "Distance_from_Home",
                    "Device_Type", "Avg_Spending_Habit", "Is_Fraud"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        ts = pd.to_datetime(data["Timestamp"], errors="coerce")
        if ts.isna().any():
            raise ValueError("Timestamp contains values that could not be parsed.")
        X = pd.DataFrame({
            "Transaction_Hour": ts.dt.hour,
            "Day_of_Week": ts.dt.dayofweek,
            "Month": ts.dt.month,
            "Amount": pd.to_numeric(data["Amount"], errors="coerce"),
            "Merchant_Category": data["Merchant_Category"].astype(str),
            "Distance_from_Home": pd.to_numeric(data["Distance_from_Home"], errors="coerce"),
            "Device_Type": data["Device_Type"].astype(str),
            "Avg_Spending_Habit": pd.to_numeric(data["Avg_Spending_Habit"], errors="coerce"),
            "Is_Weekend": (ts.dt.dayofweek >= 5).astype(int),
            "Is_Night_Transaction": (ts.dt.hour < 5).astype(int),
        })
        if X.isna().any().any():
            raise ValueError("Model features contain missing or non-numeric values.")
        y = pd.to_numeric(data["Is_Fraud"], errors="raise").astype(int)
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("Is_Fraud must contain only 0 and 1.")
        return X, y


    def make_preprocessor(numeric, categorical, scale=False):
        return ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler() if scale else "passthrough", numeric),
        ])


    def choose_threshold(y_true, probabilities):
        # Search validation thresholds only. Prefer a higher threshold on F1 ties.
        candidates = np.linspace(0.05, 0.95, 181)
        scores = [(f1_score(y_true, probabilities >= t, zero_division=0), t) for t in candidates]
        best_f1 = max(score for score, _ in scores)
        return float(max(t for score, t in scores if score == best_f1))


    def evaluate(y_true, probabilities, threshold):
        predictions = (probabilities >= threshold).astype(int)
        return {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
        }


    def main():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(DATA_PATH)
        X, y = build_features(data)
        numeric = ["Transaction_Hour", "Day_of_Week", "Month", "Amount", "Distance_from_Home",
                "Avg_Spending_Habit", "Is_Weekend", "Is_Night_Transaction"]
        categorical = ["Merchant_Category", "Device_Type"]

        # Stratified 60/20/20 split: test data remains untouched until final evaluation.
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25,
            random_state=RANDOM_STATE, stratify=y_train_val)

        models = {
            "Logistic Regression": Pipeline([
                ("preprocessing", make_preprocessor(numeric, categorical, True)),
                ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE))]),
            "Random Forest": Pipeline([
                ("preprocessing", make_preprocessor(numeric, categorical)),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                    class_weight="balanced", min_samples_leaf=2, n_jobs=-1))]),
            "Gradient Boosting": Pipeline([
                ("preprocessing", make_preprocessor(numeric, categorical)),
                ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                    max_depth=3, random_state=RANDOM_STATE))]),
            "AdaBoost": Pipeline([
                ("preprocessing", make_preprocessor(numeric, categorical)),
                ("classifier", AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE))]),
        }

        validation_results, trained = {}, {}
        print("Training models; selecting using validation data...")
        for name, pipeline in models.items():
            started = time.perf_counter()
            if name in ("Gradient Boosting", "AdaBoost"):
                weights = compute_sample_weight(class_weight="balanced", y=y_train)
                pipeline.fit(X_train, y_train, classifier__sample_weight=weights)
            else:
                pipeline.fit(X_train, y_train)
            elapsed = time.perf_counter() - started
            probabilities = pipeline.predict_proba(X_val)[:, 1]
            threshold = choose_threshold(y_val, probabilities)
            result = evaluate(y_val, probabilities, threshold)
            result["training_seconds"] = round(float(elapsed), 4)
            result["model_file"] = name.lower().replace(" ", "_") + "_model.pkl"
            validation_results[name] = result
            trained[name] = pipeline
            joblib.dump(pipeline, MODEL_DIR / result["model_file"])
            print(f"{name:20s} | val F1={result['f1_score']:.3f} | PR-AUC={result['pr_auc']:.3f} | threshold={threshold:.3f}")

        # Validation F1 is primary; validation PR-AUC breaks ties.
        selected = max(validation_results, key=lambda n: (
            validation_results[n]["f1_score"], validation_results[n]["pr_auc"]))
        threshold = validation_results[selected]["threshold"]
        selected_model = trained[selected]

        # Evaluate only the chosen model on the held-out test set, after selection.
        test_probabilities = selected_model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, test_probabilities, threshold)
        joblib.dump(selected_model, MODEL_DIR / "fraud_model.pkl")

        counts = {
            "dataset_rows": int(len(data)), "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)), "test_rows": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "split_note": "Stratified 60% train / 20% validation / 20% final test split.",
        }
        note = ("Models and thresholds are selected using validation data. Final test metrics are "
                "reported only for the selected model on the untouched test split. Dataset-specific "
                "results do not guarantee real-world performance.")
        comparison = {
            **counts, "selection_metric": "validation_f1_score", "selected_model": selected,
            "selected_threshold": threshold, "validation_results": validation_results,
            "models": validation_results, "best_model_by_f1": selected,
            "final_test_results": test_metrics, "metric_note": note,
            "target_note": "Class 0 is treated as Normal and class 1 as Suspicious, assuming dataset labels use this mapping.",
        }
        (MODEL_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        metrics = {**counts, "selected_model": selected, **test_metrics, "note": note}
        (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"\nSelected model: {selected} (validation threshold {threshold:.3f})")
        print("FINAL TEST | " + " | ".join(
            f"{key}: {test_metrics[key]:.3f}" for key in ("accuracy", "precision", "recall", "f1_score", "pr_auc")))
        print("Saved selected model, metrics.json, and model_comparison.json.")


    if __name__ == "__main__":
        main()
