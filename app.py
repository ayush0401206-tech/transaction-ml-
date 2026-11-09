from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

model = joblib.load(BASE_DIR / "model" / "fraud_model.pkl")

with open(BASE_DIR / "model" / "metrics.json", "r", encoding="utf-8") as file:
    METRICS = json.load(file)

THRESHOLD = float(METRICS.get("threshold", 0.5))

with open(BASE_DIR / "model" / "model_comparison.json", "r", encoding="utf-8") as file:
    COMPARISON = json.load(file)

# The fitted pipeline contains preprocessing followed by the selected classifier.
preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

# TreeExplainer is suitable for the selected Gradient Boosting classifier.
# It explains the classifier's raw output, not a percentage probability.
shap_explainer = shap.TreeExplainer(classifier)


def explain_prediction(row):
    """Return the strongest feature contributions for class 1 (Suspicious)."""
    transformed_row = preprocessor.transform(row)
    shap_output = shap_explainer.shap_values(transformed_row)

    # SHAP versions/models can return either a list per class or an array.
    if isinstance(shap_output, list):
        values = np.asarray(shap_output[1 if len(shap_output) > 1 else 0])[0]
    else:
        values_array = np.asarray(shap_output)
        if values_array.ndim == 3:
            # Some SHAP versions return (rows, features, classes).
            values = values_array[0, :, 1]
        else:
            values = values_array[0]

    feature_names = preprocessor.get_feature_names_out()
    contributions = []

    for name, impact in zip(feature_names, values):
        clean_name = str(name).replace("categorical__", "").replace("numeric__", "")
        contributions.append({
            "feature": clean_name.replace("_", " "),
            "impact": float(impact),
            "direction": "toward_suspicious" if impact > 0 else "toward_normal",
        })

    # Return the five largest absolute contributions for a readable UI.
    contributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return contributions[:5]


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/model-comparison")
def model_comparison():
    return render_template("model_comparison.html", comparison=COMPARISON)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        timestamp = datetime.fromisoformat(data["timestamp"])

        row = pd.DataFrame([{
            "Transaction_Hour": timestamp.hour,
            "Day_of_Week": timestamp.weekday(),
            "Month": timestamp.month,
            "Amount": float(data["amount"]),
            "Merchant_Category": str(data["merchant_category"]),
            "Distance_from_Home": float(data["distance_from_home"]),
            "Device_Type": str(data["device_type"]),
            "Avg_Spending_Habit": float(data["avg_spending_habit"]),
            "Is_Weekend": int(timestamp.weekday() >= 5),
            "Is_Night_Transaction": int(timestamp.hour < 5),
        }])

        # Validate numeric inputs against the ranges observed in the dataset.
        limits = {
            "amount": (2.58, 7987.24),
            "distance_from_home": (0.0123, 4.8088),
            "avg_spending_habit": (-36.68, 200.55),
        }
        for key, (low, high) in limits.items():
            value = float(data[key])
            if not low <= value <= high:
                message = (
                    f"{key.replace('_', ' ').title()} must be between "
                    f"{low} and {high}, based on the dataset range."
                )
                return jsonify({"error": message}), 400

        valid_merchants = [
            "Electronics", "Food_Dining", "Groceries",
            "Luxury_Goods", "Online_Services", "Transport",
        ]
        valid_devices = ["Android", "iOS", "Web_Browser"]

        if data["merchant_category"] not in valid_merchants:
            return jsonify({"error": "Please choose a valid merchant category."}), 400

        if data["device_type"] not in valid_devices:
            return jsonify({"error": "Please choose a valid device type."}), 400

        probabilities = model.predict_proba(row)[0]
        classes = list(classifier.classes_)
        fraud_probability = (
            float(probabilities[classes.index(1)]) if 1 in classes else 0.0
        )
        normal_probability = (
            float(probabilities[classes.index(0)]) if 0 in classes else 0.0
        )
        prediction = int(fraud_probability >= THRESHOLD)

        # SHAP explanations are attached to each successful prediction.
        explanations = explain_prediction(row)

        return jsonify({
            "prediction": "Suspicious" if prediction == 1 else "Normal",
            "class_id": prediction,
            "fraud_probability": round(fraud_probability * 100, 2),
            "normal_probability": round(normal_probability * 100, 2),
            "decision_threshold": round(THRESHOLD * 100, 2),
            "shap_explanations": explanations,
            "explanation_note": (
                "SHAP impact shows how each feature moves the model's raw output: "
                "toward Suspicious or toward Normal. It is not a percentage."
            ),
            "disclaimer": (
                "This is a model prediction for an academic project, "
                "not a definitive fraud determination."
            ),
        })

    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({
            "error": "Please complete all fields with valid values."
        }), 400
    except Exception:
        app.logger.exception("Prediction or SHAP explanation failed.")
        return jsonify({
            "error": "The prediction explanation could not be generated. "
                     "Check the Flask terminal for details."
        }), 500


if __name__ == "__main__":
    app.run(debug=False)
