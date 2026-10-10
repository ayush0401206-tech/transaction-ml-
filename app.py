from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
model = joblib.load(BASE_DIR / "model" / "fraud_model.pkl")
with open(BASE_DIR / "model" / "metrics.json", "r", encoding="utf-8") as f:
    METRICS = json.load(f)

@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)

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
            "Is_Weekend": int(bool(data["is_weekend"])),
            "Is_Night_Transaction": int(bool(data["is_night_transaction"])),
        }])
        # Keep inputs within the observed dataset ranges to avoid unsupported extrapolation.
        limits = {
            "amount": (2.58, 7987.24),
            "distance_from_home": (0.0123, 4.8088),
            "avg_spending_habit": (-36.68, 200.55),
        }
        for key, (low, high) in limits.items():
            value = float(data[key])
            if not low <= value <= high:
                return jsonify({"error": f"{key.replace('_', ' ').title()} must be between {low} and {high}, based on the dataset range."}), 400
        if data["merchant_category"] not in ["Electronics", "Food_Dining", "Groceries", "Luxury_Goods", "Online_Services", "Transport"]:
            return jsonify({"error": "Please choose a valid merchant category."}), 400
        if data["device_type"] not in ["Android", "iOS", "Web_Browser"]:
            return jsonify({"error": "Please choose a valid device type."}), 400
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        classes = list(model.named_steps["classifier"].classes_)
        fraud_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
        normal_probability = float(probabilities[classes.index(0)]) if 0 in classes else 0.0
        return jsonify({
            "prediction": "Suspicious" if prediction == 1 else "Normal",
            "class_id": prediction,
            "fraud_probability": round(fraud_probability * 100, 2),
            "normal_probability": round(normal_probability * 100, 2),
            "disclaimer": "This is a model prediction for an academic project, not a definitive fraud determination."
        })
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": "Please complete all fields with valid values."}), 400

if __name__ == "__main__":
    app.run(debug=True)
