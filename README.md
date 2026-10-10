# Suspicious Transaction Detection — Basic Phase

A simple Flask + HTML/CSS/JavaScript application using a Random Forest classifier. The dataset and model intentionally exclude `IP_Risk_Score`, as requested.

## Structure
- `dataset/finance_fraud_data.csv`: dataset with `IP_Risk_Score` removed.
- `model/train_model.py`: script to retrain the model and recalculate test metrics.
- `model/fraud_model.pkl`: trained Random Forest pipeline.
- `model/metrics.json`: held-out test metrics.
- `app.py`: Flask server and prediction endpoint.
- `templates/index.html`: basic web page.
- `static/css/style.css`: styling.
- `static/js/script.js`: form behavior and API call.

## Run on Windows
1. Install Python 3.10 or newer.
2. Open this folder in VS Code and open Terminal.
3. `python -m venv venv`
4. PowerShell: `venv\Scripts\Activate.ps1` (or Command Prompt: `venv\Scripts\activate.bat`)
5. `pip install -r requirements.txt`
6. `python app.py`
7. Open `http://127.0.0.1:5000` in your browser.

## Retrain
From the project root run `python model/train_model.py`.

## Model details
The model uses timestamp-derived hour, weekday and month; amount; merchant category; distance from home; device type; average spending habit; weekend; and night-transaction flags. Transaction and customer IDs are excluded. Metrics are calculated on a stratified 80/20 held-out test split. They are dataset-specific and not a guarantee of real-world fraud detection.

## Dataset caveat
The dataset does not document a physical unit for `Distance_from_Home`, so the app labels it as a dataset value and does not call it kilometers or miles. The average spending feature contains negative values in the provided dataset; the UI allows the observed range.
