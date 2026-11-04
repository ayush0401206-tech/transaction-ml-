# Suspicious Transaction Detection — Four-Model Comparison Edition

A Flask + HTML/CSS/JavaScript academic project for suspicious transaction prediction. The original dark navy/cyan dashboard theme is retained. The main transaction prediction page continues to use Random Forest, and a new **Model Comparison** page is linked from the main dashboard.

## Included models
- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost

The comparison page shows accuracy, precision, recall, F1 score, training time, metric comparison bars, algorithm descriptions, and a confusion matrix for every model. ROC-AUC is intentionally not included. The highest F1 score is used to identify the comparison winner, while recall and false positives should also be considered for fraud detection.

## Project structure
- `dataset/finance_fraud_data.csv`: supplied dataset.
- `model/train_model.py`: trains all four algorithms and generates metrics.
- `model/fraud_model.pkl`: Random Forest model used by the main prediction page.
- `model/logistic_regression_model.pkl`, `model/random_forest_model.pkl`, `model/gradient_boosting_model.pkl`, `model/adaboost_model.pkl`: saved model pipelines.
- `model/metrics.json`: Random Forest metrics used by the original dashboard.
- `model/model_comparison.json`: comparison metrics for the new page.
- `app.py`: Flask server and prediction endpoints.
- `templates/index.html`: main prediction dashboard.
- `templates/model_comparison.html`: new model comparison page.
- `static/css/style.css`: shared original theme and comparison page styling.
- `static/js/script.js`: main prediction form behavior.

## Run on Windows
1. Install Python 3.10 or newer.
2. Open this project folder in VS Code and open Terminal.
3. Create a virtual environment: `python -m venv venv`
4. PowerShell: `venv\\Scripts\\Activate.ps1` (or Command Prompt: `venv\\Scripts\\activate.bat`)
5. Install the dependencies already listed in `requirements.txt`: `pip install -r requirements.txt`
6. Start the app: `python app.py`
7. Open `http://127.0.0.1:5000`.
8. Click **Compare 4 Models** in the main dashboard to open the new page.

## Retrain all models
From the project root run:

```bash
python model/train_model.py
```

This recreates the four saved pipelines and both JSON metrics files. It uses a stratified 80/20 train-test split with `random_state=42` shared across all models. Class weights/sample weights are used to account for class imbalance.

## Dependencies
`requirements.txt` is unchanged from the previous project version because all four algorithms and their preprocessing/evaluation tools are already provided by the existing `scikit-learn` dependency. No additional package was required for the comparison page; the charts use CSS rather than a new charting library.

## Important notes
- `IP_Risk_Score` is excluded if present and is not used by the input form.
- Class `0` is treated as Normal and class `1` as Suspicious, assuming that is how the dataset labels are defined.
- The current prediction page continues to use Random Forest. The comparison page evaluates all four models on the same held-out test split.
- Metrics are specific to this dataset and are not a guarantee of real-world fraud detection. A suspicious prediction is a review signal, not proof of fraud.
- The dataset does not document a physical unit for `Distance_from_Home`, so the UI does not label it as kilometers or miles.
