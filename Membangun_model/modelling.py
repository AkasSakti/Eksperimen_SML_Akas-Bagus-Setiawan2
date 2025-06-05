import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import mlflow
import mlflow.sklearn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

# === Setup Base Directory ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Membangun_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    default=str(BASE_DIR / 'preprocessing/olshopdatapreprocesed/preprocessed.csv')
)
args = parser.parse_args()

# === Load Dataset ===
df = pd.read_csv(args.data_path)
df["Revenue"] = df["Revenue"].astype(int)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Setup MLflow ===
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan2.mlflow")
mlflow.set_experiment("CI-Online-Shopper")

# === Start MLflow Run ===
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log model ke MLflow
    mlflow.sklearn.log_model(model, "model")

    # Simpan model ke lokal dan log sebagai artefak
    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(str(model_path))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    conf_matrix_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()
    mlflow.log_artifact(str(conf_matrix_path))

    # Classification report as JSON
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    metrics_path = MODEL_DIR / "metric_info.json"
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    mlflow.log_artifact(str(metrics_path))

    # Classification report as HTML
    html_path = MODEL_DIR / "estimator.html"
    with open(html_path, "w") as f:
        f.write(f"<html><body><h1>Model Summary</h1><pre>{classification_report(y_test, y_pred)}</pre></body></html>")
    mlflow.log_artifact(str(html_path))

    # Requirements
    req_path = MODEL_DIR / "requirements.txt"
    with open(req_path, "w") as f:
        f.write("\n".join([
            "scikit-learn",
            "mlflow",
            "matplotlib",
            "seaborn",
            "pandas",
            "joblib"
        ]) + "\n")
    mlflow.log_artifact(str(req_path))

print("✅ Semua artefak berhasil disimpan dan dilog ke MLflow DagsHub.")
