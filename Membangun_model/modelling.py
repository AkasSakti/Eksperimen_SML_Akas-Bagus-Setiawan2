import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
import os
import mlflow
import dagshub
from pathlib import Path
import matplotlib.pyplot as plt
import json

# === Setup Base Directory ===
BASE_DIR = Path(__file__).resolve().parent.parent

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

# === Inisialisasi MLflow Autologging (panggil sebelum start_run) ===
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan2.mlflow")
dagshub.init(
    repo_owner='AkasSakti',
    repo_name='Eksperimen_SML_Akas-Bagus-Setiawan2',
    mlflow=True
)
mlflow.set_experiment("CI-Online-Shopper")

mlflow.autolog()  # <= Penting! Ditaruh sebelum start_run

# === Training dan Tracking ===
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # === Simpan model lokal (opsional/manual)
    MODEL_DIR = "Membangun_model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model disimpan sebagai {model_path}")

    # === Log confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # === Simpan metric_info.json
    metrics_path = os.path.join(MODEL_DIR, "metric_info.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(metrics_path)

    # === Simpan estimator.html (opsional ringkasan HTML)
    html_path = os.path.join(MODEL_DIR, "estimator.html")
    with open(html_path, "w") as f:
        f.write(f"<html><body><h1>Random Forest Summary</h1><pre>{classification_report(y_test, y_pred)}</pre></body></html>")
    mlflow.log_artifact(html_path)

    # === Log manual model.pkl (meskipun autolog sudah menangani)
    mlflow.log_artifact(model_path)
