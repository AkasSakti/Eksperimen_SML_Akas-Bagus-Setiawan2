import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import mlflow
import mlflow.sklearn
import dagshub
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

# === Inisialisasi MLflow dan DagsHub ===
#mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan2.mlflow")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
dagshub.init(
    repo_owner='AkasSakti',
    repo_name='Eksperimen_SML_Akas-Bagus-Setiawan2',
    mlflow=True
)
mlflow.set_experiment("CI-Online-Shopper")

# === Mulai eksperimen ===
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # === Log model secara eksplisit ===
    mlflow.sklearn.log_model(model, "model")

    # === Simpan model secara manual (opsional)
    joblib.dump(model, MODEL_DIR / "model.pkl")
    mlflow.log_artifact(str(MODEL_DIR / "model.pkl"))

    # === Log confusion matrix PNG ===
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

    # === Log metric info JSON ===
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    metrics_path = MODEL_DIR / "metric_info.json"
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    mlflow.log_artifact(str(metrics_path))

    # === Log HTML estimator summary ===
    html_path = MODEL_DIR / "estimator.html"
    with open(html_path, "w") as f:
        f.write(f"<html><body><h1>Model Summary</h1><pre>{classification_report(y_test, y_pred)}</pre></body></html>")
    mlflow.log_artifact(str(html_path))

    # === Log manual requirements.txt ===
    req_path = MODEL_DIR / "requirements.txt"
    with open(req_path, "w") as f:
        f.write("scikit-learn\nmlflow\ndagshub\nmatplotlib\nseaborn\npandas\njoblib\n")
    mlflow.log_artifact(str(req_path))

print("✅ Semua artefak berhasil disimpan dan dilog ke MLflow DagsHub.")
