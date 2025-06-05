import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
import mlflow
import dagshub
from pathlib import Path

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
    print(classification_report(y_test, y_pred))

    # Simpan model lokal (boleh untuk backup/keperluan manual)
    MODEL_DIR = "Membangun_model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    print(f"✅ Model disimpan sebagai {os.path.join(MODEL_DIR, 'model.pkl')}")

    # Tidak perlu mlflow.log_param, log_metric, atau log_artifact
