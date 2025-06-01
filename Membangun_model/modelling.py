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

# Path dataset (ganti sesuai lokasi lokal kamu, atau gunakan path relatif)
BASE_DIR = Path(__file__).resolve().parent.parent  # Masuk ke root project

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    default=str(BASE_DIR / 'preprocessing/olshopdatapreprocesed/preprocessed.csv')
)
args = parser.parse_args()

df = pd.read_csv(args.data_path)

#Revenuefix
df["Revenue"] = df["Revenue"].astype(int)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Simpan model ke file pkl
MODEL_DIR = "Membangun_model"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
print(f"Model disimpan sebagai {os.path.join(MODEL_DIR, 'model.pkl')}")

# 1. Set tracking URI dulu
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan2.mlflow")

# 2. Inisialisasi koneksi DagsHub (autentikasi & pengikatan MLflow)
dagshub.init(repo_owner='AkasSakti', repo_name='Eksperimen_SML_Akas-Bagus-Setiawan2', mlflow=True)

# 3. Set nama eksperimen
mlflow.set_experiment("CI-Online-Shopper")

# 4. Jalankan logging dalam run
with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("acc", 0.9)
    mlflow.log_artifact("Membangun_model/model.pkl")
    mlflow.autolog()  # Opsional: bisa ditaruh sebelum atau dalam run
