# modelling.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Path dataset (ganti sesuai lokasi lokal kamu, atau gunakan path relatif)
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='preprocessing/olshopdatapreprocesed/online_shoppers_intention_preprocessed.csv')
args = parser.parse_args()

df = pd.read_csv(args.data_path)
# Fitur dan target
X = df.drop("Revenue", axis=1)
print(X.columns.tolist())
y = df["Revenue"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Simpan model ke file pkl
MODEL_DIR = "Membangun_model"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
print(f"✅ Model disimpan sebagai {os.path.join(MODEL_DIR, 'model.pkl')}")
