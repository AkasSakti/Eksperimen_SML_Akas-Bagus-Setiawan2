import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
import joblib
import mlflow
from dagshub import dagshub_logger, login

# ========== CONFIG ==========
REPO_NAME = "Eksperimen_SML_Akas-Bagus-Setiawan2"
MODEL_OUTPUT = "Membangun_model/model.pkl"
DATA_PATH = f"{REPO_NAME}/preprocessing/online_shoppers_intention_preprocessed.csv"
PROCESSED_PATH = f"{REPO_NAME}/preprocessing/olshopdatapreprocesed/preprocessed.csv"
PROCESSED_DIR = os.path.dirname(PROCESSED_PATH)

# ========== ENV CHECK ==========
MLFLOW_USER = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TOKEN = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_USER or not MLFLOW_TOKEN or not MLFLOW_URI:
    raise EnvironmentError("❌ Env MLFLOW_TRACKING_USERNAME, PASSWORD, dan URI wajib diset di GitHub Secrets!")

# ========== LOGIN ==========
try:
    login(username=MLFLOW_USER, token=MLFLOW_TOKEN)
    mlflow.set_tracking_uri(MLFLOW_URI)
    dagshub_logger.init(repo_owner="AkasSakti", repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("CI-Online-Shopper")
    print("✅ Login dan MLflow init berhasil.")
except Exception as e:
    raise RuntimeError(f"❌ Gagal login atau init MLflow/DagsHub: {e}")

# ========== PREPROCESS ==========
def load_and_preprocess(path):
    print(f"[INFO] Loading dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)
    print("[INFO] Dataset loaded successfully.")

    # Imputasi
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    df.drop_duplicates(inplace=True)
    df = df.convert_dtypes()

    # Skala numerik
    numerical_features = df.select_dtypes(include=['number']).columns
    for col in numerical_features:
        if df[col].dtype not in ['float64', 'int64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numerical_features, inplace=True)

    if len(numerical_features) > 0:
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        z_scores = np.abs(stats.zscore(df[numerical_features]))
        df = df[(z_scores < 3).all(axis=1)]

    # Encode kategori
    categorical_features = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_features:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"[WARNING] Encoding gagal pada kolom {col}: {e}")

    # Optional binning
    if 'Administrative_Duration' in df.columns and not df['Administrative_Duration'].isnull().all():
        try:
            df['Administrative_Duration_Bin'] = pd.qcut(
                df['Administrative_Duration'],
                q=4,
                labels=False,
                duplicates='drop'
            )
        except Exception as e:
            print(f"[WARNING] Binning error: {e}")

    # Target normalisasi
    if 'Revenue' in df.columns:
        if df['Revenue'].dtype != 'int' and df['Revenue'].dtype != 'bool':
            if set(df['Revenue'].unique()) <= {0.0, 1.0}:
                df['Revenue'] = df['Revenue'].astype(int)
            else:
                df['Revenue'] = (df['Revenue'] > 0.5).astype(int)

    print("[INFO] Preprocessing complete.")
    return df

# ========== MAIN ==========
if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_ready = load_and_preprocess(DATA_PATH)
    df_ready.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Preprocessed file saved to: {PROCESSED_PATH}")

    # ========== MODELLING & LOGGING ==========
    X = df_ready.drop(columns=["Revenue"])
    y = df_ready["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            print(report)

            # Logging MLflow
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", acc)

            # Save model
            os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
            joblib.dump(model, MODEL_OUTPUT)
            print(f"✅ Model disimpan sebagai {MODEL_OUTPUT}")

            if os.path.exists(MODEL_OUTPUT):
                mlflow.log_artifact(MODEL_OUTPUT)
            else:
                print(f"[WARNING] File model tidak ditemukan saat log_artifact.")
    except Exception as e:
        print(f"❌ Error saat modelling atau logging MLflow: {e}")
