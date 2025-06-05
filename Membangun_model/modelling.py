import os
import json
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import plotly.graph_objects as go

# ================================
# Setup MLflow Tracking untuk DagsHub
# ================================

# Ambil kredensial dari environment (GitHub Actions secret)
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'your_username_here')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'your_token_here')

# Set tracking URI ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan2.mlflow")

# Autolog harus dideklarasikan sebelum start_run
mlflow.autolog()

# ================================
# Buat direktori lokal untuk artefak
# ================================
local_base_dir = os.path.join("Membangun_model", "artefak")
os.makedirs(local_base_dir, exist_ok=True)

# ================================
# Load Dataset
# ================================
data_path = os.path.join("preprocessing", "online_shoppers_intention_preprocessed.csv")
df = pd.read_csv(data_path)
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Mulai Eksperimen MLflow
# ================================
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    cm_path = os.path.join(local_base_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Classification Report (JSON)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(local_base_dir, "metric_info.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path)

    # Save model (pkl)
    model_path = os.path.join(local_base_dir, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # Feature Importance Plot
    fig = go.Figure(go.Bar(
        x=model.feature_importances_,
        y=X.columns,
        orientation='h'
    ))
    fig.update_layout(title="Feature Importance", yaxis={'autorange': 'reversed'})
    estimator_html_path = os.path.join(local_base_dir, "estimator.html")
    fig.write_html(estimator_html_path)
    mlflow.log_artifact(estimator_html_path)

print("Run selesai. Artefak tersimpan di:", local_base_dir)
