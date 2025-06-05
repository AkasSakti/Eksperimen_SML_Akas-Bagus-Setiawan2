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

# Direktori lokal untuk simpan artefak (sesuai request)
local_base_dir = os.path.join(os.getcwd(), "Membangun_model", "Membangun_model")
os.makedirs(local_base_dir, exist_ok=True)

# Load dataset
data_path = os.path.join(os.getcwd(), "..", "preprocessing", "online_shoppers_intention_preprocessed.csv")
df = pd.read_csv(data_path)
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set mlflow tracking URI ke DagsHub kamu
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan2.mlflow")

# Aktifkan autolog mlflow SEBELUM start_run
mlflow.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    cm_path = os.path.join(local_base_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(local_base_dir, "metric_info.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path)

    # Save model.pkl lokal dan log ke mlflow
    model_path = os.path.join(local_base_dir, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # Feature importance (interactive)
    fi = model.feature_importances_
    fig = go.Figure(go.Bar(
        x=fi,
        y=X.columns,
        orientation='h'
    ))
    fig.update_layout(title="Feature Importance", yaxis={'autorange': 'reversed'})
    estimator_html_path = os.path.join(local_base_dir, "estimator.html")
    fig.write_html(estimator_html_path)
    mlflow.log_artifact(estimator_html_path)

print("Run selesai dan artefak sudah di-log ke DagsHub serta disimpan lokal di:", local_base_dir)
