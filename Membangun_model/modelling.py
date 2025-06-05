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

# Path relatif di repo
base_dir = "Membangun_model/Membangun_model"
os.makedirs(base_dir, exist_ok=True)

data_path = "preprocessing/online_shoppers_intention_preprocessed.csv"

# Pastikan autolog aktif dulu
mlflow.autolog()

# Load data
df = pd.read_csv(data_path)
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Simpan confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Simpan classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(base_dir, "metric_info.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Simpan model (opsional)
    model_path = os.path.join(base_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Visualisasi feature importance simpan ke html
    fi = model.feature_importances_
    fig = go.Figure(go.Bar(
        x=fi,
        y=X.columns,
        orientation='h'
    ))
    fig.update_layout(title="Feature Importance", yaxis={'autorange': 'reversed'})
    estimator_html_path = os.path.join(base_dir, "estimator.html")
    fig.write_html(estimator_html_path)

print("Training selesai, semua artefak tersimpan di:", base_dir)
