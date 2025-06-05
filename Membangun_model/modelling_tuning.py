import os
import json
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)

# =========================== ARGUMENT PARSING ============================
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='preprocessing/online_shoppers_intention_preprocessed.csv')
parser.add_argument('--mlflow_uri', type=str, default='https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan2.mlflow')
parser.add_argument('--username', type=str, required=True)
parser.add_argument('--token', type=str, required=True)
args = parser.parse_args()

# =========================== MLFLOW CONFIG ============================
os.environ['MLFLOW_TRACKING_USERNAME'] = args.username
os.environ['MLFLOW_TRACKING_PASSWORD'] = args.token
mlflow.set_tracking_uri(args.mlflow_uri)

# =========================== LOAD DATASET ============================
df = pd.read_csv(args.data_path)
X = df.drop("Revenue", axis=1)
y = df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================== GRID SEARCH ============================
params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, None]
}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid=params, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# =========================== EVALUATION ============================
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# =========================== LOGGING MLFLOW ============================
with mlflow.start_run(run_name="RandomForest Tuning Experiment"):
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Save classification report
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/classification_report_tuning.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path)

    # Log model
    mlflow.sklearn.log_model(best_model, artifact_path="model")

print("Model tuning & logging selesai.")
