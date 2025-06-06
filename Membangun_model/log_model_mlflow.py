import mlflow
import joblib

# Path ke model.pkl Anda
model_path = r"D:\nang jember\Akas Bagus Setiawan\DICODING\Eksperimen_SML_Akas-Bagus-Setiawan2\Membangun_model\Membangun_model\model.pkl"
model = joblib.load(model_path)

# Log model ke MLflow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")

print("Model berhasil di-log ke MLflow.")
print("Run ID:", run.info.run_id)