# Inference.py
import joblib
import pandas as pd

# Path absolut ke model
model_path = r'Eksperimen_SML_Akas-Bagus-Setiawan2\Membangun_model\Membangun_model\model.pkl'

# Load model
model = joblib.load(model_path)

# Contoh input (harus cocok urutan & nama kolomnya)
sample_input = pd.DataFrame([{
    'Administrative': 2,
    'Administrative_Duration': 30.0,
    'Informational': 0,
    'Informational_Duration': 0.0,
    'ProductRelated': 20,
    'ProductRelated_Duration': 400.0,
    'BounceRates': 0.02,
    'ExitRates': 0.04,
    'PageValues': 0.0,
    'SpecialDay': 0.0,
    'Month': 6,  # jika Month sudah dalam bentuk numerik
    'OperatingSystems': 2,
    'Browser': 2,
    'Region': 1,
    'TrafficType': 1,
    'VisitorType': 1,  # pastikan sudah encode ke 0/1
    'Weekend': 0,
    'Administrative_Duration_Bin':0
}])

# Prediksi
prediction = model.predict(sample_input)

print("✅ Hasil Prediksi:", prediction[0])  # 0 = Tidak beli, 1 = Beli
