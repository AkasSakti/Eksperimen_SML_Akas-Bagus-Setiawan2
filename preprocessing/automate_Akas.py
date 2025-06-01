import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats


def load_and_preprocess(path):
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print("[INFO] Dataset loaded successfully.")

    # 1. Imputasi Missing Value
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # 2. Drop Duplikat
    df.drop_duplicates(inplace=True)

    # 3. Konversi tipe data
    df = df.convert_dtypes()

    # 4. Pastikan kolom numerik benar-benar numerik
    numerical_features = df.select_dtypes(include=['number']).columns
    for col in numerical_features:
        if df[col].dtype not in ['float64', 'int64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numerical_features, inplace=True)

    if len(numerical_features) == 0:
        print("[WARNING] Tidak ada kolom numerik ditemukan setelah konversi.")
    else:
        # 5. Scaling (StandardScaler)
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        # 6. Deteksi & Penanganan Outlier (z-score)
        z_scores = np.abs(stats.zscore(df[numerical_features]))
        if z_scores.shape[1] == len(numerical_features):
            df = df[(z_scores < 3).all(axis=1)]
        else:
            print("[WARNING] Dimensi z_scores tidak sesuai, outlier tidak dideteksi.")

    # 7. Encoding Data Kategorikal
    categorical_features = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_features:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"[WARNING] Encoding gagal pada kolom {col}: {e}")

    # 8. Binning pada 'Administrative_Duration' jika ada
    if 'Administrative_Duration' in df.columns:
        if not df['Administrative_Duration'].isnull().all():
            try:
                df['Administrative_Duration_Bin'] = pd.qcut(
                    df['Administrative_Duration'],
                    q=4,
                    labels=False,
                    duplicates='drop'
                )
            except Exception as e:
                print(f"[WARNING] Binning error: {e}")
        else:
            print("[WARNING] Kolom 'Administrative_Duration' kosong, tidak bisa di-binning.")
    else:
        print("[WARNING] Kolom 'Administrative_Duration' tidak ditemukan.")

    # 9. Pastikan kolom Revenue bertipe int (0/1)
    if 'Revenue' in df.columns:
        if df['Revenue'].dtype != 'int' and df['Revenue'].dtype != 'bool':
            # Jika nilai 0.0 / 1.0, ubah ke int
            if set(df['Revenue'].unique()) <= {0.0, 1.0}:
                df['Revenue'] = df['Revenue'].astype(int)
            else:
                df['Revenue'] = (df['Revenue'] > 0.5).astype(int)

    print("[INFO] Preprocessing complete.")
    return df


if __name__ == "__main__":
    output_dir = "Eksperimen_SML_Akas-Bagus-Setiawan2/preprocessing/olshopdatapreprocesed/"
    os.makedirs(output_dir, exist_ok=True)

    input_path = "Eksperimen_SML_Akas-Bagus-Setiawan2/preprocessing/online_shoppers_intention_preprocessed.csv"
    output_path = os.path.join(output_dir, "preprocessed.csv")

    df_ready = load_and_preprocess(input_path)
    df_ready.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed file saved to: {output_path}")
