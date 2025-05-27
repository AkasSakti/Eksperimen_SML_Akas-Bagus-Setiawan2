import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print("[INFO] Dataset loaded successfully.")

    scaler = StandardScaler()
    features = df.drop(columns=["Revenue"])
    target = df["Revenue"]

    print("[INFO] Scaling features...")
    X_scaled = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(X_scaled, columns=features.columns)
    df_scaled["Revenue"] = target.values

    print("[INFO] Preprocessing complete.")
    return df_scaled

if __name__ == "__main__":
    os.makedirs("Eksperimen_SML_Akas-Bagus-Setiawan2/preprocessing/olshopdatapreprocesed/", exist_ok=True)

    output_path = "Eksperimen_SML_Akas-Bagus-Setiawan2/preprocessing/olshopdatapreprocesed/preprocessed.csv"
    df_ready = load_and_preprocess("Eksperimen_SML_Akas-Bagus-Setiawan2/preprocessing/online_shoppers_intention_preprocessed.csv")
    df_ready.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed file saved to: {output_path}")
