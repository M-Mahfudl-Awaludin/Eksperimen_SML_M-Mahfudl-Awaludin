import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Fungsi untuk mendeteksi outlier menggunakan IQR ---
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# --- Fungsi untuk menghapus outlier menggunakan IQR ---
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# --- Fungsi utama untuk preprocessing ---
def preprocess_data(file_path, output_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # --- Menampilkan jumlah nilai unik per kolom ---
    print("=== Unique Values Per Column ===")
    for col in df.columns:
        unique_vals = df[col].nunique()
        print(f"{col}: {unique_vals} unique values")

    # --- Transformasi Kolom Result menjadi Kolom heart_attack ---
    df["heart_attack"] = df["Result"].apply(lambda x: 1 if x == "positive" else 0)
    df = df.drop(columns=["Result"])

    # --- Kolom numerik untuk deteksi outlier ---
    numeric_cols = ['Age', 'Heart rate', 'Systolic blood pressure',
                    'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
    
    # --- Deteksi outlier ---
    print("\n=== Outlier Detection ===")
    for col in df.select_dtypes(include=np.number).columns:
        outliers = detect_outliers_iqr(df, col)
        print(f"{col}: {len(outliers)} outliers")
    
    # --- Salin dataframe asli dan hapus outlier ---
    df_clean = df.copy()
    df_clean = remove_outliers_iqr(df_clean, numeric_cols)

    # --- Pisahkan fitur dan label ---
    X = df_clean.drop(columns='heart_attack')
    y = df_clean['heart_attack']

    # --- Normalisasi (StandardScaler) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Split train-test ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Cek hasil
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- Simpan dataset yang telah diproses ke dalam file CSV ---
    # Membuat dataframe dengan fitur yang telah diproses
    processed_data = pd.DataFrame(X_scaled, columns=X.columns)
    processed_data['heart_attack'] = y.values
    
    # Simpan dataframe ke dalam file CSV
    processed_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    # Return data yang telah diproses
    return X_train, X_test, y_train, y_test

# --- Jalankan fungsi preprocessing dengan file path ---
if __name__ == "__main__":
    input_file_path = 'heart_attack_raw.csv'  # Ganti dengan path file dataset Anda
    output_file_path = 'processed_data.csv'  # Ganti dengan path output yang diinginkan
    X_train, X_test, y_train, y_test = preprocess_data(input_file_path, output_file_path)
