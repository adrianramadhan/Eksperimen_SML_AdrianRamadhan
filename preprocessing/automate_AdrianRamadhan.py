"""
automate_AdrianRamadhan.py

Skrip ini mengotomasi proses preprocessing dataset mental health:
- Membaca CSV input
- Memisahkan kolom numerik dan kategorikal
- Membangun pipeline preprocessing (imputasi, scaling, encoding)
- Split data train/test
- Transformasi dan simpan hasil preprocess (train/test) ke folder output

Fungsi utama:
- load_dataset(path)
- build_preprocessor(df)
- run_preprocessing(input_path, output_dir, test_size, random_state)

Usage:
$ python automate_AdrianRamadhan.py --input data/mental_health.csv \
    --output data/mental_health_dataset_preprocessing \
    --test_size 0.2 --random_state 42
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_dataset(path: str) -> pd.DataFrame:
    """Membaca dataset CSV dan mengembalikan DataFrame."""
    return pd.read_csv(path)


def build_preprocessor(df: pd.DataFrame):
    """
    Membangun ColumnTransformer untuk preprocessing:
    - Numerik: median imputation + StandardScaler
    - Kategorikal: most_frequent imputation + OneHotEncoder
    """
    # Identifikasi kolom
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'mental_health_risk' in num_cols:
        num_cols.remove('mental_health_risk')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'mental_health_risk' in cat_cols:
        cat_cols.remove('mental_health_risk')

    # Pipeline numerik
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline kategorikal
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])
    return preprocessor, num_cols, cat_cols


def run_preprocessing(input_path: str, output_dir: str, test_size: float=0.2, random_state: int=42):
    """
    Jalankan keseluruhan preprocessing:
    1. Load dataset
    2. Build preprocessor
    3. Split data
    4. Fit & transform train, transform test
    5. Simpan hasil ke output_dir

    Mengembalikan tuple:
    (train_df, test_df)
    """
    # Pastikan folder output ada
    os.makedirs(output_dir, exist_ok=True)

    # Load
    df = load_dataset(input_path)
    y = df['mental_health_risk']
    X = df.drop(columns=['mental_health_risk'])

    # Build
    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit & Transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Buat nama kolom hasil
    feature_names = preprocessor.get_feature_names_out()

    # DataFrame hasil
    train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    train_df['mental_health_risk'] = y_train.reset_index(drop=True)
    test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    test_df['mental_health_risk'] = y_test.reset_index(drop=True)

    # Save
    train_path = os.path.join(output_dir, 'mental_health_train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'mental_health_test_preprocessed.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved preprocessed train to: {train_path}")
    print(f"Saved preprocessed test to:  {test_path}")
    return train_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automate preprocessing mental health dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--output', type=str, required=True, help='Directory to save preprocessed files')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for split')
    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
