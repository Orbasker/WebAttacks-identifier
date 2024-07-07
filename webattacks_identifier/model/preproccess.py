from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import joblib


def preprocess_data(df, n_components=50):
    # Separate labels
    y = df["class"]
    df = df.drop(columns=["class"])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Identify categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = df.columns.difference(categorical_cols)

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_preprocessed)

    # Save preprocessing objects
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(pca, "pca.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    return X_pca, y, preprocessor, pca, label_encoder


def preprocess_new_data(df, preprocessor, pca):
    # Transform the datax
    # df = df.apply(pd.to_numeric, errors="coerce")
    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    df = df.fillna(0)

    X_preprocessed = preprocessor.transform(df)

    # Apply PCA
    X_new_pca = pca.transform(X_preprocessed)

    return X_new_pca
