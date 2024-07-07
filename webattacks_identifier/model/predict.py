import joblib
import pandas as pd

from model.extract_feautres import extract_features

# from extract_feautres import extract_features
# from preproccess import preprocess_new_data
from dataclasses import dataclass

from model.preproccess import preprocess_new_data


@dataclass
class Predict:
    pcap_file: str
    features: pd.DataFrame
    predictions: list[str]
    attack_types: list[str]
    safe: bool


def detect_attack_type(predictions):
    attack_types = set(predictions)
    attack_types.discard("normal.")
    return list(attack_types)


def define_safe(predictions):
    for prediction in predictions:
        if prediction != "normal.":
            return False
    return True


def predict_pcap(pcap_file):
    # Load the trained model and preprocessing objects
    svm_model = joblib.load("svm_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    pca = joblib.load("pca.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Extract features from the PCAP file
    df = extract_features(pcap_file)

    # Preprocess the extracted features
    X_new = preprocess_new_data(df, preprocessor, pca)

    # Predict
    predictions = svm_model.predict(X_new)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    attack_types = detect_attack_type(decoded_predictions)
    safe = define_safe(decoded_predictions)

    prediction = Predict(pcap_file, df, decoded_predictions, attack_types, safe)

    return prediction
