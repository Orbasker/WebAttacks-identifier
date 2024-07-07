from load_combine import load_and_combine_csv_files
from preproccess import preprocess_data
from train import train_svm_model, train_svm_model_sample
import joblib

dataset_path = "dataset/Network Intrusion dataset"
combined_df = load_and_combine_csv_files(dataset_path)

print(combined_df.head())

X, y, label_encoder_protocol, label_encoder_ip, scaler, pca = preprocess_data(
    combined_df
)

print(X.shape, y.shape)

# Train the SVM model on a sample of the dataset
sample_size = 100000  # Adjust the sample size as needed
svm_model_sample = train_svm_model_sample(X, y, sample_size)

# Save preprocessing objects
joblib.dump(label_encoder_protocol, "label_encoder_protocol.pkl")
joblib.dump(label_encoder_ip, "label_encoder_ip.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
