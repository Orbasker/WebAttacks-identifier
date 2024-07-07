import logging
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from preproccess import preprocess_data

logging.basicConfig(level=logging.INFO)


def train_svm_model(X, y):
    # Split data into training and test sets
    logging.info("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train SVM model
    logging.info("Training SVM model")
    svm_model = SVC(kernel="linear", C=10)
    logging.info("Fitting SVM model")
    svm_model.fit(X_train, y_train)

    # Evaluate model
    logging.info("Evaluating model")
    y_pred = svm_model.predict(X_test)
    logging.info("Classification report:")
    # print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info(report)
    pd.DataFrame(report).to_csv("classification_report.csv")
    # Save the model
    joblib.dump(svm_model, "svm_model.pkl")
    logging.info("Model saved as svm_model.pkl")

    return svm_model


def train_svm_model_sample(X, y, sample_size=100000):
    # Stratified sampling to ensure all types are represented
    stratified_split = StratifiedShuffleSplit(
        n_splits=1, train_size=sample_size, random_state=42
    )
    for train_index, _ in stratified_split.split(X, y):
        X_sample, y_sample = X[train_index], y[train_index]

    # Split the sample data into training and test sets
    logging.info("Splitting sample data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )

    # Train Linear SVM model
    logging.info("Training Linear SVM model on sample")
    svm_model = SVC(kernel="linear", C=5)
    logging.info("Fitting Linear SVM model")
    svm_model.fit(X_train, y_train)

    # Evaluate model
    logging.info("Evaluating sample-trained model")
    y_pred = svm_model.predict(X_test)
    logging.info("Classification report for sample-trained model:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(svm_model, "svm_model_sample.pkl")
    logging.info("Sample-trained model saved as svm_model_sample.pkl")

    return svm_model


def main():
    dataset_path = "dataset/kddcup99_data.csv"
    combined_df = pd.read_csv(dataset_path)

    # Preprocess the data
    X, y, scaler, pca, label_encoder = preprocess_data(combined_df)

    # Train the SVM model on the full dataset
    logging.info("Training on the full dataset")
    train_svm_model(X, y)


if __name__ == "__main__":
    main()
