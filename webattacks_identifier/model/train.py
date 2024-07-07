import logging
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedShuffleSplit,
)
from sklearn.metrics import classification_report
from preproccess import preprocess_data

logging.basicConfig(level=logging.INFO)


def train_svm_model(X, y):
    # Split data into training and test sets
    logging.info("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Perform grid search to find the best parameters
    logging.info("Performing grid search for hyperparameter tuning")
    param_grid = {
        "C": [0.1, 1, 3, 5, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],  # Only relevant for 'rbf' kernel
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters found: {grid_search.best_params_}")

    # Train the best model
    best_svm_model = grid_search.best_estimator_
    logging.info("Fitting the best SVM model")
    best_svm_model.fit(X_train, y_train)

    # Evaluate model
    logging.info("Evaluating model")
    y_pred = best_svm_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info(report)
    pd.DataFrame(report).to_csv("classification_report.csv")

    # Save the model
    joblib.dump(best_svm_model, "svm_model_best.pkl")
    logging.info("Model saved as svm_model_best.pkl")

    return best_svm_model


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

    # Perform grid search to find the best parameters
    logging.info("Performing grid search for hyperparameter tuning on sample")
    param_grid = {
        "C": [0.1, 1, 3, 5, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],  # Only relevant for 'rbf' kernel
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters found for sample: {grid_search.best_params_}")

    # Train the best model
    best_svm_model = grid_search.best_estimator_
    logging.info("Fitting the best SVM model on sample")
    best_svm_model.fit(X_train, y_train)

    # Evaluate model
    logging.info("Evaluating sample-trained model")
    y_pred = best_svm_model.predict(X_test)
    logging.info("Classification report for sample-trained model:")
    logging.info(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(best_svm_model, "svm_model_sample_best.pkl")
    logging.info("Sample-trained model saved as svm_model_sample_best.pkl")

    return best_svm_model


def main():
    dataset_path = "dataset/kddcup99_data.csv"
    combined_df = pd.read_csv(dataset_path)

    # Preprocess the data
    X, y, scaler, pca, label_encoder = preprocess_data(combined_df)

    # Train the SVM model on the full dataset
    logging.info("Training on the full dataset")
    train_svm_model(X, y)

    # Train the SVM model on a sample of the dataset
    # logging.info("Training on a sample of the dataset")
    # train_svm_model_sample(X, y)


if __name__ == "__main__":
    main()
