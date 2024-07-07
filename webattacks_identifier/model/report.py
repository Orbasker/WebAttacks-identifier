import pandas as pd
import os
from dataclasses import dataclass


@dataclass
class TrainingReport:
    report: pd.DataFrame
    training_data: pd.DataFrame
    amount_of_labels: int
    labels: list[str]
    labels_count: pd.DataFrame
    total_rows: int


def detect_labels(data):
    labels = data["class"].unique()
    return labels


def detect_length_for_class(data, label):
    count = data[data["class"] == label].shape[0]
    return count


def create_training_report(data, report):
    labels = detect_labels(data)
    labels_count = pd.DataFrame(
        {
            "Label": labels,
            "Count": [detect_length_for_class(data, label) for label in labels],
        }
    )
    total_rows = data.shape[0]
    amount_of_labels = len(labels)
    return TrainingReport(
        report, data, amount_of_labels, labels, labels_count, total_rows
    )
