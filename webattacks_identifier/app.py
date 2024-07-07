import streamlit as st
import pandas as pd
import os
from model.predict import predict_pcap
from model.report import create_training_report

# Load the classification report and training data
classification_report = pd.read_csv("classification_report.csv")
training_data = pd.read_csv("webattacks_identifier/dataset/kddcup99_data.csv")

# Generate the training report
training_report = create_training_report(training_data, classification_report)

# Streamlit app
st.title("PCAP File Attack Type Prediction")

# List files in the pcap_files directory
pcap_directory = "webattacks_identifier/pcap_files"
pcap_files = [
    f for f in os.listdir(pcap_directory) if f.endswith((".pcap", ".pcapng", ".cap"))
]

# File selection
selected_file = st.selectbox("Select a PCAP file from directory", pcap_files)

if selected_file:
    pcap_file_path = os.path.join(pcap_directory, selected_file)

    # Predict attack types
    prediction = predict_pcap(pcap_file_path)

    st.write(f"Total packets: {len(prediction.predictions)}")
    st.write(f"Attack Types found: {prediction.attack_types}")
    st.write(f"Safe: {prediction.safe}")

    # Display predictions in a dataframe
    st.write("### Features")
    prediction_df = pd.DataFrame(prediction.features)
    st.write(prediction_df)

# File upload option
uploaded_file = st.file_uploader(
    "Or choose a PCAP file to upload", type=["pcap", "pcapng"]
)

if uploaded_file is not None:
    with open("uploaded_pcap.pcap", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict attack types
    prediction = predict_pcap("uploaded_pcap.pcap")

    st.write("### Features")
    st.write(prediction.features)
    st.write(f"Total predictions: {len(prediction.predictions)}")
    st.write(f"Attack Types: {prediction.attack_types}")
    st.write(f"Safe: {prediction.safe}")

# Display Report of the training model
st.write("### Training Report")
st.write("This report is based on the training data used to train the model.")
st.write(f"Total Rows: {training_report.total_rows}")
st.write(f"Amount of Labels: {training_report.amount_of_labels}")
st.write("Training Data:")
st.write(training_report.training_data.head())
st.write("Classification Report:")
st.write(training_report.report)
st.write("### Labels")
st.write(training_report.labels_count)
# Plot chart of the labels count
st.write("### Labels Count")
st.bar_chart(training_report.labels_count.set_index("Label"))
