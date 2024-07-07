import pandas as pd
import os


def load_and_combine_csv_files(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    df_list = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df
