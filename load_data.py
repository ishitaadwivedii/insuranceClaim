import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv('data/health_insurance.csv')
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
