import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def data_overview(df):
    print("\n📊 Data Overview:")
    print(df.info())
    print("\n🧮 Summary Statistics:")
    print(df.describe())
    print("\n🔍 Unique Values per Column:")
    print(df.nunique())

def check_missing_and_duplicates(df):
    print("\n📌 Checking for Duplicates:")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✅ Dropped. New shape: {df.shape}")
    
    print("\n📌 Checking for Missing Values:")
    missing = df.isnull().sum()
    missing_percent = (missing / df.shape[0]) * 100
    print(missing_percent[missing_percent > 0])
    return df
