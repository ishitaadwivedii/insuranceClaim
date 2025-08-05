# eda.py

import pandas as pd
from logger_config import setup_logger

logger = setup_logger(__name__)

def data_overview(df):
    logger.info("Displaying data overview...")
    logger.debug(df.info())
    logger.debug(df.describe())
    logger.debug(f"Unique values per column:\n{df.nunique()}")

def check_missing_and_duplicates(df):
    logger.info("Checking for duplicates and missing values...")
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate rows: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info(f"Duplicates dropped. New shape: {df.shape}")

    missing = df.isnull().sum()
    missing_percent = (missing / df.shape[0]) * 100
    logger.info("Missing values (in %):")
    logger.debug(missing_percent[missing_percent > 0])

    return df

def impute_missing_values(df):
    logger.info("Imputing missing values...")

    if 'age' in df.columns:
        df['age'] = df['age'].interpolate()
        logger.debug("Interpolated missing 'age' values.")

    if 'bmi' in df.columns:
        mean_bmi = df['bmi'].mean()
        df['bmi'].fillna(mean_bmi)
        logger.debug(f"Filled missing 'bmi' values with mean: {mean_bmi:.2f}")

    return df

