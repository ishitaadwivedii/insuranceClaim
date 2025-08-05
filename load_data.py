import pandas as pd
from logger_config import setup_logger

logger = setup_logger(__name__)  # FIX: get logger instance

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info("Data loaded successfully from %s", filepath)
        return df
    except FileNotFoundError:
        logger.error("File not found. Please check the file path: %s", filepath)
        return None
    except Exception as e:
        logger.error("An unexpected error occurred while loading data: %s", e)
        return None

