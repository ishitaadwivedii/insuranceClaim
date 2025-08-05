import numpy as np
from logger_config import setup_logger

logger = setup_logger(__name__)

def handle_bloodpressure_anomalies(df):
    """
    Replaces physiologically implausible blood pressure values (0) with NaN,
    then imputes missing values with the mean.
    """
    logger.info(" Checking for anomalies in 'bloodpressure' column...")
    zero_count = (df['bloodpressure'] == 0).sum()
    logger.debug(f"Found {zero_count} zero values in 'bloodpressure' column.")

    if zero_count > 0:
        df['bloodpressure'] = df['bloodpressure'].replace(0, np.nan)
        mean_bp = df['bloodpressure'].mean()
        df['bloodpressure'].fillna(mean_bp, inplace=True)
        logger.info(f" Replaced zeros and filled missing 'bloodpressure' with mean: {mean_bp:.2f}")
    else:
        logger.info(" No anomalies detected in 'bloodpressure' column.")

    return df


def drop_irrelevant_features(df):
    """
    Drops features that are irrelevant or cause multicollinearity.
    - 'weight' is dropped due to correlation with 'bmi'.
    - 'regular_ex' is dropped due to low correlation with target.
    """
    cols_to_drop = ['weight', 'regular_ex']
    logger.info(f" Dropping columns: {cols_to_drop}")
    
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(existing_cols, axis=1)

    logger.debug(f"Remaining columns: {df.columns.tolist()}")
    return df
