from eda import data_overview, check_missing_and_duplicates, impute_missing_values
from load_data import load_data
from preprocessing import handle_bloodpressure_anomalies, drop_irrelevant_features
from transform.feature_engineering import full_feature_engineering
from logger_config import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info(" Starting Insurance Claim Prediction Pipeline...")

    df = load_data("data/health_insurance.csv")
    if df is None:
        logger.error(" Terminating pipeline due to data loading failure.")
        return

    df = check_missing_and_duplicates(df)
    df = impute_missing_values(df)
    df = handle_bloodpressure_anomalies(df)
    df = drop_irrelevant_features(df)
    df = full_feature_engineering(df)

    df.to_csv("processed_data.csv", index=False)
    logger.info(" Final processed data saved to processed_data.csv")

if __name__ == "__main__":
    main()

