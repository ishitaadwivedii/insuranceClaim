from load_data import load_data
from eda import data_overview, check_missing_and_duplicates

claim_df = load_data("data/health_insurance.csv")

# EDA Step 1: Basic Checks
data_overview(claim_df)

# EDA Step 2: Missing and Duplicate Check
claim_df = check_missing_and_duplicates(claim_df)
