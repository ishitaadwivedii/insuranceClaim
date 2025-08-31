import pandas as pd

def preprocess_input(smoker, bmi_category, bloodpressure, age_group, no_of_dependents,
                     diabetes, hereditary_heart, is_male, hereditary_alzheimer):
    # Binary encoding
    smoker_val = 1 if smoker == "Yes" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    hereditary_heart_val = 1 if hereditary_heart == "Yes" else 0
    hereditary_alzheimer_val = 1 if hereditary_alzheimer == "Yes" else 0
    is_male_val = 1 if is_male == "Male" else 0

    # One-hot for BMI category
    bmi_obese = 1 if bmi_category == "Obese" else 0
    bmi_overweight = 1 if bmi_category == "Overweight" else 0

    # One-hot for Age group
    age_middle = 1 if age_group == "Middle-Aged" else 0
    age_adult = 1 if age_group == "Adult" else 0

    # Build DataFrame in the exact feature order
    data = pd.DataFrame([[
        smoker_val,
        bmi_obese,
        bloodpressure,
        age_middle,
        no_of_dependents,
        diabetes_val,
        hereditary_heart_val,
        is_male_val,
        age_adult,
        hereditary_alzheimer_val,
        bmi_overweight
    ]], columns=[
        'smoker',
        'bmi_category_Obese',
        'bloodpressure',
        'age_group_Middle-Aged',
        'no_of_dependents',
        'diabetes',
        'hereditary_disease_HeartDisease',
        'is_male',
        'age_group_Adult',
        'hereditary_disease_Alzheimer',
        'bmi_category_Overweight'
    ])

    return data
