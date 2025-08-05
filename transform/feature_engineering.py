import pandas as pd
import numpy as np
from logger_config import setup_logger

logger = setup_logger(__name__)

def encode_gender(df):
    df['is_male'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df = df.drop('sex', axis=1)
    return df

def encode_hereditary_diseases(df):
    df = pd.get_dummies(df, columns=['hereditary_diseases'], prefix='hereditary_disease')
    if 'hereditary_disease_NoDisease' in df.columns:
        df = df.drop('hereditary_disease_NoDisease', axis=1)
    return df

def bin_age(df):
    bin_edges = [18, 30, 45, 65]
    bin_labels = ['Young Adult', 'Adult', 'Middle-Aged']
    df['age_group'] = pd.cut(df['age'], bins=bin_edges, labels=bin_labels, right=False)
    df = pd.get_dummies(df, columns=['age_group'], drop_first=True)
    df = df.drop(['age'], axis=1)
    return df

def bin_bmi(df):
    bmi_bin_edges = [0, 18.5, 25.0, 30.0, 100.0]
    bmi_bin_labels = ['Underweight', 'Normalweight', 'Overweight', 'Obese']
    df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bin_edges, labels=bmi_bin_labels, right=False)
    df = pd.get_dummies(df, columns=['bmi_category'], drop_first=True)
    df = df.drop(['bmi'], axis=1)
    return df

def encode_region(df):
    region_mapping = {
  "Northeast": ["Augusta","Bangor","Portland","Lewiston","South Portland","Montpelier","Burlington","South Burlington","Rutland","Bennington",
    "Concord","Manchester","Nashua","Dover","Rochester","Boston","Worcester","Springfield","Lowell","Cambridge","Providence","Warwick",
    "Cranston","Pawtucket","East Providence","Hartford","New Haven","Bridgeport","Stamford","Waterbury","New York City","Buffalo","Rochester",
    "Yonkers","Syracuse","Philadelphia","Phildelphia","Pittsburgh","Pittsburg","Allentown","Erie","Reading","Newark","Jersey City","Paterson","Elizabeth","Edison","Dover",
    "Wilmington","Newark","Middletown","Smyrna","NewYork","AtlanticCity","York","Trenton", "Harrisburg", "Newport"],
  "Southeast": ["Annapolis","Baltimore","Rockville","Gaithersburg","Frederick","Richmond","Virginia Beach","Norfolk","Chesapeake","Arlington",
    "Washington D.C.","Charleston","Huntington","Parkersburg","Wheeling","Morgantown","Raleigh","Charlotte","Greensboro","Durham","Winston-Salem",
    "Columbia","Charleston","North Charleston","Mount Pleasant","Rock Hill","Atlanta","Augusta","Columbus","Savannah","Athens","Jacksonville",
    "Miami","Tampa","Orlando","St. Petersburg","Montgomery","Birmingham","Huntsville","Mobile","Tuscaloosa","Jackson","Gulfport","Biloxi",
    "Hattiesburg","Southaven","Baton Rouge","New Orleans","Shreveport","Lafayette","Lake Charles","Nashville","Memphis","Knoxville","Chattanooga",
    "Clarksville","WashingtonDC","Brimingham","Louisville","NewOrleans","Georgia","Macon","Florence","PanamaCity","Kingsport"],
  "Southwest": ["Austin","Houston","San Antonio","Dallas","Fort Worth","Oklahoma City","Tulsa","Norman","Lawton","Edmond","Little Rock",
    "Fort Smith","Fayetteville","Springdale","Jonesboro","Albuquerque","Santa Fe","Las Cruces","Rio Rancho","Roswell","Phoenix","Tucson",
    "Mesa","Chandler","Scottsdale","Las Vegas","Henderson","Reno","North Las Vegas","Sparks","Salt Lake City","West Valley City","Provo",
    "Orem","Sandy","Oklahoma","Prescott","LasVegas","Denver","Kingman","SilverCity","Pheonix","SantaFe","Lovelock"],
  "Northwest": ["Helena","Billings","Missoula","Great Falls","Bozeman","Boise","Meridian","Nampa","Idaho Falls","Caldwell","Cheyenne","Casper",
    "Laramie","Gillette","Rock Springs","Portland","Salem","Eugene","Gresham","Hillsboro","Seattle","Spokane","Tacoma","Vancouver","Bellevue",
    "Anchorage","Fairbanks","Juneau","Sitka","Ketchikan","FallsCity","SantaRosa","Eureka","SanFrancisco","SanJose","LosAngeles","Oxnard","SanDeigo",
                "Oceanside","Carlsbad","Montrose","Fresno","SanLuis","Bakersfield","Mexicali"],
  "Northcentral": ["Marshall","Mandan","Waterloo","IowaCity","Indianapolis","Cincinnati","Bloomington","Salina","KanasCity","Brookings",
                   "Minot","Chicago","Lincoln","GrandForks","Fargo","Cleveland","Canton","Minneapolis","JeffersonCity","Escabana","Youngstown"]
}

    def get_region(city):
        for region, cities in region_mapping.items():
            if city in cities:
                return region
        return None

    df["region"] = df["city"].apply(get_region)
    missing_cities = df[df["region"].isnull()]["city"].unique()
    if missing_cities.size > 0:
        logger.warning(f"Missing regions for cities: {missing_cities}")
    else:
        logger.info(" All cities have been mapped to regions.")

    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    df = df.drop(['city'], axis=1)
    return df

def drop_job_title(df):
    if 'job_title' in df.columns:
        df = df.drop('job_title', axis=1)
    return df

def full_feature_engineering(df):
    logger.info(" Starting feature engineering pipeline...")
    df = encode_gender(df)
    df = encode_hereditary_diseases(df)
    df = bin_age(df)
    df = bin_bmi(df)
    df = encode_region(df)
    df = drop_job_title(df)
    logger.info(" Feature engineering completed.")
    return df
