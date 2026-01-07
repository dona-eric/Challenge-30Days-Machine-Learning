from sklearn.preprocessing import OneHotEncoder,LabelEncoder, TargetEncoder, OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from src.init.paths import PROCESSED_DATA
from src.init.config import configs

params = configs()

target = params['feature_engineering']['target']

data = PROCESSED_DATA / "df_prepare.xlsx"
def feature_engineering(df: pd.DataFrame):
    """
    Prepare features for ML model from preprocessed DPE dataframe.

    Steps:
    - Generate age of building
    - Round latitude/longitude
    - Create season feature
    - Identify numerical and categorical features
    - Return preprocessor (ColumnTransformer) and feature lists

    :param df: preprocessed dataframe
    :return: tuple (X, y, preprocessor)
    """

    # Target
    TARGET = target
    
    # Feature engineering
    df["age_batiment"] = df["year"] - df["annee_construction"]
    df["age_batiment"] = df["age_batiment"].clip(lower=0)

    # Rounded coordinates
    df["lat_round"] = df["latitude"].round(2)
    df["lon_round"] = df["longitude"].round(2)

    # Season feature
    df["season"] = df["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn"
    })

    # Features lists
    NUM_FEATURES = [
        "estimation_ges",
        "surface_thermique_lot",
        "age_batiment",
        "lat_round",
        "lon_round",
    ]

    CAT_FEATURES = [
        "classe_estimation_ges",
        "tv016_departement_code",
        "season",
    ]

    # Drop target
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    return X, y, preprocessor


# def get_pipeline(preprocessor, model=None):
#     """
#     Create full sklearn pipeline with preprocessor + model.
#     Default model: RandomForestRegressor
#     """
#     if model is None:
#         model = RandomForestRegressor(
#             n_estimators=200,
#             random_state=42,
#             n_jobs=-1
#         )

#     pipeline = Pipeline(
#         steps=[
#             ("preprocessing", preprocessor),
#             ("model", model)
#         ]
#     )
#     return pipeline
