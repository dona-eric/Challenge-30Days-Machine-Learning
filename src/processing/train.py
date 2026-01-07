import pandas as pd
import json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.processing.feature import feature_engineering, get_pipeline
from src.init.paths import PROCESSED_DATA, MODELS_DIR
from src.init.config import configs


params = configs()

test_size, random_state, shuffle = params['split']['test_size'], params['split']['random_state'], params['split']['shuffle']
model_params = params['model_params']

def train():
    """
    Train ML model on preprocessed and feature-engineered data.

    Steps:
    - Load preprocessed data
    - Feature engineering
    - Train/test split
    - Build pipeline with preprocessor and model
    - Fit model
    - Evaluate model
    - Save model and metrics
    """
    # --- Load preprocessed data ---
    df = pd.read_excel(PROCESSED_DATA / "df_prepare.xlsx")

    # --- Feature engineering ---
    X, y, preprocessor = feature_engineering(df)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        shuffle=shuffle
    )

    # --- Build pipeline ---
    pipeline = get_pipeline(preprocessor,
                            model=None)
    pipeline.named_steps["model"].set_params(**model_params)

    # --- Fit model ---
    pipeline.fit(X_train, y_train)

    # --- Save model ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "pde_model.pkl")


    # --- sauvegarde des indices test pour evaluate.py ---
    output_path_X = MODELS_DIR / "X_test.csv"
    output_path_y = MODELS_DIR / "y_test.csv"

    X_test.to_csv(output_path_X, index=False)
    y_test.to_csv(output_path_y, index=False)
    print("Training done, model saved at:", MODELS_DIR / "pde_model.pkl")