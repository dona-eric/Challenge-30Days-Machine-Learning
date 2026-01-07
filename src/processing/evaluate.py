import pandas as pd
from pathlib import Path
import joblib,json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from src.init.paths import MODELS_DIR

# --- Load trained model ---
pipeline_path = Path(MODELS_DIR) / "pde_model.pkl"
pipeline = joblib.load(pipeline_path)

def evaluate():
    """
    Evaluate trained ML model on test set.

    Steps:
    - Load trained model
    - Load test set
    - Predict on test set
    - Calculate evaluation metrics (RMSE, R2)
    - Save metrics to JSON file
    """
    # --- Load test set (sauvegardé par train.py) ---
    X_test = pd.read_csv(Path(MODELS_DIR) / "X_test.csv")
    y_test = pd.read_csv(Path(MODELS_DIR) / "y_test.csv").squeeze()  # Convert to Series

    # --- Evaluate ---
    y_pred = pipeline.predict(X_test)
    metrics = {
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "r2": r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': root_mean_squared_error(y_test, y_pred)
    }

    print("Evaluation metrics:", metrics)

    # --- Save metrics ---
    with open(Path(MODELS_DIR) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved at:", Path(MODELS_DIR) / "metrics.json")