import pandas as pd
from pathlib import Path
import joblib,yaml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from src.init.paths import MODELS_DIR, PLOT_DIR
import matplotlib.pyplot as plt
# --- Load trained model ---
pipeline_path = MODELS_DIR / "pde_model.pkl"
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
    print(f"***"*30)
    print(f"EVALUATION OF MODELS")
    # --- Load test set (sauvegardé par train.py) ---
    X_test = pd.read_csv(MODELS_DIR / "X_test.csv")
    y_test = pd.read_csv(MODELS_DIR / "y_test.csv").squeeze()  # Convert to Series

    # --- Evaluate ---
    y_pred = pipeline.predict(X_test)
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        "Erreur_Marge": (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100
    }

    print("Evaluation metrics:", metrics)

    # --- Save metrics ---
    with open(MODELS_DIR / "metrics.yml", "w") as f:
        yaml.dump(metrics, f, indent=4)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"**"*20)
    print(f"EVALUATION FINISHED")
    print("Metrics saved at:", MODELS_DIR / "metrics.yml")

    print("********************")
    print("PLOTS LIVE FOR COMPARISON")
    print("********************")

    # 1️⃣ Prédit vs Réel
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title("Prédit vs Réel")
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle="--")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "predicted_vs_actual.png")
    plt.show()
    plt.close()

    # 2️⃣ Résidus vs Prédictions
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus")
    plt.title("Résidus vs Prédictions")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "residuals_vs_predictions.png")
    plt.show()
    plt.close()

    # 3️⃣ Distribution des erreurs
    plt.figure(figsize=(6, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Erreur")
    plt.ylabel("Fréquence")
    plt.title("Distribution des erreurs")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "error_distribution.png")
    plt.show()
    plt.close()

    print(f"Plots sauvegardés dans : {PLOT_DIR}")

if __name__=="__main__":
    evaluate()