from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from src.init.paths import MODELS_DIR, PROCESSED_DATA
import pandas as pd
import joblib,yaml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from src.processing.feature import feature_engineering
from src.init.config import configs, configs_tuning
from dvclive import Live

params = configs()
target = params['feature_engineering']['target']
params_tuning = configs_tuning()

def tuning_evaluation():
    """
    Perform hyperparameter tuning and evaluate multiple models.

    Steps:
    - Load preprocessed data
    - Feature engineering
    - Train/test split
    - Define models and hyperparameter grids
    - Perform GridSearchCV for each model
    - Evaluate best models and save results
    """
    # --- Load preprocessed data ---
    df = pd.read_excel(PROCESSED_DATA / "df_prepare.xlsx")

    # --- Feature engineering ---
    X, y, preprocessor = feature_engineering(df)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=params['split']['test_size'],
        random_state=params['split']['random_state'],
        shuffle=params['split']['shuffle']
    )

    # Define models and hyperparameter grids
    models = {
        'RandomForest': (RandomForestRegressor()),
        'GradientBoosting': (GradientBoostingRegressor()),
        'ElasticNet': (ElasticNet()),
        'LinearRegression': (LinearRegression()),

    }

    best_estimators = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with Live() as live:
        for model_name, model in models.items():
            print(f"*********** Tuning {model_name} *************")

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid={'model__' + k: v for k, v in params_tuning['models'][model_name].items()},
                scoring='neg_mean_squared_error',
                cv=5,
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(X_train, y_train)
            preds = grid_search.predict(X_test)
            best_model = grid_search.best_estimator_

            best_estimators[model_name] = {
            "best_params": grid_search.best_params_,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "mse": mean_squared_error(y_test, preds),
            "rmse": root_mean_squared_error(y_test, preds),
            "cv_score": -grid_search.best_score_
        }
            
             # Metrics
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = root_mean_squared_error(y_test, preds)
            print(f"METRIC TRACKING EXPERIMENT - {model_name}")
            live.log_metric(f"{model_name}_r2", r2)
            live.log_metric(f"{model_name}_mae", mae)
            live.log_metric(f"{model_name}_mse", mse)
            live.log_metric(f"{model_name}_rmse", rmse)

            joblib.dump(best_model, MODELS_DIR / f"{model_name}_best.pkl")
        # Save best estimators and their performance
    with open(MODELS_DIR / "metrics_best.yml", "w") as f:
        yaml.dump(best_estimators, f, indent=4)

    print("Tuning and evaluation completed. Results saved.")

if __name__=='__main__':
    tuning_evaluation()