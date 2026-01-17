import pandas as pd
from src.init.load_data import data_loaded, file_path
from src.init.paths import PROCESSED_DATA
from sklearn.impute import SimpleImputer
from src.init.config import configs

params = configs()
imputation_strategy = params['preprocess']['imputation_method']


def preprocessing_data(file: str) -> pd.DataFrame:
    """
    Preprocess DPE dataset:
    - date feature engineering
    - drop useless columns
    - median imputation on numeric columns
    - save processed dataset
    """
    try :
        # Load data
        df = data_loaded(file=file)

        # Datetime processing
        df["date_etablissement_dpe"] = pd.to_datetime(
            df["date_etablissement_dpe"], errors="coerce"
        )

        df["year"] = df["date_etablissement_dpe"].dt.year
        df["month"] = df["date_etablissement_dpe"].dt.month

        # Drop useless columns
        df = df.drop(
            columns=[
                "date_etablissement_dpe",
                "numero_dpe",
                "geo_adresse",
                "nom_methode_dpe",
                "version_methode_dpe",
                "tr002_type_batiment_description",
                "tr001_modele_dpe_type_libelle",
            ],
            errors="ignore",
        )

        # Columns to impute
        columns_imputer = [
            "latitude",
            "longitude"]
        # Keep only existing numeric columns
        columns_imputer = [
            col for col in columns_imputer if col in df.columns
        ]

        imputer = SimpleImputer(strategy=imputation_strategy)

        df[columns_imputer] = imputer.fit_transform(df[columns_imputer])
        df.dropna(subset=["code_insee_commune_actualise"])

        # Save processed data
        output_path = PROCESSED_DATA / "df_prepare.xlsx"
        df.to_excel(output_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du prétraitement des données : {e}")
    return df


if __name__ == "__main__":
    preprocessing_data(file=file_path)
