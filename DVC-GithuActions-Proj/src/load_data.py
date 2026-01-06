import pandas as pd
import requests
from typing import List, Dict, Any
from src.paths import RAW_DATA, PROCESSED_DATA


file_path = RAW_DATA / "dpe-france.xlsx"

def load_data(file_path: str = file_path) -> pd.DataFrame:
        """Charge les données depuis le fichier Excel et retourne un DataFrame pandas."""
if not file_path.exists():
    raise FileNotFoundError(f"Fichier introuvable : {file_path}")

data = pd.read_excel(file_path)

if __name__ == "__main__":
    print(data.head())
