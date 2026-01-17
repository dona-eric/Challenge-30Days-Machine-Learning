import pandas as pd
import requests
from typing import List, Dict, Any
from src.init.paths import RAW_DATA, PROCESSED_DATA
from tabulate import tabulate


file_path = RAW_DATA / "dpe-france.xlsx"

def data_loaded(file) -> pd.DataFrame:
    """
    function for data_loaded
    
    :param file: Description
    :return: Description
    :rtype: DataFrame
    
    """

    if not file.exists():
         raise FileNotFoundError(f"Fichier introuvable : {file}")
    try:
        data = pd.read_excel(file)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du fichier : {e}")
    return data

if __name__ == "__main__":
        data = data_loaded(file=file_path)
        print(f""" INFOS ON DATA """)
        print(f"**"*20)
        print(tabulate(data.head(20), headers='keys',tablefmt='simple'))
        print(f"**"*20)
        print(len(data.columns))
        print(f"**"*20)
        print(data.isnull().sum())
        print(f"**"*20)
        print(data.describe(include="all"))