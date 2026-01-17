from pathlib import Path

## definir le chemin de base du projet
BASE_DIR = Path(__file__).resolve().parent.parent

## definir le chemin du dossier data
data_path = BASE_DIR / "DATA"
## definir le chemin du dossier models, des données brutes
RAW_DATA = data_path / "raw"
PROCESSED_DATA = data_path / "processed"
MODELS_DIR = BASE_DIR / "DPE_MODELS"    