from pathlib import Path

## definir le chemin de base du projet
BASE_DIR = Path(__file__).resolve().parents[1]

## definir le chemin du dossier data
data_path = BASE_DIR / "DATA"
## definir le chemin du dossier models
RAW_DATA = data_path / "raw"
PROCESSED_DATA = data_path / "processed"