import yaml 
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yml"



def configs()->dict:
    """
    Load configuration parameters from a YAML file.

    :return: Configuration parameters as a dictionary
    :rtype: dict
    """
    try:
        with open(PARAMS_PATH, 'r') as file:
            params = yaml.safe_load(file)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des configurations : {e}")
    return params