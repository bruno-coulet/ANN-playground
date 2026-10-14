from pathlib import Path
from typing import Final

import os

# config.py -> src/ann_playground/config.py
# parent = src/ann_playground/
# parent.parent = src/
# parent.parent.parent = project root

# /home/.../.../.../building-perceptron/src/ann_playground
# .parents[0] -> .../src/ann_playground
# .parents[1] -> .../src
# .parents[2] -> .../building-perceptron (racine)
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw_data"
CLEAN_DATA_DIR: Final[Path] = DATA_DIR / "clean_data"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed_data"

DATA: Final[Path] = RAW_DATA_DIR / "data.csv"
TARGET: Final[str] = "Target"

# MLflow configuration
MLFLOW_DB_PATH: Final[Path] = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "mlartifacts"

# Priorité à la variable d'environnement MLFLOW_TRACKING_URI (ex: http://localhost:5000)
# Fallback direct sur la base SQLite locale
MLFLOW_TRACKING_URI: Final[str] = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.resolve()}",
)
