"""
Module : config.py
Description : Global configuration, filesystem paths, and MLflow experiment constants.

Constants :
    - BASE_DIR : Path to the project root directory.
    - SRC_DIR : Path to the Python package source tree.
    - DATA_DIR : Base directory for datasets.
    - RAW_DATA_DIR : Directory containing raw tabular files.
    - PROCESSED_DATA_DIR : Directory containing train/test processed parquet partitions.
    - DATA : Default path to raw dataset file.
    - EXPERIMENT_NAME : Target MLflow tracking experiment identifier.
"""

import os
from pathlib import Path
from typing import Final

# config.py -> src/ann_playground/config.py
# parent = src/ann_playground/
# parent.parent = src/
# parent.parent.parent = project root

# /home/.../.../.../Ann-playground/src/ann_playground
# .parents[0] -> .../src/ann_playground
# .parents[1] -> .../src
# .parents[2] -> .../Ann-playground (racine)
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]

SRC_DIR: Final[Path] = BASE_DIR / "src" / "ann_playground"
DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw_data"
CLEAN_DATA_DIR: Final[Path] = DATA_DIR / "clean_data"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed_data"

DATA: Final[Path] = RAW_DATA_DIR / "data.csv"
TARGET: Final[str] = "Target"

# MLflow configuration
MLFLOW_DB_PATH: Final[Path] = BASE_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR: Final[Path] = BASE_DIR / "mlartifacts"
# Priorité à la variable d'environnement MLFLOW_TRACKING_URI (ex: http://localhost:5000)
# Fallback direct sur la base SQLite locale
MLFLOW_TRACKING_URI: Final[str] = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.resolve()}",
)

EXPERIMENT_NAME: str = "ANN_Playground"
