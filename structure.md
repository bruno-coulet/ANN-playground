.
├── Dockerfile
├── README.md
├── data
│   ├── processed_data
│   │   ├── test_prepared.parquet
│   │   └── train_prepared.parquet
│   └── raw_data
│       └── data.csv
├── docker-compose.yml
├── mlartifacts
│   └── 0
│       ├── 5592fe8dba32461f8fd9487bd380fdae
│       │   └── artifacts
│       │       └── confusion_matrix.html
│       └── 62bf99db18a24deabcef3721d24a4b72
│           └── artifacts
│               └── confusion_matrix.html
├── mlflow.db
├── notebooks
│   └── 1_clean.ipynb
├── presentation
├── pyproject.toml
├── src
│   └── ann_playground
│       ├── __init__.py
│       ├── apps
│       │   ├── app.py
│       │   ├── app_v0.py
│       │   └── pages
│       │       ├── 1_data_studio.py
│       │       └── 2_ann_lab.py
│       ├── cleaning_utils.py
│       ├── cleaning_utils_old.py
│       ├── config.py
│       ├── data
│       ├── eda_utils.py
│       ├── model_utils.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── keras_mlp.py
│       │   ├── numpy_mlp.py
│       │   └── train_baseline.py
│       └── tracker_utils.py
├── uv.lock
└── ~$ANN playground.pptx

22 directories, 40 files
