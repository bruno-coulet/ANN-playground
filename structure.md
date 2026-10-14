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
│       └── 5592fe8dba32461f8fd9487bd380fdae
│           └── artifacts
│               └── confusion_matrix.html
├── mlflow.db
├── mlruns
├── notebooks
│   └── 1_clean.ipynb
├── presentation
├── pyproject.toml
├── src
│   └── ann_playground
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-312.pyc
│       │   ├── cleaning_utils.cpython-312.pyc
│       │   ├── config.cpython-312.pyc
│       │   ├── eda_utils.cpython-312.pyc
│       │   └── tracker_utils.cpython-312.pyc
│       ├── apps
│       │   ├── app.py
│       │   ├── app_v0.py
│       │   └── pages
│       │       ├── 1_data_studio.py
│       │       └── 2_ann_lab.py
│       ├── cleaning_utils.py
│       ├── config.py
│       ├── data
│       ├── eda_utils.py
│       ├── model_utils.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-312.pyc
│       │   │   ├── keras_mlp.cpython-312.pyc
│       │   │   └── numpy_mlp.cpython-312.pyc
│       │   ├── keras_mlp.py
│       │   ├── numpy_mlp.py
│       │   └── train_baselinne.py
│       └── tracker_utils.py
├── structure.md
├── temp
├── tutorials
│   ├── keras_first_network.py
│   ├── main.py
│   └── pytorch_gradient_accumulation.py
└── uv.lock

21 directories, 37 files
