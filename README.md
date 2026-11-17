# ann-playground

Guide d'installation et de préparation de l'environnement (branche `v4.app.val`).

## Prérequis

- Git
- Un accès au dépôt `https://github.com/bruno-coulet/ANN-playground.git`
- Le fichier de données `data.csv`

## 1. Récupération du projet

```bash
git clone https://github.com/bruno-coulet/ANN-playground.git ann-playground-v1
cd ann-playground-v1

git fetch --all
git switch main
```

## 2. Création de l'arborescence

```bash
mkdir -p data/raw_data data/clean_data data/processed_data
mkdir mlartifacts
```

Télécharger le data set https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
et enregistrer le dans `data/raw_data`

## 3. Fichiers de configuration

- `.streamlit/secrets.toml`
- `mlflow-data/basic_auth.ini`
- `.env`

Trois fichiers sont à créer à partir des modèles `.example` fournis.<br>
Remplacez chaque `xxx` par une valeur adaptée à votre environnement et ne commitez jamais ces fichiers.

### 3.1 Streamlit : `.streamlit/secrets.toml`

```bash
cp -p .streamlit/secrets.toml.exemple .streamlit/secrets.toml
```

```toml
AUTH_USER = "admin"
AUTH_PASSWORD = "xxx"
```

### 3.2 MLflow : `mlflow-data/basic_auth.ini`

```bash
cp -p mlflow-data/basic_auth.ini.example mlflow-data/basic_auth.ini
```

```ini
[mlflow]
default_permission = READ
database_uri = sqlite:////app/mlflow-data/mlflow.db
admin_username = admin
admin_password = xxx
```

### 3.3 Environnement : `.env`

```bash
cp -p .env.example .env
```

```dotenv
# ==========================================
# Docker & Image Configuration
# ==========================================
APP_VERSION=v1.0.0

# ==========================================
# Network & Port Configuration
# ==========================================
STREAMLIT_PORT=8501
MLFLOW_PORT=5000
API_PORT=8000

# ==========================================
# MLflow Authentication & Security
# ==========================================
MLFLOW_AUTH_ADMIN_USERNAME=admin
MLFLOW_AUTH_ADMIN_PASSWORD=xxx
MLFLOW_FLASK_SERVER_SECRET_KEY=<clé secrète à générer>

# ==========================================
# External Host Configuration (si nécessaire)
# ==========================================
EXTERNAL_HOST=xxx.xxx.xxx
```

Pour générer la clé secrète Flask :

```bash
openssl rand -hex 32
```

## 4. Données

Copiez le fichier de données dans le répertoire des données brutes.

```bash
# Depuis la racine du projet, si data.csv se trouve dans le répertoire parent
cp -p ../data.csv data/raw_data/
```

## Récapitulatif des ports

| Service   | Variable        | Valeur par défaut |
|-----------|-----------------|-------------------|
| Streamlit | `STREAMLIT_PORT`| 8511              |
| MLflow    | `MLFLOW_PORT`   | 5010              |
| API       | `API_PORT`      | 8010              |
