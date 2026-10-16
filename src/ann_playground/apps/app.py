"""
Module : app.py
Description : Main entry point for the Multi-Page Streamlit application.
              Coordinates Data Studio (EDA / Preparation) and ANN Lab (MLP Benchmarking).

Usage :
    uv run streamlit run src/ann_playground/apps/app.py
"""

import streamlit as st

from ann_playground.config import DATA, PROCESSED_DATA_DIR

st.set_page_config(
    page_title="ANN Playground · Home",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ANN Playground : Deep Learning & Multi-Layer Perceptron")
st.markdown(
    """
    Bienvenue sur la plateforme d'expérimentation pour réseaux de neurones artificiels.
    Cette suite applicative est découpée en deux modules indépendants accessibles via le menu latéral :

    1. **Data Studio (`1_data_studio`)** :
       - Diagnostic structurel et détection des types statistiques réels.
       - Analyse exploratoire des données (EDA) et corrélations linéaires.
       - Suppression des variables redondantes (multicolinéarité).
       - Partition étanche Train/Test et export au format Parquet.

    2. **ANN Lab (`2_ann_lab`)** :
       - Laboratoire interactif de conception de topologies MLP (Keras & NumPy).
       - Suivi des courbes d'apprentissage (*loss*, *accuracy*) par epoch.
       - Diagnostic automatisé du surapprentissage / sous-apprentissage.
       - Analyse des matrices de confusion et des rapports de classification multiclasse.
       - Traçabilité des hyperparamètres et métriques via MLflow.
    """
)

st.divider()

col_status_1, col_status_2 = st.columns(2)

with col_status_1:
    st.markdown("### Source brute")
    if DATA.exists():
        st.success(f"Données brutes détectées : `{DATA.resolve()}`")
    else:
        st.error(f"Fichier brut manquant : `{DATA.resolve()}`")

with col_status_2:
    st.markdown("### Données préparées (Artefacts)")
    train_parquet = PROCESSED_DATA_DIR / "train_prepared.parquet"
    test_parquet = PROCESSED_DATA_DIR / "test_prepared.parquet"

    if train_parquet.exists() and test_parquet.exists():
        st.success(f"Artefacts d'entraînement disponibles dans `{PROCESSED_DATA_DIR}`")
    else:
        st.warning(
            "Aucune matrice préparée détectée. Veuillez exécuter le module "
            "`1_data_studio` pour générer le jeu de données d'apprentissage."
        )
