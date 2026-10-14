"""
Module : app.py
Description : Streamlit modular and rusable  Interface
              EDA, cleaning, feature selection
"""

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from ann_playground.cleaning_utils import (
    count_duplicates,
    missing_summary,
    special_columns,
)
from ann_playground.eda_utils import (
    correlated_features,
    drop_redundant_features,
    target_correlations,
)

# Configuration de la page
st.set_page_config(
    page_title="Data Studio EDA & Prep",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_dataset(file) -> pd.DataFrame:
    """Charge un fichier CSV ou Parquet dans un DataFrame Pandas."""
    if file.name.endswith(".csv"):
        # Détection automatique du séparateur
        return pd.read_csv(file, sep=None, engine="python")
    elif file.name.endswith(".parquet"):
        return pd.read_parquet(file)
    raise ValueError("Format non supporté. Fournir un CSV ou Parquet.")


def detect_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Détecte les types statistiques usuels."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    bin_cols = [
        col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1, True, False})
    ]
    return num_cols, cat_cols, bin_cols


# --- Barre latérale : Chargement et Paramètres globaux ---
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier (CSV / Parquet)", type=["csv", "parquet"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier dans la barre latérale pour commencer.")
    st.stop()

df_raw = load_dataset(uploaded_file)
# Nettoyage des espaces résiduels sur les colonnes
df_raw.columns = df_raw.columns.str.strip()

target_col = st.sidebar.selectbox("Sélectionner la Target (Optionnel)", ["None"] + list(df_raw.columns))

tabs = st.tabs(["1. Inspection & Diagnostic", "2. EDA & Distributions", "3. Corrélations & Sélection", "4. Export Data Prep"])

# --- Onglet 1 : Inspection & Diagnostic ---
with tabs[0]:
    st.subheader("Diagnostic structurel des données")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lignes", df_raw.shape[0])
    col2.metric("Colonnes", df_raw.shape[1])
    col3.metric("Doublons stricts", count_duplicates(df_raw))
    col4.metric("Colonnes constantes", len(special_columns(df_raw)["constant"]))

    st.markdown("#### Aperçu des premières lignes")
    st.dataframe(df_raw.head(), use_container_width=True)

    st.markdown("#### Bilan des valeurs manquantes et types")
    df_missing = missing_summary(df_raw)
    st.dataframe(df_missing[df_missing["missing_count"] > 0], use_container_width=True)

# --- Onglet 2 : EDA & Distributions ---
with tabs[1]:
    st.subheader("Visualisations univariées et bivariées")

    num_cols, cat_cols, _ = detect_types(df_raw)

    col_eda_1, col_eda_2 = st.columns(2)
    with col_eda_1:
        st.markdown("##### Distribution continue")
        selected_num = st.selectbox("Variable numérique", num_cols if num_cols else ["Aucune"])
        if selected_num != "Aucune":
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(df_raw[selected_num].dropna(), kde=True, ax=ax, color="#439cc8")
            st.pyplot(fig)
            plt.close(fig)

    with col_eda_2:
        st.markdown("##### Distribution catégorielle / discrète")
        selected_cat = st.selectbox("Variable catégorielle", cat_cols if cat_cols else ["Aucune"])
        if selected_cat != "Aucune":
            fig, ax = plt.subplots(figsize=(6, 3.5))
            order = df_raw[selected_cat].value_counts().head(15).index
            sns.countplot(data=df_raw, y=selected_cat, order=order, ax=ax, color="#439cc8")
            st.pyplot(fig)
            plt.close(fig)

# --- Onglet 3 : Corrélations & Sélection ---
with tabs[2]:
    st.subheader("Analyse de colinéarité & filtrage redondance")

    threshold = st.slider("Seuil de colinéarité absolue (|r|)", 0.50, 0.99, 0.85, 0.05)

    if len(num_cols) > 1:
        high_corr_pairs = correlated_features(df_raw[num_cols], threshold=threshold)
        st.markdown(f"**Paires fortement corrélées identifiées (|r| > {threshold}) :** {len(high_corr_pairs)}")

        if high_corr_pairs:
            df_corr_pairs = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Corrélation"])
            st.dataframe(df_corr_pairs, use_container_width=True)

        if target_col != "None" and pd.api.types.is_numeric_dtype(df_raw[target_col]):
            st.markdown("##### Corrélation avec la Target")
            top_corr = target_correlations(df_raw.drop(columns=[target_col]), df_raw[target_col], n_top=10)
            st.dataframe(top_corr, use_container_width=True)

            if st.button("Filtrer les variables redondantes"):
                df_reduced, dropped = drop_redundant_features(df_raw, target_col, threshold=threshold)
                st.session_state["df_prep"] = df_reduced
                st.success(f"{len(dropped)} colonnes supprimées : {', '.join(dropped)}")
    else:
        st.warning("Pas assez de variables numériques pour calculer la matrice de corrélation.")

# --- Onglet 4 : Data Prep & Export ---
with tabs[3]:
    st.subheader("Finalisation et Exportation")

    df_to_export = st.session_state.get("df_prep", df_raw.copy())

    st.markdown(f"Dimensions actuelles : `{df_to_export.shape}`")

    cols_to_remove = st.multiselect("Suppression manuelle de colonnes", df_to_export.columns)
    if cols_to_remove:
        df_to_export = df_to_export.drop(columns=cols_to_remove)

    csv_data = df_to_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger le dataset nettoyé (CSV)",
        data=csv_data,
        file_name="dataset_prepared.csv",
        mime="text/csv",
    )
