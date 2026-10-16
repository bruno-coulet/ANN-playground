"""Module d'entraînement d'un réseau dense de référence (baseline).

Construit, entraîne et sauvegarde un modèle séquentiel Keras simple
sur des données synthétiques.
"""

from pathlib import Path
from typing import Final

import keras
import numpy as np
from keras import layers

# Constantes du modèle
INPUT_DIM: Final[int] = 8
NUM_CLASSES: Final[int] = 2
BATCH_SIZE: Final[int] = 32
EPOCHS: Final[int] = 5
ARTIFACTS_DIR: Final[Path] = Path("models")


def build_mlp_classifier(input_dim: int, num_classes: int) -> keras.Model:
    """Instancie et compile un modèle perceptron multicouche."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="baseline_mlp",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_pipeline() -> None:
    """Génère un jeu synthétique, entraîne le réseau et sauvegarde l'artefact."""
    # Données synthétiques pour test
    x_train: np.ndarray = np.random.randn(200, INPUT_DIM).astype(np.float32)
    y_train: np.ndarray = np.random.randint(0, NUM_CLASSES, size=(200,)).astype(
        np.int64
    )

    model: keras.Model = build_mlp_classifier(
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES
    )
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1,
    )

    # Sauvegarde au format standard Keras 3 (.keras)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path: Path = ARTIFACTS_DIR / "baseline_model.keras"
    model.save(save_path)
    print(f"\nModèle exporté avec succès : {save_path}")


if __name__ == "__main__":
    run_pipeline()
