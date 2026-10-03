"""Chargement du jeu de données Wisconsin Breast Cancer (intégré à scikit-learn).

Le jeu d'origine code 0 = malin et 1 = bénin. On inverse ce codage pour que la
classe positive soit la tumeur maligne : la « sensibilité » (rappel de la classe 1)
mesure alors directement la part de tumeurs malignes détectées, ce qui est la
métrique qui compte en aide au diagnostic.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

MALIN, BENIN = 1, 0


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series  # 1 = malin, 0 = bénin

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def load() -> Dataset:
    raw = load_breast_cancer(as_frame=True)
    y = (1 - raw.target).rename("malin")
    return Dataset(X=raw.data, y=y)


def holdout(ds: Dataset, test_size: float = 0.25, seed: int = 42):
    """Découpage stratifié train/test, utilisé pour les explications et la matrice de confusion."""
    return train_test_split(ds.X, ds.y, test_size=test_size, random_state=seed, stratify=ds.y)
