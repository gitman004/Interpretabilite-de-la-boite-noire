"""Les deux modèles comparés.

La normalisation de la régression logistique est placée dans un pipeline : en
validation croisée, elle est ainsi recalculée sur chaque pli d'entraînement et
n'utilise jamais les données de test.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def black_box(seed: int = 42) -> RandomForestClassifier:
    """Random Forest : performant, mais sa décision agrège 300 arbres et ne se lit pas directement."""
    return RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed, n_jobs=-1)


def glass_box(seed: int = 42) -> Pipeline:
    """Régression logistique : chaque coefficient se lit comme l'effet d'une variable normalisée."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
    ])


MODELS = {"random_forest": black_box, "logistic_regression": glass_box}
