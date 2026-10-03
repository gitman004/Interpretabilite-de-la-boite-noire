"""Évaluation des modèles.

Un seul découpage train/test (143 cas de test ici) ne suffit pas pour affirmer
qu'un modèle est meilleur qu'un autre : un ou deux cas mal classés changent le
classement. On utilise donc une validation croisée stratifiée répétée
(5 plis x 10 répétitions = 50 évaluations par modèle) et un test de Student
corrigé (Nadeau et Bengio, 2003) adapté aux plis qui partagent des données
d'entraînement.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

METRICS = ["roc_auc", "sensibilite", "precision", "f1", "accuracy"]


def score(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "sensibilite": recall_score(y_true, pred),          # tumeurs malignes détectées
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
    }


def cross_validate(models: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                   n_repeats: int = 10, seed: int = 42):
    """Renvoie un DataFrame (une ligne par pli et par modèle) et les coefficients
    de la régression logistique obtenus sur chaque pli."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, coefs = [], []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        for name, factory in models.items():
            model = clone(factory(seed))
            model.fit(X.iloc[tr], y.iloc[tr])
            proba = model.predict_proba(X.iloc[te])[:, 1]
            rows.append({"fold": fold, "model": name, **score(y.iloc[te], proba)})
            if name == "logistic_regression":
                coefs.append(model.named_steps["clf"].coef_[0])
    folds = pd.DataFrame(rows)
    coefs = pd.DataFrame(coefs, columns=X.columns)
    return folds, coefs


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Moyenne et écart-type de chaque métrique, par modèle."""
    return folds.groupby("model")[METRICS].agg(["mean", "std"])


@dataclass
class Comparison:
    metric: str
    mean_diff: float      # modèle a - modèle b
    p_value: float
    a_wins: float         # part des plis où a fait strictement mieux que b


def corrected_t_test(folds: pd.DataFrame, a: str, b: str, metric: str,
                     test_fraction: float) -> Comparison:
    """Test de Student apparié corrigé de Nadeau et Bengio pour la validation croisée répétée.

    La variance est gonflée d'un facteur (1/k + n_test/n_train) pour tenir compte du
    recouvrement des jeux d'entraînement entre plis, que le test classique ignore et
    qui le rend beaucoup trop optimiste.
    """
    pa = folds[folds.model == a].set_index("fold")[metric]
    pb = folds[folds.model == b].set_index("fold")[metric]
    d = (pa - pb).to_numpy()
    k = len(d)
    var = d.var(ddof=1)
    ratio = test_fraction / (1 - test_fraction)
    if var == 0:
        p = 1.0 if d.mean() == 0 else 0.0
    else:
        t = d.mean() / np.sqrt((1 / k + ratio) * var)
        p = float(2 * stats.t.sf(abs(t), df=k - 1))
    return Comparison(metric, float(d.mean()), p, float((d > 0).mean()))


def coefficient_stability(coefs: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque variable : coefficient moyen, écart-type et part des plis où le signe
    est le même que celui du coefficient moyen. Un signe instable signifie que le sens
    de l'effet « lu » dans le modèle dépend des données d'entraînement."""
    mean = coefs.mean()
    same_sign = (np.sign(coefs) == np.sign(mean)).mean()
    out = pd.DataFrame({"coef_moyen": mean, "ecart_type": coefs.std(), "signe_stable": same_sign})
    return out.reindex(mean.abs().sort_values(ascending=False).index)


def threshold_for_sensitivity(model, X_train, y_train, target: float = 0.97,
                              seed: int = 42) -> float:
    """Seuil de décision choisi sur les seules données d'entraînement (prédictions hors
    pli) pour détecter au moins `target` des tumeurs malignes.

    Le seuil 0,5 par défaut traite un faux négatif (tumeur maligne manquée) comme un
    faux positif (examen complémentaire inutile) ; en aide au diagnostic, le premier
    coûte bien plus cher. Le jeu de test n'est jamais utilisé pour fixer le seuil."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    malignant = np.sort(proba[np.asarray(y_train) == 1])
    k = int(np.floor((1 - target) * len(malignant)))  # nombre de malins qu'on accepte de manquer
    return float(malignant[k])


def correlated_pairs(X: pd.DataFrame, threshold: float = 0.9) -> list[tuple[str, str, float]]:
    """Paires de variables très corrélées : elles portent la même information, et un
    modèle peut répartir le crédit entre elles de façon arbitraire."""
    corr = X.corr().abs()
    cols = list(corr.columns)
    return [(a, b, round(float(corr.loc[a, b]), 3))
            for i, a in enumerate(cols) for b in cols[i + 1:] if corr.loc[a, b] >= threshold]


def holdout_report(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {**{k: round(v, 4) for k, v in score(y_true, proba, threshold).items()},
            "vrais_positifs": int(tp), "faux_negatifs": int(fn),
            "faux_positifs": int(fp), "vrais_negatifs": int(tn)}
