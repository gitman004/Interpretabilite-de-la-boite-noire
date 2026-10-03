"""Explications SHAP des deux modèles et mesure de leur cohérence.

- Random Forest : TreeExplainer, valeurs exprimées en probabilité de « malin ».
- Régression logistique : LinearExplainer sur les données normalisées, valeurs
  exprimées en log-odds.

Les deux échelles diffèrent, on compare donc les explications par leur
classement (corrélation de Spearman, variables communes dans le top k) et par
la part d'importance de chaque variable, pas par leurs valeurs brutes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from scipy import stats


@dataclass
class Explanation:
    values: np.ndarray        # (n_cas, n_variables), contribution vers « malin »
    base_value: float
    data: pd.DataFrame        # valeurs d'origine des variables (non normalisées)
    unit: str

    def importance(self) -> pd.Series:
        """Importance globale : moyenne des |SHAP|, normalisée pour sommer à 1."""
        imp = pd.Series(np.abs(self.values).mean(axis=0), index=self.data.columns)
        return (imp / imp.sum()).sort_values(ascending=False)

    def for_case(self, i: int) -> shap.Explanation:
        return shap.Explanation(values=self.values[i], base_values=self.base_value,
                                data=self.data.iloc[i].to_numpy(),
                                feature_names=list(self.data.columns))


def _positive_class(values, base):
    """Selon la version de SHAP, un classifieur renvoie une liste par classe, un tableau
    (n, variables, classes) ou directement la classe positive."""
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]
    base = np.atleast_1d(base)
    return values, float(base[-1])


def explain_forest(model, X_test: pd.DataFrame) -> Explanation:
    explainer = shap.TreeExplainer(model)
    values, base = _positive_class(explainer.shap_values(X_test), explainer.expected_value)
    return Explanation(values, base, X_test.reset_index(drop=True), "probabilité de malin")


def explain_logistic(pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Explanation:
    scaler, clf = pipeline.named_steps["scaler"], pipeline.named_steps["clf"]
    explainer = shap.LinearExplainer(clf, shap.maskers.Independent(scaler.transform(X_train)))
    values = explainer.shap_values(scaler.transform(X_test))
    values, base = _positive_class(values, explainer.expected_value)
    return Explanation(values, base, X_test.reset_index(drop=True), "log-odds de malin")


@dataclass
class Agreement:
    spearman: float           # corrélation des classements d'importance (1 = identiques)
    top_k: int
    common: list[str]         # variables présentes dans le top k des deux modèles


def agreement(a: pd.Series, b: pd.Series, k: int = 10) -> Agreement:
    b = b.reindex(a.index)
    rho = float(stats.spearmanr(a.to_numpy(), b.to_numpy()).statistic)
    top_a = list(a.sort_values(ascending=False).index[:k])
    top_b = set(b.sort_values(ascending=False).index[:k])
    return Agreement(rho, k, [f for f in top_a if f in top_b])


def pick_case(y_true, proba_a, proba_b, threshold: float = 0.5) -> int:
    """Choisit le cas le plus instructif à expliquer : d'abord un cas où les deux modèles
    ne sont pas d'accord, sinon un cas mal classé, sinon le plus incertain."""
    y = np.asarray(y_true)
    pa, pb = (np.asarray(proba_a) >= threshold), (np.asarray(proba_b) >= threshold)
    uncertainty = -np.abs(np.asarray(proba_a) - 0.5) - np.abs(np.asarray(proba_b) - 0.5)
    for mask in (pa != pb, (pa != y) | (pb != y)):
        if mask.any():
            idx = np.flatnonzero(mask)
            return int(idx[np.argmax(uncertainty[idx])])
    return int(np.argmax(uncertainty))
