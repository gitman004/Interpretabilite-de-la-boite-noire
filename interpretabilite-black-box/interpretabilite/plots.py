"""Graphiques enregistrés dans outputs/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

LABELS = {"random_forest": "Random Forest (boîte noire)",
          "logistic_regression": "Régression logistique (glass box)"}
COLORS = {"random_forest": "#3b6ea8", "logistic_regression": "#d08a2c"}
MALIN, BENIN = "#b5432f", "#2f7d4f"


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cv_comparison(summary: pd.DataFrame, path: Path, metrics: list[str]):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(metrics))
    width = 0.38
    for i, model in enumerate(summary.index):
        means = [summary.loc[model, (m, "mean")] for m in metrics]
        stds = [summary.loc[model, (m, "std")] for m in metrics]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
               color=COLORS[model], label=LABELS[model])
    names = {"roc_auc": "ROC-AUC", "sensibilite": "Sensibilité\n(malins détectés)",
             "precision": "Précision", "f1": "F1", "accuracy": "Accuracy"}
    ax.set_xticks(x, [names[m] for m in metrics])
    lows = [summary[(m, "mean")].min() - 2 * summary[(m, "std")].max() for m in metrics]
    ax.set_ylim(max(0.0, min(lows)), 1.0)
    ax.set_ylabel("Score moyen (± écart-type)")
    ax.set_title("Validation croisée stratifiée 5 plis × 10 répétitions")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def importance_comparison(imp_rf: pd.Series, imp_lr: pd.Series, path: Path, top: int = 12):
    order = (imp_rf + imp_lr.reindex(imp_rf.index)).sort_values(ascending=False).index[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(order))
    ax.barh(y + 0.2, imp_rf[order], 0.4, color=COLORS["random_forest"], label=LABELS["random_forest"])
    ax.barh(y - 0.2, imp_lr[order], 0.4, color=COLORS["logistic_regression"],
            label=LABELS["logistic_regression"])
    ax.set_yticks(y, order)
    ax.set_xlabel("Part de l'importance SHAP totale")
    ax.set_title("Les deux modèles s'appuient-ils sur les mêmes variables ?")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def coefficients(stability: pd.DataFrame, path: Path):
    data = stability.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [MALIN if v > 0 else BENIN for v in data.coef_moyen]
    bars = ax.barh(data.index, data.coef_moyen, xerr=data.ecart_type, capsize=3, color=colors)
    for bar, stable in zip(bars, data.signe_stable):
        if stable < 0.95:
            bar.set_hatch("///")
            bar.set_alpha(0.55)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Coefficient (variables normalisées), moyenne ± écart-type sur 50 plis")
    ax.set_title("Régression logistique : les 30 coefficients\n"
                 "rouge = pousse vers « malin », vert = vers « bénin », hachuré = signe instable")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def shap_summary(explanation, path: Path):
    plt.figure()
    shap.summary_plot(explanation.values, explanation.data, show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle(f"Random Forest — impact de chaque variable ({explanation.unit})", y=1.0)
    _save(fig, path)


def waterfall(explanation, case: int, title: str, path: Path):
    plt.figure()
    shap.plots.waterfall(explanation.for_case(case), max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle(title, y=1.0)
    _save(fig, path)
