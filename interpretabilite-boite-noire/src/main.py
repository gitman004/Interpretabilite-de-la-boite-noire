"""
Interpretabilite de modele "boite noire" vs modele "glass box"
================================================================

Cas d'usage : aide au diagnostic (classification de tumeurs malignes/benignes
a partir de mesures cellulaires - dataset Wisconsin Breast Cancer, integre a
scikit-learn, aucune donnee patient reelle).

Objectif du projet : montrer qu'un modele tres precis (Random Forest) est
une boite noire difficile a justifier, et utiliser SHAP pour l'expliquer
(globalement et pour une prediction individuelle), puis comparer avec un
modele intrinsequement interpretable (regression logistique).

C'est exactement le type de question que se pose une entreprise avant de
mettre un modele en production : "je fais confiance au modele le plus
precis, ou a celui que je peux expliquer a un client / un auditeur / un
medecin ?"
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

OUT = "outputs"

def main():
    # ------------------------------------------------------------------
    # 1. Donnees
    # ------------------------------------------------------------------
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # y=0 malin, y=1 benin
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Modele "boite noire" : Random Forest (precis, peu interpretable)
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 3. Modele "glass box" : regression logistique (interpretable nativement)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]

    # ------------------------------------------------------------------
    # 4. Comparaison des performances
    # ------------------------------------------------------------------
    results = {
        "random_forest": {
            "accuracy": round(accuracy_score(y_test, rf_pred), 4),
            "f1": round(f1_score(y_test, rf_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, rf_proba), 4),
        },
        "logistic_regression": {
            "accuracy": round(accuracy_score(y_test, lr_pred), 4),
            "f1": round(f1_score(y_test, lr_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, lr_proba), 4),
        },
    }
    print("=== Performances ===")
    print(json.dumps(results, indent=2))
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # 5. Interpretabilite native du modele glass box (coefficients)
    # ------------------------------------------------------------------
    coefs = pd.Series(lr.coef_[0], index=feature_names).sort_values()
    top_coefs = pd.concat([coefs.head(5), coefs.tail(5)])
    plt.figure(figsize=(8, 6))
    colors = ["#c0392b" if v < 0 else "#2e7d32" for v in top_coefs.values]
    plt.barh(top_coefs.index, top_coefs.values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Regression logistique — coefficients les plus influents\n(rouge = pousse vers 'malin', vert = pousse vers 'benin')")
    plt.tight_layout()
    plt.savefig(f"{OUT}/glassbox_coefficients.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 6. SHAP sur le modele boite noire (Random Forest)
    # ------------------------------------------------------------------
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    # shap_values peut être une liste [classe0, classe1] ou un array (n, features, classes) selon la version
    if isinstance(shap_values, list):
        sv_class1 = shap_values[1]
    elif shap_values.ndim == 3:
        sv_class1 = shap_values[:, :, 1]
    else:
        sv_class1 = shap_values

    # 6a. Importance globale (summary plot)
    plt.figure()
    shap.summary_plot(sv_class1, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_summary_global.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6b. Explication d'UNE prediction individuelle (waterfall)
    idx = 0
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]

    explanation = shap.Explanation(
        values=sv_class1[idx],
        base_values=expected_value,
        data=X_test.iloc[idx].values,
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_waterfall_patient0.png", dpi=150, bbox_inches="tight")
    plt.close()

    predicted_label = "benin" if rf_pred[idx] == 1 else "malin"
    true_label = "benin" if y_test.iloc[idx] == 1 else "malin"
    print(f"\nExemple explique (outputs/shap_waterfall_patient0.png) :")
    print(f"  Prediction du modele : {predicted_label} (proba benin = {rf_proba[idx]:.3f})")
    print(f"  Vraie classe         : {true_label}")

    print("\nTermine. Graphiques et metriques dans outputs/.")


if __name__ == "__main__":
    main()
