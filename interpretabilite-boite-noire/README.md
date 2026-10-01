# Interprétabilité de modèle — Boîte noire vs Glass Box

Un modèle très précis mais qu'on ne peut pas expliquer pose un vrai problème en entreprise : impossible de justifier une décision à un client, un auditeur, ou un médecin. Ce projet compare une approche "boîte noire" (Random Forest) expliquée a posteriori avec SHAP, et une approche nativement interprétable (régression logistique), sur un cas de classification binaire.

## Cas d'usage

Classification de tumeurs (malignes / bénignes) à partir de 30 mesures cellulaires — dataset *Wisconsin Breast Cancer* (intégré à scikit-learn, aucune donnée patient réelle). L'explicabilité est un enjeu réel dans ce contexte : un médecin doit comprendre *pourquoi* un modèle propose un diagnostic, pas seulement le résultat.

## Ce que fait le script

1. Entraîne un **Random Forest** (précis, difficile à interpréter directement) et une **régression logistique** (glass box, coefficients directement lisibles)
2. Compare leurs performances (accuracy, F1, ROC-AUC)
3. Applique **SHAP** (`TreeExplainer`) sur le Random Forest :
   - un *summary plot* : quelles variables pèsent le plus, globalement, sur les prédictions
   - un *waterfall plot* : explique **une prédiction individuelle** précise (pourquoi ce cas a été classé ainsi)
4. Affiche les coefficients les plus influents de la régression logistique, pour comparaison directe

## Résultat obtenu

Sur ce dataset, la régression logistique (interprétable nativement) égale voire dépasse légèrement le Random Forest en performance (ROC-AUC 0,998 contre 0,993) — un résultat qu'on retrouve souvent en pratique : la complexité d'un modèle ne garantit pas un gain de précision, et sacrifier l'interprétabilité n'est pas toujours justifié.

## Lancer le projet

```bash
pip install -r requirements.txt
python src/main.py
```

Génère dans `outputs/` :
- `metrics.json` — performances des deux modèles
- `glassbox_coefficients.png` — variables les plus influentes de la régression logistique
- `shap_summary_global.png` — importance globale des variables (Random Forest, vue SHAP)
- `shap_waterfall_patient0.png` — explication détaillée d'une prédiction individuelle

## Stack technique

Python · scikit-learn · SHAP · pandas · matplotlib
