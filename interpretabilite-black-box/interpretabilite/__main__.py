"""Pipeline complet : python -m interpretabilite [--out outputs] [--repeats 10] [--seed 42]"""

import argparse
import json
from pathlib import Path

from . import data, evaluation, explain, plots
from .models import MODELS


def run(out: Path, repeats: int = 10, seed: int = 42) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds = data.load()

    # 1. Comparaison robuste des performances
    folds, coefs = evaluation.cross_validate(MODELS, ds.X, ds.y, n_repeats=repeats, seed=seed)
    summary = evaluation.summarize(folds)
    comparisons = [evaluation.corrected_t_test(folds, "logistic_regression", "random_forest", m,
                                               test_fraction=1 / 5)
                   for m in ("roc_auc", "sensibilite", "f1")]
    stability = evaluation.coefficient_stability(coefs)

    # 2. Modèles entraînés sur un découpage fixe, pour les explications
    X_tr, X_te, y_tr, y_te = data.holdout(ds, seed=seed)
    rf = MODELS["random_forest"](seed).fit(X_tr, y_tr)
    lr = MODELS["logistic_regression"](seed).fit(X_tr, y_tr)
    p_rf, p_lr = rf.predict_proba(X_te)[:, 1], lr.predict_proba(X_te)[:, 1]

    exp_rf = explain.explain_forest(rf, X_te)
    exp_lr = explain.explain_logistic(lr, X_tr, X_te)
    imp_rf, imp_lr = exp_rf.importance(), exp_lr.importance()
    agree = explain.agreement(imp_rf, imp_lr, k=10)
    case = explain.pick_case(y_te, p_rf, p_lr)

    # 3. Graphiques
    plots.cv_comparison(summary, out / "comparaison_cv.png", ["roc_auc", "sensibilite", "precision", "f1"])
    plots.importance_comparison(imp_rf, imp_lr, out / "importance_shap_deux_modeles.png")
    plots.coefficients(stability, out / "coefficients_regression_logistique.png")
    plots.shap_summary(exp_rf, out / "shap_summary_random_forest.png")
    verdict = "malin" if y_te.iloc[case] == 1 else "bénin"
    plots.waterfall(exp_rf, case, f"Random Forest — cas n°{case} (vraie classe : {verdict}, "
                    f"P(malin) = {p_rf[case]:.2f})", out / "waterfall_random_forest.png")
    plots.waterfall(exp_lr, case, f"Régression logistique — même cas (P(malin) = {p_lr[case]:.2f})",
                    out / "waterfall_regression_logistique.png")

    unstable = stability[stability.signe_stable < 0.95]
    thresholds = {name: evaluation.threshold_for_sensitivity(MODELS[name](seed), X_tr, y_tr, seed=seed)
                  for name in MODELS}
    pairs = evaluation.correlated_pairs(ds.X)
    report = {
        "validation_croisee": {
            "plis": len(folds) // len(MODELS),
            "moyennes": {m: {k: round(float(summary.loc[m, (k, "mean")]), 4)
                             for k in evaluation.METRICS} for m in summary.index},
            "ecarts_types": {m: {k: round(float(summary.loc[m, (k, "std")]), 4)
                                 for k in evaluation.METRICS} for m in summary.index},
            "logistique_moins_foret": {c.metric: {"difference": round(c.mean_diff, 4),
                                                  "p_value_corrigee": round(c.p_value, 4),
                                                  "part_plis_gagnes_par_logistique": round(c.a_wins, 3)}
                                       for c in comparisons},
        },
        "jeu_de_test": {
            "taille": len(y_te),
            "random_forest": evaluation.holdout_report(y_te, p_rf),
            "logistic_regression": evaluation.holdout_report(y_te, p_lr),
            "seuil_ajuste_sensibilite_97": {
                "random_forest": {"seuil": round(thresholds["random_forest"], 3),
                                  **evaluation.holdout_report(y_te, p_rf, thresholds["random_forest"])},
                "logistic_regression": {"seuil": round(thresholds["logistic_regression"], 3),
                                        **evaluation.holdout_report(y_te, p_lr,
                                                                    thresholds["logistic_regression"])},
            },
        },
        "variables_correlees": {"paires_corr_sup_0_9": len(pairs),
                                "exemples": [list(p) for p in sorted(pairs, key=lambda p: -p[2])[:5]]},
        "explications": {
            "top5_random_forest": list(imp_rf.index[:5]),
            "top5_regression_logistique": list(imp_lr.index[:5]),
            "spearman_importances": round(agree.spearman, 3),
            f"variables_communes_top{agree.top_k}": agree.common,
            "cas_explique": {"index": case, "vraie_classe": verdict,
                             "p_malin_random_forest": round(float(p_rf[case]), 3),
                             "p_malin_regression_logistique": round(float(p_lr[case]), 3)},
        },
        "coefficients_signe_instable": {f: round(float(s), 2)
                                        for f, s in unstable.signe_stable.items()},
    }
    (out / "resultats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--repeats", type=int, default=10, help="répétitions de la validation croisée 5 plis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run(args.out, args.repeats, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nGraphiques et résultats enregistrés dans {args.out}/")


if __name__ == "__main__":
    main()
