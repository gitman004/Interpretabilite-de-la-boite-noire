import json

import numpy as np
import pandas as pd
import pytest

from interpretabilite import data, evaluation, explain
from interpretabilite.__main__ import run
from interpretabilite.models import MODELS


@pytest.fixture(scope="module")
def ds():
    return data.load()


@pytest.fixture(scope="module")
def fitted(ds):
    X_tr, X_te, y_tr, y_te = data.holdout(ds)
    rf = MODELS["random_forest"]().fit(X_tr, y_tr)
    lr = MODELS["logistic_regression"]().fit(X_tr, y_tr)
    return X_tr, X_te, y_tr, y_te, rf, lr


# --- Données -----------------------------------------------------------------------

def test_malignant_is_the_positive_class(ds):
    # Le jeu contient 212 tumeurs malignes et 357 bénignes
    assert ds.y.sum() == 212 and len(ds.y) == 569
    assert ds.X.shape[1] == 30


def test_holdout_is_stratified(ds):
    _, _, y_tr, y_te = data.holdout(ds)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


# --- Évaluation --------------------------------------------------------------------

def test_sensitivity_counts_detected_malignant_tumours():
    y = np.array([1, 1, 1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.4, 0.7, 0.1, 0.6])
    s = evaluation.score(y, proba)
    assert s["sensibilite"] == 0.75          # 3 malins détectés sur 4
    assert s["precision"] == 0.75            # 1 faux positif sur 4 alertes


def test_cross_validation_produces_one_row_per_fold_and_model(ds):
    folds, coefs = evaluation.cross_validate(MODELS, ds.X, ds.y, n_splits=3, n_repeats=1)
    assert len(folds) == 3 * len(MODELS)
    assert list(coefs.columns) == ds.feature_names and len(coefs) == 3


def test_corrected_t_test_identical_models_is_not_significant():
    folds = pd.DataFrame({"fold": [0, 1, 2, 0, 1, 2], "model": ["a"] * 3 + ["b"] * 3,
                          "roc_auc": [0.9, 0.95, 0.92] * 2})
    c = evaluation.corrected_t_test(folds, "a", "b", "roc_auc", test_fraction=0.2)
    assert c.mean_diff == 0 and c.p_value == 1.0 and c.a_wins == 0


def test_corrected_t_test_is_more_conservative_than_naive_test():
    rng = np.random.default_rng(0)
    d = 0.004 + rng.normal(0, 0.01, 50)
    folds = pd.DataFrame({"fold": list(range(50)) * 2, "model": ["a"] * 50 + ["b"] * 50,
                          "roc_auc": np.r_[0.95 + d, np.full(50, 0.95)]})
    from scipy import stats
    naive = stats.ttest_1samp(d, 0).pvalue
    corrected = evaluation.corrected_t_test(folds, "a", "b", "roc_auc", test_fraction=0.2).p_value
    assert corrected > naive


def test_coefficient_stability_flags_sign_flips():
    coefs = pd.DataFrame({"stable": [1.0, 1.2, 0.9, 1.1], "instable": [0.3, -0.2, 0.25, 0.1]})
    out = evaluation.coefficient_stability(coefs)
    assert out.loc["stable", "signe_stable"] == 1.0
    assert out.loc["instable", "signe_stable"] == 0.75
    assert out.index[0] == "stable"          # trié par force du coefficient


def test_threshold_reaches_target_sensitivity_on_training_data(fitted):
    X_tr, _, y_tr, _, _, _ = fitted
    model = MODELS["logistic_regression"]()
    t = evaluation.threshold_for_sensitivity(model, X_tr, y_tr, target=0.97)
    assert 0 < t < 0.5                       # plus bas que le seuil par défaut
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    assert (proba[y_tr.to_numpy() == 1] >= t).mean() >= 0.97


def test_correlated_pairs_finds_size_measures(ds):
    pairs = {(a, b) for a, b, _ in evaluation.correlated_pairs(ds.X, 0.95)}
    assert ("mean radius", "mean perimeter") in pairs


def test_holdout_report_confusion_matrix_adds_up(fitted):
    _, X_te, _, y_te, rf, _ = fitted
    r = evaluation.holdout_report(y_te, rf.predict_proba(X_te)[:, 1])
    assert r["vrais_positifs"] + r["faux_negatifs"] == y_te.sum()
    assert sum(r[k] for k in ("vrais_positifs", "faux_negatifs", "faux_positifs", "vrais_negatifs")) == len(y_te)


# --- Explications ------------------------------------------------------------------

def test_shap_values_add_up_to_forest_probability(fitted):
    _, X_te, _, _, rf, _ = fitted
    e = explain.explain_forest(rf, X_te)
    reconstructed = e.base_value + e.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, rf.predict_proba(X_te)[:, 1], atol=1e-6)


def test_shap_values_add_up_to_logistic_log_odds(fitted):
    X_tr, X_te, _, _, _, lr = fitted
    e = explain.explain_logistic(lr, X_tr, X_te)
    reconstructed = e.base_value + e.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, lr.decision_function(X_te), atol=1e-6)


def test_importance_is_normalised(fitted):
    _, X_te, _, _, rf, _ = fitted
    imp = explain.explain_forest(rf, X_te).importance()
    assert imp.sum() == pytest.approx(1.0) and imp.is_monotonic_decreasing


def test_agreement_of_identical_rankings():
    s = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    a = explain.agreement(s, s, k=2)
    assert a.spearman == pytest.approx(1.0) and a.common == ["a", "b"]


def test_pick_case_prefers_disagreement_then_errors():
    y = [1, 0, 1, 0]
    assert explain.pick_case(y, [0.9, 0.2, 0.6, 0.1], [0.9, 0.2, 0.4, 0.1]) == 2   # désaccord
    assert explain.pick_case(y, [0.9, 0.7, 0.8, 0.1], [0.9, 0.6, 0.8, 0.1]) == 1   # erreur commune
    assert explain.pick_case(y, [0.9, 0.2, 0.55, 0.1], [0.9, 0.2, 0.6, 0.1]) == 2  # le plus incertain


# --- Pipeline complet --------------------------------------------------------------

def test_run_writes_report_and_figures(tmp_path):
    report = run(tmp_path, repeats=1)
    saved = json.loads((tmp_path / "resultats.json").read_text(encoding="utf-8"))
    assert saved["validation_croisee"]["plis"] == 5 == report["validation_croisee"]["plis"]
    assert len(list(tmp_path.glob("*.png"))) == 6
