"""Training: imbalance remedies, model comparison, Optuna tuning, final model.

STEP 1 - Imbalance remedies (requirement 3)
STEP 2 - Comparison of 4 models with default hyperparameters (requirement 4)
STEP 3 - Optuna tuning of the 4 models + run log (requirements 6 and 7)
STEP 4 - Final model and decision threshold
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import PrecisionRecallDisplay

from src.prepare import (
    FIGURES_DIR,
    MODEL_NAMES,
    MODELS_DIR,
    N_TRIALS,
    REPORTS_DIR,
    SEED,
    best_f1_threshold,
    compute_metrics,
    cross_validate,
    load_splits,
    log_run,
    make_pipeline,
)


def evaluate(pipe, X, y):
    """Cross-validate a pipeline and summarise it in one line of metrics."""
    oof, scores = cross_validate(pipe, X, y)
    at_05 = compute_metrics(y, oof, threshold=0.5)
    threshold = best_f1_threshold(y, oof)
    metrics = {
        **scores,
        "precision": at_05["precision"],   # at the default threshold 0.5
        "recall": at_05["recall"],
        "f1": at_05["f1"],
        "brier": at_05["brier"],
        "f1_best_threshold": compute_metrics(y, oof, threshold)["f1"],
    }
    return metrics, oof


# STEP 1 - Imbalance remedies (session 7)

def compare_remedies(X, y):
    """Same model (LightGBM), four ways of handling the 76 / 24 imbalance.

    SMOTE is not tested: it creates points 'between' two rows, which has no
    meaning for one-hot encoded categories (a limit mentioned in session 7).
    """
    ratio = (y == 0).sum() / (y == 1).sum()  # class-ratio heuristic, about 3.2
    base = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31}
    remedies = {
        "none": make_pipeline("lightgbm", base),
        "class_weight": make_pipeline("lightgbm", {**base, "scale_pos_weight": float(ratio)}),
        "undersampling": make_pipeline("lightgbm", base, sampler=RandomUnderSampler(random_state=SEED)),
        "oversampling": make_pipeline("lightgbm", base, sampler=RandomOverSampler(random_state=SEED)),
    }
    results = {}
    for name, pipe in remedies.items():
        results[name], _ = evaluate(pipe, X, y)
        log_run("lightgbm", f"remedy_{name}", base, results[name], "reports/imbalance_remedies.csv")

    table = pd.DataFrame(results).T.round(4)
    table.to_csv(REPORTS_DIR / "imbalance_remedies.csv")
    print("\nSTEP 1 - Imbalance remedies (5-fold CV, LightGBM)\n")
    print(table.to_string())
    print(
        "\nReading: PR-AUC and ROC-AUC do not move, so no remedy ranks people better.\n"
        "Weights and resampling only trade precision for recall at threshold 0.5, and\n"
        "they make the probabilities worse (higher Brier). Tuning the threshold gives\n"
        "the same F1 (last column) without that cost.\n"
        "DECISION: no reweighting; PR-AUC as main metric + a tuned decision threshold."
    )


# STEP 2 - Model comparison, default hyperparameters

def compare_models(X, y):
    results = {}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    print("\nSTEP 2 - Model comparison (default hyperparameters)")
    for name in ["dummy"] + MODEL_NAMES:
        print(f"  cross-validating {name}...")
        results[name], oof = evaluate(make_pipeline(name), X, y)
        log_run(name, "default_params", {}, results[name], "reports/model_comparison.csv")
        if name != "dummy":
            PrecisionRecallDisplay.from_predictions(y, oof, name=name, ax=ax)

    ax.axhline(y.mean(), ls="--", c="grey", label="no skill")
    ax.legend(loc="lower left")
    ax.set_title("Precision-Recall curves (out-of-fold predictions)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_pr_curves.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame(results).T.round(4)
    table.to_csv(REPORTS_DIR / "model_comparison.csv")
    print("\n" + table.to_string())


# STEP 3 - Optuna tuning (session 6)

def suggest_params(trial, name):
    """The hyperparameters Optuna is allowed to try, for each model."""
    if name == "logreg":
        # C = inverse of the regularisation strength (small C = strong penalty)
        return {"C": trial.suggest_float("C", 0.001, 100, log=True)}
    if name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": 200,
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5]),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }


def tune_model(name, X, y):
    """Optuna study for one model. One trial = 5-fold CV of the whole Pipeline."""

    def objective(trial):
        params = suggest_params(trial, name)
        _, scores = cross_validate(make_pipeline(name, params), X, y)
        trial.set_user_attr("params", params)
        trial.set_user_attr("scores", scores)
        return scores["cv_pr_auc"]  # Optuna maximises the mean CV PR-AUC

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial

    # Artifact 1: plot of all the trials of this model
    values = [t.value for t in study.trials]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(values, "o", label="trial")
    ax.plot(np.maximum.accumulate(values), label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean CV PR-AUC")
    ax.set_title(f"Optuna - {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"optuna_{name}.png", dpi=150)
    plt.close(fig)

    # Artifact 2: the best configuration, refitted on the whole development set
    best_params = best.user_attrs["params"]
    best_pipe = make_pipeline(name, best_params).fit(X, y)
    joblib.dump(best_pipe, MODELS_DIR / f"{name}_tuned.joblib")

    # Run log: one line per trial, with the artifact it corresponds to
    for t in study.trials:
        if t.number == best.number:
            stage, artifact = f"optuna_trial_{t.number:02d}_BEST", f"models/{name}_tuned.joblib"
        else:
            stage, artifact = f"optuna_trial_{t.number:02d}", f"reports/figures/optuna_{name}.png"
        log_run(name, stage, t.user_attrs["params"], t.user_attrs["scores"], artifact)

    print(f"  {name:<14} best CV PR-AUC = {best.value:.4f}   {best_params}")
    return best_params, best.user_attrs["scores"], best_pipe


def main():
    X_dev, y_dev, _, _ = load_splits()  # the test set is NOT used here
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    compare_remedies(X_dev, y_dev)
    compare_models(X_dev, y_dev)

    print(f"\nSTEP 3 - Optuna tuning ({N_TRIALS} trials per model, 5-fold CV each)")
    tuned = {name: tune_model(name, X_dev, y_dev) for name in MODEL_NAMES}
    table = pd.DataFrame({name: scores for name, (_, scores, _) in tuned.items()}).T.round(4)
    table = table.sort_values("cv_pr_auc", ascending=False)
    table.to_csv(REPORTS_DIR / "tuned_models.csv")
    print("\n" + table.to_string())

    # STEP 4 - Final model = best mean CV PR-AUC
    best_name = table.index[0]
    best_params, best_scores, best_pipe = tuned[best_name]

    # Threshold chosen on out-of-fold predictions: each development row is
    # scored by a model that never saw it, so the choice is not optimistic.
    oof, _ = cross_validate(make_pipeline(best_name, best_params), X_dev, y_dev)
    threshold = best_f1_threshold(y_dev, oof)

    print(f"\nSTEP 4 - Final model: {best_name}, decision threshold = {threshold:.3f}")
    compare = pd.DataFrame(
        {"threshold 0.5": compute_metrics(y_dev, oof, 0.5), "tuned threshold": compute_metrics(y_dev, oof, threshold)}
    ).T[["precision", "recall", "f1", "accuracy"]]
    print(compare.round(4).to_string())

    # Saved for evaluate.py
    joblib.dump(best_pipe, MODELS_DIR / "final_model.joblib")
    np.save(MODELS_DIR / "final_model_oof.npy", oof)
    with open(REPORTS_DIR / "final_model.json", "w") as f:
        json.dump({"model": best_name, "params": best_params, "threshold": threshold, **best_scores}, f, indent=2)
    log_run(best_name, "final_model", {**best_params, "threshold": threshold}, best_scores, "models/final_model.joblib")


if __name__ == "__main__":
    main()
