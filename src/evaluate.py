"""Evaluation of the final model: test metrics, calibration, SHAP, fairness.

STEP 1 - Final metrics on the test set
STEP 2 - Calibration: reliability curve, Brier score, ECE (requirement 9)
STEP 3 - SHAP: global importance + local explanations (requirement 8)
STEP 4 - Fairness audit by sex + mitigation, before / after (requirement 10)
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score

from src.prepare import (
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    SENSITIVE,
    compute_metrics,
    load_splits,
)


# STEP 1 - Final metrics on the test set

def test_metrics(y_test, proba, threshold, cv_pr_auc):
    table = pd.DataFrame(
        {
            "threshold 0.5": compute_metrics(y_test, proba, 0.5),
            f"tuned threshold {threshold:.3f}": compute_metrics(y_test, proba, threshold),
        }
    ).T.round(4)
    table.to_csv(REPORTS_DIR / "test_metrics.csv")
    print("\nSTEP 1 - Final model on the untouched test set\n")
    print(table.to_string())
    print(f"\nCV PR-AUC on development was {cv_pr_auc:.4f}; test PR-AUC is {table['pr_auc'].iloc[0]:.4f}.")

    ConfusionMatrixDisplay.from_predictions(
        y_test, proba >= threshold, display_labels=["<=50K", ">50K"], cmap="Blues"
    )
    plt.title(f"Confusion matrix (test, threshold {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

# STEP 2 - Calibration

def calibration(y_test, proba):
    """Do predicted probabilities match reality?"""

    observed, predicted = calibration_curve(y_test, proba, n_bins=10)
    brier = np.mean((proba - y_test) ** 2)

    # ECE = average |predicted - observed| over the bins, weighted by bin size
    bins = np.minimum((proba * 10).astype(int), 9)
    sizes = np.bincount(bins, minlength=10)[np.unique(bins)] / len(proba)
    ece = np.sum(sizes * np.abs(predicted - observed))

    table = pd.DataFrame({"mean_predicted": predicted, "observed": observed})
    table["gap"] = table["mean_predicted"] - table["observed"]  # > 0: model over-estimates
    print("\nSTEP 2 - Calibration on the test set\n")
    print(table.round(3).to_string(index=False))
    print(f"\nBrier score = {brier:.4f}   ECE = {ece:.4f}")

    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "--", c="grey", label="perfect calibration")
    plt.plot(predicted, observed, "o-", label="final model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed share of >50K")
    plt.title(f"Reliability curve (test)  Brier = {brier:.3f}, ECE = {ece:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=150)
    plt.close()

# STEP 3 - SHAP

def shap_analysis(pipe, X_test, y_test, proba, threshold):
    """SHAP value = how much one feature pushed one prediction up or down."""

    sample = X_test.sample(2000, random_state=SEED)       # enough for stable results
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    X_transformed = pd.DataFrame(pipe[:-1].transform(sample), columns=names, index=sample.index)

    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    explanation = explainer(X_transformed)
    if explanation.values.ndim == 3:                      # some models return one array per class
        explanation = explanation[:, :, 1]

    # Global: which features matter most on average?
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.Series(np.abs(explanation.values).mean(axis=0), index=names).sort_values(ascending=False)
    print("\nSTEP 3 - SHAP\n\nGlobal importance (mean |SHAP|, top 10):")
    print(importance.head(10).round(3).to_string())

    # Local: one person of each kind (the most confident one)
    p = pd.Series(proba, index=X_test.index)[sample.index]
    y = y_test[sample.index]
    pred = (p >= threshold).astype(int)
    people = {
        "true_positive": p[(pred == 1) & (y == 1)].idxmax(),
        "true_negative": p[(pred == 0) & (y == 0)].idxmin(),
        "false_positive": p[(pred == 1) & (y == 0)].idxmax(),
        "false_negative": p[(pred == 0) & (y == 1)].idxmin(),
    }
    for kind, idx in people.items():
        row = sample.index.get_loc(idx)
        shap.plots.waterfall(explanation[row], max_display=8, show=False)
        plt.title(f"{kind}: predicted probability = {p[idx]:.2f}")
        plt.savefig(FIGURES_DIR / f"shap_local_{kind}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print("\nLocal explanations saved: reports/figures/shap_local_*.png")

# STEP 4 - Fairness

def fairness_table(y_true, pred, groups):
    """Metrics per group, plus the gap between groups (max - min).

    selection_rate gap -> demographic parity
    recall (TPR) gap   -> equal opportunity (the definition we target)
    """
    frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "recall_TPR": true_positive_rate,
            "FPR": false_positive_rate,
            "precision": precision_score,
        },
        y_true=y_true,
        y_pred=pred,
        sensitive_features=groups,
    )
    table = frame.by_group.astype(float)
    table.loc["GAP"] = frame.difference()
    return table.round(3)


def fairness(y_dev, oof_dev, groups_dev, y_test, proba_test, groups_test, threshold):
    """Audit by sex, then mitigation with one threshold per group."""
    pred_before = (proba_test >= threshold).astype(int)
    before = fairness_table(y_test, pred_before, groups_test)
    print(f"\nSTEP 4 - Fairness by {SENSITIVE} on the test set\n\nBefore (single threshold {threshold:.3f}):")
    print(before.to_string())

    # A recall of r means: catch the top r share of a group's true positives,
    # so the threshold is the (1 - r) quantile of their scores.
    target_recall = recall_score(y_dev, oof_dev >= threshold)
    group_thresholds = {}
    for group in groups_dev.unique():
        scores_of_positives = oof_dev[(groups_dev == group) & (y_dev == 1)]
        group_thresholds[group] = float(np.quantile(scores_of_positives, 1 - target_recall))

    pred_after = (proba_test >= groups_test.map(group_thresholds)).astype(int)
    after = fairness_table(y_test, pred_after, groups_test)
    rounded = {g: round(t, 3) for g, t in group_thresholds.items()}
    print(f"\nAfter (group thresholds {rounded}):")
    print(after.to_string())

    summary = pd.DataFrame(
        {
            "before": {
                "equal_opportunity_gap": before.loc["GAP", "recall_TPR"],
                "demographic_parity_gap": before.loc["GAP", "selection_rate"],
                "precision": precision_score(y_test, pred_before),
                "recall": recall_score(y_test, pred_before),
                "f1": f1_score(y_test, pred_before),
            },
            "after": {
                "equal_opportunity_gap": after.loc["GAP", "recall_TPR"],
                "demographic_parity_gap": after.loc["GAP", "selection_rate"],
                "precision": precision_score(y_test, pred_after),
                "recall": recall_score(y_test, pred_after),
                "f1": f1_score(y_test, pred_after),
            },
        }
    ).round(4)
    summary.to_csv(REPORTS_DIR / "fairness_before_after.csv")
    print("\nBefore / after on the test set:\n")
    print(summary.to_string())

    # Figure: recall per group, before and after
    recalls = pd.DataFrame({"before": before["recall_TPR"], "after": after["recall_TPR"]}).drop("GAP")
    recalls.plot.bar(rot=0, figsize=(5, 4), title=f"Recall by {SENSITIVE} (test set)")
    plt.ylabel("Recall (TPR)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_recall_by_sex.png", dpi=150)
    plt.close()


def main():
    X_dev, y_dev, X_test, y_test = load_splits()
    pipe = joblib.load(MODELS_DIR / "final_model.joblib")
    oof_dev = np.load(MODELS_DIR / "final_model_oof.npy")
    with open(REPORTS_DIR / "final_model.json") as f:
        info = json.load(f)
    threshold = info["threshold"]
    print(f"Final model: {info['model']}, threshold = {threshold:.3f}")

    proba = pipe.predict_proba(X_test)[:, 1]

    test_metrics(y_test, proba, threshold, info["cv_pr_auc"])
    calibration(y_test, proba)
    shap_analysis(pipe, X_test, y_test, proba, threshold)
    fairness(y_dev, oof_dev, X_dev[SENSITIVE], y_test, proba, X_test[SENSITIVE], threshold)


if __name__ == "__main__":
    main()
