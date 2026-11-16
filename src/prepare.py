"""Toolbox shared by the three scripts (eda.py, train.py, evaluate.py).

This file is imported, never run on its own. It contains:
1. Settings          -> seed, split size, number of folds, folders
2. Data              -> download, cleaning, development / test split
3. Features          -> new features + preprocessing + the single Pipeline
4. Models            -> the 4 models we compare
5. Evaluation tools  -> metrics, cross-validation, threshold, run log
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Settings

SEED = 42          # one seed everywhere -> reproducible results
TEST_SIZE = 0.20   # 80% development / 20% test
N_FOLDS = 5        # 5-fold stratified cross-validation inside the development set
N_TRIALS = 15      # Optuna trials per model

TARGET = "class"
SENSITIVE = "sex"  

ROOT = Path(__file__).resolve().parents[1]   
DATA_DIR = ROOT / "data"                     
MODELS_DIR = ROOT / "models"                 
REPORTS_DIR = ROOT / "reports"              
FIGURES_DIR = REPORTS_DIR / "figures"
RUN_LOG = REPORTS_DIR / "run_log.csv"

for folder in (DATA_DIR, MODELS_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# 2. Data

def load_raw():
    """Download Adult from OpenML once, then read the local copy."""
    path = DATA_DIR / "adult_raw.csv"
    if not path.exists():
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        adult.frame.to_csv(path, index=False)
    return pd.read_csv(path)


def load_splits():
    """Return X_dev, y_dev, X_test, y_test.

    Cleaning done here is row by row, so it cannot leak information:
    - target: 1 if income > 50K, else 0
    - drop 'fnlwgt'    : census sampling weight, not a property of the person
    - drop 'education' : exact duplicate of 'education-num'
    Everything that LEARNS from the data (imputation, scaling, encoding) is in
    the Pipeline instead, so it is fitted on training folds only.

    Split: stratified, so both parts keep ~24% of positives. The test set is
    never used for a decision; validation is done by cross-validation inside
    the development set.
    """
    df = load_raw()
    y = (df[TARGET] == ">50K").astype(int)
    X = df.drop(columns=[TARGET, "fnlwgt", "education"])
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return (
        X_dev.reset_index(drop=True),
        y_dev.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# 3. Features and Pipeline

def add_features(X):
    """Feature engineering: 4 new features justified by the EDA.

    capital_net : signed log of (capital gain - capital loss).
                  92% of zeros and a huge tail -> the log squashes the tail.
    has_capital : 1 if the person has any capital gain or loss (wealth signal).
    overtime    : 1 if more than 40 hours per week (47% work exactly 40h).
    native_us   : 1 if born in the United States (91% of rows; the 40 other
                  countries are too rare for one-hot encoding).
    """
    X = X.copy()
    net = X["capital-gain"] - X["capital-loss"]
    X["capital_net"] = np.sign(net) * np.log1p(np.abs(net))
    X["has_capital"] = ((X["capital-gain"] > 0) | (X["capital-loss"] > 0)).astype(int)
    X["overtime"] = (X["hours-per-week"] > 40).astype(int)
    X["native_us"] = (X["native-country"] == "United-States").astype(int)
    return X.drop(columns=["capital-gain", "capital-loss", "native-country"])


# Columns after feature engineering
NUMERIC = ["age", "education-num", "hours-per-week", "capital_net"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]
BINARY = ["has_capital", "overtime", "native_us"]


def build_pipeline(model, scale=False, sampler=None):
    """ONE Pipeline: new features -> preprocessing -> (resampling) -> model.

    - Numeric    : median imputation, and scaling only when the model needs it
                   (logistic regression; trees do not care about scale).
    - Categorical: missing values become their own category 'Missing' (the EDA
                   shows they are informative), then one-hot encoding.
    - sampler    : optional over / under-sampler. imblearn's Pipeline applies
                   it to the training folds only, never to validation rows.

    Because everything is in one Pipeline, cross-validation re-fits the
    preprocessing inside each fold: no leakage.
    """
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    categorical = [
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline(numeric), NUMERIC),
            ("cat", Pipeline(categorical), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ],
        verbose_feature_names_out=False,
    )
    steps = [("features", FunctionTransformer(add_features)), ("preprocess", preprocess)]
    if sampler is not None:
        return ImbPipeline(steps + [("sampler", sampler), ("model", model)])
    return Pipeline(steps + [("model", model)])


# 4. Models

MODEL_NAMES = ["logreg", "decision_tree", "random_forest", "lightgbm"]

# Reasonable starting hyperparameters (before Optuna)
DEFAULT_PARAMS = {
    "logreg": {"C": 1.0},
    "decision_tree": {"max_depth": 8, "min_samples_leaf": 20},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 5},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
}


def make_pipeline(name, params=None, sampler=None):
    """Build the full Pipeline for one of our models."""
    params = DEFAULT_PARAMS.get(name, {}) if params is None else params
    if name == "dummy":            # no-skill floor (session 1)
        return build_pipeline(DummyClassifier(strategy="prior"))
    if name == "logreg":           # linear baseline (sessions 2-3)
        model = LogisticRegression(max_iter=3000, random_state=SEED, **params)
        return build_pipeline(model, scale=True, sampler=sampler)
    if name == "decision_tree":    # one tree (session 4)
        model = DecisionTreeClassifier(random_state=SEED, **params)
    elif name == "random_forest":  # many trees averaged (session 5)
        model = RandomForestClassifier(n_jobs=-1, random_state=SEED, **params)
    elif name == "lightgbm":       # gradient boosting (sessions 5-6)
        model = LGBMClassifier(random_state=SEED, verbose=-1, **params)
    return build_pipeline(model, sampler=sampler)


# 5. Evaluation tools

def compute_metrics(y_true, proba, threshold=0.5):
    """The same metrics for every model.

    Accuracy alone is misleading here: always predicting '<=50K' gives 76%.
    PR-AUC (average precision) is our main metric; its no-skill value is the
    positive rate (0.24).
    """
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }


def cross_validate(pipe, X, y):
    """5-fold stratified cross-validation of a complete Pipeline.

    In each fold, a fresh copy of the Pipeline is fitted on 4 folds and
    predicts the 5th one. Returns:
    - oof    : 'out-of-fold' probabilities. Every development row is predicted
               by a model that never saw it.
    - scores : mean and standard deviation of PR-AUC and ROC-AUC over the folds.
    """
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    pr_auc, roc_auc = [], []
    for train_idx, val_idx in folds.split(X, y):
        model = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        pr_auc.append(average_precision_score(y.iloc[val_idx], oof[val_idx]))
        roc_auc.append(roc_auc_score(y.iloc[val_idx], oof[val_idx]))
    scores = {
        "cv_pr_auc": np.mean(pr_auc),
        "cv_pr_auc_std": np.std(pr_auc),
        "cv_roc_auc": np.mean(roc_auc),
    }
    return oof, scores


def best_f1_threshold(y_true, proba):
    """Decision threshold with the highest F1 (session 7: threshold tuning).

    We have no business costs for false positives / negatives on this dataset,
    so we treat them as comparable and use F1. Always computed on out-of-fold
    predictions of the development set, never on the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def log_run(model, stage, params, metrics, artifact):
    """Run log (requirement 7): one CSV line per training / tuning trial."""
    is_new = not RUN_LOG.exists()
    with open(RUN_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "model", "stage", "params", "metrics", "artifact"])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                model,
                stage,
                json.dumps(params),
                json.dumps({k: round(float(v), 4) for k, v in metrics.items()}),
                artifact,
            ]
        )
