"""Exploratory Data Analysis

Prints the key numbers in the terminal and saves the figures used in the
report to reports/figures/. Each section ends with the DECISION it led to
(those decisions are implemented in prepare.py).

We explore the RAW data (before cleaning), on the full dataset: EDA only
looks at the data, it does not fit anything, so it cannot leak into the model.

"""

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.prepare import FIGURES_DIR, TARGET, load_raw

sns.set_theme(style="whitegrid", palette="deep")


def title(text):
    """Print a visible section title in the terminal."""
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print(f"[figure saved] reports/figures/{name}")

# 1. Overview

def overview(df):
    title("1. Overview")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns\n")
    print(df.dtypes.to_string())
    print("\nFirst rows:")
    print(df.head(3).T.to_string())

# 2. Target: how imbalanced is it?

def target_distribution(df, target):
    title("2. Target distribution")
    counts = df[target].value_counts()
    print(pd.DataFrame({"count": counts, "share": (counts / len(df)).round(3)}).to_string())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=df, x=target, ax=ax)
    ax.set_title("Target distribution")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    save(fig, "eda_target_distribution.png")

    print(
        "\nDECISION: ~76% '<=50K' vs ~24% '>50K'. Always predicting '<=50K' gives 76% "
        "accuracy -> accuracy is misleading. Main metric = PR-AUC, plus ROC-AUC, F1, "
        "precision, recall, balanced accuracy (see prepare.compute_metrics)."
    )

# 3. Missing values: random or informative?

def missing_values(df, target):
    title("3. Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print((missing / len(df) * 100).round(2).astype(str) + " %")

    both = (df["workclass"].isna() & df["occupation"].isna()).sum()
    print(f"\nRows missing BOTH workclass and occupation: {both}")

    # Positive rate when the value is missing vs present
    print()
    for col in missing.index:
        rate_missing = (df.loc[df[col].isna(), target] == ">50K").mean()
        rate_present = (df.loc[df[col].notna(), target] == ">50K").mean()
        print(f"{col:<15} P(>50K | missing) = {rate_missing:.3f}   P(>50K | present) = {rate_present:.3f}")

    print(
        "\nDECISION: missingness is NOT random (same rows lack workclass and occupation, "
        "and they earn >50K much less often). We keep 'Missing' as its own category "
        "instead of dropping rows or imputing the mode (prepare.build_pipeline)."
    )

# 4. Duplicates

def duplicates(df):
    title("4. Duplicates")
    n = df.duplicated().sum()
    print(f"Exact duplicate rows: {n} ({n / len(df):.2%})")
    print(
        "\nDECISION: kept. In census data two different people can share the same "
        "attributes, and 0.1% has no real impact on the split."
    )

# 5. Numeric features

def numeric_features(df, target):
    title("5. Numeric features")
    num_cols = df.select_dtypes("number").columns.tolist()
    print(df[num_cols].describe().T.round(1).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.histplot(data=df, x=col, hue=target, bins=40, stat="density", common_norm=False, ax=ax)
        ax.set_title(col)
    save(fig, "eda_numeric_distributions.png")

    print()
    for col in ["capital-gain", "capital-loss"]:
        print(f"{col}: {(df[col] == 0).mean():.1%} zeros, max = {df[col].max():,}")
    print(f"hours-per-week == 40: {(df['hours-per-week'] == 40).mean():.1%} of rows")

    print(
        "\nDECISIONS:\n"
        "- capital-gain / capital-loss: >90% zeros + very long tail -> new features "
        "'capital_net' (signed log) and 'has_capital' (flag).\n"
        "- hours-per-week: big spike at 40 (full-time) -> new flag 'overtime' (> 40h).\n"
        "- fnlwgt: census sampling weight, not a property of the person, same "
        "distribution in both classes -> dropped."
    )
    return num_cols

# 6. Categorical features

def categorical_features(df, target):
    title("6. Categorical features")
    cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c != target]
    print("Number of categories per column:")
    print(df[cat_cols].nunique().sort_values().to_string())

    # Positive rate per category: which categories go with high income?
    global_rate = (df[target] == ">50K").mean()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    cols = ["workclass", "marital-status", "relationship", "occupation", "race", "sex"]
    for ax, col in zip(axes.ravel(), cols):
        rate = (df[target] == ">50K").groupby(df[col].fillna("Missing")).mean().sort_values()
        rate.plot.barh(ax=ax)
        ax.axvline(global_rate, ls="--", c="k", lw=1)  # global positive rate
        ax.set_title(f"P(>50K) by {col}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    save(fig, "eda_positive_rate_by_category.png")

    # Is "education" just a text version of "education-num"?
    levels_per_education = pd.crosstab(df["education"], df["education-num"]).gt(0).sum(axis=1)
    print(f"\nEach 'education' value maps to {levels_per_education.unique().tolist()} 'education-num' value(s)")

    print("\nnative-country, top 3:")
    print(df["native-country"].value_counts(normalize=True).head(3).round(3).to_string())

    print(
        "\nDECISIONS:\n"
        "- marital-status / relationship: married people earn >50K far more often -> strong predictors, kept.\n"
        "- education is a 1-to-1 copy of education-num -> 'education' dropped.\n"
        "- native-country: 91% United-States, 40 rare countries -> new flag 'native_us' "
        "instead of 41 sparse one-hot columns."
    )

# 7. Sensitive attributes (preview of the fairness audit)

def sensitive_attributes(df, target):
    title("7. Sensitive attributes")
    for col in ["sex", "race"]:
        summary = pd.DataFrame(
            {
                "share_of_data": df[col].value_counts(normalize=True),
                "P(>50K)": (df[target] == ">50K").groupby(df[col]).mean(),
            }
        ).round(3)
        print(summary.to_string(), "\n")

    print(
        "TAKEAWAY: large base-rate gaps already exist in the data (~30% of men vs ~11% of "
        "women earn >50K; women are 1/3 of rows; some race groups are <1%).\n"
        "Note: 'relationship' (Husband / Wife) is a near-perfect proxy for sex, so simply "
        "removing 'sex' would NOT make the model blind to it -> to discuss in the fairness audit."
    )

# 8. Correlations

def correlations(df, target, num_cols):
    title("8. Correlations (numeric features + target)")
    corr = df[num_cols].assign(target=(df[target] == ">50K").astype(int)).corr()
    print(corr["target"].drop("target").sort_values(ascending=False).round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric + target)")
    save(fig, "eda_correlations.png")

    print("\nTAKEAWAY: no strong correlation between numeric features (no multicollinearity issue).")


def main():
    df = load_raw()
    target = TARGET

    overview(df)
    target_distribution(df, target)
    missing_values(df, target)
    duplicates(df)
    num_cols = numeric_features(df, target)
    categorical_features(df, target)
    sensitive_attributes(df, target)
    correlations(df, target, num_cols)


if __name__ == "__main__":
    main()
