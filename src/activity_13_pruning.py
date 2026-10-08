"""
Activity 13: Decision Tree Regularization & Cost-Complexity Pruning
Data Mining & Modern AI Systems (IIND4417) — Team 5
Dataset: Smart Grid Substation Disturbance (Big99 / Grid)
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "pruning_team5_smartgrid.csv"
REPORTS_DIR = ROOT / "reports"
TARGET = "grid_disturbance_flag"
RANDOM_STATE = 42

CANDIDATES = [
    ("alpha_1 (Unconstrained)", 0.001),
    ("alpha_2 (Optimal Pruned)", 0.015),
    ("alpha_3 (Over-Pruned)", 0.080),
]


def load_and_split():
    """Step 2: load the team CSV and perform a stratified 75/25 split."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


def run_experiment(X_train, X_test, y_train, y_test):
    """Steps 3-5: fit one tree per alpha, measure complexity, error and cost."""
    rows = []
    for label, alpha in CANDIDATES:
        clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

        n_leaves = clf.get_n_leaves()
        train_error = 1.0 - clf.score(X_train, y_train)
        test_error = 1.0 - clf.score(X_test, y_test)
        rows.append({
            "candidate": label,
            "ccp_alpha": alpha,
            "leaves": n_leaves,
            "max_depth": clf.get_depth(),
            "train_error": train_error,
            "test_error": test_error,
            "total_cost": test_error + alpha * n_leaves,
        })
    return pd.DataFrame(rows)


def build_report(df, X_train, X_test, y_train, results):
    best = results.loc[results["total_cost"].idxmin()]
    table = results.to_string(
        index=False,
        formatters={
            "ccp_alpha": "{:.3f}".format,
            "train_error": "{:.2%}".format,
            "test_error": "{:.2%}".format,
            "total_cost": "{:.4f}".format,
        },
    )
    lines = [
        "=" * 78,
        "ACTIVITY 13 — COST-COMPLEXITY PRUNING | TEAM 5 SMART GRID",
        "=" * 78,
        f"Dataset: {DATA_PATH.name} | rows={df.shape[0]} | features={X_train.shape[1]}",
        f"Features: {', '.join(X_train.columns)}",
        f"Target: {TARGET}",
        f"Train size: {len(X_train)} | Test size: {len(X_test)}",
        f"Class balance (train): {y_train.value_counts(normalize=True).round(3).to_dict()}",
        "",
        "Total cost: R_alpha(T) = R_test(T) + alpha * |T|",
        "",
        table,
        "",
        f"Optimal alpha*: {best['ccp_alpha']:.3f} -> {best['candidate']} "
        f"(test error {best['test_error']:.2%}, total cost {best['total_cost']:.4f})",
        "=" * 78,
    ]
    return "\n".join(lines)


def main():
    df, X_train, X_test, y_train, y_test = load_and_split()
    results = run_experiment(X_train, X_test, y_train, y_test)
    report = build_report(df, X_train, X_test, y_train, results)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "activity_13_pruning_results.csv", index=False)
    (REPORTS_DIR / "activity_13_pruning_log.txt").write_text(report + "\n")
    print(f"\nSaved results and log to {REPORTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
