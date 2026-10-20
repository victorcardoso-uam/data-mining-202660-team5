"""
IIND4417 — Capstone Milestone 2, Team 5 (Big99 Inventory Intelligence)
src/capstone_modeling.py

Trains and validates the slow-mover classifier on the anonymised modelling
table produced by src/capstone_preprocessing.py.

Validation design (why the numbers can be trusted):
  * Units of one purchase lot share their arrival date and therefore their
    label, so every split is GROUPED BY LOT: no lot appears in both train
    and test.
  * Imputation, log1p, scaling and one-hot encoding live inside a sklearn
    Pipeline, so they are fitted on training folds only.
  * The decision threshold is chosen on out-of-fold training predictions,
    never on the test set.

Outputs: reports/capstone/*.png, metrics.json, modeling_log.txt

From the project root:
    python src/capstone_modeling.py
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "big99_model_input.csv"
OUT_DIR = ROOT / "reports" / "capstone"

RANDOM_STATE = 42
TARGET = "slow_mover"
GROUP = "lot_id"
CATEGORICAL = ["brand", "line", "warehouse", "appearance", "copy_quality", "working"]
NUMERIC = ["page_counter_k", "landed_cost_mxn"]
MIN_RECALL = 0.75          # Milestone 1 commitment
MILESTONE1_F1 = 0.70       # Milestone 1 commitment
MILESTONE1_SILHOUETTE = 0.40
BASELINE_SILHOUETTE = 0.305

NAVY, AMBER, ICE, MUTED, RED, GREEN = (
    "#1E2761", "#E08A3C", "#8FA3D9", "#6B7280", "#C0392B", "#2E7D32")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def build_preprocessor(accessory_cols):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
        ("num", numeric, NUMERIC),
        ("acc", "passthrough", accessory_cols),
    ])


def candidates():
    return {
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__max_depth": [3, 4, 5, 6], "model__min_samples_leaf": [5, 10, 20, 40]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [5, 10, 20]},
        ),
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=3000),
            {"model__C": [0.03, 0.1, 0.3, 1.0]},
        ),
    }


def pick_threshold(y_true, proba):
    """Highest-F1 threshold that still meets the recall commitment."""
    best = (0.5, -1.0, 0.0)
    for t in np.round(np.arange(0.10, 0.91, 0.01), 2):
        pred = proba >= t
        rec = recall_score(y_true, pred)
        f1 = f1_score(y_true, pred)
        if rec >= MIN_RECALL and f1 > best[1]:
            best = (t, f1, rec)
    return best


def scores(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "confusion": confusion_matrix(y_true, pred).tolist(),
    }


def section(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# ------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------
def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure -> {path.relative_to(ROOT)}")


def plot_confusion(test, champion_name):
    cm = np.array(test["confusion"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow([[1, 0], [0, 1]], cmap=matplotlib.colors.ListedColormap(["#F3F4F6", "#E8ECF8"]))
    labels = [["True negative\nhealthy, cleared", "False alarm\nflagged, sold fast"],
              ["Missed stall\nno warning", "Caught stall\nflagged in time"]]
    for i in range(2):
        for j in range(2):
            color = NAVY if i == j else (RED if i == 1 else AMBER)
            ax.text(j, i - 0.08, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=color)
            ax.text(j, i + 0.25, labels[i][j], ha="center", va="center",
                    fontsize=8.5, color=MUTED)
    ax.set_xticks([0, 1], ["Predicted healthy", "Predicted slow mover"], fontsize=9.5)
    ax.set_yticks([0, 1], ["Actual healthy", "Actual slow mover"], fontsize=9.5)
    ax.set_title(f"{champion_name} — hold-out lots (n = {cm.sum()})\n"
                 f"threshold {test['threshold']:.2f}  ·  recall {test['recall']:.2f}  ·  "
                 f"precision {test['precision']:.2f}  ·  F1 {test['f1']:.2f}",
                 fontsize=10, fontweight="bold", color=NAVY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "confusion_matrix.png")


def plot_curves(y_test, proba, test):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    fpr, tpr, roc_t = roc_curve(y_test, proba)
    left.plot(fpr, tpr, color=NAVY, lw=2.4, label=f"Champion (AUC = {test['roc_auc']:.3f})")
    left.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.4, label="Random guess (AUC = 0.500)")
    idx = np.argmin(np.abs(roc_t - test["threshold"]))
    left.scatter(fpr[idx], tpr[idx], s=70, color=AMBER, zorder=5,
                 label=f"Policy threshold {test['threshold']:.2f}")
    left.set(xlabel="False positive rate (healthy units flagged)",
             ylabel="True positive rate (recall on slow movers)")
    left.set_title("ROC curve", fontweight="bold", color=NAVY)
    left.legend(fontsize=8.5, loc="lower right")
    left.grid(ls=":", alpha=0.5)

    prec, rec, pr_t = precision_recall_curve(y_test, proba)
    base = y_test.mean()
    right.plot(rec, prec, color=NAVY, lw=2.4, label=f"Champion (PR-AUC = {test['pr_auc']:.3f})")
    right.axhline(base, color=MUTED, ls=":", lw=1.4, label=f"No-skill baseline ({base:.2f})")
    right.axvline(MIN_RECALL, color=RED, ls="--", lw=1.4, label=f"Recall target {MIN_RECALL:.2f}")
    right.scatter(test["recall"], test["precision"], s=70, color=AMBER, zorder=5,
                  label=f"Operating point (P {test['precision']:.2f}, R {test['recall']:.2f})")
    right.set(xlabel="Recall (slow movers caught)", ylabel="Precision (flags that are real)",
              ylim=(0, 1.02))
    right.set_title("Precision / recall trade-off", fontweight="bold", color=NAVY)
    right.legend(fontsize=8.5, loc="lower left")
    right.grid(ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "roc_pr_curves.png")


def plot_leakage(random_split, grouped):
    metrics = ["roc_auc", "f1", "recall"]
    names = ["ROC-AUC", "F1 (slow mover)", "Recall"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    a = [random_split[m] for m in metrics]
    b = [grouped[m] for m in metrics]
    ax.bar(x - 0.19, a, 0.36, color=ICE, label="Random row split (leaky)")
    ax.bar(x + 0.19, b, 0.36, color=NAVY, label="Split by purchase lot (honest)")
    for xi, (va, vb) in enumerate(zip(a, b)):
        ax.text(xi - 0.19, va + 0.015, f"{va:.2f}", ha="center", fontsize=9, color=MUTED)
        ax.text(xi + 0.19, vb + 0.015, f"{vb:.2f}", ha="center", fontsize=9,
                color=NAVY, fontweight="bold")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score on held-out units")
    ax.set_title("Same Random Forest, two validation designs\n"
                 "units of one lot share their label → random splits leak it",
                 fontsize=10, fontweight="bold", color=NAVY)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5)
    save(fig, "leakage_check.png")


def plot_importance(importance):
    top = importance.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.barh(top.index, top["mean"], xerr=top["std"], color=NAVY, alpha=0.9,
            error_kw={"ecolor": MUTED, "capsize": 3})
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("What drives stall risk (permutation importance, hold-out lots)",
                 fontsize=10, fontweight="bold", color=NAVY)
    ax.grid(axis="x", ls=":", alpha=0.5)
    save(fig, "feature_importance.png")


def plot_bands(bands):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(bands))
    colors = [GREEN, AMBER, RED]
    ax.bar(x, bands["slow_rate"] * 100, color=colors, width=0.6)
    for xi, row in zip(x, bands.itertuples()):
        ax.text(xi, row.slow_rate * 100 + 2,
                f"{row.slow_rate:.0%} stall\n{row.units} units · MXN {row.capital_mxn / 1e6:.2f}M",
                ha="center", fontsize=8.5, color=NAVY)
    ax.set_xticks(x, bands.index)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Actual slow-mover rate (%)")
    ax.set_title("Risk bands on out-of-fold scores (all 918 units)",
                 fontsize=10, fontweight="bold", color=NAVY)
    ax.grid(axis="y", ls=":", alpha=0.5)
    save(fig, "risk_bands.png")


def plot_silhouette(sil):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(sil.index, sil.values, color=NAVY, marker="o", lw=2.2,
            markerfacecolor="white", markeredgewidth=1.6, label="Extended feature space")
    ax.axhline(BASELINE_SILHOUETTE, color=MUTED, ls=":", lw=1.5,
               label=f"Milestone 1 baseline ({BASELINE_SILHOUETTE})")
    ax.axhline(MILESTONE1_SILHOUETTE, color=RED, ls="--", lw=1.5,
               label=f"Milestone 1 target ({MILESTONE1_SILHOUETTE})")
    ax.set(xlabel="Number of clusters (K)", ylabel="Silhouette coefficient",
           ylim=(0, 0.6))
    ax.set_xticks(sil.index)
    ax.set_title("K-Means cohesion after widening the feature space",
                 fontsize=10, fontweight="bold", color=NAVY)
    ax.legend(fontsize=8.5)
    ax.grid(ls=":", alpha=0.5)
    save(fig, "kmeans_silhouette.png")


# ------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    accessory_cols = [c for c in df.columns if c.startswith("acc_")]
    features = CATEGORICAL + NUMERIC + accessory_cols
    X, y, groups = df[features], df[TARGET], df[GROUP]

    print("=" * 78)
    print("BIG99 CAPSTONE — SLOW-MOVER CLASSIFIER (MILESTONE 2)")
    print("=" * 78)
    print(f"Rows {len(df)} | features {len(features)} | lots {groups.nunique()} | "
          f"positives {y.mean():.1%}")

    section("1. LOT-GROUPED HOLD-OUT SPLIT (~75/25)")
    outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(outer.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    g_tr = groups.iloc[train_idx]
    shared = set(g_tr) & set(groups.iloc[test_idx])
    print(f"Train {len(train_idx)} units / {g_tr.nunique()} lots | "
          f"Test {len(test_idx)} units / {groups.iloc[test_idx].nunique()} lots | "
          f"lots in both: {len(shared)}")
    print(f"Positive rate: train {y_tr.mean():.1%} | test {y_te.mean():.1%}")

    section("2. MODEL SELECTION (5-fold grouped CV on train, scoring = ROC-AUC)")
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted = {}
    for name, (model, grid) in candidates().items():
        pipe = Pipeline([("prep", build_preprocessor(accessory_cols)), ("model", model)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X_tr, y_tr, groups=g_tr)
        fitted[name] = search
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"  {name:<20} CV ROC-AUC {search.best_score_:.3f}  {params}")
    champion_name = max(fitted, key=lambda n: fitted[n].best_score_)
    champion = fitted[champion_name].best_estimator_
    print(f"Champion: {champion_name}")

    section("3. DECISION THRESHOLD (out-of-fold on train only)")
    oof = cross_val_predict(champion, X_tr, y_tr, groups=g_tr, cv=inner,
                            method="predict_proba")[:, 1]
    threshold, oof_f1, oof_rec = pick_threshold(y_tr, oof)
    print(f"Threshold {threshold:.2f}: OOF recall {oof_rec:.3f}, OOF F1 {oof_f1:.3f} "
          f"(highest F1 with recall >= {MIN_RECALL})")

    section("4. HOLD-OUT TEST RESULTS")
    proba = champion.predict_proba(X_te)[:, 1]
    train_scores = scores(y_tr, champion.predict_proba(X_tr)[:, 1], threshold)
    test = scores(y_te, proba, threshold)
    default = scores(y_te, proba, 0.5)
    print(f"{'':<26}{'F1':>8}{'Recall':>9}{'Prec.':>8}{'ROC-AUC':>9}{'PR-AUC':>8}")
    for label, s in [("Train (policy threshold)", train_scores),
                     ("Test @ 0.50", default), (f"Test @ {threshold:.2f} (policy)", test)]:
        print(f"{label:<26}{s['f1']:>8.3f}{s['recall']:>9.3f}{s['precision']:>8.3f}"
              f"{s['roc_auc']:>9.3f}{s['pr_auc']:>8.3f}")
    tn, fp, fn, tp = np.array(test["confusion"]).ravel()
    print(f"Confusion (policy): TN {tn} | FP {fp} | FN {fn} | TP {tp}")
    print(f"Milestone 1 targets: recall >= {MIN_RECALL} -> "
          f"{'MET' if test['recall'] >= MIN_RECALL else 'MISSED'} ({test['recall']:.2f}); "
          f"F1 >= {MILESTONE1_F1} -> "
          f"{'MET' if test['f1'] >= MILESTONE1_F1 else 'MISSED'} ({test['f1']:.2f})")

    section("5. ROBUSTNESS: 4-fold grouped CV over all lots (champion settings)")
    fold_rows = []
    for a, b in outer.split(X, y, groups):
        model = Pipeline(champion.steps).fit(X.iloc[a], y.iloc[a])
        fold_rows.append(scores(y.iloc[b], model.predict_proba(X.iloc[b])[:, 1], threshold))
    folds = pd.DataFrame(fold_rows)[["roc_auc", "f1", "recall", "precision"]]
    print(folds.agg(["mean", "std"]).round(3).to_string())

    section("6. LEAKAGE CHECK: random row split vs lot-grouped split")
    leaky_tr, leaky_te = train_test_split(np.arange(len(df)), test_size=0.25,
                                          stratify=y, random_state=RANDOM_STATE)
    rf = Pipeline([("prep", build_preprocessor(accessory_cols)),
                   ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                    random_state=RANDOM_STATE, n_jobs=-1))])
    leaky_cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    leaky_oof = cross_val_predict(rf, X.iloc[leaky_tr], y.iloc[leaky_tr], cv=leaky_cv,
                                  method="predict_proba")[:, 1]
    leaky_t = pick_threshold(y.iloc[leaky_tr], leaky_oof)[0]
    rf.fit(X.iloc[leaky_tr], y.iloc[leaky_tr])
    leaky = scores(y.iloc[leaky_te], rf.predict_proba(X.iloc[leaky_te])[:, 1], leaky_t)
    honest_fold = folds.mean().to_dict()
    print(f"Random split : ROC-AUC {leaky['roc_auc']:.3f} | F1 {leaky['f1']:.3f} | "
          f"recall {leaky['recall']:.3f}  <- inflated: same-lot twins on both sides")
    print(f"Lot-grouped  : ROC-AUC {honest_fold['roc_auc']:.3f} | F1 {honest_fold['f1']:.3f} | "
          f"recall {honest_fold['recall']:.3f}  <- reported number")

    section("7. FEATURE IMPORTANCE (permutation on hold-out, grouped by input)")
    perm = permutation_importance(champion, X_te, y_te, scoring="roc_auc",
                                  n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.DataFrame({"mean": perm.importances_mean, "std": perm.importances_std},
                              index=features).sort_values("mean", ascending=False)
    print(importance.head(8).round(4).to_string())

    section("8. RISK BANDS FOR THE OPERATING POLICY (out-of-fold, all units)")
    oof_all = cross_val_predict(Pipeline(champion.steps), X, y, groups=groups, cv=outer,
                                method="predict_proba")[:, 1]
    high = 0.60
    band = pd.cut(oof_all, [-0.01, threshold, high, 1.01],
                  labels=[f"Low (< {threshold:.2f})", f"Watch ({threshold:.2f}-{high:.2f})",
                          f"High (>= {high:.2f})"])
    capital = df["landed_cost_mxn"]
    bands = pd.DataFrame({"band": band, "slow": y, "capital": capital}).groupby(
        "band", observed=False).agg(units=("slow", "size"), slow_rate=("slow", "mean"),
                                    capital_mxn=("capital", "sum"))
    bands["stalled_capital_mxn"] = pd.DataFrame(
        {"band": band, "c": capital * y}).groupby("band", observed=False)["c"].sum()
    print(bands.round(3).to_string())
    flagged = oof_all >= threshold
    stalled_total = (capital * y).sum()
    caught = (capital * y)[flagged].sum()
    print(f"Stalled capital in stock: MXN {stalled_total:,.0f}")
    print(f"Flagged by the policy threshold: MXN {caught:,.0f} "
          f"({caught / stalled_total:.1%} of stalled capital)")

    section("9. K-MEANS ON THE EXTENDED FEATURE SPACE")
    Z = build_preprocessor(accessory_cols).fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    sil = pd.Series({k: silhouette_score(
        Z, KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z))
        for k in range(2, 9)})
    peak_k = int(sil.idxmax())
    print(sil.round(4).to_string())
    print(f"Peak silhouette {sil.max():.3f} at K = {peak_k} "
          f"(baseline {BASELINE_SILHOUETTE}, target {MILESTONE1_SILHOUETTE}) -> "
          f"{'MET' if sil.max() >= MILESTONE1_SILHOUETTE else 'MISSED'}")

    section("10. FIGURES")
    plot_confusion(test, champion_name)
    plot_curves(y_te, proba, test)
    plot_leakage(leaky, honest_fold)
    plot_importance(importance)
    plot_bands(bands)
    plot_silhouette(sil)

    metrics = {
        "champion": champion_name,
        "champion_params": {k.replace("model__", ""): v
                            for k, v in fitted[champion_name].best_params_.items()},
        "cv_auc": {n: round(s.best_score_, 4) for n, s in fitted.items()},
        "threshold": threshold, "train": train_scores, "test_default": default,
        "test_policy": test, "folds_mean": folds.mean().round(4).to_dict(),
        "folds_std": folds.std().round(4).to_dict(), "leaky_random_split": leaky,
        "risk_bands": bands.reset_index().astype({"band": str}).to_dict("records"),
        "stalled_capital_mxn": float(stalled_total), "flagged_stalled_mxn": float(caught),
        "silhouette": sil.round(4).to_dict(), "importance": importance.head(8).round(4).to_dict(),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print("\nMetrics -> reports/capstone/metrics.json")
    print("=" * 78)


def main():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run()
    log = buffer.getvalue()
    print(log, end="")
    (OUT_DIR / "modeling_log.txt").write_text(log)


if __name__ == "__main__":
    main()
