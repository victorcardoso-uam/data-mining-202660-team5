"""
IIND4417 — Capstone Milestone 3, Team 5 (Big99 Inventory Intelligence)
capstone_streamlit_app/train_models.py

Trains every model the dashboard serves and serializes it to models/*.joblib,
so the app only loads artifacts and never trains at start-up.

Validation design, inherited from Milestone 2:
  * Hold-out and every CV fold are grouped by purchase lot, because units of
    one lot share their arrival date and therefore their label.
  * Preprocessing is fitted on training data only.
  * Decision thresholds come from out-of-fold training predictions.

From capstone_streamlit_app/:
    python train_models.py
"""
import copy
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
    r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP = Path(__file__).resolve().parent
SOURCE = APP.parent / "data" / "processed" / "big99_model_input.csv"
DATA = APP / "data" / "team_dataset.csv"
MODELS = APP / "models"

RANDOM_STATE = 42
TARGET = "slow_mover_flag"
GROUP = "lot_id"
AGE = "days_in_stock"
CATEGORICAL = ["brand", "line", "warehouse", "appearance", "copy_quality", "working"]
NUMERIC = ["page_counter_k", "landed_cost_mxn"]
MIN_RECALL = 0.75
HIGH_FLOOR = 0.60
ANN_MAX_EPOCHS = 300
ANN_PATIENCE = 30


def build_preprocessor(accessories):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
         CATEGORICAL),
        ("num", numeric, NUMERIC),
        ("acc", "passthrough", accessories),
    ])


def pick_threshold(y_true, scores, grid):
    """Highest-F1 cut that still meets the recall commitment from Milestone 1."""
    best_t, best_f1 = float(np.median(scores)), -1.0
    for t in grid:
        pred = scores >= t
        if recall_score(y_true, pred) >= MIN_RECALL and f1_score(y_true, pred) > best_f1:
            best_t, best_f1 = float(t), f1_score(y_true, pred)
    return best_t


PROB_GRID = np.round(np.arange(0.10, 0.91, 0.01), 2)


def fit_ann(Z, y, groups):
    """Train an MLP epoch by epoch, stopping on the loss of held-out lots."""
    fit_idx, val_idx = next(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(Z, y, groups))
    y = np.asarray(y)
    net = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, learning_rate_init=0.001,
                        batch_size=32, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    train_loss, val_loss = [], []
    best, best_loss, best_epoch = None, np.inf, 0
    for epoch in range(1, ANN_MAX_EPOCHS + 1):
        order = rng.permutation(fit_idx)
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            net.partial_fit(Z[batch], y[batch], classes=[0, 1])
        train_loss.append(log_loss(y[fit_idx], net.predict_proba(Z[fit_idx]), labels=[0, 1]))
        val_loss.append(log_loss(y[val_idx], net.predict_proba(Z[val_idx]), labels=[0, 1]))
        if val_loss[-1] < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(net), val_loss[-1], epoch
        elif epoch - best_epoch >= ANN_PATIENCE:
            break
    return best, {"train_loss": train_loss, "val_loss": val_loss, "best_epoch": best_epoch}


def ann_oof(X, y, groups, accessories, cv):
    oof = np.zeros(len(X))
    for fit, hold in cv.split(X, y, groups):
        prep = build_preprocessor(accessories).fit(X.iloc[fit])
        net, _ = fit_ann(prep.transform(X.iloc[fit]), y.iloc[fit], groups.iloc[fit])
        oof[hold] = net.predict_proba(prep.transform(X.iloc[hold]))[:, 1]
    return oof


def evaluate(y_true, scores, threshold, Z_test, r2):
    pred = (scores >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int))
    sil = silhouette_score(Z_test, pred) if len(set(pred)) == 2 else None
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "r2": r2,
        "silhouette": sil,
        "confusion": confusion_matrix(y_true, pred).tolist(),
        "fpr": fpr[keep].round(4).tolist(),
        "tpr": tpr[keep].round(4).tolist(),
    }


def main():
    warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
    warnings.filterwarnings("ignore", message="Got `batch_size`")
    MODELS.mkdir(exist_ok=True)
    if SOURCE.exists():
        DATA.parent.mkdir(exist_ok=True)
        pd.read_csv(SOURCE).rename(columns={"slow_mover": TARGET}).to_csv(DATA, index=False)
    df = pd.read_csv(DATA)
    accessories = [c for c in df.columns if c.startswith("acc_")]
    features = CATEGORICAL + NUMERIC + accessories
    X, y, groups = df[features], df[TARGET], df[GROUP]
    log_age = np.log1p(df[AGE])

    outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(outer.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    g_tr = groups.iloc[train_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Train {len(X_tr)} units / {g_tr.nunique()} lots | "
          f"Test {len(X_te)} units / {groups.iloc[test_idx].nunique()} lots | "
          f"shared lots {len(set(g_tr) & set(groups.iloc[test_idx]))}")

    prep = build_preprocessor(accessories).fit(X_tr)
    Z_tr, Z_te = prep.transform(X_tr), prep.transform(X_te)
    results, fitted, policy = {}, {}, {}

    grids = {
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                          {"model__max_depth": [3, 4, 5, 6],
                           "model__min_samples_leaf": [5, 10, 20, 40]}),
        "Random Forest": (RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=-1),
                          {"model__max_depth": [4, 6, 8],
                           "model__min_samples_leaf": [5, 10, 20]}),
    }
    for name, (model, grid) in grids.items():
        pipe = Pipeline([("prep", build_preprocessor(accessories)), ("model", model)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X_tr, y_tr, groups=g_tr)
        oof = cross_val_predict(search.best_estimator_, X_tr, y_tr, groups=g_tr, cv=inner,
                                method="predict_proba")[:, 1]
        threshold = pick_threshold(y_tr, oof, PROB_GRID)
        fitted[name] = clone(model).set_params(
            **{k.replace("model__", ""): v for k, v in search.best_params_.items()}).fit(Z_tr, y_tr)
        scores = fitted[name].predict_proba(Z_te)[:, 1]
        results[name] = evaluate(y_te, scores, threshold, Z_te, r2_score(y_te, scores))
        policy[name] = {"kind": "probability", "watch": threshold,
                        "high": max(HIGH_FLOOR, round(threshold + 0.10, 2))}
        print(f"{name}: {search.best_params_} threshold {threshold:.2f}")

    # Multiple Linear Regression predicts log1p(days in stock); R-squared is its native metric.
    mlr_pipe = Pipeline([("prep", build_preprocessor(accessories)), ("model", LinearRegression())])
    # Folds come from the binary label: a stratified splitter rejects a continuous target.
    folds = list(inner.split(X_tr, y_tr, g_tr))
    oof = cross_val_predict(mlr_pipe, X_tr, log_age.iloc[train_idx], cv=folds)
    grid = np.unique(np.quantile(oof, np.linspace(0.02, 0.98, 193)))
    cut = pick_threshold(y_tr, oof, grid)
    fitted["MLR"] = LinearRegression().fit(Z_tr, log_age.iloc[train_idx])
    scores = fitted["MLR"].predict(Z_te)
    results["MLR"] = evaluate(y_te, scores, cut, Z_te, r2_score(log_age.iloc[test_idx], scores))
    policy["MLR"] = {"kind": "regression", "watch": cut, "high": cut + 0.25,
                     "watch_days": float(np.expm1(cut))}
    print(f"MLR: flag at predicted {np.expm1(cut):.0f} days")

    oof = ann_oof(X_tr, y_tr, g_tr, accessories, inner)
    threshold = pick_threshold(y_tr, oof, PROB_GRID)
    fitted["ANN"], curves = fit_ann(Z_tr, y_tr, g_tr)
    scores = fitted["ANN"].predict_proba(Z_te)[:, 1]
    results["ANN"] = evaluate(y_te, scores, threshold, Z_te, r2_score(y_te, scores))
    policy["ANN"] = {"kind": "probability", "watch": threshold,
                     "high": max(HIGH_FLOOR, round(threshold + 0.10, 2))}
    print(f"ANN: stopped at epoch {len(curves['train_loss'])}, best {curves['best_epoch']}, "
          f"threshold {threshold:.2f}")

    # One hold-out of 24 lots is noisy, so the champion is chosen on 4-fold grouped CV.
    cv_auc = {name: [] for name in results}
    for fit, hold in outer.split(X, y, groups):
        fold_prep = build_preprocessor(accessories).fit(X.iloc[fit])
        Z_fit, Z_hold = fold_prep.transform(X.iloc[fit]), fold_prep.transform(X.iloc[hold])
        for name in ("Decision Tree", "Random Forest"):
            model = clone(fitted[name]).fit(Z_fit, y.iloc[fit])
            cv_auc[name].append(roc_auc_score(y.iloc[hold], model.predict_proba(Z_hold)[:, 1]))
        mlr = LinearRegression().fit(Z_fit, log_age.iloc[fit])
        cv_auc["MLR"].append(roc_auc_score(y.iloc[hold], mlr.predict(Z_hold)))
        net, _ = fit_ann(Z_fit, y.iloc[fit], groups.iloc[fit])
        cv_auc["ANN"].append(roc_auc_score(y.iloc[hold], net.predict_proba(Z_hold)[:, 1]))
    for name, values in cv_auc.items():
        results[name]["cv_auc_mean"] = float(np.mean(values))
        results[name]["cv_auc_std"] = float(np.std(values, ddof=1))
    champion = max(cv_auc, key=lambda n: np.mean(cv_auc[n]))
    print(f"Champion by 4-fold grouped CV ROC-AUC: {champion}")

    joblib.dump(prep, MODELS / "preprocessor.joblib", compress=3)
    joblib.dump(prep.named_transformers_["num"].named_steps["scale"],
                MODELS / "standard_scaler.joblib", compress=3)
    files = {"Random Forest": "random_forest_model.joblib", "Decision Tree": "decision_tree_model.joblib",
             "MLR": "mlr_model.joblib", "ANN": "ann_model.joblib"}
    for name, file in files.items():
        joblib.dump(fitted[name], MODELS / file, compress=3)

    # Real hold-out units used as one-click what-if presets.
    rf_scores = fitted["Random Forest"].predict_proba(Z_te)[:, 1]
    columns = features + [AGE]
    test_rows = df.iloc[test_idx][columns].reset_index(drop=True)
    presets = {"low": test_rows.iloc[int(np.argmin(rf_scores))].to_dict(),
               "high": test_rows.iloc[int(np.argmax(rf_scores))].to_dict()}

    meta = {
        "files": files, "features": features, "categorical": CATEGORICAL, "numeric": NUMERIC,
        "accessories": accessories, "encoded_columns": [n.split("__", 1)[1] for n in prep.get_feature_names_out()],
        "results": results, "policy": policy, "champion": champion, "ann_curves": curves, "presets": presets,
        "sklearn_version": sklearn.__version__, "train_units": len(train_idx), "test_units": len(test_idx),
        "test_lots": int(groups.iloc[test_idx].nunique()), "train_lots": int(g_tr.nunique()),
        "imputed_median_counter": float(prep.named_transformers_["num"].named_steps["impute"].statistics_[0]),
        "imputed_median_cost": float(prep.named_transformers_["num"].named_steps["impute"].statistics_[1]),
    }
    (MODELS / "metrics.json").write_text(json.dumps(meta, indent=1, default=float))

    table = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_auc_mean",
                                     "cv_auc_std", "r2", "silhouette"]]
    print(table.astype(float).round(3).to_string())


if __name__ == "__main__":
    main()
