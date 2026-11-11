"""
Activity 21: ANN Architecture, Hyperparameter Optimization & Production Model Export
Data Mining & Modern AI Systems (IIND4417) - Team 5
Dataset: ann_team5_smartgrid.csv (750 rows, 6 continuous predictors)
Target: active_power_load_mw (MW)

Run from the repo root:  py src/activity_21_ann.py
Outputs:
  reports/activity_21_ann_log.txt
  reports/activity_21_ann_benchmark.csv
  reports/figures/activity_21_ann_diagnostics.png
  models/best_ann_model_team5.joblib
  models/ann_scaler_team5.joblib
"""
from pathlib import Path
import sys
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ann_team5_smartgrid.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
TARGET = "active_power_load_mw"
RANDOM_STATE = 42

# Training budget shared by all 5 candidates.
# With sklearn's default tol=1e-4 the optimizer stops after 37-166 epochs,
# before the loss has settled, so the ranking depends on when each model
# happened to stop. tol=1e-6 and max_iter=3000 let every model train until
# its loss is flat; all 5 models use the same rule.
MAX_ITER = 3000
TOL = 1e-6

# The 5 architectures fixed by the Activity 21 guide.
CANDIDATES = [
    {"model": "M1 Shallow Baseline", "hidden": (16,), "activation": "relu", "alpha": 1e-4, "early_stopping": False},
    {"model": "M2 Single Wide Layer", "hidden": (64,), "activation": "relu", "alpha": 1e-4, "early_stopping": False},
    {"model": "M3 Deep 2-Layer", "hidden": (64, 32), "activation": "relu", "alpha": 1e-4, "early_stopping": False},
    {"model": "M4 Deep Tanh", "hidden": (128, 64, 32), "activation": "tanh", "alpha": 1e-4, "early_stopping": False},
    {"model": "M5 Regularized Early-Stop", "hidden": (64, 32), "activation": "relu", "alpha": 0.01, "early_stopping": True},
]


def make_ann(cfg, seed=RANDOM_STATE, scale_target=True):
    mlp = MLPRegressor(
        hidden_layer_sizes=cfg["hidden"],
        activation=cfg["activation"],
        solver="adam",
        alpha=cfg["alpha"],
        early_stopping=cfg["early_stopping"],  # M5 holds out 10% of TRAIN only
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=seed,
    )
    if not scale_target:
        return mlp
    # The target (~325 MW) is also standardized inside the model so the output
    # neuron works on the same scale as the inputs; predict() returns MW.
    return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())


def n_params(n_inputs, hidden):
    sizes = [n_inputs, *hidden, 1]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def fit_quiet(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def main():
    out = []

    def log(line=""):
        print(line)
        out.append(line)

    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # ---------------- Step 0: load + split ----------------
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    p = X.shape[1]

    log("=" * 78)
    log("ACTIVITY 21 - ANN MODEL TRAINING | TEAM 5 SMART GRID")
    log("=" * 78)
    log(f"Dataset: {DATA_PATH.name} | rows={len(df)} | features={p} | nulls={int(df.isna().sum().sum())}")
    log(f"Target: {TARGET} (mean {y.mean():.1f} MW, std {y.std():.1f} MW)")
    log(f"Split 80/20, random_state={RANDOM_STATE}: train={len(X_train)} | test={len(X_test)}")
    log(f"scikit-learn {sklearn.__version__} | max_iter={MAX_ITER} | tol={TOL}")

    # ---------------- Step 1: StandardScaler (fit on TRAIN only) ----------------
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    log("\nSTEP 1 - FEATURE SCALING (StandardScaler fit on train only)")
    scale_tbl = pd.DataFrame({
        "raw_mean": X_train.mean(),
        "raw_std": X_train.std(ddof=0),
        "scaled_train_mean": X_train_s.mean(axis=0),
        "scaled_train_std": X_train_s.std(axis=0),
        "scaled_test_mean": X_test_s.mean(axis=0),
    })
    log(scale_tbl.to_string(float_format=lambda v: f"{v:,.3f}"))
    log("Train columns end at mean 0 / std 1. Test columns are close but not exact,")
    log("which is expected: the test set was transformed with the train statistics.")

    # Scaling ablation on M3 (relu) and M4 (tanh), same budget for every run.
    log("\nScaling ablation (M3 relu and M4 tanh, same max_iter/tol/seed):")
    ablation = [
        ("raw X, raw y", X_train.values, X_test.values, False),
        ("scaled X, raw y", X_train_s, X_test_s, False),
        ("scaled X, scaled y (used)", X_train_s, X_test_s, True),
    ]
    abl_rows = []
    for cfg in (CANDIDATES[2], CANDIDATES[3]):
        for label, xtr, xte, st in ablation:
            mdl, conv = fit_quiet(make_ann(cfg, scale_target=st), xtr, y_train)
            reg = mdl.regressor_ if st else mdl
            m = metrics(y_test, mdl.predict(xte))
            abl_rows.append({"model": cfg["model"][:2], "setup": label, "epochs": reg.n_iter_,
                             "converged": conv, "test_r2": m["r2"], "test_rmse": m["rmse"]})
    log(pd.DataFrame(abl_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------- Step 2: train the 5 ANNs ----------------
    log("\nSTEP 2 - 5-MODEL ANN TRAINING (same split for every model)")
    rows, fitted = [], {}
    for cfg in CANDIDATES:
        model, conv = fit_quiet(make_ann(cfg), X_train_s, y_train)
        reg = model.regressor_
        tr = metrics(y_train, model.predict(X_train_s))
        te = metrics(y_test, model.predict(X_test_s))
        fitted[cfg["model"]] = model
        rows.append({
            "model": cfg["model"],
            "hidden_layers": str(cfg["hidden"]),
            "activation": cfg["activation"],
            "alpha": cfg["alpha"],
            "early_stop": cfg["early_stopping"],
            "params": n_params(p, cfg["hidden"]),
            "epochs": reg.n_iter_,
            "converged": conv,
            "train_r2": tr["r2"],
            "test_r2": te["r2"],
            "test_adj_r2": adjusted_r2(te["r2"], len(y_test), p),
            "test_mae": te["mae"],
            "test_rmse": te["rmse"],
            "train_rmse": tr["rmse"],
        })
        log(f"  {cfg['model']:<27} epochs={reg.n_iter_:<5} converged={conv}")

    # ---------------- Step 3: benchmark table ----------------
    bench = pd.DataFrame(rows)
    bench.to_csv(REPORTS_DIR / "activity_21_ann_benchmark.csv", index=False)
    log("\nSTEP 3 - MULTI-MODEL BENCHMARK (holdout = 150 test rows)")
    show = bench[["model", "params", "epochs", "train_r2", "test_r2", "test_adj_r2", "test_mae", "test_rmse"]]
    log(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_idx = bench["test_rmse"].idxmin()
    best = bench.loc[best_idx]
    agree = {
        "test_r2": bench["test_r2"].idxmax() == best_idx,
        "test_adj_r2": bench["test_adj_r2"].idxmax() == best_idx,
        "test_mae": bench["test_mae"].idxmin() == best_idx,
    }
    log(f"\nSelection rule: lowest test RMSE -> {best['model']} "
        f"(RMSE {best['test_rmse']:.2f} MW, MAE {best['test_mae']:.2f} MW, R2 {best['test_r2']:.4f})")
    log(f"Same winner on the other holdout metrics: {agree}")
    log(f"Error relative to mean load: {best['test_rmse'] / y_test.mean():.2%} of {y_test.mean():.1f} MW")

    # Seed sensitivity: does the winner depend on the weight initialization?
    log("\nSeed sensitivity (same split, weight-init seed 0-4, test RMSE in MW):")
    sens = {}
    for seed in range(5):
        sens[f"seed {seed}"] = [
            mean_squared_error(y_test, fit_quiet(make_ann(cfg, seed=seed), X_train_s, y_train)[0].predict(X_test_s)) ** 0.5
            for cfg in CANDIDATES
        ]
    sens_df = pd.DataFrame(sens, index=[c["model"] for c in CANDIDATES])
    sens_df["mean"] = sens_df.mean(axis=1)
    sens_df["std"] = sens_df.iloc[:, :5].std(axis=1)
    log(sens_df.to_string(float_format=lambda v: f"{v:.2f}"))
    wins = sens_df.iloc[:, :5].idxmin(axis=0).value_counts().to_dict()
    log(f"Wins by seed: {wins}")

    # Linear reference (not one of the 5 candidates, used only as a sanity check).
    lin = LinearRegression().fit(X_train_s, y_train)
    lin_m = metrics(y_test, lin.predict(X_test_s))
    log(f"\nLinear regression reference on the same split: R2 {lin_m['r2']:.4f} | "
        f"MAE {lin_m['mae']:.2f} | RMSE {lin_m['rmse']:.2f} MW")

    # ---------------- Step 4: 2-panel diagnostic figure ----------------
    winner = fitted[best["model"]]
    y_pred = winner.predict(X_test_s)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2))
    for cfg in CANDIDATES:
        reg = fitted[cfg["model"]].regressor_
        lw = 2.6 if cfg["model"] == best["model"] else 1.4
        ax_a.plot(reg.loss_curve_, lw=lw, label=f"{cfg['model']} ({reg.n_iter_} ep.)")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Training loss (squared error / 2, standardized target, log scale)")
    ax_a.set_title("Panel A. Loss convergence (loss_curve_)")
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=8)

    lo, hi = min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10
    ax_b.scatter(y_test, y_pred, s=18, alpha=0.75, edgecolor="none")
    ax_b.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect prediction (y = x)")
    ax_b.set_xlim(lo, hi)
    ax_b.set_ylim(lo, hi)
    ax_b.set_xlabel("Actual active power load (MW)")
    ax_b.set_ylabel("Predicted active power load (MW)")
    ax_b.set_title(f"Panel B. Predicted vs actual, {best['model']} (test set, n={len(y_test)})")
    ax_b.text(0.04, 0.95, f"R2 = {best['test_r2']:.4f}\nMAE = {best['test_mae']:.2f} MW\nRMSE = {best['test_rmse']:.2f} MW",
              transform=ax_b.transAxes, va="top", fontsize=9,
              bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig_path = FIG_DIR / "activity_21_ann_diagnostics.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log(f"\nSTEP 4 - Figure saved: {fig_path.relative_to(ROOT)}")

    # ---------------- Step 5: serialization ----------------
    model_path = MODELS_DIR / "best_ann_model_team5.joblib"
    scaler_path = MODELS_DIR / "ann_scaler_team5.joblib"
    joblib.dump(winner, model_path)
    joblib.dump(scaler, scaler_path)

    # Reload check: the two files alone must reproduce the test predictions.
    m_back = joblib.load(model_path)
    s_back = joblib.load(scaler_path)
    reloaded = m_back.predict(s_back.transform(X_test))
    max_diff = float(np.abs(reloaded - y_pred).max())
    sample = X_test.iloc[[0]]
    log("\nSTEP 5 - SERIALIZATION")
    log(f"  {model_path.relative_to(ROOT)} ({model_path.stat().st_size / 1024:.0f} KB)")
    log(f"  {scaler_path.relative_to(ROOT)} ({scaler_path.stat().st_size / 1024:.1f} KB)")
    log(f"  Reload check: max |reloaded - original| = {max_diff:.2e} MW")
    log(f"  Example: first test row -> predicted {m_back.predict(s_back.transform(sample))[0]:.2f} MW, "
        f"actual {y_test.iloc[0]:.2f} MW")
    log("  Usage: model.predict(scaler.transform(X_new)) returns MW directly.")
    log("=" * 78)

    (REPORTS_DIR / "activity_21_ann_log.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
