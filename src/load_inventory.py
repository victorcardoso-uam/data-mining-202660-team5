"""
IIND4417 — Capstone Milestone 1, Team 5
src/load_inventory.py

Sourcing verification for the Big99 used-MFP inventory dataset.
Loads the secured export, validates the schema, and prints the evidence
required by Milestone 1 (row count, feature types, target balance).

From the project root:
    python src/load_inventory.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend: must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SOURCE_FILE = "inventario_mfp_usado.csv"

# Slow-mover threshold in days. Units above this are the positive class.
SLOW_MOVER_DAYS = 120

# Candidate cluster counts explored in the preliminary K-Means diagnostic.
K_RANGE = range(2, 11)
RANDOM_STATE = 42

# Palette, matching the Milestone 1 deck.
NAVY = "#1E2761"
AMBER = "#E08A3C"
ICE = "#8FA3D9"
MUTED = "#6B7280"

# Features proposed for the modelling phase, grouped by role.
CATEGORICAL_FEATURES = [
    "MARCA",
    "LINEA",
    "UBICACION",
    "APARIENCIA GRAL.",
    "CALIDAD COPIA",
    "EQUIPO FUNCIONANDO",
]

NUMERIC_FEATURES = [
    "CONTADOR",
    "COST W/FREIGHT",
]

MULTI_LABEL_FEATURE = "ACCESORIOS"
TARGET_SOURCE = "DIAS EN INVENTARIO"
ID_COLUMN = "CLAVE"


def clean_counter(series):
    """CONTADOR may arrive as text with non-breaking spaces and thousands commas.

    The released CSV is already cleaned, but the original Excel export is not,
    so this runs defensively and is a no-op on numeric input.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def read_source(path):
    """Read either the CSV release or the original Excel export."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the file in data/raw/ or "
            f"update SOURCE_FILE at the top of this script."
        )
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if extension in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension: {extension}")


def load_inventory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_path = os.path.join(project_dir, "data", "raw", SOURCE_FILE)

    print("=" * 78)
    print("BIG99 USED-MFP INVENTORY — SOURCING VERIFICATION")
    print("=" * 78)

    df = read_source(input_path)

    print(f"\nSource file : {SOURCE_FILE}")
    print(f"Source path : {input_path}")
    print(f"Provenance  : Big99 S.A. de C.V. internal inventory export")
    print(f"Shape       : {df.shape[0]} rows x {df.shape[1]} columns")

    required = (
        CATEGORICAL_FEATURES + NUMERIC_FEATURES
        + [MULTI_LABEL_FEATURE, TARGET_SOURCE, ID_COLUMN]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columns absent from the export: {missing}")

    df["CONTADOR"] = clean_counter(df["CONTADOR"])

    # Guard the numeric block before any statistic is computed.
    for column in NUMERIC_FEATURES + [TARGET_SOURCE]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    bad = df[NUMERIC_FEATURES + [TARGET_SOURCE]].isna().sum()
    if bad.any():
        raise ValueError(
            "Non-numeric values found in the numeric block after cleaning:\n"
            + bad[bad > 0].to_string()
        )

    # --------------------------------------------------------------------------
    # Row-count check against the Milestone 1 requirement
    # --------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("ROW COUNT CHECK")
    print("-" * 78)
    print(f"Rows secured       : {len(df)}")
    print(f"Milestone minimum  : 1000")
    if len(df) >= 1000:
        print("Status             : MEETS the 1,000-row requirement")
    else:
        print(f"Status             : SHORT by {1000 - len(df)} rows — "
              "widen the export filter before the pitch")

    # --------------------------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("LABELED FEATURE SCHEMA")
    print("-" * 78)

    schema = []
    for col in CATEGORICAL_FEATURES:
        schema.append({
            "feature": col,
            "role": "categorical",
            "dtype": str(df[col].dtype),
            "unique": df[col].nunique(),
            "nulls": int(df[col].isna().sum()),
        })
    for col in NUMERIC_FEATURES:
        schema.append({
            "feature": col,
            "role": "numeric",
            "dtype": str(df[col].dtype),
            "unique": df[col].nunique(),
            "nulls": int(df[col].isna().sum()),
        })
    schema.append({
        "feature": MULTI_LABEL_FEATURE,
        "role": "multi-label (Apriori)",
        "dtype": str(df[MULTI_LABEL_FEATURE].dtype),
        "unique": df[MULTI_LABEL_FEATURE].nunique(),
        "nulls": int(df[MULTI_LABEL_FEATURE].isna().sum()),
    })
    schema.append({
        "feature": TARGET_SOURCE,
        "role": "target source",
        "dtype": str(df[TARGET_SOURCE].dtype),
        "unique": df[TARGET_SOURCE].nunique(),
        "nulls": int(df[TARGET_SOURCE].isna().sum()),
    })
    print(pd.DataFrame(schema).to_string(index=False))

    # --------------------------------------------------------------------------
    # Target definition and class balance
    # --------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"TARGET VARIABLE: slow_mover = {TARGET_SOURCE} > {SLOW_MOVER_DAYS}")
    print("-" * 78)

    df["slow_mover"] = (df[TARGET_SOURCE] > SLOW_MOVER_DAYS).astype(int)
    balance = df["slow_mover"].value_counts().sort_index()
    for label, count in balance.items():
        name = "slow mover" if label else "healthy turnover"
        print(f"  class {label} ({name:<17}): {count:>4} units "
              f"({count / len(df) * 100:>5.1f}%)")

    print("\nThreshold sensitivity:")
    for threshold in (90, 120, 180, 365):
        positives = int((df[TARGET_SOURCE] > threshold).sum())
        print(f"  > {threshold:>3} days: {positives:>4} positives "
              f"({positives / len(df) * 100:>5.1f}%)")

    # --------------------------------------------------------------------------
    # Distribution diagnostics
    # --------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DISTRIBUTION DIAGNOSTICS")
    print("-" * 78)
    numeric_block = df[[TARGET_SOURCE] + NUMERIC_FEATURES]
    print(numeric_block.describe().to_string(
        float_format=lambda v: f"{v:>14.2f}"))
    print("\nSkewness (raw):")
    print(numeric_block.skew().to_string(
        float_format=lambda v: f"{v:>10.4f}"))
    print("\nSkewness after np.log1p:")
    print(np.log1p(numeric_block.clip(lower=0)).skew().to_string(
        float_format=lambda v: f"{v:>10.4f}"))

    # --------------------------------------------------------------------------
    # Business framing figures
    # --------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("CAPITAL EXPOSURE")
    print("-" * 78)
    total_capital = df["COST W/FREIGHT"].sum()
    print(f"Total inventory capital        : MXN {total_capital:>14,.2f}")
    for threshold in (120, 180, 365):
        subset = df[df[TARGET_SOURCE] > threshold]
        print(f"Capital held > {threshold:>3} days        : "
              f"MXN {subset['COST W/FREIGHT'].sum():>14,.2f}  "
              f"({len(subset):>3} units, "
              f"{subset['COST W/FREIGHT'].sum() / total_capital * 100:>4.1f}%)")

    print("\nBy warehouse:")
    by_site = df.groupby("UBICACION").agg(
        units=(ID_COLUMN, "count"),
        capital_mxn=("COST W/FREIGHT", "sum"),
        avg_days=(TARGET_SOURCE, "mean"),
        slow_mover_rate=("slow_mover", "mean"),
    )
    by_site["slow_mover_rate"] = by_site["slow_mover_rate"] * 100
    print(by_site.to_string(float_format=lambda v: f"{v:>14.2f}"))

    # --------------------------------------------------------------------------
    # Multi-label accessory field, for the Apriori branch
    # --------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("ACCESSORY CONFIGURATION FIELD (Apriori input)")
    print("-" * 78)
    tokens = (
        df[MULTI_LABEL_FEATURE].astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    tokens = tokens[tokens.ne("") & tokens.ne("nan")]
    print(f"Distinct configuration strings: {df[MULTI_LABEL_FEATURE].nunique()}")
    print(f"Distinct accessory tokens     : {tokens.nunique()}")
    print("\nMost frequent accessory tokens:")
    print(tokens.value_counts().head(10).to_string())

    print("\n" + "=" * 78)
    print("VERIFICATION COMPLETE")
    print("=" * 78)

    return df




def figures_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    path = os.path.join(project_dir, "reports", "figures")
    os.makedirs(path, exist_ok=True)
    return path


# ==============================================================================
# Figure 1 — Inventory age distribution and capital by warehouse
# ==============================================================================
def plot_inventory_profile(df):
    print("\n" + "-" * 78)
    print("FIGURE 1: INVENTORY AGE DISTRIBUTION & CAPITAL BY WAREHOUSE")
    print("-" * 78)

    out = os.path.join(figures_dir(), "inventory_profile.png")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left: age distribution, capped for readability ---
    ages = df[TARGET_SOURCE]
    cap = 400
    clipped = ages.clip(upper=cap)
    bins = np.arange(0, cap + 25, 25)

    counts, edges, patches = ax_left.hist(
        clipped, bins=bins, color=NAVY, alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )
    for patch, left_edge in zip(patches, edges[:-1]):
        if left_edge >= SLOW_MOVER_DAYS:
            patch.set_facecolor(AMBER)

    ax_left.axvline(
        x=SLOW_MOVER_DAYS, color="#C0392B", linestyle="--", linewidth=1.8,
        label=f"slow-mover threshold = {SLOW_MOVER_DAYS} days",
    )
    ax_left.axvline(
        x=ages.median(), color="#2E7D32", linestyle=":", linewidth=1.8,
        label=f"median = {ages.median():.0f} days",
    )

    n_beyond = int((ages > cap).sum())
    ax_left.text(
        cap - 8, counts.max() * 0.86,
        f"{n_beyond} units beyond {cap} days\n(max {ages.max():,.0f})",
        ha="right", fontsize=9, color=MUTED, style="italic",
    )

    ax_left.set_xlabel("Days in inventory", fontsize=11, fontweight="bold")
    ax_left.set_ylabel("Units", fontsize=11, fontweight="bold")
    ax_left.set_title(
        f"Inventory age distribution (n = {len(df)})",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax_left.grid(axis="y", linestyle=":", alpha=0.45)
    ax_left.set_axisbelow(True)
    ax_left.legend(fontsize=9, frameon=True, framealpha=0.95)

    # --- Right: capital by warehouse, split healthy vs stalled ---
    by_site = df.groupby("UBICACION").apply(
        lambda g: pd.Series({
            "healthy": g.loc[g["slow_mover"] == 0, "COST W/FREIGHT"].sum(),
            "stalled": g.loc[g["slow_mover"] == 1, "COST W/FREIGHT"].sum(),
            "units": len(g),
            "rate": g["slow_mover"].mean() * 100,
        }),
        include_groups=False,
    )
    by_site = by_site.sort_values("healthy", ascending=True)

    y = np.arange(len(by_site))
    healthy = by_site["healthy"].to_numpy() / 1000
    stalled = by_site["stalled"].to_numpy() / 1000

    ax_right.barh(y, healthy, color=NAVY, alpha=0.9,
                  edgecolor="white", linewidth=0.8, label="Healthy turnover")
    ax_right.barh(y, stalled, left=healthy, color=AMBER, alpha=0.95,
                  edgecolor="white", linewidth=0.8,
                  label=f"Stalled (> {SLOW_MOVER_DAYS} days)")

    for i, (h, st, row) in enumerate(zip(healthy, stalled, by_site.itertuples())):
        ax_right.text(
            h + st + 45, i,
            f"MXN {(h + st) * 1000:,.0f}   ·   {row.rate:.1f}% stalled",
            va="center", fontsize=9, color=NAVY,
        )

    ax_right.set_yticks(y)
    ax_right.set_yticklabels(
        [f"{site}\n({int(row.units)} units)"
         for site, row in zip(by_site.index, by_site.itertuples())],
        fontsize=10,
    )
    ax_right.set_xlabel("Landed capital (thousands of MXN)",
                        fontsize=11, fontweight="bold")
    ax_right.set_title(
        f"Capital by warehouse — MXN {df['COST W/FREIGHT'].sum():,.0f} total",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax_right.set_xlim(0, (healthy + stalled).max() * 1.42)
    ax_right.grid(axis="x", linestyle=":", alpha=0.45)
    ax_right.set_axisbelow(True)
    ax_right.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.95)

    fig.suptitle(
        "Big99 Used-MFP Inventory — Where the Capital Sits",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure written to: {out}")
    return out


# ==============================================================================
# Figure 2 — Preliminary K-Means diagnostic
# ==============================================================================
def plot_cluster_diagnostic(df):
    print("\n" + "-" * 78)
    print("FIGURE 2: PRELIMINARY K-MEANS DIAGNOSTIC")
    print("-" * 78)
    print("Exploratory only — this is Milestone 1 evidence that the clustering "
          "branch is viable, not a fitted deliverable.")

    # log1p first (the raw features are heavily right-skewed), then scale.
    block = df[[TARGET_SOURCE] + NUMERIC_FEATURES].clip(lower=0)
    logged = np.log1p(block)
    scaled = StandardScaler().fit_transform(logged)

    records = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(scaled)
        records.append({
            "K": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(scaled, labels),
        })
    tuning = pd.DataFrame(records).set_index("K")
    tuning["drop_pct"] = -tuning["inertia"].pct_change() * 100

    print("\nInertia and silhouette, K = "
          f"{K_RANGE[0]} to {K_RANGE[-1]} (random_state={RANDOM_STATE}):")
    print(tuning.to_string(float_format=lambda v: f"{v:>14.4f}"))

    peak_k = int(tuning["silhouette"].idxmax())
    print(f"\nPeak silhouette at K = {peak_k} "
          f"({tuning.loc[peak_k, 'silhouette']:.4f})")

    out = os.path.join(figures_dir(), "cluster_diagnostic.png")
    ks = tuning.index.to_numpy()

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5.8))

    ax_left.plot(ks, tuning["inertia"], color=NAVY, linewidth=2.2,
                 marker="o", markersize=6, markerfacecolor="white",
                 markeredgewidth=1.6, label="Inertia (WCSS)")
    ax_left.axvline(x=peak_k, color="#C0392B", linestyle="--", linewidth=1.8,
                    label=f"K = {peak_k}")
    ax_left.annotate(
        f"K = {peak_k}\nWCSS = {tuning.loc[peak_k, 'inertia']:.1f}",
        xy=(peak_k, tuning.loc[peak_k, "inertia"]),
        xytext=(26, 26), textcoords="offset points",
        fontsize=9, color="#8B3A3E", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#C0392B", linewidth=1.2),
    )
    ax_left.set_xlabel("Number of clusters (K)", fontsize=11, fontweight="bold")
    ax_left.set_ylabel("Within-cluster sum of squares",
                       fontsize=11, fontweight="bold")
    ax_left.set_title("Inertia elbow curve", fontsize=12,
                      fontweight="bold", pad=12)
    ax_left.set_xticks(ks)
    ax_left.grid(linestyle=":", alpha=0.45)
    ax_left.set_axisbelow(True)
    ax_left.legend(fontsize=9, frameon=True, framealpha=0.95)

    ax_right.plot(ks, tuning["silhouette"], color="#2E7D32", linewidth=2.2,
                  marker="s", markersize=6, markerfacecolor="white",
                  markeredgewidth=1.6, label="Silhouette coefficient")
    ax_right.axvline(x=peak_k, color="#C0392B", linestyle="--", linewidth=1.8,
                     label=f"Peak at K = {peak_k}")
    ax_right.annotate(
        f"{tuning.loc[peak_k, 'silhouette']:.4f}",
        xy=(peak_k, tuning.loc[peak_k, "silhouette"]),
        xytext=(26, -28), textcoords="offset points",
        fontsize=9, color="#8B3A3E", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#C0392B", linewidth=1.2),
    )
    ax_right.set_xlabel("Number of clusters (K)", fontsize=11, fontweight="bold")
    ax_right.set_ylabel("Silhouette coefficient", fontsize=11, fontweight="bold")
    ax_right.set_title("Silhouette coefficient by K", fontsize=12,
                       fontweight="bold", pad=12)
    ax_right.set_xticks(ks)
    ax_right.grid(linestyle=":", alpha=0.45)
    ax_right.set_axisbelow(True)
    ax_right.legend(fontsize=9, frameon=True, framealpha=0.95)

    fig.suptitle(
        "Preliminary Cluster Diagnostic — log1p + StandardScaler on "
        "age, page counter and landed cost",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure written to: {out}")
    return out


if __name__ == "__main__":
    inventory = load_inventory()
    plot_inventory_profile(inventory)
    plot_cluster_diagnostic(inventory)
    print("\n" + "=" * 78)
    print("FIGURES COMPLETE")
    print("=" * 78)