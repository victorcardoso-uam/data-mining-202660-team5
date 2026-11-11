"""
IIND4417 — Capstone Milestone 3, Team 5
Big99 Inventory Intelligence: slow-mover decision dashboard.

Loads pre-trained artifacts from models/*.joblib (see train_models.py); nothing
is trained when the app starts.

From capstone_streamlit_app/:
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

APP = Path(__file__).resolve().parent
MODELS = APP / "models"
TARGET = "slow_mover_flag"
AGE = "days_in_stock"
SLOW_DAYS = 120
DISCOUNT = 0.25
GREEN, RED, AMBER, BLUE, VIOLET = "#10B981", "#EF4444", "#E08A3C", "#8FA3D9", "#C084FC"
STATUS = {0: "Healthy turnover", 1: "Slow mover (> 120 days)"}
STATUS_COLORS = {STATUS[0]: GREEN, STATUS[1]: RED}
MODEL_COLORS = {"Random Forest": AMBER, "Decision Tree": BLUE, "MLR": GREEN, "ANN": VIOLET}
FAMILY = {"Random Forest": "Classical ensemble", "Decision Tree": "Classical tree",
          "MLR": "Classical regression", "ANN": "Neural network"}
LABELS = {"page_counter_k": "Page counter (thousands)", "landed_cost_mxn": "Landed cost (MXN)",
          AGE: "Days in stock", "brand": "Brand", "line": "Product line",
          "warehouse": "Warehouse", "appearance": "Appearance", "copy_quality": "Copy quality"}

st.set_page_config(page_title="Big99 Inventory Intelligence", page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data
def load_data():
    return pd.read_csv(APP / "data" / "team_dataset.csv")


@st.cache_data
def load_metrics():
    return json.loads((MODELS / "metrics.json").read_text())


@st.cache_resource
def load_artifacts(files):
    """Deserialize the fitted preprocessor, scaler and every model once per server."""
    models = {name: joblib.load(MODELS / file) for name, file in files.items()}
    return joblib.load(MODELS / "preprocessor.joblib"), joblib.load(MODELS / "standard_scaler.joblib"), models


def raw_score(name, row):
    """Stall probability, or predicted log1p(days) for the regression model."""
    Z = preprocessor.transform(row)
    if policy[name]["kind"] == "regression":
        return float(models[name].predict(Z)[0])
    return float(models[name].predict_proba(Z)[0, 1])


def risk_band(name, score):
    if score >= policy[name]["high"]:
        return "HIGH"
    return "WATCH" if score >= policy[name]["watch"] else "LOW"


df = load_data()
meta = load_metrics()
preprocessor, scaler, models = load_artifacts(meta["files"])
policy, results = meta["policy"], meta["results"]
model_names = sorted(results, key=lambda n: -results[n]["cv_auc_mean"])

# ------------------------------------------------------------------------------
# Session state: what-if inputs survive tab switches and script reruns
# ------------------------------------------------------------------------------
DEFAULTS = {
    "wi_days": 60,
    "wi_cost": int(round(df["landed_cost_mxn"].median(), -2)),
    "wi_counter": int(df["page_counter_k"].median()),
    "wi_brand": df["brand"].mode()[0],
    "wi_line": df["line"].mode()[0],
    "wi_warehouse": df["warehouse"].mode()[0],
    "wi_appearance": df["appearance"].mode()[0],
    "wi_quality": df["copy_quality"].mode()[0],
    "wi_working": "SI",
    "wi_acc": ["acc_duplex", "acc_dsdf"],
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("history", [])
st.session_state.setdefault("active_model", meta["champion"])


def reset_scenario():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def load_preset(which):
    row = meta["presets"][which]
    counter = row["page_counter_k"]
    st.session_state.update({
        "wi_days": int(min(row[AGE], 730)),
        "wi_cost": int(min(round(row["landed_cost_mxn"], -2), 45000)),
        "wi_counter": DEFAULTS["wi_counter"] if pd.isna(counter) else int(min(counter, 1700)),
        "wi_brand": row["brand"], "wi_line": row["line"], "wi_warehouse": row["warehouse"],
        "wi_appearance": row["appearance"], "wi_quality": row["copy_quality"], "wi_working": row["working"],
        "wi_acc": [c for c in meta["accessories"] if row[c] == 1],
    })


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
st.sidebar.title("📦 Big99 Inventory Intelligence")
st.sidebar.caption("IIND4417 · Capstone Milestone 3 · Team 5")
st.sidebar.divider()
st.sidebar.selectbox("Active model", model_names, key="active_model",
                     format_func=lambda n: f"{n} ★ champion" if n == meta["champion"] else n)
active = st.session_state["active_model"]
rule = policy[active]
if rule["kind"] == "regression":
    st.sidebar.caption(f"Flags a unit when predicted days in stock ≥ {rule['watch_days']:.0f}.")
else:
    st.sidebar.caption(f"WATCH from {rule['watch']:.2f} · HIGH from {rule['high']:.2f} stall probability. "
                       "Cut-offs were tuned on out-of-fold training lots.")
st.sidebar.divider()
st.sidebar.markdown("**Serialized artifacts loaded**")
st.sidebar.code("\n".join(["preprocessor.joblib", "standard_scaler.joblib", *meta["files"].values()]), language=None)
if meta["sklearn_version"] != sklearn.__version__:
    st.sidebar.warning(f"Models were trained with scikit-learn {meta['sklearn_version']}; "
                       f"this server runs {sklearn.__version__}. Run: pip install -r requirements.txt")

# ------------------------------------------------------------------------------
# Header and KPI cards
# ------------------------------------------------------------------------------
st.title("Big99 Inventory Intelligence")
st.caption("Procurement & Commercial Director · Used-MFP slow-mover early warning across MID, MEX, GDL and MTY")

total_capital = df["landed_cost_mxn"].sum()
stalled = df.loc[df[TARGET] == 1, "landed_cost_mxn"].sum()
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Units in stock", f"{len(df):,}", f"{df['lot_id'].nunique()} purchase lots", delta_color="off")
kpi2.metric("Landed capital", f"MXN {total_capital / 1e6:.2f}M", "4 warehouses", delta_color="off")
kpi3.metric("Stalled > 120 days", f"MXN {stalled / 1e6:.2f}M", f"{stalled / total_capital:.1%} of capital",
            delta_color="off")
kpi4.metric("Active model ROC-AUC", f"{results[active]['cv_auc_mean']:.3f}",
            f"{active} · 4-fold grouped CV", delta_color="off")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer & EDA", "⚙️ Preprocessing & Correlation",
                                  "🤖 Multi-Model Benchmark", "🔮 What-If Inference"])

# ------------------------------------------------------------------------------
# Tab 1: Data Explorer
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Anonymised inventory table")
    left, right = st.columns([3, 2])
    left.dataframe(df.head(100), width="stretch", height=330)
    right.markdown("**Summary statistics**")
    right.dataframe(df[["page_counter_k", "landed_cost_mxn", AGE, TARGET]].describe().T.round(2),
                    width="stretch")

    st.divider()
    st.subheader("Feature distribution by turnover status")
    view = df.assign(status=df[TARGET].map(STATUS))
    choice = st.selectbox("Feature", list(LABELS), format_func=LABELS.get, key="eda_feature")
    plot_data = view.copy()
    if choice == AGE:
        plot_data[AGE] = plot_data[AGE].clip(upper=730)
        st.caption("Days in stock are capped at 730 for readability; the oldest unit has 2,526.")
    if pd.api.types.is_numeric_dtype(plot_data[choice]):
        fig = px.histogram(plot_data, x=choice, color="status", barmode="overlay", nbins=40,
                           color_discrete_map=STATUS_COLORS, labels={choice: LABELS[choice], "status": ""})
    else:
        counts = plot_data.groupby([choice, "status"]).size().reset_index(name="units")
        fig = px.bar(counts, x=choice, y="units", color="status", barmode="stack",
                     color_discrete_map=STATUS_COLORS, labels={choice: LABELS[choice], "status": "", "units": "Units"})
        fig.update_xaxes(categoryorder="total descending")
    fig.update_yaxes(title="Units")
    fig.update_layout(legend=dict(orientation="h", y=1.1), margin=dict(t=40))
    st.plotly_chart(fig, width="stretch", key="eda_hist")

    site = view.groupby(["warehouse", "status"])["landed_cost_mxn"].sum().reset_index()
    fig_site = px.bar(site, y="warehouse", x="landed_cost_mxn", color="status", orientation="h",
                      color_discrete_map=STATUS_COLORS, title="Landed capital by warehouse",
                      labels={"landed_cost_mxn": "Landed capital (MXN)", "warehouse": "", "status": ""})
    fig_site.update_layout(legend=dict(orientation="h", y=1.15), height=320)
    st.plotly_chart(fig_site, width="stretch", key="eda_site")

# ------------------------------------------------------------------------------
# Tab 2: Preprocessing
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Leak-free feature pipeline")
    st.caption(f"Imputer, scaler and encoder were fitted on {meta['train_units']} training units only "
               f"and are served from models/preprocessor.joblib.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Missing-value treatment**")
        nulls = df[meta["numeric"]].isna().sum()
        st.dataframe(pd.DataFrame({
            "Missing": nulls,
            "Treatment": ["Median imputation", "Median imputation"],
            "Train median": [meta["imputed_median_counter"], meta["imputed_median_cost"]],
        }), width="stretch")
        st.caption("A page counter of 0 on a used MFP is a missing reading, so it is treated as NaN.")
    with col2:
        st.markdown("**StandardScaler parameters** (after log1p)")
        st.dataframe(pd.DataFrame({"Mean": scaler.mean_, "Std. deviation": scaler.scale_},
                                  index=meta["numeric"]).round(4), width="stretch")
        encoded = meta["encoded_columns"]
        passthrough = len(meta["numeric"]) + len(meta["accessories"])
        st.markdown(f"**One-hot encoding:** {len(meta['categorical'])} categorical inputs become "
                    f"{len(encoded) - passthrough} columns; rare levels (< 5 units) are pooled and "
                    "unseen levels are ignored.")
        st.markdown(f"**Model matrix:** {len(encoded)} columns from {len(meta['features'])} inputs.")

    st.divider()
    st.markdown("**Scaling effect**")
    feature = st.selectbox("Numeric feature", meta["numeric"], format_func=LABELS.get, key="scale_feature")
    position = meta["numeric"].index(feature)
    before = df[feature].dropna()
    after = (np.log1p(before) - scaler.mean_[position]) / scaler.scale_[position]
    raw_col, scaled_col = st.columns(2)
    raw_fig = px.histogram(before, nbins=40, title="Raw", labels={"value": LABELS[feature]},
                           color_discrete_sequence=[BLUE])
    scaled_fig = px.histogram(after, nbins=40, title="After log1p + StandardScaler",
                              labels={"value": "Standardised value"}, color_discrete_sequence=[AMBER])
    for fig in (raw_fig, scaled_fig):
        fig.update_layout(showlegend=False, height=300, yaxis_title="Units")
    raw_col.plotly_chart(raw_fig, width="stretch", key="scale_raw")
    scaled_col.plotly_chart(scaled_fig, width="stretch", key="scale_after")

    st.divider()
    numeric = df.select_dtypes(include=[np.number])
    fig_corr = px.imshow(numeric.corr(), text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r", title="Pearson correlation matrix")
    fig_corr.update_layout(height=560)
    st.plotly_chart(fig_corr, width="stretch", key="corr")
    st.info("days_in_stock defines the target (slow mover = more than 120 days), so it appears here for "
            "analysis but is never a model input. Arrival date, exchange rate, reservation status and "
            "transfer history were removed for the same reason.")

# ------------------------------------------------------------------------------
# Tab 3: Benchmark
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Classical models vs. neural network")
    st.caption(f"All scores below come from {meta['test_units']} units in {meta['test_lots']} purchase lots "
               "that no model saw during training or tuning.")
    table = pd.DataFrame([{
        "Model": ("★ " if name == meta["champion"] else "") + name,
        "Family": FAMILY[name],
        "Accuracy": results[name]["accuracy"], "Precision": results[name]["precision"],
        "Recall": results[name]["recall"], "F1": results[name]["f1"], "ROC-AUC": results[name]["roc_auc"],
        "CV ROC-AUC": f"{results[name]['cv_auc_mean']:.3f} ± {results[name]['cv_auc_std']:.3f}",
        "R²": results[name]["r2"], "Silhouette": results[name]["silhouette"],
    } for name in model_names])
    st.dataframe(table.round(3), width="stretch", hide_index=True)
    st.caption("★ champion = best 4-fold grouped CV ROC-AUC. R²: log days in stock for MLR, probability vs. "
               "label for the classifiers. Silhouette: separation of the predicted groups in the model matrix.")

    champion_row = results[meta["champion"]]
    st.markdown(f"**Milestone 1 targets for the champion:** recall ≥ 0.75 → "
                f"{'✅' if champion_row['recall'] >= 0.75 else '❌'} {champion_row['recall']:.2f} · "
                f"F1 ≥ 0.70 → {'✅' if champion_row['f1'] >= 0.70 else '❌'} {champion_row['f1']:.2f}")

    st.divider()
    st.subheader("Confusion matrices at each model's policy cut-off")
    for column, name in zip(st.columns(4), model_names):
        fig_cm = px.imshow(results[name]["confusion"], text_auto=True, color_continuous_scale="Blues",
                           x=["Pred. healthy", "Pred. slow"], y=["Healthy", "Slow"], title=name)
        fig_cm.update_layout(coloraxis_showscale=False, height=300, margin=dict(l=10, r=10, t=50, b=10))
        column.plotly_chart(fig_cm, width="stretch", key=f"cm_{name}")

    st.divider()
    roc_col, loss_col = st.columns(2)
    with roc_col:
        fig_roc = go.Figure()
        for name in model_names:
            fig_roc.add_trace(go.Scatter(x=results[name]["fpr"], y=results[name]["tpr"], mode="lines",
                                         line=dict(color=MODEL_COLORS[name], width=2.5),
                                         name=f"{name} (AUC {results[name]['roc_auc']:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random chance",
                                     line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(title="ROC curves on hold-out lots", xaxis_title="False positive rate",
                              yaxis_title="True positive rate", height=430,
                              legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_roc, width="stretch", key="roc")
    with loss_col:
        curves = meta["ann_curves"]
        epochs = list(range(1, len(curves["train_loss"]) + 1))
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=curves["train_loss"], name="Training loss",
                                      line=dict(color=BLUE, width=2.5)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=curves["val_loss"], name="Validation loss (held-out lots)",
                                      line=dict(color=AMBER, width=2.5)))
        fig_loss.add_vline(x=curves["best_epoch"], line_dash="dash", line_color=GREEN,
                           annotation_text=f"Weights restored: epoch {curves['best_epoch']}")
        fig_loss.update_layout(title="ANN loss convergence", xaxis_title="Epoch", yaxis_title="Log loss",
                               height=430, legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_loss, width="stretch", key="loss")
        best = curves["best_epoch"] - 1
        st.caption(f"Validation loss bottoms at {curves['val_loss'][best]:.3f} in epoch {curves['best_epoch']} and "
                   f"ends at {curves['val_loss'][-1]:.3f}, while training loss reaches "
                   f"{curves['train_loss'][-1]:.3f}. Training stops once held-out loss stops improving, "
                   "and the best-epoch weights are the ones serialized.")

# ------------------------------------------------------------------------------
# Tab 4: What-If
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Real-time what-if scenario engine")
    b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 3])
    b1.button("Load real low-risk unit", on_click=load_preset, args=("low",), width="stretch")
    b2.button("Load real high-risk unit", on_click=load_preset, args=("high",), width="stretch")
    b3.button("Reset", on_click=reset_scenario, width="stretch")

    inputs, output = st.columns(2)
    with inputs:
        st.markdown("##### Unit profile")
        st.slider("Days in stock (inventory age)", 0, 730, key="wi_days")
        st.slider("Landed cost (MXN)", 500, 45000, step=100, key="wi_cost")
        st.slider("Page counter (thousands)", 1, 1700, step=5, key="wi_counter")
        c1, c2 = st.columns(2)
        c1.selectbox("Brand", sorted(df["brand"].unique()), key="wi_brand")
        c2.selectbox("Warehouse", sorted(df["warehouse"].unique()), key="wi_warehouse")
        c1.selectbox("Product line", sorted(df["line"].unique()), key="wi_line")
        c2.selectbox("Appearance", sorted(df["appearance"].unique()), key="wi_appearance")
        c1.selectbox("Copy quality", sorted(df["copy_quality"].unique()), key="wi_quality")
        c2.selectbox("Working unit", sorted(df["working"].unique()), key="wi_working")
        st.multiselect("Accessories", meta["accessories"], key="wi_acc",
                       format_func=lambda c: c.replace("acc_", "").upper())

    row = pd.DataFrame([{
        "brand": st.session_state.wi_brand, "line": st.session_state.wi_line,
        "warehouse": st.session_state.wi_warehouse, "appearance": st.session_state.wi_appearance,
        "copy_quality": st.session_state.wi_quality, "working": st.session_state.wi_working,
        "page_counter_k": float(st.session_state.wi_counter),
        "landed_cost_mxn": float(st.session_state.wi_cost),
        **{c: int(c in st.session_state.wi_acc) for c in meta["accessories"]},
    }])[meta["features"]]

    with output:
        st.markdown(f"##### Live inference · {active}")
        score = raw_score(active, row)
        band = risk_band(active, score)
        if rule["kind"] == "regression":
            days_pred = float(np.expm1(score))
            st.metric("Predicted days in stock", f"{days_pred:.0f} days")
            st.progress(min(days_pred / 365, 1.0))
            shown = f"{days_pred:.0f} days"
        else:
            st.metric("Stall probability", f"{score:.1%}")
            st.progress(score)
            shown = f"{score:.1%}"
        st.markdown(f"**Risk band: {band}**")

        days = st.session_state.wi_days
        cost = st.session_state.wi_cost
        recovered = cost * (1 - DISCOUNT)
        if days >= SLOW_DAYS:
            action = "25% discount rule"
            st.error(f"🚨 25% DISCOUNT RULE TRIGGERED: unit is at {days} days, past the {SLOW_DAYS}-day limit.")
            st.markdown(f"**Commercial Director:** launch the 25% discount now. On landed cost that means selling "
                        f"at MXN {recovered:,.0f} to release the cash (MXN {cost - recovered:,.0f} concession).")
        elif band == "HIGH":
            action = "Priority list"
            st.warning(f"⚠️ HIGH RISK: {SLOW_DAYS - days} days left before the discount rule.")
            st.markdown("**Procurement:** reopen the price or skip this configuration at the buy. "
                        "**Commercial:** add it to Monday's priority list and avoid sending it to MEX.")
        elif band == "WATCH":
            action = "Watch list"
            st.info(f"👀 WATCH: {SLOW_DAYS - days} days left before the discount rule.")
            st.markdown("**Inventory:** assign it to the lowest-risk warehouse and review the price at day 60.")
        else:
            action = "Normal flow"
            st.success(f"✅ NORMAL FLOW: low stall risk, {SLOW_DAYS - days} days left before the discount rule.")
            st.markdown("**Procurement:** no manual action needed.")
        st.caption("Inventory age drives the 25% business rule only. It is not a model input: "
                   "the label is defined from it, so feeding it to the model would leak the answer.")

        if st.button("Save this scenario", width="stretch"):
            st.session_state.history.append({
                "Model": active, "Brand": row.at[0, "brand"], "Warehouse": row.at[0, "warehouse"],
                "Days": days, "Cost (MXN)": cost, "Result": shown, "Band": band, "Action": action,
            })

    if st.session_state.history:
        st.divider()
        st.markdown("**Scenarios saved in this session**")
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", hide_index=True)
        if st.button("Clear scenarios"):
            st.session_state.history = []
            st.rerun()
