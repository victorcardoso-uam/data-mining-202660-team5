"""
What the dashboard shows and how it is used, as plain text for the assistant.

The assistant answers only from what it is sent, so everything a user can see or
click is described here: the screens and controls (app_guide), the numbers the
tables and charts display (screen_facts) and the rows listed in the inventory
table (table_rows). The constants below are the ones app.py builds its widgets
from, so the description cannot drift away from the app. No Streamlit import.
"""
import numpy as np
import pandas as pd

TARGET, AGE, COST, COUNTER = "slow_mover_flag", "days_in_stock", "landed_cost_mxn", "page_counter_k"

TAB_NAMES = ("📊 Data Explorer & EDA", "⚙️ Preprocessing & Correlation",
             "🤖 Multi-Model Benchmark", "🔮 What-If Inference")
ASSISTANT_PLACES = ("Side panel", "Wide side panel", "Below the tabs", "Hidden")
SIDE_RATIOS = {"Side panel": (3, 2), "Wide side panel": (1, 1)}      # tabs : assistant
SLIDERS = {                                                          # key: (label, min, max, step)
    "wi_days": ("Days in stock (inventory age)", 0, 730, 1),
    "wi_cost": ("Landed cost (MXN)", 500, 45000, 100),
    "wi_counter": ("Page counter (thousands)", 1, 1700, 5),
}
TABLE_ROWS = 100
STATUS = {0: "Healthy turnover", 1: "Slow mover (> 120 days)"}
FAMILY = {"Random Forest": "Classical ensemble", "Decision Tree": "Classical tree",
          "MLR": "Classical regression", "ANN": "Neural network"}
LABELS = {COUNTER: "Page counter (thousands)", COST: "Landed cost (MXN)",
          AGE: "Days in stock", "brand": "Brand", "line": "Product line",
          "warehouse": "Warehouse", "appearance": "Appearance", "copy_quality": "Copy quality"}
RANGES = {                                                           # how the histograms are summarised in words
    COUNTER: [0, 50, 100, 150, 200, 300, 500, 2000],
    COST: [0, 2500, 5000, 7500, 10000, 15000, 20000, 45000],
    AGE: [0, 30, 60, 90, 120, 180, 365, 730, 1500],
}


def default_scenario(df):
    """The unit the What-If tab starts with and goes back to on Reset."""
    return {
        "wi_days": 60,
        "wi_cost": int(round(df[COST].median(), -2)),
        "wi_counter": int(df[COUNTER].median()),
        "wi_brand": df["brand"].mode()[0],
        "wi_line": df["line"].mode()[0],
        "wi_warehouse": df["warehouse"].mode()[0],
        "wi_appearance": df["appearance"].mode()[0],
        "wi_quality": df["copy_quality"].mode()[0],
        "wi_working": "SI",
        "wi_acc": ["acc_duplex", "acc_dsdf"],
    }


def preset_values(row, defaults, accessories):
    """A real unit from metrics.json as the What-If controls show it (values clipped to the slider ranges)."""
    counter = row[COUNTER]
    return {
        "wi_days": int(min(row[AGE], SLIDERS["wi_days"][2])),
        "wi_cost": int(min(round(row[COST], -2), SLIDERS["wi_cost"][2])),
        "wi_counter": defaults["wi_counter"] if pd.isna(counter) else int(min(counter, SLIDERS["wi_counter"][2])),
        "wi_brand": row["brand"], "wi_line": row["line"], "wi_warehouse": row["warehouse"],
        "wi_appearance": row["appearance"], "wi_quality": row["copy_quality"], "wi_working": row["working"],
        "wi_acc": [c for c in accessories if row[c] == 1],
    }


def _unit(values):
    accessories = ", ".join(a.replace("acc_", "").upper() for a in values["wi_acc"]) or "none"
    return (f"{values['wi_brand']}, line {values['wi_line']}, warehouse {values['wi_warehouse']}, appearance "
            f"{values['wi_appearance']}, copy quality {values['wi_quality']}, working {values['wi_working']}, page counter "
            f"{values['wi_counter']:,} thousand, landed cost MXN {values['wi_cost']:,}, {values['wi_days']} days in "
            f"stock, accessories {accessories}")


def _plain(tab_name):
    return tab_name.split(" ", 1)[1]


def app_guide(df, meta, suggested, max_chars, max_turns):
    """Screens and controls, in the order a user meets them."""
    results, policy, champion = meta["results"], meta["policy"], meta["champion"]
    order = sorted(results, key=lambda n: -results[n]["cv_auc_mean"])
    capital, stalled = df[COST].sum(), df.loc[df[TARGET] == 1, COST].sum()
    defaults = default_scenario(df)
    presets = {which: preset_values(meta["presets"][which], defaults, meta["accessories"]) for which in ("low", "high")}
    encoded = meta["encoded_columns"]
    one_hot = len(encoded) - len(meta["numeric"]) - len(meta["accessories"])
    files = ["preprocessor.joblib", "standard_scaler.joblib", *meta["files"].values()]
    captions = []
    for name in order:
        rule = policy[name]
        captions.append(f"{name}: \"Flags a unit when predicted days in stock ≥ {rule['watch_days']:.0f}.\""
                        if rule["kind"] == "regression" else
                        f"{name}: \"WATCH from {rule['watch']:.2f} · HIGH from {rule['high']:.2f} stall probability.\"")
    sliders = "; ".join(f"\"{label}\" from {low:,} to {high:,}" + (f" in steps of {step}" if step != 1 else "")
                        for label, low, high, step in SLIDERS.values())
    tab1, tab2, tab3, tab4 = (_plain(name) for name in TAB_NAMES)
    side, wide, below, hidden = ASSISTANT_PLACES
    ratio = {place: round(100 * b / (a + b)) for place, (a, b) in SIDE_RATIOS.items()}
    return "\n".join([
        "APP GUIDE (every screen and control of the dashboard, with the exact names shown on screen)",
        "- The dashboard is a Streamlit app with a dark theme (.streamlit/config.toml). It is started from the "
        "capstone_streamlit_app folder with `pip install -r requirements.txt` and `streamlit run app.py`, and opens at "
        "http://localhost:8501. It trains nothing: the dataset and the serialized models are loaded once and cached "
        "(st.cache_data and st.cache_resource), and every control only re-scores the unit.",
        "- Page layout: a sidebar on the left; at the top the title \"Big99 Inventory Intelligence\" and four KPI cards; "
        "below them four tabs; and the \"Inventory assistant\" panel (this chat).",
        f"- KPI cards: \"Units in stock\" {len(df):,} ({df['lot_id'].nunique()} purchase lots); \"Landed capital\" MXN "
        f"{capital / 1e6:.2f}M (4 warehouses); \"Stalled > 120 days\" MXN {stalled / 1e6:.2f}M ({stalled / capital:.1%} of "
        "capital); \"Active model ROC-AUC\", the 4-fold grouped CV ROC-AUC of the model chosen in the sidebar.",
        f"- Sidebar, \"Active model\" selector: {', '.join(n + (' ★ champion' if n == champion else '') for n in order)}. "
        f"The app starts on {champion}. Changing it changes the fourth KPI card and the \"Live inference\" result in the "
        f"{tab4} tab; the benchmark table always shows all four models. Under the selector a caption gives the active "
        f"model's cut-off: {'; '.join(captions)}",
        f"- Sidebar, \"AI assistant\" radio, moves this panel: \"{side}\" (default, {ratio[side]}% of the width, beside "
        f"the tabs), \"{wide}\" ({ratio[wide]}%), \"{below}\" (full width under the tabs, best on a small screen) and "
        f"\"{hidden}\". Switching between the two side panels keeps the open tab; moving between a side panel and "
        f"\"{below}\" or \"{hidden}\" redraws the page on the first tab. The what-if values and the conversation are "
        "always kept.",
        f"- Sidebar, \"Serialized artifacts loaded\": the files read from the models folder: {', '.join(files)}. "
        "preprocessor.joblib holds the imputer, scaler and encoder; standard_scaler.joblib holds the StandardScaler "
        "parameters shown in the second tab; the other four are the trained models.",
        f"- Tab \"{tab1}\": (1) \"Anonymised inventory table\", the first {TABLE_ROWS} of the {len(df):,} rows with all "
        f"{df.shape[1]} columns (see TABLE ROWS). (2) \"Summary statistics\" table. (3) \"Feature distribution by turnover "
        f"status\": the \"Feature\" selector offers {', '.join(LABELS.values())}; a numeric feature is drawn as two "
        "overlaid histograms and a category as stacked bars, green for healthy turnover and red for slow movers; days in "
        "stock are capped at 730 in that chart. (4) Bar chart \"Landed capital by warehouse\", split the same way.",
        f"- Tab \"{tab2}\" (heading \"Leak-free feature pipeline\"): \"Missing-value treatment\" table; \"StandardScaler "
        f"parameters (after log1p)\" table; the "
        f"one-hot summary ({len(meta['categorical'])} categorical inputs become {one_hot} columns, model matrix "
        f"{len(encoded)} columns from {len(meta['features'])} inputs); \"Scaling effect\", where the \"Numeric feature\" "
        "selector (Page counter or Landed cost) draws the raw histogram next to the one after log1p + StandardScaler; the "
        "\"Pearson correlation matrix\" heat map; and a note on why days_in_stock is never a model input.",
        f"- Tab \"{tab3}\": under \"Classical models vs. neural network\", the benchmark table with columns Model, "
        "Family, Accuracy, Precision, Recall, F1, ROC-AUC, CV ROC-AUC, R² and Silhouette (the champion carries a ★) and "
        "the line \"Milestone 1 targets for the champion\"; under \"Confusion matrices at each model's policy cut-off\", "
        "one matrix per model; then the chart \"ROC curves on hold-out lots\" and the chart \"ANN "
        "loss convergence\" with a dashed line at the epoch whose weights were restored.",
        f"- Tab \"{tab4}\" (heading \"Real-time what-if scenario engine\"), controls: buttons \"Load real low-risk unit\" "
        "and \"Load real high-risk unit\" put a real unit "
        f"of the dataset into the controls, and \"Reset\" goes back to the default unit. Under \"Unit profile\": sliders "
        f"{sliders}; selectors \"Brand\", \"Warehouse\", \"Product line\", \"Appearance\", \"Copy quality\" and \"Working "
        f"unit\"; and the \"Accessories\" multi-select ({', '.join(a.replace('acc_', '').upper() for a in meta['accessories'])}).",
        f"- Tab \"{tab4}\", result: \"Live inference\" shows \"Predicted days in stock\" for MLR or \"Stall probability\" "
        "for the other models, a progress bar, the line \"Risk band\" and a coloured alert with the action: red \"25% "
        "DISCOUNT RULE TRIGGERED\" at 120 days or more, yellow \"HIGH RISK\", blue \"WATCH\" and green \"NORMAL FLOW\". "
        "The result updates as soon as a control moves. \"Save this scenario\" adds the current unit to the table "
        "\"Scenarios saved in this session\" (columns Model, Brand, Warehouse, Days, Cost (MXN), Result, Band, Action) and "
        "\"Clear scenarios\" empties it; the table appears under the controls once one scenario is saved and lasts until "
        "the browser tab is closed. Saving is the way to compare models or units side by side.",
        f"- Default unit (app start and \"Reset\"): {_unit(defaults)}.",
        f"- \"Load real low-risk unit\" loads: {_unit(presets['low'])}.",
        f"- \"Load real high-risk unit\" loads: {_unit(presets['high'])}.",
        f"- Assistant panel: the buttons {', '.join(suggested)} each send a fixed question; any other question goes in the "
        f"box \"Ask about the data, the models or this unit\" (up to {max_chars} characters, English or Spanish). The "
        f"assistant is sent the last {max_turns} exchanges, this fact sheet and the unit currently set in the {tab4} "
        "tab. \"Clear conversation\" erases the exchanges. The panel needs an Anthropic API key in "
        ".streamlit/secrets.toml and an internet connection; without a key it shows a notice and the four tabs still work.",
    ])


def _status_counts(series, target):
    table = pd.crosstab(series, target).reindex(columns=[0, 1], fill_value=0)
    return table.assign(total=table.sum(axis=1))


def screen_facts(df, meta, scaler=None):
    """The numbers behind every table and chart that the other sections do not already give."""
    lines = ["ON-SCREEN NUMBERS (what the tables and charts display, tab by tab)"]

    stats = df[[COUNTER, COST, AGE, TARGET]].describe().T.round(2)
    lines.append("- Tab 1, \"Summary statistics\" table (count / mean / std / min / 25% / 50% / 75% / max):")
    lines += [f"  {column}: " + " / ".join(f"{row[k]:,.2f}" for k in stats.columns) for column, row in stats.iterrows()]

    lines.append("- Tab 1, \"Feature distribution by turnover status\", bars per level (healthy units / slow units):")
    for column in [c for c in LABELS if not pd.api.types.is_numeric_dtype(df[c])]:
        table = _status_counts(df[column], df[TARGET]).sort_values("total", ascending=False)
        lines.append(f"  {column}: " + "; ".join(f"{level} {row[0]} / {row[1]}" for level, row in table.iterrows()))
    lines.append("- Tab 1, same chart for the numeric features, summarised as units per range (healthy / slow; the "
                 "chart draws about 40 finer bins; each range excludes its lower bound and includes its upper bound):")
    for column, edges in RANGES.items():
        cut = pd.cut(df[column], edges)
        assert cut.notna().sum() == df[column].notna().sum(), f"{column}: a value falls outside the ranges"
        table = _status_counts(cut, df[TARGET])
        lines.append(f"  {column}: " + "; ".join(
            f"{int(interval.left):,}-{int(interval.right):,}: {row[0]} / {row[1]}" for interval, row in table.iterrows()))

    capital = df.groupby(["warehouse", TARGET])[COST].sum().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    capital = capital.assign(total=capital.sum(axis=1)).sort_values("total", ascending=False)
    lines.append("- Tab 1, \"Landed capital by warehouse\" chart (MXN in healthy units / in slow movers / total):")
    lines.append("  " + "; ".join(f"{site} {row[0]:,.0f} / {row[1]:,.0f} / {row['total']:,.0f}"
                                  for site, row in capital.iterrows()))

    missing = df[meta["numeric"]].isna().sum()
    medians = [meta["imputed_median_counter"], meta["imputed_median_cost"]]
    lines.append("- Tab 2, \"Missing-value treatment\" table (missing values, treatment, training median): " + "; ".join(
        f"{column} {missing[column]}, median imputation, {median:,.2f}" for column, median in zip(meta["numeric"], medians)))
    if scaler is not None:
        lines.append("- Tab 2, \"StandardScaler parameters (after log1p)\" table (mean / std. deviation, in log1p units): "
                     + "; ".join(f"{column} {mean:.4f} / {scale:.4f} (the mean equals {np.expm1(mean):,.0f} in the "
                                 "original unit)" for column, mean, scale in zip(meta["numeric"], scaler.mean_, scaler.scale_)))
    corr = df.select_dtypes(include=[np.number]).corr()
    names = list(corr.columns)
    lines.append("- Tab 2, \"Pearson correlation matrix\" (every pair of numeric columns):")
    lines += [f"  {a} with " + ", ".join(f"{b} {corr.at[a, b]:+.2f}" for b in names[i + 1:])
              for i, a in enumerate(names[:-1])]

    results = meta["results"]
    order = sorted(results, key=lambda n: -results[n]["cv_auc_mean"])
    lines.append("- Tab 3, \"Family\" column: " + "; ".join(f"{name} = {FAMILY[name]}" for name in order) + ".")
    lines.append("- Tab 3, \"ROC curves on hold-out lots\": true positive rate at a false positive rate of 10% / 20% / "
                 "30% / 50%, read off each curve (approximate): " + "; ".join(
                     f"{name} " + " / ".join(f"{v:.2f}" for v in np.interp([0.1, 0.2, 0.3, 0.5], results[name]["fpr"],
                                                                         results[name]["tpr"])) for name in order) + ".")
    curves = meta["ann_curves"]
    last = len(curves["train_loss"])
    epochs = sorted({1, *range(10, last, 10), curves["best_epoch"], last})
    lines.append("- Tab 3, \"ANN loss convergence\" (epoch: training loss / validation loss): " + "; ".join(
        f"{e}: {curves['train_loss'][e - 1]:.3f} / {curves['val_loss'][e - 1]:.3f}" for e in epochs)
        + f". The dashed line marks epoch {curves['best_epoch']}.")
    return "\n".join(lines)


def table_rows(df):
    """The rows listed in Tab 1, as CSV with the row number the table shows."""
    head = df.head(TABLE_ROWS).round(2)
    return (f"TABLE ROWS (the first {len(head)} of the {len(df):,} rows, as the \"Anonymised inventory table\" of Tab 1 "
            f"lists them; they cover {head['lot_id'].nunique()} of the {df['lot_id'].nunique()} lots and are not a "
            "sample of the stock. An empty value is a missing reading.)\n"
            + head.to_csv(index_label="row", lineterminator="\n").strip())
