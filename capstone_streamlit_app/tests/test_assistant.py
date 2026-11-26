"""
Checks for the dashboard's LLM assistant. No network and no API key needed:
the Anthropic call is replaced by a stub.

From capstone_streamlit_app/:
    pip install pytest
    python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import anthropic
import httpx2
import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import app_guide  # noqa: E402
import assistant  # noqa: E402

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(APP / "data" / "team_dataset.csv")


@pytest.fixture(scope="module")
def meta():
    return json.loads((APP / "models" / "metrics.json").read_text())


@pytest.fixture(scope="module")
def explain():
    return json.loads((APP / "models" / "explainability.json").read_text())


@pytest.fixture(scope="module")
def scaler():
    return joblib.load(APP / "models" / "standard_scaler.joblib")


@pytest.fixture(scope="module")
def sheet(df, meta, explain, scaler):
    return assistant.build_fact_sheet(df, meta, explain, scaler)


# ------------------------------------------------------------------------------
# Fact sheet: every number must come from the data or the stored metrics
# ------------------------------------------------------------------------------
def test_fact_sheet_matches_the_data(df, meta, sheet):
    slow = int(df["slow_mover_flag"].sum())
    assert ((df["days_in_stock"] > assistant.SLOW_DAYS) == (df["slow_mover_flag"] == 1)).all()
    assert f"{len(df):,} units in {df['lot_id'].nunique()} purchase lots" in sheet
    assert f"{slow} units are slow movers ({slow / len(df):.1%})" in sheet
    assert f"MXN {df['landed_cost_mxn'].sum():,.0f}" in sheet
    missing = int(df["page_counter_k"].isna().sum())
    assert f"page_counter_k {missing} " in sheet
    for warehouse, days in df.groupby("warehouse")["days_in_stock"].median().items():
        assert f"{warehouse} {days:.0f}" in sheet


def test_fact_sheet_matches_the_benchmark(meta, explain, sheet):
    for name, r in meta["results"].items():
        assert (f"accuracy {r['accuracy']:.3f}, precision {r['precision']:.3f}, recall {r['recall']:.3f}, "
                f"F1 {r['f1']:.3f}, hold-out ROC-AUC {r['roc_auc']:.3f}") in sheet
    assert f"- {meta['champion']} (champion):" in sheet
    top = explain["models"][meta["champion"]]["permutation_importance"][0]
    assert f"- {meta['champion']}: {top['feature']} {top['auc_drop_mean']:+.3f}" in sheet


def test_fact_sheet_reports_weak_results(meta, sheet):
    champion = meta["results"][meta["champion"]]
    assert champion["f1"] < 0.70 and "F1 >= 0.70 is missed" in sheet
    assert "false alarms" in sheet and "Censored label" in sheet and "confounded with stock age" in sheet


def test_explainability_was_built_from_the_served_models(meta, explain):
    assert explain["basis"]["holdout_units"] == meta["test_units"]
    for name, r in meta["results"].items():
        assert explain["models"][name]["holdout_roc_auc"] == round(r["roc_auc"], 4)
        assert len(explain["models"][name]["permutation_importance"]) == len(meta["features"])


def test_pooled_levels_are_not_quoted_as_model_effects(df, meta, sheet):
    effects = sheet.split("LEARNED EFFECTS")[1].split("DECISION TREE RULES")[0]
    assert "LANIER" not in effects and "SIN LINEA" not in effects
    assert "brand=LANIER" in sheet                      # but they are disclosed as a limitation


def test_policy_never_uses_warehouse_as_a_lever():
    """Warehouse risk mirrors stock age (MID holds the newest units), so no action may rely on it."""
    for text in ((APP / "app.py").read_text(encoding="utf-8"), assistant.POLICY_ACTIONS):
        assert "lowest-risk warehouse" not in text and "sending it to" not in text


# ------------------------------------------------------------------------------
# Request building and error wording
# ------------------------------------------------------------------------------
def test_messages_carry_the_scenario_and_cap_history():
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]
    messages = assistant.build_messages(history, "  What now?  ", "Days in stock: 95.")
    assert len(messages) == 2 * assistant.MAX_TURNS + 1
    assert messages[0] == {"role": "user", "content": f"q{20 - assistant.MAX_TURNS}"}
    assert [m["role"] for m in messages[-3:]] == ["user", "assistant", "user"]
    last = messages[-1]["content"]
    assert "<scenario>\nDays in stock: 95.\n</scenario>" in last and "<question>\nWhat now?\n</question>" in last


def test_long_questions_are_truncated():
    messages = assistant.build_messages([], "x" * 5000, "s")
    assert messages[-1]["content"].count("x") == assistant.MAX_QUESTION_CHARS


def test_api_errors_become_actionable_messages():
    unauthorized = anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx2.Response(401, request=REQUEST), body=None)
    assert "ANTHROPIC_API_KEY" in assistant.explain_error(unauthorized)
    assert "internet" in assistant.explain_error(anthropic.APIConnectionError(request=REQUEST))
    busy = anthropic.InternalServerError(
        "overloaded", response=httpx2.Response(529, request=REQUEST), body=None)
    assert "busy" in assistant.explain_error(busy)


# ------------------------------------------------------------------------------
# The app itself, driven headless
# ------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch):
    """Tests never touch the developer's key: the environment is cleared here and
    run_app overrides whatever .streamlit/secrets.toml holds."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)


def run_app(key=""):
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=120)
    at.secrets["ANTHROPIC_API_KEY"] = key
    at.secrets["ANTHROPIC_MODEL"] = ""
    return at.run()


def test_dashboard_runs_without_a_key():
    at = run_app()
    assert not at.exception
    assert len(at.tabs) == 4
    assert any("no API key was found" in box.value for box in at.info)
    assert not at.chat_input


def test_question_reaches_the_llm_with_the_live_scenario(monkeypatch):
    sent = {}

    def fake_stream(api_key, model, fact_sheet, messages):
        sent.update(api_key=api_key, model=model, fact_sheet=fact_sheet, messages=messages)
        yield "Launch the 25% discount "
        yield "now."

    monkeypatch.setattr(assistant, "stream_answer", fake_stream)
    at = run_app(key="test-key")
    at.slider(key="wi_days").set_value(130).run()
    at.selectbox(key="wi_warehouse").set_value("GDL").run()
    at.button(key="suggest_This scenario").click().run()

    assert not at.exception
    assert sent["api_key"] == "test-key" and sent["model"] == assistant.DEFAULT_MODEL
    assert sent["fact_sheet"].startswith("FACT SHEET")
    question = sent["messages"][-1]["content"]
    assert "Days in stock: 130." in question and "warehouse GDL" in question
    assert "discount rule is triggered" in question
    assert assistant.SUGGESTED["This scenario"] in question
    assert at.session_state["assistant_chat"] == [
        {"question": assistant.SUGGESTED["This scenario"], "answer": "Launch the 25% discount now."}]

    at.chat_input(key="assistant_input").set_value("And if it were at 40 days?").run()
    assert len(at.session_state["assistant_chat"]) == 2
    assert sent["messages"][0]["content"] == assistant.SUGGESTED["This scenario"]   # history is resent
    assert len(sent["messages"]) == 3


def test_api_failure_does_not_break_the_dashboard(monkeypatch):
    def failing_stream(*args):
        raise anthropic.APIConnectionError(request=REQUEST)
        yield  # pragma: no cover

    monkeypatch.setattr(assistant, "stream_answer", failing_stream)
    at = run_app(key="test-key")
    at.chat_input(key="assistant_input").set_value("Which inputs matter most?").run()
    assert not at.exception
    assert any("Could not reach the Anthropic API" in box.value for box in at.error)
    assert at.session_state["assistant_chat"] == []
    assert len(at.tabs) == 4


def test_panel_can_move_below_the_tabs_or_hide(monkeypatch):
    monkeypatch.setattr(assistant, "stream_answer", lambda *args: iter(["ok"]))
    at = run_app(key="test-key")
    assert at.chat_input
    at.radio(key="assistant_place").set_value("Below the tabs").run()
    assert not at.exception and at.chat_input and len(at.tabs) == 4
    at.button(key="suggest_Dataset").click().run()
    assert at.session_state["assistant_chat"][0]["answer"] == "ok"
    at.radio(key="assistant_place").set_value("Hidden").run()
    assert not at.exception and not at.chat_input and len(at.tabs) == 4
    assert len(at.session_state["assistant_chat"]) == 1          # the conversation survives hiding


# ------------------------------------------------------------------------------
# The assistant must cover everything the app shows and how it is used
# ------------------------------------------------------------------------------
def save_button(at):
    return next(button for button in at.button if button.label == "Save this scenario")


def test_guide_names_every_control_the_app_draws(sheet, monkeypatch):
    monkeypatch.setattr(assistant, "stream_answer", lambda *args: iter(["ok"]))
    at = run_app(key="test-key")
    at.button(key="suggest_Dataset").click().run()               # "Clear conversation" appears
    save_button(at).click().run()                                # the saved table and "Clear scenarios" appear
    assert not at.exception
    labels = [w.label for kind in (at.button, at.selectbox, at.slider, at.radio, at.multiselect, at.metric)
              for w in kind]
    labels += [tab.label.split(" ", 1)[1] for tab in at.tabs] + [heading.value for heading in at.subheader]
    assert {"Clear conversation", "Clear scenarios", "Active model", "Accessories"} <= set(labels)
    missing = sorted({label for label in labels if label not in sheet})
    assert not missing, f"the APP GUIDE does not mention: {missing}"
    for key, (label, low, high, step) in app_guide.SLIDERS.items():
        slider = at.slider(key=key)
        assert (slider.label, slider.min, slider.max, slider.step) == (label, low, high, step)
        assert f"\"{label}\" from {low:,} to {high:,}" in sheet
    assert list(at.radio(key="assistant_place").options) == list(app_guide.ASSISTANT_PLACES)
    assert all(f"\"{place}\"" in sheet for place in app_guide.ASSISTANT_PLACES)


def test_on_screen_numbers_match_the_data(df, meta, scaler, sheet):
    for total in df.groupby("warehouse")["landed_cost_mxn"].sum():           # Tab 1, capital by warehouse
        assert f"/ {total:,.0f}" in sheet
    assert f"/ {df['landed_cost_mxn'].std():,.2f} /" in sheet                 # Tab 1, summary statistics
    for mean, scale in zip(scaler.mean_, scaler.scale_):                     # Tab 2, scaler table
        assert f"{mean:.4f} / {scale:.4f}" in sheet
    pair = df[["page_counter_k", "landed_cost_mxn"]].corr().iloc[0, 1]       # Tab 2, full correlation matrix
    assert f"page_counter_k with landed_cost_mxn {pair:+.2f}" in sheet
    for file in ["preprocessor.joblib", "standard_scaler.joblib", *meta["files"].values()]:    # sidebar
        assert file in sheet
    high = meta["presets"]["high"]                                           # Tab 4, the real units behind the buttons
    assert f"\"Load real high-risk unit\" loads: {high['brand']}, line {high['line']}, warehouse {high['warehouse']}" in sheet


def test_table_rows_are_the_ones_tab_1_lists(df):
    lines = app_guide.table_rows(df).split("\n")
    assert f"first {app_guide.TABLE_ROWS} of the {len(df):,} rows" in lines[0]
    assert len(lines) == 2 + app_guide.TABLE_ROWS                            # caption, CSV header, rows
    first, last = df.iloc[0], df.iloc[app_guide.TABLE_ROWS - 1]
    assert lines[2].startswith(f"0,{first['lot_id']},{first['brand']},{first['line']},{first['warehouse']}")
    assert lines[-1].startswith(f"{app_guide.TABLE_ROWS - 1},{last['lot_id']},{last['brand']}")


def test_saved_scenarios_and_panel_placement_reach_the_llm(monkeypatch):
    sent = {}

    def fake_stream(api_key, model, fact_sheet, messages):
        sent.update(fact_sheet=fact_sheet, messages=messages)
        yield "ok"

    monkeypatch.setattr(assistant, "stream_answer", fake_stream)
    at = run_app(key="test-key")
    at.button(key="suggest_Dataset").click().run()
    first = sent["messages"][-1]["content"]
    assert "nothing has been saved yet" in first and "Assistant panel placement: Side panel." in first
    assert "APP GUIDE" in sent["fact_sheet"] and "StandardScaler parameters" in sent["fact_sheet"]
    assert "TABLE ROWS" in sent["fact_sheet"]

    at.selectbox(key="wi_brand").set_value("XEROX").run()
    save_button(at).click().run()
    at.radio(key="assistant_place").set_value("Wide side panel").run()
    at.button(key="suggest_This scenario").click().run()
    assert not at.exception and at.chat_input and len(at.tabs) == 4
    last = sent["messages"][-1]["content"]
    assert "- 1. Model MLR, Brand XEROX" in last and "Assistant panel placement: Wide side panel." in last


def test_rules_keep_the_scope_and_guard_the_table_rows():
    rules = " ".join(assistant.SYSTEM_RULES.replace("\\\n", "").split())
    assert "Decline requests unrelated to this dashboard without answering them" in rules
    assert "never count, total or rank the whole stock from it" in rules
    assert "never send the user to the developers" in rules
