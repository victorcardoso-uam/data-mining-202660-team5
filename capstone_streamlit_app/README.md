# Big99 Inventory Intelligence dashboard

Streamlit app for the final capstone. It loads the trained models from models/ when it starts. It does not train anything.

## Run it

Tested with Python 3.13. From this folder:

```
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. The package versions are pinned because the .joblib files only load with the scikit-learn version that created them.

## What each tab shows

| Tab | Contents |
|---|---|
| Data Explorer & EDA | The anonymised table, summary statistics, the distribution of any feature split by slow mover or not, and landed capital by warehouse |
| Preprocessing & Correlation | How missing values, scaling and one-hot encoding are handled, the effect of scaling on a numeric feature, and the Pearson correlation matrix |
| Multi-Model Benchmark | Scores of the four models on the hold-out lots, their confusion matrices, the ROC curves and the ANN loss curve |
| What-If Inference | Sliders and selectors to describe a unit. The app shows the model score, the risk band and the action the operating policy gives for it |

The "Active model" box in the sidebar switches between the decision tree, the random forest, the MLR and the ANN.

## Turn on the assistant

The assistant answers questions in English or Spanish about the data, the models and the unit set in the What-If tab. It uses the Anthropic API, so it needs an API key and an internet connection.

1. Create an API key in the Claude Console (platform.claude.com).
2. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml.
3. Open secrets.toml and paste the key between the quotes after ANTHROPIC_API_KEY.
4. Stop the app with Ctrl+C and start it again.

Git ignores secrets.toml. Do not commit a key or share it in a chat.

Without a key the four tabs still work and the assistant panel shows a setup message. The "AI assistant" option in the sidebar places the panel beside the tabs, below them, or hides it. Below the tabs works better on a small screen.

## Files

| File | Purpose |
|---|---|
| app.py | The dashboard |
| assistant.py | Builds the fact sheet that is sent to the LLM and makes the API call |
| train_models.py | Retrains the four models and rewrites models/ |
| explain_models.py | Computes feature importance and writes models/explainability.json. Run it after retraining |
| data/team_dataset.csv | The dataset. The columns are described in docs/data_dictionary.md |
| models/ | The serialized preprocessor, scaler and four models (.joblib), plus metrics.json |
| tests/test_assistant.py | 13 checks that run without a key |

## Tests

```
pip install pytest
python -m pytest tests -q
```

The last line should read "13 passed".
