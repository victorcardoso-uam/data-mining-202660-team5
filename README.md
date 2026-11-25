# Data Mining 202660 - Team 5

Coursework for Data Mining & Modern AI Systems (IIND4417) at Universidad Anáhuac Mayab, academic period 202660, taught by Dr. Víctor Cardoso. The repository holds the class activities, Partial Exam 2 and the final capstone.

## Final capstone: Big99 Inventory Intelligence

Big99 resells used multifunction printers from four warehouses. A unit that reaches 120 days in stock counts as a slow mover and gets a 25% discount so the company can recover the cash. The capstone tries to spot those units earlier, using what is known when they are bought.

What is in the repository:

- An anonymised table of 1,091 units from 97 purchase lots, taken from the inventory snapshot of 20 October 2026.
- Four trained models: a decision tree, a random forest, a multiple linear regression (MLR) and a neural network (ANN).
- A Streamlit dashboard with four tabs and an LLM assistant that answers questions about the data, the models and the unit set in the What-If tab.

The champion is the MLR. On 25 purchase lots that no model saw during training it catches 56 of 70 slow movers (recall 0.80, above the 0.75 target). Its F1 is 0.50, below the 0.70 target, and about 64 of every 100 units it flags are false alarms. We use it to decide which units to review first. A person approves every discount and every purchase.

## Run the dashboard

Tested with Python 3.13.

```
cd capstone_streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The four tabs work right away. The assistant also needs an Anthropic API key. The steps are in capstone_streamlit_app/README.md.

## Where things are

| Path | Contents |
|---|---|
| capstone_streamlit_app/ | Final dashboard: app.py, the LLM assistant, the training scripts, the dataset and the serialized models (.joblib) |
| src/ | Scripts for the class activities and for the capstone preprocessing and modeling |
| data/processed/ | Processed tables the scripts read. The raw Big99 export is confidential and is not in the repository |
| models/ | Serialized models from the ANN activity and Partial Exam 2 |
| reports/ | Run logs, metrics and figures |
| exam2_app/ and exam2_pipeline.py | Partial Exam 2 (smart grid) |
| docs/ | Data dictionary and glossary of the capstone |

## Team 5

- Farid Isaac Alfaro Omaña
- Andrés Correa Solís
- Lorenzo Casares
- Patricio Millet Esquivel
- Carlos Yael Pérez
- José Monroy Mercado
