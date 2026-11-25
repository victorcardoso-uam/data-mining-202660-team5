# Glossary

Terms used in the capstone, in plain language, with the number each one takes in this project. The numbers come from capstone_streamlit_app/models/metrics.json and models/explainability.json.

| Term | What it means | In this project |
|---|---|---|
| Slow mover | A unit that has been in stock for more than 120 days. It is what the models predict (slow_mover_flag = 1). | 281 of 1,091 units (25.8%). |
| Purchase lot | Units bought and received together. They share the arrival date, so they share days in stock and the label. | 97 lots. The largest has 133 units. |
| Data leakage | Giving a model an input that already contains the answer. The score looks great and means nothing. | days_in_stock defines the target, so it is not an input. The arrival date, exchange rate, reservation flag and transfer history were removed for the same reason. |
| Lot-grouped split | Splitting training and test data by lot, so every unit of a lot stays on the same side. A random split by rows puts near-identical units on both sides and inflates the score. | In Milestone 2, on an earlier snapshot, a random forest scored a ROC-AUC of 0.877 with a random split and 0.786 with a lot-grouped split. |
| Hold-out | Data set aside before training and used only to score the finished model. | 275 units from 25 lots. Training used 816 units from 72 lots. |
| Cross-validation (4-fold, lot-grouped) | The lots are divided into four groups. Each group takes a turn as the test set while the other three train the model. The result is a mean and a standard deviation. | Used to choose the champion, because one hold-out of 25 lots is noisy. |
| Champion | The model we recommend. It is the one with the highest mean ROC-AUC in cross-validation. | MLR, with 0.747 ± 0.061. |
| MLR (multiple linear regression) | A straight-line model that predicts a number from several inputs. Here it predicts the logarithm of days in stock. | A unit is WATCH from 58 predicted days and HIGH from 74. |
| Decision tree | A model that asks yes/no questions about the inputs until it reaches an answer. | Depth 4, at least 10 units per leaf. |
| Random forest | Many decision trees that vote. | 400 trees of depth 8. |
| ANN (artificial neural network) | Layers of simple units that learn weights from the data. It works well with a lot of data and overfits easily with little. | Two hidden layers of 32 and 16 units, 1,825 parameters, trained on 816 units. |
| Overfitting | The model memorises the training data and gets worse on new data. | The ANN's validation loss was lowest at epoch 41 (0.330) and rose to 0.403 by epoch 71, while the training loss kept falling. |
| Early stopping | Keeping the weights from the epoch where the validation loss was lowest. | Epoch 41. |
| Recall | Of the units that really are slow movers, the share the model catches. | 56 of 70 in the hold-out: 0.80. The target was 0.75, so it is met. |
| Precision | Of the units the model flags, the share that really are slow movers. | 56 of 156: 0.36. About 64 of every 100 flags are false alarms. |
| F1 | One number that combines precision and recall (their harmonic mean). It is low when either of them is low. | 0.50. The target was 0.70 and no model reached it. |
| Accuracy | Share of all units classified correctly. It misleads when one class is much smaller than the other. | 0.585 for the champion. Answering "not slow" for every unit would score 0.745 and catch no slow mover. |
| Confusion matrix | A 2x2 table that counts hits and misses. | Champion on the hold-out: 56 slow movers caught, 14 missed, 100 false alarms, 105 healthy units left alone. |
| ROC-AUC | The chance that a slow mover gets a higher score than a healthy unit picked at random. 0.5 is a coin flip and 1.0 is perfect. | 0.717 on the hold-out for the champion. |
| R² | Share of the variation in the target that the model explains. | 0.127 for the MLR on log days in stock, about 13%. |
| Cut-off (threshold) | The score from which a unit is flagged. A lower cut-off catches more slow movers and raises more false alarms. | Chosen on the training lots as the highest-F1 value that keeps recall at 0.75 or more. |
| Risk band | LOW, WATCH or HIGH, according to the model score, for units that have not reached 120 days. | The What-If tab shows the band and the action that goes with it. |
| Permutation importance | Shuffle one input and measure how much the score drops. A big drop means the model leans on that input. | Champion: appearance 0.087, page counter 0.066, warehouse 0.058, brand 0.053. |
| One-hot encoding | Turning a category into 0/1 columns, one per level. | 14 inputs become 39 columns. |
| Median imputation | Filling a missing value with the median of the training data. | 7 missing page counters, filled with 137. |
| StandardScaler | Rescaling a numeric column to mean 0 and standard deviation 1, so no input weighs more just because of its units. | Applied to the page counter and the landed cost after a log transform. |
| Confounding | Two things move together because of a third one. | MID shows fewer slow movers because its stock is newer (median 20 days in stock, against 69 in MEX and MTY and 106 in GDL). It does not mean MID sells faster. |
| Censored label | We only see units that are still in stock, so "not slow" means "has not reached 120 days yet". | Sale dates are not in the table. Real time to sale is not validated. |
