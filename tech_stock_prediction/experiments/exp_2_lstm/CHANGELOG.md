# LSTM Experiment Changelog

## Aktuelle Erweiterung: Feature-Normalisierung und erweitertes Top-K

### Geaenderte Dateien

- `tech_stock_prediction/experiments/exp_2_lstm/conf/params.yaml`
- `scripts/02_sequence_creation/create_sequences.py`
- `scripts/03_model_training/train_lstm.py`
- `scripts/04_model_testing/evaluate_lstm.py`
- `scripts/10_top_k_backtest/top_k_backtest.py`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue generierte Dateien

- `models/lstm_feature_scaler.pkl`

Diese Datei entsteht beim Erstellen der LSTM-Sequenzen. Sie wird nicht
committed, weil sie in `.gitignore` ueber `experiments/*/models/*.pkl`
ausgeschlossen ist.

### Was wurde fachlich ergaenzt?

Die LSTM-Features werden jetzt vor der Sequenz-Erstellung mit einem
`StandardScaler` normalisiert. Der Scaler wird nur auf den Trainingsdaten
gefittet. Validation- und Testdaten werden danach nur transformiert.

Das verhindert Data Leakage, weil keine Informationen aus Validation oder Test
in die Vorbereitung der Trainingsdaten einfliessen.

Die Normalisierung ist fuer LSTM sinnvoll, weil neuronale Netze mit Gradient
Descent lernen. Features auf sehr unterschiedlichen Skalen koennen das Training
erschweren. Beim Random Forest war das weniger wichtig, weil Baum-Modelle mit
Split-Schwellen arbeiten.

Der Top-K Backtest testet jetzt:

- Top 1
- Top 2
- Top 3
- Top 4
- Top 5

Top 2 und Top 4 helfen, die Handelsentscheidung feiner zu vergleichen.

### Ausfuehrung

Die komplette LSTM-Pipeline kann so gestartet werden:

```bash
python tech_stock_prediction/run_lstm_pipeline.py
```

Oder in IntelliJ mit:

```text
11 Run LSTM Pipeline
```

Wichtige Ergebnisdateien:

- `data/processed/lstm_test_predictions.csv`
- `data/processed/lstm_backtest_results.csv`
- `data/processed/lstm_threshold_results.csv`
- `data/processed/lstm_top_k_results.csv`
- `data/processed/lstm_tuning_results.csv`

Beim Interpretieren ist besonders wichtig:

- `lstm_backtest_results.csv`: einfache Strategie mit `Prediction = 1`
- `lstm_threshold_results.csv`: strengere Wahrscheinlichkeitsgrenzen
- `lstm_top_k_results.csv`: nur die staerksten Signale pro Tag
- `lstm_tuning_results.csv`: Validation-Vergleich verschiedener LSTM-Settings

## Vorherige Erweiterung: LSTM-Tuning und Top-K Backtest

### Geaenderte Dateien

- `tech_stock_prediction/run_lstm_pipeline.py`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue Dateien

- `scripts/08_threshold_backtest/threshold_backtest.py`
- `scripts/08_threshold_backtest/__init__.py`
- `scripts/09_lstm_tuning/lstm_tuning.py`
- `scripts/09_lstm_tuning/__init__.py`
- `scripts/10_top_k_backtest/top_k_backtest.py`
- `scripts/10_top_k_backtest/__init__.py`

### Warum ist diese Aenderung sinnvoll?

Das LSTM erzeugt Wahrscheinlichkeiten. Wenn daraus sofort mit einem festen
Threshold eine `Prediction` gebaut wird, kann das Modell sehr viele
Kaufsignale erzeugen. Ein hoher Recall sieht dann gut aus, aber die Strategie
kauft eventuell zu oft.

Der Threshold Backtest prueft, ob ein strengerer Threshold bessere
Trading-Entscheidungen liefert. Dadurch koennen wir besser sehen, ob das Modell
nur oft `steigt` sagt oder ob hohe Wahrscheinlichkeiten wirklich bessere
Signale sind.

Das neue LSTM-Tuning testet mehrere einfache Modellkonfigurationen. So koennen
wir pruefen, ob andere Sequenzlaengen, Hidden Sizes, Dropout-Werte oder Learning
Rates bessere Validation-Ergebnisse liefern.

Der neue Top-K Backtest nutzt nicht mehr jedes `Prediction = 1` Signal.
Stattdessen werden pro Handelstag nur die Aktien mit der hoechsten
LSTM-Wahrscheinlichkeit gekauft. Das ist fachlich sinnvoll, weil eine
Tradingentscheidung nicht nur aus `steigt` oder `faellt` bestehen sollte,
sondern auch die Staerke des Signals beruecksichtigen kann.

### Ausfuehrung

Die komplette LSTM-Pipeline kann so gestartet werden:

```bash
python tech_stock_prediction/run_lstm_pipeline.py
```

Oder in IntelliJ mit:

```text
11 Run LSTM Pipeline
```

Nur der Threshold Backtest kann so gestartet werden:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/08_threshold_backtest/threshold_backtest.py
```

Dafuer muss vorher `evaluate_lstm.py` gelaufen sein, damit
`lstm_test_predictions.csv` existiert.

Nur der Top-K Backtest kann so gestartet werden:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/10_top_k_backtest/top_k_backtest.py
```

Nur das LSTM-Tuning kann so gestartet werden:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/09_lstm_tuning/lstm_tuning.py
```

Das Tuning laeuft bewusst nicht automatisch in der normalen Pipeline, weil es
mehrere Modelle trainiert und dadurch laenger dauern kann. In
`run_lstm_pipeline.py` kann `RUN_TUNING = True` gesetzt werden, wenn es mit der
Pipeline laufen soll.

### Interpretation

Wichtige Spalten in `lstm_threshold_results.csv`:

- `Threshold`: getesteter Grenzwert fuer ein Kaufsignal
- `Predicted_Up_Share`: Anteil der Kaufsignale
- `Buy_Signals`: Anzahl der Kaufsignale
- `F1_Score`: Klassifikationsqualitaet
- `Strategy_Return`: Rendite der LSTM-Strategie
- `Buy_And_Hold_Return`: Vergleichsrendite ohne Modell
- `Difference`: Strategie minus Buy-and-Hold

Ein guter Threshold sollte nicht nur einen hohen F1-Score haben, sondern auch
eine bessere oder zumindest naeher an Buy-and-Hold liegende Strategy Return
erzielen.

Wichtige Spalten in `lstm_top_k_results.csv`:

- `top_k`: Wie viele Aktien pro Tag gekauft werden
- `strategy_return`: Rendite der Top-K Strategie
- `buy_and_hold_return`: Vergleichsrendite ohne Modell
- `difference`: Top-K Strategie minus Buy-and-Hold
- `average_number_of_positions`: Durchschnittliche Anzahl gehaltener Aktien
- `number_of_trading_days`: Anzahl getesteter Handelstage

Wichtige Spalten in `lstm_tuning_results.csv`:

- `sequence_length`: Anzahl vergangener Tage pro LSTM-Sequenz
- `hidden_size`: Groesse des LSTM
- `dropout`: Regularisierung gegen Overfitting
- `learning_rate`: Lerngeschwindigkeit
- `accuracy`, `precision`, `recall`, `f1_score`: Klassifikationsmetriken
- `predicted_up_share`: Anteil der Kaufsignale
- `strategy_return`: Rendite mit Prediction = 1
- `difference`: Strategy Return minus Buy-and-Hold

### Git-Hinweis

Generierte Dateien sollen nicht committed werden. Dazu gehoeren besonders:

- `data/processed/*.csv`
- `data/processed/*.npz`
- `models/*.pth`

Diese Dateien sind in `.gitignore` bereits abgedeckt und koennen lokal jederzeit
neu erzeugt werden.
