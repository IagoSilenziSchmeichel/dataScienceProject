# Experiment 2: LSTM

Dieses Experiment testet ein LSTM-Modell fuer die Tech-Stock-Prediction.
Das LSTM nutzt dieselben finalen Features wie das Random-Forest-Experiment,
damit beide Modelle vergleichbar bleiben.

## Ablauf

Die komplette LSTM-Pipeline kann ueber den Runner gestartet werden:

```bash
python tech_stock_prediction/run_lstm_pipeline.py
```

In IntelliJ kann dafuer die Run Configuration `11 Run LSTM Pipeline` genutzt werden.

Die Pipeline fuehrt diese Schritte aus:

1. Datencheck
2. Sequenzen erstellen
3. LSTM trainieren
4. LSTM testen
5. Einfacher Backtest
6. Threshold Backtest

Der feste Threshold fuer die normale LSTM-Evaluation steht in:

```text
conf/params.yaml
```

Aktuell:

```text
PREDICTION_THRESHOLD: 0.50
```

## Warum Thresholds?

Das LSTM gibt fuer jede Zeile eine Wahrscheinlichkeit aus. Eine einfache
Vorhersage `Prediction = 1` ist zu grob, weil sie nur sagt, ob die
Wahrscheinlichkeit ueber einem festen Grenzwert liegt.

Wenn das Modell sehr oft `1` vorhersagt, sieht der Recall oft gut aus, aber die
Strategie kann trotzdem zu viele Kaufsignale erzeugen. Deshalb testen wir
mehrere Thresholds.

Ein hoeherer Threshold bedeutet:

- weniger Kaufsignale
- nur staerkere Modell-Signale werden gehandelt
- Recall kann sinken
- Precision und Strategiequalitaet koennen steigen

## Threshold Backtest

Das Skript

```text
scripts/08_threshold_backtest/threshold_backtest.py
```

testet diese Thresholds:

```text
0.50, 0.55, 0.60, 0.65, 0.70
```

Pro Threshold werden berechnet:

- Accuracy
- Precision
- Recall
- F1-Score
- Predicted-Up-Share
- Anzahl der Kaufsignale
- Strategy Return
- Buy-and-Hold Return
- Difference

Die Ergebnisse werden gespeichert unter:

```text
data/processed/lstm_threshold_results.csv
```

## Generierte Dateien

Diese Dateien entstehen beim Ausfuehren der Pipeline:

```text
data/processed/lstm_sequences.npz
data/processed/lstm_test_metadata.csv
data/processed/lstm_test_predictions.csv
data/processed/lstm_backtest_results.csv
data/processed/lstm_threshold_results.csv
data/processed/lstm_training_history.csv
models/lstm_model.pth
```

Generierte Daten und Modelle gehoeren normalerweise nicht in Git. Besonders
gross oder automatisch reproduzierbar sind:

```text
data/processed/lstm_sequences.npz
models/lstm_model.pth
```

Sie werden ueber `.gitignore` ausgeschlossen.
