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
7. Top-K Backtest

Der feste Threshold fuer die normale LSTM-Evaluation steht in:

```text
conf/params.yaml
```

Aktuell:

```text
PREDICTION_THRESHOLD: 0.50
```

## Feature-Normalisierung

Vor der Sequenz-Erstellung werden die LSTM-Features mit einem `StandardScaler`
normalisiert.

Wichtig:

- Der Scaler wird nur auf den Trainingsdaten gefittet
- Validation-Daten werden nur transformiert
- Test-Daten werden nur transformiert
- Dadurch vermeiden wir Data Leakage

Warum ist das wichtig?

Ein LSTM lernt mit Gradient Descent. Wenn einzelne Features sehr grosse Werte
haben und andere sehr kleine Werte, kann das Training instabiler werden.
Normalisierung bringt die Features auf eine vergleichbare Skala.

Beim Random Forest war das weniger wichtig, weil Entscheidungsbaeume mit
Schwellenwerten arbeiten und nicht direkt Gewichte ueber Gradienten lernen.

Der trainierte Scaler wird gespeichert unter:

```text
models/lstm_feature_scaler.pkl
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

## Warum LSTM-Tuning?

Das LSTM hat mehrere Stellschrauben. Zum Beispiel:

- Wie viele vergangene Tage als Sequenz genutzt werden
- Wie gross das LSTM ist
- Wie stark Dropout regularisiert
- Wie schnell das Modell lernt

Das Tuning-Skript testet mehrere einfache Kombinationen und vergleicht sie auf
dem Validation-Set. Das ist wichtig, weil das Test-Set nicht zum Optimieren
benutzt werden sollte.

Das Skript liegt hier:

```text
scripts/09_lstm_tuning/lstm_tuning.py
```

Die Ergebnisse werden gespeichert unter:

```text
data/processed/lstm_tuning_results.csv
```

Die normale Pipeline startet das Tuning nicht automatisch, weil es deutlich
laenger dauern kann. In `run_lstm_pipeline.py` kann dafuer `RUN_TUNING = True`
gesetzt werden.

## Warum Top-K Backtesting?

`Prediction = 1 -> kaufen` ist sehr einfach, aber fuer Trading oft zu grob.
Wenn das Modell sehr oft `steigt` sagt, entstehen zu viele Kaufsignale.

Top-K nutzt stattdessen die Wahrscheinlichkeiten des LSTM:

- Pro Tag werden die Aktien nach Wahrscheinlichkeit sortiert
- Es werden nur die besten Signale gekauft
- Getestet werden Top 1, Top 2, Top 3, Top 4 und Top 5 Aktien pro Tag

Dadurch pruefen wir, ob die hoechsten Modellwahrscheinlichkeiten wirklich die
besseren Tradingentscheidungen liefern.

Top 2 und Top 4 wurden ergaenzt, damit wir nicht nur sehr grobe Spruenge
zwischen 1, 3 und 5 Aktien vergleichen. So sehen wir genauer, wie stark die
Strategie von der Anzahl gekaufter Aktien pro Tag abhaengt.

Das Skript liegt hier:

```text
scripts/10_top_k_backtest/top_k_backtest.py
```

Die Ergebnisse werden gespeichert unter:

```text
data/processed/lstm_top_k_results.csv
```

## Warum schlaegt das Modell Buy-and-Hold bisher nicht?

Das LSTM erkennt viele steigende Tage, deshalb sind Recall und F1 besser als
beim Random Forest. Fuer eine Handelsstrategie reicht das aber noch nicht
automatisch aus. Wenn zu viele Signale gekauft werden, nimmt die Strategie auch
viele schwache oder falsche Signale mit.

Deshalb betrachten wir jetzt nicht nur Klassifikationsmetriken, sondern auch:

- Wie viele Kaufsignale entstehen
- Welche Wahrscheinlichkeiten besonders stark sind
- Ob Top-K-Auswahl bessere Renditen liefert
- Ob Strategy Return naeher an Buy-and-Hold herankommt

## Wichtige neue Ergebnisdateien

```text
data/processed/lstm_threshold_results.csv
data/processed/lstm_top_k_results.csv
data/processed/lstm_tuning_results.csv
models/lstm_feature_scaler.pkl
```

Diese CSV-Dateien sind generierte Ergebnisse. Sie koennen lokal angeschaut
werden, gehoeren aber normalerweise nicht in Git, weil sie jederzeit durch die
Skripte neu erzeugt werden koennen.

## Naechster Vergleich

Nach dieser Erweiterung sollten wir vergleichen:

- Standard-LSTM ohne neue Normalisierung gegen Standard-LSTM mit Normalisierung
- Top 1 bis Top 5 im Backtest
- Ob Top-K stabiler ist als die einfache Regel `Prediction = 1 -> kaufen`
- Optional spaeter: `NUM_LAYERS = 2`, damit LSTM-Dropout technisch aktiv wird
