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
8. Tuned Final Test
9. Ergebnisvergleich erstellen
10. Outperformance-LSTM testen
11. Ergebnisplots erstellen

## Saubere Vergleichslogik

Die LSTM-Auswertung trennt jetzt zwei Arten von Vergleichen:

- `native_period`: Jedes Modell wird auf seinem eigenen verfuegbaren Testzeitraum bewertet.
- `common_overlap_period`: Alle Modelle werden nur auf dem gemeinsamen ueberschneidenden Zeitraum bewertet.

Das ist wichtig, weil unterschiedliche `sequence_length`-Werte unterschiedlich
viele erste Testtage verlieren. Eine Sequenzlaenge von 60 startet spaeter als
eine Sequenzlaenge von 10.

Buy-and-Hold darf deshalb nur dann unterschiedlich sein, wenn auch der
Testzeitraum unterschiedlich ist. Fuer jeden Backtest wird Buy-and-Hold immer
auf exakt demselben Zeitraum wie die Strategie berechnet.

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

## Finaler Test mit Tuning

Das Tuning allein ist noch kein finales Testergebnis. Es zeigt nur, welche
Konfiguration auf dem Validation-Set gut war.

Deshalb gibt es jetzt einen zusaetzlichen finalen Tuned-Test:

```text
scripts/12_tuned_final_test/tuned_final_test.py
```

Dieses Skript macht:

- beste Konfiguration nach Validation-F1 auswaehlen
- beste Konfiguration nach Validation-Strategy-Return auswaehlen
- mit diesen Konfigurationen ein neues LSTM trainieren
- dafuer Train + Validation als Trainingsdaten verwenden
- danach ehrlich auf dem Testset evaluieren

Dadurch wird das Tuning wirklich in finale Testergebnisse eingebaut.

Die Ergebnisse werden gespeichert unter:

```text
data/processed/lstm_tuned_final_results.csv
data/processed/lstm_tuned_top_k_results.csv
```

Wichtig: Unterschiedliche `sequence_length`-Werte erzeugen unterschiedlich
lange Testzeiträume. Deshalb wird jedes Tuned-Ergebnis gegen Buy-and-Hold im
gleichen Zeitraum verglichen.

## Ergebnisvergleich

Die zentrale Zusammenfassung wird hier gespeichert:

```text
data/processed/lstm_model_comparison_summary.csv
```

Sie enthaelt:

- Standard-LSTM mit einfacher `Prediction = 1` Regel
- Standard-LSTM mit bester Top-K-Regel
- Tuned-LSTM nach bestem Validation-F1
- Tuned-LSTM nach bestem Validation-Strategy-Return
- native Zeitraeume
- gemeinsamen ueberschneidenden Zeitraum

Diese Datei ist die wichtigste Datei, wenn Ergebnisse fair verglichen werden
sollen.

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

## Outperformance-LSTM

Bisher hat das Standard-LSTM vorhergesagt, ob eine Aktie am naechsten Tag
absolut steigt. Fuer eine Top-K-Strategie ist aber oft die bessere Frage:

```text
Welche Aktie ist morgen staerker als der Markt?
```

Deshalb gibt es jetzt eine zusaetzliche LSTM-Variante:

```text
scripts/14_outperformance_lstm/outperformance_lstm.py
```

Dieses Modell sagt nicht mehr direkt `steigt` oder `faellt` voraus, sondern:

```text
Outperform_QQQ_Target = 1, wenn die Aktie morgen eine bessere Rendite als QQQ hat
```

Warum passt das besser zu Top-K?

- Top-K kauft nur die besten Signale pro Tag
- Dafuer ist ein Ranking zwischen Aktien wichtig
- Outperformance fragt genau nach relativer Staerke
- Absolute Steigt/Faellt-Vorhersagen sind schwer, weil der Gesamtmarkt viele Aktien gleichzeitig bewegt

Neu eingebaute Marktfeatures:

- `QQQ_Return`
- `SPY_Return`
- `VIX_Change`
- `QQQ_Momentum_20`
- `SPY_Momentum_20`
- `QQQ_Distance_to_MA200`
- `SPY_Distance_to_MA200`

Neu eingebaute Relative-Strength-Features:

- `Relative_Return_QQQ`
- `Relative_Return_SPY`
- `Relative_Momentum_20_QQQ`
- `Relative_Momentum_20_SPY`

Wichtig: Die Future-Markt-Rendite wird nur fuer das Target genutzt, nicht als
Feature. So vermeiden wir Lookahead und Data Leakage.

Die wichtigsten Ergebnisdateien sind:

```text
data/processed/lstm_outperformance_predictions.csv
data/processed/lstm_outperformance_results.csv
data/processed/lstm_outperformance_top_k_results.csv
data/processed/lstm_outperformance_comparison_summary.csv
plots/09_outperformance_lstm_metrics.png
plots/10_outperformance_top_k_return.png
plots/11_outperformance_cumulative_comparison.png
```

Nur das Outperformance-LSTM kann so gestartet werden:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/14_outperformance_lstm/outperformance_lstm.py
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
data/processed/lstm_tuned_final_results.csv
data/processed/lstm_tuned_top_k_results.csv
data/processed/lstm_model_comparison_summary.csv
data/processed/lstm_outperformance_results.csv
data/processed/lstm_outperformance_top_k_results.csv
data/processed/lstm_outperformance_comparison_summary.csv
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

## Ergebnisplots

Die wichtigsten LSTM-Ergebnisse werden als PNG-Dateien gespeichert unter:

```text
plots/
```

Das Plot-Skript liegt hier:

```text
scripts/11_visualization/generate_lstm_plots.py
```

Es erzeugt diese Visualisierungen:

- Trainingsverlauf
- Testmetriken
- Confusion Matrix
- kumulierte Rendite LSTM vs. Buy-and-Hold
- Threshold Backtest
- Top-K Backtest
- Tuning-Vergleich
- Gesamtvergleich aller Modelle
- Outperformance-LSTM Metriken
- Outperformance Top-K Backtest
- kumulierter Vergleich Standard-LSTM vs. Outperformance-LSTM vs. Buy-and-Hold

Die Plots sind generierte Dateien und gehoeren nicht in Git.
