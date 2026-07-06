# LSTM Experiment Changelog

## Aktuelle Erweiterung: Stündliche Alpaca-Signale

### Geaenderte Dateien

- `tech_stock_prediction/alpaca_trading/alpaca_config.py`
- `tech_stock_prediction/alpaca_trading/signal_generator.py`
- `tech_stock_prediction/alpaca_trading/paper_trading_engine.py`
- `tech_stock_prediction/alpaca_trading/portfolio_rebalancer.py`
- `tech_stock_prediction/alpaca_trading/alpaca_client.py`
- `tech_stock_prediction/alpaca_trading/run_paper_trading.py`
- `tech_stock_prediction/alpaca_trading/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/scripts/17_export_alpaca_model/export_outperformance_alpaca_model.py`

### Neue Dateien

- `tech_stock_prediction/experiments/exp_2_lstm/models/outperformance_lstm_metadata.json`

### Was wurde fachlich geaendert?

Die Alpaca-Pipeline unterstuetzt jetzt einen konfigurierbaren Timeframe:

- `1Hour` fuer stündliche Paper-Trading-Signale
- `1Day` fuer den bisherigen Tagesmodus

Standard fuer Phase 2 ist `1Hour`. Das Modell bleibt aber das finale
Outperformance-LSTM mit der Featuregruppe `Technical + Relative Strength`.
Es wird kein Standard-LSTM und keine Threshold-Regel verwendet.

Wichtig: Das aktuelle Modell wurde auf Tagesdaten trainiert. Bei `1Hour`
werden die gleichen Feature-Namen auf Stundenbars berechnet. Deshalb gibt die
Pipeline eine Warnung aus, dass sich die Bedeutung der Features aendert und
die Daily-Backtest-Ergebnisse nicht direkt auf Stundenbasis uebertragbar sind.

### Neue Log-Spalten

`paper_signals.csv` enthaelt jetzt zusaetzlich:

- `timeframe`
- `bar_timestamp`
- `feature_window_end`
- `top_k`

Alte Logdateien mit altem Schema werden automatisch als Legacy-Datei
archiviert, bevor eine neue CSV mit sauberem Header geschrieben wird.

### Ausfuehrung

Stündliche Signale:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --signals-only --top-k 5 --timeframe 1Hour
```

Stündlicher Dry-Run:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --dry-run --top-k 5 --timeframe 1Hour
```

Daily-Signale:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --signals-only --top-k 5 --timeframe 1Day
```

## Aktuelle Erweiterung: Finaler Alpaca-Export

### Geaenderte Dateien

- `tech_stock_prediction/alpaca_trading/signal_generator.py`
- `tech_stock_prediction/alpaca_trading/paper_trading_engine.py`
- `tech_stock_prediction/alpaca_trading/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue Dateien

- `tech_stock_prediction/experiments/exp_2_lstm/conf/outperformance_alpaca_features.txt`
- `tech_stock_prediction/experiments/exp_2_lstm/scripts/17_export_alpaca_model/export_outperformance_alpaca_model.py`
- `tech_stock_prediction/experiments/exp_2_lstm/scripts/17_export_alpaca_model/__init__.py`

### Neue generierte Dateien

- `tech_stock_prediction/experiments/exp_2_lstm/models/outperformance_lstm_model.pth`
- `tech_stock_prediction/experiments/exp_2_lstm/models/outperformance_lstm_scaler.pkl`
- `tech_stock_prediction/experiments/exp_2_lstm/data/processed/lstm_outperformance_alpaca_export_summary.csv`

### Was wurde fachlich geaendert?

Fuer Alpaca wird jetzt explizit die finale Variante exportiert:

- Outperformance-LSTM
- Featuregruppe `Technical + Relative Strength`
- Benchmark `QQQ` fuer Tech-Universen
- Top-K-Strategie mit Standardwert `TOP_K=1`

Der Signal-Generator laedt nicht das Standard-LSTM, sondern erwartet bewusst
die separaten Outperformance-Dateien. Dadurch wird verhindert, dass im Paper
Trading aus Versehen das falsche Modell verwendet wird.

### Ausfuehrung

Finales Modell exportieren:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/17_export_alpaca_model/export_outperformance_alpaca_model.py
```

Nur Signale testen:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --signals-only
```

Dry-Run testen:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --dry-run
```

Erst danach Paper Orders senden:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --execute
```

`--execute` braucht Alpaca Paper-Trading-Keys in der lokalen `.env`.

## Aktuelle Erweiterung: Struktur aufgeraeumt

### Geaenderte Dateien

- `tech_stock_prediction/run_lstm_pipeline.py`
- `tech_stock_prediction/experiments/exp_2_lstm/conf/params.yaml`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue Dateien

- `tech_stock_prediction/experiments/exp_2_lstm/conf/lstm_features.txt`

### Verschobene Dateien

Diese alten Zwischenexperimente wurden nicht geloescht, sondern nach
`scripts/archive/` verschoben:

- `scripts/archive/08_threshold_backtest/`
- `scripts/archive/09_lstm_tuning/`
- `scripts/archive/12_tuned_final_test/`
- `scripts/archive/13_comparison/`

### Was wurde fachlich geaendert?

Die normale LSTM-Hauptpipeline enthaelt jetzt nur noch die aktuell relevanten
Schritte:

- Data Preparation
- Sequence Creation
- Model Training
- Model Testing
- Backtesting
- Top-K Backtest
- Outperformance-LSTM
- Feature-Ablation
- Robustheitspruefung
- Visualization

Threshold-Backtest, LSTM-Tuning, Tuned-Final-Test und der alte
Modellvergleich bleiben im Archiv erhalten. Sie werden nicht mehr
standardmaessig ausgefuehrt.

Zusaetzlich besitzt `exp_2_lstm` jetzt eine eigene Featureliste unter
`conf/lstm_features.txt`. Dadurch haengt die LSTM-Pipeline nicht mehr von einem
entfernten Random-Forest-Skriptordner ab.

### Warum ist das sinnvoll?

Die Projektstruktur zeigt jetzt klarer, was zur aktuellen Hauptpipeline gehoert
und was aeltere Experimente waren. Dadurch koennen Teammitglieder schneller
erkennen, welche Skripte fuer den aktuellen Stand wichtig sind.

## Aktuelle Erweiterung: Robustheitspruefung

### Geaenderte Dateien

- `tech_stock_prediction/run_lstm_pipeline.py`
- `tech_stock_prediction/experiments/exp_2_lstm/conf/params.yaml`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue Dateien

- `scripts/16_robustness/robustness_check.py`
- `scripts/16_robustness/__init__.py`

### Neue generierte Dateien

- `data/processed/lstm_robustness_walk_forward_summary.csv`
- `data/processed/lstm_robustness_sensitivity_summary.csv`
- `data/processed/lstm_robustness_concentration_summary.csv`
- `reports/lstm_robustness_report.md`
- `plots/lstm_robustness_walk_forward_difference.png`
- `plots/lstm_robustness_top_k_sensitivity.png`
- `plots/lstm_robustness_cost_sensitivity.png`
- `plots/lstm_robustness_concentration.png`

### Was wurde fachlich ergaenzt?

Die zwei aktuell besten Varianten werden jetzt auf Robustheit geprueft:

- `standard_lstm` mit `Technical + Market`
- `outperformance_lstm` mit `Technical + Relative Strength`

Es wird keine neue Modellarchitektur eingefuehrt. Stattdessen wird dieselbe
LSTM-Logik ueber mehrere Walk-Forward-Fenster neu trainiert und getestet.

Geprueft werden:

- mehrere Testfenster
- Top-K von 1 bis 5
- Transaktionskosten von 0 %, 0.05 %, 0.10 % und 0.20 %
- Sharpe Ratio
- Max Drawdown
- Volatilitaet
- Turnover
- Anzahl Trades
- Konzentration auf einzelne Aktien oder wenige Handelstage

### Ausfuehrung

Separat starten:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/16_robustness/robustness_check.py
```

Oder in `run_lstm_pipeline.py` aktivieren:

```text
RUN_ROBUSTNESS_CHECK = True
```

Die normale Pipeline startet diesen Check bewusst nicht automatisch, weil
mehrere LSTMs neu trainiert werden.

## Aktuelle Erweiterung: Feature-Gruppen-Ablation

### Geaenderte Dateien

- `tech_stock_prediction/run_lstm_pipeline.py`
- `tech_stock_prediction/experiments/exp_2_lstm/conf/params.yaml`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue Dateien

- `scripts/15_feature_ablation/lstm_feature_ablation.py`
- `scripts/15_feature_ablation/__init__.py`

### Neue generierte Dateien

- `data/processed/lstm_feature_ablation_summary.csv`
- `plots/lstm_feature_ablation_returns.png`
- `plots/lstm_feature_ablation_sharpe.png`
- `plots/lstm_feature_ablation_drawdown.png`
- `plots/lstm_feature_ablation_top_k.png`

### Was wurde fachlich ergaenzt?

Es wurde eine Feature-Gruppen-Ablation fuer das LSTM ergaenzt. Dabei wird nicht
blind ein neues Modell gebaut, sondern dieselbe LSTM-Struktur mit
unterschiedlichen Featuregruppen getestet.

Getestet werden:

- Technical only
- Technical + Volatility
- Technical + Volume
- Technical + Momentum/Trend
- Technical + Market
- Technical + Relative Strength
- Final Feature Set

Fuer jede Gruppe werden zwei Varianten trainiert:

- Standard-LSTM mit Target `Target`
- Outperformance-LSTM mit Target `Outperform_QQQ_Target`

Danach wird fuer jede Variante ein Top-K Backtest auf den Testdaten berechnet.
Zusaetzlich zu Return und Difference werden jetzt auch realistischere
Backtest-Metriken gespeichert:

- Transaktionskosten von 0.1 Prozent
- Sharpe Ratio
- Max Drawdown
- Volatilitaet
- Anzahl Trades
- durchschnittlicher Turnover

### Ausfuehrung

Die Ablation wird bewusst nicht automatisch in der normalen Pipeline gestartet,
weil mehrere LSTMs trainiert werden.

Separat starten:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/15_feature_ablation/lstm_feature_ablation.py
```

Oder in `run_lstm_pipeline.py` aktivieren:

```text
RUN_FEATURE_ABLATION = True
```

### Interpretation

Die wichtigste Datei ist:

```text
data/processed/lstm_feature_ablation_summary.csv
```

Wichtigste Spalten:

- `feature_group`: getestete Featuregruppe
- `model_name`: Standard-LSTM oder Outperformance-LSTM
- `best_top_k`: beste Anzahl gekaufter Aktien pro Tag
- `f1_score`: Klassifikationsmetrik auf Testdaten
- `strategy_return`: Top-K Rendite nach Transaktionskosten
- `buy_and_hold_return`: Vergleichsrendite im gleichen Zeitraum
- `difference`: Strategy Return minus Buy-and-Hold
- `sharpe_ratio`: Rendite im Verhaeltnis zum Risiko
- `max_drawdown`: groesster zwischenzeitlicher Verlust
- `number_of_trades`: Anzahl Positionswechsel
- `average_turnover`: durchschnittlicher Portfolio-Wechsel

Die Plots zeigen Returns, Sharpe Ratio, Max Drawdown und das beste Top-K je
Featuregruppe.

## Aktuelle Erweiterung: Outperformance-LSTM

### Geaenderte Dateien

- `tech_stock_prediction/run_lstm_pipeline.py`
- `tech_stock_prediction/experiments/exp_2_lstm/conf/params.yaml`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue Dateien

- `scripts/14_outperformance_lstm/outperformance_lstm.py`
- `scripts/14_outperformance_lstm/__init__.py`

### Neue generierte Dateien

- `data/processed/lstm_outperformance_predictions.csv`
- `data/processed/lstm_outperformance_results.csv`
- `data/processed/lstm_outperformance_top_k_results.csv`
- `data/processed/lstm_outperformance_comparison_summary.csv`
- `plots/09_outperformance_lstm_metrics.png`
- `plots/10_outperformance_top_k_return.png`
- `plots/11_outperformance_cumulative_comparison.png`

### Was wurde fachlich ergaenzt?

Das neue LSTM sagt nicht mehr nur voraus, ob eine Aktie morgen steigt.
Stattdessen sagt es voraus, ob eine Aktie morgen QQQ schlaegt.

Das passt besser zur Top-K-Strategie. Top-K kauft pro Tag nur die Aktien mit den
staerksten Modellwahrscheinlichkeiten. Dafuer ist ein relatives Ranking oft
sinnvoller als eine einfache absolute Steigt/Faellt-Vorhersage.

Neu hinzugekommen sind Marktfeatures fuer QQQ, SPY und VIX:

- QQQ Return
- SPY Return
- VIX Veraenderung
- QQQ Momentum 20 Tage
- SPY Momentum 20 Tage
- Abstand von QQQ zum 200-Tage-Durchschnitt
- Abstand von SPY zum 200-Tage-Durchschnitt

Zusaetzlich berechnet das Skript Relative-Strength-Features. Damit sieht das
Modell, ob eine Aktie staerker oder schwaecher als QQQ/SPY war.

Das neue Target ist:

```text
Outperform_QQQ_Target = 1, wenn die Aktie am naechsten Handelstag besser laeuft als QQQ
```

Die naechste QQQ-Rendite wird nur fuer das Target benutzt, nicht als Feature.
Dadurch entsteht kein Lookahead.

### Ausfuehrung

Die komplette LSTM-Pipeline kann wie gewohnt gestartet werden:

```bash
python tech_stock_prediction/run_lstm_pipeline.py
```

Nur das neue Outperformance-LSTM kann so gestartet werden:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/14_outperformance_lstm/outperformance_lstm.py
```

### Interpretation

Wichtige Dateien:

- `lstm_outperformance_results.csv`: Klassifikationsmetriken und einfache Strategie
- `lstm_outperformance_top_k_results.csv`: Top 1 bis Top 5 Outperformance-Strategien
- `lstm_outperformance_comparison_summary.csv`: kurze Zusammenfassung
- `plots/11_outperformance_cumulative_comparison.png`: kumulierter Vergleich gegen Standard-LSTM und Buy-and-Hold

Wichtig beim Vergleichen:

- Nicht nur Accuracy anschauen
- Besonders Top-K Strategy Return gegen Buy-and-Hold vergleichen
- Pruefen, ob Outperformance-LSTM bessere Rankings liefert als das Standard-LSTM

## Aktuelle Erweiterung: Feature-Normalisierung und erweitertes Top-K

### Geaenderte Dateien

- `tech_stock_prediction/experiments/exp_2_lstm/conf/params.yaml`
- `scripts/02_sequence_creation/create_sequences.py`
- `scripts/03_model_training/train_lstm.py`
- `scripts/04_model_testing/evaluate_lstm.py`
- `scripts/10_top_k_backtest/top_k_backtest.py`
- `scripts/09_lstm_tuning/lstm_tuning.py`
- `scripts/12_tuned_final_test/tuned_final_test.py`
- `scripts/13_comparison/compare_lstm_results.py`
- `scripts/11_visualization/generate_lstm_plots.py`
- `tech_stock_prediction/run_lstm_pipeline.py`
- `tech_stock_prediction/experiments/exp_2_lstm/README.md`
- `tech_stock_prediction/experiments/exp_2_lstm/CHANGELOG.md`

### Neue Dateien

- `scripts/11_visualization/generate_lstm_plots.py`
- `scripts/11_visualization/__init__.py`
- `scripts/12_tuned_final_test/tuned_final_test.py`
- `scripts/12_tuned_final_test/__init__.py`
- `scripts/13_comparison/compare_lstm_results.py`
- `scripts/13_comparison/__init__.py`

### Neue generierte Dateien

- `models/lstm_feature_scaler.pkl`
- `plots/01_lstm_training_history.png`
- `plots/02_lstm_test_metrics.png`
- `plots/03_lstm_confusion_matrix.png`
- `plots/04_lstm_cumulative_backtest.png`
- `plots/05_lstm_threshold_backtest.png`
- `plots/06_lstm_top_k_backtest.png`
- `plots/07_lstm_tuning_f1.png`
- `data/processed/lstm_tuned_final_results.csv`
- `data/processed/lstm_tuned_top_k_results.csv`
- `data/processed/lstm_tuned_test_predictions_*.csv`
- `data/processed/lstm_standard_results.csv`
- `data/processed/lstm_standard_top_k_results.csv`
- `data/processed/lstm_model_comparison_summary.csv`

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

Zusaetzlich wurde ein Plot-Skript ergaenzt. Es visualisiert die wichtigsten
Ergebnisse fuer die Praesentation.

Das Tuning wurde ausserdem richtig in finale Testergebnisse eingebaut. Vorher
wurden Tuning-Konfigurationen nur auf dem Validation-Set verglichen. Jetzt
werden die besten Validation-Konfigurationen genommen, neu mit Train +
Validation trainiert und danach auf dem Testset bewertet.

Die Ergebnisse werden jetzt in einer zentralen Summary zusammengefuehrt. Dabei
werden native Testzeitraeume und der gemeinsame ueberschneidende Zeitraum
getrennt ausgewiesen. So ist sichtbar, wann Buy-and-Hold wegen einer anderen
Sequenzlaenge unterschiedlich ist.

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
- `data/processed/lstm_tuned_final_results.csv`
- `data/processed/lstm_tuned_top_k_results.csv`
- `data/processed/lstm_model_comparison_summary.csv`
- `plots/*.png`

Beim Interpretieren ist besonders wichtig:

- `lstm_backtest_results.csv`: einfache Strategie mit `Prediction = 1`
- `lstm_threshold_results.csv`: strengere Wahrscheinlichkeitsgrenzen
- `lstm_top_k_results.csv`: nur die staerksten Signale pro Tag
- `lstm_tuning_results.csv`: Validation-Vergleich verschiedener LSTM-Settings
- `lstm_tuned_final_results.csv`: echte Testmetriken der besten Tuning-Modelle
- `lstm_tuned_top_k_results.csv`: Top-K Backtest der besten Tuning-Modelle
- `lstm_model_comparison_summary.csv`: wichtigste faire Gesamtuebersicht

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
