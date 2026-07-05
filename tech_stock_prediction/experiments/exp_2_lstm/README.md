# Experiment 2: LSTM

Dieses Experiment enthaelt die aktuelle LSTM-Pipeline fuer die
Tech-Stock-Prediction. Der Fokus liegt auf einer klaren, erklaerbaren
Projektstruktur:

- Standard-LSTM fuer kurzfristige Steigt/Faellt-Vorhersagen
- Top-K Backtest fuer bessere Handelsentscheidungen
- Outperformance-LSTM gegen QQQ
- Feature-Gruppen-Ablation
- Robustheitspruefung
- Ergebnisplots
- Export des finalen Outperformance-LSTM fuer Alpaca Paper Trading

Alte Zwischenexperimente wurden nicht geloescht, sondern nach
`scripts/archive/` verschoben.

## Pipeline starten

Die normale LSTM-Pipeline kann ueber den Runner gestartet werden:

```bash
python tech_stock_prediction/run_lstm_pipeline.py
```

In IntelliJ kann dafuer die Run Configuration `11 Run LSTM Pipeline` genutzt
werden.

## Aktuelle Hauptpipeline

Die Hauptpipeline besteht aus diesen Schritten:

1. `scripts/01_data_preparation/prepare_lstm_data.py`
2. `scripts/02_sequence_creation/create_sequences.py`
3. `scripts/03_model_training/train_lstm.py`
4. `scripts/04_model_testing/evaluate_lstm.py`
5. `scripts/05_backtesting/backtest_lstm.py`
6. `scripts/10_top_k_backtest/top_k_backtest.py`
7. `scripts/14_outperformance_lstm/outperformance_lstm.py`
8. `scripts/15_feature_ablation/lstm_feature_ablation.py`
9. `scripts/16_robustness/robustness_check.py`
10. `scripts/11_visualization/generate_lstm_plots.py`

Der Alpaca-Export ist ein separater Schritt, weil er ein Modell fuer die
Inference-Dateien speichert:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/17_export_alpaca_model/export_outperformance_alpaca_model.py
```

Hinweis: Feature-Ablation und Robustheitspruefung trainieren mehrere LSTMs.
Die komplette Pipeline ist deshalb langsamer als ein einzelner Modelllauf,
enthaelt dafuer aber den aktuellen vollstaendigen Projektstand.

## Archivierte Zwischenexperimente

Diese aelteren Experimente bleiben erhalten, sind aber nicht mehr Teil der
Hauptpipeline:

```text
scripts/archive/08_threshold_backtest/
scripts/archive/09_lstm_tuning/
scripts/archive/12_tuned_final_test/
scripts/archive/13_comparison/
```

Sie koennen weiterhin separat gestartet werden, falls alte Ergebnisse
nachvollzogen werden sollen. Fuer die aktuelle Projektlogik sind sie aber nur
noch Archiv.

## Feature-Normalisierung

Die Standard-LSTM-Features stehen in:

```text
conf/lstm_features.txt
```

Diese Datei enthaelt die 19 Features, mit denen das Standard-LSTM arbeitet.
Dadurch haengt `exp_2_lstm` nicht mehr von alten oder entfernten
Random-Forest-Skriptordnern ab.

Vor der Sequenz-Erstellung werden die LSTM-Features mit einem `StandardScaler`
normalisiert.

Wichtig:

- Der Scaler wird nur auf den Trainingsdaten gefittet
- Validation-Daten werden nur transformiert
- Test-Daten werden nur transformiert
- Dadurch vermeiden wir Data Leakage

Der trainierte Scaler wird gespeichert unter:

```text
models/lstm_feature_scaler.pkl
```

## Top-K Backtest

`Prediction = 1 -> kaufen` ist fuer Trading oft zu grob. Top-K nutzt deshalb
die Modellwahrscheinlichkeiten:

- Pro Tag werden Aktien nach Wahrscheinlichkeit sortiert
- Es werden nur die besten Signale gekauft
- Getestet werden Top 1 bis Top 5 Aktien pro Tag

Wichtige Datei:

```text
data/processed/lstm_top_k_results.csv
```

## Outperformance-LSTM

Das Outperformance-LSTM fragt nicht mehr nur:

```text
Steigt die Aktie morgen?
```

Sondern:

```text
Schlaegt die Aktie morgen QQQ?
```

Das passt besser zur Top-K-Strategie, weil wir nicht irgendeine steigende Aktie
suchen, sondern die staerksten Aktien relativ zum Tech-Markt.

Wichtige Dateien:

```text
data/processed/lstm_outperformance_predictions.csv
data/processed/lstm_outperformance_results.csv
data/processed/lstm_outperformance_top_k_results.csv
data/processed/lstm_outperformance_comparison_summary.csv
```

## Finaler Alpaca-Export

Fuer Phase 2 wird explizit die finale Outperformance-Top-K-Variante exportiert:

- Modell: Outperformance-LSTM
- Featuregruppe: Technical + Relative Strength
- Target: Aktie schlaegt QQQ am naechsten Handelstag
- Standardstrategie: Top-K, zuerst `TOP_K=1`

Die finale Featureliste liegt hier:

```text
conf/outperformance_alpaca_features.txt
```

Der Export erzeugt diese Inference-Dateien:

```text
models/outperformance_lstm_model.pth
models/outperformance_lstm_scaler.pkl
data/processed/lstm_outperformance_alpaca_export_summary.csv
```

Diese Modell- und Ergebnisdateien sind generiert und gehoeren nicht in Git.

## Feature-Ablation

Die Feature-Ablation testet, welche Featuregruppen wirklich helfen.

Getestet werden unter anderem:

- Technical only
- Technical + Volatility
- Technical + Volume
- Technical + Momentum/Trend
- Technical + Market
- Technical + Relative Strength
- Final Feature Set

Wichtige Datei:

```text
data/processed/lstm_feature_ablation_summary.csv
```

## Robustheitspruefung

Die Robustheitspruefung testet die besten Modellvarianten ueber mehrere
Zeitfenster und mit unterschiedlichen Handelsannahmen.

Geprueft werden:

- Walk-Forward-Zeitfenster
- Top-K von 1 bis 5
- Transaktionskosten
- Sharpe Ratio
- Max Drawdown
- Volatilitaet
- Turnover
- Anzahl Trades
- Konzentration auf einzelne Aktien oder wenige Tage

Wichtige Dateien:

```text
data/processed/lstm_robustness_walk_forward_summary.csv
data/processed/lstm_robustness_sensitivity_summary.csv
data/processed/lstm_robustness_concentration_summary.csv
reports/lstm_robustness_report.md
```

## Generierte Dateien

Diese Dateien entstehen beim Ausfuehren der Pipeline und gehoeren normalerweise
nicht in Git:

```text
data/processed/*.npz
models/*.pth
models/*.pkl
```

Die wichtigsten grossen Dateien sind:

```text
data/processed/lstm_sequences.npz
models/lstm_model.pth
models/lstm_feature_scaler.pkl
```

Sie werden ueber `.gitignore` ausgeschlossen.
