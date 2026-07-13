# Presentation Reporting

Erzeugt ein einheitliches Reporting - exakt 5 Standardplots je Universum
(+ 1 optionaler Modellinterpretations-Plot) + Vergleichsplots/CSV +
Validierungs-/Zusammenfassungs-/Folienbericht - aus bereits vorhandenen
Backtest-, Alpaca- und Log-Dateien.

Diese Skripte trainieren kein Modell neu, veraendern keine bestehende
Trading-Logik und schreiben ausschliesslich in
`alpaca_trading/presentation_reporting/output/`.

## Ausfuehren

```bash
python alpaca_trading/presentation_reporting/generate_presentation_reports.py
python alpaca_trading/presentation_reporting/generate_presentation_reports.py --universe original_tech
```

## Dateien

- `reporting_config.py` - zentrale Pfade + Plot-Stil (Farben, Schrift, DPI, Groesse) - identisch fuer alle Universen
- `presentation_data_loader.py` - laedt und validiert alle Quelldateien, liefert READY/PRELIMINARY/MISSING/INVALID
- `presentation_metrics.py` - Sharpe, Max Drawdown, Top-K-Rekonstruktion, Signal-/Order-Statistiken (alles berechnet, nichts hartkodiert)
- `presentation_plots.py` - gemeinsame Matplotlib-Plot-Bausteine (>= 220 DPI, identische Schrift/Farben/Groesse)
- `generate_presentation_reports.py` - Orchestrierung + Markdown-Berichte
- `tests/test_presentation_metrics.py` - Plausibilitaetstests (kein pytest noetig, einfacher Assert-Runner)

## Plots je Universum (`output/<universe>/`)

- `01_backtest_vs_benchmark.png` - historischer Backtest, Top-K-Strategie vs. Benchmark
- `02_top_k_results.png` - Top-1 bis Top-5 vs. Buy-and-Hold der Universums-Aktien
- `03_hourly_alpaca_vs_benchmark.png` - Hybrid-Szenario fuer den stuendlichen Serverbetrieb (real + klar markierte Simulation)
- `04_daily_alpaca_vs_benchmark.png` - Daily Paper Trading (isolierte Simulation je Universum)
- `05_signal_selection_analysis.png` - Auswahlhaeufigkeit, Probability, Rang, Kauf/Verkauf, Haltedauer je Aktie
- `06_signal_probability_distribution.png` (optional) - Verteilung der Modell-Wahrscheinlichkeiten

Fehlt echte Datenbasis fuer einen Plot, erscheint eine klar beschriftete
Status-Grafik (MISSING/PRELIMINARY). Plot 03 nutzt nur dann eine Simulation,
wenn reale Hourly-Punkte als Anker und eine belastbare Hourly-Renditebasis
vorhanden sind; simulierte Abschnitte sind sichtbar markiert.

## Vergleichsplots (`output/comparison/`)

`01_backtest_comparison.png`, `02_hourly_comparison.png`,
`03_daily_comparison.png`, `04_signal_comparison.png`,
`05_final_universe_comparison.png`, plus `final_universe_comparison.csv`.

## Wichtigster Datenstand (siehe data_validation_report.md fuer Details)

Die Vollstaendigkeitspruefung am Ende jedes Laufs zeigt pro Universum, ob
alle 5 Pflichtplots aus echten Daten erzeugt werden konnten.
