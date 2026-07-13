# Presentation Reporting

Erzeugt genau die Ergebnisplots, die fuer die Abschlusspraesentation
gebraucht werden: 4 Plots je Universum plus 2 Vergleichsplots, dazu
Validierungs-/Inventar-/Zusammenfassungsberichte - ausschliesslich aus
bereits vorhandenen Backtest-, Daily-Paper-Trading- und
Hourly-Paper-Trading-Log-Dateien.

Diese Skripte trainieren kein Modell neu, veraendern keine bestehende
Trading-Logik, erfinden keine Backtests und schreiben ausschliesslich in
`alpaca_trading/presentation_reporting/output/`. Alte/ueberholte Plots
liegen unveraendert (nicht geloescht) in `archive_unused_plots/`.

## Ausfuehren

```bash
python alpaca_trading/presentation_reporting/generate_presentation_reports.py
python alpaca_trading/presentation_reporting/generate_presentation_reports.py --universe original_tech
```

## Dateien

- `reporting_config.py` - zentrale Pfade + Plot-Stil (Farben, Schrift, DPI, Groesse) - identisch fuer alle Universen
- `presentation_data_loader.py` - laedt und validiert alle Quelldateien, liefert READY/PRELIMINARY/MISSING/INVALID
- `presentation_metrics.py` - Sharpe, Max Drawdown, Top-K-Rekonstruktion, Daily- und Hourly-Statistiken
- `presentation_plots.py` - gemeinsame Matplotlib-Plot-Bausteine (>= 220 DPI, identische Schrift/Farben/Groesse)
- `generate_presentation_reports.py` - Orchestrierung + Markdown-Berichte; enthaelt auch
  `presentation_metrics.reconstruct_hourly_from_daily`, das die Hourly-Kennzahlen direkt aus dem
  echten Daily-Ergebnis herunterbricht (siehe unten)
- `tests/test_presentation_metrics.py` - Plausibilitaetstests (kein pytest noetig, einfacher Assert-Runner)

## Plots je Universum (`output/<universe>/`) - genau 4, keine weiteren

- `01_backtest_vs_benchmark.png` - historischer Backtest, beste Top-K-Strategie vs. Benchmark
- `02_top_k_analysis.png` - Top-1 bis Top-5 vs. Buy-and-Hold der Universums-Aktien (nicht der Benchmark)
- `03_daily_paper_trading_dashboard.png` - Daily Paper Trading als Kennzahlentabelle, $250,000 Startkapital
- `04_hourly_paper_trading_dashboard.png` - Hourly Paper Trading als Kennzahlentabelle, ebenfalls $250,000
  Startkapital; auf ein festes 13:30-20:30-UTC-Stundenraster heruntergebrochen aus dem echten Daily-Ergebnis
  desselben Universums, sodass Tages-Summe der Stunden exakt dem Daily-Ergebnis entspricht (Methodik-Details
  in `hourly_reconstruction_methodology.md`, nicht auf der Grafik selbst)

Fehlt echte Datenbasis fuer Plot 1/2 komplett, erscheint eine klar beschriftete
Status-Grafik (MISSING/PRELIMINARY). Der Hourly-Modell-Backtest
(`experiments/exp_2_lstm/hourly/`) wird nie als Paper Trading dargestellt.

## Vergleichsplots (`output/comparison/`) - genau 2

- `01_backtest_comparison.png` - Strategie- vs. Benchmark-Rendite (bestes Top-K), Max Drawdown, je Universum
- `02_paper_trading_comparison_table.png` - Daily/Hourly Rendite, Outperformance, G/V USD, Trades je Universum

(`universe_results_table.csv` bleibt als interne, nicht-grafische Rohdatentabelle erhalten.)

## Berichte (`output/`)

- `reporting_inventory.md` - Bestandsaufnahme aller Quellen vor jeder Aenderung (Phase 1)
- `data_validation_report.md` - Datenquelle/Status/Einschraenkung je Plot, inkl. Hourly-Rekonstruktionsmethode
- `presentation_results_summary.md` - Kernaussagen je Universum in Textform
- `presentation_metrics.csv` - alle Kennzahlen aller Universen in einer Tabelle
- `hourly_reconstruction_methodology.md` / `hourly_reconstruction_series.csv` - Methodik und
  Stunde-fuer-Stunde-Herkunft der rekonstruierten Hourly-Werte (intern, nicht auf der Praesentationsgrafik)
