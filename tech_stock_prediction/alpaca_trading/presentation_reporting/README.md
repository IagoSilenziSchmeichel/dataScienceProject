# Presentation Reporting

Erzeugt ein einheitliches Reporting (4 Standardplots pro Universum +
Vergleichsplots/CSV + Validierungs-/Zusammenfassungs-/Folienbericht) aus
bereits vorhandenen Backtest-, Alpaca- und Log-Dateien.

Diese Skripte trainieren kein Modell neu, veraendern keine bestehende
Trading-Logik und schreiben ausschliesslich in
`alpaca_trading/presentation_reporting/output/`.

## Ausfuehren

```bash
python alpaca_trading/presentation_reporting/generate_presentation_reports.py
python alpaca_trading/presentation_reporting/generate_presentation_reports.py --universe original_tech
```

## Dateien

- `reporting_config.py` - zentrale Pfade + Plot-Stil (Farben, Schrift, DPI, Groesse)
- `presentation_data_loader.py` - laedt und validiert alle Quelldateien, liefert READY/PRELIMINARY/MISSING/INVALID
- `presentation_metrics.py` - Sharpe, Max Drawdown, Top-K-Rekonstruktion, Signal-Statistiken (alles berechnet, nichts hartkodiert)
- `presentation_plots.py` - gemeinsame Matplotlib-Plot-Bausteine
- `generate_presentation_reports.py` - Orchestrierung + Markdown-Berichte
- `tests/test_presentation_metrics.py` - Plausibilitaetstests (kein pytest noetig, einfacher Assert-Runner)
- `output/<universe>/01..04_*.png` - die vier Standardplots je Universum
- `output/comparison/` - universenuebergreifende Vergleichsplots + `final_universe_comparison.csv`
- `output/data_validation_report.md`, `presentation_results_summary.md`, `presentation_slide_plan.md`

## Wichtigster Datenstand (siehe data_validation_report.md fuer Details)

- Backtest (Plot 1) liegt aktuell nur fuer `original_tech` vor.
- Hourly Alpaca (Plot 2) hat nur 1-2 Zeitstempel je Universum - keine Zeitreihe moeglich.
- Daily Alpaca (Plot 3) hat 5 Handelstage je Universum aus der isolierten
  Simulation unter `alpaca_trading/per_universe_daily_mark_to_market/` -
  bewusst nicht aus den geteilten Konto-Logs unter `alpaca_trading/logs/`,
  da diese alle vier Universen im selben Account fahren und daher nicht
  sauber pro Universum zurechenbar sind.
