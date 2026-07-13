# Archiv-Manifest

Datum: 2026-07-13

Diese Dateien wurden beim Cleanup des Presentation-Reportings aus dem aktiven
`output/`-Ordner hierher verschoben (nicht geloescht). Sie sind ueberholt,
doppelt oder verletzen die neue Regel "genau 4 Plots je Universum + 2
Vergleichsplots, keine Signal-/Wahrscheinlichkeits-/Debug-Plots".

## Pro Universum (`<universum>/`)

| Datei | Grund |
| --- | --- |
| `02_top_k_results.png` | Altes Schema (vor Umbenennung auf `02_top_k_analysis.png`), inhaltlich ueberholt. |
| `03_daily_paper_trading_table.png` | Alter Dateiname vor Umbenennung auf `03_daily_paper_trading_dashboard.png`. |
| `03_hourly_alpaca_vs_benchmark.png` | Altes Schema, ersetzt durch die Hourly-Kennzahlentabelle. |
| `03_hourly_paper_trading_dashboard.png` | **Kritisch:** zeigte den historischen Hourly-Modell-Backtest (`hourly_outperformance_predictions.csv`, 712 Stunden) faelschlich als "Hourly Dashboard"/Paper Trading. Verstoesst gegen die Regel "Backtest niemals als Paper Trading zeigen". Archiviert, nicht wiederverwendet. |
| `04_daily_alpaca_vs_benchmark.png` | Altes Schema, ersetzt durch die Daily-Kennzahlentabelle. |
| `04_hourly_paper_trading_table.png` | Alter Dateiname vor Umbenennung auf `04_hourly_paper_trading_dashboard.png`. |
| `04_signal_stability_analysis.png` | Signalanalyse-Plot, laut neuer Vorgabe nicht Teil der 4 Pflichtplots. Nutzte zudem denselben Hourly-Backtest wie oben. |
| `05_daily_vs_hourly_comparison.png` | Fuenfter Plot pro Universum, laut neuer Vorgabe entfaellt er ("genau 4 Plots je Universum"). |
| `05_signal_selection_analysis.png` | Signalanalyse-Plot, explizit ausgeschlossen ("keine zusaetzlichen Signalanalyse-Plots"). |
| `06_signal_probability_distribution.png` | Wahrscheinlichkeitsverteilungs-Plot, explizit ausgeschlossen. |
| `wissenschaftliche_analyse.md` | Aelterer, per-Universum-Analysebericht aus einem frueheren Schema; Kernaussagen stehen jetzt in `presentation_results_summary.md`. |

## `comparison/`

| Datei | Grund |
| --- | --- |
| `02_hourly_comparison.png` ... `12_overall_ranking.png` | Fruehere Serie von 12 Einzelvergleichsplots (Signalanalyse, Wahrscheinlichkeiten, Ranking, Heatmaps etc.). Laut neuer Vorgabe genau 2 Vergleichsgrafiken statt vieler Einzelplots. |
| `backtest_comparison.png` | Alter, unnummerierter Vorlaeufer von `01_backtest_comparison.png`. |
| `hourly_dashboard_comparison.png`, `signal_stability_comparison.png`, `top_k_comparison.png` | Weitere ueberholte Einzelvergleichsplots. |
| `universe_results_table.png` | Altes Vergleichs-Schema (Gesamttabelle als drittes Vergleichsbild); ersetzt durch `01_backtest_comparison.png` + `02_paper_trading_comparison_table.png`. Die zugrundeliegende CSV (`universe_results_table.csv`) bleibt als interne Rohdatentabelle aktiv, ist aber keine Praesentationsgrafik. |
| `final_universe_comparison.csv`, `presentation_comparison_metrics.csv` | Aeltere Vergleichs-CSVs aus dem ueberholten Schema; enthielten teils den faelschlich als Paper Trading gelabelten Hourly-Backtest (`hourly_trades`-Werte aus `hourly_outperformance_predictions.csv`). Nicht mehr verwenden. |
| `gesamtanalyse.md` | Aeltere Gesamtanalyse aus einem frueheren Schema, inhaltlich durch `data_validation_report.md` + `presentation_results_summary.md` ersetzt. |

## Root (`output/`)

| Datei | Grund |
| --- | --- |
| `paper_trading_validation_report.md` | Alter Dateiname/Inhalt des Validierungsberichts, jetzt konsolidiert unter `data_validation_report.md` (inkl. Quelldateien- und Hourly-Szenario-Abschnitt). |
| `paper_trading_period_validation.md` | Inhalt (Quelldateien je Universum) in `data_validation_report.md` -> Abschnitt "Verwendete Quelldateien je Universum" integriert. |
| `plot_validation_report.md` | Dokumentierte explizit das fehlerhafte Schema, das den Hourly-Backtest als "Hourly Dashboard" auswies. Historisch relevant fuer die Fehlerursache, aber nicht als aktiver Bericht verwenden. |
| `presentation_slide_plan.md` | Folienplanung fuer eine PowerPoint-Praesentation; laut Auftrag wird aktuell noch keine PowerPoint erstellt. |

Nichts wurde geloescht. Bei Bedarf lassen sich alle Dateien aus diesem
Ordner in den passenden `output/`-Unterordner zurueckverschieben.
