# Gesamtanalyse - Vergleich aller vier Universen

Automatisch erzeugt von generate_cross_universe_analysis.py. Betrachtet das Projekt als ein gesamtes Forschungsprojekt statt vier Einzeluntersuchungen. Keine Modelle wurden neu trainiert, keine Handelslogik veraendert - nur Auswertung, Vergleich und Visualisierung bereits vorhandener Ergebnisse (siehe auch die einzelnen `wissenschaftliche_analyse.md`-Dateien je Universum, aus denen dieselben zugrundeliegenden Zahlen stammen).

Kernfrage: Welches Universum eignet sich fuer unser Modell am besten und warum?

## Vergleich 1: Datensatz

| Universum | Aktien | Benchmark | Ø Volatilitaet (30T) | Ø Korrelation (Diversifikation) | Trainingszeitraum | Testzeitraum |
| --- | --- | --- | --- | --- | --- | --- |
| Original Tech | 10 | QQQ | 2.66% | 0.56 | 2019-10-16 - 2024-06-28 | 2025-07-07 - 2026-07-09 |
| Tech ohne Nvidia | 9 | QQQ | 2.60% | 0.55 | 2019-10-16 - 2024-06-28 | 2025-07-07 - 2026-07-09 |
| Neue Tech-Aktien | 10 | QQQ | 3.42% | 0.38 | 2019-10-16 - 2024-06-28 | 2025-07-07 - 2026-07-09 |
| Defensive Non-Tech | 10 | SPY | 1.50% | 0.43 | 2019-10-16 - 2024-06-28 | 2025-07-07 - 2026-07-09 |

Branchenzusammensetzung (qualitativ, allgemein bekannte Einordnung, nicht Teil der Pipeline-Daten):

- Original Tech: Automobil/Tech, E-Commerce/Cloud, Halbleiter, Software, Technologie/Hardware, Technologie/Internet, Technologie/Software
- Tech ohne Nvidia: Automobil/Tech, E-Commerce/Cloud, Halbleiter, Software, Technologie/Hardware, Technologie/Internet, Technologie/Software
- Neue Tech-Aktien: Cloud-Infrastruktur, Cybersecurity, Halbleiter, Hardware, Hardware/Server, Netzwerktechnik, Software/Analytics, Software/Cloud, Software/Datenbanken
- Defensive Non-Tech: Einzelhandel, Energie, Gesundheit/Pharma, Getraenke, Getraenke/Konsumgueter, Konsumgueter, Pharma

Marktkapitalisierung wurde in der Pipeline nicht erhoben (kein Fundamentaldaten-Feld dafuer vorhanden) und wird hier bewusst nicht geschaetzt oder erfunden.

Hoechste durchschnittliche Volatilitaet: Neue Tech-Aktien. Niedrigste: Defensive Non-Tech. Staerkste Diversifikation (niedrigste Durchschnittskorrelation): Neue Tech-Aktien. Geringste Diversifikation (hoechste Durchschnittskorrelation): Original Tech.

Auswirkung auf das Modell: hoehere Volatilitaet bedeutet groessere Tag-zu-Tag-Preisschwankungen, was die Klassifikationsaufgabe (naechster Tag outperformt Benchmark oder nicht) tendenziell schwieriger macht, da das Signal-Rausch-Verhaeltnis sinkt. Eine hohe durchschnittliche Korrelation zwischen den Aktien eines Universums bedeutet, dass sie sich groesstenteils gemeinsam bewegen - das reduziert den potenziellen Mehrwert einer Top-K-Auswahl gegenueber Buy-and-Hold, da es weniger Gelegenheit fuer 'die eine Aktie, die sich vom Rest abhebt' gibt.

## Vergleich 2: Modellleistung

| Universum | Modell | Accuracy | Precision | Recall | Test-F1 | Validation-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Original Tech | Standard-LSTM | 0.521 | 0.523 | 0.864 | 0.651 | 0.646 |
| Original Tech | Outperformance-LSTM | 0.523 | 0.502 | 0.588 | 0.541 | 0.534 |
| Tech ohne Nvidia | Standard-LSTM | 0.518 | 0.522 | 0.823 | 0.639 | 0.641 |
| Tech ohne Nvidia | Outperformance-LSTM | 0.517 | 0.496 | 0.503 | 0.500 | 0.527 |
| Neue Tech-Aktien | Standard-LSTM | 0.506 | 0.518 | 0.788 | 0.625 | 0.649 |
| Neue Tech-Aktien | Outperformance-LSTM | 0.476 | 0.465 | 0.487 | 0.475 | 0.551 |
| Defensive Non-Tech | Standard-LSTM | 0.506 | 0.509 | 0.932 | 0.659 | 0.676 |
| Defensive Non-Tech | Outperformance-LSTM | 0.528 | 0.506 | 0.541 | 0.523 | 0.527 |

Beste Test-F1 (Outperformance-LSTM): Original Tech (0.541). Schwaechste: Neue Tech-Aktien (0.475).

Erklaerungsansatz: die Modellkennzahlen haengen direkt mit den in Vergleich 1 gemessenen Datensatzeigenschaften zusammen - Universen mit hoeherer durchschnittlicher Korrelation zwischen den Aktien liefern dem Modell tendenziell konsistentere, leichter lernbare Muster (die Aktien bewegen sich aehnlicher, wodurch das allgemeine Marktsignal deutlicher wird), waehrend Universen mit sehr heterogenen Einzeltiteln (z. B. neue, kleinere Tech-Werte mit eigenen, titelspezifischen Nachrichtenlagen) schwerer vorherzusagen sein koennen.

## Vergleich 3: Backtest

| Universum | Buy-and-Hold | Standard-LSTM | Outperformance-LSTM (bestes Top-K) | Sharpe | Max Drawdown |
| --- | --- | --- | --- | --- | --- |
| Original Tech | +54.3% | +39.9% | Top-1: +105.7% | 2.17 | -24.3% |
| Tech ohne Nvidia | +58.5% | +36.8% | Top-5: +47.2% | 1.81 | -13.9% |
| Neue Tech-Aktien | +57.6% | +30.4% | Top-2: +23.5% | 0.70 | -42.9% |
| Defensive Non-Tech | +21.0% | +15.2% | Top-1: +20.7% | 1.14 | -13.2% |

Hoechste Stabilitaet (Sharpe): Original Tech. Geringster Max Drawdown: Defensive Non-Tech. Staerkste Outperformance gegenueber Benchmark: Original Tech (+74.7%).

Robustheit: ein hoher Sharpe bei gleichzeitig geringem Max Drawdown deutet auf eine Strategie hin, die nicht nur im Mittel gut abschneidet, sondern auch wenige extreme Verlustphasen durchlebt - das ist fuer die praktische Einsetzbarkeit wichtiger als die reine absolute Rendite, da grosse Drawdowns in der Praxis zu vorzeitigem Strategieabbruch fuehren koennen.

## Vergleich 4: Top-K

| Universum | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| Original Tech | +105.7% | +96.4% | +67.0% | +60.5% | +68.2% |
| Tech ohne Nvidia | +14.5% | +40.0% | +34.2% | +46.3% | +47.2% |
| Neue Tech-Aktien | -4.5% | +23.5% | +22.3% | +2.1% | +4.6% |
| Defensive Non-Tech | +20.7% | +5.6% | +15.6% | +19.9% | +20.4% |

Profitiert am staerksten von Konzentration (Top-1 deutlich besser als Top-5): Original Tech (+37.5% Unterschied). Am wenigsten (oder sogar umgekehrt): Tech ohne Nvidia (-32.7%).

Diagramm: `comparison/08_top_k_heatmap.png`

## Vergleich 5: Signalanalyse

| Universum | Ø Probability | Std | Haeufigste Top-1-Aktie | Nie gewaehlt |
| --- | --- | --- | --- | --- |
| Original Tech | 0.36 | 0.05 | AMD | 5 von 10 Aktien |
| Tech ohne Nvidia | 0.36 | 0.05 | AMD | 5 von 9 Aktien |
| Neue Tech-Aktien | 0.42 | 0.10 | MU | 6 von 10 Aktien |
| Defensive Non-Tech | 0.36 | 0.02 | COST | 7 von 10 Aktien |

Eindeutigste Signale im Mittel (hoechste Ø Probability): Neue Tech-Aktien. Unsicherste Signale (Ø Probability am naechsten an 0.50): Original Tech.

Von den Universen bevorzugte Branchen (haeufigste Top-1-Aktie je Universum):

- Original Tech: AMD (Halbleiter)
- Tech ohne Nvidia: AMD (Halbleiter)
- Neue Tech-Aktien: MU (Halbleiter)
- Defensive Non-Tech: COST (Einzelhandel)

## Vergleich 6: Paper Trading

### Hourly Paper Trading

Fuer alle vier Universen liegt aktuell keine belastbare Hourly-Zeitreihe vor (nur 1-2 Zeitstempel aus einem einmaligen Dry-Run vor Umstellung auf den taeglichen Scheduler, siehe data_validation_report.md). Ein Vergleich ist auf dieser Datenbasis nicht seriös moeglich.

### Daily Paper Trading

| Universum | Zeitraum | Portfolio-Rendite | Benchmark-Rendite | Differenz | Trades (Kauf/Verkauf) | Max Drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| Original Tech | 2026-07-06 - 2026-07-10 | -3.0% | +0.4% | -3.4% | 12/5 | -6.7% |
| Tech ohne Nvidia | 2026-07-06 - 2026-07-10 | -4.2% | +0.4% | -4.6% | 7/3 | -7.4% |
| Neue Tech-Aktien | 2026-07-06 - 2026-07-10 | +1.6% | +0.4% | +1.2% | 7/3 | -2.4% |
| Defensive Non-Tech | 2026-07-06 - 2026-07-10 | -2.4% | +0.5% | -2.8% | 6/3 | -3.2% |

Beste Daily-Paper-Trading-Performance (vorlaeufig, wenige Tage): Neue Tech-Aktien. Hoechste Handelsaktivitaet: Original Tech (17 Trades).

Alle Daily-Alpaca-Ergebnisse beruhen auf sehr kurzen Beobachtungsfenstern (wenige Handelstage je Universum) und sind daher nur als Plausibilitaetssignal zu werten, nicht als belastbarer Live-Vergleich.

## Vergleich 7: Modellverhalten

Fuer ein LSTM am ehesten 'einfacher' erscheint Original Tech (hoechste Test-F1), am 'schwierigsten' Neue Tech-Aktien (niedrigste Test-F1).

Rolle der einzelnen Aspekte:

- Momentum: in allen vier Universen identisch als Feature-Gruppe vorhanden; seine Nuetzlichkeit haengt davon ab, wie stark Trends in der jeweiligen Aktiengruppe tatsaechlich anhalten (Trendpersistenz), was wiederum von der Branchenstruktur abhaengt (z. B. reagieren Halbleiterwerte oft staerker/schneller auf Nachrichten als defensive Konsumguetertitel).
- Volatilitaet: hoehere durchschnittliche Volatilitaet (siehe Vergleich 1) erschwert tendenziell die Klassifikationsaufgabe, da kurzfristiges Rauschen das eigentliche Richtungssignal ueberlagert.
- Benchmark: QQQ (techlastig) und SPY (breiter Markt) unterscheiden sich in Zusammensetzung und Volatilitaet - ein techlastiges Universum gegen QQQ zu vergleichen bedeutet, dass Universum und Benchmark aehnlicheren Kraeften unterliegen, als wenn ein defensives Universum gegen SPY antritt.
- Korrelation/Relative Strength: Universen mit hoeherer interner Korrelation (siehe Vergleich 1) bieten der Outperformance-Formulierung (relativ zum Benchmark) potenziell weniger Differenzierungsspielraum zwischen den Einzeltiteln, da sie sich aehnlicher bewegen.

## Vergleich 8: Visualisierung

Alle folgenden Diagramme verwenden dieselbe Farbpalette (ein fester Farbcode je Universum, `UNIVERSE_COLORS` in `reporting_config.py`), dieselbe Schriftgroesse, Achsenbeschriftung und DPI (220) wie alle anderen Plots dieses Projekts.

- `06_model_metrics_comparison.png` - Accuracy/Precision/Recall/F1 je Universum und Modell
- `07_backtest_strategy_comparison.png` - Buy-and-Hold vs. Standard-LSTM vs. Outperformance-LSTM
- `08_top_k_heatmap.png` - Top-1 bis Top-5 Rendite je Universum (Heatmap)
- `09_average_probability_comparison.png` - durchschnittliche Modell-Wahrscheinlichkeit je Universum
- `10_daily_paper_trading_portfolio_comparison.png` - Portfolioentwicklung (Index) aller Universen
- `11_sharpe_drawdown_comparison.png` - Sharpe Ratio und Max Drawdown je Universum
- `12_overall_ranking.png` - Gesamtranking (siehe Vergleich 9)

Bereits vorhandene Vergleichsplots aus `generate_presentation_reports.py` (01-05, gleiche Palette/Stil) ergaenzen dies um Backtest-, Hourly-, Daily- und Signal-Outperformance je Universum.

## Vergleich 9: Ranking

Methodik: jedes Kriterium wird pro Universum min-max-normalisiert (0 = schwaechstes, 1 = staerkstes beobachtetes Universum bei diesem Kriterium); der Gesamtscore ist der ungewichtete Mittelwert aller verfuegbaren normalisierten Kriterien. 'Praktische Einsetzbarkeit' und 'wissenschaftliche Qualitaet' werden als Synthese der unten stehenden Kriterien interpretiert, nicht als zusaetzliche eigene Zahlen (um dieselbe Evidenz nicht doppelt zu zaehlen).

| Kriterium | Original Tech | Tech ohne Nvidia | Neue Tech-Aktien | Defensive Non-Tech |
| --- | --- | --- | --- | --- |
| Aussagekraft (# Backtest-Handelstage) | 235.000 | 235.000 | 235.000 | 235.000 |
| Generalisierung (# Top-K mit positiver Differenz) | 5.000 | 0.000 | 0.000 | 0.000 |
| Modellqualitaet (Test-F1) | 0.541 | 0.500 | 0.475 | 0.523 |
| Paper Trading (Daily-Alpaca-Differenz) | -0.034 | -0.046 | 0.012 | -0.028 |
| Robustheit (Max Drawdown, weniger negativ = besser) | -0.243 | -0.139 | -0.429 | -0.132 |
| Stabilitaet (Sharpe) | 2.170 | 1.808 | 0.696 | 1.140 |
| Trading Performance (Backtest-Differenz) | 0.747 | 0.162 | -0.075 | -0.007 |

| Universum | Gesamtscore (0-1) | Rang |
| --- | --- | --- |
| Original Tech | 0.761 | 1 |
| Defensive Non-Tech | 0.415 | 2 |
| Tech ohne Nvidia | 0.412 | 3 |
| Neue Tech-Aktien | 0.214 | 4 |

Diagramm: `comparison/12_overall_ranking.png`

## Vergleich 10: Key Findings

Ueberraschend vs. erwartungskonform:

- Erwartungskonform waere, dass die drei Tech-Universen (Ø Differenz +27.8%) deutlich staerker abschneiden als das defensive Kontroll-Universum (-0.7%), da die Strategie urspruenglich fuer Tech-Aktien entwickelt wurde.
- Gesamtranking: Original Tech > Defensive Non-Tech > Tech ohne Nvidia > Neue Tech-Aktien (siehe Vergleich 9 fuer die Methodik).
- Groesster Unterschied im Backtest: Original Tech (+74.7%) vs. Neue Tech-Aktien (-7.5%) - eine Spanne von +82.1%.
- Neue Tech-Aktien zeigt die im Mittel eindeutigsten Modell-Wahrscheinlichkeiten - ein Hinweis darauf, dass das Modell in diesem Universum konsistentere Muster erkennt.

## Praesentationsempfehlungen

### Muss unbedingt in die Praesentation

- `comparison/12_overall_ranking.png` + Vergleich-9-Tabelle: beantwortet die Kernfrage direkt und objektiv nachvollziehbar.
- `comparison/01_backtest_comparison.png` und `comparison/07_backtest_strategy_comparison.png`: zeigt den zentralen wissenschaftlichen Befund (schlaegt die Strategie den Benchmark, und wo).
- `comparison/08_top_k_heatmap.png`: kompakte, auf einen Blick verstaendliche Darstellung eines der aufwendigsten Analyseteile.
- Je Universum Plot 01 (Backtest) aus dem eigenen Ordner: zeigt die Story pro Universum konkret, nicht nur aggregiert.

### Darf optional gezeigt werden

- `comparison/06_model_metrics_comparison.png`: interessant fuer ein technisches Publikum, aber fuer die Kernaussage nicht zwingend.
- `comparison/09_average_probability_comparison.png` und Signalanalyse-Details: gute Ergaenzung, falls Zeit fuer eine Modellverhalten-Folie ist.
- `comparison/11_sharpe_drawdown_comparison.png`: relevant fuer ein risikofokussiertes Publikum, sonst optional.
- Je Universum Plot 05/06 (Signalanalyse/Wahrscheinlichkeitsverteilung): gut als Vertiefung bei Nachfragen.

### Kann weggelassen werden

- Hourly-Paper-Trading-Plots (Plot 03 je Universum, `comparison/02_hourly_comparison.png`): zeigen aktuell durchgaengig nur den Hinweistext 'keine ausreichenden Daten' - liefert keine zusaetzliche Erkenntnis in der Praesentation.
- Die vollstaendige Kriterientabelle aus Vergleich 9 als eigene Folie: zu dicht fuer eine Praesentationsfolie, besser nur das Endergebnis (Ranking-Balken) zeigen und die Tabelle im Backup-Anhang bereithalten.
- Detaillierte Order-/Trade-Listen aus dem Daily Paper Trading: fuer die Kernaussage irrelevant, da die Stichprobe ohnehin zu klein fuer belastbare Aussagen ist.

Ziel: eine schluessige Geschichte - Datensatz unterscheidet sich (Vergleich 1) -> das beeinflusst Modellleistung (Vergleich 2) -> das zeigt sich im Backtest (Vergleich 3/4) -> Ranking fasst es objektiv zusammen (Vergleich 9). Nicht jede erzeugte Grafik muss gezeigt werden.
