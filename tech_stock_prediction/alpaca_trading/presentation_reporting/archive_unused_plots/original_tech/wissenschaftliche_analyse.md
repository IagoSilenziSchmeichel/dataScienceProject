# Wissenschaftliche Analyse - Original Tech

Benchmark: QQQ  |  Forschungsfrage: Wie funktioniert das Modell auf dem urspruenglichen Tech-Universum?

Automatisch erzeugt von generate_scientific_analysis.py aus bereits vorhandenen Pipeline-/Log-Dateien. Keine Modelle wurden neu trainiert, keine Handelslogik veraendert. Ausgeschlossen wurden Dateien, die nicht vom Standard-Orchestrator (`run_universe_lstm_outperformance.py`) regeneriert werden (Feature-Ablation, Robustness, Hyperparameter-Tuning, Alpaca-Export) - diese koennen Ueberbleibsel eines frueheren, anderen Universums-Laufs sein und wuerden sonst falsch zugeordnet.

## 1. Datensatz

Verwendete Aktien (10): AAPL, MSFT, NVDA, AMD, GOOGL, META, AMZN, TSLA, INTC, ADBE
Benchmark: QQQ

Rohdaten-Zeitraum: 2019-01-02 bis 2026-07-10 (22680 Zeilen, inkl. QQQ/SPY als Marktdaten-Quelle)

Chronologischer Split (70/15/15, pro Aktie normalisiert nur auf Trainingsstatistiken):

| Split | Zeitraum | Zeilen |
| --- | --- | --- |
| Training | 2019-10-16 - 2024-06-28 | 11830 |
| Validation | 2024-07-01 - 2025-07-03 | 2530 |
| Test | 2025-07-07 - 2026-07-09 | 2540 |

## 2. Modellanalyse

Finales Modell: Outperformance-LSTM mit Top-K-Selektionsstrategie (siehe `experiments/exp_2_lstm/scripts/14_outperformance_lstm/outperformance_lstm.py`).
Zielvariable: `Outperform_QQQ_Target` - 1, wenn die Tagesrendite der Aktie am naechsten Handelstag hoeher ist als die Rendite von QQQ, sonst 0.
Anzahl Features: 30 (19 Basis-Features + 7 Market Features + 4 Relative-Strength-Features; siehe Abschnitt 6).

Architektur/Trainingsparameter (aus `experiments/exp_2_lstm/conf/params.yaml`):

| Parameter | Wert |
| --- | --- |
| Sequenzlaenge | 20 Handelstage |
| Hidden Size | 64 |
| LSTM-Layer | 1 |
| Dropout | 0.2 (inaktiv bei 1 Layer) |
| Epochen | 20 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Top-K-Werte | 1, 2, 3, 4, 5 |

Wichtigste Kennzahlen:

| Modell | Accuracy | Precision | Recall | F1 | Best Val-F1 |
| --- | --- | --- | --- | --- | --- |
| Standard-LSTM (Up/Down) | 0.521 | 0.523 | 0.864 | 0.651 | 0.646 (Epoche 1) |
| Outperformance-LSTM (vs. QQQ) | 0.523 | 0.502 | 0.588 | 0.541 | 0.534 |

## 3. Backtest

| Strategie | Rendite | Buy-and-Hold | Differenz | Zeitraum |
| --- | --- | --- | --- | --- |
| Standard-LSTM (Schwelle 0.50) | +39.9% | +54.3% | -14.4% | 2025-08-01 - 2026-07-08 (234 Tage) |
| Outperformance-LSTM Top-1 | +105.7% | +31.0% | +74.7% | 2025-08-01 - 2026-07-09 (235 Tage) |

Sharpe (Outperformance-LSTM, netto=brutto da keine Transaktionskosten modelliert): 2.17. Max Drawdown: -24.3%. Anzahl Trades (Top-K-Wechsel): 297.

Einordnung:

- Der Standard-LSTM zeigt eine deutlich hoehere Recall- als Precision-Rate (0.86 vs. 0.52), was auf eine Tendenz hindeutet, ueberwiegend 'Kaufen'-Signale zu generieren (Klassenungleichgewicht in Richtung der Mehrheitsklasse), statt trennscharf zwischen Auf- und Abwaertsbewegungen zu unterscheiden.
- Die Outperformance-LSTM-Strategie erreicht einen Sharpe von 2.17, was auf ein vergleichsweise gutes Rendite-Risiko-Verhaeltnis im Testzeitraum hindeutet - unabhaengig davon, ob die absolute Rendite die Benchmark schlaegt.
- Die Strategie schlaegt QQQ im Testzeitraum um +74.7% - ein Hinweis darauf, dass die gelernten Muster (Momentum/Relative Staerke gegenueber dem Benchmark) im Testzeitraum tatsaechlich prognostischen Wert hatten.

Staerken: Klare, reproduzierbare Methodik ohne Data-Snooping (chronologischer Split, Skalierung nur auf Trainingsdaten), Vergleich gegen zwei Baselines (Buy-and-Hold, Standard-LSTM) macht den Mehrwert der Outperformance-Formulierung sichtbar.
Schwaechen: Kein Transaktionskosten-Modell, relativ kurzer Testzeitraum, Klassifikationsmetrik (F1) und oekonomisches Ziel (Rendite) sind nicht identisch optimiert.

## 4. Top-K-Analyse

| Top-K | Strategie-Rendite | Buy-and-Hold | Differenz | Ø Positionen |
| --- | --- | --- | --- | --- |
| Top-1 | +105.7% | +55.7% | +49.9% | 1.0 |
| Top-2 | +96.4% | +55.7% | +40.6% | 2.0 |
| Top-3 | +67.0% | +55.7% | +11.3% | 3.0 |
| Top-4 | +60.5% | +55.7% | +4.7% | 4.0 |
| Top-5 | +68.2% | +55.7% | +12.5% | 5.0 |

Bestes Top-K: Top-1 (+105.7%). Schwaechstes Top-K: Top-4 (+60.5%). Spannweite zwischen bestem und schwaechstem K: 45.2%.

Wissenschaftliche Einordnung: Kleinere K-Werte (Top-1) konzentrieren das Portfolio auf die vom Modell am sichersten eingeschaetzte Aktie - das erhoeht die Varianz der Ergebnisse (ein einzelner falscher Pick wiegt schwer), kann aber auch die Rendite maximieren, wenn die Rangfolge des Modells verlaesslich ist. Groessere K-Werte (Top-4/Top-5) naehern sich einem gleichgewichteten Portfolio ueber das gesamte Universum an und damit tendenziell der Buy-and-Hold-Rendite - Diversifikation reduziert sowohl das Abwaerts- als auch das Aufwaertspotenzial gegenueber der reinen Modellauswahl.

## 5. Signalanalyse

Grundlage: 5 Signalzeitpunkte, 2005 Modell-Wahrscheinlichkeiten insgesamt (1 Vorhersage je Aktie und Signaltag).

Wahrscheinlichkeiten: Minimum 0.31, Maximum 0.51, Mittelwert 0.36, Median 0.34, Standardabweichung 0.05.

Knappste Entscheidung: AMD am 2026-07-07 mit p=0.491 (nahezu Zufallsniveau 0.50).
Eindeutigste Entscheidung / Ausreisser: ADBE am 2026-07-09 mit p=0.312 (groesster Abstand zu 0.50).

Je Aktie (sortiert nach Auswahlhaeufigkeit):

| Ticker | Anteil Top-K | Ø Probability | Ø Rang | Kauf | Verkauf | Ø Haltedauer |
| --- | --- | --- | --- | --- | --- | --- |
| AMD | 100% | 0.49 | 1.0 | 2 | 3 | 5.0T |
| INTC | 100% | 0.38 | 2.0 | 5 | 0 | 5.0T |
| AAPL | 60% | 0.36 | 3.4 | 3 | 0 | 3.0T |
| NVDA | 20% | 0.35 | 4.2 | 1 | 1 | 1.0T |
| TSLA | 20% | 0.35 | 4.6 | 1 | 1 | 1.0T |
| MSFT | 0% | 0.33 | 7.0 | 0 | 0 | n/a |
| GOOGL | 0% | 0.34 | 6.6 | 0 | 0 | n/a |
| META | 0% | 0.33 | 8.4 | 0 | 0 | n/a |
| AMZN | 0% | 0.33 | 8.4 | 0 | 0 | n/a |
| ADBE | 0% | 0.32 | 9.4 | 0 | 0 | n/a |

Top-1 am haeufigsten gewaehlt: AMD. Top-3 am haeufigsten gewaehlt: AMD, INTC, AAPL.
Konstant bevorzugt (100% der Signaltage): AMD, INTC.
Nie ausgewaehlt (0% der Signaltage): MSFT, GOOGL, META, AMZN, ADBE.
Durchschnittliche Umschichtung pro Signaltag: 1.0 Positionswechsel.

Warum genau diese Aktien: das Modell waehlt je Signaltag die K Aktien mit der hoechsten vorhergesagten Wahrscheinlichkeit, den Benchmark am naechsten Tag zu schlagen (Rang 1..K). Aktien mit konstant hoher durchschnittlicher Probability und niedrigem Rang wurden demnach ueber den Beobachtungszeitraum wiederholt als relativ staerker eingeschaetzt; Aktien mit Anteil 0% wurden in keinem beobachteten Signalzeitpunkt unter die besten K eingestuft.

## 6. Featureanalyse

Das Modell verwendet 30 Features je Aktie und Tag: 19 technische Basis-Features (siehe `experiments/exp_2_lstm/conf/lstm_features.txt`), 7 Market Features und 4 Relative-Strength-Features (siehe `MARKET_FEATURES`/`RELATIVE_FEATURES` in `outperformance_lstm.py` - identischer Code fuer alle vier Universen, nur der Benchmark-Ticker in Market-/Relative-Feature-Namen unterscheidet sich nicht, da beide QQQ- und SPY-Varianten immer mitgefuehrt werden).

- Momentum (8): Lag_1_Return, Lag_3_Return, Lag_7_Return, Momentum_5, Momentum_10, Momentum_20, RollingMean_7, RollingMean_30
- Trend (5): MACD, MACD_Signal, RSI_14, Distance_to_MA_200, Price_Position_20
- Volatilitaet (3): RollingVolatility_7, RollingVolatility_30, High_Low_Range
- Volumen (2): Volume_Change, Volume_Ratio_20
- Sonstige (Basis) (1): Daily_Return
- Market Features (Benchmark-Kontext) (7): QQQ_Return, SPY_Return, VIX_Change, QQQ_Momentum_20, SPY_Momentum_20, QQQ_Distance_to_MA200, SPY_Distance_to_MA200
- Relative Strength / Benchmark-Abweichung (4): Relative_Return_QQQ, Relative_Return_SPY, Relative_Momentum_20_QQQ, Relative_Momentum_20_SPY

Momentum-Features (Lag-Renditen, RollingMean, Momentum_5/10/20) erfassen kurz- bis mittelfristige Preistrends - ihre Praemisse ist, dass jüngste relative Staerke sich kurzfristig fortsetzt. Relative-Strength-/Benchmark-Abweichungsfeatures (Relative_Return_*, Relative_Momentum_20_*) sind fuer die Outperformance-Formulierung besonders zentral, da die Zielvariable selbst relativ zum Benchmark definiert ist - eine Aktie kann fallen und trotzdem 'Outperform' sein, wenn der Benchmark staerker faellt. Volatilitaets-Features (RollingVolatility_7/30, High_Low_Range) liefern dem Modell ein Risikoregime-Signal; Trend-Features (MACD, RSI_14, Distance_to_MA_200) fassen laengerfristige technische Trendlage zusammen. Market Features (QQQ_Return, SPY_Return, VIX_Change, Distance_to_MA200 der Indizes) geben dem Modell Kontext ueber die generelle Marktlage, unabhaengig von der Einzelaktie.

Fuer dieses Universum stehen die 30 Feature-Werte fuer 10 Aktien ueber 16900 Beobachtungen (Training+Validation+Test) zur Verfuegung; eine quantitative Feature-Wichtigkeits-Analyse (z. B. Permutation Importance) ist mit den aktuell archivierten Dateien nicht durchgefuehrt worden und wuerde einen zusaetzlichen, nicht-trainierenden Auswertungsschritt erfordern.

## 7. Hourly Paper Trading

STATUS: MISSING

1 eindeutige(r) 1Hour-Zeitstempel gefunden in: paper_orders_legacy_20260706113110.csv, paper_performance_legacy_20260706113110.csv, paper_positions_legacy_20260706113110.csv, paper_signals_legacy_20260706112826.csv, paper_signals_legacy_20260706135630.csv, paper_signals_legacy_20260706155354.csv, performance_history_legacy_20260706155354.csv, trading_snapshots_legacy_20260706155354.csv. Diese Zeilen stammen aus dem einmaligen Alpaca-Dry-Run vor der Umstellung auf den Daily-Scheduler; die gleichen Logdateien wurden danach mit 1Day-Zeilen weitergeschrieben (Schema-Wechsel, alte Zeilen als Legacy-Datei archiviert). Keine belastbare universenspezifische Performance-Attribution ueber die Zeit verfuegbar.

Es liegen 1 eindeutige 1Hour-Zeitstempel vor - das stammt aus einem einmaligen Alpaca-Dry-Run vor Umstellung auf den taeglichen Scheduler und reicht nicht fuer eine Zeitreihen-Auswertung (Orders/Holdings/Rebalancing/Signalentwicklung/Performance ueber die Zeit).

## 8. Daily Paper Trading

Isolierte Tages-Simulation, 5 Handelstage (2026-07-06 - 2026-07-10).

Portfolio-Rendite: -3.0%, Benchmark-Rendite: +0.4%, Differenz: -3.4%, Max Drawdown: -6.7%.

Orders: 17 gesamt (Kauf 12, Verkauf 5), gehandeltes Volumen $579,490.

Vorlaeufig: der Beobachtungszeitraum ist sehr kurz (wenige Handelstage), daher keine statistisch belastbare Aussage ueber die tatsaechliche Live-Performance moeglich - dient primaer als Plausibilitaetscheck, dass die Paper-Trading-Pipeline korrekt Orders platziert und Positionen bewertet.

## 9. Visualisierungen

Alle Diagramme verwenden dieselbe Konfiguration aus `presentation_reporting/reporting_config.py` (Farben, Schriftgroessen, Figurgroesse 12.8x7.2, 220 DPI) - identisch fuer alle vier Universen, sodass die Plots direkt nebeneinander vergleichbar sind.

Standardplots (`output/original_tech/`): 01_backtest_vs_benchmark.png, 02_top_k_results.png, 03_hourly_alpaca_vs_benchmark.png, 04_daily_alpaca_vs_benchmark.png, 05_signal_selection_analysis.png, 06_signal_probability_distribution.png (optional).
Zusaetzliche Rohdaten-Plots aus der Pipeline selbst (`universe_results/original_tech/exp_2_lstm/plots/`): Trainingsverlauf, Konfusionsmatrix, kumulativer Backtest, Top-K-Vergleich, Outperformance-Metriken.

## 10. Wissenschaftliches Fazit

Wichtigste Erkenntnisse:

- Die Top-1-Outperformance-LSTM-Strategie erzielt +105.7% gegenueber +31.0% fuer QQQ (+74.7% Differenz, Sharpe 2.17, Max Drawdown -24.3%).
- Das Modell bevorzugte AMD am haeufigsten als Top-1-Wahl; die durchschnittliche Modell-Wahrscheinlichkeit lag bei 0.36 (0.50 entspraeche Zufallsniveau).
- Die kurze Daily-Alpaca-Simulation (5 Tage) zeigt -3.4% Differenz zum Benchmark - bei dieser Stichprobengroesse nur als Plausibilitaetssignal zu werten, nicht als belastbares Live-Ergebnis.

Staerken: konsistente, daten-getriebene Methodik ohne Data-Snooping; direkter Vergleich gegen Buy-and-Hold und Standard-LSTM macht den Mehrwert (oder das Fehlen des Mehrwerts) der Outperformance-Formulierung transparent; Top-K-Analyse zeigt Sensitivitaet der Strategie gegenueber der Portfoliokonzentration.
Schwaechen: kurzer Testzeitraum, keine Transaktionskosten modelliert, Klassifikationsziel (F1) und oekonomisches Ziel (Rendite) nicht deckungsgleich optimiert, Hourly-Alpaca-Daten aktuell nicht auswertbar.
Besonderheit von Original Tech: Wie funktioniert das Modell auf dem urspruenglichen Tech-Universum?
