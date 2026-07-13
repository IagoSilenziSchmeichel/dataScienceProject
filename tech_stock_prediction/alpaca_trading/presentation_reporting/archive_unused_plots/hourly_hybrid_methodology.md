# Hourly Hybrid Methodology

Diese Datei dokumentiert die Hybrid-Visualisierung fuer das stundenbasierte Paper Trading.

Wichtig: Simulierte Daten sind keine echten Alpaca-Paper-Trading-Ergebnisse.

Random Seed: 42

## Methode

- Echte Beobachtungen werden nur verwendet, wenn `timeframe == 1Hour` eindeutig gesetzt ist.
- Legacy- und aktuelle CSV-Schemas werden auf ein gemeinsames Format normalisiert.
- Daily-Zeilen werden nicht als Hourly-Zeilen interpretiert.
- Modell- und Benchmark-Reihe werden gemeinsam auf Startwert 100 normiert.
- Wenn weniger als 30 echte Hourly-Punkte vorhanden sind, wird eine reproduzierbare Simulation ergaenzt.
- Simulation nutzt zuerst reale Hourly-Renditen; falls zu wenige vorhanden sind, die Hourly-Backtest-Renditeverteilung fuer denselben Benchmark.
- Wenn keine belastbare Simulationsbasis vorhanden ist, wird nicht simuliert.

## Universen

### original_tech

- Benchmark: QQQ
- Reale Beobachtungen: 0
- Simulierte Beobachtungen: 0
- Verwendete reale Dateien: keine
- Simulationsmethode: not_possible
- Seed: 
- Hinweis: Keine realen Hourly-Paper-Trading-Punkte mit Portfolio- und Benchmark-Wert gefunden. Simulation nicht moeglich: no_real_anchor_for_simulation.

### tech_no_nvda

- Benchmark: QQQ
- Reale Beobachtungen: 0
- Simulierte Beobachtungen: 0
- Verwendete reale Dateien: keine
- Simulationsmethode: not_possible
- Seed: 
- Hinweis: Keine realen Hourly-Paper-Trading-Punkte mit Portfolio- und Benchmark-Wert gefunden. Simulation nicht moeglich: no_real_anchor_for_simulation.

### new_tech

- Benchmark: QQQ
- Reale Beobachtungen: 0
- Simulierte Beobachtungen: 0
- Verwendete reale Dateien: keine
- Simulationsmethode: not_possible
- Seed: 
- Hinweis: Keine realen Hourly-Paper-Trading-Punkte mit Portfolio- und Benchmark-Wert gefunden. Simulation nicht moeglich: no_real_anchor_for_simulation.

### defensive_non_tech

- Benchmark: SPY
- Reale Beobachtungen: 1
- Simulierte Beobachtungen: 29
- Verwendete reale Dateien: paper_performance_legacy_20260706160330.csv
- Simulationsmethode: hourly_backtest_distribution_top_5
- Seed: 42
- Hinweis: keine

## Einschraenkungen

- Die echte Hourly-Historie ist unvollstaendig, weil fruehe Dry-Run-Logs spaeter durch Schemawechsel archiviert wurden.
- Die Hybrid-Kurve ist ein Szenario fuer den stundenbasierten Serverbetrieb, kein vollstaendiges Paper-Trading-Ergebnis.
- Die Simulation wird nicht auf ein positives Ergebnis optimiert.
- Fehlende Werte werden nicht als 0 interpretiert.
