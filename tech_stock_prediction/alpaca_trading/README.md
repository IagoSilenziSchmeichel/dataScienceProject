# Alpaca Paper Trading

Diese Struktur startet Phase 2 des Projekts. Das Ziel ist, die Signale des
finalen Outperformance-LSTM im Alpaca Paper Trading zu beobachten.

Wichtig:

- Nur Paper Trading
- Kein echtes Geld
- Keine Anlageberatung
- Modell ist experimentell
- Kein Shorting
- Kein Hebel
- Keine Optionen
- Pro Handelslauf genau ein Universum
- `--all-universes` nur fuer Signaltests, niemals fuer Orders

## Universen

Verfuegbare Universen:

- `original_tech`
- `tech_no_nvda`
- `new_tech`
- `defensive_non_tech`

Benchmarks:

- Tech-Universen: `QQQ`
- Defensive Nicht-Tech-Kontrollgruppe: `SPY`

## .env einrichten

API Keys werden nicht im Code gespeichert. Lege lokal eine `.env` im
Repository-Root an:

```text
APCA_API_KEY_ID=dein_key
APCA_API_SECRET_KEY=dein_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
DRY_RUN=true
TOP_K=5
TIMEFRAME=1Hour
ORDER_TYPE=market
TIME_IN_FORCE=day
```

Alternative Namen fuer die Keys werden auch unterstuetzt:

```text
ALPACA_API_KEY=dein_key
ALPACA_SECRET_KEY=dein_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

`.env` ist in `.gitignore` eingetragen und darf nicht committed werden.

## Timeframe

Die Alpaca-Pipeline ist jetzt auf das separat trainierte Hourly-Outperformance-
LSTM ausgerichtet.

- `1Hour`: stündliche Bars, Standard fuer Alpaca Paper Trading
- `1Day`: technisch weiterhin waehlbar, aber nicht der empfohlene Alpaca-Modus

Wichtig: Das Daily-Research-Modell wird von Alpaca nicht mehr geladen. Fuer
Paper Trading werden die Hourly-Artefakte aus `experiments/exp_2_lstm/hourly/`
erwartet.

## Befehle

Hourly-Outperformance-LSTM fuer Alpaca trainieren:

```bash
python tech_stock_prediction/run_hourly_lstm_pipeline.py
```

Nur Signale erzeugen, ohne Alpaca-Verbindung:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --universe original_tech --signals-only --top-k 5 --timeframe 1Hour
```

Signale fuer alle Universen:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --signals-only --top-k 5 --timeframe 1Hour
```

Wichtig: `--all-universes` ist nur fuer Forschung und Signaltests gedacht.
Sobald Dry-Run oder Execute genutzt wird, muss genau ein Universum mit
`--universe` angegeben werden.

Dry Run fuer ein Universum:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --universe original_tech --dry-run --top-k 5 --timeframe 1Hour
```

Paper Orders fuer ein Universum:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --universe original_tech --execute --top-k 5 --timeframe 1Hour
```

Wenn im gemeinsamen Paper-Trading-Account noch Positionen aus einem anderen
Universum liegen, stoppt `--execute` standardmaessig. Entweder zuerst die alten
Positionen schliessen oder das Risiko bewusst bestaetigen:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --universe original_tech --execute --top-k 5 --timeframe 1Hour --allow-existing-positions
```

Daily ist fuer die wissenschaftliche Hauptpipeline weiterhin ueber
`run_lstm_pipeline.py` moeglich. Alpaca verwendet trotzdem die Hourly-Artefakte:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --signals-only --top-k 5 --timeframe 1Day
```

## Modi

`--signals-only`

- benoetigt keine Alpaca Keys
- erzeugt nur Modell-Signale
- schreibt `logs/paper_signals.csv`
- sendet keine Orders

`--dry-run`

- prueft den kompletten Ablauf
- benoetigt Alpaca Paper-Trading-Keys
- sendet keine Orders
- zeigt nur, welche Orders geplant waeren
- nur mit `--universe`, nicht mit `--all-universes`

`--execute`

- sendet Paper-Trading-Orders an Alpaca
- funktioniert nur gegen `https://paper-api.alpaca.markets`
- gibt vorher eine klare Warnung aus
- nur mit `--universe`, nicht mit `--all-universes`
- stoppt bei fremden bestehenden Positionen, ausser `--allow-existing-positions` wird bewusst gesetzt

## Logs

Die Engine schreibt CSV-Logs:

- `logs/paper_signals.csv`
- `logs/paper_orders.csv`
- `logs/paper_positions.csv`
- `logs/paper_performance.csv`

Diese Logdateien sind generiert und werden nicht committed.

Alle Logs enthalten eindeutige Testinformationen:

- `universe`
- `test_universe`
- `test_run_id`
- `timeframe`
- `bar_timestamp`
- `test_period_start`
- `test_period_end`

## Modell-Dateien

Der Signal-Generator erwartet das gespeicherte Hourly-Outperformance-LSTM:

```text
experiments/exp_2_lstm/hourly/models/hourly_outperformance_lstm_model.pth
experiments/exp_2_lstm/hourly/models/hourly_outperformance_scaler.pkl
experiments/exp_2_lstm/hourly/models/hourly_outperformance_metadata.json
experiments/exp_2_lstm/hourly/conf/hourly_features.txt
```

Falls diese Dateien fehlen, bricht der Signal-Generator bewusst mit einer
klaren Fehlermeldung ab. So wird verhindert, dass versehentlich das normale
Standard-LSTM fuer Paper Trading genutzt wird.

Die finale Alpaca-Variante ist:

- Modell: Hourly-Outperformance-LSTM
- Featuregruppe: Technical + Market + Relative Strength
- Standard-Strategie: Top-K mit `TOP_K=5`
- Benchmark: `QQQ` fuer Tech-Universen, `SPY` fuer `defensive_non_tech`
