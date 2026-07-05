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
TOP_K=1
ORDER_TYPE=market
TIME_IN_FORCE=day
```

`.env` ist in `.gitignore` eingetragen und darf nicht committed werden.

## Befehle

Finales Outperformance-LSTM fuer Alpaca exportieren:

```bash
python tech_stock_prediction/experiments/exp_2_lstm/scripts/17_export_alpaca_model/export_outperformance_alpaca_model.py
```

Nur Signale erzeugen, ohne Alpaca-Verbindung:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --universe original_tech --signals-only
```

Signale fuer alle Universen:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --signals-only
```

Dry Run fuer ein Universum:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --universe original_tech --dry-run
```

Dry Run fuer alle Universen:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --dry-run
```

Paper Orders fuer ein Universum:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --universe original_tech --execute
```

Paper Orders fuer alle Universen:

```bash
python tech_stock_prediction/alpaca_trading/run_paper_trading.py --all-universes --execute
```

## Modi

`--signals-only`

- benoetigt keine Alpaca Keys
- erzeugt nur Modell-Signale
- schreibt `logs/paper_signals.csv`
- sendet keine Orders

`--dry-run`

- prueft den kompletten Ablauf
- benoetigt keine Alpaca Keys
- sendet keine Orders
- zeigt nur, welche Orders geplant waeren

`--execute`

- sendet Paper-Trading-Orders an Alpaca
- funktioniert nur gegen `https://paper-api.alpaca.markets`
- gibt vorher eine klare Warnung aus

## Logs

Die Engine schreibt CSV-Logs:

- `logs/paper_signals.csv`
- `logs/paper_orders.csv`
- `logs/paper_positions.csv`
- `logs/paper_performance.csv`

Diese Logdateien sind generiert und werden nicht committed.

## Modell-Dateien

Der Signal-Generator erwartet ein gespeichertes finales Outperformance-LSTM:

```text
experiments/exp_2_lstm/models/outperformance_lstm_model.pth
experiments/exp_2_lstm/models/outperformance_lstm_scaler.pkl
experiments/exp_2_lstm/conf/outperformance_alpaca_features.txt
```

Falls diese Dateien fehlen, bricht der Signal-Generator bewusst mit einer
klaren Fehlermeldung ab. So wird verhindert, dass versehentlich das normale
Standard-LSTM fuer Paper Trading genutzt wird.

Die finale Alpaca-Variante ist:

- Modell: Outperformance-LSTM
- Featuregruppe: Technical + Relative Strength
- Standard-Strategie: Top-K mit `TOP_K=1`
- Benchmark: `QQQ` fuer Tech-Universen, `SPY` fuer `defensive_non_tech`
