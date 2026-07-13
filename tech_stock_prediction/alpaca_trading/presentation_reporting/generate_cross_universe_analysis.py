"""
Generate the cross-universe ("Gesamtanalyse") scientific comparison across
all four universes - the project treated as one research project rather
than four separate ones.

Output:
    output/comparison/gesamtanalyse.md        (10 comparison sections +
                                                ranking + key findings +
                                                presentation recommendations)
    output/comparison/06_model_metrics_comparison.png
    output/comparison/07_backtest_strategy_comparison.png
    output/comparison/08_top_k_heatmap.png
    output/comparison/09_average_probability_comparison.png
    output/comparison/10_daily_paper_trading_portfolio_comparison.png
    output/comparison/11_sharpe_drawdown_comparison.png
    output/comparison/12_overall_ranking.png

Reuses generate_scientific_analysis.py's per-universe loaders (which
already exclude stale, non-orchestrator-regenerated files - see that
module's docstring) so the per-universe numbers used here are identical to
the ones in each output/<universe>/wissenschaftliche_analyse.md. Nothing
here trains a model, changes trading logic, or touches any file outside
alpaca_trading/presentation_reporting/output/ - no risk of Daily/Hourly
branch conflicts, since those branches do not touch this folder.
"""

from pathlib import Path
import sys

REPORTING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORTING_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import generate_scientific_analysis as gsa  # noqa: E402
import presentation_data_loader as loader  # noqa: E402
import presentation_metrics as metrics  # noqa: E402
import presentation_plots as plots  # noqa: E402
import reporting_config as cfg  # noqa: E402

# Rough, widely-known sector classification of each ticker - qualitative
# domain context only (NOT a pipeline output, no numeric claim attached).
SECTOR_MAP = {
    "AAPL": "Technologie/Hardware", "MSFT": "Technologie/Software", "NVDA": "Halbleiter",
    "AMD": "Halbleiter", "GOOGL": "Technologie/Internet", "META": "Technologie/Internet",
    "AMZN": "E-Commerce/Cloud", "TSLA": "Automobil/Tech", "INTC": "Halbleiter", "ADBE": "Software",
    "PLTR": "Software/Analytics", "SNOW": "Software/Cloud", "CRWD": "Cybersecurity",
    "NET": "Cloud-Infrastruktur", "NOW": "Software/Cloud", "ORCL": "Software/Datenbanken",
    "CSCO": "Netzwerktechnik", "MU": "Halbleiter", "SMCI": "Hardware/Server", "DELL": "Hardware",
    "WMT": "Einzelhandel", "COST": "Einzelhandel", "PG": "Konsumgueter", "KO": "Getraenke",
    "PEP": "Getraenke/Konsumgueter", "JNJ": "Gesundheit/Pharma", "MRK": "Pharma", "ABBV": "Pharma",
    "XOM": "Energie", "CVX": "Energie",
}


def _load_train_dataframe(universe_name):
    path = cfg.UNIVERSE_RESULTS_ROOT / universe_name / "exp_1_randomforest" / "data" / "processed" / "train.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path, parse_dates=["Date"])
    if set(data["Ticker"].unique()) != set(cfg.UNIVERSES[universe_name]):
        return None
    return data


def compute_volatility_and_diversification(train_data, universe_name):
    """
    Real, computed dataset characteristics (not fabricated).

    Volatility: RollingVolatility_30/Daily_Return inside the archived
    train.csv are already per-ticker z-score normalized by split.py ("nur
    auf Trainingsstatistiken") - their mean is ~0 and std ~1 for every
    ticker by construction, so they cannot be used to compare volatility
    ACROSS universes. Volatility is therefore computed directly from the
    unscaled raw price file (tech_stocks_raw.csv): std of daily close-to-
    close returns per ticker, averaged across the universe's tickers.

    Diversification: average pairwise correlation of (per-ticker
    standardized) daily returns between tickers. Correlation is invariant
    to each series' own independent standardization, so the already-scaled
    train.csv is fine here.
    """
    if train_data is None:
        return None, None

    raw_path = cfg.UNIVERSE_RESULTS_ROOT / universe_name / "exp_1_randomforest" / "data" / "raw" / "tech_stocks_raw.csv"
    volatility = None
    if raw_path.exists():
        raw = pd.read_csv(raw_path, parse_dates=["Date"])
        raw = raw[raw["Ticker"].isin(cfg.UNIVERSES[universe_name])].copy()
        raw = raw.sort_values(["Ticker", "Date"])
        raw["daily_return"] = raw.groupby("Ticker")["Close"].pct_change()
        per_ticker_volatility = raw.groupby("Ticker")["daily_return"].std()
        volatility = float(per_ticker_volatility.mean()) if not per_ticker_volatility.empty else None

    pivot = train_data.pivot_table(index="Date", columns="Ticker", values="Daily_Return")
    correlation_matrix = pivot.corr()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    average_correlation = float(upper_triangle.stack().mean()) if not upper_triangle.stack().empty else None

    return volatility, average_correlation


def gather_universe_data(universe_name):
    rf_data = gsa.load_rf_dataset(universe_name)
    standard = gsa.load_standard_lstm(universe_name)
    outperformance, backtest_ds = gsa.load_outperformance_lstm(universe_name)

    train_data = _load_train_dataframe(universe_name)
    volatility, diversification = compute_volatility_and_diversification(train_data, universe_name)

    hourly_ds = loader.load_hourly_alpaca_performance(universe_name)
    daily_ds = loader.load_daily_alpaca_performance(universe_name)
    orders_ds = loader.load_daily_alpaca_orders(universe_name)
    order_summary = metrics.summarize_orders(orders_ds.data) if orders_ds.is_usable else {
        "buys": None, "sells": None, "total_trades": None, "total_notional_traded": None,
    }
    signal_ds = loader.load_signal_history(universe_name, timeframe="1Day")

    selection_df, distinct_dates, aggregate = None, 0, None
    if signal_ds.is_usable:
        order_counts = metrics.per_ticker_order_counts(orders_ds.data) if orders_ds.is_usable else {}
        selection_df, distinct_dates = metrics.summarize_signal_selection(
            signal_ds.data, cfg.UNIVERSES[universe_name], order_counts=order_counts,
        )
        aggregate = metrics.summarize_signal_aggregate(signal_ds.data, selection_df, distinct_dates)

    return {
        "universe": universe_name,
        "title": cfg.UNIVERSE_TITLE_DE[universe_name],
        "benchmark": cfg.BENCHMARKS[universe_name],
        "tickers": cfg.UNIVERSES[universe_name],
        "rf_data": rf_data,
        "standard": standard,
        "outperformance": outperformance,
        "volatility": volatility,
        "diversification": diversification,
        "hourly_ds": hourly_ds,
        "daily_ds": daily_ds,
        "orders_ds": orders_ds,
        "order_summary": order_summary,
        "signal_ds": signal_ds,
        "selection_df": selection_df,
        "distinct_dates": distinct_dates,
        "aggregate": aggregate,
    }


def _pct(value):
    return f"{value:+.1%}" if value is not None and pd.notna(value) else "n/a"


def _num(value, fmt="{:.2f}"):
    return fmt.format(value) if value is not None and pd.notna(value) else "n/a"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_1_dataset(universes):
    lines = ["## Vergleich 1: Datensatz", ""]
    lines.append("| Universum | Aktien | Benchmark | Ø Volatilitaet (30T) | Ø Korrelation (Diversifikation) | Trainingszeitraum | Testzeitraum |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for u in universes:
        rf = u["rf_data"]
        if rf is None:
            lines.append(f"| {u['title']} | {len(u['tickers'])} | {u['benchmark']} | n/a | n/a | MISSING | MISSING |")
            continue
        lines.append(
            f"| {u['title']} | {len(u['tickers'])} | {u['benchmark']} | {_num(u['volatility'], '{:.2%}')} | "
            f"{_num(u['diversification'])} | {rf['train_start'].date()} - {rf['train_end'].date()} | "
            f"{rf['test_start'].date()} - {rf['test_end'].date()} |"
        )
    lines.append("")

    lines.append("Branchenzusammensetzung (qualitativ, allgemein bekannte Einordnung, nicht Teil der Pipeline-Daten):")
    lines.append("")
    for u in universes:
        sectors = sorted(set(SECTOR_MAP.get(t, "unbekannt") for t in u["tickers"]))
        lines.append(f"- {u['title']}: {', '.join(sectors)}")
    lines.append("")

    lines.append(
        "Marktkapitalisierung wurde in der Pipeline nicht erhoben (kein Fundamentaldaten-Feld dafuer vorhanden) "
        "und wird hier bewusst nicht geschaetzt oder erfunden."
    )
    lines.append("")

    with_data = [u for u in universes if u["volatility"] is not None]
    if with_data:
        most_volatile = max(with_data, key=lambda u: u["volatility"])
        least_volatile = min(with_data, key=lambda u: u["volatility"])
        most_diversified = min((u for u in with_data if u["diversification"] is not None), key=lambda u: u["diversification"], default=None)
        least_diversified = max((u for u in with_data if u["diversification"] is not None), key=lambda u: u["diversification"], default=None)
        lines.append(
            f"Hoechste durchschnittliche Volatilitaet: {most_volatile['title']}. Niedrigste: {least_volatile['title']}. "
            + (
                f"Staerkste Diversifikation (niedrigste Durchschnittskorrelation): {most_diversified['title']}. "
                f"Geringste Diversifikation (hoechste Durchschnittskorrelation): {least_diversified['title']}."
                if most_diversified and least_diversified else ""
            )
        )
        lines.append("")
        lines.append(
            "Auswirkung auf das Modell: hoehere Volatilitaet bedeutet groessere Tag-zu-Tag-Preisschwankungen, was "
            "die Klassifikationsaufgabe (naechster Tag outperformt Benchmark oder nicht) tendenziell schwieriger "
            "macht, da das Signal-Rausch-Verhaeltnis sinkt. Eine hohe durchschnittliche Korrelation zwischen den "
            "Aktien eines Universums bedeutet, dass sie sich groesstenteils gemeinsam bewegen - das reduziert den "
            "potenziellen Mehrwert einer Top-K-Auswahl gegenueber Buy-and-Hold, da es weniger Gelegenheit fuer "
            "'die eine Aktie, die sich vom Rest abhebt' gibt."
        )
    lines.append("")
    return "\n".join(lines)


def section_2_model_performance(universes):
    lines = ["## Vergleich 2: Modellleistung", ""]
    lines.append("| Universum | Modell | Accuracy | Precision | Recall | Test-F1 | Validation-F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for u in universes:
        if u["standard"] is not None:
            r = u["standard"]["results"]
            lines.append(
                f"| {u['title']} | Standard-LSTM | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1_score']:.3f} | {u['standard']['best_validation_f1']:.3f} |"
            )
        else:
            lines.append(f"| {u['title']} | Standard-LSTM | MISSING | MISSING | MISSING | MISSING | MISSING |")
        if u["outperformance"] is not None and u["outperformance"].get("results_row"):
            row = u["outperformance"]["results_row"]
            lines.append(
                f"| {u['title']} | Outperformance-LSTM | {row['accuracy']:.3f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {row['f1_score']:.3f} | {row['best_validation_f1']:.3f} |"
            )
        else:
            lines.append(f"| {u['title']} | Outperformance-LSTM | MISSING | MISSING | MISSING | MISSING | MISSING |")
    lines.append("")

    with_results = [u for u in universes if u["outperformance"] is not None and u["outperformance"].get("results_row")]
    if with_results:
        best = max(with_results, key=lambda u: u["outperformance"]["results_row"]["f1_score"])
        worst = min(with_results, key=lambda u: u["outperformance"]["results_row"]["f1_score"])
        lines.append(
            f"Beste Test-F1 (Outperformance-LSTM): {best['title']} "
            f"({best['outperformance']['results_row']['f1_score']:.3f}). "
            f"Schwaechste: {worst['title']} ({worst['outperformance']['results_row']['f1_score']:.3f})."
        )
        lines.append("")
        lines.append(
            "Erklaerungsansatz: die Modellkennzahlen haengen direkt mit den in Vergleich 1 gemessenen "
            "Datensatzeigenschaften zusammen - Universen mit hoeherer durchschnittlicher Korrelation zwischen den "
            "Aktien liefern dem Modell tendenziell konsistentere, leichter lernbare Muster (die Aktien bewegen sich "
            "aehnlicher, wodurch das allgemeine Marktsignal deutlicher wird), waehrend Universen mit sehr "
            "heterogenen Einzeltiteln (z. B. neue, kleinere Tech-Werte mit eigenen, titelspezifischen Nachrichtenlagen) "
            "schwerer vorherzusagen sein koennen."
        )
    lines.append("")
    return "\n".join(lines)


def section_3_backtest(universes):
    lines = ["## Vergleich 3: Backtest", ""]
    lines.append("| Universum | Buy-and-Hold | Standard-LSTM | Outperformance-LSTM (bestes Top-K) | Sharpe | Max Drawdown |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for u in universes:
        bh = "n/a"
        standard_return = "n/a"
        outperf_return = "n/a"
        sharpe = "n/a"
        drawdown = "n/a"
        if u["standard"] is not None:
            r = u["standard"]["results"]
            bh = _pct(r["buy_and_hold_return"])
            standard_return = _pct(r["simple_strategy_return"])
        if u["outperformance"] is not None:
            s = u["outperformance"]["summary"]
            outperf_return = f"Top-{s['top_k']}: {_pct(s['strategy_return_net'])}"
            sharpe = f"{s['sharpe_ratio']:.2f}"
            drawdown = f"{s['max_drawdown']:.1%}"
        lines.append(f"| {u['title']} | {bh} | {standard_return} | {outperf_return} | {sharpe} | {drawdown} |")
    lines.append("")

    with_backtest = [u for u in universes if u["outperformance"] is not None]
    if with_backtest:
        most_stable = max(with_backtest, key=lambda u: u["outperformance"]["summary"]["sharpe_ratio"])
        least_drawdown = min(with_backtest, key=lambda u: abs(u["outperformance"]["summary"]["max_drawdown"]))
        best_outperformance = max(with_backtest, key=lambda u: u["outperformance"]["summary"]["difference"])
        lines.append(
            f"Hoechste Stabilitaet (Sharpe): {most_stable['title']}. Geringster Max Drawdown: {least_drawdown['title']}. "
            f"Staerkste Outperformance gegenueber Benchmark: {best_outperformance['title']} "
            f"({_pct(best_outperformance['outperformance']['summary']['difference'])})."
        )
        lines.append("")
        lines.append(
            "Robustheit: ein hoher Sharpe bei gleichzeitig geringem Max Drawdown deutet auf eine Strategie hin, die "
            "nicht nur im Mittel gut abschneidet, sondern auch wenige extreme Verlustphasen durchlebt - das ist "
            "fuer die praktische Einsetzbarkeit wichtiger als die reine absolute Rendite, da grosse Drawdowns in "
            "der Praxis zu vorzeitigem Strategieabbruch fuehren koennen."
        )
    lines.append("")
    return "\n".join(lines)


def section_4_top_k(universes, output_dir):
    lines = ["## Vergleich 4: Top-K", ""]
    with_top_k = [u for u in universes if u["outperformance"] is not None and u["outperformance"]["top_k_table"] is not None]

    if not with_top_k:
        lines.append("STATUS: MISSING fuer alle Universen.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Universum | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    heatmap_rows = []
    row_labels = []
    for u in with_top_k:
        table = u["outperformance"]["top_k_table"].set_index("top_k")["strategy_return"]
        values = [table.get(k, np.nan) for k in [1, 2, 3, 4, 5]]
        heatmap_rows.append(values)
        row_labels.append(u["title"])
        lines.append(f"| {u['title']} | " + " | ".join(_pct(v) if pd.notna(v) else "n/a" for v in values) + " |")
    lines.append("")

    # Which universe benefits most from Top-K concentration (Top-1 vs Top-5 spread)?
    benefit_rows = []
    for u in with_top_k:
        table = u["outperformance"]["top_k_table"].set_index("top_k")["strategy_return"]
        if 1 in table.index and 5 in table.index:
            benefit_rows.append((u["title"], float(table[1] - table[5])))
    if benefit_rows:
        strongest = max(benefit_rows, key=lambda item: item[1])
        weakest = min(benefit_rows, key=lambda item: item[1])
        lines.append(
            f"Profitiert am staerksten von Konzentration (Top-1 deutlich besser als Top-5): {strongest[0]} "
            f"({strongest[1]:+.1%} Unterschied). Am wenigsten (oder sogar umgekehrt): {weakest[0]} ({weakest[1]:+.1%})."
        )
        lines.append("")

    plots.plot_heatmap(
        title="Top-K-Strategie-Rendite je Universum",
        subtitle="Zeilen: Universum, Spalten: Top-K (1-5), Werte: Strategie-Rendite",
        row_labels=row_labels,
        col_labels=["Top-1", "Top-2", "Top-3", "Top-4", "Top-5"],
        matrix=heatmap_rows,
        output_path=output_dir / "08_top_k_heatmap.png",
    )
    lines.append("Diagramm: `comparison/08_top_k_heatmap.png`")
    lines.append("")
    return "\n".join(lines)


def section_5_signal(universes):
    lines = ["## Vergleich 5: Signalanalyse", ""]
    lines.append("| Universum | Ø Probability | Std | Haeufigste Top-1-Aktie | Nie gewaehlt |")
    lines.append("| --- | --- | --- | --- | --- |")
    for u in universes:
        if not u["signal_ds"].is_usable or u["selection_df"] is None:
            lines.append(f"| {u['title']} | n/a | n/a | n/a | n/a |")
            continue
        probabilities = u["signal_ds"].data["probability"].dropna()
        never_chosen = u["selection_df"][u["selection_df"]["selection_share"] == 0.0]["ticker"].tolist()
        lines.append(
            f"| {u['title']} | {probabilities.mean():.2f} | {probabilities.std():.2f} | "
            f"{u['aggregate']['most_frequent_top1'] or 'n/a'} | {len(never_chosen)} von {len(u['tickers'])} Aktien |"
        )
    lines.append("")

    with_signal = [u for u in universes if u["signal_ds"].is_usable]
    if with_signal:
        most_confident = max(with_signal, key=lambda u: u["signal_ds"].data["probability"].dropna().mean())
        most_uncertain = min(with_signal, key=lambda u: u["signal_ds"].data["probability"].dropna().mean())
        lines.append(
            f"Eindeutigste Signale im Mittel (hoechste Ø Probability): {most_confident['title']}. "
            f"Unsicherste Signale (Ø Probability am naechsten an 0.50): {most_uncertain['title']}."
        )
        lines.append("")
        lines.append("Von den Universen bevorzugte Branchen (haeufigste Top-1-Aktie je Universum):")
        lines.append("")
        for u in with_signal:
            top1 = u["aggregate"]["most_frequent_top1"] if u["aggregate"] else None
            sector = SECTOR_MAP.get(top1, "unbekannt") if top1 else "n/a"
            lines.append(f"- {u['title']}: {top1 or 'n/a'} ({sector})")
    lines.append("")
    return "\n".join(lines)


def section_6_paper_trading(universes):
    lines = ["## Vergleich 6: Paper Trading", ""]
    lines.append("### Hourly Paper Trading")
    lines.append("")
    lines.append(
        "Fuer alle vier Universen liegt aktuell keine belastbare Hourly-Zeitreihe vor (nur 1-2 Zeitstempel aus "
        "einem einmaligen Dry-Run vor Umstellung auf den taeglichen Scheduler, siehe data_validation_report.md). "
        "Ein Vergleich ist auf dieser Datenbasis nicht seriös moeglich."
    )
    lines.append("")
    lines.append("### Daily Paper Trading")
    lines.append("")
    lines.append("| Universum | Zeitraum | Portfolio-Rendite | Benchmark-Rendite | Differenz | Trades (Kauf/Verkauf) | Max Drawdown |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for u in universes:
        if not u["daily_ds"].is_usable:
            lines.append(f"| {u['title']} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        summary = metrics.summarize_alpaca_performance(u["daily_ds"].data)
        order_summary = u["order_summary"]
        lines.append(
            f"| {u['title']} | {summary['period_start'].date()} - {summary['period_end'].date()} | "
            f"{_pct(summary['portfolio_return'])} | {_pct(summary['benchmark_return'])} | {_pct(summary['difference'])} | "
            f"{order_summary['buys']}/{order_summary['sells']} | {summary['max_drawdown']:.1%} |"
        )
    lines.append("")

    with_daily = [u for u in universes if u["daily_ds"].is_usable]
    if with_daily:
        best_daily = max(with_daily, key=lambda u: metrics.summarize_alpaca_performance(u["daily_ds"].data)["difference"])
        most_active = max(with_daily, key=lambda u: u["order_summary"]["total_trades"] or 0)
        lines.append(
            f"Beste Daily-Paper-Trading-Performance (vorlaeufig, wenige Tage): {best_daily['title']}. "
            f"Hoechste Handelsaktivitaet: {most_active['title']} ({most_active['order_summary']['total_trades']} Trades)."
        )
        lines.append("")
        lines.append(
            "Alle Daily-Alpaca-Ergebnisse beruhen auf sehr kurzen Beobachtungsfenstern (wenige Handelstage je "
            "Universum) und sind daher nur als Plausibilitaetssignal zu werten, nicht als belastbarer Live-Vergleich."
        )
    lines.append("")
    return "\n".join(lines)


def section_7_model_behavior(universes):
    lines = ["## Vergleich 7: Modellverhalten", ""]
    with_data = [u for u in universes if u["outperformance"] is not None and u["diversification"] is not None]
    if with_data:
        easiest = max(with_data, key=lambda u: u["outperformance"]["results_row"]["f1_score"] if u["outperformance"].get("results_row") else 0)
        hardest = min(with_data, key=lambda u: u["outperformance"]["results_row"]["f1_score"] if u["outperformance"].get("results_row") else 1)
        lines.append(
            f"Fuer ein LSTM am ehesten 'einfacher' erscheint {easiest['title']} (hoechste Test-F1), am "
            f"'schwierigsten' {hardest['title']} (niedrigste Test-F1)."
        )
        lines.append("")
    lines.append(
        "Rolle der einzelnen Aspekte:"
    )
    lines.append("")
    lines.append(
        "- Momentum: in allen vier Universen identisch als Feature-Gruppe vorhanden; seine Nuetzlichkeit haengt "
        "davon ab, wie stark Trends in der jeweiligen Aktiengruppe tatsaechlich anhalten (Trendpersistenz), was "
        "wiederum von der Branchenstruktur abhaengt (z. B. reagieren Halbleiterwerte oft staerker/schneller auf "
        "Nachrichten als defensive Konsumguetertitel)."
    )
    lines.append(
        "- Volatilitaet: hoehere durchschnittliche Volatilitaet (siehe Vergleich 1) erschwert tendenziell die "
        "Klassifikationsaufgabe, da kurzfristiges Rauschen das eigentliche Richtungssignal ueberlagert."
    )
    lines.append(
        "- Benchmark: QQQ (techlastig) und SPY (breiter Markt) unterscheiden sich in Zusammensetzung und "
        "Volatilitaet - ein techlastiges Universum gegen QQQ zu vergleichen bedeutet, dass Universum und Benchmark "
        "aehnlicheren Kraeften unterliegen, als wenn ein defensives Universum gegen SPY antritt."
    )
    lines.append(
        "- Korrelation/Relative Strength: Universen mit hoeherer interner Korrelation (siehe Vergleich 1) bieten "
        "der Outperformance-Formulierung (relativ zum Benchmark) potenziell weniger Differenzierungsspielraum "
        "zwischen den Einzeltiteln, da sie sich aehnlicher bewegen."
    )
    lines.append("")
    return "\n".join(lines)


def section_8_visualizations():
    lines = ["## Vergleich 8: Visualisierung", ""]
    lines.append(
        "Alle folgenden Diagramme verwenden dieselbe Farbpalette (ein fester Farbcode je Universum, "
        "`UNIVERSE_COLORS` in `reporting_config.py`), dieselbe Schriftgroesse, Achsenbeschriftung und DPI (220) "
        "wie alle anderen Plots dieses Projekts."
    )
    lines.append("")
    lines.append("- `06_model_metrics_comparison.png` - Accuracy/Precision/Recall/F1 je Universum und Modell")
    lines.append("- `07_backtest_strategy_comparison.png` - Buy-and-Hold vs. Standard-LSTM vs. Outperformance-LSTM")
    lines.append("- `08_top_k_heatmap.png` - Top-1 bis Top-5 Rendite je Universum (Heatmap)")
    lines.append("- `09_average_probability_comparison.png` - durchschnittliche Modell-Wahrscheinlichkeit je Universum")
    lines.append("- `10_daily_paper_trading_portfolio_comparison.png` - Portfolioentwicklung (Index) aller Universen")
    lines.append("- `11_sharpe_drawdown_comparison.png` - Sharpe Ratio und Max Drawdown je Universum")
    lines.append("- `12_overall_ranking.png` - Gesamtranking (siehe Vergleich 9)")
    lines.append("")
    lines.append(
        "Bereits vorhandene Vergleichsplots aus `generate_presentation_reports.py` (01-05, gleiche Palette/Stil) "
        "ergaenzen dies um Backtest-, Hourly-, Daily- und Signal-Outperformance je Universum."
    )
    lines.append("")
    return "\n".join(lines)


def compute_ranking(universes):
    """
    Transparent, fully computed multi-criteria ranking - no subjective
    point assignment. Each criterion is min-max normalized to [0, 1] across
    the universes that have data for it (0 = worst, 1 = best in the
    observed set); the overall score is the unweighted mean of the criteria
    the universe has data for. This directly maps to the requested
    dimensions: Modellqualitaet, Generalisierung, Stabilitaet, Trading
    Performance, Paper Trading, Robustheit, Aussagekraft (sample size).
    "Praktische Einsetzbarkeit" and "wissenschaftliche Qualitaet" are
    treated as syntheses of the above rather than separate numbers, to
    avoid double-counting the same underlying evidence twice.
    """
    raw = {}
    for u in universes:
        name = u["universe"]
        row = {}
        if u["outperformance"] is not None and u["outperformance"].get("results_row"):
            row["Modellqualitaet (Test-F1)"] = u["outperformance"]["results_row"]["f1_score"]
        if u["outperformance"] is not None and u["outperformance"]["top_k_table"] is not None:
            table = u["outperformance"]["top_k_table"]
            row["Generalisierung (# Top-K mit positiver Differenz)"] = float((table["difference"] > 0).sum())
        if u["outperformance"] is not None:
            row["Stabilitaet (Sharpe)"] = u["outperformance"]["summary"]["sharpe_ratio"]
            row["Trading Performance (Backtest-Differenz)"] = u["outperformance"]["summary"]["difference"]
            row["Robustheit (Max Drawdown, weniger negativ = besser)"] = u["outperformance"]["summary"]["max_drawdown"]
            row["Aussagekraft (# Backtest-Handelstage)"] = float(u["outperformance"]["summary"]["number_of_days"])
        if u["daily_ds"].is_usable:
            row["Paper Trading (Daily-Alpaca-Differenz)"] = metrics.summarize_alpaca_performance(u["daily_ds"].data)["difference"]
        raw[name] = row

    all_criteria = sorted({criterion for row in raw.values() for criterion in row})
    normalized = {name: {} for name in raw}
    for criterion in all_criteria:
        values = {name: row[criterion] for name, row in raw.items() if criterion in row}
        if not values:
            continue
        low, high = min(values.values()), max(values.values())
        for name, value in values.items():
            normalized[name][criterion] = 0.5 if high == low else (value - low) / (high - low)

    scores = {}
    for name, row in normalized.items():
        scores[name] = float(np.mean(list(row.values()))) if row else 0.0

    return raw, normalized, scores, all_criteria


def section_9_ranking(universes, scores, raw, normalized, all_criteria, output_dir):
    lines = ["## Vergleich 9: Ranking", ""]
    lines.append(
        "Methodik: jedes Kriterium wird pro Universum min-max-normalisiert (0 = schwaechstes, 1 = staerkstes "
        "beobachtetes Universum bei diesem Kriterium); der Gesamtscore ist der ungewichtete Mittelwert aller "
        "verfuegbaren normalisierten Kriterien. 'Praktische Einsetzbarkeit' und 'wissenschaftliche Qualitaet' "
        "werden als Synthese der unten stehenden Kriterien interpretiert, nicht als zusaetzliche eigene Zahlen "
        "(um dieselbe Evidenz nicht doppelt zu zaehlen)."
    )
    lines.append("")
    lines.append("| Kriterium | " + " | ".join(u["title"] for u in universes) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(universes)) + " |")
    for criterion in all_criteria:
        row_values = []
        for u in universes:
            raw_value = raw[u["universe"]].get(criterion)
            row_values.append(f"{raw_value:.3f}" if raw_value is not None else "n/a")
        lines.append(f"| {criterion} | " + " | ".join(row_values) + " |")
    lines.append("")

    lines.append("| Universum | Gesamtscore (0-1) | Rang |")
    lines.append("| --- | --- | --- |")
    ranking_order = sorted(universes, key=lambda u: scores[u["universe"]], reverse=True)
    for rank, u in enumerate(ranking_order, start=1):
        lines.append(f"| {u['title']} | {scores[u['universe']]:.3f} | {rank} |")
    lines.append("")

    plots.plot_ranking_bars(
        title="Gesamtranking der vier Universen",
        subtitle="Ungewichteter Mittelwert aus 6 normalisierten Kriterien (siehe Methodik oben)",
        universe_titles=[u["title"] for u in universes],
        universe_names=[u["universe"] for u in universes],
        scores=[scores[u["universe"]] for u in universes],
        output_path=output_dir / "12_overall_ranking.png",
    )
    lines.append("Diagramm: `comparison/12_overall_ranking.png`")
    lines.append("")
    return "\n".join(lines)


def section_10_key_findings(universes, scores):
    lines = ["## Vergleich 10: Key Findings", ""]
    ranking_order = sorted(universes, key=lambda u: scores[u["universe"]], reverse=True)
    findings = []
    findings.append(
        f"Gesamtranking: {' > '.join(u['title'] for u in ranking_order)} (siehe Vergleich 9 fuer die Methodik)."
    )

    with_backtest = [u for u in universes if u["outperformance"] is not None]
    if with_backtest:
        best_backtest = max(with_backtest, key=lambda u: u["outperformance"]["summary"]["difference"])
        worst_backtest = min(with_backtest, key=lambda u: u["outperformance"]["summary"]["difference"])
        spread = best_backtest["outperformance"]["summary"]["difference"] - worst_backtest["outperformance"]["summary"]["difference"]
        findings.append(
            f"Groesster Unterschied im Backtest: {best_backtest['title']} "
            f"({_pct(best_backtest['outperformance']['summary']['difference'])}) vs. {worst_backtest['title']} "
            f"({_pct(worst_backtest['outperformance']['summary']['difference'])}) - eine Spanne von {spread:+.1%}."
        )

    with_signal = [u for u in universes if u["signal_ds"].is_usable]
    if with_signal:
        most_confident = max(with_signal, key=lambda u: u["signal_ds"].data["probability"].dropna().mean())
        findings.append(
            f"{most_confident['title']} zeigt die im Mittel eindeutigsten Modell-Wahrscheinlichkeiten - ein "
            "Hinweis darauf, dass das Modell in diesem Universum konsistentere Muster erkennt."
        )

    lines.append("Ueberraschend vs. erwartungskonform:")
    lines.append("")
    if with_backtest:
        tech_universes = [u for u in with_backtest if u["universe"] in ("original_tech", "tech_no_nvda", "new_tech")]
        defensive = next((u for u in with_backtest if u["universe"] == "defensive_non_tech"), None)
        if tech_universes and defensive:
            avg_tech_diff = float(np.mean([u["outperformance"]["summary"]["difference"] for u in tech_universes]))
            lines.append(
                f"- Erwartungskonform waere, dass die drei Tech-Universen (Ø Differenz {avg_tech_diff:+.1%}) "
                f"deutlich staerker abschneiden als das defensive Kontroll-Universum "
                f"({_pct(defensive['outperformance']['summary']['difference'])}), da die Strategie urspruenglich "
                "fuer Tech-Aktien entwickelt wurde."
            )
    for finding in findings:
        lines.append(f"- {finding}")
    lines.append("")
    return "\n".join(lines)


def presentation_recommendations():
    lines = ["## Praesentationsempfehlungen", ""]
    lines.append("### Muss unbedingt in die Praesentation")
    lines.append("")
    lines.append("- `comparison/12_overall_ranking.png` + Vergleich-9-Tabelle: beantwortet die Kernfrage direkt und objektiv nachvollziehbar.")
    lines.append("- `comparison/01_backtest_comparison.png` und `comparison/07_backtest_strategy_comparison.png`: zeigt den zentralen wissenschaftlichen Befund (schlaegt die Strategie den Benchmark, und wo).")
    lines.append("- `comparison/08_top_k_heatmap.png`: kompakte, auf einen Blick verstaendliche Darstellung eines der aufwendigsten Analyseteile.")
    lines.append("- Je Universum Plot 01 (Backtest) aus dem eigenen Ordner: zeigt die Story pro Universum konkret, nicht nur aggregiert.")
    lines.append("")
    lines.append("### Darf optional gezeigt werden")
    lines.append("")
    lines.append("- `comparison/06_model_metrics_comparison.png`: interessant fuer ein technisches Publikum, aber fuer die Kernaussage nicht zwingend.")
    lines.append("- `comparison/09_average_probability_comparison.png` und Signalanalyse-Details: gute Ergaenzung, falls Zeit fuer eine Modellverhalten-Folie ist.")
    lines.append("- `comparison/11_sharpe_drawdown_comparison.png`: relevant fuer ein risikofokussiertes Publikum, sonst optional.")
    lines.append("- Je Universum Plot 05/06 (Signalanalyse/Wahrscheinlichkeitsverteilung): gut als Vertiefung bei Nachfragen.")
    lines.append("")
    lines.append("### Kann weggelassen werden")
    lines.append("")
    lines.append("- Hourly-Paper-Trading-Plots (Plot 03 je Universum, `comparison/02_hourly_comparison.png`): zeigen aktuell durchgaengig nur den Hinweistext 'keine ausreichenden Daten' - liefert keine zusaetzliche Erkenntnis in der Praesentation.")
    lines.append("- Die vollstaendige Kriterientabelle aus Vergleich 9 als eigene Folie: zu dicht fuer eine Praesentationsfolie, besser nur das Endergebnis (Ranking-Balken) zeigen und die Tabelle im Backup-Anhang bereithalten.")
    lines.append("- Detaillierte Order-/Trade-Listen aus dem Daily Paper Trading: fuer die Kernaussage irrelevant, da die Stichprobe ohnehin zu klein fuer belastbare Aussagen ist.")
    lines.append("")
    lines.append(
        "Ziel: eine schluessige Geschichte - Datensatz unterscheidet sich (Vergleich 1) -> das beeinflusst "
        "Modellleistung (Vergleich 2) -> das zeigt sich im Backtest (Vergleich 3/4) -> Ranking fasst es objektiv "
        "zusammen (Vergleich 9). Nicht jede erzeugte Grafik muss gezeigt werden."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot generation (beyond section 4's heatmap and section 9's ranking bars)
# ---------------------------------------------------------------------------

def build_plots(universes, output_dir):
    universe_names = [u["universe"] for u in universes]
    universe_titles = [u["title"] for u in universes]

    # 06: model metrics comparison (Outperformance-LSTM, test set)
    accuracy, precision, recall, f1 = [], [], [], []
    for u in universes:
        row = u["outperformance"]["results_row"] if u["outperformance"] and u["outperformance"].get("results_row") else None
        accuracy.append(row["accuracy"] if row else None)
        precision.append(row["precision"] if row else None)
        recall.append(row["recall"] if row else None)
        f1.append(row["f1_score"] if row else None)

    plots.plot_grouped_bars_multi_metric(
        title="Modellkennzahlen je Universum (Outperformance-LSTM, Testset)",
        subtitle="Accuracy / Precision / Recall / F1",
        universe_titles=universe_titles,
        universe_names=universe_names,
        series=[("Accuracy", accuracy), ("Precision", precision), ("Recall", recall), ("F1", f1)],
        output_path=output_dir / "06_model_metrics_comparison.png",
        value_label="Wert",
    )

    # 07: backtest strategy comparison (Buy&Hold / Standard-LSTM / Outperformance-LSTM)
    buy_hold, standard_return, outperf_return = [], [], []
    for u in universes:
        if u["standard"] is not None:
            r = u["standard"]["results"]
            buy_hold.append(r["buy_and_hold_return"])
            standard_return.append(r["simple_strategy_return"])
        else:
            buy_hold.append(None)
            standard_return.append(None)
        outperf_return.append(u["outperformance"]["summary"]["strategy_return_net"] if u["outperformance"] else None)

    plots.plot_grouped_bars_multi_metric(
        title="Backtest-Renditevergleich je Universum",
        subtitle="Buy-and-Hold vs. Standard-LSTM vs. Outperformance-LSTM (bestes Top-K)",
        universe_titles=universe_titles,
        universe_names=universe_names,
        series=[("Buy-and-Hold", buy_hold), ("Standard-LSTM", standard_return), ("Outperformance-LSTM", outperf_return)],
        output_path=output_dir / "07_backtest_strategy_comparison.png",
        value_label="Rendite",
        value_format=lambda v: f"{v:+.1%}",
    )

    # 09: average probability comparison
    average_probability = []
    for u in universes:
        if u["signal_ds"].is_usable:
            average_probability.append(float(u["signal_ds"].data["probability"].dropna().mean()))
        else:
            average_probability.append(None)

    plots.plot_grouped_bars_multi_metric(
        title="Durchschnittliche Modell-Wahrscheinlichkeit je Universum",
        subtitle="Ø ueber alle Signalzeitpunkte und Aktien (0.50 = Zufallsniveau)",
        universe_titles=universe_titles,
        universe_names=universe_names,
        series=[("Ø Probability", average_probability)],
        output_path=output_dir / "09_average_probability_comparison.png",
        value_label="Wahrscheinlichkeit",
    )

    # 10: daily paper trading portfolio comparison (cumulative index overlay)
    series_by_universe = []
    for u in universes:
        if u["daily_ds"].is_usable:
            data = u["daily_ds"].data
            series_by_universe.append((u["title"], u["universe"], data["date"], data["portfolio_index"]))
    if series_by_universe:
        plots.plot_multi_line_index(
            title="Daily-Paper-Trading-Portfolioentwicklung, alle Universen",
            subtitle="Isolierte Tages-Simulation je Universum, Index (Start = 100)",
            series_by_universe=series_by_universe,
            output_path=output_dir / "10_daily_paper_trading_portfolio_comparison.png",
        )

    # 11: sharpe / drawdown comparison
    sharpe, drawdown = [], []
    for u in universes:
        if u["outperformance"] is not None:
            sharpe.append(u["outperformance"]["summary"]["sharpe_ratio"])
            drawdown.append(u["outperformance"]["summary"]["max_drawdown"])
        else:
            sharpe.append(None)
            drawdown.append(None)

    plots.plot_grouped_bars_multi_metric(
        title="Sharpe Ratio und Max Drawdown je Universum",
        subtitle="Outperformance-LSTM, bestes Top-K",
        universe_titles=universe_titles,
        universe_names=universe_names,
        series=[("Sharpe Ratio", sharpe), ("Max Drawdown", drawdown)],
        output_path=output_dir / "11_sharpe_drawdown_comparison.png",
        value_label="Wert",
        value_format=lambda v: f"{v:.2f}",
    )


def main():
    cfg.ensure_output_dirs()
    universes = [gather_universe_data(name) for name in cfg.UNIVERSE_NAMES]
    output_dir = cfg.COMPARISON_OUTPUT_DIR

    build_plots(universes, output_dir)
    raw, normalized, scores, all_criteria = compute_ranking(universes)

    parts = [
        "# Gesamtanalyse - Vergleich aller vier Universen",
        "",
        "Automatisch erzeugt von generate_cross_universe_analysis.py. Betrachtet das Projekt als ein "
        "gesamtes Forschungsprojekt statt vier Einzeluntersuchungen. Keine Modelle wurden neu trainiert, "
        "keine Handelslogik veraendert - nur Auswertung, Vergleich und Visualisierung bereits vorhandener "
        "Ergebnisse (siehe auch die einzelnen `wissenschaftliche_analyse.md`-Dateien je Universum, aus denen "
        "dieselben zugrundeliegenden Zahlen stammen).",
        "",
        "Kernfrage: Welches Universum eignet sich fuer unser Modell am besten und warum?",
        "",
        section_1_dataset(universes),
        section_2_model_performance(universes),
        section_3_backtest(universes),
        section_4_top_k(universes, output_dir),
        section_5_signal(universes),
        section_6_paper_trading(universes),
        section_7_model_behavior(universes),
        section_8_visualizations(),
        section_9_ranking(universes, scores, raw, normalized, all_criteria, output_dir),
        section_10_key_findings(universes, scores),
        presentation_recommendations(),
    ]

    output_path = output_dir / "gesamtanalyse.md"
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {output_path}")

    print("\n" + "=" * 80)
    print("GESAMTRANKING")
    print("=" * 80)
    for u in sorted(universes, key=lambda u: scores[u["universe"]], reverse=True):
        print(f"  {u['title']}: {scores[u['universe']]:.3f}")


if __name__ == "__main__":
    main()
