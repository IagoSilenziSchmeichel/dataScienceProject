"""
Generate the unified presentation report (4 standard plots per universe +
comparison plots/CSV + validation/summary/slide-plan markdown) for one or
all four stock universes.

Usage:
    python alpaca_trading/presentation_reporting/generate_presentation_reports.py
    python alpaca_trading/presentation_reporting/generate_presentation_reports.py --universe original_tech

This script only reads existing result/backtest/Alpaca log files (see
presentation_data_loader.py). It does not train any model, does not run or
change any existing training or trading pipeline, and does not modify or
delete any existing log, model, or data file. It only writes into
alpaca_trading/presentation_reporting/output/.
"""

from pathlib import Path
import argparse
import sys

REPORTING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORTING_ROOT))

import pandas as pd  # noqa: E402

import presentation_data_loader as loader  # noqa: E402
import presentation_metrics as metrics  # noqa: E402
import presentation_plots as plots  # noqa: E402
import reporting_config as cfg  # noqa: E402


def build_universe_report(universe_name):
    benchmark_ticker = cfg.BENCHMARKS[universe_name]
    universe_title = cfg.UNIVERSE_TITLE_DE[universe_name]
    universe_tickers = cfg.UNIVERSES[universe_name]
    output_dir = cfg.universe_output_dir(universe_name)

    result = {
        "universe": universe_name,
        "title": universe_title,
        "benchmark": benchmark_ticker,
        "research_question": cfg.RESEARCH_QUESTION_DE[universe_name],
        "sources": {},
        "backtest_summary": None,
        "daily_alpaca_summary": None,
        "daily_alpaca_orders_summary": None,
        "signal_summary": None,
    }

    # ---- Plot 1: historical backtest vs benchmark --------------------------
    backtest_ds = loader.load_backtest_predictions(universe_name)
    result["sources"]["backtest"] = backtest_ds.to_dict()
    plot1_path = output_dir / "01_backtest_vs_benchmark.png"

    if not backtest_ds.is_usable:
        plots.plot_status_message(
            title=f"Historischer Backtest - {universe_title} vs. {benchmark_ticker}",
            status=backtest_ds.status,
            message_lines=[backtest_ds.note],
            output_path=plot1_path,
        )
    else:
        best_top_k, candidates = metrics.select_best_top_k(backtest_ds.source_path)
        if best_top_k is None:
            best_top_k = max(
                candidates,
                key=lambda k: metrics.total_return_from_returns(
                    metrics.reconstruct_daily_topk_backtest(backtest_ds.data, benchmark_ticker, k)["Strategy_Return"]
                ),
            )
        daily = metrics.reconstruct_daily_topk_backtest(backtest_ds.data, benchmark_ticker, best_top_k)
        summary = metrics.summarize_backtest(daily, benchmark_ticker)
        summary["top_k"] = best_top_k
        result["backtest_summary"] = summary

        strategy_index = metrics.build_cumulative_index(daily["Strategy_Return"])
        benchmark_index = metrics.build_cumulative_index(daily["Benchmark_Return"])

        metric_lines = [
            f"Strategie (netto=brutto): {summary['strategy_return_net']:+.1%}",
            f"{benchmark_ticker}: {summary['benchmark_return']:+.1%}",
            f"Differenz: {summary['difference']:+.1%}",
            f"Sharpe: {summary['sharpe_ratio']:.2f}",
            f"MaxDD: {summary['max_drawdown']:.1%}",
            f"Trades: {summary['number_of_trades']}",
            "Kosten: 0% (im Daily-Backtest nicht modelliert)",
        ]

        plots.plot_cumulative_lines(
            title=f"Historischer Backtest - {universe_title} vs. {benchmark_ticker}",
            subtitle=(
                f"Top-{best_top_k} Outperformance-LSTM, "
                f"{daily.index.min().date()} bis {daily.index.max().date()} ({len(daily)} Handelstage)"
            ),
            dates=daily.index,
            strategy_index=strategy_index,
            benchmark_index=benchmark_index,
            strategy_label=f"Top-{best_top_k} Strategie",
            benchmark_label=f"{benchmark_ticker} (Index)",
            output_path=plot1_path,
            metric_lines=metric_lines,
            preliminary=(backtest_ds.status == cfg.STATUS_PRELIMINARY),
        )

    # ---- Plot 2: hourly Alpaca vs benchmark ---------------------------------
    hourly_ds = loader.load_hourly_alpaca_performance(universe_name)
    result["sources"]["hourly_alpaca"] = hourly_ds.to_dict()
    plot2_path = output_dir / "02_hourly_alpaca_vs_benchmark.png"

    plots.plot_status_message(
        title=f"Hourly Alpaca Paper Trading - {universe_title}",
        status=hourly_ds.status,
        message_lines=[
            "Keine belastbare universenspezifische Performance-Attribution ueber die Zeit verfuegbar."
            if hourly_ds.status == cfg.STATUS_MISSING
            else "Zu wenige Beobachtungen fuer eine belastbare Kurve.",
            hourly_ds.note,
        ],
        output_path=plot2_path,
    )

    # ---- Plot 3: daily Alpaca vs benchmark ----------------------------------
    daily_alpaca_ds = loader.load_daily_alpaca_performance(universe_name)
    result["sources"]["daily_alpaca"] = daily_alpaca_ds.to_dict()
    plot3_path = output_dir / "03_daily_alpaca_vs_benchmark.png"

    orders_ds = loader.load_daily_alpaca_orders(universe_name)
    result["sources"]["daily_alpaca_orders"] = orders_ds.to_dict()
    order_summary = metrics.summarize_orders(orders_ds.data) if orders_ds.is_usable else {
        "buys": None, "sells": None, "total_trades": None,
    }
    result["daily_alpaca_orders_summary"] = order_summary

    if not daily_alpaca_ds.is_usable:
        plots.plot_status_message(
            title=f"Daily Alpaca Paper Trading - {universe_title} vs. {benchmark_ticker}",
            status=daily_alpaca_ds.status,
            message_lines=[daily_alpaca_ds.note],
            output_path=plot3_path,
        )
    else:
        data = daily_alpaca_ds.data
        summary = metrics.summarize_alpaca_performance(data)
        result["daily_alpaca_summary"] = summary

        metric_lines = [
            f"{summary['period_start'].date()} - {summary['period_end'].date()}",
            f"n={summary['observations']} Tage",
            f"Portfolio: {summary['portfolio_return']:+.1%}",
            f"{benchmark_ticker}: {summary['benchmark_return']:+.1%}",
            f"Diff: {summary['difference']:+.1%}",
            f"Trades: {order_summary['total_trades']}",
        ]

        plots.plot_cumulative_lines(
            title=f"Daily Alpaca Paper Trading - {universe_title} vs. {benchmark_ticker}",
            subtitle=f"Isolierte Tages-Simulation je Universum, {summary['observations']} Beobachtungen",
            dates=data["date"],
            strategy_index=data["portfolio_index"],
            benchmark_index=data["benchmark_index"],
            strategy_label="Portfolio (Top-3)",
            benchmark_label=f"{benchmark_ticker} (Index)",
            output_path=plot3_path,
            metric_lines=metric_lines,
            preliminary=(daily_alpaca_ds.status == cfg.STATUS_PRELIMINARY),
        )

    # ---- Plot 4: signal / selection behaviour -------------------------------
    signal_ds = loader.load_signal_history(universe_name, timeframe="1Day")
    result["sources"]["signal_history"] = signal_ds.to_dict()
    plot4_path = output_dir / "04_signal_selection_analysis.png"

    if not signal_ds.is_usable:
        plots.plot_status_message(
            title=f"Top-K-Auswahlhaeufigkeit - {universe_title}",
            status=signal_ds.status,
            message_lines=[signal_ds.note],
            output_path=plot4_path,
        )
    else:
        selection_df, distinct_dates = metrics.summarize_signal_selection(signal_ds.data, universe_tickers)
        result["signal_summary"] = {"distinct_dates": distinct_dates, "table": selection_df.to_dict(orient="records")}

        secondary_annotations = [
            f"Ø p={row['average_probability']:.2f}, Ø Rang={row['average_rank']:.1f}, {int(row['new_entries'])}x neu"
            if pd.notna(row["average_probability"])
            else "keine Signale"
            for _, row in selection_df.iterrows()
        ]

        plots.plot_horizontal_bars(
            title=f"Top-K-Auswahlhaeufigkeit - {universe_title}",
            subtitle=f"Anteil der Signaltage in Top-K, n={distinct_dates} Tage (Daily Alpaca-Signale)",
            labels=selection_df["ticker"],
            values=selection_df["selection_share"],
            value_format=lambda value: f"{value:.0%}",
            output_path=plot4_path,
            value_label="Anteil der Signaltage in Top-K",
            metric_lines=[f"n={distinct_dates} Signaltage", "Quelle: alpaca_trading/logs/paper_signals.csv (1Day)"],
            preliminary=(signal_ds.status == cfg.STATUS_PRELIMINARY),
            secondary_annotations=secondary_annotations,
        )

    return result


def build_comparison_outputs(universe_results):
    cfg.COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universes = [r["universe"] for r in universe_results]
    universe_titles = [r["title"] for r in universe_results]

    backtest_values, backtest_statuses = [], []
    for r in universe_results:
        if r["backtest_summary"] is None:
            backtest_values.append(None)
            backtest_statuses.append(cfg.STATUS_MISSING)
        else:
            backtest_values.append(r["backtest_summary"]["difference"])
            backtest_statuses.append(r["sources"]["backtest"]["status"])

    plots.plot_comparison_bars(
        title="Backtest-Outperformance je Universum",
        subtitle="Differenz Top-K-Strategie minus Benchmark (Zeitraeume koennen je Universum abweichen)",
        universes=universe_titles,
        values=backtest_values,
        statuses=backtest_statuses,
        output_path=cfg.COMPARISON_OUTPUT_DIR / "01_backtest_outperformance_by_universe.png",
    )

    hourly_values = [None for _ in universe_results]
    hourly_statuses = [cfg.STATUS_MISSING for _ in universe_results]

    plots.plot_comparison_bars(
        title="Hourly-Alpaca-Outperformance je Universum",
        subtitle="Keine belastbare universenspezifische Attribution ueber die Zeit verfuegbar",
        universes=universe_titles,
        values=hourly_values,
        statuses=hourly_statuses,
        output_path=cfg.COMPARISON_OUTPUT_DIR / "02_hourly_alpaca_outperformance_by_universe.png",
    )

    daily_values, daily_statuses = [], []
    for r in universe_results:
        if r["daily_alpaca_summary"] is None:
            daily_values.append(None)
            daily_statuses.append(cfg.STATUS_MISSING)
        else:
            daily_values.append(r["daily_alpaca_summary"]["difference"])
            daily_statuses.append(r["sources"]["daily_alpaca"]["status"])

    plots.plot_comparison_bars(
        title="Daily-Alpaca-Outperformance je Universum",
        subtitle="Isolierte Tages-Simulation, 5 Handelstage - vorlaeufig",
        universes=universe_titles,
        values=daily_values,
        statuses=daily_statuses,
        output_path=cfg.COMPARISON_OUTPUT_DIR / "03_daily_alpaca_outperformance_by_universe.png",
    )

    buys, sells, trade_statuses = [], [], []
    for r in universe_results:
        order_summary = r.get("daily_alpaca_orders_summary") or {}
        if order_summary.get("total_trades") is None:
            buys.append(0)
            sells.append(0)
            trade_statuses.append(cfg.STATUS_MISSING)
        else:
            buys.append(order_summary["buys"])
            sells.append(order_summary["sells"])
            trade_statuses.append(cfg.STATUS_READY)

    plots.plot_trade_activity_comparison(
        universes=universe_titles,
        buys=buys,
        sells=sells,
        statuses=trade_statuses,
        output_path=cfg.COMPARISON_OUTPUT_DIR / "04_trade_activity_by_universe.png",
        title="Handelsaktivitaet je Universum (Daily Alpaca, isolierte Simulation)",
    )

    plots.plot_final_comparison(
        universes=universe_titles,
        backtest_values=backtest_values,
        hourly_values=hourly_values,
        daily_values=daily_values,
        statuses_by_metric={"backtest": backtest_statuses, "hourly": hourly_statuses, "daily": daily_statuses},
        output_path=cfg.COMPARISON_OUTPUT_DIR / "05_final_universe_comparison.png",
        title="Gesamtvergleich: Backtest vs. Hourly Alpaca vs. Daily Alpaca",
    )

    rows = []
    for r in universe_results:
        backtest_summary = r["backtest_summary"] or {}
        daily_summary = r["daily_alpaca_summary"] or {}
        order_summary = r.get("daily_alpaca_orders_summary") or {}
        hourly_source = r["sources"]["hourly_alpaca"]

        data_quality_flags = []
        if r["sources"]["backtest"]["status"] == cfg.STATUS_MISSING:
            data_quality_flags.append("backtest_missing")
        if r["sources"]["daily_alpaca"]["status"] in (cfg.STATUS_PRELIMINARY, cfg.STATUS_MISSING):
            data_quality_flags.append("daily_alpaca_preliminary")
        data_quality_flags.append("hourly_alpaca_missing")

        rows.append(
            {
                "universe": r["universe"],
                "benchmark": r["benchmark"],
                "backtest_start": backtest_summary.get("period_start"),
                "backtest_end": backtest_summary.get("period_end"),
                "backtest_strategy_return": backtest_summary.get("strategy_return_net"),
                "backtest_benchmark_return": backtest_summary.get("benchmark_return"),
                "backtest_difference": backtest_summary.get("difference"),
                "backtest_sharpe": backtest_summary.get("sharpe_ratio"),
                "backtest_max_drawdown": backtest_summary.get("max_drawdown"),
                "backtest_trades": backtest_summary.get("number_of_trades"),
                "backtest_transaction_cost": backtest_summary.get("transaction_cost_assumption"),
                "hourly_start": None,
                "hourly_end": None,
                "hourly_observations": hourly_source["observations"],
                "hourly_strategy_return": None,
                "hourly_benchmark_return": None,
                "hourly_difference": None,
                "hourly_trades": None,
                "hourly_max_drawdown": None,
                "daily_start": daily_summary.get("period_start"),
                "daily_end": daily_summary.get("period_end"),
                "daily_observations": daily_summary.get("observations"),
                "daily_strategy_return": daily_summary.get("portfolio_return"),
                "daily_benchmark_return": daily_summary.get("benchmark_return"),
                "daily_difference": daily_summary.get("difference"),
                "daily_trades": order_summary.get("total_trades"),
                "daily_max_drawdown": daily_summary.get("max_drawdown"),
                "data_quality_status": ";".join(data_quality_flags) if data_quality_flags else "ok",
                "notes": r["sources"]["backtest"]["note"] if r["sources"]["backtest"]["status"] == cfg.STATUS_MISSING else "",
            }
        )

    comparison_df = pd.DataFrame(rows)
    csv_path = cfg.COMPARISON_OUTPUT_DIR / "final_universe_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return comparison_df


# ---------------------------------------------------------------------------
# Markdown reports (sections 6, 7, 8 of the specification)
# ---------------------------------------------------------------------------

def write_validation_report(universe_results, read_errors):
    lines = ["# Data Validation Report", "", "Automatisch erzeugt von generate_presentation_reports.py. Nur Lesevorgaenge - keine Datei wurde veraendert.", ""]

    for r in universe_results:
        lines.append(f"## {r['title']} ({r['universe']})")
        lines.append("")
        lines.append(f"Benchmark: {r['benchmark']}")
        lines.append("")
        lines.append("| Plot | Datenquelle | Zeitraum/Beobachtungen | Status | Einschraenkung |")
        lines.append("| --- | --- | --- | --- | --- |")

        backtest_source = r["sources"]["backtest"]
        backtest_period = "n/a"
        if r["backtest_summary"]:
            backtest_period = (
                f"{r['backtest_summary']['period_start'].date()} - "
                f"{r['backtest_summary']['period_end'].date()} "
                f"({r['backtest_summary']['number_of_days']} Tage)"
            )
        lines.append(
            f"| 01 Backtest | {backtest_source['source_path'] or 'n/a'} | {backtest_period} | "
            f"{backtest_source['status']} | {backtest_source['note']} |"
        )

        hourly_source = r["sources"]["hourly_alpaca"]
        lines.append(
            f"| 02 Hourly Alpaca | alpaca_trading/logs/*.csv (inkl. Legacy) | "
            f"{hourly_source['observations']} Zeitstempel | {hourly_source['status']} | {hourly_source['note']} |"
        )

        daily_source = r["sources"]["daily_alpaca"]
        daily_period = "n/a"
        if r["daily_alpaca_summary"]:
            daily_period = (
                f"{r['daily_alpaca_summary']['period_start'].date()} - "
                f"{r['daily_alpaca_summary']['period_end'].date()} ({r['daily_alpaca_summary']['observations']} Tage)"
            )
        lines.append(
            f"| 03 Daily Alpaca | {daily_source['source_path'] or 'n/a'} | {daily_period} | "
            f"{daily_source['status']} | {daily_source['note']} |"
        )

        signal_source = r["sources"]["signal_history"]
        lines.append(
            f"| 04 Signal-Auswahl | {signal_source['source_path'] or 'n/a'} | "
            f"{signal_source['observations']} Tage | {signal_source['status']} | {signal_source['note']} |"
        )
        lines.append("")

    lines.append("## Uebergreifende Einschraenkungen")
    lines.append("")
    lines.append(
        "- Der Daily-Outperformance-LSTM-Backtest (Plot 1) existiert aktuell nur fuer original_tech "
        "(letzter Lauf in experiments/exp_2_lstm/data/processed). Fuer die anderen drei Universen "
        "muss er erst ueber `python run_universe_lstm_outperformance.py --universe <name>` erzeugt werden."
    )
    lines.append(
        "- Hourly Alpaca Paper Trading hat fuer alle Universen nur 1-2 Zeitstempel (der einmalige Dry-Run "
        "vom 2026-07-06). Die urspruenglich stuendlichen Logs wurden danach vom Daily-Scheduler mit "
        "1Day-Zeilen ueberschrieben/archiviert (Schema-Wechsel -> Legacy-Dateien). Es existiert keine "
        "fortlaufende Hourly-Zeitreihe."
    )
    lines.append(
        "- Die geteilten Konto-Logs unter alpaca_trading/logs/ (performance_history.csv, paper_performance.csv, "
        "orders_history.csv, positions_history.csv) fahren alle vier Universen im selben Alpaca-Paper-Account. "
        "Portfolio-Wert und -Rendite in diesen Dateien sind daher NICHT sauber einem einzelnen Universum "
        "zurechenbar und wurden fuer Plot 2/3 bewusst NICHT verwendet. Stattdessen wurde die isolierte "
        "Simulation unter alpaca_trading/per_universe_daily_mark_to_market/<universe>/ verwendet, die pro "
        "Universum ein eigenes Startkapital fuehrt."
    )
    lines.append(
        "- Der Daily-Backtest-Benchmark (Plot 1) ist die tatsaechliche Index-Rendite von QQQ/SPY "
        "(`Next_Day_QQQ_Return`/`Next_Day_SPY_Return` aus den Testdaten), nicht die gleichgewichtete "
        "Buy-and-Hold-Rendite der 10 Einzelaktien, die in `lstm_outperformance_top_k_results.csv` als "
        "`buy_and_hold_return` ausgewiesen ist. Beide Zahlen beantworten unterschiedliche Fragen und sollten "
        "nicht verwechselt werden."
    )
    lines.append(
        "- Im Daily-Outperformance-LSTM-Backtest sind keine Transaktionskosten modelliert (Kostenannahme 0%); "
        "Brutto- und Nettorendite sind daher identisch. Die isolierte Alpaca-Tagessimulation modelliert "
        "ebenfalls keine expliziten Gebuehren/Spreads."
    )
    unique_read_errors = list(dict.fromkeys(read_errors))
    if unique_read_errors:
        lines.append("- Folgende Logdateien konnten nicht eingelesen werden (defektes/altes Schema) und wurden uebersprungen:")
        for error in unique_read_errors:
            lines.append(f"  - {error}")
    lines.append("")

    output_path = cfg.OUTPUT_ROOT / "data_validation_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


def _interpret_backtest(r):
    summary = r["backtest_summary"]
    if summary is None:
        return f"Kein Backtest-Ergebnis vorhanden ({r['sources']['backtest']['note']})"
    direction = "schlaegt" if summary["difference"] > 0 else "unterliegt"
    return (
        f"Die Top-{summary['top_k']}-Strategie erzielt {summary['strategy_return_net']:+.1%} "
        f"gegenueber {summary['benchmark_return']:+.1%} fuer {r['benchmark']} "
        f"({summary['period_start'].date()} bis {summary['period_end'].date()}, {summary['number_of_days']} Handelstage). "
        f"Die Strategie {direction} den Benchmark um {summary['difference']:+.1%} "
        f"(Sharpe {summary['sharpe_ratio']:.2f}, Max Drawdown {summary['max_drawdown']:.1%}, "
        f"{summary['number_of_trades']} Trades, keine Transaktionskosten modelliert)."
    )


def _interpret_daily_alpaca(r):
    summary = r["daily_alpaca_summary"]
    if summary is None:
        return f"Keine belastbaren Daily-Alpaca-Daten ({r['sources']['daily_alpaca']['note']})"
    direction = "vor" if summary["difference"] > 0 else "hinter"
    return (
        f"Ueber {summary['observations']} Handelstage ({summary['period_start'].date()} bis "
        f"{summary['period_end'].date()}) liegt das isolierte Paper-Portfolio mit "
        f"{summary['portfolio_return']:+.1%} {direction} {r['benchmark']} ({summary['benchmark_return']:+.1%}). "
        "Vorlaeufig, da nur wenige Handelstage vorliegen."
    )


def _interpret_signal(r):
    if r["signal_summary"] is None:
        return f"Keine belastbaren Signaldaten ({r['sources']['signal_history']['note']})"
    table = r["signal_summary"]["table"]
    if not table:
        return "Keine Signaldaten."
    top_pick = table[0]
    return (
        f"Ueber {r['signal_summary']['distinct_dates']} Signaltage wurde {top_pick['ticker']} am haeufigsten "
        f"ausgewaehlt ({top_pick['selection_share']:.0%} der Tage, durchschnittliche Wahrscheinlichkeit "
        f"{top_pick['average_probability']:.2f}). Vorlaeufig, da nur wenige Signaltage vorliegen."
    )


def write_presentation_results_summary(universe_results):
    lines = ["# Presentation Results Summary", ""]

    for r in universe_results:
        lines.append(f"## {r['title']}")
        lines.append("")
        lines.append("Forschungsfrage:")
        lines.append(r["research_question"])
        lines.append("")
        lines.append("Backtest:")
        lines.append(_interpret_backtest(r))
        lines.append("")
        lines.append("Hourly Alpaca:")
        lines.append(
            "Keine belastbare universenspezifische Performance-Attribution verfuegbar "
            f"({r['sources']['hourly_alpaca']['note']})"
        )
        lines.append("")
        lines.append("Daily Alpaca:")
        lines.append(_interpret_daily_alpaca(r))
        lines.append("")
        lines.append("Signalverhalten:")
        lines.append(_interpret_signal(r))
        lines.append("")

        core_claim = "Backtest-Ergebnis noch nicht verfuegbar."
        if r["backtest_summary"] is not None:
            summary = r["backtest_summary"]
            core_claim = (
                f"Im verfuegbaren Testzeitraum {'schlaegt' if summary['difference'] > 0 else 'schlaegt nicht'} "
                f"die Top-{summary['top_k']}-Outperformance-LSTM-Strategie {r['benchmark']} "
                f"({summary['difference']:+.1%})."
            )
        lines.append("Kernaussage:")
        lines.append(core_claim)
        lines.append("")

    lines.append("## Gesamtvergleich")
    lines.append("")

    with_backtest = [r for r in universe_results if r["backtest_summary"] is not None]
    if with_backtest:
        best_backtest = max(with_backtest, key=lambda r: r["backtest_summary"]["difference"])
        lowest_drawdown = min(with_backtest, key=lambda r: r["backtest_summary"]["max_drawdown"])
        lines.append(f"- Staerkstes Backtest-Universum: {best_backtest['title']} ({best_backtest['backtest_summary']['difference']:+.1%})")
        lines.append(f"- Geringster Max Drawdown (Backtest): {lowest_drawdown['title']} ({lowest_drawdown['backtest_summary']['max_drawdown']:.1%})")
    else:
        lines.append("- Kein Universum hat aktuell ein vollstaendiges Backtest-Ergebnis.")

    lines.append("- Staerkstes Hourly-Universum: nicht auswertbar (keine belastbare Hourly-Attribution verfuegbar).")

    with_daily = [r for r in universe_results if r["daily_alpaca_summary"] is not None]
    if with_daily:
        best_daily = max(with_daily, key=lambda r: r["daily_alpaca_summary"]["difference"])
        lines.append(f"- Staerkstes Daily-Alpaca-Universum (vorlaeufig, 5 Tage): {best_daily['title']} ({best_daily['daily_alpaca_summary']['difference']:+.1%})")
    else:
        lines.append("- Staerkstes Daily-Alpaca-Universum: keine Daten.")

    trade_counts = [
        (r["title"], (r.get("daily_alpaca_orders_summary") or {}).get("total_trades"))
        for r in universe_results
    ]
    trade_counts = [(title, count) for title, count in trade_counts if count is not None]
    if trade_counts:
        most_active = max(trade_counts, key=lambda item: item[1])
        lines.append(f"- Hoechste Handelsaktivitaet (Daily Alpaca): {most_active[0]} ({most_active[1]} Trades)")

    lines.append(
        "- Wichtigste wissenschaftliche Einschraenkung: Der Backtest deckt fuer die meisten Universen noch "
        "keinen Zeitraum ab (nur original_tech ist vollstaendig durchgelaufen), und alle Alpaca-Daten "
        "(hourly wie daily) beruhen auf sehr kurzen Testfenstern (1-5 Beobachtungen). Keine der Aussagen "
        "hier ist statistisch belastbar genug fuer eine endgueltige Generalisierungsaussage."
    )
    lines.append(
        "- Abschliessende Interpretation: Auf Basis der bisher vorliegenden Daten kann noch nicht "
        "abschliessend beurteilt werden, ob die Strategie tech-spezifisch ist oder generalisiert. "
        "original_tech zeigt im Backtest eine deutliche Outperformance; fuer tech_no_nvda, new_tech und "
        "defensive_non_tech fehlt der vergleichbare Backtest noch."
    )
    lines.append("")

    output_path = cfg.OUTPUT_ROOT / "presentation_results_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


def write_slide_plan():
    lines = [
        "# Presentation Slide Plan",
        "",
        "Empfohlene Foliennummerierung, Plot und Kernaussage je Folie.",
        "",
        "## Folie 1: Warum vier Universen?",
        "- Plot: keiner (Text/Diagramm der vier Universen)",
        "- Kennzahlen: 4 Universen, 2 Benchmarks (QQQ/SPY)",
        "- Kernaussage: Wir testen Generalisierung ueber Tech-Zusammensetzung und Tech-vs-Non-Tech.",
        "",
        "## Folie 2: Versuchsaufbau und Benchmarks",
        "- Plot: keiner (Tabelle: Universum, Ticker, Benchmark, Forschungsfrage)",
        "- Kennzahlen: Top-K-Strategie, Modell = Outperformance-LSTM",
        "- Kernaussage: Gleiches Modell, gleiche Strategie, vier verschiedene Aktienmengen.",
        "",
        "## Folie 3: Original Tech - vier Ergebnisdimensionen",
        "- Plots: original_tech/01-04",
        "- Kennzahlen: Backtest-Differenz, Sharpe, Top-K-Auswahl",
        "- Kernaussage: siehe presentation_results_summary.md, Abschnitt Original Tech",
        "",
        "## Folie 4: Tech ohne Nvidia - vier Ergebnisdimensionen",
        "- Plots: tech_no_nvda/01-04",
        "- Kennzahlen: Backtest-Differenz (falls vorhanden), Daily-Alpaca-Differenz",
        "- Kernaussage: siehe presentation_results_summary.md, Abschnitt Tech ohne Nvidia",
        "",
        "## Folie 5: Neue Tech-Aktien - vier Ergebnisdimensionen",
        "- Plots: new_tech/01-04",
        "- Kennzahlen: Backtest-Differenz (falls vorhanden), Daily-Alpaca-Differenz",
        "- Kernaussage: siehe presentation_results_summary.md, Abschnitt Neue Tech-Aktien",
        "",
        "## Folie 6: Defensive Non-Tech - vier Ergebnisdimensionen",
        "- Plots: defensive_non_tech/01-04",
        "- Kennzahlen: Backtest-Differenz (falls vorhanden), Daily-Alpaca-Differenz",
        "- Kernaussage: siehe presentation_results_summary.md, Abschnitt Defensive Non-Tech",
        "",
        "## Folie 7: Direkter Vergleich aller Universen",
        "- Plots: comparison/01, comparison/03, comparison/04",
        "- Kennzahlen: final_universe_comparison.csv",
        "- Kernaussage: welches Universum performt im Backtest/Daily-Alpaca am besten, wo ist die Datenlage noch duenn",
        "",
        "## Folie 8: Backtest vs. Alpaca - Unterschiede",
        "- Plot: comparison/05_final_universe_comparison.png",
        "- Kennzahlen: Zeitraum Backtest (Monate) vs. Alpaca (Tage)",
        "- Kernaussage: Backtest und Live-Paper-Trading sind nicht direkt gleichwertig - unterschiedliche Zeitraeume und Stichprobengroessen.",
        "",
        "## Folie 9: Generalisiert das Modell?",
        "- Plot: comparison/01_backtest_outperformance_by_universe.png",
        "- Kennzahlen: Backtest-Differenz aller vier Universen (grau = fehlend)",
        "- Kernaussage: siehe Gesamtvergleich in presentation_results_summary.md",
        "",
        "## Folie 10: Limitierungen und Fazit",
        "- Plot: keiner (Textfolie)",
        "- Kennzahlen: Anzahl fehlender/vorlaeufiger Datenquellen aus data_validation_report.md",
        "- Kernaussage: Ergebnisse sind vorlaeufig; Backtest fehlt fuer 3 von 4 Universen, Alpaca-Historie ist sehr kurz.",
        "",
    ]

    output_path = cfg.OUTPUT_ROOT / "presentation_slide_plan.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


def write_readme():
    content = """# Presentation Reporting

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
"""
    output_path = REPORTING_ROOT / "README.md"
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate the unified presentation report.")
    parser.add_argument("--universe", choices=cfg.UNIVERSE_NAMES, default=None)
    args = parser.parse_args()

    cfg.ensure_output_dirs()
    universe_names = [args.universe] if args.universe else cfg.UNIVERSE_NAMES

    universe_results = [build_universe_report(name) for name in universe_names]

    if not args.universe:
        build_comparison_outputs(universe_results)
        write_validation_report(universe_results, loader.get_read_errors())
        write_presentation_results_summary(universe_results)
        write_slide_plan()
    else:
        print("Hinweis: --universe erzeugt nur die vier Plots dieses Universums, keine Vergleichsberichte.")

    write_readme()

    print("\n" + "=" * 80)
    print("ABSCHLUSSAUSGABE")
    print("=" * 80)
    for r in universe_results:
        print(f"\n{r['title']} ({r['universe']}):")
        for source_name, source in r["sources"].items():
            print(f"  - {source_name}: {source['status']} ({source['source_path'] or 'kein Pfad'})")


if __name__ == "__main__":
    main()
