"""
Shared plotting helpers for the presentation reporting system.

Matplotlib only (no seaborn). Every plot goes through the same style
helpers in reporting_config.py, so a design change only has to happen in
one place. Two plot "modes" exist:

  - a real data plot (line/bar chart) when enough observations exist
  - a plain status graphic ("MISSING" / "PRELIMINARY: n=...") when they
    do not, per the project's no-fabrication rule

Nothing in this module reads trading/model files directly - it only
receives already-validated DataFrames and metric dicts from the loader and
metrics modules.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from reporting_config import (
    ANNOTATION_FONT_SIZE,
    AXIS_LABEL_FONT_SIZE,
    COLOR_ALWAYS_BUY,
    COLOR_BENCHMARK,
    COLOR_GRID,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_STRATEGY,
    DPI,
    FIGURE_SIZE_WIDE,
    LEGEND_FONT_SIZE,
    STATUS_MISSING,
    STATUS_PRELIMINARY,
    SUBTITLE_FONT_SIZE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
)


def _new_figure(figsize=FIGURE_SIZE_WIDE):
    fig = plt.figure(figsize=figsize)
    return fig


def _style_axis(ax):
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linewidth=0.6)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def _footer_text(fig, lines):
    text = "   |   ".join(lines)
    fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE, family="monospace")


def _percent(value):
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return "n/a"
    return f"{value:+.1%}"


def plot_status_message(title, status, message_lines, output_path, subtitle=None):
    """
    A clean, presentation-ready placeholder for MISSING/insufficient data.
    Never fabricates a line/bar chart - explicitly says what is missing.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.78])
    ax.axis("off")

    ax.text(0.5, 0.82, title, ha="center", va="center", fontsize=TITLE_FONT_SIZE, fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.72, subtitle, ha="center", va="center", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    badge_color = COLOR_NEUTRAL if status == STATUS_MISSING else "#B8860B"
    ax.text(
        0.5, 0.58, f"STATUS: {status}", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": badge_color, "edgecolor": "none"},
    )

    body = "\n".join(message_lines)
    ax.text(0.5, 0.32, body, ha="center", va="center", fontsize=12, wrap=True)

    _save(fig, output_path)


def plot_cumulative_lines(
    title,
    subtitle,
    dates,
    strategy_index,
    benchmark_index,
    strategy_label,
    benchmark_label,
    output_path,
    metric_lines,
    preliminary=False,
    extra_series=None,
):
    """
    Shared line-chart renderer for Plot 1 (backtest) and Plot 2/3 (Alpaca):
    two (or more) series normalized to 100 at the first observation,
    identical colors/legend position across every universe.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.08, 0.22, 0.88, 0.62])

    ax.plot(dates, strategy_index, color=COLOR_STRATEGY, linewidth=2.2, label=strategy_label)
    ax.plot(dates, benchmark_index, color=COLOR_BENCHMARK, linewidth=2.0, linestyle="--", label=benchmark_label)

    if extra_series:
        for label, series, color in extra_series:
            ax.plot(dates, series, color=color, linewidth=1.6, linestyle=":", label=label)

    ax.axhline(100, color="#999999", linewidth=0.8)
    ax.set_ylabel("Index (Start = 100)", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, frameon=True)

    title_text = title
    if preliminary:
        title_text += "  (vorlaeufig)"
    fig.suptitle(title_text, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _footer_text(fig, metric_lines)
    _save(fig, output_path)


def plot_horizontal_bars(
    title,
    subtitle,
    labels,
    values,
    value_format,
    output_path,
    metric_lines=None,
    color=COLOR_STRATEGY,
    value_label="",
    preliminary=False,
    secondary_annotations=None,
):
    fig = _new_figure()
    bottom = 0.16 if metric_lines else 0.08
    ax = fig.add_axes([0.28, bottom, 0.60, 0.68])

    bars = ax.barh(labels, values, color=color)
    ax.invert_yaxis()
    ax.set_xlabel(value_label, fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)

    bar_labels = [value_format(value) for value in values]
    if secondary_annotations:
        bar_labels = [f"{label}  ({extra})" for label, extra in zip(bar_labels, secondary_annotations)]
    ax.bar_label(bars, labels=bar_labels, padding=4, fontsize=ANNOTATION_FONT_SIZE)

    title_text = title
    if preliminary:
        title_text += "  (vorlaeufig)"
    fig.suptitle(title_text, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    if metric_lines:
        _footer_text(fig, metric_lines)
    _save(fig, output_path)


def plot_comparison_bars(
    title,
    subtitle,
    universes,
    values,
    statuses,
    output_path,
    value_label="Outperformance (Prozentpunkte)",
    value_is_fraction=True,
):
    """
    One bar per universe; universes with STATUS_MISSING are rendered as an
    empty/hatched bar with a "n/a" label instead of a fabricated 0.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.10, 0.22, 0.85, 0.62])

    display_values = []
    colors = []
    labels = []
    for universe, value, status in zip(universes, values, statuses):
        if status == STATUS_MISSING or value is None:
            display_values.append(0.0)
            colors.append("#DDDDDD")
            labels.append("n/a")
        else:
            plotted = value * 100 if value_is_fraction else value
            display_values.append(plotted)
            colors.append(COLOR_POSITIVE if plotted >= 0 else COLOR_NEGATIVE)
            labels.append(f"{plotted:+.1f}%")

    bars = ax.bar(universes, display_values, color=colors)
    for bar, status, label in zip(bars, statuses, labels):
        if status == STATUS_MISSING:
            bar.set_hatch("//")
            bar.set_edgecolor("#999999")
    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_ylabel(value_label, fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.bar_label(bars, labels=labels, padding=4, fontsize=ANNOTATION_FONT_SIZE)

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _save(fig, output_path)


def plot_trade_activity_comparison(universes, buys, sells, statuses, output_path, title, subtitle=None):
    fig = _new_figure()
    ax = fig.add_axes([0.10, 0.20, 0.85, 0.64])

    index_positions = range(len(universes))
    width = 0.38

    buys_plot = [0 if status == STATUS_MISSING else value for value, status in zip(buys, statuses)]
    sells_plot = [0 if status == STATUS_MISSING else value for value, status in zip(sells, statuses)]

    bars_buys = ax.bar([i - width / 2 for i in index_positions], buys_plot, width, color=COLOR_POSITIVE, label="Kaeufe")
    bars_sells = ax.bar([i + width / 2 for i in index_positions], sells_plot, width, color=COLOR_NEGATIVE, label="Verkaeufe")

    ax.set_xticks(list(index_positions))
    ax.set_xticklabels(universes)
    ax.set_ylabel("Anzahl", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    ax.bar_label(bars_buys, padding=3, fontsize=ANNOTATION_FONT_SIZE)
    ax.bar_label(bars_sells, padding=3, fontsize=ANNOTATION_FONT_SIZE)

    for index, status in enumerate(statuses):
        if status == STATUS_MISSING:
            ax.text(index, 0.5, "n/a", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE, color="#777777")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _save(fig, output_path)


def plot_simulated_paper_trading_extension(
    universe_title,
    benchmark_ticker,
    sim_dates,
    sim_strategy_index,
    sim_benchmark_index,
    sim_top_k,
    real_dates,
    real_strategy_index,
    real_benchmark_index,
    real_observations,
    output_path,
    sim_metric_lines,
    real_metric_lines=None,
):
    """
    Supplementary plot (not one of the 4 standard plots): a two-panel chart
    that always keeps SIMULATION and real Alpaca data visually and
    numerically separate, per the project rule "Backtest und Alpaca Paper
    Trading nicht vermischen".

    Left panel: the same day-by-day Top-K backtest reconstruction used for
    Plot 1 (real historical prices, real model predictions) shown as a
    stand-in for "what a longer paper-trading run would look like". This is
    a recomputation of Plot 1's own methodology, not a new/fabricated
    number - but it is explicitly labeled SIMULATION and is never an actual
    Alpaca broker result.

    Right panel: only the real Alpaca daily observations for this universe
    (if any). Never merged into the left panel's series.
    """
    has_real = real_dates is not None and len(real_dates) > 0

    fig = plt.figure(figsize=(13.5, 6))
    fig.suptitle(
        f"Simulierte Paper-Trading-Fortschreibung - {universe_title} vs. {benchmark_ticker}",
        fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.905,
        "LINKS: SIMULATION - echte Kurse & Modellvorhersagen, KEIN echtes Alpaca-Broker-Ergebnis"
        + ("   |   RECHTS: echte Alpaca-Daten" if has_real else "   |   RECHTS: keine echten Alpaca-Daten vorhanden"),
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#8B4500",
    )

    left_width = 0.54 if has_real else 0.86
    left = fig.add_axes([0.06, 0.16, left_width, 0.62])
    left.plot(sim_dates, sim_strategy_index, color=COLOR_STRATEGY, linewidth=2.0, label=f"Top-{sim_top_k} Simulation")
    left.plot(sim_dates, sim_benchmark_index, color=COLOR_BENCHMARK, linewidth=1.8, linestyle="--", label=f"{benchmark_ticker} (Index)")
    left.axhline(100, color="#999999", linewidth=0.8)
    _style_axis(left)
    left.set_ylabel("Index (Start = 100)", fontsize=AXIS_LABEL_FONT_SIZE)
    left.set_title(f"Simulation ({len(sim_dates)} Tage, Backtest-Methodik)", fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")
    left.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, frameon=True)

    if has_real:
        right = fig.add_axes([0.68, 0.16, 0.28, 0.62])
        right.plot(real_dates, real_strategy_index, color=COLOR_STRATEGY, linewidth=2.2, marker="o", label="Portfolio (echt)")
        right.plot(real_dates, real_benchmark_index, color=COLOR_BENCHMARK, linewidth=2.0, linestyle="--", marker="o", label=f"{benchmark_ticker} (echt)")
        right.axhline(100, color="#999999", linewidth=0.8)
        _style_axis(right)
        right.set_title(f"Echt (n={real_observations} Tage)", fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")
        right.legend(loc="upper left", fontsize=9, frameon=True)
        right.tick_params(axis="x", labelrotation=30)
    else:
        right = fig.add_axes([0.68, 0.16, 0.28, 0.62])
        right.axis("off")
        right.text(0.5, 0.5, "Keine echten\nAlpaca-Daten\nverfuegbar", ha="center", va="center", fontsize=11, color="#777777")

    footer_lines = list(sim_metric_lines)
    if real_metric_lines:
        footer_lines = footer_lines + real_metric_lines
    _footer_text(fig, footer_lines)
    _save(fig, output_path)


def plot_final_comparison(universes, backtest_values, hourly_values, daily_values, statuses_by_metric, output_path, title):
    """
    Grouped bar chart: backtest / hourly / daily outperformance per
    universe, side by side, with missing series rendered as hatched
    'n/a' bars instead of zero.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.09, 0.24, 0.87, 0.60])

    index_positions = list(range(len(universes)))
    width = 0.26
    series_config = [
        ("Backtest", backtest_values, statuses_by_metric["backtest"], COLOR_STRATEGY, -width),
        ("Hourly Alpaca", hourly_values, statuses_by_metric["hourly"], COLOR_BENCHMARK, 0),
        ("Daily Alpaca", daily_values, statuses_by_metric["daily"], COLOR_ALWAYS_BUY, width),
    ]

    for label, values, statuses, color, offset in series_config:
        plotted = []
        bar_colors = []
        for value, status in zip(values, statuses):
            if status == STATUS_MISSING or value is None:
                plotted.append(0.0)
                bar_colors.append("#DDDDDD")
            else:
                plotted.append(value * 100)
                bar_colors.append(color)
        positions = [position + offset for position in index_positions]
        bars = ax.bar(positions, plotted, width=width, color=bar_colors, label=label)
        for bar, status in zip(bars, statuses):
            if status == STATUS_MISSING:
                bar.set_hatch("//")
                bar.set_edgecolor("#999999")

    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_xticks(index_positions)
    ax.set_xticklabels(universes)
    ax.set_ylabel("Outperformance (Prozentpunkte)", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    ax.set_title(
        "Grau/schraffiert = keine belastbaren Daten verfuegbar. Zeitraeume unterscheiden sich zwischen Backtest und Alpaca (siehe Validierungsbericht).",
        fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left",
    )

    _save(fig, output_path)
