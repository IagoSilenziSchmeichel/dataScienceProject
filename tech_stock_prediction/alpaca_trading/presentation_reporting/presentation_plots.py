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
import numpy as np
import pandas as pd

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
    UNIVERSE_COLORS,
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


def plot_hourly_hybrid_series(
    title,
    subtitle,
    data,
    benchmark_label,
    output_path,
    metric_lines,
):
    """
    Plot a clearly labeled hybrid hourly scenario.

    Real and simulated sections are visually separated. Simulated points are
    never drawn as if they were real Paper-Trading observations.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.08, 0.24, 0.88, 0.58])

    real = data[data["source_type"] == "real"].copy()
    simulated = data[data["source_type"] == "simulated"].copy()

    if not real.empty:
        ax.plot(
            real["timestamp"],
            real["model_index"],
            color=COLOR_STRATEGY,
            linewidth=2.4,
            label="Modell - real",
        )
        ax.plot(
            real["timestamp"],
            real["benchmark_index"],
            color=COLOR_BENCHMARK,
            linewidth=2.1,
            label=f"{benchmark_label} - real",
        )

    if not simulated.empty:
        # Include the last real point as visual anchor for the dashed continuation.
        model_sim = simulated[["timestamp", "model_index"]].copy()
        bench_sim = simulated[["timestamp", "benchmark_index"]].copy()
        if not real.empty:
            anchor = real.iloc[[-1]]
            model_sim = pd.concat(
                [anchor[["timestamp", "model_index"]], model_sim],
                ignore_index=True,
            )
            bench_sim = pd.concat(
                [anchor[["timestamp", "benchmark_index"]], bench_sim],
                ignore_index=True,
            )
            transition = real["timestamp"].max()
            ax.axvline(transition, color="#333333", linestyle=":", linewidth=1.2)
            ax.text(
                transition,
                0.96,
                "Simulation beginnt",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#333333",
            )

        ax.plot(
            model_sim["timestamp"],
            model_sim["model_index"],
            color=COLOR_STRATEGY,
            linewidth=2.2,
            linestyle="--",
            label="Modell - simuliert",
        )
        ax.plot(
            bench_sim["timestamp"],
            bench_sim["benchmark_index"],
            color=COLOR_BENCHMARK,
            linewidth=2.0,
            linestyle="--",
            label=f"{benchmark_label} - simuliert",
        )

    ax.axhline(100, color="#999999", linewidth=0.8)
    ax.set_ylabel("Index (Start = 100)", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, frameon=True)

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")
    ax.text(
        0.99,
        0.04,
        "HYBRID: REAL + SIMULATION",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": COLOR_NEUTRAL, "edgecolor": "none"},
    )
    fig.autofmt_xdate()
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


def plot_top_k_results(
    title,
    subtitle,
    top_k_values,
    strategy_returns,
    buy_and_hold_returns,
    differences,
    average_positions,
    output_path,
    metric_lines,
    preliminary=False,
):
    """
    Plot 2: grouped bar chart, Top-1..Top-5 strategy return vs. buy-and-hold
    return, straight from the pipeline's own lstm_outperformance_top_k_results.csv
    (nothing recomputed/estimated). The best strategy bar is highlighted.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.08, 0.22, 0.88, 0.60])

    index_positions = list(range(len(top_k_values)))
    width = 0.36
    # Average position count is folded into the tick label itself (not a
    # floating annotation) so it can never overlap the axis or bar labels.
    labels = [f"Top-{k}\n(Ø Pos. {positions:.1f})" for k, positions in zip(top_k_values, average_positions)]

    strategy_pct = [v * 100 for v in strategy_returns]
    buy_hold_pct = [v * 100 for v in buy_and_hold_returns]

    best_index = int(np.nanargmax(strategy_pct)) if strategy_pct else 0
    strategy_colors = [COLOR_STRATEGY for _ in strategy_pct]
    strategy_colors[best_index] = COLOR_POSITIVE

    bars_strategy = ax.bar(
        [i - width / 2 for i in index_positions], strategy_pct,
        width, color=strategy_colors, label="Strategie-Rendite",
    )
    bars_buy_hold = ax.bar(
        [i + width / 2 for i in index_positions], buy_hold_pct,
        width, color=COLOR_BENCHMARK, alpha=0.75, label="Buy-and-Hold Universum",
    )
    bars_strategy[best_index].set_edgecolor("#111111")
    bars_strategy[best_index].set_linewidth(2.0)

    ax.set_xticks(index_positions)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_ylabel("Rendite (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)

    # Headroom above the tallest bar so the difference labels and the legend
    # never overlap a bar or each other, regardless of how tall the tallest
    # bar in a given universe turns out to be.
    all_bar_values = strategy_pct + buy_hold_pct + [0]
    tallest_bar = max(all_bar_values)
    lowest_bar = min(all_bar_values)
    ax.set_ylim(
        bottom=lowest_bar * 1.25 if lowest_bar < 0 else None,
        top=tallest_bar * 1.35 if tallest_bar > 0 else 1,
    )

    strategy_labels = [f"{value:+.1f}%" for value in strategy_pct]
    buy_hold_labels = [f"{value:+.1f}%" for value in buy_hold_pct]
    ax.bar_label(bars_strategy, labels=strategy_labels, padding=3, fontsize=ANNOTATION_FONT_SIZE)
    ax.bar_label(bars_buy_hold, labels=buy_hold_labels, padding=3, fontsize=ANNOTATION_FONT_SIZE)
    for index, diff in enumerate(differences):
        ax.text(
            index,
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04,
            f"Diff {diff:+.1%}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            color=COLOR_POSITIVE if diff >= 0 else COLOR_NEGATIVE,
            fontweight="bold",
        )

    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    title_text = title
    if preliminary:
        title_text += "  (vorlaeufig)"
    fig.suptitle(title_text, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _footer_text(fig, metric_lines)
    _save(fig, output_path)


def plot_hourly_dashboard_table(title, subtitle, sections, output_path, footnote=None):
    """
    Plot 3: presentation dashboard as a compact table. `sections` is a list
    of (section_title, [(metric, value, tone), ...]) where tone can be
    positive/negative/neutral.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.78])
    ax.axis("off")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.text(0.0, 0.98, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    table_rows = []
    row_colors = []
    for section_title, rows in sections:
        table_rows.append([section_title, ""])
        row_colors.append(["#E5E7EB", "#E5E7EB"])
        for metric, value, tone in rows:
            table_rows.append([metric, value])
            if tone == "positive":
                value_color = "#DDEFE5"
            elif tone == "negative":
                value_color = "#F3D9D8"
            else:
                value_color = "#F3F4F6"
            row_colors.append(["#FFFFFF", value_color])

    table = ax.table(
        cellText=table_rows,
        cellColours=row_colors,
        colLabels=["Kennzahl", "Wert"],
        colWidths=[0.58, 0.34],
        cellLoc="left",
        bbox=[0.02, 0.10, 0.96, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_facecolor("#D1D5DB")
        if row > 0 and table_rows[row - 1][1] == "":
            cell.set_text_props(weight="bold")

    if footnote:
        ax.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE, color="#555555")

    _save(fig, output_path)


def plot_paper_trading_metric_table(title, subtitle, sections, output_path, footnote=None):
    """
    Presentation table for Daily and Hourly Paper Trading with identical
    structure: metric, model value, benchmark value, and difference.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
    ax.axis("off")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.text(0.0, 0.96, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    table_rows = []
    cell_colors = []
    for section_title, rows in sections:
        table_rows.append([section_title, "", "", ""])
        cell_colors.append(["#E5E7EB"] * 4)
        for metric, model_value, benchmark_value, difference_value, tone in rows:
            diff_color = "#F3F4F6"
            if tone == "positive":
                diff_color = "#DDEFE5"
            elif tone == "negative":
                diff_color = "#F3D9D8"
            table_rows.append([metric, model_value, benchmark_value, difference_value])
            cell_colors.append(["#FFFFFF", "#FFFFFF", "#FFFFFF", diff_color])

    table = ax.table(
        cellText=table_rows,
        cellColours=cell_colors,
        colLabels=["Kennzahl", "Modell", "Benchmark", "Differenz"],
        colWidths=[0.36, 0.20, 0.20, 0.20],
        cellLoc="center",
        bbox=[0.0, 0.08, 1.0, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_facecolor("#D1D5DB")
        elif row > 0 and table_rows[row - 1][1] == "":
            cell.set_text_props(weight="bold")
            if col > 0:
                cell.get_text().set_text("")
        if col == 0:
            cell.set_text_props(ha="left")

    if footnote:
        ax.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE, color="#555555")

    _save(fig, output_path)


def plot_paper_trading_summary_table(title, subtitle, summary_rows, detail_rows, detail_columns, output_path):
    """
    Presentation-ready Daily/Hourly Paper-Trading table.

    The PNG intentionally contains only presentation numbers. Technical
    source notes and limitations are written to markdown validation reports.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
    ax.axis("off")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    ax.text(0.0, 0.965, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    main_table = ax.table(
        cellText=summary_rows,
        colLabels=["Kennzahl", "Wert", "Kennzahl", "Wert"],
        colWidths=[0.28, 0.20, 0.30, 0.20],
        cellLoc="center",
        bbox=[0.0, 0.42, 1.0, 0.48],
    )
    main_table.auto_set_font_size(False)
    main_table.set_fontsize(9.2)
    main_table.scale(1.0, 1.18)

    for (row, col), cell in main_table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#D1D5DB")
            cell.set_text_props(weight="bold")
        elif col in (1, 3):
            text = cell.get_text().get_text()
            if text.startswith("+"):
                cell.set_facecolor("#DDEFE5")
            elif text.startswith("-"):
                cell.set_facecolor("#F3D9D8")
            else:
                cell.set_facecolor("#F8F8F8")
        if col in (0, 2):
            cell.set_text_props(ha="left")

    ax.text(0.0, 0.365, "Detailwerte", ha="left", va="center", fontsize=11, fontweight="bold")
    detail_table = ax.table(
        cellText=detail_rows,
        colLabels=detail_columns,
        colWidths=[0.30, 0.20, 0.24, 0.22, 0.12][: len(detail_columns)],
        cellLoc="center",
        bbox=[0.0, 0.02, 1.0, 0.31],
    )
    detail_table.auto_set_font_size(False)
    detail_table.set_fontsize(8.8)
    detail_table.scale(1.0, 1.15)

    for (row, col), cell in detail_table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#D1D5DB")
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")
        if row > 0 and col in (1, 2):
            text = cell.get_text().get_text()
            if text.startswith("+"):
                cell.set_facecolor("#DDEFE5")
            elif text.startswith("-"):
                cell.set_facecolor("#F3D9D8")

    _save(fig, output_path)


def plot_daily_hourly_comparison_table(title, subtitle, rows, output_path):
    """Optional per-universe table: metric | Daily | Hourly."""
    fig = _new_figure()
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.76])
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Kennzahl", "Daily", "Hourly"],
        colWidths=[0.44, 0.26, 0.26],
        cellLoc="center",
        bbox=[0.0, 0.08, 1.0, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#D1D5DB")
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.text(0.0, 0.96, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    _save(fig, output_path)


def plot_signal_stability_matrix(
    title,
    subtitle,
    selection_df,
    output_path,
    metric_lines=None,
    preliminary=False,
):
    """
    Plot 4: compact ranking matrix. Left side shows Top-K selection share;
    right side shows numeric stability columns without long bar annotations.
    """
    data = selection_df.copy().reset_index(drop=True)
    fig = _new_figure()
    ax_bar = fig.add_axes([0.10, 0.22, 0.37, 0.58])
    ax_table = fig.add_axes([0.50, 0.22, 0.44, 0.58])
    ax_table.axis("off")

    labels = data["ticker"].tolist()
    values = data["selection_share"].fillna(0.0).tolist()
    colors = [COLOR_POSITIVE if index == 0 and value > 0 else COLOR_STRATEGY for index, value in enumerate(values)]
    bars = ax_bar.barh(labels, values, color=colors)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Anteil in Top-K", fontsize=AXIS_LABEL_FONT_SIZE)
    ax_bar.set_xlim(0, 1)
    _style_axis(ax_bar)
    ax_bar.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=3, fontsize=ANNOTATION_FONT_SIZE)

    def fmt_number(value, pattern):
        if pd.isna(value):
            return "n. v."
        return pattern.format(value)

    table_rows = []
    for _, row in data.iterrows():
        table_rows.append(
            [
                fmt_number(row["average_rank"], "{:.1f}"),
                fmt_number(row["average_probability"], "{:.2f}"),
                str(int(row["buys"])),
                str(int(row["sells"])),
                fmt_number(row["average_holding_duration"], "{:.1f}"),
            ]
        )

    table = ax_table.table(
        cellText=table_rows,
        rowLabels=labels,
        colLabels=["Ø Rang", "Ø p", "Kauf", "Verkauf", "Ø Halt"],
        cellLoc="center",
        rowLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#D1D5DB")
            cell.set_text_props(weight="bold")
        elif row == 1:
            cell.set_facecolor("#E8F3ED")

    title_text = title
    if preliminary:
        title_text += "  (vorlaeufig)"
    fig.suptitle(title_text, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    ax_bar.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")
    if metric_lines:
        _footer_text(fig, metric_lines)

    _save(fig, output_path)


def plot_comparison_table(title, subtitle, rows, output_path):
    """Comparison dashboard table across all universes."""
    fig = _new_figure()
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
    ax.axis("off")

    columns = [
        "Universum",
        "Benchmark",
        "Strategie",
        "Benchmark",
        "Outperf.",
        "Trades",
        "Beste Aktie",
        "Ø p",
    ]
    table_rows = []
    cell_colors = []
    for row in rows:
        diff = row.get("difference")
        table_rows.append(
            [
                row.get("universe_title", "n. v."),
                row.get("benchmark", "n. v."),
                row.get("strategy_return", "n. v."),
                row.get("benchmark_return", "n. v."),
                row.get("difference_label", "n. v."),
                row.get("trades", "n. v."),
                row.get("best_ticker", "n. v."),
                row.get("average_probability", "n. v."),
            ]
        )
        diff_color = "#F3F4F6"
        if diff is not None:
            diff_color = "#DDEFE5" if diff >= 0 else "#F3D9D8"
        cell_colors.append(["#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", diff_color, "#FFFFFF", "#FFFFFF", "#FFFFFF"])

    table = ax.table(
        cellText=table_rows,
        cellColours=cell_colors,
        colLabels=columns,
        cellLoc="center",
        bbox=[0.0, 0.05, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#D1D5DB")
            cell.set_text_props(weight="bold")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.text(0.0, 0.95, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    _save(fig, output_path)


def plot_universe_results_table(title, subtitle, table_df, output_path):
    """Large final comparison table with one row per universe."""
    display_columns = [
        "Universum",
        "Benchmark",
        "Top-K",
        "Backtest",
        "Bench.",
        "BT-Outp.",
        "Sharpe",
        "MaxDD",
        "Daily",
        "Daily-Outp.",
        "Hourly",
        "Hourly-Outp.",
        "Trades",
    ]

    fig = _new_figure()
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.80])
    ax.axis("off")

    text_rows = table_df[display_columns].values.tolist()
    colors = [["#FFFFFF"] * len(display_columns) for _ in text_rows]

    highlight_columns = ["BT-Outp.", "Sharpe", "MaxDD", "Daily-Outp.", "Hourly-Outp."]
    for column in highlight_columns:
        if column not in table_df.columns:
            continue
        raw_column = f"__raw_{column}"
        if raw_column not in table_df.columns:
            continue
        raw_values = pd.to_numeric(table_df[raw_column], errors="coerce")
        if raw_values.dropna().empty:
            continue
        best_index = raw_values.idxmax()
        if column == "MaxDD":
            best_index = raw_values.idxmax()  # closest to zero is best because values are negative
        col_index = display_columns.index(column)
        row_position = list(table_df.index).index(best_index)
        colors[row_position][col_index] = "#DDEFE5"

    table = ax.table(
        cellText=text_rows,
        cellColours=colors,
        colLabels=display_columns,
        colWidths=[0.11, 0.07, 0.07, 0.085, 0.075, 0.085, 0.07, 0.08, 0.075, 0.09, 0.075, 0.095, 0.065],
        cellLoc="center",
        bbox=[0.0, 0.04, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.30)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#D1D5DB")
            cell.set_text_props(weight="bold")
        if col == 0:
            cell.set_text_props(ha="left")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.text(0.0, 0.96, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    _save(fig, output_path)


def plot_signal_probability_distribution(
    title,
    subtitle,
    probabilities,
    output_path,
    metric_lines,
    preliminary=False,
):
    """
    Optional Plot 6 (model interpretation only): histogram of every
    predicted probability the model produced for this universe's signal
    history - shows whether the model mostly outputs confident (far from
    0.50) or uncertain (near 0.50) predictions.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.09, 0.22, 0.86, 0.60])

    ax.hist(probabilities, bins=20, range=(0, 1), color=COLOR_STRATEGY, edgecolor="white")
    ax.axvline(0.5, color=COLOR_NEGATIVE, linewidth=1.5, linestyle="--", label="p = 0.50 (Zufall)")
    ax.set_xlabel("Vorhergesagte Wahrscheinlichkeit", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Anzahl Beobachtungen", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    title_text = title
    if preliminary:
        title_text += "  (vorlaeufig)"
    fig.suptitle(title_text, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _footer_text(fig, metric_lines)
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


# ---------------------------------------------------------------------------
# Cross-universe comparison helpers (generate_cross_universe_analysis.py).
# Every universe gets one fixed identity color (UNIVERSE_COLORS) so it is
# visually consistent across every chart in this section.
# ---------------------------------------------------------------------------

def plot_grouped_bars_multi_metric(
    title,
    subtitle,
    universe_titles,
    universe_names,
    series,
    output_path,
    value_label,
    value_format=lambda value: f"{value:.2f}",
    missing_mask=None,
):
    """
    One group of bars per metric, one colored bar per universe within each
    group (fixed UNIVERSE_COLORS). `series` is an ordered dict/list of
    (metric_label, [value_per_universe]). Missing values (None) are drawn as
    hatched "n/a" bars, never as 0.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.08, 0.24, 0.88, 0.58])

    metric_labels = [label for label, _ in series]
    n_universes = len(universe_names)
    n_metrics = len(metric_labels)
    width = 0.8 / max(n_universes, 1)
    group_positions = list(range(n_metrics))

    all_values = []
    for universe_index, universe_name in enumerate(universe_names):
        values = []
        colors = []
        labels = []
        for metric_index, (_, metric_values) in enumerate(series):
            value = metric_values[universe_index]
            is_missing = value is None or (missing_mask and missing_mask[metric_index][universe_index])
            if is_missing:
                values.append(0.0)
                colors.append("#DDDDDD")
                labels.append("n/a")
            else:
                values.append(value)
                colors.append(UNIVERSE_COLORS.get(universe_name, COLOR_NEUTRAL))
                labels.append(value_format(value))
        all_values.extend(values)
        offset = (universe_index - (n_universes - 1) / 2) * width
        positions = [g + offset for g in group_positions]
        bars = ax.bar(positions, values, width=width, color=colors, label=universe_titles[universe_index])
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8, rotation=90)

    ax.set_xticks(group_positions)
    ax.set_xticklabels(metric_labels)
    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_ylabel(value_label, fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)

    # Headroom above the tallest bar (in both directions if negative values
    # are present) so rotated bar-value labels never collide with the
    # legend, regardless of the metric's scale in a given chart.
    positive_values = [v for v in all_values if v >= 0]
    negative_values = [v for v in all_values if v < 0]
    top = max(positive_values) * 1.45 if positive_values else None
    bottom = min(negative_values) * 1.3 if negative_values else None
    if top is not None or bottom is not None:
        ax.set_ylim(bottom=bottom, top=top)

    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, ncol=2)

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _save(fig, output_path)


def plot_heatmap(
    title,
    subtitle,
    row_labels,
    col_labels,
    matrix,
    output_path,
    value_format=lambda value: f"{value:+.1%}",
    cmap_name="RdYlGn",
    center_zero=True,
):
    """Diverging heatmap (e.g. Top-K return per universe x K)."""
    fig = _new_figure()
    ax = fig.add_axes([0.22, 0.18, 0.68, 0.62])

    array = np.array(matrix, dtype=float)
    if center_zero:
        limit = np.nanmax(np.abs(array)) if array.size else 1.0
        limit = limit if limit > 0 else 1.0
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = np.nanmin(array), np.nanmax(array)

    image = ax.imshow(array, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(labelsize=TICK_FONT_SIZE)

    for row_index in range(array.shape[0]):
        for col_index in range(array.shape[1]):
            value = array[row_index, col_index]
            if np.isnan(value):
                text = "n/a"
            else:
                text = value_format(value)
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=ANNOTATION_FONT_SIZE, color="#111111")

    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _save(fig, output_path)


def plot_multi_line_index(
    title,
    subtitle,
    series_by_universe,
    output_path,
    metric_lines=None,
    y_label="Index (Start = 100)",
):
    """
    Overlays one cumulative-index line per universe (fixed UNIVERSE_COLORS).
    `series_by_universe` is a list of (universe_title, universe_name, dates, index_values).
    """
    fig = _new_figure()
    ax = fig.add_axes([0.08, 0.20, 0.88, 0.64])

    for universe_title, universe_name, dates, index_values in series_by_universe:
        ax.plot(
            dates, index_values, color=UNIVERSE_COLORS.get(universe_name, COLOR_NEUTRAL),
            linewidth=2.0, label=universe_title,
        )

    ax.axhline(100, color="#999999", linewidth=0.8)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, frameon=True)

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    if metric_lines:
        _footer_text(fig, metric_lines)
    _save(fig, output_path)


def plot_ranking_bars(
    title,
    subtitle,
    universe_titles,
    universe_names,
    scores,
    output_path,
    metric_lines=None,
):
    """Horizontal bar chart of the final composite ranking score, best on top."""
    fig = _new_figure()
    ax = fig.add_axes([0.26, 0.16, 0.62, 0.68])

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    labels = [universe_titles[i] for i in order]
    values = [scores[i] for i in order]
    colors = [UNIVERSE_COLORS.get(universe_names[i], COLOR_NEUTRAL) for i in order]

    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Gesamtscore (hoeher = besser)", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=ANNOTATION_FONT_SIZE)

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    if metric_lines:
        _footer_text(fig, metric_lines)
    _save(fig, output_path)


def plot_backtest_comparison(
    title,
    subtitle,
    universe_titles,
    universe_names,
    strategy_returns,
    benchmark_returns,
    best_top_k,
    max_drawdowns,
    output_path,
):
    """
    Grouped bar chart: Strategie-Rendite (bestes Top-K) vs. Benchmark-Rendite,
    je Universum. Bestes Top-K und Max Drawdown werden als kompakte
    Annotation unter jeder Gruppe angezeigt statt als eigene Balken.
    """
    fig = _new_figure()
    ax = fig.add_axes([0.08, 0.24, 0.88, 0.6])

    n = len(universe_titles)
    x = np.arange(n)
    width = 0.36

    strategy_pct = [v * 100 if v is not None else 0.0 for v in strategy_returns]
    benchmark_pct = [v * 100 if v is not None else 0.0 for v in benchmark_returns]

    bars_strategy = ax.bar(x - width / 2, strategy_pct, width, color=COLOR_STRATEGY, label="Strategie (bestes Top-K)")
    bars_benchmark = ax.bar(x + width / 2, benchmark_pct, width, color=COLOR_BENCHMARK, label="Benchmark")

    ax.bar_label(bars_strategy, labels=[f"{v:+.1f}%" for v in strategy_pct], padding=3, fontsize=ANNOTATION_FONT_SIZE)
    ax.bar_label(bars_benchmark, labels=[f"{v:+.1f}%" for v in benchmark_pct], padding=3, fontsize=ANNOTATION_FONT_SIZE)

    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(universe_titles)
    ax.set_ylabel("Rendite (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    top = max(strategy_pct + benchmark_pct + [0]) * 1.28 + 1
    bottom = min(strategy_pct + benchmark_pct + [0]) * 1.35 - 1
    ax.set_ylim(bottom=bottom, top=top)

    for i, (top_k, max_dd) in enumerate(zip(best_top_k, max_drawdowns)):
        top_k_text = f"Top-{int(top_k)}" if top_k is not None else "n. v."
        dd_text = f"{max_dd:.1%}" if max_dd is not None else "n. v."
        ax.text(
            i, bottom + (top - bottom) * 0.02, f"bestes K: {top_k_text}\nMaxDD: {dd_text}",
            ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE, color="#444444",
        )

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _save(fig, output_path)


def plot_paper_trading_comparison_table(title, subtitle, rows, output_path):
    """
    Vergleichstabelle: Universum, Benchmark, Daily-Rendite, Daily-Outperformance,
    Hourly-Rendite, Hourly-Outperformance, Gewinn/Verlust Daily USD,
    Gewinn/Verlust Hourly USD, Trades Daily, Trades Hourly.

    Bester Wert je numerischer Spalte dezent gruen, schwaechster dezent rot
    hervorgehoben (nur unter bereits vorhandenen/berechneten Werten - 'n. v.'
    wird nie in die Best/Worst-Bewertung einbezogen).
    """
    fig = _new_figure(figsize=(12.8, 6.4))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.axis("off")

    columns = [
        "Universum", "Benchmark",
        "Daily-\nRendite", "Daily-\nOutperf.",
        "Hourly-\nRendite", "Hourly-\nOutperf.",
        "G/V Daily\n(USD)", "G/V Hourly\n(USD)",
        "Trades\nDaily", "Trades\nHourly",
    ]
    # Nur Performance-Kennzahlen werden best/worst-markiert. Trades sind eine
    # Aktivitaetskennzahl ohne "besser/schlechter"-Richtung und werden bewusst
    # nicht hervorgehoben.
    numeric_keys = [
        "daily_return", "daily_outperformance",
        "hourly_return", "hourly_outperformance",
        "daily_pnl_usd", "hourly_pnl_usd",
    ]
    column_widths = [0.14, 0.09, 0.10, 0.11, 0.10, 0.11, 0.11, 0.12, 0.06, 0.06]

    def best_worst(key, higher_is_better=True):
        values = [(i, r[key]) for i, r in enumerate(rows) if r.get(key) is not None]
        if len(values) < 2:
            return None, None
        best_i = max(values, key=lambda t: t[1])[0] if higher_is_better else min(values, key=lambda t: t[1])[0]
        worst_i = min(values, key=lambda t: t[1])[0] if higher_is_better else max(values, key=lambda t: t[1])[0]
        return best_i, worst_i

    highlight = {key: best_worst(key) for key in numeric_keys}

    table_rows = []
    cell_colors = []
    for i, r in enumerate(rows):
        table_rows.append(
            [
                r.get("universe_title", "n. v."),
                r.get("benchmark", "n. v."),
                r.get("daily_return_label", "n. v."),
                r.get("daily_outperformance_label", "n. v."),
                r.get("hourly_return_label", "n. v."),
                r.get("hourly_outperformance_label", "n. v."),
                r.get("daily_pnl_usd_label", "n. v."),
                r.get("hourly_pnl_usd_label", "n. v."),
                r.get("daily_trades_label", "n. v."),
                r.get("hourly_trades_label", "n. v."),
            ]
        )
        row_colors = ["#FFFFFF", "#FFFFFF"]
        for key in numeric_keys:
            best_i, worst_i = highlight[key]
            if best_i == i:
                row_colors.append("#DDEFE5")
            elif worst_i == i:
                row_colors.append("#F3D9D8")
            else:
                row_colors.append("#FFFFFF")
        row_colors.extend(["#FFFFFF", "#FFFFFF"])  # Trades Daily / Trades Hourly: nicht hervorgehoben
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=table_rows, cellColours=cell_colors, colLabels=columns,
        cellLoc="center", bbox=[0.0, 0.05, 1.0, 0.82], colWidths=column_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        if row == 0:
            cell.set_facecolor("#D1D5DB")
            cell.set_text_props(weight="bold", ha="center" if col != 0 else "left")

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.text(0.0, 0.95, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONT_SIZE, color="#444444")

    _save(fig, output_path)
