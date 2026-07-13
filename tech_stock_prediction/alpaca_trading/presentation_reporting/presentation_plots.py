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
    (nothing recomputed/estimated). Difference and average number of
    positions are annotated per group.
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

    bars_strategy = ax.bar(
        [i - width / 2 for i in index_positions], strategy_pct,
        width, color=COLOR_STRATEGY, label="Strategie-Rendite",
    )
    bars_buy_hold = ax.bar(
        [i + width / 2 for i in index_positions], buy_hold_pct,
        width, color=COLOR_BENCHMARK, label="Buy-and-Hold-Rendite",
    )

    ax.set_xticks(index_positions)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_ylabel("Rendite (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    _style_axis(ax)

    # Headroom above the tallest bar so the difference labels and the legend
    # never overlap a bar or each other, regardless of how tall the tallest
    # bar in a given universe turns out to be.
    tallest_bar = max(strategy_pct + buy_hold_pct + [0])
    ax.set_ylim(top=tallest_bar * 1.3 if tallest_bar > 0 else 1)

    diff_labels = [f"{diff:+.1%}" for diff in differences]
    ax.bar_label(bars_strategy, labels=diff_labels, padding=3, fontsize=ANNOTATION_FONT_SIZE)

    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    title_text = title
    if preliminary:
        title_text += "  (vorlaeufig)"
    fig.suptitle(title_text, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=SUBTITLE_FONT_SIZE, color="#444444", loc="left")

    _footer_text(fig, metric_lines)
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
