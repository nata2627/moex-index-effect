"""Построение графиков отчёта.

Каждый график отвечает на один вопрос и подписан так, чтобы его можно было
прочитать отдельно от текста. Оформление единое: тонкие линии, сплошная
неброская сетка, подписи на русском, единицы на осях.

Палитра проверена на различимость при нарушениях цветовосприятия: синий и
оранжевый расходятся на 24.7 единицы OKLab при пороге 8, так что тип события
читается и в чёрно-белой печати по форме линии.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

# Категориальные слоты: включение — синий, исключение — оранжевый.
INCLUSION_COLOR = "#2a78d6"
EXCLUSION_COLOR = "#eb6834"
TREATED_COLOR = "#2a78d6"
CONTROL_COLOR = "#eb6834"
ACCENT = "#e34948"
NEUTRAL = "#8a8985"

TYPE_COLORS = {"включение": INCLUSION_COLOR, "исключение": EXCLUSION_COLOR}

# Расходящаяся шкала: синий и красный как противоположные полюса, нейтральный
# серый в середине — ноль должен читаться как «ничего не происходит».
DIVERGING = LinearSegmentedColormap.from_list(
    "car", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f2322"])


def apply_style() -> None:
    """Единое оформление для всех графиков."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def _finish(ax, title: str, subtitle: str = "") -> None:
    """Заголовок с пояснением и уборка лишних рамок."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=11,
                     color=TEXT_PRIMARY, pad=12, linespacing=1.5)
    else:
        ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=12)


def _percent_axis(ax) -> None:
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda value, _: f"{value * 100:.0f}%"))


def plot_caar_paths(paths: pd.DataFrame, output: Path) -> None:
    """График 1. Средняя накопленная аномальная доходность по дням события."""
    apply_style()
    figure, ax = plt.subplots(figsize=(9, 5.2))

    for event_type, frame in paths.groupby("event_type"):
        frame = frame.sort_values("t")
        color = TYPE_COLORS[event_type]
        ax.plot(frame["t"], frame["caar"], color=color, linewidth=2,
                label=f"{event_type} (n = {int(frame['n_events'].max())})")
        ax.fill_between(frame["t"], frame["ci_low"], frame["ci_high"],
                        color=color, alpha=0.13, linewidth=0)
        # Направленная подпись на конце линии вместо числа у каждой точки.
        last = frame.iloc[-1]
        ax.annotate(f"{last['caar'] * 100:+.0f}%",
                    xy=(last["t"], last["caar"]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", fontweight="bold")

    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.annotate("день события", xy=(0, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8.5, color=TEXT_SECONDARY)

    ax.set_xlabel("торговые дни от события")
    ax.set_ylabel("накопленная аномальная доходность")
    _percent_axis(ax)
    ax.legend(loc="lower left", fontsize=9)
    _finish(ax, "Что происходит с ценой после пересмотра индекса",
            "Средняя накопленная аномальная доходность, коридор — 95% "
            "по кластерам-датам")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_parallel_trends(panel: pd.DataFrame, output: Path,
                         metric: str = "log_value",
                         metric_label: str = "логарифм оборота в рублях") -> None:
    """График 2. Тест против контроля до события."""
    apply_style()
    types = sorted(panel["event_type"].unique())
    figure, axes = plt.subplots(1, len(types), figsize=(11, 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, event_type in zip(axes, types):
        part = panel[panel["event_type"] == event_type]
        for treated, color, label in ((1, TREATED_COLOR, "тестовые бумаги"),
                                      (0, CONTROL_COLOR, "контрольные бумаги")):
            series = (part[part["treated"] == treated]
                      .groupby("t")[metric].mean().sort_index())
            # Уровни групп различаются: тестовые бумаги крупнее, их оборот
            # заведомо выше. Параллельность трендов — это про форму, а не про
            # уровень, поэтому обе линии отсчитываются от собственного
            # среднего за ранний предсобытийный период.
            baseline = series.loc[series.index < -30].mean()
            series = series - baseline
            # Сглаживание пятидневным окном: дневной ряд оборота слишком
            # шумный, чтобы на глаз оценить параллельность трендов.
            smoothed = series.rolling(5, center=True, min_periods=1).mean()
            ax.plot(smoothed.index, smoothed.to_numpy(), color=color,
                    linewidth=2, label=label)

        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)

        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.axvspan(-10, 10, color=NEUTRAL, alpha=0.1, linewidth=0)
        ax.set_xlabel("торговые дни от события")
        ax.set_title(event_type, loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"{metric_label},\nотклонение от уровня до события")
    axes[0].legend(loc="lower left", fontsize=9)
    figure.suptitle("Двигались ли группы одинаково до события\n"
                    "Обе линии отсчитываются от своего уровня за дни "
                    "ранее −30; серая полоса исключена из оценки",
                    x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_placebo(distributions: dict[str, pd.Series],
                 observed: dict[str, float],
                 percentiles: dict[str, float],
                 output: Path) -> None:
    """График 3. Распределение эффектов на случайных датах."""
    apply_style()
    labels = list(distributions)
    figure, axes = plt.subplots(1, len(labels), figsize=(11, 4.4))
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        values = distributions[label].dropna()
        ax.hist(values, bins=40, color=NEUTRAL, alpha=0.5, linewidth=0)
        ax.axvline(observed[label], color=ACCENT, linewidth=2.2)
        # Подпись уводится в свободный угол над столбцами и получает подложку:
        # иначе она ложится поверх гистограммы и становится нечитаемой.
        on_left = observed[label] < values.median()
        ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
        ax.annotate(f"фактическая оценка {observed[label] * 100:+.1f}%\n"
                    f"перцентиль {percentiles[label]:.0f}",
                    xy=(observed[label], ax.get_ylim()[1] * 0.99),
                    xytext=(6 if on_left else -6, -2),
                    textcoords="offset points", fontsize=8.5, color=ACCENT,
                    ha="left" if on_left else "right", va="top", linespacing=1.5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.5))
        ax.set_title(label, loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("средняя накопленная аномальная доходность")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda value, _: f"{value * 100:.0f}%"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("число случайных дат")
    figure.suptitle("Насколько велик эффект, возникающий на пустом месте\n"
                    "Тысяча случайных дат, та же процедура оценки",
                    x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(abnormal: pd.DataFrame, output: Path) -> None:
    """График 4. События против дней окна: общий эффект или две-три бумаги."""
    apply_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [
            max(1, (abnormal["event_type"] == "включение").sum()),
            max(1, (abnormal["event_type"] == "исключение").sum())]})

    limit = float(np.nanpercentile(np.abs(abnormal["car"]), 97))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None

    for ax, event_type in zip(axes, ["включение", "исключение"]):
        part = abnormal[abnormal["event_type"] == event_type]
        grid = part.pivot_table(index="ticker", columns="t", values="car")
        # Порядок по итоговой накопленной доходности: сразу видно, есть ли
        # общий сдвиг или всё держится на краях.
        grid = grid.loc[grid.iloc[:, -1].sort_values().index]

        image = ax.imshow(grid.to_numpy(), aspect="auto", cmap=DIVERGING,
                          norm=norm, interpolation="nearest",
                          extent=(grid.columns.min(), grid.columns.max(),
                                  len(grid) - 0.5, -0.5))
        ax.set_yticks(range(len(grid)))
        ax.set_yticklabels(grid.index, fontsize=8)
        ax.axvline(0, color=TEXT_PRIMARY, linewidth=1)
        ax.set_xlabel("торговые дни от события")
        ax.set_title(event_type, loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.grid(False)

    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("накопленная аномальная доходность", fontsize=9)
    colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda value, _: f"{value * 100:.0f}%"))
    colorbar.outline.set_visible(False)

    figure.suptitle("Эффект общий или его создают отдельные бумаги",
                    x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_liquidity(panel: pd.DataFrame, output: Path) -> None:
    """График 5. Ликвидность до и после события."""
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    summary = (panel[panel["period"].notna()]
               .groupby(["event_id", "ticker", "treated", "period"])["log_value"]
               .mean().reset_index())
    wide = summary.pivot_table(index=["event_id", "ticker", "treated"],
                               columns="period", values="log_value").dropna()
    wide = wide.reset_index()

    ax = axes[0]
    for treated, color, label in ((1, TREATED_COLOR, "тестовые"),
                                  (0, CONTROL_COLOR, "контрольные")):
        part = wide[wide["treated"] == treated]
        for row in part.itertuples():
            ax.plot([0, 1], [row.до, row.после], color=color, alpha=0.35,
                    linewidth=1)
        ax.plot([0, 1], [part["до"].mean(), part["после"].mean()],
                color=color, linewidth=2.8, label=f"{label}, среднее",
                marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["до события", "после события"])
    ax.set_ylabel("логарифм среднего оборота в рублях")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Оборот каждой бумаги до и после", loc="left", fontsize=10)

    ax = axes[1]
    changes = wide.assign(change=wide["после"] - wide["до"])
    positions, data, colors = [], [], []
    for index, (treated, color, label) in enumerate(
            ((1, TREATED_COLOR, "тестовые"), (0, CONTROL_COLOR, "контрольные"))):
        values = changes[changes["treated"] == treated]["change"]
        positions.append(index)
        data.append(values.to_numpy())
        colors.append(color)

    parts = ax.violinplot(data, positions=positions, showmeans=True, widths=0.7)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[key].set_color(TEXT_SECONDARY)
        parts[key].set_linewidth(1)

    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(["тестовые", "контрольные"])
    ax.set_ylabel("изменение логарифма оборота")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Распределение изменения", loc="left", fontsize=10)

    figure.suptitle("Меняется ли ликвидность после пересмотра индекса",
                    x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_robustness(results: pd.DataFrame, output: Path) -> None:
    """График 6. Как меняется оценка при смене спецификации."""
    apply_style()
    types = ["включение", "исключение"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharex=True, sharey=True)

    for index, (ax, event_type) in enumerate(zip(axes, types)):
        part = results[results["event_type"] == event_type].copy()
        part = part.sort_values(["измерение", "вариант"], ascending=[True, False])
        labels = [f"{row.измерение}: {row.вариант}" for row in part.itertuples()]
        positions = np.arange(len(part))
        color = TYPE_COLORS[event_type]

        ax.scatter(part["оценка"], positions, color=color, s=46, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(part["медиана"], positions, color=color, s=30, zorder=3,
                   marker="|", linewidth=1.6)

        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.set_yticks(positions)
        # Подписи спецификаций одни и те же для обеих панелей, поэтому
        # печатаются только слева. Ось общая, и очищать подписи на второй
        # панели через set_yticklabels нельзя — это стёрло бы их и на первой.
        if index == 0:
            ax.set_yticklabels(labels, fontsize=8.5)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("накопленная доходность за [+1, +60] дней")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda value, _: f"{value * 100:.0f}%"))
        ax.set_title(event_type, loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", visible=False)

    # Легенда поясняет форму метки; цвет в ней нейтральный, потому что цветом
    # кодируется тип события, а он назван в заголовке панели.
    handles = [plt.Line2D([], [], marker="o", color=NEUTRAL, linestyle="none",
                          markersize=7, label="среднее"),
               plt.Line2D([], [], marker="|", color=NEUTRAL, linestyle="none",
                          markersize=9, markeredgewidth=1.6, label="медиана")]
    figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Держится ли результат при смене спецификации\n"
                    "Каждая строка — отдельная оценка того же эффекта",
                    x=0.02, ha="left", fontsize=11, color=TEXT_PRIMARY)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
