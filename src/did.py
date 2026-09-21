"""Разность разностей по ликвидности.

По цене индексный эффект может быть слабым: рынок способен заранее выкупить
предстоящий спрос фондов. По ликвидности он обычно заметнее — индексные
фонды не только покупают бумагу один раз, но и продолжают ею торговать,
ребалансируя портфель вслед за индексом.

Оценивается разность разностей:

    эффект = (после - до) в тесте - (после - до) в контроле

Метрики: оборот в рублях, число сделок, доля дней без торгов и прокси спреда
(HIGH - LOW) / CLOSE. Первые две берутся в логарифмах — распределены они с
тяжёлым правым хвостом, и в уровнях оценка определялась бы двумя крупнейшими
бумагами.

Окрестность самого события (плюс-минус десять торговых дней) из обоих
периодов исключена. В эти дни проходит сама сделка индексных фондов, и оборот
подскакивает механически. Нас интересует, изменилась ли ликвидность надолго,
а всплеск в день ребалансировки показывается отдельно и в оценку не входит.

Стандартные ошибки кластеризуются по бумаге: наблюдения одной бумаги в
соседние дни зависимы, и без кластеризации ошибка занижается в разы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src import events as events_module

# Периоды сравнения в торговых днях относительно события.
PRE_PERIOD = (-60, -11)
POST_PERIOD = (11, 60)

# Окрестность события, исключаемая из обоих периодов.
EVENT_NEIGHBOURHOOD = (-10, 10)

METRICS = {
    "log_value": "логарифм оборота в рублях",
    "log_trades": "логарифм числа сделок",
    "spread_proxy": "прокси спреда (HIGH-LOW)/CLOSE",
    "no_trade": "день без торгов",
}


def build_liquidity_panel(matches: pd.DataFrame,
                          quotes: pd.DataFrame,
                          calendar: pd.DatetimeIndex,
                          pre_period: tuple[int, int] = PRE_PERIOD,
                          post_period: tuple[int, int] = POST_PERIOD) -> pd.DataFrame:
    """Дневная панель «бумага x день» для тестовых и контрольных бумаг.

    Дни без торгов не выбрасываются, а помечаются: отсутствие сделок — это
    само по себе показатель ликвидности, и молча исключать такие дни значило
    бы систематически завышать ликвидность неликвидных бумаг.
    """
    by_ticker = {ticker: frame.set_index("TRADEDATE")
                 for ticker, frame in quotes.groupby("SECID")}

    pieces = []
    for pair in matches.itertuples():
        offsets = events_module.relative_day_index(calendar, pair.event_date)
        window = offsets[(offsets >= pre_period[0]) & (offsets <= post_period[1])]

        for ticker, treated in ((pair.ticker, 1), (pair.control, 0)):
            frame = by_ticker.get(ticker)
            if frame is None:
                continue
            piece = frame.reindex(window.index)
            panel = pd.DataFrame({
                "event_id": pair.event_id,
                "event_type": pair.event_type,
                "event_date": pair.event_date,
                "ticker": ticker,
                "treated": treated,
                "t": window.to_numpy(),
                "TRADEDATE": window.index,
                "value": piece["VALUE"].to_numpy(),
                "numtrades": piece["NUMTRADES"].to_numpy(),
                "high": piece["HIGH"].to_numpy(),
                "low": piece["LOW"].to_numpy(),
                "close": piece["CLOSE"].to_numpy(),
            })
            pieces.append(panel)

    panel = pd.concat(pieces, ignore_index=True)

    panel["no_trade"] = (panel["close"].isna() |
                         (panel["value"].fillna(0) <= 0)).astype(float)
    panel["log_value"] = np.log(panel["value"].where(panel["value"] > 0))
    panel["log_trades"] = np.log(panel["numtrades"].where(panel["numtrades"] > 0))
    panel["spread_proxy"] = ((panel["high"] - panel["low"]) /
                             panel["close"].where(panel["close"] > 0))

    panel["period"] = np.where(panel["t"].between(*pre_period), "до",
                       np.where(panel["t"].between(*post_period), "после", None))
    panel["post"] = (panel["period"] == "после").astype(float)
    # Уникальная метка бумаги внутри события: одна и та же контрольная бумага
    # может обслуживать несколько событий, и смешивать эти наблюдения нельзя.
    panel["unit"] = panel["event_id"] + "|" + panel["ticker"]
    return panel


def estimate_did(panel: pd.DataFrame, metric: str) -> dict:
    """Оценка разности разностей с фиксированными эффектами.

    Спецификация: фиксированные эффекты бумаги-в-событии снимают постоянные
    различия между тестом и контролем, фиксированные эффекты относительного
    дня — общую динамику вокруг события. Интересующий коэффициент — при
    произведении treated x post.
    """
    frame = panel[panel["period"].notna() & panel[metric].notna()].copy()
    # Метрика без вариации оценке не поддаётся: у ликвидных бумаг дней без
    # торгов не бывает вовсе, и регрессия на константу вырождается.
    if (frame.empty or frame["treated"].nunique() < 2 or
            frame[metric].nunique() < 2):
        return {"metric": metric, "metric_name": METRICS.get(metric, metric),
                "coefficient": np.nan, "se": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n": len(frame), "n_clusters": frame["ticker"].nunique(),
                "note": "нет вариации метрики: оценка невозможна"}

    frame["interaction"] = frame["treated"] * frame["post"]
    model = smf.ols(f"{metric} ~ interaction + C(unit) + C(t)", data=frame)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": frame["ticker"]})

    return {
        "metric": metric,
        "metric_name": METRICS.get(metric, metric),
        "coefficient": result.params["interaction"],
        "se": result.bse["interaction"],
        "t_stat": result.tvalues["interaction"],
        "p_value": result.pvalues["interaction"],
        "ci_low": result.conf_int().loc["interaction", 0],
        "ci_high": result.conf_int().loc["interaction", 1],
        "n": int(result.nobs),
        "n_clusters": frame["ticker"].nunique(),
    }


def parallel_trends_test(panel: pd.DataFrame, metric: str,
                         pre_period: tuple[int, int] = PRE_PERIOD,
                         n_bins: int = 5) -> dict:
    """Формальная проверка параллельности трендов до события.

    Предсобытийный период разбивается на равные отрезки, и для каждого
    оценивается разница между тестом и контролем относительно последнего
    предсобытийного отрезка. Если предпосылка верна, все эти разницы
    статистически неотличимы от нуля — это и проверяет совместный тест.

    Важно, что непрохождение теста не означает «нужно подобрать контроль
    получше»: оно означает, что разность разностей на этих данных
    неприменима, и об этом следует сказать прямо.
    """
    frame = panel[panel["t"].between(*pre_period) & panel[metric].notna()].copy()
    if frame.empty or frame[metric].nunique() < 2 or frame["treated"].nunique() < 2:
        return {"metric": metric, "metric_name": METRICS.get(metric, metric),
                "f_stat": np.nan, "p_value": np.nan, "n": len(frame), "bins": [],
                "note": "нет вариации метрики: проверка невозможна"}

    edges = np.linspace(pre_period[0], pre_period[1] + 1, n_bins + 1)
    frame["bin"] = pd.cut(frame["t"], bins=edges, right=False, labels=False)
    frame = frame[frame["bin"].notna()]
    frame["bin"] = frame["bin"].astype(int)

    # Базой служит последний предсобытийный отрезок — ближайший к событию.
    base = frame["bin"].max()
    terms = []
    for bin_index in sorted(frame["bin"].unique()):
        if bin_index == base:
            continue
        column = f"lead_{bin_index}"
        frame[column] = frame["treated"] * (frame["bin"] == bin_index).astype(float)
        terms.append(column)

    if not terms:
        return {"metric": metric, "p_value": np.nan, "n": len(frame), "bins": []}

    formula = f"{metric} ~ {' + '.join(terms)} + C(unit) + C(t)"
    result = smf.ols(formula, data=frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["ticker"]})

    hypothesis = ", ".join(f"{term} = 0" for term in terms)
    joint = result.f_test(hypothesis)

    return {
        "metric": metric,
        "metric_name": METRICS.get(metric, metric),
        "f_stat": float(np.squeeze(joint.fvalue)),
        "p_value": float(np.squeeze(joint.pvalue)),
        "n": int(result.nobs),
        "n_clusters": frame["ticker"].nunique(),
        "bins": [{"term": term,
                  "coefficient": result.params[term],
                  "se": result.bse[term],
                  "p_value": result.pvalues[term]} for term in terms],
    }


def event_time_means(panel: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Средние значения метрики по дням окна отдельно для теста и контроля.

    Нужны для графика параллельных трендов: формальный тест говорит, есть ли
    расхождение, а график — как оно выглядит.
    """
    grouped = panel.groupby(["t", "treated"])[metric].agg(["mean", "std", "count"])
    return grouped.reset_index()
