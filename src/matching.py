"""Подбор контрольной группы для событий индекса.

Задача контроля — показать, что случилось бы с бумагой, не войди она в
индекс. Значит контрольная бумага должна быть похожа на тестовую по всему,
кроме самого события.

Матчинг: **точное совпадение по отрасли плюс ближайший сосед по размеру и
ликвидности**. Признаки измеряются до события, на окне [-60, -11] торговых
дней — достаточно близко, чтобы описывать бумагу актуально, и достаточно
далеко, чтобы не захватить реакцию на объявление о ребалансировке.

Почему не propensity score. PSM оценивает вероятность попадания в индекс по
наблюдаемым признакам, и для этого нужна модель, обученная на событиях. У нас
их 26 — логистическая регрессия на такой выборке даст оценки с доверительными
интервалами шире самого эффекта, а сама процедура станет непрозрачной.
Точный матчинг по отрасли с ближайшим соседом проще, проверяем глазами и при
малой выборке устойчивее. Качество проверяется балансом ковариат, а не верой.

Кто годится в контроль. Наивное правило «бумага никогда не входила в IMOEX»
оставляет всего 72 кандидата, причём в финансах четыре, а в строительстве
два — при таком пуле «ближайший сосед» перестаёт быть близким. Поэтому
основное правило мягче: бумага не должна быть в индексе на дату события и не
должна иметь собственного включения или исключения в окрестности этого
события. Бумага, вошедшая в индекс двумя годами позже, про текущее событие
ничего не знает и контролем быть может. Строгий вариант остаётся как
проверка устойчивости.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import events as events_module

# Окно измерения признаков матчинга, торговых дней до события.
FEATURE_WINDOW = (-60, -11)

# Насколько далеко от даты события у кандидата не должно быть своих событий,
# торговых дней. Слева шире: нужно чистое окно оценки.
CONTAMINATION_WINDOW = (-250, 60)

MIN_FEATURE_DAYS = 20


def compute_features(tickers: list[str],
                     quotes: pd.DataFrame,
                     securities: pd.DataFrame,
                     calendar: pd.DatetimeIndex,
                     event_date: pd.Timestamp,
                     feature_window: tuple[int, int] = FEATURE_WINDOW) -> pd.DataFrame:
    """Размер и ликвидность бумаг на окне до события.

    Капитализация считается как число акций из справочника, умноженное на
    медианную цену окна. Число акций берётся текущее — исторического ISS не
    отдаёт, и при допэмиссиях это приближение (ограничение отмечено в README).
    С дроблениями рассогласования нет: цены к моменту расчёта уже приведены к
    той же шкале, в которой указано текущее число акций.
    """
    offsets = events_module.relative_day_index(calendar, event_date)
    window_dates = offsets[offsets.between(*feature_window)].index

    frame = quotes[quotes["SECID"].isin(tickers) &
                   quotes["TRADEDATE"].isin(window_dates) &
                   quotes["CLOSE"].notna()]

    aggregated = frame.groupby("SECID").agg(
        price=("CLOSE", "median"),
        value=("VALUE", "median"),
        numtrades=("NUMTRADES", "median"),
        days=("CLOSE", "size"),
    ).reset_index()

    aggregated = aggregated[aggregated["days"] >= MIN_FEATURE_DAYS]

    issue_size = securities.set_index("SECID")["ISSUESIZE"]
    aggregated["issue_size"] = aggregated["SECID"].map(issue_size)
    aggregated["market_cap"] = aggregated["issue_size"] * aggregated["price"]

    # Логарифмы: размер и ликвидность распределены с тяжёлым правым хвостом,
    # и без логарифма расстояние определялось бы одним Сбербанком.
    aggregated["log_cap"] = np.log(aggregated["market_cap"].where(aggregated["market_cap"] > 0))
    aggregated["log_value"] = np.log(aggregated["value"].where(aggregated["value"] > 0))
    aggregated["log_trades"] = np.log(aggregated["numtrades"].where(aggregated["numtrades"] > 0))
    return aggregated


def eligible_controls(candidate_pool: list[str],
                      composition: pd.DataFrame,
                      all_events: pd.DataFrame,
                      calendar: pd.DatetimeIndex,
                      event_date: pd.Timestamp,
                      strict: bool = False) -> list[str]:
    """Кандидаты, пригодные быть контролем для события на дату ``event_date``.

    ``strict=True`` оставляет только бумаги, никогда не входившие в индекс.
    """
    ever_in_index = set(composition["ticker"])
    if strict:
        return [t for t in candidate_pool if t not in ever_in_index]

    offsets = events_module.relative_day_index(calendar, event_date)
    left = offsets[offsets >= CONTAMINATION_WINDOW[0]].index.min()
    right = offsets[offsets <= CONTAMINATION_WINDOW[1]].index.max()

    # В индексе на дату события — контролем быть не может.
    in_index_now = set(composition.loc[
        (composition["date_from"] <= event_date) &
        (composition["date_till"] >= event_date), "ticker"])

    # Есть собственное событие рядом — тоже не может.
    nearby = set(all_events.loc[all_events["event_date"].between(left, right), "ticker"])

    return [t for t in candidate_pool
            if t not in in_index_now and t not in nearby]


def match_controls(events: pd.DataFrame,
                   quotes: pd.DataFrame,
                   securities: pd.DataFrame,
                   sectors: pd.DataFrame,
                   composition: pd.DataFrame,
                   all_events: pd.DataFrame,
                   calendar: pd.DatetimeIndex,
                   candidate_pool: list[str],
                   n_neighbours: int = 1,
                   strict: bool = False,
                   feature_window: tuple[int, int] = FEATURE_WINDOW) -> pd.DataFrame:
    """Для каждого события подобрать ``n_neighbours`` контрольных бумаг.

    Возвращает таблицу пар событие-контроль с расстоянием и значениями
    признаков обеих бумаг — баланс ковариат проверяется по ней напрямую.
    """
    sector_map = sectors.groupby("ticker")["sector_name"].apply(set).to_dict()

    pairs = []
    for event in events.itertuples():
        own_sectors = sector_map.get(event.ticker, set())
        if not own_sectors:
            continue

        pool = eligible_controls(candidate_pool, composition, all_events,
                                 calendar, event.event_date, strict=strict)
        # Точное совпадение по отрасли: у бумаги и контроля должен быть общий
        # отраслевой индекс.
        pool = [t for t in pool if sector_map.get(t, set()) & own_sectors]
        if not pool:
            continue

        features = compute_features(pool + [event.ticker], quotes, securities,
                                    calendar, event.event_date, feature_window)
        features = features.dropna(subset=["log_cap", "log_trades"])
        if event.ticker not in set(features["SECID"]):
            continue

        target = features[features["SECID"] == event.ticker].iloc[0]
        controls = features[features["SECID"] != event.ticker]
        if controls.empty:
            continue

        # Стандартизация по пулу кандидатов этого события: расстояние должно
        # измеряться в единицах разброса того множества, из которого выбираем.
        distances = np.zeros(len(controls))
        for column in ("log_cap", "log_trades"):
            spread = controls[column].std()
            if not np.isfinite(spread) or spread == 0:
                spread = 1.0
            distances += ((controls[column] - target[column]) / spread) ** 2
        controls = controls.assign(distance=np.sqrt(distances))

        for rank, row in enumerate(
                controls.nsmallest(n_neighbours, "distance").itertuples(), start=1):
            pairs.append({
                "event_id": event.event_id,
                "ticker": event.ticker,
                "event_type": event.event_type,
                "event_date": event.event_date,
                "control": row.SECID,
                "rank": rank,
                "distance": row.distance,
                "target_log_cap": target["log_cap"],
                "control_log_cap": row.log_cap,
                "target_log_trades": target["log_trades"],
                "control_log_trades": row.log_trades,
                "n_pool": len(controls),
            })

    return pd.DataFrame(pairs)


def covariate_balance(matches: pd.DataFrame) -> pd.DataFrame:
    """Баланс признаков между тестовой и контрольной группой.

    Стандартизованная разность средних — принятая мера качества матчинга;
    значения по модулю до 0.1 считаются хорошим балансом, до 0.25 — приемлемым.
    """
    rows = []
    for label, target_column, control_column in (
            ("логарифм капитализации", "target_log_cap", "control_log_cap"),
            ("логарифм числа сделок", "target_log_trades", "control_log_trades")):
        target = matches[target_column]
        control = matches[control_column]
        pooled = np.sqrt((target.var() + control.var()) / 2)
        rows.append({
            "признак": label,
            "среднее в тесте": target.mean(),
            "среднее в контроле": control.mean(),
            "стандартизованная разность": (target.mean() - control.mean()) / pooled,
        })
    return pd.DataFrame(rows)
