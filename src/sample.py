"""Отбор аналитической выборки событий.

Из 42 событий окна для анализа цены годятся далеко не все, и причины отсева
содержательные, а не технические. Правила здесь зафиксированы кодом, чтобы
выборку можно было воспроизвести и оспорить.

**Исключения, вызванные уходом бумаги с торгов, отбрасываются.** Индексный
эффект — это реакция на вынужденную сделку индексных фондов. Если бумага
одновременно теряет листинг, уезжает в другую юрисдикцию или её эмитент
лишается лицензии, то оценка смешивает индексный эффект с реакцией на само
корпоративное событие, и разделить их данными нельзя. Признак воспроизводим:
бумага перестала торговаться раньше конца периода данных. Под него попадают
и те случаи, где данных на окно физически нет (POGR, GLTR, FIVE, TCSG, AGRO,
YNDX), и те, где окно закрывается, но причина исключения корпоративная
(HHRU, DSKY, FIXP, QIWI, POLY).

**Включения без окна оценки отбрасываются только из анализа цены.** Недавние
IPO и новые тикеры не дают оценить бету, и CAR по ним посчитать нельзя. Но
разность разностей по ликвидности беты не требует — ей нужен лишь период
«до». Поэтому пригодность проверяется раздельно.

**Возвраты после редомициляции помечаются флагом.** Пять российских компаний
сменили юрисдикцию и вернулись на биржу под новым тикером. Формально это
включение новой бумаги, содержательно — возвращение известного эмитента,
которого рынок ждал. Эффект здесь может быть слабее просто потому, что
включение предсказуемо. Из выборки эти события не удаляются, но результат
показывается с ними и без них.
"""

from __future__ import annotations

import pandas as pd

# Пары «ушедший тикер -> вернувшийся тикер» при смене юрисдикции эмитента.
# Это знание о корпоративных действиях, а не вывод из котировок, поэтому
# задано явно; функция validate_redomicile_pairs проверяет, что котировки
# ему не противоречат.
REDOMICILE_PAIRS = {
    "TCSG": "T",      # ТКС Холдинг -> Т-Технологии
    "YNDX": "YDEX",   # Yandex N.V. -> МКПАО «Яндекс»
    "FIVE": "X5",     # X5 Retail Group -> ПАО «Корпоративный центр ИКС 5»
    "HHRU": "HEAD",   # HeadHunter Group -> МКПАО «Хэдхантер»
    "AGRO": "RAGR",   # Ros Agro -> МКПАО «Русагро»
}

# Минимум наблюдений для оценки рыночной модели. 120 дневных наблюдений дают
# стандартную ошибку беты около 0.1 при типичном для наших бумаг R^2 — меньше
# уже не позволяет отличить бету от единицы.
MIN_ESTIMATION_DAYS = 120

# Сколько торговых дней бумага должна прожить после события, чтобы окно
# события [-30, +60] закрылось целиком.
MIN_POST_EVENT_DAYS = 60

# Требования для разности разностей по ликвидности: столько дней с торгами
# нужно по обе стороны от события.
LIQUIDITY_WINDOW = 60
MIN_LIQUIDITY_DAYS = 30


def validate_redomicile_pairs(quotes: pd.DataFrame,
                              pairs: dict[str, str] = REDOMICILE_PAIRS) -> pd.DataFrame:
    """Проверка, что котировки согласуются с заявленными парами.

    Ожидание: старый тикер перестаёт торговаться, новый начинает — и первые
    торги нового не предшествуют последним торгам старого более чем на месяц.
    """
    traded = quotes.loc[quotes["CLOSE"].notna()].groupby("SECID")["TRADEDATE"]
    span = traded.agg(["min", "max"])

    rows = []
    for old, new in pairs.items():
        if old not in span.index or new not in span.index:
            rows.append({"old": old, "new": new, "consistent": False,
                         "note": "нет котировок по одному из тикеров"})
            continue
        last_old = span.loc[old, "max"]
        first_new = span.loc[new, "min"]
        gap_days = (first_new - last_old).days
        rows.append({
            "old": old, "new": new,
            "last_old": last_old, "first_new": first_new,
            "gap_days": gap_days,
            "consistent": gap_days > -31,
            "note": "новый тикер начал торговаться после ухода старого"
                    if gap_days > -31 else "периоды торгов пересекаются — пара под вопросом",
        })
    return pd.DataFrame(rows)


def classify_events(events: pd.DataFrame,
                    quotes: pd.DataFrame,
                    parameters: pd.DataFrame,
                    data_end: pd.Timestamp) -> pd.DataFrame:
    """Разметка событий признаками пригодности и причинами отсева."""
    frame = events.merge(parameters[["event_id", "n_estimation"]], on="event_id", how="left")

    quoted = quotes.loc[quotes["CLOSE"].notna()]
    last_trade = quoted.groupby("SECID")["TRADEDATE"].max()
    frame["last_trade"] = frame["ticker"].map(last_trade)

    # Бумага не дожила до конца периода данных — значит ушла с торгов.
    frame["delisted"] = frame["last_trade"] < data_end

    frame["is_redomicile"] = (frame["ticker"].isin(REDOMICILE_PAIRS.keys()) |
                              frame["ticker"].isin(REDOMICILE_PAIRS.values()))

    frame["has_estimation"] = frame["n_estimation"].fillna(0) >= MIN_ESTIMATION_DAYS
    frame["has_post_event"] = frame["last_quote_t"] >= MIN_POST_EVENT_DAYS

    frame["liquidity_days_before"] = _count_days(frame, quoted, -LIQUIDITY_WINDOW, -1)
    frame["liquidity_days_after"] = _count_days(frame, quoted, 1, LIQUIDITY_WINDOW)
    frame["has_liquidity_window"] = (
        (frame["liquidity_days_before"] >= MIN_LIQUIDITY_DAYS) &
        (frame["liquidity_days_after"] >= MIN_LIQUIDITY_DAYS))

    frame["in_price_sample"] = (~frame["delisted"] & frame["has_estimation"] &
                                frame["has_post_event"])
    frame["in_liquidity_sample"] = ~frame["delisted"] & frame["has_liquidity_window"]

    frame["drop_reason"] = frame.apply(_drop_reason, axis=1)
    return frame


def _count_days(events: pd.DataFrame, quoted: pd.DataFrame,
                left: int, right: int) -> pd.Series:
    """Число дней с ценой в окне [left, right] календарных торговых дней бумаги."""
    counts = []
    by_ticker = {ticker: pd.DatetimeIndex(frame["TRADEDATE"]).sort_values()
                 for ticker, frame in quoted.groupby("SECID")}
    for event in events.itertuples():
        dates = by_ticker.get(event.ticker)
        if dates is None:
            counts.append(0)
            continue
        position = dates.searchsorted(event.event_date, side="left")
        start = max(0, position + left) if left < 0 else position + left
        stop = position + right + 1
        counts.append(max(0, len(dates[start:stop])))
    return pd.Series(counts, index=events.index)


def _drop_reason(row: pd.Series) -> str:
    if row["delisted"]:
        return "бумага ушла с торгов: событие смешано с корпоративным"
    if not row["has_estimation"]:
        return f"нет окна оценки: {int(row['n_estimation'] or 0)} наблюдений"
    if not row["has_post_event"]:
        return "окно события не закрывается"
    return ""
