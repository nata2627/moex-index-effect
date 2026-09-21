"""Сборка событий включения/исключения из истории состава индекса.

ISS отдаёт членство интервалами ``[date_from, date_till]``. Из интервала
получается до двух событий:

* **включение** — бумага впервые появляется в базе индекса;
* **исключение** — бумага из базы выбывает.

Дата события определена симметрично: ``t = 0`` — первый торговый день, в
котором действует новый статус бумаги. Для включения это ``date_from``
(первый день в индексе), для исключения — первый торговый день после
``date_till`` (последний день в индексе). При таком определении сделка
индексного фонда приходится на закрытие ``t = -1`` в обоих случаях, и
включения с исключениями сравнимы напрямую.

Ловушка: у бумаг, которые в индексе сейчас, ``date_till`` равен дате
последнего доступного среза состава. Это не исключение, а край данных.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INCLUSION = "включение"
EXCLUSION = "исключение"


def build_events(composition: pd.DataFrame,
                 trading_days: pd.DatetimeIndex,
                 window_start: str | pd.Timestamp,
                 window_end: str | pd.Timestamp) -> pd.DataFrame:
    """Список событий индекса в окне ``[window_start, window_end]``.

    Parameters
    ----------
    composition
        Таблица членства: ticker, date_from, date_till.
    trading_days
        Торговый календарь (даты торгов индекса), отсортированный.
    window_start, window_end
        Границы окна отбора событий по дате события.

    Returns
    -------
    DataFrame с колонками event_id, ticker, event_type, event_date,
    index_from, index_till, still_in_index.
    """
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    trading_days = pd.DatetimeIndex(trading_days).sort_values()

    # Край данных: бумаги с таким date_till из индекса не выбывали.
    last_snapshot = composition["date_till"].max()

    records = []
    for row in composition.itertuples():
        # Включение: датой события считаем первый день бумаги в индексе.
        records.append({
            "ticker": row.ticker,
            "event_type": INCLUSION,
            "event_date": row.date_from,
            "index_from": row.date_from,
            "index_till": row.date_till,
            "still_in_index": row.date_till >= last_snapshot,
        })

        if row.date_till >= last_snapshot:
            continue  # бумага всё ещё в индексе — исключения не было

        exit_date = _next_trading_day(trading_days, row.date_till)
        if exit_date is None:
            continue  # исключение на самом краю календаря: дня «после» ещё нет
        records.append({
            "ticker": row.ticker,
            "event_type": EXCLUSION,
            "event_date": exit_date,
            "index_from": row.date_from,
            "index_till": row.date_till,
            "still_in_index": False,
        })

    events = pd.DataFrame.from_records(records)
    events = events[events["event_date"].between(window_start, window_end)]
    events = events.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    events.insert(0, "event_id", events["ticker"] + "_" +
                  events["event_type"].str[:3] + "_" +
                  events["event_date"].dt.strftime("%Y%m%d"))
    return events


def _next_trading_day(trading_days: pd.DatetimeIndex,
                      date: pd.Timestamp) -> pd.Timestamp | None:
    """Первый торговый день строго после ``date``."""
    position = trading_days.searchsorted(date, side="right")
    if position >= len(trading_days):
        return None
    return trading_days[position]


def relative_day_index(trading_days: pd.DatetimeIndex,
                       event_date: pd.Timestamp) -> pd.Series:
    """Номера торговых дней относительно события: t = 0 в день события.

    Если сам день события нерабочий (такого быть не должно, но данные бывают
    неполными), нулём считается ближайший следующий торговый день.
    """
    trading_days = pd.DatetimeIndex(trading_days).sort_values()
    zero_position = trading_days.searchsorted(event_date, side="left")
    offsets = np.arange(len(trading_days)) - zero_position
    return pd.Series(offsets, index=trading_days, name="t")


def data_coverage(events: pd.DataFrame,
                  quotes: pd.DataFrame,
                  trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Сколько торговых дней с котировками есть у каждого события до и после.

    Считаются не календарные дни, а дни с фактической ценой закрытия —
    именно они ограничивают окно оценки и окно события.
    """
    trading_days = pd.DatetimeIndex(trading_days).sort_values()
    by_ticker = {ticker: frame for ticker, frame in quotes.groupby("SECID")}

    rows = []
    for event in events.itertuples():
        frame = by_ticker.get(event.ticker)
        if frame is None or frame.empty:
            rows.append({"event_id": event.event_id, "days_before": 0,
                         "days_after": 0, "first_quote": pd.NaT,
                         "last_quote": pd.NaT, "gap_days": np.nan})
            continue

        dates = pd.DatetimeIndex(frame.loc[frame["CLOSE"].notna(), "TRADEDATE"]).sort_values()
        before = dates[dates < event.event_date]
        after = dates[dates >= event.event_date]

        # Пропуски внутри окна: торговый день есть в календаре индекса, а
        # котировки бумаги нет (приостановка торгов, неликвид).
        calendar_window = trading_days[(trading_days >= dates.min()) &
                                       (trading_days <= dates.max())]
        gap_days = len(calendar_window) - len(dates)

        rows.append({
            "event_id": event.event_id,
            "days_before": len(before),
            "days_after": len(after),
            "first_quote": dates.min(),
            "last_quote": dates.max(),
            "gap_days": gap_days,
        })

    return events.merge(pd.DataFrame(rows), on="event_id", how="left")


def window_coverage(events: pd.DataFrame,
                    quotes: pd.DataFrame,
                    trading_days: pd.DatetimeIndex,
                    estimation_window: tuple[int, int],
                    event_window: tuple[int, int]) -> pd.DataFrame:
    """Фактическая наполненность окна оценки и окна события по каждому событию.

    Календарь берётся по индексу: в дни остановки торгов ISS отдаёт строки с
    пустой ценой, и считать их торговыми нельзя. «Наполненность» — доля дней
    окна, в которые у бумаги есть цена закрытия.
    """
    trading_days = pd.DatetimeIndex(trading_days).sort_values()
    quoted_dates = {
        ticker: set(frame.loc[frame["CLOSE"].notna(), "TRADEDATE"])
        for ticker, frame in quotes.groupby("SECID")
    }

    rows = []
    for event in events.itertuples():
        offsets = relative_day_index(trading_days, event.event_date)
        available = quoted_dates.get(event.ticker, set())

        record = {"event_id": event.event_id}
        for label, (left, right) in (("est", estimation_window),
                                      ("evt", event_window)):
            wanted = offsets[(offsets >= left) & (offsets <= right)].index
            have = [date for date in wanted if date in available]
            record[f"{label}_days_expected"] = right - left + 1
            record[f"{label}_days_in_calendar"] = len(wanted)
            record[f"{label}_days_with_price"] = len(have)

        # Последний день с ценой относительно события: показывает, дожила ли
        # бумага до конца окна или торги прекратились.
        if available:
            last_quote = max(available)
            position = offsets.reindex([last_quote]).iloc[0]
            record["last_quote"] = last_quote
            record["last_quote_t"] = position
        else:
            record["last_quote"] = pd.NaT
            record["last_quote_t"] = np.nan

        rows.append(record)

    return events.merge(pd.DataFrame(rows), on="event_id", how="left")
