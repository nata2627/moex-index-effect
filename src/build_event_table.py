"""Сводная таблица событий с диагностикой доступности данных.

Запускается после ETL. Показывает по каждому событию, хватает ли данных на
окно оценки и на окно события, и отмечает случаи, где исключение из индекса
совпало с уходом бумаги с торгов.

Результат: data/processed/events.parquet и печать таблицы.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, events, moex


def main() -> None:
    composition = pd.read_parquet(moex.DATA_RAW / "index_composition.parquet")
    index_prices = pd.read_parquet(moex.DATA_RAW / "index_prices.parquet")
    quotes = pd.read_parquet(moex.DATA_RAW / "quotes_events.parquet")
    calendar = pd.DatetimeIndex(index_prices["TRADEDATE"])

    frame = events.build_events(composition, calendar,
                                config.EVENT_WINDOW_START, config.EVENT_WINDOW_END)
    frame = events.window_coverage(frame, quotes, calendar,
                                   config.ESTIMATION_WINDOW, config.EVENT_WINDOW)

    frame["est_fill"] = frame["est_days_with_price"] / frame["est_days_expected"]
    frame["evt_fill"] = frame["evt_days_with_price"] / frame["evt_days_expected"]

    # Пригодность: хватает истории на оценку беты и есть чем мерить последействие.
    frame["ok_estimation"] = frame["est_days_with_price"] >= config.MIN_ESTIMATION_DAYS
    frame["ok_post_event"] = frame["last_quote_t"] >= config.MIN_POST_EVENT_DAYS
    frame["eligible"] = frame["ok_estimation"] & frame["ok_post_event"]

    moex.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(moex.DATA_PROCESSED / "events.parquet", index=False)

    columns = ["ticker", "event_type", "event_date", "est_days_with_price",
               "evt_days_with_price", "last_quote", "last_quote_t",
               "ok_estimation", "ok_post_event", "eligible"]
    view = frame[columns].copy()
    view["event_date"] = view["event_date"].dt.date
    view["last_quote"] = view["last_quote"].dt.date
    print(view.to_string(index=False))
    print(f"\nПригодны по формальным критериям: {frame['eligible'].sum()} из {len(frame)}")


if __name__ == "__main__":
    main()
