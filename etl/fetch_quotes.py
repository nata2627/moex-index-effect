"""Выгрузка дневных котировок бумаг в режиме TQBR.

По умолчанию качает бумаги, участвующие в событиях индекса; с флагом
``--controls`` — пул кандидатов в контрольную группу.

Цены отдаются биржей БЕЗ поправки на дробления акций: скрипт сохраняет их
как есть, корректировка — отдельный шаг (etl/detect_splits.py).

Результат: data/raw/quotes_events.parquet или quotes_controls.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, events, moex

KEEP_COLUMNS = ["TRADEDATE", "SECID", "OPEN", "LOW", "HIGH", "CLOSE",
                "VOLUME", "VALUE", "NUMTRADES", "WAPRICE"]


def event_tickers() -> list[str]:
    composition = pd.read_parquet(moex.DATA_RAW / "index_composition.parquet")
    index_prices = pd.read_parquet(moex.DATA_RAW / "index_prices.parquet")
    calendar = pd.DatetimeIndex(index_prices["TRADEDATE"])
    frame = events.build_events(composition, calendar,
                                config.EVENT_WINDOW_START, config.EVENT_WINDOW_END)
    return sorted(frame["ticker"].unique())


def control_tickers() -> list[str]:
    """Пул кандидатов: все бумаги TQBR, кроме участвовавших в событиях.

    Бумаги, входившие в индекс когда-либо, из пула не исключаются здесь —
    отбор по «никогда не был в индексе» делается на этапе матчинга, где
    видна дата конкретного события.
    """
    securities = pd.read_parquet(moex.DATA_RAW / "tqbr_securities.parquet")
    excluded = set(event_tickers())
    return sorted(set(securities["SECID"]) - excluded)


def fetch_many(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for number, ticker in enumerate(tickers, start=1):
        try:
            frame = moex.share_history(ticker, config.QUOTES_START, config.QUOTES_END)
        except RuntimeError as error:
            print(f"  [{number}/{len(tickers)}] {ticker}: ошибка — {error}")
            continue
        if frame.empty:
            print(f"  [{number}/{len(tickers)}] {ticker}: нет данных")
            continue
        frame = frame[[column for column in KEEP_COLUMNS if column in frame.columns]]
        frames.append(frame)
        print(f"  [{number}/{len(tickers)}] {ticker}: {len(frame)} дней "
              f"({frame['TRADEDATE'].min()} … {frame['TRADEDATE'].max()})")

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    quotes = pd.concat(frames, ignore_index=True)
    quotes["TRADEDATE"] = pd.to_datetime(quotes["TRADEDATE"])
    return quotes.sort_values(["SECID", "TRADEDATE"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", action="store_true",
                        help="качать пул кандидатов в контроль вместо бумаг событий")
    args = parser.parse_args()

    tickers = control_tickers() if args.controls else event_tickers()
    name = "quotes_controls" if args.controls else "quotes_events"
    print(f"Выгрузка {len(tickers)} бумаг → {name}.parquet")

    quotes = fetch_many(tickers)
    moex.DATA_RAW.mkdir(parents=True, exist_ok=True)
    quotes.to_parquet(moex.DATA_RAW / f"{name}.parquet", index=False)
    print(f"Итого: {len(quotes)} строк, {quotes['SECID'].nunique()} бумаг")


if __name__ == "__main__":
    main()
