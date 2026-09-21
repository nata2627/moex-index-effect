"""Выгрузка дневной истории индекса МосБиржи.

Индекс нужен дважды: как рыночная доходность R_mt в модели рынка и как
торговый календарь — список дат, в которые биржа вообще работала.

Результат: data/raw/index_prices.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, moex


def main() -> None:
    moex.DATA_RAW.mkdir(parents=True, exist_ok=True)

    frame = moex.index_history(config.MARKET_INDEX, config.QUOTES_START, config.QUOTES_END)
    frame["TRADEDATE"] = pd.to_datetime(frame["TRADEDATE"])
    frame = frame.sort_values("TRADEDATE").reset_index(drop=True)

    keep = ["TRADEDATE", "SECID", "OPEN", "HIGH", "LOW", "CLOSE", "VALUE", "CAPITALIZATION"]
    frame = frame[[column for column in keep if column in frame.columns]]
    frame.to_parquet(moex.DATA_RAW / "index_prices.parquet", index=False)

    print(f"{config.MARKET_INDEX}: {len(frame)} торговых дней, "
          f"{frame['TRADEDATE'].min().date()} … {frame['TRADEDATE'].max().date()}")

    # Разрывы в календаре видны сразу и важны: весной 2022 торги акциями
    # останавливались почти на месяц, и окна оценки это задевает.
    gaps = frame["TRADEDATE"].diff().dt.days
    for position in gaps[gaps > 7].index:
        print(f"  перерыв {int(gaps[position])} дней: "
              f"{frame['TRADEDATE'][position - 1].date()} → {frame['TRADEDATE'][position].date()}")


if __name__ == "__main__":
    main()
