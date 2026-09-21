"""Выгрузка истории состава индекса МосБиржи и отраслевых индексов.

Состав IMOEX — ключевые данные проекта: из интервалов членства собираются
события включения и исключения. Отраслевые индексы нужны для матчинга
контрольной группы по сектору, справочник TQBR — для пула кандидатов.

Результат: data/raw/index_composition.parquet, sector_composition.parquet,
tqbr_securities.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import moex

# Отраслевые индексы МосБиржи. Бумага может входить в несколько, поэтому при
# матчинге сектор определяется как множество, а не как одно значение.
SECTOR_INDICES = {
    "MOEXOG": "нефть и газ",
    "MOEXEU": "электроэнергетика",
    "MOEXTL": "телекоммуникации",
    "MOEXMM": "металлы и добыча",
    "MOEXFN": "финансы",
    "MOEXCN": "потребительский сектор",
    "MOEXCH": "химия и нефтехимия",
    "MOEXTN": "транспорт",
    "MOEXIT": "информационные технологии",
    "MOEXRE": "строительство",
}


def main() -> None:
    moex.DATA_RAW.mkdir(parents=True, exist_ok=True)

    imoex = moex.index_composition("IMOEX")
    imoex.to_parquet(moex.DATA_RAW / "index_composition.parquet", index=False)
    print(f"IMOEX: {len(imoex)} интервалов членства, "
          f"{imoex['date_from'].min().date()} … {imoex['date_till'].max().date()}")

    sectors = []
    for index_id, name in SECTOR_INDICES.items():
        try:
            frame = moex.index_composition(index_id)
        except (RuntimeError, KeyError) as error:
            print(f"  {index_id}: пропущен ({error})")
            continue
        frame["sector_name"] = name
        sectors.append(frame)
        print(f"  {index_id} ({name}): {len(frame)} бумаг")

    sector_frame = pd.concat(sectors, ignore_index=True)
    sector_frame.to_parquet(moex.DATA_RAW / "sector_composition.parquet", index=False)
    print(f"Отраслевые индексы: {len(sector_frame)} записей, "
          f"{sector_frame['ticker'].nunique()} уникальных бумаг")

    securities = moex.tqbr_securities()
    securities.to_parquet(moex.DATA_RAW / "tqbr_securities.parquet", index=False)
    print(f"Справочник TQBR: {len(securities)} бумаг")


if __name__ == "__main__":
    main()
