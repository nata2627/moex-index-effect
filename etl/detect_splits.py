"""Поиск дроблений акций и выпуск скорректированных котировок.

Запускается после fetch_quotes.py. Печатает разбор всех резких скачков цены
с вердиктом по каждому — этот вывод предназначен для проверки глазами,
автоматике здесь доверять нельзя.

Результат: data/interim/splits.csv (разбор кандидатов),
data/processed/quotes_adjusted.parquet (котировки с поправкой).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import moex, splits


def main() -> None:
    frames = []
    for name in ("quotes_events", "quotes_controls"):
        path = moex.DATA_RAW / f"{name}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            print(f"Нет файла {path.name} — пропущен")

    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.drop_duplicates(["SECID", "TRADEDATE"])
    print(f"Котировки: {len(quotes)} строк, {quotes['SECID'].nunique()} бумаг")

    # Доходность индекса нужна, чтобы отличить дробление от общерыночного
    # обвала: цена бумаги может упасть ровно вдвое вместе со всем рынком.
    index_prices = pd.read_parquet(moex.DATA_RAW / "index_prices.parquet")
    index_prices = index_prices.sort_values("TRADEDATE")
    market_returns = index_prices.set_index("TRADEDATE")["CLOSE"].pct_change()

    detected = splits.detect_splits(quotes, market_returns)
    moex.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    detected.to_csv(moex.DATA_INTERIM / "splits.csv", index=False)

    confirmed = detected[detected["is_split"]]
    print(f"\nСкачков цены свыше {splits.JUMP_THRESHOLD:.0%}: {len(detected)}")
    print(f"Из них признано дроблениями: {len(confirmed)}\n")

    print("=== Признаны дроблениями ===")
    if confirmed.empty:
        print("  нет")
    for row in confirmed.itertuples():
        print(f"  {row.SECID:7s} {row.TRADEDATE.date()}  "
              f"{row.prev_close:>10.2f} → {row.CLOSE:<10.2f} "
              f"коэффициент {row.split_ratio:g}, оборот в рублях "
              f"{row.value_ratio:.2f} от обычного")

    print("\n=== Отклонены (реальные движения цены) ===")
    rejected = detected[~detected["is_split"]]
    for row in rejected.itertuples():
        print(f"  {row.SECID:7s} {row.TRADEDATE.date()}  "
              f"{row.price_change:+7.1%}  {row.verdict}")

    registry = splits.load_registry(moex.PROJECT_ROOT / splits.REGISTRY_PATH)
    audit = splits.audit_registry(detected, registry)
    print("\n=== Сверка гипотез детектора с реестром ===")
    for row in audit.itertuples():
        ratio = row.shares_ratio if pd.notna(row.shares_ratio) else row.split_ratio
        print(f"  {row.secid:7s} {row.date.date()}  коэффициент {ratio:g}  {row.status}")

    adjusted = splits.adjust_for_splits(quotes, registry)
    moex.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    adjusted.to_parquet(moex.DATA_PROCESSED / "quotes_adjusted.parquet", index=False)
    print(f"\nСкорректированные котировки сохранены: {len(adjusted)} строк")


if __name__ == "__main__":
    main()
