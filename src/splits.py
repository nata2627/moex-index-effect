"""Обнаружение дроблений акций и корректировка котировок.

Московская биржа отдаёт исторические цены без поправки на дробления, а сами
коэффициенты через открытый ISS API не публикуются — их приходится выводить
из котировок. Наивный порог «скачок цены больше 30%» даёт список кандидатов,
но разделить в нём дробления и реальные обвалы он не может: 24 февраля 2022
обыкновенные акции Мечела упали на 46.9%, и это отношение цен неотличимо от
дробления 1:2.

Разделяют их четыре признака, проверяемые вместе:

1. **Отношение цен кратно осмысленному коэффициенту** (1:2, 1:10, 100:1 …).
   Необходимое условие, но далеко не достаточное — см. пример выше.
2. **Рынок в этот день стоял на месте.** Дробление — решение одного эмитента,
   индекс на него не реагирует. Обвал же 24.02.2022 был общерыночным: IMOEX
   упал на треть вместе с бумагой. Это самый надёжный из признаков.
3. **Оборот в рублях не изменился.** Денег вложено столько же, изменилось
   лишь число бумаг, на которые они поделены. При реальной новости оборот
   взлетает в разы: у QIWI в день отзыва лицензии он вырос в 69 раз.
4. **Объём в штуках изменился обратно пропорционально цене.** Цена вдесятеро
   меньше — бумаг вдесятеро больше.

Признаки 3 и 4 считаются по медианам за окно до и после, а не по одному дню:
дробление часто приходится на возобновление торгов после паузы, и первый день
идёт с повышенным объёмом, который сбивает однодневное сравнение.

Корректировка: все котировки ДО даты дробления делятся на коэффициент, объём
в штуках умножается на него. После этого ряд непрерывен, и доходность на
стыке корректна. Оборот в рублях и число сделок не трогаем — они выражены в
единицах, не зависящих от номинала бумаги.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Порог скачка цены для попадания в кандидаты. Самое мелкое практически
# встречающееся дробление (1:2) даёт -50%, так что 30% берутся с запасом.
JUMP_THRESHOLD = 0.30

# Коэффициенты, которые рассматриваются. Дробление — цена падает (ratio > 1),
# обратное дробление (консолидация) — цена растёт (ratio < 1). Крайние
# значения не экзотика: у ВТБ в 2024 году была консолидация 5000:1.
PLAUSIBLE_RATIOS = [2, 3, 4, 5, 10, 20, 25, 50, 100, 1000, 5000]

# Допуск на отношение цен: дробление даёт точный коэффициент с точностью до
# дневного движения цены.
RATIO_TOLERANCE = 0.08

# Движение индекса, выше которого скачок считается общерыночным, а не
# корпоративным действием. 5% за день для IMOEX — уже заметное событие.
MARKET_MOVE_LIMIT = 0.05

# Во сколько раз медианный оборот в рублях может измениться. При дроблении не
# меняется вовсе; коридор оставляет запас на изменение интереса к бумаге.
VALUE_RATIO_BOUNDS = (0.4, 4.0)

# Коридор для отношения «фактический рост объёма в штуках / ожидаемый».
# Не симметричный допуск вокруг единицы: после дробления бумага дешевеет,
# становится доступнее мелкому инвестору, и объём растёт СИЛЬНЕЕ коэффициента —
# у Норникеля после дробления 1:100 объём вырос в 339 раз. А вот вырасти
# заметно слабее коэффициента он не может: это означало бы, что бумаг не
# прибавилось, то есть дробления не было.
VOLUME_RATIO_BOUNDS = (0.6, 6.0)

# Окно для медиан по обе стороны от кандидата, торговых дней.
MEDIAN_WINDOW = 10


def find_split_candidates(quotes: pd.DataFrame,
                          jump_threshold: float = JUMP_THRESHOLD) -> pd.DataFrame:
    """Дни с резким изменением цены — сырые кандидаты, до фильтрации.

    Сравнение идёт с последней ИМЕЮЩЕЙСЯ ценой, а не с предыдущей календарной
    датой: в дни остановки торгов ISS отдаёт строки с пустой ценой, а
    дробления часто приходятся как раз на возобновление торгов.
    """
    frame = quotes.loc[quotes["CLOSE"].notna()].sort_values(["SECID", "TRADEDATE"]).copy()
    grouped = frame.groupby("SECID", group_keys=False)

    frame["prev_close"] = grouped["CLOSE"].shift(1)
    frame["prev_date"] = grouped["TRADEDATE"].shift(1)
    frame["price_ratio"] = frame["prev_close"] / frame["CLOSE"]
    frame["price_change"] = frame["CLOSE"] / frame["prev_close"] - 1

    candidates = frame.loc[frame["price_change"].abs() > jump_threshold]
    return candidates[["SECID", "TRADEDATE", "prev_date", "prev_close", "CLOSE",
                       "price_change", "price_ratio"]].reset_index(drop=True)


def _nearest_plausible_ratio(ratio: float) -> float | None:
    """Ближайший осмысленный коэффициент дробления, если он достаточно близок."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    for candidate in PLAUSIBLE_RATIOS:
        for value in (float(candidate), 1.0 / candidate):
            if abs(ratio / value - 1) <= RATIO_TOLERANCE:
                return value
    return None


def _window_medians(series_frame: pd.DataFrame, date: pd.Timestamp,
                    window: int = MEDIAN_WINDOW) -> tuple[pd.Series, pd.Series]:
    """Медианы показателей за ``window`` торговых дней до и с даты ``date``."""
    before = series_frame.loc[series_frame["TRADEDATE"] < date].tail(window)
    after = series_frame.loc[series_frame["TRADEDATE"] >= date].head(window)
    columns = [c for c in ("CLOSE", "VALUE", "VOLUME") if c in series_frame.columns]
    return before[columns].median(), after[columns].median()


def classify_candidates(candidates: pd.DataFrame,
                        quotes: pd.DataFrame,
                        market_returns: pd.Series | None = None) -> pd.DataFrame:
    """Разделение кандидатов на дробления и реальные движения цены.

    ``market_returns`` — дневная доходность индекса, индексированная датой.
    Без неё проверка на общерыночное движение не выполняется, и обвалы вроде
    24.02.2022 могут быть приняты за дробления.

    Возвращает исходную таблицу с колонками ``split_ratio``, ``verdict`` и
    ``is_split``. Вердикт пишется словами: разбор рассчитан на проверку
    глазами, автоматике здесь доверять нельзя.
    """
    prepared = quotes.loc[quotes["CLOSE"].notna()].sort_values(["SECID", "TRADEDATE"])
    by_ticker = {ticker: frame for ticker, frame in prepared.groupby("SECID")}

    records = []
    for row in candidates.itertuples():
        record = {"split_ratio": np.nan, "market_return": np.nan,
                  "value_ratio": np.nan, "volume_ratio": np.nan}

        ratio = _nearest_plausible_ratio(row.price_ratio)
        if ratio is None:
            record["verdict"] = "движение цены: отношение не кратно коэффициенту дробления"
            records.append(record)
            continue

        if market_returns is not None:
            market_move = market_returns.get(row.TRADEDATE, np.nan)
            record["market_return"] = market_move
            if np.isfinite(market_move) and abs(market_move) > MARKET_MOVE_LIMIT:
                record["verdict"] = (f"движение цены: рынок в этот день сам сдвинулся "
                                     f"на {market_move:+.1%}")
                records.append(record)
                continue

        before, after = _window_medians(by_ticker[row.SECID], row.TRADEDATE)

        value_ratio = after.get("VALUE", np.nan) / before.get("VALUE", np.nan)
        volume_ratio = after.get("VOLUME", np.nan) / before.get("VOLUME", np.nan)
        record["value_ratio"] = value_ratio
        record["volume_ratio"] = volume_ratio

        low, high = VALUE_RATIO_BOUNDS
        if not (np.isfinite(value_ratio) and low <= value_ratio <= high):
            record["verdict"] = (f"движение цены: медианный оборот в рублях изменился "
                                 f"в {value_ratio:.1f} раза")
            records.append(record)
            continue

        volume_low, volume_high = VOLUME_RATIO_BOUNDS
        relative_volume = volume_ratio / ratio if np.isfinite(volume_ratio) else np.nan
        if not (np.isfinite(relative_volume) and volume_low <= relative_volume <= volume_high):
            record["verdict"] = ("движение цены: объём в штуках не вырос обратно "
                                 f"цене (ожидалось около x{ratio:g}, фактически x{volume_ratio:.1f})")
            records.append(record)
            continue

        record["split_ratio"] = ratio
        record["verdict"] = (f"дробление с коэффициентом {ratio:g}: цена /{ratio:g}, "
                             f"объём x{volume_ratio:.1f}, оборот {value_ratio:.2f} от прежнего")
        records.append(record)

    frame = pd.concat([candidates.reset_index(drop=True),
                       pd.DataFrame.from_records(records)], axis=1)
    frame["is_split"] = frame["split_ratio"].notna()
    return frame


def detect_splits(quotes: pd.DataFrame,
                  market_returns: pd.Series | None = None) -> pd.DataFrame:
    """Полный проход: поиск кандидатов и их классификация."""
    return classify_candidates(find_split_candidates(quotes), quotes, market_returns)


REGISTRY_PATH = "splits_confirmed.csv"


def load_registry(path: str | Path = REGISTRY_PATH) -> pd.DataFrame:
    """Реестр подтверждённых корпоративных действий.

    Детектор умеет только предлагать гипотезы: по котировкам невозможно
    надёжно отличить дробление от крупной дивидендной отсечки — у Лензолота
    в июле 2021 цена упала вдвое, объём вырос вчетверо, и от дробления 1:2
    это неотличимо. Поэтому источником истины служит реестр, каждая строка
    которого подтверждена изменением числа акций в справочнике бумаг, а не
    только формой скачка цены.
    """
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def audit_registry(candidates: pd.DataFrame,
                   registry: pd.DataFrame) -> pd.DataFrame:
    """Сверка гипотез детектора с реестром.

    Расхождения в обе стороны нормальны и должны быть объяснимы: детектор
    предлагает лишнее (дивидендные отсечки), а реестр содержит случаи,
    которых детектор не видит (бонусная эмиссия не даёт точного отношения цен).
    """
    proposed = candidates.loc[candidates["is_split"], ["SECID", "TRADEDATE", "split_ratio"]]
    proposed = proposed.rename(columns={"SECID": "secid", "TRADEDATE": "date"})

    merged = proposed.merge(registry, on=["secid", "date"], how="outer", indicator=True)
    status = {
        "both": "подтверждено: предложено детектором и внесено в реестр",
        "left_only": "гипотеза отклонена при проверке: в реестр не внесено",
        "right_only": "внесено в реестр вручную: детектор такой формы скачка не ловит",
    }
    merged["status"] = merged["_merge"].map(status)
    return merged.drop(columns="_merge")


def adjust_for_splits(quotes: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Привести котировки к шкале, действующей на конец ряда.

    Цены до корпоративного действия делятся на коэффициент изменения числа
    акций, объём в штуках умножается на него. Несколько действий по одной
    бумаге применяются последовательно и дают кумулятивный эффект.

    Оборот в рублях и число сделок не трогаем: они выражены в единицах, не
    зависящих от номинала бумаги, и корректировать их означало бы исказить
    ровно тот показатель ликвидности, который мы измеряем.
    """
    if registry.empty:
        return quotes.copy()

    frame = quotes.copy()
    price_columns = [c for c in ("OPEN", "LOW", "HIGH", "CLOSE", "WAPRICE")
                     if c in frame.columns]

    for action in registry.itertuples():
        # Строго раньше даты действия: сам день уже торгуется в новой шкале.
        mask = (frame["SECID"] == action.secid) & (frame["TRADEDATE"] < action.date)
        if not mask.any():
            continue
        frame.loc[mask, price_columns] = frame.loc[mask, price_columns] / action.shares_ratio
        if "VOLUME" in frame.columns:
            frame.loc[mask, "VOLUME"] = frame.loc[mask, "VOLUME"] * action.shares_ratio

    return frame
