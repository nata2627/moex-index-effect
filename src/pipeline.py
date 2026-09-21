"""Загрузка подготовленных данных и сборка аналитической выборки.

Один вход для всех скриптов и ноутбуков: конвейер собирается здесь, чтобы
анализ и графики работали с одной и той же выборкой.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import (abnormal_returns as arm, config, events as events_module,
                 matching, moex, sample)


@dataclass
class Dataset:
    """Всё, что нужно для анализа, в одном месте."""

    events: pd.DataFrame           # все 42 события с разметкой пригодности
    price_sample: pd.DataFrame     # события, пригодные для анализа цены
    liquidity_sample: pd.DataFrame  # события, пригодные для анализа ликвидности
    quotes: pd.DataFrame           # котировки с поправкой на дробления
    index_prices: pd.DataFrame
    calendar: pd.DatetimeIndex
    composition: pd.DataFrame
    securities: pd.DataFrame
    sectors: pd.DataFrame
    parameters: pd.DataFrame       # alpha, beta по каждому событию
    abnormal: pd.DataFrame         # AR и CAR по дням окна события
    control_pool: list[str]


def load() -> Dataset:
    """Собрать выборку из выгруженных данных."""
    composition = pd.read_parquet(moex.DATA_RAW / "index_composition.parquet")
    index_prices = pd.read_parquet(moex.DATA_RAW / "index_prices.parquet")
    securities = pd.read_parquet(moex.DATA_RAW / "tqbr_securities.parquet")
    sectors = pd.read_parquet(moex.DATA_RAW / "sector_composition.parquet")
    quotes = pd.read_parquet(moex.DATA_PROCESSED / "quotes_adjusted.parquet")
    calendar = pd.DatetimeIndex(index_prices["TRADEDATE"]).sort_values()

    events = events_module.build_events(composition, calendar,
                                        config.EVENT_WINDOW_START,
                                        config.EVENT_WINDOW_END)
    events = events_module.window_coverage(events, quotes, calendar,
                                           config.ESTIMATION_WINDOW,
                                           config.EVENT_WINDOW)

    panel = arm.build_event_panel(events, quotes, index_prices,
                                  config.ESTIMATION_WINDOW, config.EVENT_WINDOW)
    parameters = arm.fit_market_model(panel)
    events = sample.classify_events(events, quotes, parameters, calendar.max())

    abnormal = arm.compute_abnormal_returns(panel, parameters)
    price_sample = events[events["in_price_sample"]].reset_index(drop=True)
    abnormal = abnormal[abnormal["event_id"].isin(price_sample["event_id"])]

    control_pool = sorted(set(securities["SECID"]) & set(sectors["ticker"]) -
                          set(events["ticker"]))

    return Dataset(
        events=events,
        price_sample=price_sample,
        liquidity_sample=events[events["in_liquidity_sample"]].reset_index(drop=True),
        quotes=quotes,
        index_prices=index_prices,
        calendar=calendar,
        composition=composition,
        securities=securities,
        sectors=sectors,
        parameters=parameters,
        abnormal=abnormal.reset_index(drop=True),
        control_pool=control_pool,
    )


def match(data: Dataset, events: pd.DataFrame | None = None,
          n_neighbours: int = 1, strict: bool = False,
          caliper: float = matching.CALIPER) -> pd.DataFrame:
    """Подобрать контрольную группу для переданной выборки событий."""
    events = data.price_sample if events is None else events
    return matching.match_controls(
        events, data.quotes, data.securities, data.sectors, data.composition,
        data.events, data.calendar, data.control_pool,
        n_neighbours=n_neighbours, strict=strict, caliper=caliper)
