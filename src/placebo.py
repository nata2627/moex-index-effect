"""Плацебо-тест: та же процедура на датах, где события не было.

Смысл теста в том, чтобы узнать, насколько велик эффект, который наша
процедура выдаёт на пустом месте. Если настоящая оценка лежит внутри
распределения плацебо-оценок, находки нет, каким бы ни был t-статистик.

Две детали, без которых тест теряет смысл.

**Кластерная структура сохраняется.** Биржа пересматривает индекс пачками:
21 марта 2025 года в выборку попало шесть событий сразу. Если в плацебо
раздавать каждой бумаге независимую случайную дату, наблюдения станут
независимыми, разброс плацебо-распределения сожмётся, и настоящий эффект
окажется «значимым» просто по построению. Поэтому плацебо переносит пачку
целиком: те же бумаги, тот же размер группы, другая дата.

**Плацебо-даты держатся подальше от настоящих событий.** Иначе окно
плацебо-события пересечётся с реальным, и тест будет измерять тот же эффект,
против которого проверяется.

Вывод делается рандомизационный: доля плацебо-оценок, по модулю не меньших
фактической, и есть двусторонний p-значение. Он не требует ни нормальности,
ни асимптотики по числу кластеров — а именно эти предпосылки на выборке из
шести кластеров не выполняются.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Насколько далеко плацебо-дата должна отстоять от любого настоящего события
# той же бумаги, торговых дней.
EXCLUSION_RADIUS = 90


class ReturnMatrix:
    """Доходности бумаг, выровненные по торговому календарю индекса.

    Плацебо гоняет одну и ту же процедуру тысячу раз, поэтому доходности
    один раз раскладываются в массивы numpy, а оценка беты считается напрямую
    через ковариацию — это на два порядка быстрее повторной сборки таблиц.
    """

    def __init__(self, quotes: pd.DataFrame, index_prices: pd.DataFrame):
        index_prices = index_prices.sort_values("TRADEDATE")
        self.calendar = pd.DatetimeIndex(index_prices["TRADEDATE"])
        self.position = {date: i for i, date in enumerate(self.calendar)}

        close = index_prices["CLOSE"].to_numpy(dtype=float)
        self.market = np.full(len(close), np.nan)
        self.market[1:] = np.log(close[1:] / close[:-1])

        self.returns: dict[str, np.ndarray] = {}
        quoted = quotes.loc[quotes["CLOSE"].notna()]
        for ticker, frame in quoted.groupby("SECID"):
            frame = frame.sort_values("TRADEDATE")
            aligned = pd.Series(frame["CLOSE"].to_numpy(dtype=float),
                                index=frame["TRADEDATE"]).reindex(self.calendar)
            prices = aligned.to_numpy(dtype=float)
            series = np.full(len(prices), np.nan)
            # Доходность считается по последней имеющейся цене: пауза в торгах
            # даёт одну доходность «через пропуск», а не обрыв ряда.
            filled = pd.Series(prices).ffill().to_numpy()
            series[1:] = np.log(filled[1:] / filled[:-1])
            series[np.isnan(prices)] = np.nan
            self.returns[ticker] = series

    def car(self, ticker: str, position: int,
            estimation_window: tuple[int, int],
            event_window: tuple[int, int],
            min_estimation: int = 120) -> float:
        """CAR бумаги вокруг позиции ``position`` в торговом календаре."""
        series = self.returns.get(ticker)
        if series is None:
            return np.nan

        est_start, est_end = position + estimation_window[0], position + estimation_window[1]
        evt_start, evt_end = position + event_window[0], position + event_window[1]
        if est_start < 0 or evt_end >= len(series):
            return np.nan

        est_stock = series[est_start:est_end + 1]
        est_market = self.market[est_start:est_end + 1]
        usable = np.isfinite(est_stock) & np.isfinite(est_market)
        if usable.sum() < min_estimation:
            return np.nan

        stock, market = est_stock[usable], est_market[usable]
        variance = market.var()
        if variance <= 0:
            return np.nan
        beta = ((market - market.mean()) * (stock - stock.mean())).mean() / variance
        alpha = stock.mean() - beta * market.mean()

        evt_stock = series[evt_start:evt_end + 1]
        evt_market = self.market[evt_start:evt_end + 1]
        abnormal = evt_stock - (alpha + beta * evt_market)
        return float(np.nansum(abnormal))


def admissible_positions(matrix: ReturnMatrix,
                         ticker: str,
                         real_event_dates: pd.Series,
                         estimation_window: tuple[int, int],
                         event_window: tuple[int, int],
                         exclusion_radius: int = EXCLUSION_RADIUS) -> np.ndarray:
    """Позиции календаря, годные как плацебо-дата для данной бумаги."""
    total = len(matrix.calendar)
    left = -estimation_window[0]
    right = total - event_window[1] - 1
    if right <= left:
        return np.array([], dtype=int)

    positions = np.arange(left, right)
    for date in real_event_dates:
        centre = matrix.position.get(pd.Timestamp(date))
        if centre is None:
            continue
        positions = positions[np.abs(positions - centre) > exclusion_radius]
    return positions


def run_placebo(events: pd.DataFrame,
                matrix: ReturnMatrix,
                all_event_dates: pd.DataFrame,
                estimation_window: tuple[int, int],
                event_window: tuple[int, int],
                n_iterations: int = 1000,
                seed: int = 20260921) -> pd.Series:
    """Распределение средней CAR на случайных датах.

    Пачки событий переносятся целиком: если в реальности шесть бумаг вошли в
    индекс в один день, то и в плацебо те же шесть бумаг получают одну общую
    случайную дату.
    """
    rng = np.random.default_rng(seed)
    groups = [frame["ticker"].tolist() for _, frame in events.groupby("event_date")]

    # Для каждой бумаги — свой список допустимых дат, считается один раз.
    own_events = all_event_dates.groupby("ticker")["event_date"].apply(list).to_dict()
    admissible = {}
    for ticker in {t for group in groups for t in group}:
        admissible[ticker] = admissible_positions(
            matrix, ticker, own_events.get(ticker, []), estimation_window, event_window)

    results = []
    for _ in range(n_iterations):
        values = []
        for group in groups:
            # Общая для пачки дата — из пересечения допустимых позиций её бумаг.
            shared = None
            for ticker in group:
                positions = admissible.get(ticker, np.array([], dtype=int))
                shared = positions if shared is None else np.intersect1d(shared, positions)
            if shared is None or len(shared) == 0:
                continue
            position = int(rng.choice(shared))
            for ticker in group:
                value = matrix.car(ticker, position, estimation_window, event_window)
                if np.isfinite(value):
                    values.append(value)
        results.append(np.mean(values) if values else np.nan)

    return pd.Series(results, name="placebo_car")


def randomization_p_value(observed: float, placebo: pd.Series) -> dict:
    """Двусторонний рандомизационный p-значение и положение факта в плацебо.

    К числителю и знаменателю добавляется единица: фактическая оценка сама
    является одной из возможных перестановок, и без этой поправки p-значение
    может оказаться нулевым, чего при конечном числе итераций быть не должно.
    """
    clean = placebo.dropna()
    if clean.empty or not np.isfinite(observed):
        return {"p_value": np.nan, "percentile": np.nan, "n_placebo": len(clean)}

    at_least_as_extreme = int((clean.abs() >= abs(observed)).sum())
    return {
        "p_value": (at_least_as_extreme + 1) / (len(clean) + 1),
        "percentile": float((clean < observed).mean() * 100),
        "n_placebo": len(clean),
        "placebo_mean": clean.mean(),
        "placebo_std": clean.std(),
    }
