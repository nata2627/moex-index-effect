"""Рыночная модель, аномальные доходности и накопленный эффект.

Обычная доходность бумаги включает движение всего рынка, и без его вычитания
событие 24 февраля 2022 года выглядело бы как эффект исключения из индекса.
Рыночная компонента снимается моделью рынка:

    R_it = alpha_i + beta_i * R_mt + eps_it,      t из окна оценки
    AR_it = R_it - (alpha_i + beta_i * R_mt),     t из окна события
    CAR_i[t1, t2] = сумма AR_it по t от t1 до t2

Параметры alpha и beta оцениваются МНК на окне оценки, которое с окном
события не пересекается: иначе реакция на событие попала бы в оценку беты и
частично вычлась бы из самой себя.

Доходности логарифмические. Причина в аддитивности: CAR — это сумма AR, и
логарифмические доходности складываются точно, тогда как для простых сумма за
90 дней уже заметно расходится с фактическим изменением цены. Плата за это —
небольшое смещение при переводе обратно в проценты, которое на наших порядках
величин несущественно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import events as events_module

ESTIMATION = "оценка"
EVENT = "событие"


def log_returns(prices: pd.Series) -> pd.Series:
    """Логарифмическая доходность ряда цен."""
    return np.log(prices / prices.shift(1))


def build_event_panel(events: pd.DataFrame,
                      quotes: pd.DataFrame,
                      index_prices: pd.DataFrame,
                      estimation_window: tuple[int, int],
                      event_window: tuple[int, int]) -> pd.DataFrame:
    """Панель «событие x торговый день» с доходностями бумаги и рынка.

    Календарь берётся по индексу: в дни остановки торгов ISS отдаёт строки с
    пустой ценой, и торговыми днями они не являются. Доходность бумаги
    считается по имеющимся ценам подряд, так что пауза в торгах даёт одну
    доходность «через пропуск», а не цепочку пустых значений.
    """
    index_prices = index_prices.sort_values("TRADEDATE")
    calendar = pd.DatetimeIndex(index_prices["TRADEDATE"])
    market = pd.DataFrame({
        "TRADEDATE": calendar,
        "mkt_ret": log_returns(index_prices["CLOSE"].reset_index(drop=True)).to_numpy(),
    })

    by_ticker = {}
    for ticker, frame in quotes.loc[quotes["CLOSE"].notna()].groupby("SECID"):
        frame = frame.sort_values("TRADEDATE")[["TRADEDATE", "CLOSE"]].copy()
        frame["ret"] = log_returns(frame["CLOSE"])
        by_ticker[ticker] = frame

    left = min(estimation_window[0], event_window[0])
    right = max(estimation_window[1], event_window[1])

    pieces = []
    for event in events.itertuples():
        frame = by_ticker.get(event.ticker)
        if frame is None:
            continue

        offsets = events_module.relative_day_index(calendar, event.event_date)
        wanted = offsets[(offsets >= left) & (offsets <= right)]

        piece = pd.DataFrame({"TRADEDATE": wanted.index, "t": wanted.to_numpy()})
        piece = piece.merge(frame, on="TRADEDATE", how="left")
        piece = piece.merge(market, on="TRADEDATE", how="left")
        piece["event_id"] = event.event_id
        piece["ticker"] = event.ticker
        piece["event_type"] = event.event_type
        piece["event_date"] = event.event_date

        piece["window"] = np.where(
            piece["t"].between(*estimation_window), ESTIMATION,
            np.where(piece["t"].between(*event_window), EVENT, None))
        pieces.append(piece)

    panel = pd.concat(pieces, ignore_index=True)
    return panel[panel["window"].notna()].reset_index(drop=True)


def fit_market_model(panel: pd.DataFrame, min_observations: int = 60) -> pd.DataFrame:
    """Оценка alpha и beta по окну оценки для каждого события."""
    rows = []
    for event_id, frame in panel.groupby("event_id"):
        estimation = frame[(frame["window"] == ESTIMATION) &
                           frame["ret"].notna() & frame["mkt_ret"].notna()]

        if len(estimation) < min_observations:
            rows.append({"event_id": event_id, "alpha": np.nan, "beta": np.nan,
                         "sigma_ar": np.nan, "n_estimation": len(estimation),
                         "r_squared": np.nan})
            continue

        design = sm.add_constant(estimation["mkt_ret"].to_numpy())
        model = sm.OLS(estimation["ret"].to_numpy(), design).fit()
        rows.append({
            "event_id": event_id,
            "alpha": model.params[0],
            "beta": model.params[1],
            # Остаточное СКО на окне оценки — масштаб «обычного» шума бумаги,
            # относительно которого оценивается величина аномальной доходности.
            "sigma_ar": float(np.sqrt(model.mse_resid)),
            "n_estimation": len(estimation),
            "r_squared": model.rsquared,
        })

    return pd.DataFrame(rows)


def compute_abnormal_returns(panel: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    """Аномальные доходности на окне события и их накопленная сумма."""
    frame = panel[panel["window"] == EVENT].merge(parameters, on="event_id", how="left")
    frame["ar"] = frame["ret"] - (frame["alpha"] + frame["beta"] * frame["mkt_ret"])
    frame = frame.sort_values(["event_id", "t"])
    # Пропуски (бумага не торговалась) не обрывают накопление: считаем, что в
    # эти дни аномальная доходность равна нулю, а не неизвестна.
    frame["car"] = frame.groupby("event_id")["ar"].transform(
        lambda series: series.fillna(0).cumsum())
    return frame.reset_index(drop=True)


def car_by_event(abnormal: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    """CAR каждого события на заданном подокне."""
    left, right = window
    selected = abnormal[abnormal["t"].between(left, right)]
    grouped = selected.groupby(["event_id", "ticker", "event_type", "event_date"])

    result = grouped.agg(
        car=("ar", lambda series: series.fillna(0).sum()),
        n_days=("ar", "count"),
        sigma_ar=("sigma_ar", "first"),
    ).reset_index()

    # Стандартизованная CAR: во сколько собственных СКО уложился эффект.
    # Нужна, чтобы волатильные бумаги не перевешивали спокойные при усреднении.
    result["scar"] = result["car"] / (result["sigma_ar"] * np.sqrt(result["n_days"]))
    return result


def matched_returns(matches: pd.DataFrame,
                    quotes: pd.DataFrame,
                    calendar: pd.DatetimeIndex,
                    window: tuple[int, int]) -> pd.DataFrame:
    """Накопленная разность доходностей тестовой и контрольной бумаги.

    Прямая оценка эффекта «разность разностей» по цене: вместо вычитания
    рыночной компоненты по оценённой бете вычитается фактическая доходность
    сопоставимой бумаги. Бета здесь не нужна вовсе, поэтому оценка не зависит
    от качества окна оценки — но зависит от качества подбора контроля.

    Две оценки полезны именно вместе: если они расходятся, значит результат
    держится на одном из этих двух допущений, и это надо знать.
    """
    quoted = quotes.loc[quotes["CLOSE"].notna()]
    by_ticker = {ticker: pd.Series(frame["CLOSE"].to_numpy(dtype=float),
                                   index=pd.DatetimeIndex(frame["TRADEDATE"]))
                 for ticker, frame in quoted.groupby("SECID")}

    rows = []
    for pair in matches.itertuples():
        offsets = events_module.relative_day_index(calendar, pair.event_date)
        window_dates = offsets[offsets.between(*window)].index

        def cumulative(ticker: str) -> float:
            series = by_ticker.get(ticker)
            if series is None:
                return np.nan
            values = series.reindex(window_dates).ffill().dropna()
            if len(values) < 2:
                return np.nan
            return float(np.log(values.iloc[-1] / values.iloc[0]))

        treated_return = cumulative(pair.ticker)
        control_return = cumulative(pair.control)
        rows.append({
            "event_id": pair.event_id,
            "ticker": pair.ticker,
            "event_type": pair.event_type,
            "event_date": pair.event_date,
            "control": pair.control,
            "treated_return": treated_return,
            "control_return": control_return,
            "difference": treated_return - control_return,
        })

    return pd.DataFrame(rows)
