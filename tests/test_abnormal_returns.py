"""Тесты рыночной модели и накопленной аномальной доходности."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import abnormal_returns as arm


@pytest.fixture
def calendar():
    return pd.bdate_range("2023-01-02", periods=400)


@pytest.fixture
def synthetic(calendar):
    """Бумага с заданными альфой и бетой плюс известный скачок после события.

    Альфа нулевая, бета 1.5, а начиная с первого дня после события бумага
    теряет по 0.2% в день сверх рынка. За 60 дней это складывается в 12%.
    """
    # Шум намеренно мал: за 60 дней окна он накапливается, и при реалистичной
    # дневной волатильности разброс оценки перекрыл бы проверяемый эффект.
    # Тест должен ловить ошибку в расчёте, а не удачу конкретного посева.
    rng = np.random.default_rng(7)
    market = rng.normal(0.0004, 0.012, len(calendar))
    stock = 1.5 * market + rng.normal(0, 0.001, len(calendar))

    event_position = 300
    stock[event_position + 1:event_position + 61] -= 0.002

    index_prices = pd.DataFrame({
        "TRADEDATE": calendar,
        "CLOSE": 3000 * np.exp(np.cumsum(np.insert(market[1:], 0, 0.0))),
    })
    quotes = pd.DataFrame({
        "SECID": "TEST",
        "TRADEDATE": calendar,
        "CLOSE": 100 * np.exp(np.cumsum(np.insert(stock[1:], 0, 0.0))),
        "VALUE": 1e6, "NUMTRADES": 1000, "HIGH": 101.0, "LOW": 99.0,
    })
    events = pd.DataFrame({
        "event_id": ["TEST_вкл_20240101"],
        "ticker": ["TEST"],
        "event_type": ["включение"],
        "event_date": [calendar[event_position]],
    })
    return events, quotes, index_prices


def test_market_model_recovers_beta(synthetic):
    events, quotes, index_prices = synthetic
    panel = arm.build_event_panel(events, quotes, index_prices, (-250, -40), (-30, 60))
    parameters = arm.fit_market_model(panel)

    assert parameters.loc[0, "beta"] == pytest.approx(1.5, abs=0.06)
    assert parameters.loc[0, "alpha"] == pytest.approx(0.0, abs=0.001)
    assert parameters.loc[0, "n_estimation"] == 211


def test_car_recovers_injected_effect(synthetic):
    """Заложенные 0.2% в день за 60 дней должны найтись как -12%."""
    events, quotes, index_prices = synthetic
    panel = arm.build_event_panel(events, quotes, index_prices, (-250, -40), (-30, 60))
    parameters = arm.fit_market_model(panel)
    abnormal = arm.compute_abnormal_returns(panel, parameters)

    car = arm.car_by_event(abnormal, (1, 60))
    assert car.loc[0, "car"] == pytest.approx(-0.12, abs=0.02)


def test_no_effect_before_event(synthetic):
    """До события аномальной доходности быть не должно."""
    events, quotes, index_prices = synthetic
    panel = arm.build_event_panel(events, quotes, index_prices, (-250, -40), (-30, 60))
    abnormal = arm.compute_abnormal_returns(panel, arm.fit_market_model(panel))

    car = arm.car_by_event(abnormal, (-30, -1))
    assert abs(car.loc[0, "car"]) < 0.02


def test_estimation_window_excluded_from_event_window(synthetic):
    """Окна не пересекаются: иначе эффект просочился бы в оценку беты."""
    events, quotes, index_prices = synthetic
    panel = arm.build_event_panel(events, quotes, index_prices, (-250, -40), (-30, 60))
    estimation = panel[panel["window"] == arm.ESTIMATION]["t"]
    event = panel[panel["window"] == arm.EVENT]["t"]
    assert estimation.max() < event.min()
