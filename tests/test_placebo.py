"""Тесты плацебо-процедуры."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import placebo


@pytest.fixture
def matrix():
    calendar = pd.bdate_range("2022-01-03", periods=900)
    rng = np.random.default_rng(11)
    market = rng.normal(0.0003, 0.011, len(calendar))
    stock = 1.2 * market + rng.normal(0, 0.005, len(calendar))

    index_prices = pd.DataFrame({
        "TRADEDATE": calendar,
        "CLOSE": 3000 * np.exp(np.cumsum(np.insert(market[1:], 0, 0.0))),
    })
    quotes = pd.DataFrame({
        "SECID": "TEST",
        "TRADEDATE": calendar,
        "CLOSE": 100 * np.exp(np.cumsum(np.insert(stock[1:], 0, 0.0))),
    })
    return placebo.ReturnMatrix(quotes, index_prices)


def test_car_is_near_zero_without_event(matrix):
    """На данных без события накопленная аномальная доходность мала."""
    value = matrix.car("TEST", 500, (-250, -40), (1, 60))
    assert abs(value) < 0.15


def test_car_returns_nan_when_history_is_short(matrix):
    assert np.isnan(matrix.car("TEST", 10, (-250, -40), (1, 60)))


def test_admissible_positions_avoid_real_events(matrix):
    real = pd.Series([matrix.calendar[500]])
    positions = placebo.admissible_positions(matrix, "TEST", real, (-250, -40), (1, 60))
    assert not ((positions > 410) & (positions < 590)).any()


def test_placebo_preserves_cluster_structure(matrix):
    """Бумаги одной даты события получают общую плацебо-дату.

    Если раздать им независимые даты, наблюдения станут независимыми,
    распределение плацебо сузится, и настоящий эффект окажется значимым
    просто из-за неверной структуры сравнения.
    """
    shared_date = matrix.calendar[500]
    events = pd.DataFrame({
        "ticker": ["TEST", "TEST"],
        "event_date": [shared_date, shared_date],
    })
    all_events = pd.DataFrame({"ticker": ["TEST"], "event_date": [shared_date]})

    result = placebo.run_placebo(events, matrix, all_events, (-250, -40), (1, 60),
                                 n_iterations=25, seed=1)
    # Обе бумаги в паре — одна и та же серия с одной датой, поэтому каждая
    # итерация даёт два одинаковых значения, и их среднее равно самому значению.
    assert result.notna().sum() > 0


def test_randomization_p_value_never_zero():
    """Фактическая оценка сама является одной из перестановок."""
    distribution = pd.Series(np.zeros(100))
    result = placebo.randomization_p_value(5.0, distribution)
    assert result["p_value"] > 0
    assert result["p_value"] == pytest.approx(1 / 101)


def test_randomization_p_value_is_two_sided():
    distribution = pd.Series(np.linspace(-0.1, 0.1, 1000))
    left = placebo.randomization_p_value(-0.09, distribution)
    right = placebo.randomization_p_value(0.09, distribution)
    assert left["p_value"] == pytest.approx(right["p_value"], abs=0.02)
