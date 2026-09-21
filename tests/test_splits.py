"""Тесты обнаружения дроблений.

Проверяются оба типа ошибок: принять обвал за дробление и пропустить
настоящее дробление. Оба реально случились на наших данных.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import splits


def make_series(dates, close, volume, value):
    return pd.DataFrame({
        "SECID": ["TEST"] * len(dates),
        "TRADEDATE": dates,
        "OPEN": close, "LOW": close, "HIGH": close, "CLOSE": close,
        "VOLUME": volume, "VALUE": value, "WAPRICE": close,
    })


@pytest.fixture
def calendar():
    return pd.bdate_range("2024-01-01", periods=40)


@pytest.fixture
def split_quotes(calendar):
    """Дробление 1:10 на 21-й день: цена /10, объём x10, оборот тот же."""
    close = np.array([1000.0] * 20 + [100.0] * 20)
    volume = np.array([1_000_000.0] * 20 + [10_000_000.0] * 20)
    value = close * volume
    return make_series(calendar, close, volume, value)


@pytest.fixture
def crash_quotes(calendar):
    """Обвал вдвое вместе со всем рынком: объём в штуках не меняется."""
    close = np.array([1000.0] * 20 + [500.0] * 20)
    volume = np.array([1_000_000.0] * 40)
    value = close * volume
    return make_series(calendar, close, volume, value)


def quiet_market(calendar):
    return pd.Series(0.001, index=calendar)


def test_split_is_detected(split_quotes, calendar):
    detected = splits.detect_splits(split_quotes, quiet_market(calendar))
    assert len(detected) == 1
    assert detected.iloc[0]["is_split"]
    assert detected.iloc[0]["split_ratio"] == 10


def test_market_wide_crash_is_not_a_split(crash_quotes, calendar):
    """Падение вдвое кратно коэффициенту 2, но рынок упал вместе с бумагой."""
    market = pd.Series(0.001, index=calendar)
    market.iloc[20] = -0.33
    detected = splits.detect_splits(crash_quotes, market)
    assert not detected.iloc[0]["is_split"]
    assert "рынок" in detected.iloc[0]["verdict"]


def test_crash_without_volume_change_is_not_a_split(crash_quotes, calendar):
    """Даже при спокойном рынке: цена упала вдвое, а бумаг больше не стало."""
    detected = splits.detect_splits(crash_quotes, quiet_market(calendar))
    assert not detected.iloc[0]["is_split"]


def test_news_driven_jump_is_not_a_split(calendar):
    """Реальная новость: цена /10, но оборот в рублях взлетел в 20 раз."""
    close = np.array([1000.0] * 20 + [100.0] * 20)
    volume = np.array([1_000_000.0] * 20 + [200_000_000.0] * 20)
    quotes = make_series(calendar, close, volume, close * volume)
    detected = splits.detect_splits(quotes, quiet_market(calendar))
    assert not detected.iloc[0]["is_split"]
    assert "оборот" in detected.iloc[0]["verdict"]


def test_adjustment_makes_series_continuous(split_quotes, calendar):
    """После поправки скачок доходности на стыке исчезает."""
    registry = pd.DataFrame({"secid": ["TEST"],
                             "date": [calendar[20]],
                             "shares_ratio": [10.0]})
    adjusted = splits.adjust_for_splits(split_quotes, registry)

    returns = adjusted.sort_values("TRADEDATE")["CLOSE"].pct_change().dropna()
    assert returns.abs().max() < 1e-9

    # Оборот в рублях поправка не трогает: он не зависит от номинала.
    pd.testing.assert_series_equal(adjusted["VALUE"], split_quotes["VALUE"])
    # Объём в штуках до дробления приведён к новой шкале.
    assert adjusted.loc[0, "VOLUME"] == pytest.approx(10_000_000.0)


def test_halt_before_split_does_not_break_detection(calendar):
    """Дробление на возобновлении торгов после паузы всё равно находится."""
    close = np.array([1000.0] * 20 + [np.nan] * 4 + [100.0] * 16)
    volume = np.array([1_000_000.0] * 20 + [0.0] * 4 + [10_000_000.0] * 16)
    value = np.nan_to_num(close, nan=0.0) * volume
    quotes = make_series(calendar, close, volume, value)
    detected = splits.detect_splits(quotes, quiet_market(calendar))
    assert detected["is_split"].sum() == 1


def test_volume_growing_faster_than_ratio_is_still_a_split(calendar):
    """Дробление 1:100 у Норникеля дало рост объёма в 339 раз, а не в 100.

    После дробления бумага дешевеет, становится доступнее мелкому инвестору,
    и торговая активность растёт сильнее коэффициента. Требовать точного
    совпадения объёма с коэффициентом нельзя — так пропускаются настоящие
    дробления.
    """
    close = np.array([15000.0] * 20 + [150.0] * 20)
    volume = np.array([90_000.0] * 20 + [30_000_000.0] * 20)
    quotes = make_series(calendar, close, volume, close * volume)
    detected = splits.detect_splits(quotes, quiet_market(calendar))
    assert detected.iloc[0]["is_split"]
    assert detected.iloc[0]["split_ratio"] == 100


def test_volume_growing_slower_than_ratio_is_not_a_split(calendar):
    """Цена упала вдвое, а бумаг почти не прибавилось — дробления не было.

    Так выглядит крупная дивидендная отсечка: у привилегированных акций
    Лензолота в июле 2021 цена упала более чем вдвое при росте объёма
    в 3.8 раза, что коэффициенту 2 соответствует лишь формально.
    """
    close = np.array([6000.0] * 20 + [3000.0] * 20)
    volume = np.array([20_000.0] * 20 + [22_000.0] * 20)
    quotes = make_series(calendar, close, volume, close * volume)
    detected = splits.detect_splits(quotes, quiet_market(calendar))
    assert not detected.iloc[0]["is_split"]


def test_registry_adjustment_is_cumulative(calendar):
    """Два дробления подряд перемножаются."""
    close = np.array([1000.0] * 14 + [100.0] * 13 + [10.0] * 13)
    volume = np.array([1e6] * 14 + [1e7] * 13 + [1e8] * 13)
    quotes = make_series(calendar, close, volume, close * volume)
    registry = pd.DataFrame({"secid": ["TEST", "TEST"],
                             "date": [calendar[14], calendar[27]],
                             "shares_ratio": [10.0, 10.0]})
    adjusted = splits.adjust_for_splits(quotes, registry)
    returns = adjusted.sort_values("TRADEDATE")["CLOSE"].pct_change().dropna()
    assert returns.abs().max() < 1e-9
    assert adjusted.loc[0, "CLOSE"] == pytest.approx(10.0)


def test_audit_reports_both_kinds_of_mismatch(calendar, split_quotes):
    """Аудит помечает и отклонённые гипотезы, и ручные записи реестра."""
    detected = splits.detect_splits(split_quotes, quiet_market(calendar))
    registry = pd.DataFrame({"secid": ["OTHER"],
                             "date": [pd.Timestamp("2024-05-01")],
                             "shares_ratio": [8.0]})
    audit = splits.audit_registry(detected, registry)
    statuses = set(audit["status"])
    assert any("гипотеза отклонена" in s for s in statuses)
    assert any("внесено в реестр вручную" in s for s in statuses)
