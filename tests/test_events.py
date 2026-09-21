"""Тесты сборки событий из истории состава индекса."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import events


@pytest.fixture
def calendar():
    """Пять рабочих недель подряд как торговый календарь."""
    return pd.bdate_range("2024-01-01", "2024-02-02")


@pytest.fixture
def composition(calendar):
    """Три бумаги: действующий член, выбывшая и та, что вошла в середине."""
    last = calendar[-1]
    return pd.DataFrame({
        "ticker": ["STAY", "GONE", "NEWC"],
        "date_from": pd.to_datetime(["2020-01-02", "2020-01-02", "2024-01-15"]),
        "date_till": [last, pd.Timestamp("2024-01-19"), last],
    })


def test_current_member_gives_no_exclusion(composition, calendar):
    """Бумага, чей date_till равен краю данных, из индекса не выбывала.

    STAY входила в индекс в 2020-м, поэтому в широком окне у неё есть
    включение, но исключения быть не должно.
    """
    frame = events.build_events(composition, calendar, "2020-01-01", "2024-02-02")
    stay = frame[frame["ticker"] == "STAY"]["event_type"].tolist()
    gone = frame[frame["ticker"] == "GONE"]["event_type"].tolist()
    assert stay == [events.INCLUSION]
    assert gone == [events.INCLUSION, events.EXCLUSION]


def test_exclusion_date_is_first_day_outside_index(composition, calendar):
    """t = 0 при исключении — первый торговый день после последнего дня в индексе."""
    frame = events.build_events(composition, calendar, "2020-01-01", "2024-02-02")
    exclusion = frame[(frame["ticker"] == "GONE") &
                      (frame["event_type"] == events.EXCLUSION)].iloc[0]
    # 19.01.2024 — пятница, последний день в индексе; следующий торговый — понедельник.
    assert exclusion["event_date"] == pd.Timestamp("2024-01-22")


def test_inclusion_date_is_first_day_in_index(composition, calendar):
    frame = events.build_events(composition, calendar, "2024-01-01", "2024-02-02")
    inclusion = frame[frame["ticker"] == "NEWC"].iloc[0]
    assert inclusion["event_type"] == events.INCLUSION
    assert inclusion["event_date"] == pd.Timestamp("2024-01-15")


def test_window_filters_events_outside_range(composition, calendar):
    """Включения 2020 года не попадают в окно 2024-го."""
    frame = events.build_events(composition, calendar, "2024-01-01", "2024-02-02")
    assert set(frame["ticker"]) == {"NEWC", "GONE"}


def test_relative_day_index_zero_at_event(calendar):
    offsets = events.relative_day_index(calendar, pd.Timestamp("2024-01-15"))
    assert offsets.loc[pd.Timestamp("2024-01-15")] == 0
    assert offsets.loc[pd.Timestamp("2024-01-16")] == 1
    assert offsets.loc[pd.Timestamp("2024-01-12")] == -1


def test_coverage_counts_only_days_with_price(composition, calendar):
    """Строки-заглушки без цены (дни остановки торгов) торговыми не считаются."""
    quotes = pd.DataFrame({
        "SECID": ["NEWC"] * len(calendar),
        "TRADEDATE": calendar,
        "CLOSE": [100.0] * len(calendar),
    })
    quotes.loc[quotes["TRADEDATE"] >= "2024-01-16", "CLOSE"] = None

    frame = events.build_events(composition, calendar, "2024-01-01", "2024-02-02")
    covered = events.window_coverage(frame, quotes, calendar, (-5, -1), (-2, 5))
    row = covered[covered["ticker"] == "NEWC"].iloc[0]

    assert row["evt_days_with_price"] == 3  # 11, 12, 15 января
    assert row["last_quote"] == pd.Timestamp("2024-01-15")
    assert row["last_quote_t"] == 0
