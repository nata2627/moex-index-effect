"""Тесты разности разностей и подбора контрольной группы."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import did, matching


@pytest.fixture
def calendar():
    return pd.bdate_range("2024-01-01", periods=200)


def make_panel(calendar, effect: float, trend_gap: float = 0.0):
    """Панель из пар с заданным эффектом после события.

    ``trend_gap`` задаёт расхождение трендов до события — им проверяется,
    что проверка параллельности его замечает.
    """
    event_position = 100
    event_date = calendar[event_position]
    rng = np.random.default_rng(5)

    rows = []
    for pair in range(12):
        for ticker, treated in ((f"T{pair}", 1), (f"C{pair}", 0)):
            for offset, date in enumerate(calendar):
                t = offset - event_position
                if not (-60 <= t <= 60):
                    continue
                level = 15 + pair * 0.1 + rng.normal(0, 0.05)
                if treated and t > 0:
                    level += effect
                if treated and t < 0:
                    level += trend_gap * t / 60
                rows.append({
                    "event_id": f"E{pair}", "event_type": "включение",
                    "event_date": event_date, "ticker": ticker,
                    "treated": treated, "t": t, "TRADEDATE": date,
                    "log_value": level,
                })

    panel = pd.DataFrame(rows)
    panel["period"] = np.where(panel["t"].between(*did.PRE_PERIOD), "до",
                       np.where(panel["t"].between(*did.POST_PERIOD), "после", None))
    panel["post"] = (panel["period"] == "после").astype(float)
    panel["unit"] = panel["event_id"] + "|" + panel["ticker"]
    return panel


def test_did_recovers_known_effect(calendar):
    panel = make_panel(calendar, effect=0.30)
    result = did.estimate_did(panel, "log_value")
    assert result["coefficient"] == pytest.approx(0.30, abs=0.03)
    assert result["p_value"] < 0.01


def test_did_finds_nothing_when_there_is_nothing(calendar):
    panel = make_panel(calendar, effect=0.0)
    result = did.estimate_did(panel, "log_value")
    assert abs(result["coefficient"]) < 0.05
    assert result["p_value"] > 0.05


def test_parallel_trends_passes_when_trends_are_parallel(calendar):
    panel = make_panel(calendar, effect=0.30)
    result = did.parallel_trends_test(panel, "log_value")
    assert result["p_value"] > 0.05


def test_parallel_trends_detects_diverging_pre_period(calendar):
    """Если группы расходились до события, проверка обязана это заметить."""
    panel = make_panel(calendar, effect=0.0, trend_gap=0.5)
    result = did.parallel_trends_test(panel, "log_value")
    assert result["p_value"] < 0.05


def test_degenerate_metric_reports_instead_of_crashing(calendar):
    """Метрика без вариации не должна ронять расчёт."""
    panel = make_panel(calendar, effect=0.0)
    panel["no_trade"] = 0.0
    result = did.estimate_did(panel, "no_trade")
    assert np.isnan(result["coefficient"])
    assert "нет вариации" in result["note"]


def test_covariate_balance_reports_standardised_difference():
    matches = pd.DataFrame({
        "target_log_cap": [10.0, 11.0, 12.0],
        "control_log_cap": [10.0, 11.0, 12.0],
        "target_log_trades": [5.0, 6.0, 7.0],
        "control_log_trades": [4.0, 5.0, 6.0],
    })
    balance = matching.covariate_balance(matches)
    capital = balance[balance["признак"] == "логарифм капитализации"].iloc[0]
    trades = balance[balance["признак"] == "логарифм числа сделок"].iloc[0]
    assert capital["стандартизованная разность"] == pytest.approx(0.0)
    assert trades["стандартизованная разность"] == pytest.approx(1.0)


def test_control_pool_excludes_current_index_members():
    """Бумага, состоящая в индексе на дату события, контролем быть не может."""
    composition = pd.DataFrame({
        "ticker": ["INSIDE", "LEFT"],
        "date_from": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "date_till": pd.to_datetime(["2026-01-01", "2023-01-01"]),
    })
    events = pd.DataFrame({"ticker": [], "event_date": pd.to_datetime([])})
    calendar = pd.bdate_range("2020-01-01", "2026-01-01")

    eligible = matching.eligible_controls(
        ["INSIDE", "LEFT", "OTHER"], composition, events, calendar,
        pd.Timestamp("2024-06-03"))
    assert "INSIDE" not in eligible
    assert {"LEFT", "OTHER"} <= set(eligible)


def test_strict_pool_excludes_every_former_member():
    composition = pd.DataFrame({
        "ticker": ["LEFT"],
        "date_from": pd.to_datetime(["2020-01-01"]),
        "date_till": pd.to_datetime(["2023-01-01"]),
    })
    events = pd.DataFrame({"ticker": [], "event_date": pd.to_datetime([])})
    calendar = pd.bdate_range("2020-01-01", "2026-01-01")

    eligible = matching.eligible_controls(
        ["LEFT", "OTHER"], composition, events, calendar,
        pd.Timestamp("2024-06-03"), strict=True)
    assert eligible == ["OTHER"]
