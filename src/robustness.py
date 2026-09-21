"""Проверки устойчивости оценки.

Один результат при одной спецификации мало что значит, особенно на выборке
из двух десятков событий. Здесь одна и та же величина оценивается разными
способами, и интерес представляет не столько каждая оценка, сколько разброс
между ними: если знак меняется при смене окна, говорить об эффекте нельзя.

Проверяются четыре измерения:

* **окно события** — сохраняется ли эффект на горизонтах от 20 до 90 дней;
* **окно оценки** — не держится ли результат на конкретном периоде оценки беты;
* **способ снятия рыночной компоненты** — рыночная модель против прямого
  сравнения с сопоставимой бумагой;
* **состав выборки** — влияют ли возвраты после редомициляции и требование
  «контроль никогда не входил в индекс».
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import abnormal_returns as arm, config, inference, matching, pipeline

POST_WINDOWS = [(1, 20), (1, 40), (1, 60), (1, 90)]
ESTIMATION_WINDOWS = [(-250, -40), (-150, -40), (-250, -60), (-120, -30)]


def by_event_window(data: pipeline.Dataset) -> pd.DataFrame:
    """Оценка рыночной моделью на разных окнах события."""
    rows = []
    for window in POST_WINDOWS:
        panel = arm.build_event_panel(data.price_sample, data.quotes,
                                      data.index_prices, config.ESTIMATION_WINDOW,
                                      (config.EVENT_WINDOW[0], window[1]))
        parameters = arm.fit_market_model(panel)
        abnormal = arm.compute_abnormal_returns(panel, parameters)

        for event_type, subset in data.price_sample.groupby("event_type"):
            part = abnormal[abnormal["event_id"].isin(subset["event_id"])]
            car = arm.car_by_event(part, window)
            test = inference.clustered_test(car["car"], car["event_date"])
            rows.append({
                "измерение": "окно события",
                "вариант": f"[+{window[0]},+{window[1]}]",
                "event_type": event_type,
                "n": len(car),
                "оценка": car["car"].mean(),
                "медиана": car["car"].median(),
                "p_value": test["p_value"],
            })
    return pd.DataFrame(rows)


def by_estimation_window(data: pipeline.Dataset) -> pd.DataFrame:
    """Оценка на разных окнах оценки рыночной модели."""
    rows = []
    for estimation in ESTIMATION_WINDOWS:
        panel = arm.build_event_panel(data.price_sample, data.quotes,
                                      data.index_prices, estimation,
                                      config.EVENT_WINDOW)
        parameters = arm.fit_market_model(panel)
        abnormal = arm.compute_abnormal_returns(panel, parameters)

        for event_type, subset in data.price_sample.groupby("event_type"):
            part = abnormal[abnormal["event_id"].isin(subset["event_id"])]
            car = arm.car_by_event(part, (1, 60))
            test = inference.clustered_test(car["car"], car["event_date"])
            rows.append({
                "измерение": "окно оценки",
                "вариант": f"[{estimation[0]},{estimation[1]}]",
                "event_type": event_type,
                "n": len(car),
                "оценка": car["car"].mean(),
                "медиана": car["car"].median(),
                "p_value": test["p_value"],
            })
    return pd.DataFrame(rows)


def by_control_definition(data: pipeline.Dataset) -> pd.DataFrame:
    """Оценка прямым сравнением с контролем при разных правилах подбора."""
    variants = [
        ("отрасль, размер, ликвидность", dict(match_on_momentum=False, strict=False)),
        ("плюс динамика до события", dict(match_on_momentum=True, strict=False)),
        ("контроль никогда не в индексе", dict(match_on_momentum=False, strict=True)),
        ("три ближайших соседа", dict(match_on_momentum=False, strict=False,
                                      n_neighbours=3)),
    ]

    rows = []
    for label, options in variants:
        matches = matching.match_controls(
            data.price_sample, data.quotes, data.securities, data.sectors,
            data.composition, data.events, data.calendar, data.control_pool,
            index_prices=data.index_prices, **options)
        if matches.empty:
            continue
        returns = arm.matched_returns(matches, data.quotes, data.calendar, (1, 60))
        # При нескольких соседях сначала усредняем контроль внутри события.
        returns = returns.groupby(["event_id", "event_type", "event_date"],
                                  as_index=False)["difference"].mean()

        for event_type, subset in returns.groupby("event_type"):
            subset = subset.dropna(subset=["difference"])
            test = inference.clustered_test(subset["difference"], subset["event_date"])
            rows.append({
                "измерение": "подбор контроля",
                "вариант": label,
                "event_type": event_type,
                "n": len(subset),
                "оценка": subset["difference"].mean(),
                "медиана": subset["difference"].median(),
                "p_value": test["p_value"],
            })
    return pd.DataFrame(rows)


def by_sample(data: pipeline.Dataset) -> pd.DataFrame:
    """Оценка на разных подвыборках событий."""
    variants = [
        ("все пригодные события", data.price_sample),
        ("без возвратов после редомициляции",
         data.price_sample[~data.price_sample["is_redomicile"]]),
        ("без событий 2026 года",
         data.price_sample[data.price_sample["event_date"] < "2026-01-01"]),
    ]

    rows = []
    for label, subset in variants:
        if subset.empty:
            continue
        for event_type, part in subset.groupby("event_type"):
            abnormal = data.abnormal[data.abnormal["event_id"].isin(part["event_id"])]
            car = arm.car_by_event(abnormal, (1, 60))
            if car.empty:
                continue
            test = inference.clustered_test(car["car"], car["event_date"])
            rows.append({
                "измерение": "состав выборки",
                "вариант": label,
                "event_type": event_type,
                "n": len(car),
                "оценка": car["car"].mean(),
                "медиана": car["car"].median(),
                "p_value": test["p_value"],
            })
    return pd.DataFrame(rows)


def run_all(data: pipeline.Dataset) -> pd.DataFrame:
    """Все проверки устойчивости в одной таблице."""
    frames = [by_event_window(data), by_estimation_window(data),
              by_control_definition(data), by_sample(data)]
    result = pd.concat(frames, ignore_index=True)
    # Поправка на множественность по всей сетке: проверок здесь десятки, они
    # сильно зависимы, и контроль доли ложных открытий уместнее, чем FWER.
    result["p_fdr"] = inference.benjamini_hochberg(result["p_value"])
    return result
