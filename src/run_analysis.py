"""Основной расчёт: аномальные доходности, плацебо, разность разностей.

Результаты складываются в data/processed и используются скриптом построения
графиков и README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (abnormal_returns as arm, config, did, inference, matching,
                 moex, pipeline, placebo, robustness)

# Подокна, по которым считается накопленная аномальная доходность. Выбор не
# произволен: каждое отвечает на свой вопрос.
CAR_WINDOWS = {
    "[-30,-1] до события": (-30, -1),      # успел ли рынок отыграть объявление
    "[-1,+1] ребалансировка": (-1, 1),     # сама сделка индексных фондов
    "[+1,+20] ближний откат": (1, 20),     # возвращается ли цена сразу
    "[+1,+60] дальний откат": (1, 60),     # сохраняется ли эффект
    "[-30,+60] всё окно": (-30, 60),       # чистый итог
}

# Основные гипотезы, к которым применяется поправка Холма. Их немного, и
# каждая сформулирована заранее, а не выбрана по результатам.
PRIMARY_WINDOWS = ["[-30,-1] до события", "[-1,+1] ребалансировка",
                   "[+1,+60] дальний откат"]

PLACEBO_ITERATIONS = 1000


def analyse_prices(data: pipeline.Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CAR по подокнам и типам событий с тестами значимости."""
    rows, paths = [], []

    for event_type, subset in data.price_sample.groupby("event_type"):
        abnormal = data.abnormal[data.abnormal["event_id"].isin(subset["event_id"])]

        path = inference.caar_path(abnormal)
        path["event_type"] = event_type
        paths.append(path)

        for label, window in CAR_WINDOWS.items():
            car = arm.car_by_event(abnormal, window)
            naive = inference.cross_sectional_test(car["car"])
            clustered = inference.clustered_test(car["car"], car["event_date"])
            rows.append({
                "event_type": event_type,
                "window": label,
                "n_events": naive["n"],
                "mean_car": naive["mean"],
                "median_car": naive["median"],
                "share_positive": naive["share_positive"],
                "p_naive": naive["p_value"],
                "p_sign": naive["p_sign"],
                "n_clusters": clustered["n_clusters"],
                "se_clustered": clustered["se"],
                "p_clustered": clustered["p_value"],
                "ci_low": clustered.get("ci_low", np.nan),
                "ci_high": clustered.get("ci_high", np.nan),
            })

    results = pd.DataFrame(rows)

    # Поправка Холма — только по заранее заявленным основным гипотезам.
    primary = results["window"].isin(PRIMARY_WINDOWS)
    results["p_holm"] = np.nan
    results.loc[primary, "p_holm"] = inference.holm_correction(
        results.loc[primary, "p_clustered"])
    # Для полной сетки окон — контроль доли ложных открытий.
    results["p_fdr"] = inference.benjamini_hochberg(results["p_clustered"])

    return results, pd.concat(paths, ignore_index=True)


def run_placebo(data: pipeline.Dataset, results: pd.DataFrame) -> pd.DataFrame:
    """Рандомизационный вывод по основным окнам."""
    matrix = placebo.ReturnMatrix(data.quotes, data.index_prices)

    rows = []
    for event_type, subset in data.price_sample.groupby("event_type"):
        for label in PRIMARY_WINDOWS:
            window = CAR_WINDOWS[label]
            abnormal = data.abnormal[data.abnormal["event_id"].isin(subset["event_id"])]
            observed = arm.car_by_event(abnormal, window)["car"].mean()

            distribution = placebo.run_placebo(
                subset, matrix, data.events, config.ESTIMATION_WINDOW, window,
                n_iterations=PLACEBO_ITERATIONS)
            verdict = placebo.randomization_p_value(observed, distribution)

            rows.append({"event_type": event_type, "window": label,
                         "observed": observed, **verdict})
            distribution.to_frame().assign(event_type=event_type, window=label).to_parquet(
                moex.DATA_PROCESSED / f"placebo_{event_type}_{window[0]}_{window[1]}.parquet",
                index=False)
            print(f"  плацебо {event_type} {label}: факт {observed:+.2%}, "
                  f"перцентиль {verdict['percentile']:.1f}, p = {verdict['p_value']:.3f}")

    return pd.DataFrame(rows)


def analyse_liquidity(data: pipeline.Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разность разностей по метрикам ликвидности и проверка трендов."""
    matches = pipeline.match(data, data.liquidity_sample)
    panel = did.build_liquidity_panel(matches, data.quotes, data.calendar)

    estimates, trends = [], []
    for event_type, subset in matches.groupby("event_type"):
        part = panel[panel["event_id"].isin(subset["event_id"])]
        for metric in did.METRICS:
            estimate = did.estimate_did(part, metric)
            estimate["event_type"] = event_type
            estimates.append(estimate)

            trend = did.parallel_trends_test(part, metric)
            trend["event_type"] = event_type
            trend.pop("bins", None)
            trends.append(trend)

    estimate_frame = pd.DataFrame(estimates)
    estimate_frame["p_holm"] = inference.holm_correction(estimate_frame["p_value"])
    return estimate_frame, pd.DataFrame(trends), panel


def main() -> None:
    data = pipeline.load()
    moex.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"События: всего {len(data.events)}, "
          f"для анализа цены {len(data.price_sample)}, "
          f"для анализа ликвидности {len(data.liquidity_sample)}")

    matches = pipeline.match(data)
    balance = matching.covariate_balance(matches)
    matches.to_parquet(moex.DATA_PROCESSED / "matches.parquet", index=False)
    print("\nБаланс ковариат после матчинга:")
    print(balance.round(3).to_string(index=False))

    print("\nАномальные доходности")
    results, paths = analyse_prices(data)
    results.to_parquet(moex.DATA_PROCESSED / "car_results.parquet", index=False)
    paths.to_parquet(moex.DATA_PROCESSED / "caar_paths.parquet", index=False)
    print(results[["event_type", "window", "n_events", "mean_car", "p_clustered",
                   "p_holm"]].round(4).to_string(index=False))

    print("\nПлацебо-тест")
    placebo_results = run_placebo(data, results)
    placebo_results.to_parquet(moex.DATA_PROCESSED / "placebo_results.parquet", index=False)

    print("\nЛиквидность: разность разностей")
    estimates, trends, panel = analyse_liquidity(data)
    estimates.to_parquet(moex.DATA_PROCESSED / "did_results.parquet", index=False)
    trends.to_parquet(moex.DATA_PROCESSED / "parallel_trends.parquet", index=False)
    panel.to_parquet(moex.DATA_PROCESSED / "liquidity_panel.parquet", index=False)
    print(estimates[["event_type", "metric_name", "coefficient", "se",
                     "p_value", "p_holm"]].round(4).to_string(index=False))

    print("\nВсплеск оборота в дни ребалансировки")
    spike = did.rebalancing_spike(panel)
    spike.to_parquet(moex.DATA_PROCESSED / "rebalancing_spike.parquet", index=False)
    print(spike.round(4).to_string(index=False))

    print("\nПроверка параллельных трендов")
    print(trends[["event_type", "metric_name", "f_stat", "p_value"]].round(4).to_string(index=False))

    print("\nПроверки устойчивости")
    robustness_results = robustness.run_all(data)
    robustness_results.to_parquet(moex.DATA_PROCESSED / "robustness.parquet", index=False)
    print(robustness_results[["event_type", "измерение", "вариант", "оценка",
                              "p_value", "p_fdr"]].round(4).to_string(index=False))

    data.events.to_parquet(moex.DATA_PROCESSED / "events.parquet", index=False)
    data.abnormal.to_parquet(moex.DATA_PROCESSED / "abnormal_returns.parquet", index=False)
    print("\nРезультаты сохранены в data/processed")


if __name__ == "__main__":
    main()
