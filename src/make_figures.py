"""Построение всех графиков отчёта из сохранённых результатов расчёта."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import did, moex, pipeline, plots, robustness, run_analysis

FIGURES = moex.PROJECT_ROOT / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    paths = pd.read_parquet(moex.DATA_PROCESSED / "caar_paths.parquet")
    abnormal = pd.read_parquet(moex.DATA_PROCESSED / "abnormal_returns.parquet")
    panel = pd.read_parquet(moex.DATA_PROCESSED / "liquidity_panel.parquet")
    placebo_results = pd.read_parquet(moex.DATA_PROCESSED / "placebo_results.parquet")

    plots.plot_caar_paths(paths, FIGURES / "01_car_event_time.png")
    print("01_car_event_time.png")

    plots.plot_parallel_trends(panel, FIGURES / "02_parallel_trends.png")
    print("02_parallel_trends.png")

    # Для плацебо показываем главное окно — период после события.
    window = run_analysis.CAR_WINDOWS["[+1,+60] дальний откат"]
    distributions, observed, percentiles = {}, {}, {}
    for event_type in ("включение", "исключение"):
        path = moex.DATA_PROCESSED / f"placebo_{event_type}_{window[0]}_{window[1]}.parquet"
        if not path.exists():
            continue
        distributions[event_type] = pd.read_parquet(path)["placebo_car"]
        row = placebo_results[(placebo_results["event_type"] == event_type) &
                              (placebo_results["window"] == "[+1,+60] дальний откат")].iloc[0]
        observed[event_type] = row["observed"]
        percentiles[event_type] = row["percentile"]
    plots.plot_placebo(distributions, observed, percentiles,
                       FIGURES / "03_placebo.png")
    print("03_placebo.png")

    plots.plot_heatmap(abnormal, FIGURES / "04_heatmap.png")
    print("04_heatmap.png")

    plots.plot_liquidity(panel, FIGURES / "05_liquidity.png")
    print("05_liquidity.png")

    robustness_path = moex.DATA_PROCESSED / "robustness.parquet"
    if not robustness_path.exists():
        data = pipeline.load()
        robustness.run_all(data).to_parquet(robustness_path, index=False)
    plots.plot_robustness(pd.read_parquet(robustness_path),
                          FIGURES / "06_robustness.png")
    print("06_robustness.png")


if __name__ == "__main__":
    main()
