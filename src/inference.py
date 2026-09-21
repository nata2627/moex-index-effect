"""Статистический вывод: тесты значимости и поправки на множественность.

Главная трудность выборки — не её размер сам по себе, а зависимость внутри
неё. Московская биржа пересматривает состав индекса раз в квартал, поэтому
события приходят пачками: 42 события приходятся на 21 дату, а 21 марта 2025
года их сразу шесть. Аномальные доходности внутри одной даты коррелированы
через общий рыночный шок, который рыночная модель снимает лишь частично.
Если считать события независимыми наблюдениями, стандартная ошибка окажется
заниженной, а значимость — выдуманной.

Поэтому вывод строится на трёх уровнях, от наивного к надёжному:

1. кросс-секционный t-тест — считает события независимыми, приводится только
   для сравнения;
2. стандартные ошибки, кластеризованные по дате события — учитывают
   зависимость внутри даты, но опираются на асимптотику по числу кластеров,
   а кластеров у нас от 6 до 10;
3. рандомизационный вывод на плацебо-датах — не требует ни асимптотики, ни
   предположений о распределении, и потому служит основным критерием.

Третий пункт здесь не «дополнительная проверка», как его обычно подают,
а единственный инструмент, чьи предпосылки на этих данных выполняются.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cross_sectional_test(values: pd.Series) -> dict:
    """Наивный t-тест среднего против нуля плюс устойчивые непараметрические.

    Знаковый тест и тест Уилкоксона добавлены потому, что распределение CAR
    имеет тяжёлые хвосты: одна бумага с эффектом в 40% способна создать
    «значимое» среднее при медиане около нуля.
    """
    clean = pd.Series(values).dropna()
    if len(clean) < 3:
        return {"n": len(clean), "mean": np.nan, "t_stat": np.nan, "p_value": np.nan,
                "median": np.nan, "p_sign": np.nan, "p_wilcoxon": np.nan}

    t_stat, p_value = stats.ttest_1samp(clean, 0.0)
    positive = int((clean > 0).sum())
    p_sign = stats.binomtest(positive, len(clean), 0.5).pvalue
    try:
        p_wilcoxon = stats.wilcoxon(clean).pvalue
    except ValueError:
        p_wilcoxon = np.nan

    return {
        "n": len(clean),
        "mean": clean.mean(),
        "std": clean.std(),
        "t_stat": t_stat,
        "p_value": p_value,
        "median": clean.median(),
        "share_positive": positive / len(clean),
        "p_sign": p_sign,
        "p_wilcoxon": p_wilcoxon,
    }


def clustered_test(values: pd.Series, clusters: pd.Series) -> dict:
    """Среднее со стандартной ошибкой, кластеризованной по дате события.

    Оценка строится на средних внутри кластеров: при равных размерах кластеров
    это эквивалентно обычной кластерной поправке, но прозрачнее и устойчивее
    при малом их числе. Степени свободы берутся как G - 1.
    """
    frame = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    if frame["cluster"].nunique() < 3:
        return {"n_clusters": frame["cluster"].nunique(), "mean": np.nan,
                "se": np.nan, "t_stat": np.nan, "p_value": np.nan}

    cluster_means = frame.groupby("cluster")["value"].mean()
    n_clusters = len(cluster_means)
    mean = cluster_means.mean()
    se = cluster_means.std(ddof=1) / np.sqrt(n_clusters)
    t_stat = mean / se if se > 0 else np.nan
    p_value = 2 * stats.t.sf(abs(t_stat), df=n_clusters - 1) if np.isfinite(t_stat) else np.nan

    return {"n_clusters": n_clusters, "mean": mean, "se": se,
            "t_stat": t_stat, "p_value": p_value,
            "ci_low": mean - stats.t.ppf(0.975, n_clusters - 1) * se,
            "ci_high": mean + stats.t.ppf(0.975, n_clusters - 1) * se}


def holm_correction(p_values: pd.Series) -> pd.Series:
    """Поправка Холма: контроль вероятности хотя бы одной ложной находки.

    Применяется к основным гипотезам, которых немного. Холм равномерно
    мощнее Бонферрони и при этом не требует независимости тестов — это важно,
    поскольку оценки на пересекающихся окнах заведомо зависимы.
    """
    values = pd.Series(p_values).dropna()
    if values.empty:
        return pd.Series(dtype=float)

    order = values.sort_values()
    n = len(order)
    adjusted, running_max = [], 0.0
    for rank, p in enumerate(order):
        candidate = min(1.0, (n - rank) * p)
        running_max = max(running_max, candidate)
        adjusted.append(running_max)
    return pd.Series(adjusted, index=order.index).reindex(pd.Series(p_values).index)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Контроль доли ложных открытий.

    Применяется к сетке проверок устойчивости: там гипотез десятки, они
    сильно зависимы, и контроль FWER по Холму оставил бы без значимости даже
    настоящий эффект. FDR допускает известную долю ложных срабатываний в
    обмен на мощность, что для разведочной части уместно.
    """
    values = pd.Series(p_values).dropna()
    if values.empty:
        return pd.Series(dtype=float)

    order = values.sort_values()
    n = len(order)
    adjusted, running_min = [], 1.0
    for rank, p in reversed(list(enumerate(order))):
        candidate = min(1.0, p * n / (rank + 1))
        running_min = min(running_min, candidate)
        adjusted.append(running_min)
    adjusted.reverse()
    return pd.Series(adjusted, index=order.index).reindex(pd.Series(p_values).index)


def caar_path(abnormal: pd.DataFrame) -> pd.DataFrame:
    """Средняя накопленная аномальная доходность по дням окна события.

    Доверительный коридор строится по кластерам-датам: в каждый день окна
    события берутся средние по кластерам, и разброс оценивается по ним.
    """
    rows = []
    for t, frame in abnormal.groupby("t"):
        result = clustered_test(frame["car"], frame["event_date"])
        naive = frame["car"].dropna()
        rows.append({
            "t": t,
            "caar": naive.mean(),
            "n_events": len(naive),
            "se_clustered": result["se"],
            "ci_low": result.get("ci_low", np.nan),
            "ci_high": result.get("ci_high", np.nan),
        })
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
