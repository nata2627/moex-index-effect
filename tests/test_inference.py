"""Тесты статистического вывода и поправок на множественность."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import inference


def test_holm_matches_manual_computation():
    """Холм: p, умноженное на число оставшихся гипотез, монотонно."""
    p_values = pd.Series([0.001, 0.02, 0.04, 0.3, 0.8])
    adjusted = inference.holm_correction(p_values)
    assert adjusted.iloc[0] == pytest.approx(0.005)   # 0.001 * 5
    assert adjusted.iloc[1] == pytest.approx(0.08)    # 0.02 * 4
    assert (adjusted.sort_values().diff().dropna() >= 0).all()


def test_benjamini_hochberg_is_less_conservative_than_holm():
    p_values = pd.Series([0.001, 0.02, 0.04, 0.3, 0.8])
    holm = inference.holm_correction(p_values)
    fdr = inference.benjamini_hochberg(p_values)
    assert (fdr <= holm + 1e-12).all()
    assert fdr.iloc[1] == pytest.approx(0.05)         # 0.02 * 5 / 2


def test_clustering_widens_standard_error():
    """Наблюдения, зависимые внутри кластера, дают большую ошибку.

    Это главный довод в пользу кластеризации: сорок два наблюдения, собранные
    в шесть дат, несут информации меньше, чем сорок два независимых.
    """
    rng = np.random.default_rng(3)
    cluster_shock = rng.normal(0, 0.05, 6)
    values, clusters = [], []
    for cluster, shock in enumerate(cluster_shock):
        for _ in range(7):
            values.append(shock + rng.normal(0, 0.005))
            clusters.append(cluster)

    values = pd.Series(values)
    naive = inference.cross_sectional_test(values)
    clustered = inference.clustered_test(values, pd.Series(clusters))

    naive_se = naive["std"] / np.sqrt(naive["n"])
    assert clustered["se"] > naive_se
    assert clustered["n_clusters"] == 6


def test_cross_sectional_test_reports_robust_alternatives():
    """Одно большое значение при медиане около нуля видно по знаковому тесту."""
    values = pd.Series([-0.01, -0.005, 0.002, -0.003, 0.60])
    result = inference.cross_sectional_test(values)
    assert result["mean"] > 0
    assert result["median"] < 0
    assert result["share_positive"] == pytest.approx(0.4)


def test_small_sample_returns_nan_instead_of_guessing():
    result = inference.cross_sectional_test(pd.Series([0.1, 0.2]))
    assert np.isnan(result["p_value"])
