"""Клиент к открытому ISS API Московской биржи.

ISS отдаёт данные постранично: в ответе лежит не более ~100 строк, следующая
порция запрашивается параметром ``start``. Здесь это спрятано в
:func:`fetch_paged`, чтобы вызывающий код работал с целым DataFrame.

Ключа и регистрации не требуется, но биржа ограничивает частоту запросов,
поэтому между страницами выдерживается пауза, а на сетевые сбои — ретраи.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ISS_BASE = "https://iss.moex.com/iss"

# Корень проекта — на два уровня выше этого файла (src/moex.py -> index_effect/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Пауза между запросами: биржа не публикует лимит явно, 0.2 с заведомо безопасно
# и при ~30 бумагах x 5 лет даёт приемлемое общее время выгрузки.
REQUEST_DELAY_SEC = 0.2
MAX_RETRIES = 4
TIMEOUT_SEC = 30

_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Одна HTTP-сессия на процесс — переиспользование соединения."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": "index-effect-research/1.0"})
    return _session


def _get_json(url: str, params: dict) -> dict:
    """GET с ретраями и экспоненциальной задержкой."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = get_session().get(url, params=params, timeout=TIMEOUT_SEC)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"ISS не ответил после {MAX_RETRIES} попыток: {url}") from last_error


def fetch_block(path: str, block: str, params: dict | None = None) -> pd.DataFrame:
    """Одна страница одного блока ответа ISS.

    ISS упаковывает каждый блок как ``{"columns": [...], "data": [[...], ...]}``.
    """
    params = dict(params or {})
    params["iss.meta"] = "off"
    payload = _get_json(f"{ISS_BASE}/{path}.json", params)
    if block not in payload:
        raise KeyError(f"В ответе {path} нет блока '{block}'. Есть: {list(payload)}")
    return pd.DataFrame(payload[block]["data"], columns=payload[block]["columns"])


def fetch_paged(path: str, block: str, params: dict | None = None,
                page_limit: int = 500) -> pd.DataFrame:
    """Все страницы блока: идём по ``start``, пока ISS не вернёт пустую порцию.

    ``page_limit`` — предохранитель от бесконечного цикла, если биржа начнёт
    игнорировать ``start`` (тогда страницы повторялись бы вечно).
    """
    params = dict(params or {})
    frames: list[pd.DataFrame] = []
    start = 0
    for _ in range(page_limit):
        page = fetch_block(path, block, {**params, "start": start})
        if page.empty:
            break
        # Часть справочных эндпоинтов ISS игнорирует ``start`` и на каждый
        # запрос отдаёт один и тот же полный ответ. Без этой проверки цикл
        # молча накрутил бы сотни копий одних и тех же строк.
        if frames and page.equals(frames[-1]):
            raise RuntimeError(
                f"Эндпоинт {path} не поддерживает пагинацию: повторная страница "
                f"на start={start}. Используйте fetch_block."
            )
        frames.append(page)
        start += len(page)
        time.sleep(REQUEST_DELAY_SEC)
    else:
        raise RuntimeError(f"Пагинация {path} не завершилась за {page_limit} страниц")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def index_composition(index_id: str) -> pd.DataFrame:
    """История членства бумаг в индексе ``index_id``.

    Колонки: ticker, from, till, tradingsession. Одна строка на бумагу —
    последний интервал членства (ISS не отдаёт более ранние интервалы, если
    бумага включалась в индекс повторно).
    """
    path = f"statistics/engines/stock/markets/index/analytics/{index_id}/tickers"
    frame = fetch_block(path, "tickers")
    frame = frame.rename(columns={"from": "date_from", "till": "date_till"})
    frame["date_from"] = pd.to_datetime(frame["date_from"])
    frame["date_till"] = pd.to_datetime(frame["date_till"])
    frame["index_id"] = index_id
    return frame


def share_history(secid: str, date_from: str, date_till: str) -> pd.DataFrame:
    """Дневные котировки бумаги в основном режиме торгов TQBR."""
    path = f"history/engines/stock/markets/shares/boards/TQBR/securities/{secid}"
    return fetch_paged(path, "history", {"from": date_from, "till": date_till})


def index_history(index_id: str, date_from: str, date_till: str) -> pd.DataFrame:
    """Дневная история значений индекса."""
    path = f"history/engines/stock/markets/index/securities/{index_id}"
    return fetch_paged(path, "history", {"from": date_from, "till": date_till})


def tqbr_securities() -> pd.DataFrame:
    """Справочник бумаг режима TQBR — пул кандидатов в контрольную группу."""
    # Этот эндпоинт отдаёт весь справочник одной страницей и игнорирует start.
    return fetch_block("engines/stock/markets/shares/boards/TQBR/securities",
                       "securities")
