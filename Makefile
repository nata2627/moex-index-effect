VENV := .venv
PY := $(VENV)/bin/python

.PHONY: help venv data events test clean

help:
	@echo "make venv    — создать окружение и поставить зависимости"
	@echo "make data    — выгрузить состав индекса, котировки IMOEX и бумаг событий"
	@echo "make events  — собрать таблицу событий с диагностикой данных"
	@echo "make test    — прогнать тесты"

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) etl/fetch_index_history.py
	$(PY) etl/fetch_index_prices.py
	$(PY) etl/fetch_quotes.py

events:
	$(PY) src/build_event_table.py

test:
	$(VENV)/bin/pytest -q tests

clean:
	rm -rf data/raw/* data/interim/* data/processed/*
