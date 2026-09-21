VENV := .venv
PY := $(VENV)/bin/python

.PHONY: help venv data splits events analysis figures test all clean

help:
	@echo "make venv     — создать окружение и поставить зависимости"
	@echo "make data     — выгрузить состав индекса, котировки IMOEX, бумаг и контроля"
	@echo "make splits   — найти дробления и выпустить скорректированные котировки"
	@echo "make events   — собрать таблицу событий с диагностикой данных"
	@echo "make analysis — аномальные доходности, плацебо, разность разностей"
	@echo "make figures  — построить графики в figures/"
	@echo "make test     — прогнать тесты"
	@echo "make all      — весь путь от выгрузки до графиков"

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) etl/fetch_index_history.py
	$(PY) etl/fetch_index_prices.py
	$(PY) etl/fetch_quotes.py
	$(PY) etl/fetch_quotes.py --controls

splits:
	$(PY) etl/detect_splits.py

events:
	$(PY) src/build_event_table.py

analysis:
	$(PY) src/run_analysis.py

figures:
	$(PY) src/make_figures.py

test:
	$(VENV)/bin/pytest -q tests

all: data splits analysis figures

clean:
	rm -rf data/raw/* data/interim/* data/processed/* figures/*.png
