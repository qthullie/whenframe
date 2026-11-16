PY ?= python3
DIDEMO ?= data/didemo

.PHONY: install check lint test didemo-extract didemo-tune didemo-eval tables

install:
	$(PY) -m pip install -e ".[dev]"

check: lint test

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -q

# DiDeMo: see docs/TUTORIAL.md for where the videos come from
didemo-extract:
	$(PY) scripts/didemo.py extract --split train --videos $(DIDEMO)/train --annotations $(DIDEMO)/didemo_train.json \
		--list results/didemo_train_videos.txt
	$(PY) scripts/didemo.py extract --split test --videos $(DIDEMO)/test --annotations $(DIDEMO)/didemo_test.json

didemo-tune:
	$(PY) scripts/didemo.py tune --split train --annotations $(DIDEMO)/didemo_train.json \
		--train-annotations $(DIDEMO)/didemo_train.json

didemo-eval:
	$(PY) scripts/didemo.py evaluate --split test --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json

tables:
	$(PY) scripts/tables.py
