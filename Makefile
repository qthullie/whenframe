PY ?= python3
DIDEMO ?= data/didemo
TRANSLATOR ?= data/opus-mt-en-fr

.PHONY: install check lint test didemo-extract didemo-tune didemo-eval didemo-explore didemo-round2 tables figures

install:
	$(PY) -m pip install -e ".[dev]"

check: lint test
	$(PY) scripts/tables.py --check

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

figures:
	$(PY) scripts/figures.py --annotations $(DIDEMO)/didemo_test.json

# what was tried on the tuning videos after the first test run
didemo-explore:
	$(PY) scripts/explore.py --annotations $(DIDEMO)/didemo_train.json --train-annotations $(DIDEMO)/didemo_train.json

# the second round of the protocol: MobileCLIP-S2, French descriptions, then one more test run
didemo-round2:
	$(PY) scripts/didemo.py extract --split train --model mobileclip-s2 --videos $(DIDEMO)/train \
		--annotations $(DIDEMO)/didemo_train.json --list results/didemo_train_videos.txt
	$(PY) scripts/didemo.py extract --split test --model mobileclip-s2 --videos $(DIDEMO)/test \
		--annotations $(DIDEMO)/didemo_test.json --list results/didemo_test_videos.txt
	$(PY) scripts/didemo.py translate --split test --annotations $(DIDEMO)/didemo_test.json --translator-dir $(TRANSLATOR)
	$(PY) scripts/didemo.py calibrate --split train --model mobileclip-s2 --annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/didemo.py tune --split train --round 2 --annotations $(DIDEMO)/didemo_train.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/didemo.py evaluate --split test --round 2 --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json
