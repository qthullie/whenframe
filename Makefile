PY ?= python3
DIDEMO ?= data/didemo
TRANSLATOR ?= data/opus-mt-en-fr
FILMS ?= data/films
SMOLVLM ?= data/smolvlm-256m
CHARADES ?= data/charades
YOUCOOK2 ?= data/youcook2
UNAV ?= data/unav

.PHONY: install check lint test didemo-extract didemo-tune didemo-eval didemo-explore didemo-round2 tables figures \
	didemo-rerank charades didemo-round3 youcook2 round4 benchmark-figure demo site

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

# a small vision-language model re-orders the first ten moments: SmolVLM-256M-Instruct in $(SMOLVLM), with onnx/
didemo-rerank:
	$(PY) scripts/rerank.py --vlm-dir $(SMOLVLM) --videos $(DIDEMO)/train --annotations $(DIDEMO)/didemo_train.json \
		--train-annotations $(DIDEMO)/didemo_train.json

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

# Charades-STA test with the parameters tuned on DiDeMo, unchanged; after make didemo-tune
charades:
	$(PY) scripts/charades.py extract --videos $(CHARADES)/videos --annotations $(CHARADES)/test.parquet
	$(PY) scripts/charades.py evaluate --annotations $(CHARADES)/test.parquet \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/charades.py prior --annotations $(CHARADES)/test.parquet

# the third round: the DiDeMo videos embedded again with OpenVINO, compared with ONNX Runtime (after didemo-extract)
didemo-round3:
	$(PY) scripts/didemo.py extract --split train --backend openvino --videos $(DIDEMO)/train \
		--annotations $(DIDEMO)/didemo_train.json --list results/didemo_train_videos.txt
	$(PY) scripts/backends.py --split train --annotations $(DIDEMO)/didemo_train.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/didemo.py extract --split test --backend openvino --videos $(DIDEMO)/test \
		--annotations $(DIDEMO)/didemo_test.json --list results/didemo_test_videos.txt
	$(PY) scripts/backends.py --split test --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json

# long videos and speech: YouCook2 validation videos in $(YOUCOOK2)/videos, annotations in $(YOUCOOK2)/val.parquet
youcook2:
	$(PY) scripts/youcook2.py extract --videos $(YOUCOOK2)/videos --annotations $(YOUCOOK2)/val.parquet
	$(PY) scripts/youcook2.py transcribe --videos $(YOUCOOK2)/videos --annotations $(YOUCOOK2)/val.parquet
	$(PY) scripts/youcook2.py tune --annotations $(YOUCOOK2)/val.parquet --train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/youcook2.py evaluate --annotations $(YOUCOOK2)/val.parquet \
		--train-annotations $(DIDEMO)/didemo_train.json

# the fourth round, after didemo-extract, didemo-round3 and youcook2: speech with adaptive spans, SigLIP 2 So400m,
# what is heard (UnAV-100's annotations in $(UNAV)/unav100_all.json; its videos are streamed, none is kept), motion
round4:
	$(PY) scripts/youcook2.py tune-spans --annotations $(YOUCOOK2)/val.parquet \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/youcook2.py evaluate-spans --annotations $(YOUCOOK2)/val.parquet \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/didemo.py extract --split train --model siglip2-so400m --backend openvino --videos $(DIDEMO)/train \
		--annotations $(DIDEMO)/didemo_train.json --list results/didemo_train_videos.txt
	$(PY) scripts/so400m.py tune --split train --annotations $(DIDEMO)/didemo_train.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/didemo.py extract --split test --model siglip2-so400m --backend openvino --videos $(DIDEMO)/test \
		--annotations $(DIDEMO)/didemo_test.json --list results/didemo_test_videos.txt
	$(PY) scripts/so400m.py evaluate --split test --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/unav.py extract --annotations $(UNAV)/unav100_all.json
	$(PY) scripts/unav.py tune --annotations $(UNAV)/unav100_all.json
	$(PY) scripts/unav.py evaluate --annotations $(UNAV)/unav100_all.json
	$(PY) scripts/motion.py extract --split train --videos $(DIDEMO)/train --annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/motion.py tune --split train --annotations $(DIDEMO)/didemo_train.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/thumbnail.py --split test --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json

# whenframe beside published numbers and home-tool approaches (docs/figures/benchmark.png)
benchmark-figure:
	$(PY) scripts/benchmark_figure.py

# the Blender films (Big Buck Bunny, Sintel, Tears of Steel) in $(FILMS): the screenshots, then the site's data
demo:
	$(PY) scripts/demo.py --films $(FILMS)

site:
	$(PY) scripts/site.py --films $(FILMS)
	$(PY) scripts/site.py --preview
