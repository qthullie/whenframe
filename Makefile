PY ?= python3
DIDEMO ?= data/didemo
TRANSLATOR ?= data/opus-mt-en-fr
FILMS ?= data/films
SMOLVLM ?= data/smolvlm-256m
CHARADES ?= data/charades
YOUCOOK2 ?= data/youcook2
UNAV ?= data/unav

.PHONY: install check lint test didemo-extract didemo-tune didemo-eval didemo-explore didemo-round2 tables figures \
	didemo-rerank charades didemo-round3 youcook2 round4 round5 round6 round7 benchmark-figure demo site

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
	$(PY) scripts/so400m.py latency --split test --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/unav.py extract --annotations $(UNAV)/unav100_all.json
	$(PY) scripts/unav.py tune --annotations $(UNAV)/unav100_all.json
	$(PY) scripts/unav.py evaluate --annotations $(UNAV)/unav100_all.json
	$(PY) scripts/motion.py extract --split train --videos $(DIDEMO)/train --annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/motion.py tune --split train --annotations $(DIDEMO)/didemo_train.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/motion.py extract --split test --videos $(DIDEMO)/test --annotations $(DIDEMO)/didemo_test.json
	$(PY) scripts/motion.py evaluate --split test --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json
	$(PY) scripts/thumbnail.py --split test --annotations $(DIDEMO)/didemo_test.json \
		--train-annotations $(DIDEMO)/didemo_train.json

# the fifth round, whenframe's own model (docs/PROTOCOL.md): the training videos streamed and embedded, the model
# trained (pip install -e ".[train]") with the settings chosen on the tuning videos, exported, tested once
ROUND5 = --annotations $(DIDEMO)/didemo_train.json
round5:
	$(PY) scripts/train_data.py $(ROUND5)
	$(PY) scripts/temporal.py texts $(ROUND5)
	$(PY) scripts/temporal.py train $(ROUND5) --name temporal-siglip2-base --steps 3000 --batch 128 --every 250 \
		--lr 1e-4 --warmup 200 --bank-sample 1024 --head 0 --leash 100 --within 2
	$(PY) scripts/temporal.py export $(ROUND5) --model results/temporal/temporal-siglip2-base.npz
	$(PY) scripts/temporal.py evaluate $(ROUND5) --model whenframe/data/temporal-siglip2-base.npz \
		--test-annotations $(DIDEMO)/didemo_test.json --charades-annotations $(CHARADES)/test.parquet \
		--youcook2-annotations $(YOUCOOK2)/val.parquet --unav-annotations $(UNAV)/unav100_all.json

# the sixth round (docs/PROTOCOL.md): So400m and B/32 on the same frames of the four benchmarks, model v2 trained on
# DiDeMo and MSR-VTT (msrvtt_train_9k.json of friedrichor/MSR-VTT in $(MSRVTT)), the choices made on the tuning sets,
# then every candidate tested once
MSRVTT ?= data/msrvtt
ROUND6 = $(ROUND5) --test-annotations $(DIDEMO)/didemo_test.json --charades-annotations $(CHARADES)/test.parquet \
	--youcook2-annotations $(YOUCOOK2)/val.parquet --unav-annotations $(UNAV)/unav100_all.json
V2 = --steps 3000 --batch 128 --every 250 --lr 1e-4 --warmup 200 --bank-sample 1024 --head 0 --leash 100 --within 2 \
	--extra msrvtt --init-seed 0
round6:
	for d in unav charades youcook2; do $(PY) scripts/compare.py extract --dataset $$d; done
	$(PY) scripts/compare.py extract --dataset didemo-train --models siglip2-b32 --videos $(DIDEMO)/train
	$(PY) scripts/compare.py extract --dataset didemo-test --models siglip2-b32 --videos $(DIDEMO)/test
	$(PY) scripts/msrvtt.py extract --annotations $(MSRVTT)/msrvtt_train_9k.json
	$(PY) scripts/msrvtt.py texts --annotations $(MSRVTT)/msrvtt_train_9k.json
	$(PY) scripts/temporal.py train $(ROUND5) --name v2-infonce --between-loss infonce --seed 0 $(V2)
	$(PY) scripts/temporal.py train $(ROUND5) --name v2-sigmoid --between-loss sigmoid --seed 0 $(V2)
	for s in 1 2; do $(PY) scripts/temporal.py train $(ROUND5) --name v2-infonce-s$$s --between-loss infonce --seed $$s $(V2); done
	$(PY) scripts/temporal.py soup $(ROUND5) --name v2-soup \
		--models results/temporal/v2-infonce.npz,results/temporal/v2-infonce-s1.npz,results/temporal/v2-infonce-s2.npz
	$(PY) scripts/temporal.py export $(ROUND5) --model results/temporal/v2-soup.npz --out results/temporal/temporal-siglip2-base-v2.npz
	$(PY) scripts/compare.py tune $(ROUND6) --candidates b32,base,so400m,v1,v2-infonce=results/temporal/v2-infonce.npz@1,v2-sigmoid=results/temporal/v2-sigmoid.npz@1,v2-soup=results/temporal/v2-soup.npz@1
	$(PY) scripts/compare.py tune $(ROUND6) --candidates v2-soup-a0.25=results/temporal/v2-soup.npz@0.25,v2-soup-a0.5=results/temporal/v2-soup.npz@0.5,v2-soup-a0.75=results/temporal/v2-soup.npz@0.75,v2=results/temporal/temporal-siglip2-base-v2.npz@1
	$(PY) scripts/compare.py cost --models siglip2-base,siglip2-b32,siglip2-so400m --videos $(DIDEMO)/test
	$(PY) scripts/compare.py evaluate $(ROUND6) --candidates b32,base,so400m,v1,v2

# the seventh round (docs/PROTOCOL.md): model v3, taught by So400m, trained on more moments. In $(TRAIN7): Charades-STA's
# charades_sta_train.txt, YouCook2's youcookii_train.jsonl and TEMPO-HL's tempoHL+didemo_train.json; the videos are
# read from the Hugging Face Hub one at a time. scripts/round7.py makes the choices on the tuning sets by the rule;
# the test is run once, after they are recorded in docs/PROTOCOL.md. So400m's frames of DiDeMo's and MSR-VTT's
# training videos (moments_data.py teacher, then texts --teacher-texts didemo_all,msrvtt) are needed only if choice 1
# keeps the teacher; it did not (results/round7_choices.json), and round7.py waits for them if it does
TRAIN7 ?= data/train-annotations
round7:
	$(PY) scripts/moments_data.py charades --annotations $(TRAIN7)/charades_sta_train.txt
	$(PY) scripts/moments_data.py youcook2 --annotations $(TRAIN7)/youcookii_train.jsonl
	$(PY) scripts/moments_data.py texts --charades $(TRAIN7)/charades_sta_train.txt \
		--youcook2 $(TRAIN7)/youcookii_train.jsonl --tempo $(TRAIN7)/tempoHL+didemo_train.json
	$(PY) scripts/round7.py $(ROUND6)
	$(PY) scripts/compare.py evaluate --round 7 $(ROUND6) \
		--candidates b32,base,so400m,v1,v2,v3,v3-no-charades,v3-no-youcook2

# whenframe beside published numbers and home-tool approaches (docs/figures/benchmark.png)
benchmark-figure:
	$(PY) scripts/benchmark_figure.py

# the Blender films (Big Buck Bunny, Sintel, Tears of Steel) in $(FILMS): the screenshots, then the site's data
demo:
	$(PY) scripts/demo.py --films $(FILMS)

site:
	$(PY) scripts/site.py --films $(FILMS)
	$(PY) scripts/site.py --preview
