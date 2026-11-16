# Tutorial

## Install

whenframe needs Python 3.10 or later and ffmpeg.

```bash
# ffmpeg: apt install ffmpeg, brew install ffmpeg, or https://ffmpeg.org/download.html
pip install git+https://github.com/qthullie/whenframe
```

The first run downloads the SigLIP 2 model (about 1.5 GB) into
`~/.cache/whenframe` (or `$WHENFRAME_CACHE`). Nothing else goes over the
network: whenframe turns off ONNX Runtime's usage telemetry
(`ORT_DISABLE_TELEMETRY=1`) before loading it.

## Index a folder

```bash
whenframe index ~/Videos
```

Every video under the folder is read at one frame per second. Frames that
barely differ from the previous one are skipped. Run the same command again
later: only new or modified videos are indexed.

A library is built in one of three modes, chosen the first time it is indexed:

| `--mode` | model |
|---|---|
| `standard` (default) | SigLIP 2 B/16 |
| `economy` | SigLIP 2 B/32 (another 1.5 GB download) |
| `precision` | both, their scores combined |

The economy mode costs about a third of the CPU time and finds fewer moments;
the precision mode costs more and, on DiDeMo, found as many as the standard
mode. The measured numbers are in the README and in [RESULTS.md](RESULTS.md),
"Second round".

The index lives in `~/.local/share/whenframe/library` (or `$WHENFRAME_LIBRARY`,
or `--library DIR`). It takes about 5 MB per hour of video, and a search holds
it in memory as float32, about 11 MB per hour of video.

```bash
whenframe stats      # what is indexed
whenframe compact    # reclaim space after videos were re-indexed or removed
```

## Search

```bash
whenframe search "a dog jumps into the pool"
```

```
 83%     1:12 - 1:22      /home/me/Videos/2024/summer/IMG_4120.MOV
 71%     0:03 - 0:13      /home/me/Videos/2023/garden.mp4
 ...
```

Each line is a moment: how sure the model is that its best frame matches the
description, where the moment starts and ends, and the file. About a tenth of
the model's training text was not English, so descriptions in French or Spanish
can work; they were not measured here, and English is the safer bet.

## In the browser

```bash
whenframe serve            # then open http://127.0.0.1:8765
```

The model stays loaded, so each search answers in a fraction of a second. The
page shows the best frame of each moment; a click plays the video from the start
of the moment and stops at its end. It listens on this computer only (`--host`
changes that) and loads nothing from the internet. Browsers play mp4 and webm;
for other formats the page gives the file's folder, to open it in a video
player.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/demo-2-dark.webp">
    <img src="figures/demo-2.webp" alt="whenframe serve on three Blender films: 'a girl hugging a baby dragon' finds three moments of Sintel first" width="800">
  </picture>
</p>

## From Python

```python
from whenframe import Library, Searcher

lib = Library("my-library")
lib.add("holiday.mp4")
for hit in Searcher(lib).search("sunset over the sea", k=5):
    print(hit.path, hit.start, hit.end, hit.confidence)
```

Each `whenframe search` loads the text model first, which takes a few seconds;
from Python, keep one `Searcher` and the following searches take tens of
milliseconds.

`SearchConfig` holds the few settings of the search: fixed windows or spans
grown around peaks, the window length, how many moments per video, and the
weight of the query-bank correction (0 turns it off). The defaults are the ones
chosen on DiDeMo training videos.

## Reproduce the benchmark

The evaluation uses DiDeMo (Hendricks et al., 2017). The annotations and an
mp4 copy of the videos are on the Hugging Face Hub
(`friedrichor/DiDeMo`: `raw_data/didemo_{train,test}.json`,
`DiDeMo_Videos_mp4_test.tar`, `DiDeMo_Videos_mp4_train.tar.part-000`). Put them
under `data/didemo/` as:

```
data/didemo/didemo_train.json
data/didemo/didemo_test.json
data/didemo/train/   *.mp4 from the first part of the training archive
data/didemo/test/    *.mp4 from the test archive
```

then:

```bash
make didemo-extract   # frame and text embeddings, scene cuts, timings
make didemo-tune      # parameters of every method, on training videos only
make didemo-eval      # the test numbers in docs/RESULTS.md
make tables           # rewrite the tables from results/*.json
```

The second round needs the English-to-French model as well
(`Xenova/opus-mt-en-fr` on the Hugging Face Hub: `config.json`,
`tokenizer.json`, `onnx/encoder_model.onnx`, `onnx/decoder_model.onnx`, in
`data/opus-mt-en-fr/`); MobileCLIP-S2 is downloaded like SigLIP 2:

```bash
make didemo-explore   # the other score corrections tried on tuning videos
make didemo-round2    # MobileCLIP-S2, French descriptions, calibration, tuning, test
make tables
```

The protocol, including the success criteria, is in [PROTOCOL.md](PROTOCOL.md).
