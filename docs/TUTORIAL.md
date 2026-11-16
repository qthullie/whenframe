# Tutorial

## Install

whenframe needs Python 3.10 or later and ffmpeg.

```bash
# ffmpeg: apt install ffmpeg, brew install ffmpeg, or https://ffmpeg.org/download.html
pip install git+https://github.com/qthullie/whenframe
```

The first run downloads the SigLIP 2 model (about 1.5 GB) into
`~/.cache/whenframe` (or `$WHENFRAME_CACHE`). Nothing else leaves your machine.

## Index a folder

```bash
whenframe index ~/Videos
```

Every video under the folder is read at one frame per second. Frames that
barely differ from the previous one are skipped. Run the same command again
later: only new or modified videos are indexed.

The index lives in `~/.local/share/whenframe/library` (or `$WHENFRAME_LIBRARY`,
or `--library DIR`). It takes about 5 MB per hour of video.

```bash
whenframe stats      # what is indexed
whenframe compact    # reclaim space after videos were re-indexed or removed
```

## Search

```bash
whenframe search "a dog jumps into the pool"
```

```
 83%     1:12 - 1:19      /home/me/Videos/2024/summer/IMG_4120.MOV
 71%     0:03 - 0:09      /home/me/Videos/2023/garden.mp4
 ...
```

Each line is a moment: how sure the model is that its best frame matches the
description, where the moment starts and ends, and the file. The model was
trained on text in many languages, so French or Spanish descriptions work too.

## From Python

```python
from whenframe import Library, Searcher

lib = Library("my-library")
lib.add("holiday.mp4")
for hit in Searcher(lib).search("sunset over the sea", k=5):
    print(hit.path, hit.start, hit.end, hit.confidence)
```

`SearchConfig` exposes the few settings of the search (smoothing, where a
moment stops, how many moments per video).

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

The protocol, including the success criteria, is in [PROTOCOL.md](PROTOCOL.md).
