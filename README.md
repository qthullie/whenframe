# whenframe

Find a moment in your own videos by describing it. whenframe indexes a folder
of videos on your computer, with no GPU and nothing sent anywhere, and answers
"the dog jumps into the pool" with the video and the seconds where it happens.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/results-dark.png">
    <img src="docs/figures/results.png" alt="Share of DiDeMo test descriptions whose moment is among the first five results, for each search method" width="800">
  </picture>
</p>

<!-- table readme-headline -->
Searching 979 personal videos at once (DiDeMo test) for 3,902 descriptions written by people, whenframe puts the right moment among its first five results **27.2 %** of the time. Indexing one keyframe per shot, as some home video tools do, finds it 2.3 % of the time, and the best of the two baselines (fixed windows) 25.0 %.
<!-- /table -->

## Use

```bash
pip install git+https://github.com/qthullie/whenframe   # Python 3.10+, and ffmpeg on the PATH
whenframe index ~/Videos
whenframe search "the dog jumps into the pool"
```

```
 83%     1:12 - 1:22      /home/me/Videos/2024/summer/IMG_4120.MOV
 71%     0:03 - 0:13      /home/me/Videos/2023/garden.mp4
```

The first run downloads the SigLIP 2 model (1.5 GB). Indexing again only reads
new or modified videos. More in [docs/TUTORIAL.md](docs/TUTORIAL.md).

Nothing else goes over the network. ONNX Runtime 1.29 sends usage telemetry to
Microsoft by default; whenframe sets `ORT_DISABLE_TELEMETRY=1` before loading
it (checked by logging every outbound connection of an indexing and search run).

## How it works

1. ffmpeg reads one frame per second; frames that barely differ from the last
   one kept are skipped.
2. [SigLIP 2](https://arxiv.org/abs/2502.14786) embeds the frames and the
   description (ONNX Runtime, CPU). Its sigmoid gives each frame an absolute
   match score, so frames from different videos compare directly.
3. Frames that match almost any description are marked down, using a bank of
   4,096 unrelated descriptions: query-bank normalisation
   ([Bogolin et al., CVPR 2022](https://arxiv.org/abs/2112.12777)), applied to
   video frames.
4. In each of the best videos, up to three 10-second moments are placed around
   the best frames, and all moments are ranked together.

## Results

DiDeMo test split, with the protocol fixed in advance ([docs/PROTOCOL.md](docs/PROTOCOL.md)).
Every number below is written by `scripts/tables.py` from `results/didemo_test.json`.

<!-- table didemo-criteria -->
| criterion (fixed in advance) | measured on test | met |
|---|---|---|
| 1. adaptive spans beat both baselines by at least 2 points | -1.8 points | no |
| 2. skipping repeats encodes at most 70 % of frames and loses at most 1 point | 91 % of frames, +0.2 points | no |
| 3. indexing takes at most 10 min of CPU time per hour of video | 24.9 min | no |
| 4. the query bank adds at least 1 point to adaptive spans | +1.6 points | yes |
| 5. the query bank adds at least 1 point to fixed windows | +2.2 points | yes |
<!-- /table -->

All tables, the costs and the figures: [docs/RESULTS.md](docs/RESULTS.md).

## Limits

- It sees frames, not motion or sound: "the camera pans left" or "someone
  laughs" are out of its reach.
- It was measured on the first 30 seconds of short personal videos; long
  videos and other kinds of footage were not measured.
- Descriptions in English. About a tenth of SigLIP 2's training text was in
  other languages; how well they work here was not measured.
- Indexing is slow on a small machine: 25 minutes of CPU time per hour of
  video on the 4-vCPU machine of the benchmark (7.4 minutes of waiting).
- Three of the five criteria set before the test run were not met; the results
  page says which and why.

## Documents

- [docs/RESEARCH.md](docs/RESEARCH.md): what already exists, and what is new here
- [docs/PROTOCOL.md](docs/PROTOCOL.md): the evaluation, written before the test run, and every change made to it
- [docs/RESULTS.md](docs/RESULTS.md): all the numbers
- [docs/TUTORIAL.md](docs/TUTORIAL.md): installing, indexing, searching, reproducing the benchmark

## Credits

- DiDeMo: Hendricks et al., *Localizing Moments in Video with Natural
  Language*, ICCV 2017.
- SigLIP 2: Tschannen et al., 2025. The weights, downloaded at the first run,
  are Google's, under the Apache 2.0 licence.
- The query bank holds embeddings of DiDeMo training descriptions (BSD 2-Clause);
  see [whenframe/data/NOTICE](whenframe/data/NOTICE).

## Licence

MIT.
