# Changelog

## Unreleased

- whenframe's own model: a temporal adapter of 1.45 million weights rewrites
  each frame's SigLIP 2 embedding with what the frames around it show. A new
  library in the standard mode goes through it (`whenframe index
  --no-temporal` leaves it out). Model v3, trained on the moments described
  in the training videos of DiDeMo (with TEMPO-HL's sentences), MSR-VTT,
  Charades-STA and YouCook2; its weights are for research and non-commercial
  use (`whenframe/data/NOTICE`).
- The seventh round tests v3 once beside the sixth round's five: mean of the
  four primary metrics 41.90, against 40.63 for So400m (`--mode large`, sign
  test p = 0.00086), 38.65 for v2 and 38.07 for B/16 alone; on DiDeMo, the
  right moment among the first five 29.4 % of the time, against 29.5 % with
  So400m for 3.4 times the CPU time. Its lead is on Charades-STA and YouCook2,
  whose training videos it learnt from: trained without each one's videos,
  46.5 and 26.1 %, against 50.4 and 28.5 % for So400m.
- Beside the systems published for searching a whole library of DiDeMo's
  test videos, trained on DiDeMo on a GPU: whenframe puts the right video
  among its first five 48.1 % of the time, against 25.4 % for the best of
  them (EventFormer, 2024), and first 25.3 % against 8.5 %
  (`results/published_didemo.json`, as their authors report them).
- The eighth round tried teaching whenframe's model a library from automatic
  descriptions of its own frames (Florence-2-base, one frame in each 10
  seconds; `scripts/learn.py`, `scripts/round8.py`). None of its three
  recipes found more than the starting models on the tuning sets (mean 37.49
  at best, against 37.82), so it was not tested and the tool is unchanged.
  Describing costs about 20 CPU minutes per hour of video
  ([docs/RESULTS.md](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md)).
- The demo site's frames go through whenframe's model (v3), as a new library's
  do; they were SigLIP 2 B/16's alone. The site can also search a description
  of the visitor's own, embedded by a small text server (`demo-server/`: no
  log, about 1.5 GB of memory, no GPU), once one is deployed and named in
  `site/config.json`.
- The ninth round measured Perception Encoder's PE core B/16 in place of
  SigLIP 2 B/16, at the same indexing cost (1.03 times), and whenframe's
  model trained on its frames with v3's recipe (v4; `scripts/pe_core.py`,
  `scripts/round9.py`). Alone, PE finds more than SigLIP 2 B/16 (mean 39.51
  against 38.07, sign test p = 2.8e-07). v4 finds as much as v3 (41.69
  against 41.90, p = 0.77): more on DiDeMo, Charades-STA and UnAV-100, 4.7
  points less on YouCook2. Criteria 30 and 31 are not met, so the tool keeps
  SigLIP 2 B/16 and v3.
- The tenth round screened three ideas on the tuning sets, with no test
  (`scripts/screen.py`). v3's scores fused with v4's raise the mean of the
  four tuning metrics by 1.41 to 1.70 points, for both encoders' indexing
  time (about 1.7 times SigLIP 2 B/16's alone); this is proposed for a test.
  Guidance, each description pushed away from itself without its action,
  lowers it by 0.83 to 3.79 points. Learning a library only from the
  automatic descriptions that find their own 10 seconds lowers it by 0.78.
  The tool is unchanged.
- The sixth round compares SigLIP 2 B/32, B/16, So400m and B/16 with the
  model, on the same frames of DiDeMo, Charades-STA, YouCook2 and UnAV-100:
  mean of the four primary metrics 36.80, 38.07, 40.63 and 38.65, each step
  significant; on DiDeMo, 29.2 % of the moments among the first five with
  the model instead of 27.6 %, for 0.04 more CPU minutes per hour of video
  ([docs/RESULTS.md](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md)).
- Every encoder's indexing cost measured the same way: 4.3 (B/32), 7.2 (B/16)
  and 24.9 (So400m) CPU minutes per hour of video with OpenVINO.
- A search weighs what is said and what is heard as chosen with its library's
  model on the tuning sets, as the benchmarks measure each mode: sound weight
  1.5 with B/32 and So400m, and whenframe's model and B/32 count what is said
  up to 5 seconds around a frame. Every mode used the standard one's before
  (sound weight 2). whenframe's model keeps its own in its file
  (`scripts/temporal.py export --settings`).
- `scripts/train_data.py`, `scripts/msrvtt.py`, `scripts/moments_data.py`,
  `scripts/temporal.py`, `scripts/round7.py` and `scripts/compare.py` stream
  the training archives, train the model with JAX (`pip install
  'whenframe[train]'`), make the choices on the tuning sets and run the
  comparison; a head placing moments, SigLIP's own loss, WiSE-FT's scaling,
  So400m's losses as a teacher, a larger adapter and, in the seventh round, a
  soup of three runs were tried and not kept.

## 0.1.0

The first release. Every number below was measured with the protocol of
[docs/PROTOCOL.md](https://github.com/qthullie/whenframe/blob/main/docs/PROTOCOL.md), fixed before each test run; all of them,
and the criteria that were not met, are in [docs/RESULTS.md](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md).

- `whenframe index`, `search`, `serve`, `watch`, `stats`, `compact`: a library
  of videos and photos, indexed once on the CPU and searched by description,
  in the terminal or on a local page that plays each moment.
- SigLIP 2 frame embeddings, query-bank correction of the frames that match
  any description, 10-second moments around the best frames, all videos
  ranked together: the right moment among the first five 27.2 % of the time on
  the 979 DiDeMo test videos.
- Four modes: `standard`, `economy` (SigLIP 2 B/32), `precision` (both) and
  `large` (SigLIP 2 So400m, 29.5 %).
- OpenVINO when installed (`pip install 'whenframe[fast]'`): 6.5 CPU minutes per
  hour of video instead of 24.9 for the standard mode.
- `--speech`: Whisper on the CPU, what is said searched with the frames
  (YouCook2: 12.3 % to 18.8 %, 25.8 % with `--moments spans`).
- `--sounds`: CLAP, what is heard searched with the frames (UnAV-100: 42.9 % to
  53.2 % of right moments among the first ten).
- `--moments auto` (the default): in a library whose videos last more than a
  minute, moments as long as the frames match; 10-second windows otherwise.
- `serve --jellyfin`, `serve --immich`: each moment linked to its video in
  Jellyfin, or to its video or photo in Immich.
- Libraries larger than a quarter of the memory are searched from their files.
- Besides downloading the models, nothing goes over the network: ONNX
  Runtime's and OpenVINO's usage reports are turned off.
