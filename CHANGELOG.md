# Changelog

## Unreleased

- `whenframe film` cuts a film from a story, one shot per sentence, each the
  best moment found for it that the film does not use yet (a sentence ending
  with `#2` takes its second best; `--plan` prints the shots). `whenframe
  supercut` cuts the best moments of one description in the order they were
  filmed. A shot lasts three seconds from the moment's best frame; a video of
  another shape is shown over a blurred copy of itself, a photo slowly zoomed
  into; the shots are joined with cross-fades, the sentences drawn in the
  picture and written in an .srt, and `--music` puts a music under the film
  (`whenframe/film.py`, ffmpeg).
- The demo site opens on a film cut from someone's phone, three stories (one
  written in French) and one of the visitor's own, planned in the browser as
  `whenframe film` plans them, a section on whenframe's model and its numbers
  beside the other ways, and a music composed by `scripts/music.py` (CC0), off
  until asked for. The page now embeds its sentences one at a time, as a search
  does: for all 18 suggestions it shows what the tool finds (17 before, where
  a batch's padding had turned a near tie), and it plans the films the tool
  plans (`scripts/site_check.mjs`).
- The text towers in 8 bits: SigLIP 2 B/16's and PE core B/16's text towers
  keep their weights in 8 bits (NNCF, one scale per output channel), from
  whenframe's release models-2, built by GitHub Actions to the bytes measured
  (`scripts/text8.py`). A new library downloads 1.4 GB of models instead of
  2.6 GB, and a search of the demo's library holds 1.4 GB of memory instead
  of 4.8 GB; on the four test sets they find what the towers in full
  precision found (mean 43.70 against 43.72, twelfth round). The query banks
  are embedded by them; libraries already indexed keep their frames. Both
  text towers run with OpenVINO, whatever `--backend` says.
- Descriptions in French: a search translates a description it takes for
  French into English on the computer first (Opus-MT fr-en, from the
  University of Helsinki, Apache 2.0: 430 MB, downloaded the first time one is
  needed; about a tenth of a second a description), then searches it as an
  English one. `whenframe search` and the local page say what they searched;
  `--as-typed` searches the description as written. On the four test sets,
  French descriptions found 10.76 points more translated than as they were
  (mean 40.04 against 29.28; the English ones 43.72), and the detector took no
  English description for French (twelfth round, `scripts/round12.py`).
- The demo site searches someone's phone instead of three Blender films: 67
  videos and 165 photos that 217 people shared on Flickr under CC BY 2.0
  (YFCC100M), chosen by hand, named as a phone names them, with whenframe's
  one model (both encoders, v3 and v4) as `whenframe index` builds a library.
  The page searches them as the tool does (`scripts/site_check.mjs` checks
  every suggestion against the tool's own answers), photos and video moments
  together, and credits each author (`site/credits.html`). The deploy workflow
  downloads the files from YFCC100M's public copy and makes them smaller,
  without their sound or metadata (`scripts/site_media.py`); the demo's text
  server embeds a typed description with both text towers.
- One model: a new library is indexed with SigLIP 2 B/16 and Meta's PE core
  B/16, each frame decoded once, each encoder's frames through whenframe's own
  model trained on them (v3 and v4), and searched through both together. On
  the four test sets it found the most of everything measured (eleventh round,
  below). The standard, economy, precision and large modes are gone, and with
  them `--mode`; libraries built in them are searched and kept up to date as
  they were built. OpenVINO is a dependency now (PE core B/16's image tower
  runs with it), the first run downloads about 2.6 GB, and a search holds both
  text towers in memory (at most 4.7 GB for a search of three films, 37 minutes).
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
- The eleventh round tested that fusion once (`scripts/round11.py`): v3 and
  v4 together find more than v3 alone on each of the four tests, mean 43.72
  against 41.90 (sign test p = 4.9e-12), for 1.74 times the indexing cost;
  So400m's mean is 40.63, for 3.4 times. Criteria 33 to 35 are met: the
  large mode became the fusion, then whenframe's only model (above). PE core
  B/16's image tower, which no model hub serves, comes from the release
  models-1, built by GitHub Actions to the bytes measured
  (`.github/workflows/models.yml`); its SHA-256 is checked when it is
  downloaded. v5, whenframe's model taught by the fusion
  on SigLIP 2 B/16's frames alone, raised v3's tuning mean by 0.55 points, not
  the 1 asked for, so it was not tested.
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
