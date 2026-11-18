# Changelog

## Unreleased

- whenframe's own model: a temporal adapter of 1.45 million weights rewrites
  each frame's SigLIP 2 embedding with what the frames around it show. A new
  library in the standard mode goes through it (`whenframe index
  --no-temporal` leaves it out). Trained on DiDeMo's 7,745 and MSR-VTT's
  8,998 training videos, the average of three runs; its weights are for
  research and non-commercial use (`whenframe/data/NOTICE`).
- The sixth round compares SigLIP 2 B/32, B/16, So400m and B/16 with the
  model, on the same frames of DiDeMo, Charades-STA, YouCook2 and UnAV-100:
  mean of the four primary metrics 36.80, 38.07, 40.63 and 38.65, each step
  significant; on DiDeMo, 29.2 % of the moments among the first five with
  the model instead of 27.6 %, for 0.04 more CPU minutes per hour of video
  ([docs/RESULTS.md](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md)).
- Every encoder's indexing cost measured the same way: 4.3 (B/32), 7.2 (B/16)
  and 24.9 (So400m) CPU minutes per hour of video with OpenVINO.
- `scripts/train_data.py`, `scripts/msrvtt.py`, `scripts/temporal.py` and
  `scripts/compare.py` stream the training archives, train the model with JAX
  (`pip install 'whenframe[train]'`) and run the comparison; a head placing
  moments, SigLIP's own loss and WiSE-FT's scaling were tried and not kept.

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
