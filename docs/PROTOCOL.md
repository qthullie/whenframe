# Protocol

Written and committed before any test result was computed. Anything changed
after the first test run is listed at the end, with the reason.

## Question

How well can a search that runs on a CPU, with no training, find a described
moment in a library of personal videos, and at what cost?

## Task

**Corpus moment retrieval.** Given a text, rank moments (video, start, end)
across every video of the split. This is what a person searching their own
library does: they do not know which video holds the moment.

**Single-video moment retrieval** is reported as well (the video is given), to
separate finding the video from placing the moment inside it.

## Data

- **Test:** DiDeMo test split, 1,004 Flickr videos with 4,021 descriptions
  (Hendricks et al., 2017). Flickr videos are personal footage, which is the
  setting this tool is for. Only descriptions whose video could be downloaded
  are kept; the number used is reported with every result.
- **Tuning:** videos from the DiDeMo training split. The test split is never
  used to choose a parameter.
- **Reference moment:** each description was located by four annotators as a
  run of 5-second segments. The run chosen most often is the reference (ties go
  to the first one listed), converted to seconds and clipped to the video.

## Metrics

- Corpus: R@k, k = 1, 5, 10, 100: the share of queries for which one of the
  first k moments is in the right video and overlaps the reference with
  IoU >= 0.5 (and >= 0.7).
- Video R@k: the right video among the first k distinct videos.
- Single video: R1 at IoU >= 0.5 and >= 0.7, and mean IoU of the first moment.
- Cost: CPU seconds per hour of video for decoding and for encoding, index size
  per hour of video, and milliseconds per query.

**Primary metric: corpus R@5 at IoU >= 0.5.** It is the one used to tune every
method and the one the success criteria refer to.

## Methods

All methods use the same frame embeddings (SigLIP 2 base, 224 px, float32,
one frame per second) and the same text embeddings. They differ only in how
per-frame scores become ranked moments.

| Method | Moments | Ranking |
| --- | --- | --- |
| `frame_window` | a fixed window around the best frame of the video; since change 8, up to three windows, each around the best frame at least a window away from the previous ones | score of that frame |
| `scene` | shots found by ffmpeg's scene detector (threshold 0.3) | score of the frame in the middle of the shot |
| `whenframe` | grown around peaks of the smoothed curve until it falls back towards the video's median | peak score + contrast x (peak - median) |
| `whenframe_dedup` | as `whenframe` (since change 9: as the default search), using only frames that differ from the previous kept one | same |
| `whenframe_bank` (change 2) | as `whenframe`, on scores corrected with a query bank | same, corrected |
| `frame_window_bank` (change 9) | as `frame_window`, on the same corrected scores | same, corrected |

Each method gets the same tuning budget: every combination when its grid has
at most 60 points, 60 random combinations otherwise (seed 0). The `scene`
method is in the spirit of tools that index one keyframe per shot; it is not a
reproduction of any particular tool.

## Hardware

A 4-vCPU cloud container without a GPU, ONNX Runtime on the CPU, 4 threads.

## Success criteria, fixed in advance

1. `whenframe` beats both baselines on the primary metric, on the test split,
   by at least 2 points.
2. `whenframe_dedup` encodes at most 70 % of the frames and loses at most
   1 point on the primary metric against `whenframe`.
3. Indexing costs at most 10 minutes of CPU time per hour of video.

## Changes made before the first test run

Added after the protocol above was first committed, and before any result on
training or test videos was computed.

1. **CPU time.** The first version of the script timed indexing with a wall
   clock. Criterion 3 speaks of CPU time, so the script now records the CPU
   time of the process (encoding) and of its ffmpeg children (decoding), and
   keeps the wall-clock time as well. Both are reported; criterion 3 is judged
   on CPU time, for the plain `whenframe` index (every frame encoded).
2. **A fifth method, `whenframe_bank`.** In cross-modal retrieval some gallery
   items match almost any text and crowd the top of every ranking ("hubs").
   Query-bank normalisation ([Bogolin et al., 2022](https://arxiv.org/abs/2112.12777))
   corrects for this with a bank of unrelated queries. Here, each frame gets
   an offset when it is indexed: the soft maximum of its scores against a bank
   of 4,096 descriptions of training videos that are not in the tuning set,
   (1/beta) log mean_b exp(beta cos(b, frame)). A query's score for that frame
   becomes cos(query, frame) - lambda x offset, and the rest is `whenframe`.
   Its two parameters are tuned on the tuning videos on top of the tuned
   `whenframe` parameters (beta in {10, 20, 50, 100}, lambda in {0.25, 0.5, 1};
   12 combinations). The same bank serves tuning and test.
   **Criterion 4:** `whenframe_bank` beats `whenframe` by at least 1 point on the
   primary metric on the test split.
3. **Cost curve.** The primary metric against the share of frames encoded,
   when repeated frames are skipped (thresholds 0 to 24, `max_gap` 4) and when
   the frame rate is lowered instead (every k-th frame, k = 1 to 5), with the
   tuned `whenframe` parameters, on tuning and test videos. Reported as a
   figure; nothing is chosen from it, and the threshold of `whenframe_dedup`
   stays at 4.
4. **Latency** is measured one query at a time, as in the tool (scoring every
   frame, shortlisting, placing and ranking moments), on 200 queries. Embedding
   the query text is timed separately, on 50 queries.
5. **Tuning set.** The first 500 videos, in name order, of the 839 extracted
   from the first part of the DiDeMo training archive. The list is saved
   (`results/didemo_train_videos.txt`) so the same set can be used again.
6. **Smoke run.** Before these changes, the pipeline was run end to end on 11
   test videos to check that it worked. It showed that the repeat filter kept
   about 86 % of frames at the default threshold, which is why the cost curve
   was added. No parameter or criterion was changed because of it.
7. **The first 30 seconds.** DiDeMo's annotators saw only the first 30 seconds
   of each video: "The first temporal chunk corresponds to seconds 0-5 in the
   video [...] Some videos are longer than 30 seconds. These videos were
   truncated to 30 seconds during annotation" (dataset README), and every
   annotation uses segments 0 to 5. The mp4 copies used here are the whole
   videos (24 to 90 s in a random sample of 40 test videos). Only the first
   30 seconds are searched: a moment found later could never be counted as
   right. Test videos are decoded up to 30 s. The tuning videos had already
   been extracted whole when this was noticed, and are cut to 30 s when
   loaded. Costs are per hour of video decoded.

8. **After the first tuning run.** On the tuning videos (the test split was
   not touched), `whenframe` reached 22.3 % on the primary metric and
   `frame_window` 24.5 % (`results/didemo_tuned_first_run.json`). Both found the
   right video equally often (48.9 % in the first five); they differed in
   placing the moment. 77 % of the reference moments are a single 5-second
   segment, so IoU >= 0.5 needs a moment of at most 10 s around it, and every
   leading `whenframe` combination sat at the largest minimum length allowed,
   5 s. With a 10-second minimum, `whenframe` reached 28.4 %, but a fixed
   10-second window allowed three windows per video reached 29.0 %: the gain
   came from proposing several moments per video, not from their adaptive
   length. Before the second tuning run, therefore:
   - `frame_window` may propose up to three windows per video, window 5 to
     11 s: all 21 combinations are tried;
   - `whenframe` may use a minimum length of 5, 7.5 or 10 s, and its grid is
     narrowed to what the first run found useful (sigma 0, 1 or 2; alpha 0.5
     to 0.8; contrast 0, 0.5 or 1; two or three moments per video): 60 of its
     216 combinations, drawn at random;
   - `scene` and the query-bank parameters are unchanged.
   Criteria 1 to 4 are unchanged; criterion 1 is judged against the stronger
   `frame_window`. The default search of the tool is the method with the best
   primary metric on the tuning videos in the second run, among
   `frame_window`, `whenframe` and `whenframe_bank`, and is written here
   before the test run.

9. **After the second tuning run.** On the tuning videos: `frame_window`
   29.0 %, `whenframe` 28.3 %, `whenframe_bank` 30.4 %
   (`results/didemo_tuned_second_run.json`), so the rule of change 8 would
   have made `whenframe_bank` the default. But the query bank helps the fixed
   windows as much: with the bank parameters found for `whenframe_bank`
   (beta 50, lambda 0.5), tried by hand, `frame_window` rose to 31.2 %. Before
   the test run, therefore:
   - a sixth method, `frame_window_bank`: the tuned `frame_window` parameters
     with the query bank, whose two parameters are tuned as for
     `whenframe_bank` (the same 12 combinations);
   - the default search is the best of `frame_window`, `whenframe`,
     `whenframe_bank` and `frame_window_bank` on the tuning videos, chosen by
     the script and saved as `"default"` in `results/didemo_tuned.json`;
   - `whenframe_dedup` runs the default search, instead of `whenframe`, on the
     frames left after skipping repeats, and the cost curve uses the default
     search: both are about what the tool does. Criterion 2 compares
     `whenframe_dedup` with the default search;
   - **criterion 5:** `frame_window_bank` beats `frame_window` by at least
     1 point on the primary metric on the test split.
   The grids and random draws of the second run are unchanged, so the third
   run repeats its results and adds `frame_window_bank`.

10. **The third tuning run** (`results/didemo_tuned.json`) repeated the second
    one exactly and chose `frame_window_bank` as the default search: up to
    three 10-second windows per video and the query bank with beta 50 and
    lambda 1, 31.5 % on the tuning videos (`whenframe_bank` 30.4 %,
    `frame_window` 29.0 %, `whenframe` 29.0 %, `scene` 3.7 %). The defaults of
    the tool's `SearchConfig` are set to it. Correction to change 9: the 28.3 %
    given there for `whenframe` was read before the second run had finished;
    the run's final figure is 29.0 % (`results/didemo_tuned_second_run.json`).
    It does not change the rule of change 8 or its outcome.

## Changes after the first test run

None to the first round: its methods, parameters and numbers stand as they
are. What follows is a second round, developed on the tuning videos only and
written here before the test split was used a second time.

## Second round

**What was tried on the tuning videos first.**

1. Other corrections of frame scores: nearest-neighbour normalisation
   ([arXiv 2410.24114](https://arxiv.org/abs/2410.24114)), the dynamic inverted
   softmax of query-bank normalisation, centring both modalities
   ([arXiv 2507.19054](https://arxiv.org/abs/2507.19054)), a bank of all 30,431
   training descriptions instead of 4,096, frames averaged with their
   neighbours, a per-video offset, and alpha query expansion. None beat the
   default search by more than 0.2 points, and none is used
   (`scripts/explore.py`, `results/didemo_train_explore.json`).
2. Cheaper image encoders. SigLIP 2 with 4-bit weights was no faster on this
   CPU (362 ms per frame). SigLIP 2 quantized to int8 with calibration broke the
   embeddings (cosine 0.66 with the float model; 12.7 % on the primary metric);
   keeping the MLP output layers in float gave cosine 0.94 at 251 ms per frame.
   Neither is used. MobileCLIP-S2 (Xenova/mobileclip_s2, 256 px) takes about
   156 ms per frame.

**Methods added.**

- `mobileclip_bank`: `frame_window` with its tuned parameters, on MobileCLIP-S2
  scores corrected with a query bank of the same 4,096 descriptions embedded by
  MobileCLIP-S2; the bank's two parameters are tuned as for the other banked
  methods (12 combinations).
- `ensemble_bank`: `frame_window` on the sum of the two corrected scores, each
  standardised over all frames for the query, the MobileCLIP-S2 term weighted by
  w in {0.25, 0.5, 0.75, 1}, both banks as tuned.
- The tool calls them the economy and precision modes; the default search
  (standard mode) is unchanged.
- MobileCLIP-S2 has no calibrated probability; the confidence the tool shows
  for it comes from a sigmoid fitted on tuning pairs (`scripts/didemo.py
  calibrate`). It does not change any ranking.

**French descriptions.** The test descriptions are machine-translated into
French with Opus-MT (Helsinki-NLP/opus-mt-en-fr, greedy decoding,
`scripts/translate.py`) and searched with the default search and the two new
methods. Translation errors stay in, so this measures French as typed by a
translation system, which is probably worse than French typed by a person.
No criterion.

**Criteria for the second test run.**

6. `mobileclip_bank` loses at most 2 points against the default search on the
   primary metric.
7. Indexing with MobileCLIP-S2 takes at most 10 minutes of CPU time per hour of
   video (criterion 3, for the economy mode).
8. `ensemble_bank` beats the default search by at least 1 point on the primary
   metric.

**Check.** The second test run evaluates the first round's methods again; their
numbers must be identical to the first run's.
