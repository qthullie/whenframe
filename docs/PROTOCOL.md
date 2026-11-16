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
| `frame_window` | a fixed window around the best frame of the video | best frame score |
| `scene` | shots found by ffmpeg's scene detector (threshold 0.3) | score of the frame in the middle of the shot |
| `whenframe` | grown around peaks of the smoothed curve until it falls back towards the video's median | peak score + contrast x (peak - median) |
| `whenframe_dedup` | as `whenframe`, using only frames that differ from the previous kept one | same |

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

## Changes after the first test run

None yet.
