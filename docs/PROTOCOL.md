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

## Changes after the first test run

None yet.
