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

**Changed before the second test run: the cheaper encoder.** Apple licenses the
MobileCLIP-S2 weights for non-commercial research only (Apple Machine Learning
Research Model licence), so a tool for anyone's videos cannot ship them as a
mode. SigLIP 2 B/32 (google/siglip2-base-patch32-256, Apache 2.0; the
onnx-community export; 256 px, 64 patches) takes its place, measured in the same
way and tuned on the same grids:

- `siglip32_bank` and `ensemble32_bank` are added, the counterparts of
  `mobileclip_bank` and `ensemble_bank` with SigLIP 2 B/32. Its probability is
  its own checkpoint's sigmoid (scale exp(4.717003), bias -16.766876).
- The economy mode is `siglip32_bank` and the precision mode `ensemble32_bank`.
- Criteria 6 to 8 apply to them instead of the MobileCLIP-S2 methods, which are
  still evaluated and reported, without criterion.

## Charades-STA, without tuning

Added with the second round, before any Charades video was embedded. Charades-STA
(Gao et al., 2017) is the usual benchmark for placing a described moment in a
given video, so it allows a comparison with published numbers, which DiDeMo's
corpus setting does not.

- **Data:** the test split, 3,720 sentences in 1,334 Charades videos
  (Sigurdsson et al., 2016), from the lmms-eval/charades_sta copy on the Hugging
  Face Hub; whole videos, one frame per second.
- **No tuning:** `frame_window`, `frame_window_bank`, `whenframe` and
  `whenframe_bank` run with the parameters tuned on DiDeMo
  (`results/didemo_tuned.json`) and the DiDeMo query bank. Nothing is chosen on
  Charades.
- **Metrics:** for a given video, R1 at IoU >= 0.5 and >= 0.7 and mean IoU, as
  in the literature; and the corpus search over the 1,334 videos (R@5 at
  IoU >= 0.5).
- **No criterion.** Published numbers are quoted as their authors report them,
  with what their methods run on; they are not reproduced here.
- **Added after the run:** a reference that ignores the sentence (10 seconds in
  the middle of the video, `scripts/charades.py prior`), so the numbers can be
  read against what the window alone scores.

## Third round

Written before the test split is used a third time, and before any YouCook2
video is embedded or transcribed.

**What was tried on the tuning videos first: the runtime of the image encoder.**
Measured on 58 frames, 4 threads, this Xeon (with AMX, AVX512-BF16), CPU time
per frame:

- ONNX Runtime, float32 (the first two rounds): 342 ms.
- ONNX Runtime, the int8 file of the onnx-community export (dynamic
  quantization): 158 ms, but cosine 0.75 with the float32 embeddings. Not used.
- OpenVINO 2026.4, float32: 265 ms, cosine 1.00000.
- OpenVINO, bfloat16, which it picks by itself on processors that compute in it:
  67 ms, cosine 0.9998 on average.
- ONNX Runtime's transformer optimiser needs PyTorch, which is not installed
  here: not tried.

On the 500 tuning videos (`scripts/backends.py --split train`), OpenVINO in
bfloat16 encodes for 2.9 CPU minutes per hour of video against 21.6, and the
default search finds 31.0 % of the descriptions among its first five moments
against 31.5 %: 6 descriptions won, 16 lost (sign test p = 0.05).
*Correction, after the test run:* 2.9 was the OpenVINO run's CPU time divided by
the hours of the whole videos, while it embedded only their first 30 seconds
(the older ONNX Runtime cache holds whole videos). Per hour of what each run
embedded, OpenVINO takes 4.8 minutes against 21.6; `scripts/backends.py` now
divides each run by its own hours. The time per frame, 80 ms against 360, was
right. Criterion 9 below was set with the wrong figure in view.

`import openvino` sends a usage event to Google Analytics, from its model
converter (openvino.tools.ovc, through the openvino_telemetry package).
whenframe makes that package unimportable before loading OpenVINO, which then
falls back on its own silent stub; tracing every connection of an indexing and
a search run shows none.

**Changed in the tool.** The image encoder runs with OpenVINO when it is
installed (`pip install whenframe[fast]`), in the precision OpenVINO picks;
otherwise with ONNX Runtime, as before. The text encoder stays on ONNX Runtime
in float32.

**Criteria for the third test run** (DiDeMo test, the same 979 videos and 3,902
descriptions, embedded again with OpenVINO; the default search with its
first-round parameters and query bank):

9. Indexing with OpenVINO takes at most 10 minutes of CPU time per hour of
   video, decoding included (criterion 3 of the first round, missed then).
10. The primary metric loses at most 1 point against ONNX Runtime float32.

### Long videos: YouCook2

- **Data:** the YouCook2 validation split (Zhou et al., AAAI 2018), cooking
  videos from YouTube with each recipe step described and timed. Annotations from
  the lmms-eval/YouCook2 copy on the Hugging Face Hub (3,179 steps in 414
  videos); videos from the nguyenminh04/youcook2-videos copy, which holds 252 of
  them in full: 21.7 hours, 5.2 minutes on average (up to 11.3), 1,950 steps of
  19 seconds on average (median 14). The other 162 videos were not found.
- **Split:** a third of the 252 videos, drawn at random (Python's
  `random.Random(0).sample` of the sorted names), are for tuning: 84 videos,
  619 steps. The other 168 videos and their 1,331 steps are the test. Every
  search runs over all 252 videos.
- **Images:** one frame per second of the whole video, embedded with OpenVINO,
  repeats skipped as in the tool. `frame_window_bank` (the default search) and
  `whenframe_bank` (adaptive spans), with their DiDeMo parameters and the DiDeMo
  query bank: nothing is tuned on YouCook2.
- **Metrics:** on the test steps, as for Charades: R1 at IoU >= 0.3, 0.5 and 0.7
  and mean IoU in the given video; over the 252 videos, R@1, R@5 and R@10 at
  IoU >= 0.3 and 0.5, the primary metric being R@5 at IoU >= 0.5 as for DiDeMo;
  and R@5 counting a moment as right when its best frame falls inside the step,
  since a step longer than 20 seconds cannot be matched by a 10-second window at
  IoU >= 0.5.

### What is said: speech

- **Transcription:** Whisper base (faster-whisper 1.2.1, CTranslate2 int8, 4
  threads, its voice activity filter, greedy decoding, language detected),
  audio decoded by ffmpeg at 16 kHz. On two tuning videos it took 6.6 and 13.5
  CPU minutes per hour of video; Whisper small took 17.3 and 23.9.
- **Search:** the transcript segments go into an SQLite FTS5 index (Porter
  stemmer, unicode61 tokenizer). A query's words, stop words removed, are
  matched with OR, and each segment gets its BM25 score. A frame gets the best
  score of the segments of its video said around it; its search score becomes
  the image score plus `weight` times that speech score (`normalise`: either the
  BM25 score itself, or divided by the query's best segment score).
  *Changed before any YouCook2 video was embedded:* "around it" was first
  written as the segments starting before the frame plus `pad` seconds and
  ending after it minus `pad`. A unit test showed that a short segment falling
  between two kept frames then counts for neither, since frames repeating the
  previous one are skipped. A kept frame now stands for the time until the next
  kept frame, and a segment counts when it overlaps that time, both widened by
  `pad` seconds.
- **Tuning:** `weight` in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}, `pad` in
  {0, 5} seconds and `normalise` in {raw, best}, 24 combinations, chosen on the
  619 tuning steps with the primary metric; the image part is the default
  search, unchanged.
- **Criteria:**
  11. Speech raises the primary metric on the YouCook2 test steps by at least
      2 points over the default search.
  12. Transcription takes at most 15 minutes of CPU time per hour of video on
      the YouCook2 videos.

## Fourth round

Written part by part, each part before its test run.

### Speech with adaptive spans

On YouCook2's long steps the adaptive spans did better than the fixed windows
(third round), and speech had only been added to the fixed windows. The same 24
combinations were tried with speech added to the adaptive spans, on the 619
tuning steps (`scripts/youcook2.py tune-spans`): the same one came out best,
weight 0.005, no pad, raw scores, with an R@5 of 29.4 % against 19.5 % for the
spans alone and 24.1 % for speech with fixed windows.

13. On the 1,331 YouCook2 test steps, speech added to the adaptive spans beats
    speech added to the fixed windows (18.8 % in the third round) by at least 2
    points on the primary metric.

### SigLIP 2 So400m

SigLIP 2 So400m (Google's google/siglip2-so400m-patch14-224, through Immich's
ONNX export of the open_clip weights; about 400 million parameters in the image
tower, against 86 million for the base model) embeds the same frames as the
base model, at 224 pixels, with OpenVINO in bfloat16: one frame per request,
four requests at once (`scripts/didemo.py extract --model siglip2-so400m
--backend openvino`). The default search is unchanged but for the query bank's
beta and strength and the cleaning of the descriptions, chosen among 24
combinations on the 500 tuning videos (`scripts/so400m.py tune`): lower case
as for the base model, or open_clip's "canonicalize" (punctuation removed too);
beta 10, 20, 50 or 100; strength 0.25, 0.5 or 1. The best was lower case, beta
50, strength 1, the base model's own values: the right moment among the first
five for 35.1 % of the tuning descriptions, against 31.3 % for the base model
embedded by OpenVINO, both with repeats skipped. The 24 combinations ranged
from 33.6 to 35.1 %. On the tuning videos, encoding took 407 ms of CPU time per
frame (80 for the base model with OpenVINO), and indexing 26.6 CPU minutes per
hour of video, decoding included.

14. On the DiDeMo test, the default search with So400m, as tuned, finds the
    right moment among the first five at least 2 points more often than the
    same search with the base model embedded by OpenVINO (`scripts/so400m.py
    evaluate`, repeats skipped in both).
15. Indexing with So400m on OpenVINO takes at most 30 CPU minutes per hour of
    video, decoding included (the base model took 24.9 with ONNX Runtime in the
    first round).

### Motion: grids of four frames

SigLIP 2 sees still images. Each second of a video is also shown to it as a
2x2 grid of the four frames a quarter of a second apart (224 pixels, each
frame at 112), embedded with OpenVINO (`scripts/motion.py extract`); the grid's
score, corrected by the query bank as the frame's is, is added to the frame's
with a weight chosen among 0.25, 0.5, 0.75 and 1 on the 500 tuning videos
(`scripts/motion.py tune`). With 0.25, the right moment was among the first
five for 32.5 % of the tuning descriptions against 31.3 % for the frames
alone (52 won, 29 lost), and 33.4 % against 31.5 % for the 826 that say how
something moves (a fixed list of verbs, in the script, written before any grid
was scored); the grids alone found 29.0 %. The weight was picked among four on
the same descriptions, so the gain on new ones is likely smaller. Decoding and
embedding the grids took 6.8 CPU minutes per hour of video, about as much as
the frames.

16. On the DiDeMo test, adding the grids' score with weight 0.25 to the
    frames' raises the primary metric by at least 1 point (both embedded by
    OpenVINO, repeats skipped; `scripts/motion.py evaluate`).

### What is heard: CLAP on UnAV-100

UnAV-100 (Geng et al., CVPR 2023) marks where 100 kinds of events, each both
seen and heard, happen in untrimmed videos: people laughing, clapping, a dog
barking, a guitar... Its test videos are the ones found in full, in the copy at
huggingface.co/datasets/nguyenminh04/unav100-videos: 2,008 of the 2,167 are
there, but 1,616 of them last longer than UnAV-100 says, apparently the whole
YouTube video from which it cut its clip at an offset the annotations do not
give (on six of them, the sound and the frames matched the annotations best
at shifts of 20 to 165 seconds, not at 0). Only the 392 files that last as
long as the annotations say, within a second, are used: 3.8 hours, 937
events. A third of them, drawn once (seed 0), are for tuning; the other 262
for the test.

Each label is a description ("dog barking"), searched for among the videos of
the part at once with the default search; it counts where it happens at least
3 times. A moment is right when its best instant falls inside an event of the
label in that video (half of the events last under 7 seconds, which a
10-second moment cannot match at IoU 0.5). Primary metric: the share of the
first 10 moments that are right, averaged over the labels (P@10); also the
average precision of the first 100 moments, each event found once (mAP).

The sound track is cut into windows that overlap by half, embedded by LAION's
CLAP (larger_clap_general, Apache 2.0) with OpenVINO; a frame takes the best
score of the windows around its middle. Frames and sound are standardised over
the videos, per label, and added with a weight. Tuned on the 130 tuning videos
(39 labels, `scripts/unav.py tune`), among windows of 5 or 10 seconds and
weights 0.25 to 3: 5-second windows and weight 2 gave a P@10 of 41.0 %
against 32.8 % for the frames alone and 30.5 % for the sound alone. Decoding
the sound and embedding its 5-second windows took 4.7 CPU minutes per hour of
video.

17. On the 262 test videos, frames and sound together raise P@10 over the
    frames alone by at least 4 points.

## Fifth round: whenframe's own model

Until this round nothing was trained: whenframe searched SigLIP 2's
embeddings of single frames as the encoder gave them, and DiDeMo's training
annotations served only to build the query bank and to choose a few settings.
This round trains a model of its own on DiDeMo's training videos, on top of
SigLIP 2, and measures it with the same splits, metrics and search as before:
does a small model trained on one collection of personal videos find moments
better than SigLIP 2 alone, there and on the other benchmarks?

### The model

A temporal adapter (`whenframe/temporal.py`): two transformer layers (width
256, four heads) read a video's frames together, 32 at a time, with their
times as sines and cosines of periods 2 to 64 seconds counted from the
window's first frame, and add to each frame's SigLIP 2 embedding what its
neighbours show; the sum is normalised again and searched exactly as before,
with the same text embeddings, query bank and moments. Windows start every 16
frames and are blended where they overlap, so a video of any length goes
through it. Its output projection starts at zero, so that before training it
is SigLIP 2 itself (CLIP4Clip's temporal transformer, Luo et al. 2021; the
zero-initialised residual of CLIP-Adapter and ControlNet). 1.45 million
weights, run once per frame when a video is indexed (about half a millisecond
of CPU per frame, against some 80 for SigLIP 2's image tower with OpenVINO);
photos keep their embedding.

### Training data

DiDeMo's training videos, from the copy on the Hugging Face Hub
(friedrichor/DiDeMo, a tar archive cut in eight parts, read as a stream: no
video is kept; `scripts/train_data.py`), the 500 tuning videos left out:
their first 30 seconds at one frame per second, embedded by SigLIP 2 base with
OpenVINO and repeats marked, as the tuning and test videos were; their
descriptions embedded by SigLIP 2's text tower with ONNX Runtime, as the tool
embeds a query (31,048 descriptions of 7,901 videos; those whose video is
not in the archive or cannot be read are left out).

### Training (`scripts/temporal.py train`)

Losses, each after a published idea:

- between videos: each description's own video among the batch's, scored as
  the search scores it (the best frame's cosine minus the query bank's
  offset, the offset computed from bank descriptions drawn for each batch),
  with InfoNCE (CLIP4Clip; the bank: Bogolin et al. 2022);
- within a video: the share of the frames' softmax that falls inside the
  described moment (the saliency losses of Moment-DETR and QD-DETR, Lei et
  al. 2021, Moon et al. 2023; the rest of the video as hard negatives, as in
  ReLoCLNet, Zhang et al. 2021);
- a leash, 1 - cos(new, old), towards SigLIP 2's embedding of each frame.

Each batch takes one description per video; its moment is half the time one
of its four annotators', half the time the most common one. Frames are
dropped as the tool drops repeats (half the time), 30 % more of them
sometimes, and half the time the window starts and ends a few frames inside
the video, the described moment kept whole. Adam with weight decay, warm-up
then cosine decay, gradients clipped at 1. The model is checked on the 500
tuning videos as it trains, with the default search on its frames; the best
check is kept.

### Choices made on the tuning videos

Five short runs (600 steps of 64 descriptions, on the 1,454 training videos
embedded at the time), each checked on the tuning videos every 200 steps
(`results/temporal/e*.json`). Without the model: R@5 31.3 %, the right video
among the first five 51.4 %, single-video R1 at IoU 0.5 46.4 %.

| run | between | within | leash | best R@5 | right video in top 5 | single R1 |
|---|---|---|---|---|---|---|
| e1, with a head (below) | 1 | 1 | 10 | 32.9 | 51.7 | 50.1 |
| e3 (stopped at 400 steps) | 0 | 1 | 10 | 30.8 | 49.5 | 49.1 |
| e4 | 1 | 1 | 100 | 33.0 | 52.8 | 48.3 |
| e5 | 1 | 2 | 100 | 33.1 | 52.7 | 49.1 |

Without the loss between videos, the right video falls back as the moments
get better placed (e3). A leash ten times stronger keeps the frames at a
cosine of 0.997 from SigLIP 2's (0.96 in e1) for the same R@5. e5's settings
are kept: the same R@5 as e4, the moments better placed in their video.

**A head, tried and not kept.** A second small network was trained with the
adapter, run at search time on each shortlisted video for the description:
one attention layer over the frames with the description mixed into each
(as FiLM), giving each frame a corrected logit and how far the moment reaches
before and after it (an anchor-free placement, as FCOS, and UniVTG for
moments), trained with binary cross-entropy, a ranking against four other
videos and Moment-DETR's span loss (L1 and generalised IoU), without
reshaping the adapter. Its moments did much worse than the default search's
windows (e1: R@5 24.2 %), for a reason the tuning videos show: three
quarters of DiDeMo's reference moments last 5 seconds, and a 10-second window
around the right frame overlaps them at IoU 0.5 exactly, so the primary
metric favours the default windows over tighter moments (around the best
frame of SigLIP 2's scores, a fixed 5-second moment gives a single-video R1
of 31.6 %, 7 seconds 39.0 %, 10 seconds 46.4 %). Used only to score the
frames inside the default windows, it did no better either (e1: 32.3 %
against 32.9 % for the adapter alone; trained alone on SigLIP 2's frames,
e2: 31.3 %). Its code stays in `scripts/temporal.py`.

### The model trained for the test

e5's settings on every training video embedded (`scripts/train_data.py`, run
to the end of the archive): 3,000 steps of 128 descriptions, learning rate
1e-4 after 200 steps of warm-up, 1,024 bank descriptions per batch, checked on
the tuning videos every 250 steps, the best check kept. Before the test, the
evaluation was run with an untrained adapter (the identity), which gave back
every published number of the benchmarks below (DiDeMo 27.6 %, Charades-STA
45.7 %, YouCook2 12.3 and 25.8 %, UnAV-100 42.9 and 53.2 %), one DiDeMo
description in 3,902 aside (a tie broken otherwise by rounding).

The test (`scripts/temporal.py evaluate`, run once): the default search on
the adapted frames against the same search on SigLIP 2's, on the DiDeMo test
videos (OpenVINO, repeats skipped) and, each as in its own round, on
Charades-STA, YouCook2 (test steps) and UnAV-100 (test videos), with a sign
test over the descriptions (labels for UnAV-100).

18. On the DiDeMo test, the adapted frames raise the primary metric by at
    least 1 point.
19. On each of Charades-STA (R1 at IoU >= 0.5 in the given video), YouCook2
    (R@5 at IoU >= 0.5, speech added to the adaptive spans) and UnAV-100
    (P@10, frames and sound), the adapted frames lose at most 1 point.
20. The adapter adds at most 1 CPU minute per hour of video to indexing.

If 18 and 19 hold, a new library goes through the model by default
(`--no-temporal` keeps SigLIP 2's embeddings); libraries already indexed keep
theirs.

**The model trained, before the test.** 7,745 training videos were in the
archive and could be read (62.3 hours, 30,434 descriptions). The best check
on the tuning videos came at step 1,750 of 3,000: R@5 34.1 % (31.3 % without
the model), the right video among the first five 53.9 % (51.4 %), single-video
R1 at IoU 0.5 49.9 % (46.4 %), the frames at a cosine of 0.998 from SigLIP 2's
(`results/temporal/temporal-siglip2-base.json`). The checks were chosen among
twelve on the same videos, so the gain on new ones is likely smaller.
Training took 21 minutes on the four cores, checks included. Exported in
float16 to `whenframe/data/temporal-siglip2-base.npz`, the file the test runs.

## Sixth round: the models compared the same way, and whenframe's model again

Written before any of its numbers was computed. Two questions. Measured the
same way on the four benchmarks, which of whenframe's ways of embedding frames
finds the most moments? And does whenframe's model, trained again on more
varied descriptions, keep its gain on DiDeMo without its loss on UnAV-100?

### Candidates

Five, searched the same way:

1. SigLIP 2 B/32 (`--mode economy`);
2. SigLIP 2 B/16 (the standard mode), the reference;
3. SigLIP 2 So400m (`--mode large`);
4. B/16 and whenframe's model of the fifth round (`--temporal`, v1);
5. B/16 and whenframe's model v2 (below).

**The same frames.** All five see the frames at the same instants: those of
the standard mode's caches (DiDeMo's tuning and test videos with repeats
skipped, every frame of Charades-STA and UnAV-100, YouCook2 with repeats
skipped). The other encoders' frames are decoded again from the same videos,
read again from the Hugging Face Hub one at a time and not kept; only the
frames at the cached instants are embedded, and the instants are checked
equal. Every encoder runs with OpenVINO, as the tool runs it (B/32 is embedded
again on DiDeMo for this).

**The same tuning.** Each encoder keeps the query bank's beta and strength
chosen for it on DiDeMo's tuning videos (second and fourth rounds; the
standard mode's for 4 and 5) and the standard mode's moment parameters. The
weight of speech on YouCook2 and the weight of sound on UnAV-100 are chosen
again for every candidate, on the grids, tuning steps and tuning videos of
the third and fourth rounds.

**Metrics.** Each benchmark's primary metric, as in its own round: DiDeMo
test, R@5 at IoU >= 0.5 (default search); Charades-STA, R1 at IoU >= 0.5 in
the given video (DiDeMo's parameters); YouCook2, R@5 at IoU >= 0.5 over the
1,331 test steps, speech added to the adaptive spans; UnAV-100, P@10 over the
262 test videos, frames and sound. The cost of indexing is reported beside
them.

**The rule.** The candidates are ranked by the mean of the four primary
metrics. One is said to find more than another only when its mean is higher
and a sign test over the paired items of the four tests gives p < 0.05: each
description found or not among the first five (DiDeMo, YouCook2), placed or
not in its given video (Charades-STA), and each kind of event whose P@10 rose
or fell (UnAV-100).

### whenframe's model v2

Trained on DiDeMo's training videos, as v1, and on MSR-VTT's 9,000 training
clips (Xu et al., CVPR 2016: YouTube clips of 10 to 30 seconds in 20
categories, among them animals, music, vehicles, sports and cooking), read
from the archive of friedrichor/MSR-VTT on the Hub without keeping any video:
their frames embedded as DiDeMo's were, five of each clip's twenty
descriptions (drawn once, seed 0), the whole clip as the described moment (so
the loss within a video learns nothing from them). Clips cut from a YouTube
video that is also a YouCook2 or UnAV-100 video are left out: two YouCook2
videos, none of UnAV-100's (by YouTube identifier). VGGSound, from which
UnAV-100 was built, is not used.

Three changes to the fifth round's training, each after a published result,
each kept only if it does better on the tuning sets:

- the loss between videos as SigLIP's own pretraining loss, a sigmoid over
  every pair of the batch on the raw best-frame logits with SigLIP 2's scale
  and bias, instead of InfoNCE on bank-corrected scores (fine-tuning with the
  pretraining loss: FLYP, Goyal et al., CVPR 2023);
- a soup: the average of the weights of three runs from the same
  initialisation with different orders of the data (Wortsman et al., ICML
  2022);
- the adapter's correction multiplied by alpha in {0.25, 0.5, 0.75, 1} (the
  zero-shot and the fine-tuned models ensembled, as WiSE-FT, Wortsman et al.,
  CVPR 2022; for a residual adapter whose correction starts at zero, the
  ensemble of their outputs).

Each choice is made by the rule's mean over the tuning sets: DiDeMo's 500
tuning videos (R@5), YouCook2's tuning steps (R@5, speech and spans) and
UnAV-100's 130 tuning videos (P@10, frames and sound); Charades-STA has none.
The settings otherwise are the fifth round's.

### Criteria

21. v2's mean of the four primary metrics is higher than SigLIP 2 B/16's, with
    a sign test p < 0.05.
22. v2 loses at most 1 point on each of the four tests against SigLIP 2 B/16.
23. v2's mean is higher than v1's.

If 21 and 22 hold, v2 replaces v1 as `--temporal` and a new library in the
standard mode goes through it by default; otherwise v2 replaces v1 only if 23
holds, and stays an option. The test is run once, after the choices on the
tuning sets are recorded here.

**Licence of the weights.** whenframe's model is trained on videos published
for research (DiDeMo: Flickr videos under Creative Commons licences, some of
them non-commercial; MSR-VTT: YouTube clips); its weights are provided for
research and non-commercial use. whenframe's code stays under the MIT licence.

**The choices on the tuning sets, before the test** (`results/compare_tuned.json`;
the mean of DiDeMo's R@5, YouCook2's R@5 with speech and spans, UnAV-100's
P@10 with sound, each candidate's speech and sound weights chosen on the
same sets). The standard mode: 31.3, 29.4 and 41.0 %, mean 33.92; v1: 34.1,
28.9 and 41.0 %, mean 34.68.

| v2, trained on DiDeMo and MSR-VTT (8,998 clips, 44,990 descriptions) | DiDeMo | YouCook2 | UnAV-100 | mean |
|---|---|---|---|---|
| InfoNCE between videos, one run | 33.4 | 29.4 | 41.0 | 34.60 |
| SigLIP's own sigmoid loss (FLYP): not kept | 33.1 | 28.4 | 41.8 | 34.43 |
| **soup of three InfoNCE runs (data seeds 0, 1, 2): kept** | 33.2 | 29.9 | 41.3 | **34.79** |
| the soup, alpha 0.25 | 32.1 | 29.2 | 41.0 | 34.14 |
| the soup, alpha 0.5 | 32.6 | 29.4 | 41.3 | 34.43 |
| the soup, alpha 0.75 | 33.1 | 29.9 | 41.3 | 34.74 |

v2 is the soup at alpha 1 (no scaling did better), exported in float16 as
v1 was (`results/temporal/temporal-siglip2-base-v2.npz`), the file the test
runs. Of the three published ideas, only the soup is kept.

## Seventh round: whenframe's model v3, taught by So400m, trained on more moments

Written before any of its numbers was computed. The sixth round ranked
SigLIP 2 So400m (`--mode large`) first, 2.0 points of the mean above model v2,
for 3.4 times v2's cost of indexing. The question: can whenframe's model, at
the standard mode's cost, find more moments than So400m?

### What changes

**More moments, with their start and end.** v2 learnt where moments are from
DiDeMo's 5-second segments only (MSR-VTT's clips are described whole). v3 adds
three sets of training descriptions, each placing a moment in time:

- Charades-STA's 12,408 training sentences (Gao et al., ICCV 2017, as copied
  with 2D-TAN, microsoft/VideoX), on 5,338 of Charades' training videos
  (Sigurdsson et al., ECCV 2016), read from the full Charades archive on the
  Hugging Face Hub (lmms-eval/charades_sta) by byte ranges, one at a time,
  none kept. 500 of these videos, drawn once (seed 0,
  `results/charades_tuning_videos.txt`), are held out as Charades-STA's tuning
  set, which it did not have; none of its 1,334 test videos is a training
  video;
- YouCook2's training steps (Zhou et al., AAAI 2018): 8,815 steps of 1,139
  training videos, 1,106 of them in the copy nguyenminh04/youcook2-videos;
  none is one of the benchmark's 404 validation videos (tuning and test), and
  none a UnAV-100 video (by YouTube identifier);
- TEMPO-HL's sentences on DiDeMo's training videos (Hendricks et al., EMNLP
  2018, BSD 2-Clause) that are not DiDeMo's own: moments told by what happens
  before, after or while another, on the same 5-second segments; the 500
  tuning videos left out.

DiDeMo's training videos and MSR-VTT's clips stay, as in v2. The copy of
UnAV-100 read for the fourth round holds only its test videos (VGGSound's are
on YouTube): nothing of UnAV-100 is trained on, and it stays the test of what
the model has never seen. The frames are those the tool indexes: one a
second, SigLIP 2 base with OpenVINO; Charades' every frame with repeats
marked, as DiDeMo's training videos; YouCook2's with repeats skipped, as the
benchmark's (`scripts/moments_data.py`).

**A teacher.** Knowledge distillation (Hinton et al., 2015), as TeachCLIP
taught a light text-video retrieval model with stronger ones (Tian et al.,
CVPR 2024): So400m embeds every second frame of Charades-STA's training
videos (not the 500 held out), at the base model's instants, and their
descriptions (lower case, as chosen for it in the fourth round). Two losses
are added, each the Kullback-Leibler divergence from the teacher's
distribution to the model's, both scored as the search scores
(bank-corrected cosine times the encoder's own logit scale; So400m's bank
strength and beta for the teacher), over the frames the teacher saw:

- between videos: each description's distribution over the batch's videos,
  by their smooth best frames;
- within a video: its distribution over the frames of its own video.

If the teacher's losses do better on the tuning sets with Charades-STA's
frames alone, So400m also embeds every second frame of DiDeMo's training
videos and of MSR-VTT's clips (their archives read again) and their
descriptions, and the model trained with them is kept if it does better
again.

**Mixing.** Each item of a batch comes from one collection (DiDeMo with
TEMPO-HL, MSR-VTT, Charades-STA, YouCook2) drawn with probability
proportional to the square root of its number of descriptions (sampling with
a temperature, as multilingual models mix languages: XLM-R, Conneau et al.,
2020), then one of its descriptions, one per video in a batch. A moment given
in seconds holds the frames whose middle is inside it; a long video
(YouCook2) is cut to 32 frames placed at random around the moment. The
fifth round's augmentations are kept.

**Checked as it trains.** On DiDeMo's 500 tuning videos (R@5, default
search) and on Charades-STA's 500 held-out videos (R1 at IoU >= 0.5 in the
given video, every frame, DiDeMo's parameters, as its test); the check with
the best mean of the two is kept; 6,000 steps, a check every 500.

### Choices on the tuning sets

Made in this order, each kept only if it raises the rule's mean over the four
tuning sets: DiDeMo's 500 tuning videos (R@5), Charades-STA's 500 held-out
videos (R1 at IoU >= 0.5), YouCook2's tuning steps (R@5, speech and spans)
and UnAV-100's tuning videos (P@10, frames and sound), each candidate's
speech and sound weights chosen on them as in the sixth round:

1. the teacher's losses (weight 1), and then the teacher's DiDeMo and
   MSR-VTT frames, as above;
2. the leash's weight, 100 (v1, v2) or 30;
3. the adapter's size, width 256 and two layers (v1, v2) or width 512 and four;
4. a soup of three runs from the same initialisation (data seeds 0, 1, 2);
5. alpha in {0.25, 0.5, 0.75, 1} (WiSE-FT, as in the sixth round).

Reported beside, deciding nothing: v3's recipe trained without
Charades-STA's videos, and without YouCook2's, each measured on the test it
left out. The difference with v3 is the part of v3's gain there that comes
from having trained on that collection; the rest is what a collection the
model has never seen, such as a family's videos, can expect.

### Candidates and criteria

The sixth round's five, searched as in the sixth round (their numbers
recomputed by the same code, and checked equal to the sixth round's), and
v3; the same four primary metrics and the same rule. Charades-STA and
YouCook2 are no longer zero-shot for v3: as DiDeMo since the fifth round, it
has seen their training videos, never their test videos. UnAV-100 stays
zero-shot for every candidate. v3's cost of indexing is measured as v1's was
in the fifth round.

24. v3's mean of the four primary metrics is higher than So400m's, with a
    sign test p < 0.05.
25. v3's mean is higher than v2's, with p < 0.05.
26. v3 loses at most 1 point on each of the four tests against SigLIP 2
    B/16, UnAV-100 among them.

If 25 and 26 hold, v3 replaces v2 as the model a new standard library goes
through by default; if 24 holds as well, the README says that at the
standard mode's cost whenframe finds more moments than its large mode. The
test is run once, after the choices on the tuning sets are recorded here.

**The tool searches as the candidates are measured** (decided during the
choices, before the test). Each candidate is measured with the speech settings
and the sound weight chosen for it on the tuning sets; the tool searched every
mode with the standard one's (what is said within the moment, sound weight
2). It now uses those chosen with the library's model: B/32's (what is said up
to 5 seconds around a frame, sound weight 1.5), So400m's (sound weight 1.5),
and those of whenframe's model, kept in its file (v2's: sound weight 1.5; v3's
if it replaces v2). On the tuning sets, the standard mode's settings would
cost v3-noteacher 1.6 points on YouCook2 and 0.8 on UnAV-100, 0.6 of the mean,
and v2 0.8 on UnAV-100. The test itself is unchanged.

**Licence of the weights.** Charades is licensed for non-commercial research
(Allen Institute for AI), YouCook2 was published for research, TEMPO's
sentences are under the BSD 2-Clause licence, and So400m's weights, the
teacher's, under the Apache License 2.0. v3's weights, as v2's, are provided
for research and non-commercial use; whenframe's code stays under the MIT
licence.

**The choices on the tuning sets, before the test** (`scripts/round7.py`;
`results/round7_tuned.json`, `results/round7_choices.json`; the mean of
DiDeMo's R@5, Charades-STA's held-out R1 at IoU >= 0.5, YouCook2's R@5 with
speech and spans and UnAV-100's P@10 with sound, each candidate's speech and
sound weights chosen on the same sets). The standard mode: 31.3, 44.9, 29.4
and 41.0 %, mean 36.66; v2: 33.2, 45.4, 29.9 and 41.3 %, mean 37.45.

| v3, trained on DiDeMo with TEMPO-HL, MSR-VTT, Charades-STA and YouCook2 | DiDeMo | Charades-STA | YouCook2 | UnAV-100 | mean |
|---|---|---|---|---|---|
| 1. no teacher | 32.7 | 49.4 | 31.8 | 41.8 | 38.92 |
| 1. So400m's losses on Charades-STA's frames: not kept | 32.4 | 48.6 | 31.8 | 41.5 | 38.61 |
| **2. the leash's weight 30, not 100: kept** | 33.7 | 50.6 | 32.1 | 42.1 | **39.63** |
| 3. width 512 and four layers: not kept | 34.4 | 49.7 | 31.7 | 42.1 | 39.47 |
| 4. a soup of three runs (data seeds 0, 1, 2): not kept | 33.5 | 50.3 | 32.0 | 42.1 | 39.46 |
| 5. alpha 0.25 | 32.5 | 47.1 | 29.6 | 40.8 | 37.50 |
| 5. alpha 0.5 | 32.9 | 48.9 | 30.4 | 41.3 | 38.35 |
| 5. alpha 0.75 | 33.6 | 50.1 | 31.3 | 42.3 | 39.33 |

v3 is the run with the leash's weight 30 (data seed 0) at alpha 1, exported
in float16 as v2 was (`results/temporal/temporal-siglip2-base-v3.npz`, the
file the test runs; on the tuning sets it measures as the run, 39.63). Its
settings, chosen on the tuning sets: what is said weighs 0.005 and counts up
to 5 seconds around a frame, the sound weighs 2. Its training drew on 46,338
descriptions of DiDeMo's training videos with TEMPO-HL's, 44,990 of
MSR-VTT's clips, 11,256 of Charades-STA's training videos (the 500 held out
left out) and 8,563 of YouCook2's training steps, a batch's items from each
with probability 0.34, 0.34, 0.17 and 0.15; its best check came at step
5,000 of 6,000 (`results/temporal/v3-leash30.json`), and it trained in 37
minutes on the four cores, checks included. The teacher's losses did not
help on Charades-STA's frames, so So400m's frames of DiDeMo and MSR-VTT were
not computed. Neither did the larger adapter, the soup or a smaller alpha:
of the round's five choices, only the leash's lighter weight is kept. Every
choice was judged on the same tuning sets, so v3's gain on the tests is
likely smaller than on them. Beside, deciding nothing: v3's recipe without
Charades-STA's videos (checked as it trains on DiDeMo's tuning videos alone)
and without YouCook2's, one run each, as v3. On the tuning sets, without
Charades-STA's videos: 34.0, 46.4, 32.6 and 42.1 %, mean 38.79; without
YouCook2's: 34.0, 51.6, 29.1 and 41.5 %, mean 39.04.

## Eighth round: whenframe learns a library

Written before any of its numbers was computed. The seventh round's
left-out runs show where whenframe's model gains: trained without a
collection's videos, v3's recipe adds 0.8 points to SigLIP 2 B/16 alone on
Charades-STA and 0.3 on YouCook2; trained with them, 7.6 and 6.2. A user's
library is a collection the model has never seen, and nobody has described
it. The question: can the model learn a library from automatic descriptions
of its own frames, as text-video retrieval models learn a collection from an
image captioner's (Ventura et al., IJCV 2024: 50 to 80 % of the gain of
human descriptions on MSR-VTT, MSVD and ActivityNet)?

### What changes

**Automatic descriptions.** Florence-2-base (Xiao et al., CVPR 2024;
Microsoft, MIT licence; the ONNX export of onnx-community/Florence-2-base-ft,
its image encoder run by OpenVINO and the rest by ONNX Runtime) describes one
frame in each 10 seconds of every video of a library: the video is cut into
blocks of 10 seconds from its start (a last block shorter than 5 seconds is
joined to the one before), and the frame the index holds nearest to a
block's middle is decoded again and described (task `<CAPTION>`, greedy
decoding without a repeated 3-gram, at most 60 tokens). Nothing else about
the library is used: no description, label or annotation of it.

**Learning.** The model goes on training from its own weights on the
library's indexed frames, each automatic description as a query whose moment
is its block, embedded by SigLIP 2's text tower as the tool embeds a query.
The seventh round's settings are kept (the losses between and within videos,
the leash's weight 30, learning rate 1e-4, its warm-up a tenth of the steps
up to 200, the cosine decay, the augmentations), with two differences: a
batch holds at most one description of each video, so a library of fewer
than 128 videos makes smaller batches; and there is no check: a library has
nothing to check against, so the number of steps is fixed, a number of
passes over its descriptions chosen on the tuning sets, and the last weights
are kept.

**Libraries.** Each benchmark's test videos are learnt as a library from
their frames alone, then searched with the benchmark's descriptions, as the
tool searches a learnt library:

- Charades-STA's 1,334 test videos, from v3-no-charades (the seventh round's
  run that never saw Charades' videos);
- YouCook2's 252 test videos, from v3-no-youcook2;
- UnAV-100's 262 test videos, from v3 (no candidate trained on UnAV-100);
- DiDeMo's 979 test videos, from v3 (which learnt from DiDeMo's training
  videos, not these).

The tuning sets are learnt the same way, each from the same starting model,
for the choices: DiDeMo's 500 tuning videos, Charades-STA's 500 held-out
training videos, YouCook2's 84 tuning videos and UnAV-100's tuning videos.
The videos are read again as in their rounds (Charades' and YouCook2's from
the Hugging Face Hub by byte ranges, UnAV-100's archives as they download),
one at a time, and only the descriptions are kept.

### Choices on the tuning sets

Made in this order, each kept only if it raises the mean of the four tuning
metrics (DiDeMo R@5, Charades-STA R1 at IoU >= 0.5, YouCook2 R@5 with speech
and spans, UnAV-100 P@10 with sound; each candidate's speech and sound
weights chosen on the same sets, as in the seventh round), the reference
being the starting models as they are:

1. learning each library, 10 passes over its descriptions;
2. 3 or 30 passes instead (the better of the two, if better);
3. half of each batch from the starting model's own training collections,
   drawn as in the seventh round, the other half from the library.

### Test and criteria

Once, each test library learnt with the recipe chosen, against its starting
model, on its benchmark: the four primary metrics, the sign test over their
items as in the sixth and seventh rounds.

27. On Charades-STA and YouCook2, learning the library raises the mean of
    their two primary metrics by at least 2 points, with a sign test p < 0.05
    over their items.
28. Learning the library loses at most 1 point on each of the four tests.
29. Describing the frames and learning cost at most 30 CPU minutes per hour of
    video together (the process CPU time on DiDeMo's test library, decoding
    included, as the indexing costs are measured).

If 27, 28 and 29 hold, `whenframe learn` is added: it describes a library's
frames, teaches v3 the library, and the library is searched through the
model it learnt; a library not learnt keeps v3. The test is run once, after
the choices on the tuning sets are recorded here.

**The choices on the tuning sets** (`scripts/round8.py`;
`results/round8_tuned.json`, `results/round8_choices.json`). Each is scored
by the mean of four metrics:

- DiDeMo's R@5;
- Charades-STA's held-out R1 at IoU >= 0.5;
- YouCook2's R@5 with speech and spans;
- UnAV-100's P@10 with sound.

Each candidate's speech and sound weights were chosen on the same sets. The
starting models as they are: 33.7, 46.4, 29.1 and 42.1 %, mean 37.82.

| each library learnt from Florence-2's descriptions of its frames | DiDeMo | Charades-STA | YouCook2 | UnAV-100 | mean |
|---|---|---|---|---|---|
| 1. 10 passes over its descriptions: not kept | 32.2 | 45.5 | 29.7 | 42.1 | 37.38 |
| 2. 3 passes: not kept | 33.0 | 46.1 | 29.6 | 41.3 | 37.49 |
| 2. 30 passes: not kept | 31.5 | 44.0 | 28.9 | 41.3 | 36.42 |
| 3. half of each batch from the starting model's own collections, 10 passes: not kept | 32.7 | 45.4 | 28.9 | 41.5 | 37.13 |

No choice is kept.

A first version of `scripts/round8.py` stopped once choice 1 was not kept,
although the choices are made in a set order. The script was corrected
before choice 2 was measured, and choices 2 and 3 were then made as written:

- each was judged against the starting models, the run kept so far;
- choice 3 used choice 1's 10 passes, as neither 3 nor 30 was kept.

**No test.** The test compares each test library learnt with the recipe
chosen against its starting model. With no recipe chosen, it would compare
the starting models with themselves, so the round ends on the tuning sets:
criteria 27 to 29 are not judged, and `whenframe learn` is not added.

The test libraries were described ahead of the choices, to save time, from
their frames alone. DiDeMo's and UnAV-100's were described entirely, and
Charades-STA's (1,188 of 1,334 videos) and YouCook2's (244 of 252) until
the choices ended. None was learnt or searched.

On DiDeMo's test library, measured as criterion 29 measures it, decoding,
describing and embedding the descriptions cost 22.9 CPU minutes per hour of
video before learning: decoding 1.9, describing 20.3 and embedding 0.67
(`results/round8_cost.json`).

## Ninth round: Perception Encoder in place of SigLIP 2 B/16

Written before any of its numbers was computed.

**The question.** Most of the gain of whenframe's model comes from the
collections it trained on. On a library it has never seen, it adds 0.3 to
0.8 points to SigLIP 2 B/16 alone, and the eighth round could not teach it
one. What is left to gain there is in the encoder.

Meta's Perception Encoder was trained as CLIP is, then on 22 million videos
described by its authors' own captioning model (PE core, Bolya et al., 2025;
Apache License 2.0). Its B/16 has an image tower the size of SigLIP 2
B/16's: ViT-B/16 at 224 pixels, 0.09 billion weights. It would index at
about the same cost.

Its authors report, for B/16 zero-shot, MSR-VTT text-to-video R@1 47.6 and
Kinetics-400 65.6. They report no comparison with SigLIP 2 B/16 on video at
that size; on ImageNet the two are level (78.4 and 78.2 %).

So: at the same cost, does PE core B/16 find more moments than SigLIP 2
B/16? And if so, does whenframe's model, trained on its frames, find more
than v3?

### The encoder

- **Files.** PE-Core-B16-224 as exported to ONNX by onnx-community
  (`onnx-community/PE-Core-B16-224-ONNX`): the image tower in float32, as
  SigLIP 2's runs; the text tower (0.31 billion weights, 32 tokens) in
  float16 weights. The image tower runs with OpenVINO, as SigLIP 2's does,
  and so does the text tower, as So400m's does.
- **Tokens.** CLIP's tokenizer, as published with `timm/PE-Core-B-16`,
  padded with 0 to 32 tokens.
- **Frames.** Squashed to 224 pixels with bilinear filtering, as ffmpeg
  already resizes them for SigLIP 2, and scaled to [-1, 1], PE's own
  preprocessing.
- **Checked before any benchmark number**, against open_clip's
  PE-Core-B-16 in PyTorch (the weights of `timm/PE-Core-B-16`), on 16 frames
  of DiDeMo's tuning videos and 16 of their descriptions:
  - the token ids must be equal;
  - each embedding must be at a cosine of at least 0.999 to the reference's.

  If the check fails, the export is not used: PE is exported from those
  weights instead, and checked the same way.

### Stage 1, on the tuning sets: the encoders alone

**Candidates.** Two, neither with whenframe's model:

- SigLIP 2 B/16, the reference, its numbers recomputed by the same code and
  checked equal to the seventh round's;
- PE core B/16.

**The same frames.** Those at the instants of the standard mode's caches,
read again as in the sixth round, decoded at 224 pixels, the instants
checked equal, and embedded with OpenVINO:

- DiDeMo's 500 tuning videos (14,439 frames, repeats skipped);
- Charades-STA's 500 held-out training videos (15,657 frames);
- YouCook2's 252 validation videos, among which its tuning steps are
  searched (65,142 frames, repeats skipped);
- UnAV-100's 392 videos (13,779 frames): its 130 tuning videos, and the 262
  others for the test, all in the archives read.

**PE's own settings**, chosen on the tuning sets as each encoder's were:

- the query bank's beta and strength on DiDeMo's tuning videos, over the
  fourth round's grid (beta 10, 20, 50 or 100; strength 0.25, 0.5 or 1);
- the bank's strength for adaptive spans on YouCook2's tuning steps, over
  the same strengths;
- the weight of speech on YouCook2's tuning steps and the weight of sound
  on UnAV-100's tuning videos, over the sixth round's grids.

SigLIP 2 B/16 keeps its own. The search is otherwise the standard mode's.

**Metrics.** The mean of the four tuning metrics, as in the seventh and
eighth rounds:

- DiDeMo's R@5;
- Charades-STA's held-out R1 at IoU >= 0.5;
- YouCook2's R@5 with speech and spans;
- UnAV-100's P@10 with sound.

**The rule.** Stage 2 runs only if PE core B/16's mean is at least 1 point
higher than SigLIP 2 B/16's. Otherwise the round ends there: nothing is
trained and no test is run.

### Stage 2: whenframe's model on PE's frames (v4)

**Frames.** PE embeds the frames of the collections v3 trained on, at the
same instants: DiDeMo's training videos (with TEMPO-HL's sentences),
MSR-VTT's 9,000 clips, Charades-STA's training videos (the 500 held out left
out) and YouCook2's training videos. It also embeds those of the test sets
not yet embedded: DiDeMo's 979 test videos and Charades-STA's 1,334.

**Training.** v4 is v3's recipe, unchanged and with no new choice, on PE's
1,024 dimensions instead of SigLIP 2's 768:

- no teacher;
- the leash's weight 30;
- width 256 and two layers;
- data seed 0;
- the same steps and checks, the best check kept;
- alpha 1.

Its speech and sound weights are chosen on the tuning sets, as every
candidate's. Its two left-out runs (without Charades-STA's videos, without
YouCook2's) are trained as v3's were.

### Test and criteria

Once, on the four tests, with the four primary metrics and the sign test of
the sixth and seventh rounds. Four candidates:

- SigLIP 2 B/16 alone;
- PE core B/16 alone;
- v3;
- v4.

Reported beside, deciding nothing: each left-out run on the test it left
out, against v3's. This is the case of a library never seen.

30. v4's mean of the four primary metrics is at least 1 point higher than
    v3's, with a sign test p < 0.05.
31. v4 loses at most 1 point on each of the four tests against v3.
32. Indexing with PE core B/16 costs at most 1.2 times what SigLIP 2 B/16
    costs: CPU minutes per hour of DiDeMo's test videos, decoding included,
    with OpenVINO, measured as the other encoders' costs are.

If 30, 31 and 32 hold, a new library in the standard mode is indexed with PE
core B/16 and goes through v4. Libraries already indexed keep SigLIP 2 B/16
and v3. The test is run once, after stage 1's numbers and stage 2's training
are recorded here.

**The check, before any benchmark number** (`scripts/pe_core.py`,
`results/round9_check.json`).

- onnx-community's image tower was traced with a batch of exactly 2 frames
  and cannot be reshaped. Run in pairs, its embeddings came out at a cosine
  of 0.971 at worst to the reference's. It is not used: the image tower was
  exported again from `timm/PE-Core-B-16`'s weights (open_clip 4.2.2,
  PyTorch 2.15), and takes any batch size.
- The token ids are equal. The text tower's embeddings are at a cosine of
  0.99992 at worst.
- The image tower exported again is at a cosine of 0.9999999 in float32.

On this processor, OpenVINO computes every tower in bfloat16, SigLIP 2
B/16's too. PE's frames then come out at a cosine of 0.9987 at worst (0.9995
on average), and SigLIP 2's at 0.9995 at worst against its own float32.
Decided before any benchmark number: the measure runs both in bfloat16, as
the tool would on this machine.

**One frame at a time.** With any batch size, OpenVINO's memory for PE's
image tower grew with each new batch size (from 1.5 to 3.3 GB over ten
sizes), and an extraction was killed for lack of memory. It also took 1.8
times SigLIP 2 B/16's CPU time per frame. Fixed to one frame at a time, PE
runs in OpenVINO's throughput mode, as So400m's tower does: its memory
holds, and a first estimate puts it at 1.3 times SigLIP 2 B/16's CPU time
per frame (criterion 32 is measured at the test). The check was made again
on it: 0.9999999 in float32, 0.9987 at worst in bfloat16. Decided before any
benchmark number.

**The rotary position embedding, written for the export.** Stage 1 was first
measured with that tower (PE's mean 38.79, SigLIP 2 B/16's 36.66). Then,
before stage 2 and before any test number, the cost was measured on its own:
PE took 1.6 times SigLIP 2 B/16's CPU time per frame. OpenVINO's profile put
the difference in the rotary position embedding, as the export wrote it:
strided slices and concatenations of q and k in every layer (about 11 ms of
42 per frame).

The same arithmetic, written with fewer memory operations, is exported
instead (`scripts/pe_core.py`, `fast_rope`):

- the class token gets the identity rotation (cos 1, sin 0) instead of
  being cut out and put back;
- each pair of dimensions is rotated by a product with a fixed 64 by 64
  matrix.

In PyTorch it gives the reference's embeddings exactly (largest difference
0.0). PE then takes about 1.16 times SigLIP 2 B/16's CPU time per frame, and
the check holds (0.9999999 in float32, 0.9989 at worst in bfloat16).

Stage 1 is measured again with it from the start: the tuning sets embedded
again, PE's bank chosen again, the four tuning metrics measured again. Its
numbers decide, and the first measurement is reported beside.

**Stage 1's numbers** (`scripts/pe_core.py bank`, `scripts/compare.py tune
--round 9`; `results/round9_bank.json`, `results/round9_tuned.json`), with
the image tower as finally exported. SigLIP 2 B/16's, recomputed by the same
code, are the seventh round's to every decimal.

PE's own settings, chosen on the tuning sets:

- the query bank's beta 50 and strength 1 (DiDeMo's R@5 35.0; beta 50 and
  strength 0.5 next, 34.6);
- the bank's strength 0.5 for adaptive spans (YouCook2's R@5 29.9, against
  29.6 with 0.25 and with 1);
- the weight of speech 0.05 (`normalise` best), the weight of sound 1.5.

| the encoders alone, on the tuning sets | DiDeMo | Charades-STA | YouCook2 | UnAV-100 | mean |
|---|---|---|---|---|---|
| SigLIP 2 B/16 | 31.3 | 44.9 | 29.4 | 41.0 | 36.66 |
| PE core B/16 | 35.0 | 47.8 | 29.9 | 43.3 | 39.00 |
| PE core B/16, first export (reported beside) | 34.8 | 47.8 | 29.7 | 42.8 | 38.79 |

**The rule holds.** PE core B/16's mean is 2.34 points higher than SigLIP 2
B/16's (at least 1 was asked for), and it is higher on each of the four
sets. Stage 2 runs.

The first export, measured before the rotary position embedding was written
for the export, chose the same bank. It tied the spans' strengths 0.25 and
0.5 and kept the first, and its weight of sound was 1
(`results/round9_first_export.json`).

**Stage 2's numbers** (`scripts/round9.py`; `results/round9_stage2.json`,
`results/round9_tuned.json`, each run's log in `results/temporal/`),
recorded before the test.

The three runs trained with v3's flags on PE core B/16's frames, their
logged recipe checked. Each kept its best check:

- v4: step 4,500 of 6,000 (the check is the mean of DiDeMo's tuning R@5 and
  Charades-STA's held-out R1: 44.2);
- v4 without Charades-STA's videos: step 4,500 (DiDeMo's tuning R@5 alone,
  36.2);
- v4 without YouCook2's videos: step 5,000 (44.5).

v4 is exported as the package would ship it, alpha 1
(`results/temporal/temporal-pe-core-b16-v4.npz`, 1,585,408 weights).

Their speech and sound weights, chosen on the tuning sets:

- v4: speech 0.005, what is said within the moment; sound 1;
- v4 without Charades-STA's videos: speech 0.05 (`normalise` best); sound 1;
- v4 without YouCook2's videos: speech 0.005; sound 1.

| on the tuning sets | DiDeMo | Charades-STA | YouCook2 | UnAV-100 | mean |
|---|---|---|---|---|---|
| SigLIP 2 B/16 alone | 31.3 | 44.9 | 29.4 | 41.0 | 36.66 |
| PE core B/16 alone | 35.0 | 47.8 | 29.9 | 43.3 | 39.00 |
| v3 (the seventh round's) | 33.7 | 50.6 | 32.1 | 42.1 | 39.63 |
| v4 | 35.7 | 52.7 | 30.5 | 43.6 | 40.63 |
| v3 without Charades-STA's videos | 34.0 | 46.4 | 32.6 | 42.1 | 38.79 |
| v4 without Charades-STA's videos | 36.2 | 48.8 | 31.2 | 43.3 | 39.88 |
| v3 without YouCook2's videos | 34.0 | 51.6 | 29.1 | 41.5 | 39.04 |
| v4 without YouCook2's videos | 35.8 | 53.1 | 28.8 | 44.6 | 40.58 |

On the tuning sets, v4's mean is 1.00 point above v3's:

- higher on DiDeMo (+2.0), Charades-STA (+2.1) and UnAV-100 (+1.5);
- lower on YouCook2 (-1.6).

Trained on YouCook2's videos, PE gains 1.8 points on YouCook2 (28.8 to
30.5), and SigLIP 2 B/16 3.1 (29.1 to 32.1). Nothing is chosen from these
numbers. The test runs next, once, with the four candidates and the
criteria above.

## Tenth round: three ideas screened on the tuning sets

Written before any of its numbers was computed.

**The question.** The ninth round asked whether a better encoder helps. This
round asks whether three changes to the search and to what the model learns
do. Each is measured on the four tuning sets only, with the seventh round's
metrics and their mean (`scripts/screen.py`, `scripts/compare.py tune --round
10`, `results/round10_screen.json`). No test is run in this round.

**1. Guidance.** Image-text models see what a frame shows more than what
happens in it: "the dog jumps into the pool" scores the dog and the pool
more than the jump. Diffusion models had the same weakness with their
prompts, and classifier-free guidance (Ho and Salimans, 2022) answered it:
the prediction is pushed away from the one made without the prompt. Here
each query is pushed away from itself without its action:

- q_g = (1 + w) q - w q_stripped, at unit length, searched as the query was;
- q_stripped is the description without the words of a verb list mined, by a
  rule fixed before any measure, from the training descriptions of
  whenframe's model (`results/round10_verbs.json`, 914 words): the word after
  a subject when it has a verb's ending, YouCook2's frequent imperatives, the
  base forms of the -s forms, the auxiliaries, none found after a determiner
  in more than a fifth of its uses;
- a description with none of those words is searched as it is;
- w in 0.25, 0.5, 1 and 2, on v3.

**2. Fusion.** v3 (SigLIP 2 B/16's frames) and v4 (PE core B/16's, ninth
round) succeed on different descriptions. Their scores of the same frames,
each corrected by its own query bank, are standardised per query and added,
as the tool adds two models' scores, with PE's weight w in 0.5, 0.75, 1, 1.5
and 2. On YouCook2 the sum is put back on v3's scale, so that the weights of
speech keep their meaning. Indexing would cost both encoders' time.

**3. Round trip.** In the eighth round, each library learnt from every
automatic description of its frames (Florence-2-base, one per 10 seconds),
and every recipe lost. Many of those descriptions fit any video ("a person
is walking down a hallway"). Here a library learns only from those that find
their own block first in it:

- searched by the library's starting model, with its query bank, the frame
  that matches the description best in the whole library lies in the
  10-second block the description was written for;
- the libraries, the starting models and the recipe are the eighth round's
  first: 10 passes over the descriptions kept, the library alone.

**References and the rule.** Ideas 1 and 2 are measured against v3 and idea 3
against the eighth round's starting models, each measured again here by the
same code. Each candidate's speech and sound weights are chosen on the
tuning sets, as every candidate's. An idea whose best setting raises its
reference's mean by at least 1 point is proposed for an eleventh round, with
its own protocol and its test written before any test number. One that does
not is reported and dropped.

**The numbers** (`scripts/screen.py`; `results/round10_screen.json`, the
descriptions kept in `results/round10_round_trip.json`, each library's log
in `results/temporal/`). v3 and the eighth round's starting models, measured
again by the same code, are the seventh and eighth rounds' to every decimal.

| on the tuning sets | DiDeMo | Charades-STA | YouCook2 | UnAV-100 | mean | against its reference |
|---|---|---|---|---|---|---|
| v3, the reference of 1 and 2 | 33.7 | 50.6 | 32.1 | 42.1 | 39.63 | |
| 1. guidance, w 0.25 | 32.6 | 49.9 | 31.7 | 41.0 | 38.80 | -0.83 |
| 1. guidance, w 0.5 | 31.5 | 49.9 | 31.7 | 41.3 | 38.58 | -1.05 |
| 1. guidance, w 1 | 30.4 | 49.2 | 30.7 | 41.3 | 37.89 | -1.75 |
| 1. guidance, w 2 | 27.0 | 47.9 | 27.5 | 41.0 | 35.85 | -3.79 |
| 2. fusion, PE's weight 0.5 | 36.6 | 51.7 | 32.0 | 43.8 | 41.04 | +1.41 |
| 2. fusion, 0.75 | 37.4 | 52.3 | 32.1 | 43.6 | 41.34 | +1.70 |
| 2. fusion, 1 | 37.4 | 52.0 | 31.5 | 44.4 | 41.32 | +1.68 |
| 2. fusion, 1.5 | 37.1 | 52.1 | 31.8 | 43.6 | 41.16 | +1.53 |
| 2. fusion, 2 | 37.1 | 52.4 | 31.8 | 43.3 | 41.18 | +1.55 |
| the eighth round's starting models, the reference of 3 | 33.7 | 46.4 | 29.1 | 42.1 | 37.82 | |
| 3. round trip | 32.9 | 45.7 | 29.1 | 40.5 | 37.05 | -0.78 |

Each candidate's speech and sound weights, chosen on the tuning sets:

- guidance: speech 0.005, what is said up to 5 seconds around a frame, at
  every w; sound 2, 1.5, 0.75 and 1.5 at w 0.25, 0.5, 1 and 2;
- fusion: speech 0.005, up to 5 seconds around a frame at PE's weights 0.5
  and 0.75, within the moment at 1, 1.5 and 2; sound 2 at 0.5, 1.5 at the
  others;
- round trip: speech 0.05 (`normalise` best); sound 1.5.

The round trip kept 412 of DiDeMo's 1,446 descriptions, 232 of
Charades-STA's 1,551, 392 of YouCook2's 2,534 and 138 of UnAV-100's 471.

**The rule.** Fusion's best setting, PE's weight 0.75, raises v3's mean by
1.70 points: fusion is proposed for an eleventh round. Guidance's best, w
0.25, lowers v3's mean by 0.83, and the round trip lowers its reference's by
0.78: both are dropped.

## Eleventh round: v3 and v4 together, and v5 taught by them

Written before any of its numbers was computed.

**The question.** In the tenth round, v3's scores fused with v4's raised the
mean of the four tuning metrics by 1.70 points, for both encoders' indexing
time. This round tests the fusion once. It also asks whether whenframe's
model, taught by the fusion, can find as much at v3's cost
(`scripts/round11.py`).

### Part 1: the fusion

Fixed in the tenth round; nothing is chosen again:

- v3's scores of SigLIP 2 B/16's frames and v4's of PE core B/16's, each
  corrected by its own query bank, standardised per query, added with PE's
  weight 0.75 and put back on v3's scale (`scripts/compare.py`, `fused`);
- its speech and sound weights, chosen then on the tuning sets: speech
  0.005, what is said up to 5 seconds around a frame; sound 1.5.

Reported beside, deciding nothing: v3's and v4's runs without Charades-STA's
videos, fused the same way, on Charades-STA's test, against v3's run without
them; and the same without YouCook2's. Their speech and sound weights are
chosen on the tuning sets first, as every candidate's.

### Part 2: v5, whenframe's model taught by the fusion

On SigLIP 2 B/16's frames, so that indexing costs what it costs with v3:

- v3's recipe, as `scripts/round9.py` trained v4 with it: the leash's weight
  30, width 256 and two layers, data seed 0, 6,000 steps, a check every 500
  on DiDeMo's tuning R@5 and Charades-STA's held-out R1, the best check
  kept, exported with alpha 1;
- with the seventh round's teacher's losses (`--distil`): Kullback-Leibler
  divergences from the teacher's distributions to the model's, over the
  batch's videos by their smooth best frames, and over the frames of each
  description's own video (Hinton et al., 2015; for text-video retrieval,
  TeachCLIP, Tian et al., CVPR 2024).

The teacher is the fusion, scored as the search scores
(`scripts/round11.py teacher`, `scripts/temporal.py --fusion-teacher`):

- each training frame through v3 (SigLIP 2 B/16's embedding) and through v4
  (PE core B/16's, embedded at the same instants in the ninth round), a
  video at a time over all its frames;
- each against the description as its encoder's text tower embeds it,
  corrected by its encoder's query bank, computed over the whole bank
  (DiDeMo's training descriptions; beta 50 and strength 1 for both);
- standardised per description over the batch's frames, added with PE's
  weight 0.75, put back on v3's scale (its spread per description over the
  same frames), times SigLIP 2 B/16's logit scale, as the model's own scores
  are.

Every collection is taught, as the teacher has every collection's frames. In
the seventh round, So400m's losses (weight 1, on Charades-STA's frames alone)
lowered the mean by 0.31 points and were not kept.

Choices on the tuning sets, in this order. Each is kept only if it raises the
mean of the four tuning metrics, with each run's speech and sound weights
chosen there:

1. the teacher's losses with weight 1, against v3 (39.63);
2. weight 3 instead, against the run kept so far.

v5 goes to the test only if the run kept raises v3's mean on the tuning sets
by at least 1 point (the tenth round's rule). In that case, before the test,
two left-out runs are trained with the recipe chosen: one without
Charades-STA's videos, one without YouCook2's. Each is taught by the fusion
of v3's and v4's runs without those videos, and is reported beside, as v3's
were. If the rule is not met, v5 is reported and dropped.

### Test and criteria

Once for each part, on the four tests, with the four primary metrics and the
sign test of the earlier rounds:

- part 1 now, as nothing is left to choose: v3 and the fusion;
- part 2 once v5's choices are recorded here, if v5 goes to the test: v5,
  with v3 and the fusion again. Nothing of part 2's recipe changes after
  part 1's test.

Reported beside, deciding nothing: SigLIP 2 So400m (`--mode large` today) and
the left-out runs.

33. The fusion's mean of the four primary metrics is at least 1 point higher
    than v3's, with a sign test p < 0.05.
34. The fusion loses at most 1 point on each of the four tests against v3.
35. Indexing with both encoders, each frame decoded once, costs at most 2
    times what SigLIP 2 B/16 alone costs: CPU minutes per hour of DiDeMo's
    test videos, decoding included, with OpenVINO, both measured again one
    after the other, as the other encoders' costs are.

If 33, 34 and 35 hold, `--mode large` indexes a new library with SigLIP 2
B/16 and PE core B/16 and searches it through v3 and v4 together, as
measured. So400m stays available as `--mode so400m`, and libraries already
indexed keep their encoders.

36. v5's mean is at least 1 point higher than v3's, with a sign test p <
    0.05.
37. v5 loses at most 1 point on each of the four tests against v3.

If 36 and 37 hold, a new library in the standard mode goes through v5 instead
of v3; libraries already indexed keep v3.

**Part 1's test** (`scripts/round11.py test1`; `results/round11_test.json`,
the cost in `results/round11_cost.json`). Run once, after this protocol was
pushed and the cost measured on an idle machine:

- the fusion's mean is 43.72, against 41.90 for v3: +1.81 points, with 528
  items better and 326 worse, sign test p = 4.9e-12 (criterion 33 met);
- it is higher on each test: DiDeMo +2.6, Charades-STA +2.2, YouCook2 +0.9,
  UnAV-100 +1.6 (criterion 34 met);
- indexing with both encoders, each frame decoded once, costs 9.4 CPU minutes
  per hour of video, against 5.4 with SigLIP 2 B/16 alone: 1.74 times
  (criterion 35 met).

v3, So400m and v3's left-out runs gave the earlier rounds' numbers to every
decimal. Beside: So400m's mean is 40.63. v3's and v4's runs without
Charades-STA's videos, fused, place 49.9 % of its moments, against 46.5 % for
v3's run without them; without YouCook2's, 26.4 % of its steps, against 26.1 %.
Before the test, as written, their speech and sound weights were chosen on
the tuning sets (means 40.35 and 40.50).

The session's disk filled up while v5's teacher was being written, during
part 1. Part 1's numbers were not touched. Part 2 started again, with the
teacher's last arrays written to /dev/shm (`ROUND11_TEACHER_DIR`) and its
recipe unchanged.

As fixed above, `--mode large` is to index a new library with SigLIP 2 B/16
and PE core B/16 and search it through v3 and v4 together. The tool does not
do so yet: PE core B/16's image tower, exported again from timm's weights in
the ninth round, is not published anywhere the tool could download it from.

**Part 2's choices** (`scripts/round11.py v5`; `results/round11_v5.json`, each
run's tuning numbers in `results/round11_tuned.json`). Made on the tuning sets
only, in the order written above:

1. the teacher's losses with weight 1: mean 39.69, against 39.63 for v3:
   kept;
2. weight 3: mean 40.19, against 39.69: kept.

The run kept raises v3's mean by 0.55 points, short of the 1 point asked for.
As written, v5 is reported and dropped. No left-out run was trained, part 2
has no test, and criteria 36 and 37 were not measured. A new library in the
standard mode keeps v3.

**The large mode.** The workflow Models (`.github/workflows/models.yml`,
`scripts/pe_core.py export`) built PE core B/16's image tower again on
GitHub's machines, from timm/PE-Core-B-16's weights at the revision read in
the ninth round, with the same versions of PyTorch, open_clip, timm and
OpenVINO: its files are the bytes of the tower measured
(`models/pe-core-b16/check.json`). It published them once in the release
models-1, with Meta's licence and a NOTICE. Since then `--mode large` indexes
a new library as fixed above, the tower downloaded from the release and its
SHA-256 checked, and So400m is `--mode so400m`.

**One model.** Then whenframe kept a single model: every new library is
indexed as the large mode above, and the standard, economy, precision and
so400m modes are gone. This was decided on the numbers above, with no new
measurement; libraries built in the other modes are searched as they were
built. Without whenframe's models (`--no-temporal`), the two encoders are
searched together as with them, which was not measured.

## Twelfth round: descriptions in French, and text towers in 8 bits

Written before any of its numbers was computed.

**The questions.** A description typed in French is searched as it is today.
SigLIP 2 read many languages in training; PE core B/16's text tower was not
measured on any but English. In the second round, French descriptions found
about 70 % of what English ones found with SigLIP 2 B/16 alone. And
whenframe's model downloads 2.6 GB and holds both text towers in memory at
search time (4.7 GB at most, eleventh round). This round asks whether
translating a French description into English on the user's computer finds
more than searching it as it is (part 1), and whether the text towers can
keep their weights in 8 bits without finding less (part 2)
(`scripts/round12.py`).

### Part 1: French descriptions

The French descriptions stand in for French written by people, errors
included: every description of the four tuning and test sets, and UnAV-100's
labels, translated by Opus-MT en-fr (Helsinki-NLP, Apache 2.0; Xenova's ONNX
export, full precision), greedily, as in the second round
(`scripts/translate.py`; DiDeMo's are the second round's).

Three ways of searching them, with whenframe's model as the eleventh round
tested it, every setting unchanged:

- English: the original descriptions, the reference (the eleventh round's
  numbers again);
- French as it is: the French description given to both text towers, to the
  transcripts' full-text search (YouCook2) and to CLAP (UnAV-100);
- translated: the French description translated back into English on the
  computer by Opus-MT fr-en (Xenova's ONNX export of Helsinki-NLP/opus-mt-fr-en,
  Apache 2.0), greedily, then searched as an English one everywhere.

One choice, on the tuning sets: the translator in full precision (423 MB) or
its 8-bit export (107 MB). The 8-bit one is kept unless the mean of the four
tuning metrics, translated, is more than 0.3 points lower with it.

The tool cannot ask which language a description is in, so a description is
taken for French when it has more French marks than English ones: words such
as le, la, les, une, des, du, dans, sur, avec, qui, est, sont (and the
elided l', d', qu'), and accented letters; English marks are words such as
the, an, of, in, with, is, are, from, and. Words common to both languages
(a, on, son, pour, plus) count for neither. A tie, or no mark, is English.
The word lists are fixed on the tuning sets' descriptions in both languages,
before the test.

38. Translated, the French descriptions find at least 1 point more than as
    they are, on the mean of the four primary metrics, with a sign test
    p < 0.05.
39. The detector takes at most 0.5 % of the test sets' English descriptions
    for French, and at least 95 % of their French ones.
40. Translating a description takes at most 0.5 seconds (the median, one
    description at a time, on this machine's 4 vCPUs).

If 38, 39 and 40 hold, the tool translates a description it takes for French
before searching it, with the translator chosen above (downloaded the first
time one is needed); an English description is searched as before. Reported
beside, deciding nothing: what translated French descriptions find against
the English ones.

### Part 2: the text towers in 8 bits

SigLIP 2 B/16's and PE core B/16's text towers, their weights compressed to
8 bits by NNCF (`nncf.compress_weights`, INT8_ASYM, data-free, one scale per
output channel) and run with OpenVINO. The descriptions and the query banks
are embedded by them; the frames are those indexed by the image towers in
full precision, which stay unchanged in this round; whenframe's model and
every setting are those of the eleventh round's test. Nothing is chosen on
the tuning sets.

41. The mean of the four primary metrics is at most 0.3 points lower than with
    the text towers in full precision, and no test is more than 0.5 points
    lower.
42. A search of the demo's library holds at most 2.5 GB of memory (the
    process's peak), and the models to download weigh at most 1.5 GB.

If 41 and 42 hold, the tool downloads the text towers in 8 bits, built again
from the same files and published by the workflow Models with their SHA-256
pinned, and embeds the query banks with them; libraries already indexed keep
their frames. If both parts pass, the tool does both; their combination is
not measured.

**Part 1's choices** (`scripts/round12.py french`, `detect`, `back`, `tune`;
`results/round12_detector.json`, `results/round12_tuned.json`). Made on the
tuning sets only, before the test:

- The French descriptions: the four tuning and test sets and UnAV-100's
  labels have 11,464 different descriptions. DiDeMo's 5,806 are the second
  round's translations; the other 5,658 were translated by Opus-MT en-fr
  (Xenova's export at revision 2872620), greedily, 16 at a time.
- The detector's word lists. The first lists, as written above, took none of
  the tuning sets' 3,664 English descriptions for French, and 98.1 % of
  their 3,607 French ones. Most French ones missed were short, with no
  function word: "femme s'assoit", "homme en bleu", "deux poissons nagent".
  So the lists kept count each elision as a French mark (s', l', d', qu',
  and the others), and add French's other function words (en, y, lui, aussi,
  tout, comme, quand...). They also add words frequent in descriptions of
  videos and photos that English does not have (homme, femme, personne,
  enfant, deux, gauche, droite, chien, voiture, plage, neige, anniversaire...).
  None of them occurs in any English description of the tuning sets. With
  these lists (`whenframe/translate.py`), no English description of the tuning
  sets is taken for French, and 99.3 % of the French ones are. They are not
  changed after this.
- The translator. Translated back, the French descriptions of the tuning sets
  have a mean of 36.52 with the translator in full precision and 36.06 with
  its 8-bit export: 0.47 points lower, more than the 0.3 allowed, so the one in
  full precision is kept. Reported beside: it translated a description in
  113 ms, the 8-bit one in 293 ms (medians, one at a time); French as it is
  has a tuning mean of 27.80, the English descriptions 41.34.

The test is run once, after this record is pushed: the test sets' French
descriptions translated back by the translator kept, each timed alone on the
idle machine (criterion 40), then the three ways of searching them.

**Part 1's test** (`scripts/round12.py test1`; `results/round12_test.json`).
Run once, after the record above was pushed:

- translated, the French descriptions have a mean of 40.04 on the four
  primary metrics, against 29.28 as they are: +10.76 points, with 1,220
  items better and 433 worse, sign test p = 1.2e-86 (criterion 38 met);
- the detector took none of the test sets' 8,149 English descriptions for
  French, and 7,910 of their 7,942 French ones, 99.6 % (criterion 39 met);
- translating a description took 112 ms, the median of the 7,942 French
  ones, each translated alone on the idle machine; 90 % took at most 182 ms
  (criterion 40 met).

The English descriptions gave the eleventh round's numbers to every decimal
(43.72). Beside: translated, the French descriptions find 3.68 points less
than the English ones (326 items better, 586 worse). The 388 French
descriptions of the test sets already translated on the tuning sets came out
the same.

As fixed above, the tool translates a description it takes for French before
searching it, with Opus-MT fr-en in full precision, downloaded the first time
one is needed (430 MB, each file's SHA-256 checked); `--as-typed` searches it
as written.

**Part 2's test** (`scripts/round12.py towers`, `test2` and `memory`;
`results/round12_towers.json`, `results/round12_test_8bit.json`,
`results/round12_memory.json`). NNCF 3.4.0 compressed both towers with
OpenVINO 2026.4.1 (`scripts/text8.py`, the scales computed with numpy, so that
the files are the same bytes on any machine): SigLIP 2 B/16's text tower from
1,129 to 285 MB, PE core B/16's from 708 to 357 MB. The test was run once,
while part 1's searches ran, after its translations were timed:

- the mean of the four primary metrics is 43.70, against 43.72 in full
  precision, 0.01 points lower: DiDeMo −0.08, Charades-STA −0.05, YouCook2
  +0.08, UnAV-100 0.00 (criterion 41 met); 32 items better and 36 worse,
  p = 0.72;
- three searches of the demo's library (232 videos and photos), each in a
  process of its own, peaked at 1.36 GB, against 4.78 GB in full precision;
  the models a new library downloads weigh 1.43 GB, against 2.6 GB (criterion
  42 met).

**The text towers in 8 bits.** The workflow Models built both towers again
on GitHub's machines (`scripts/text8.py release`), from the source files
downloaded at the revisions measured, with the same versions of OpenVINO,
NNCF and numpy. Its first run published nothing: PE core B/16's .xml was not
the file measured. OpenVINO names the nodes it adds with a counter of the
process, and the towers stage had compressed PE's tower after compressing
SigLIP 2's and embedding its bank in the same process; compressed here after
SigLIP 2's alone, as on GitHub's machine, the file was the one built there.
The two files differ in those names only: the weights are the same bytes, the
graphs are the same once every name is set aside, and the embeddings of 1,024
bank descriptions are the same to the bit. `release` now compresses each
tower in a process of its own, and checks the weights' bytes and the graph
against the towers measured. Its second run published both towers once, in
the release models-2, with the Apache License and a NOTICE
(`results/round12_towers.json` holds their SHA-256). As fixed above,
whenframe then downloads them, each file's SHA-256 checked, embeds
descriptions with them and searches with the query banks they embed;
libraries already indexed keep their frames.

## Thirteenth round: a small model, for phones

Written before any of its numbers was computed.

**The question.** whenframe's model downloads 1.4 GB, a search holds 1.4 GB
of memory, and indexing costs 9.4 CPU minutes per hour of video: too much
for a phone, or for its browser. Can a model of at most 150 MB find most of
what it finds, by learning from it?

**The encoders.** Meta's Perception Encoder in its two small sizes, PE core
S/16 and T/16 (`facebook/PE-Core-S16-384` and `PE-Core-T16-384`, Apache
License 2.0): a ViT-S/16 and a ViT-Ti/16 image tower trained at 384 pixels,
and the same text tower, of 12 layers of width 512 and 32 tokens. Their
checkpoints weigh 349 and 278 MB in float32, most of it the text tower. Their
authors report 72.7 and 62.1 % on ImageNet, zero-shot; PE core B/16, which
whenframe's model uses beside SigLIP 2 B/16, 78.4 %.

**Files.** Exported here to ONNX from open_clip's weights
(`timm/PE-Core-S-16-384` and `timm/PE-Core-T-16-384`, at fixed revisions), in
PyTorch, as the ninth round exported PE core B/16's image tower: the image
tower at 224 and at 384 pixels (its positions computed for that size, as
open_clip computes them), the text tower with 32 tokens and CLIP's
tokenizer as published with the weights. For OpenVINO, both towers' weights
are then compressed to 8 bits as in the twelfth round (NNCF, INT8_ASYM,
data-free). **Checked before any benchmark number**, against open_clip in
PyTorch, on the ninth round's 16 frames of DiDeMo's tuning videos and 16 of
their descriptions (`scripts/pe_core.py sample`): the token ids must be equal,
and each embedding of the export in float32 at a cosine of at least 0.999 to
the reference's. If the check fails, the size is left out. The 8-bit towers
are measured on the tuning sets like any candidate.

### Stage 1, on the tuning sets: the encoders alone

**Candidates.** Four, neither with whenframe's model: S/16 and T/16, each at
224 and at 384 pixels, both towers in 8 bits.

**The same frames** as the ninth round's stage 1: the four tuning sets at
the instants of the standard mode's caches, read again and decoded at each
candidate's size. Each candidate's own settings are chosen on them as PE
core B/16's were (the query bank's beta and strength on DiDeMo's tuning
videos, the spans' strength on YouCook2's tuning steps, the weights of
speech and of sound), and the metric is the mean of the four tuning metrics.

Reported beside each: the files to download, the CPU minutes per hour of
DiDeMo's test videos to index (decoding included, OpenVINO, measured as in
the eleventh round), and the text tower's time per description.

**The rule.** The candidate kept has the highest tuning mean among those that
weigh at most 150 MB and index at most at half the current model's cost. If
none does, the round ends there: nothing is trained and no test is run.

### Stage 2: whenframe's model on its frames

**Frames.** The kept candidate embeds the frames of the collections v3 and
v4 trained on, at the same instants (DiDeMo's training videos with TEMPO-HL's
sentences, MSR-VTT's 9,000 clips, Charades-STA's training videos without the
500 held out, YouCook2's training videos), and those of the four test sets.

**Training.** Three runs of v4's recipe on its dimensions, unchanged:

- without a teacher, as v4;
- taught by the current model (v3's and v4's scores of the training frames,
  together, as in the eleventh round), the teacher's losses at weight 1;
- the same at weight 3.

The run with the highest tuning mean is kept (alpha 1); its speech and sound
weights are chosen on the tuning sets. Its two left-out runs (without
Charades-STA's videos, without YouCook2's) are trained as v4's were.

### Test and criteria

Once, on the four tests, with the four primary metrics. Candidates: the
small model, its encoder alone, and as the reference, the current model
with its text towers in 8 bits (the twelfth round's test: mean 43.70).

43. The small model's files to download weigh at most 150 MB.
44. Its mean of the four primary metrics is at least 85 % of the current
    model's: at least 37.15.
45. On DiDeMo's test, it puts the right video among its first five more often
    than the best published system (EventFormer, 25.4 %).
46. Indexing with it costs at most half of what the current model costs.
47. A search of the demo's library holds at most 0.5 GB of memory (the
    process's peak).

Reported beside, deciding nothing: its text tower's median time to embed a
description in a browser (onnxruntime-web, WebAssembly, in headless
Chromium on this machine), and each left-out run on the test it left out.

If 43 to 47 hold, whenframe ships the small model as `whenframe index
--model small`: a library built with it is indexed and searched with it, and
the demo site searches its library with it in the browser, a description of
the visitor's own embedded there without a server. The current model stays
the default. The test is run once, after stage 1's numbers and stage 2's
training are recorded here.

### Recorded before stage 2

**The check, before any benchmark number** (`scripts/small.py check`; each
export's `check.json`). For the four candidates the token ids are equal to
open_clip's, and the embeddings of the exports in float32 are at a cosine of
0.9999998 at worst to open_clip's, image and text towers alike.

**Stage 1's numbers** (`scripts/round13.py`; `results/round13_bank.json`,
`results/round13_tuned.json`, `results/round13_cost.json`,
`results/round13_stage1.json`). A restart of the machine stopped the round
during the bank's choice, after its first number; the steps resumed where
they stopped (S/16 at 224 on DiDeMo's tuning videos, chosen again, gave the
same R@5 to the decimal).

Each candidate's own settings, chosen on the tuning sets:

| candidate | bank's beta | bank's strength | spans' strength | speech | sound |
|---|---|---|---|---|---|
| S/16 at 224 | 50 | 1 | 0.5 | 0.01 | 1.5 |
| S/16 at 384 | 20 | 0.5 | 0.25 | 0.005 | 1 |
| T/16 at 224 | 50 | 1 | 0.25 | 0.01 | 1.5 |
| T/16 at 384 | 50 | 0.5 | 0.25 | 0.005 | 2 |

| the encoders alone, on the tuning sets | DiDeMo | Charades-STA | YouCook2 | UnAV-100 | mean | files | CPU min per hour | text |
|---|---|---|---|---|---|---|---|---|
| S/16 at 224 | 24.6 | 35.9 | 23.6 | 40.0 | 31.02 | 94.4 MB | 4.93 | 6.7 ms |
| S/16 at 384 | 28.5 | 41.1 | 26.3 | 42.8 | 34.69 | 95.1 MB | 10.38 | 6.1 ms |
| T/16 at 224 | 17.8 | 31.6 | 22.6 | 37.9 | 27.49 | 76.5 MB | 3.82 | 5.7 ms |
| T/16 at 384 | 22.2 | 37.1 | 23.4 | 40.3 | 30.74 | 77.0 MB | 6.45 | 6.1 ms |

The current model, measured again just after them as the eleventh round
measured it, costs 12.91 CPU minutes per hour of DiDeMo's test videos (194
ms per frame embedded); the limit is half of it, 6.45. The eleventh round
measured it at 9.36 on a machine of the same model; here everything takes
about 35 % longer, the candidates as the current model, and the rule
compares numbers measured one after the other on the same machine.
Decoding takes 2.7 to 3.1 of these minutes whatever the encoder.

**The rule.** S/16 at 384 costs more than the current model itself and T/16
at 384 just over the limit (6.4537 against 6.4530); both weigh less than
150 MB. Among the two within the limits, **S/16 at 224 is kept** (mean
31.02, against 27.49 for T/16 at 224). T/16 at 384 within them would not
have changed it (30.74). Stage 2 runs on S/16 at 224's frames.

### Recorded before the test

**Stage 2's numbers** (`scripts/round13.py texts` and `stage2`;
`results/round13_stage2.json`, `results/round13_tuned.json`, each run's log
in `results/temporal/`), recorded before the test.

S/16 at 224 embedded the frames of the four collections and of the four
tests at v4's instants, and its text tower the collections' descriptions.
The teacher, the current model's v3 and v4 fused over every training frame
as in the eleventh round, was kept in memory (`/dev/shm`), as there.

The three runs trained with v4's flags on its frames (width 256, two layers,
four heads, the leash at 30, 6,000 steps), their logged recipe checked. Each
kept its best check, the mean of DiDeMo's tuning R@5 and Charades-STA's
held-out R1:

- without a teacher: step 5,000 (34.7);
- taught, weight 1: step 5,000 (35.0);
- taught, weight 3: step 5,500 (35.5).

Their speech and sound weights, chosen on the tuning sets (what is said
within 5 seconds of a frame, for the three):

- without a teacher: speech 0.05 (`normalise` best); sound 1.5;
- taught, weight 1: speech 0.005; sound 1;
- taught, weight 3: speech 0.05 (`normalise` best); sound 1.

**Kept: the run taught at weight 3**, the highest mean, 3.72 points above
its encoder alone. It is exported as the package would ship it, alpha 1
(`results/temporal/temporal-pe-core-s16-224-small.npz`, 1,321,728 weights),
and measured again as exported, its weights in float16 as the package ships
them: the same on DiDeMo, Charades-STA and UnAV-100, one more of YouCook2's
619 tuning steps found (26.3 against 26.2), and the same speech and sound
weights chosen. That is the small model, its tuning mean 34.78.

Its two left-out runs were trained with its recipe, taught by the fusion of
v3's and v4's own left-out runs, as in the eleventh round. The teacher
without YouCook2's videos was computed while the three runs were being
measured, by the same code, ahead of its turn. Each kept its best check:

- without Charades-STA's videos: step 2,500 (27.4, DiDeMo's tuning R@5
  alone, as only a run that trains on Charades-STA is checked on its
  held-out videos);
- without YouCook2's videos: step 6,000 (35.7).

Their speech and sound weights, chosen on the tuning sets:

- without Charades-STA's videos: speech 0.005, what is said within the
  moment; sound 1.5;
- without YouCook2's videos: speech 0.01, within 5 seconds of a frame;
  sound 1.5.

| on the tuning sets | DiDeMo | Charades-STA | YouCook2 | UnAV-100 | mean |
|---|---|---|---|---|---|
| S/16 at 224 alone | 24.6 | 35.9 | 23.6 | 40.0 | 31.02 |
| without a teacher | 26.4 | 43.0 | 25.2 | 41.0 | 33.89 |
| taught, weight 1 | 27.4 | 42.5 | 25.5 | 42.1 | 34.37 |
| taught, weight 3 | 27.5 | 43.5 | 26.2 | 41.8 | 34.74 |
| **the small model, as exported** | **27.5** | **43.5** | **26.3** | **41.8** | **34.78** |
| without Charades-STA's videos | 27.4 | 37.2 | 25.7 | 41.0 | 32.83 |
| without YouCook2's videos | 27.5 | 43.8 | 24.4 | 41.0 | 34.19 |

For scale, deciding nothing: on the tuning sets the current model's mean is
41.34 (the twelfth round, its text towers then in full precision), and the
small model's 34.78 is 84.1 % of it. Criterion 44 asks for 85 % of the
current model's mean on the tests.

**How criteria 43, 46 and 47 are read**, fixed before the test:

- 43: the files a new library with the small model downloads. These are its
  encoder's five, as whenframe's cache holds them (CLIP's tokenizer, and
  each tower's XML and its weights in 8 bits), and the two whenframe's
  package adds for it (its query bank in float16, and its weights with their
  speech and sound settings). Reported beside: the two files the demo site
  downloads instead to embed a description in the browser (the text tower in
  ONNX, in 8 bits, and CLIP's merges).
- 46: indexing with the small model costs its encoder's CPU minutes per hour
  of DiDeMo's test videos as stage 1 measured them (4.93, decoding
  included), plus the CPU time of the small model over the same videos'
  frames, per hour, as the fifth round measured it. The current model's is
  its two encoders' (12.91, measured just after), plus v3's and v4's over
  their frames, measured the same way.
- 47: the demo's library (67 videos and 165 photos) is indexed with the small
  model as the package would ship it. Its query bank and weights are read
  from the files the test writes, since `whenframe/data` holds neither until
  the test decides. The library is then searched with the twelfth round's
  three descriptions ("a geyser shooting steam into the sky", "a pipe band
  marching down the street", "fireworks"), each search a process of its own,
  and the highest peak counts. Reported beside: the current model's peaks,
  on the demo's library as the site was built.

That path was tried once before the test, to check only that it runs, on
three of the demo's files copied to a scratch folder: they were indexed and
searched, and nothing was measured.

The reference, the current model with its text towers in 8 bits, is measured
again by the same code: its numbers must be the twelfth round's. The test
runs next, once.

### The test

**Run once** (`scripts/round13.py test`; `results/round13_test.json`), after
the record above was pushed, on the idle machine:

- 43: the small model's files weigh 101.2 MB, its encoder's five 94.4 MB and
  whenframe's two 6.8 MB (met);
- 44: its mean on the four tests is 36.14 (DiDeMo 21.2, Charades-STA 48.1,
  YouCook2 23.5, UnAV-100 51.7), 82.7 % of the current model's 43.70, where
  37.15 was asked (**not met**);
- 45: on DiDeMo's test it puts the right video among its first five for
  38.9 % of the descriptions, against EventFormer's 25.4 % (met);
- 46: indexing with it costs 4.97 CPU minutes per hour of DiDeMo's test
  videos, against 12.99 for the current model, 38 % (met); the small model
  adds 0.040 of them, v3 and v4 0.044 each;
- 47: a search of the demo's library peaked at 0.29 GB at most over the
  three descriptions, the current model's at 1.34 GB (met).

The reference gave the twelfth round's numbers exactly (mean 43.70). Beside:
the encoder alone has a mean of 32.72, and the small model finds more than
it (887 items better, 425 worse, sign test p = 7.2e-38); its text tower
embeds a description in a browser in 125 ms (median of 230), after 4.6
seconds to load; trained without Charades-STA's videos it finds 40.5 on
Charades-STA's test (48.1 with them), without YouCook2's 22.2 on YouCook2's
(23.5).

**Decision.** Criterion 44 is not met, so the small model is not shipped:
whenframe keeps its one model, `--model small` is not added, and the demo
site searches as it did. The test is not run again.

## Fourteenth round: a large model, at most three times the cost

Written before any of its numbers was computed.

**The question.** whenframe's model reads each frame with two image towers
of base size, SigLIP 2 B/16 and PE core B/16, for 9.4 CPU minutes per hour
of video. Does a larger tower find more, for at most three times that cost?
And can one model hear what a video says and what it sounds like, in place
of a transcription and of CLAP?

### Part A: PE core L/14 in place of PE core B/16

**The encoder.** Meta's Perception Encoder in its large size, PE core L/14
(`facebook/PE-Core-L14-336`, Apache License 2.0): a ViT-L/14 image tower of
0.32 billion weights trained at 336 pixels, and a text tower of 24 layers of
width 1024 and 32 tokens. Its authors report, zero-shot, 83.5 % on ImageNet,
73.4 % on Kinetics-400 and 50.3 on MSR-VTT's text-to-video retrieval, where
PE core B/16 has 78.4, 65.6 and 47.6.

**Files.** Exported here from open_clip's weights (`timm/PE-Core-L-14-336` at
revision 8eff41b3), as the thirteenth round exported S/16 and T/16: the
image tower read at 224 pixels, 256 patches, its positions computed for that
size as open_clip computes them, so that both encoders together cost at
most three times what the current model costs; its weights in float, as PE
core B/16's; the text tower with 32 tokens and its weights in 8 bits, as the
twelfth round's (NNCF, INT8_ASYM). **Checked before any benchmark number**
as the thirteenth round's exports were: the token ids equal to open_clip's,
and each embedding of the export in float32 at a cosine of at least 0.999
to open_clip's, on the ninth round's 16 frames and descriptions.

**Stage 1, on the tuning sets.** The four tuning sets at the standard
mode's instants, read again at 224 pixels; PE core L/14's own settings
chosen there as PE core B/16's were (the query bank's beta and strength,
the spans' strength, the speech and sound weights). Measured: PE core L/14
alone, PE core B/16 alone (its text tower in 8 bits, measured again the same
way), and the cost of SigLIP 2 B/16 and PE core L/14 together, each frame
decoded once, against the current model's, both measured again one after
the other as in the eleventh round. **The rule:** stage 2 only if PE core
L/14 alone has a higher tuning mean than PE core B/16 alone, and if the two
encoders together cost at most three times the current model.

**Stage 2.** PE core L/14 embeds the frames of the collections v3 and v4
trained on, at the same instants, and those of the four test sets. v6 is v4's
recipe, unchanged, on its frames (scripts/round9.py's flags, no teacher),
with its two left-out runs (without Charades-STA's videos, without
YouCook2's). The large model is v3 and v6 fused as the tenth round fused v3
and v4, PE's weight chosen on the tuning sets among the tenth round's (0.5,
0.75, 1, 1.5, 2); its speech and sound weights are chosen there too.

### Part B: what is heard, by EmbeddingGemma 2

**The model.** Google DeepMind's EmbeddingGemma 2 (`google/embeddinggemma-2`
at revision f9972ac2, Apache License 2.0, 740 million weights): text in
more than 100 languages, images, video and audio in one space of 768
dimensions. Its image tower costs too much for whenframe's frames: on this
machine (PyTorch, float32, one thread), a frame at its smallest vision
budget, 70 tokens, takes 6.6 times what SigLIP 2 B/16 takes, so it is not a
candidate for them. Its audio tower is: 2.0 CPU seconds for 10 seconds of
sound.

**What it replaces, where.** A video's sound, mono at 16 kHz, cut into
CLAP's windows (5 seconds, overlapping by half), each embedded by its audio
tower as its model card says (sound without a prefix), and each description
by its text tower after `task: search result | query: `. A frame takes the
best cosine of the windows around its middle; these scores are standardised
as CLAP's are and added to the frames' with a weight chosen over the grid of
the scores they replace:

- on UnAV-100's tuning videos, in place of CLAP's (the fourth round's
  weights, 0.25 to 3);
- on YouCook2's tuning steps, in place of the transcript's full-text scores
  (the third round's speech weights).

Measured on the tuning sets with the reference implementation (PyTorch,
transformers 5.24.1, float32), over the current model's frames. **The rule,
for each set:** EmbeddingGemma 2 is kept in place of what it replaces only
if that set's tuning metric is higher with it. If it is kept for either,
its audio and text towers are exported for OpenVINO and checked before the
test (each embedding at a cosine of at least 0.999 to the reference's, on
50 windows of UnAV-100's tuning videos and 50 descriptions), and its CPU
minutes per hour of sound are measured as the transcription's and CLAP's
are, one after the other.

### Test and criteria

Once, on the four tests, with the four primary metrics and the sign test of
the earlier rounds, after stage 1's numbers, stage 2's training and part B's
choices are recorded here. Candidates: the large model, with what part B
kept; the current model (the twelfth round's test: mean 43.70); and, beside,
PE core L/14 alone, the left-out runs on the test they left out, and the
large model without part B's choices.

48. Indexing with the large model costs at most three times what the
    current model costs, with what each hears (CLAP and the transcription,
    or EmbeddingGemma 2 where it replaces them).
49. Its mean of the four primary metrics is at least 1 point higher than the
    current model's (at least 44.70), with a sign test p < 0.05.
50. It loses at most 1 point on each of the four tests against the current
    model.

If 48 to 50 hold, whenframe ships it as `whenframe index --model large`. It
becomes the default for new libraries only if its mean is at least 3 points
higher than the current model's (at least 46.70); otherwise the current model
stays the default. Libraries already indexed keep their model.
