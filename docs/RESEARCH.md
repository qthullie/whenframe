# Research

What exists for finding a described moment in videos, what it costs, and where
whenframe sits. Numbers are as reported by their authors; they come from
different benchmarks and settings and are not comparable with each other.

## The problem

A person looking for a moment in their own videos types a description and
expects the moment, not a list of files. Two things set this apart from the
usual research setting:

- **They search a library, not one video.** Most moment-retrieval work assumes
  the right video is given and only the moment has to be placed. In a library,
  the video has to be found first.
- **It runs at home.** A NAS or a laptop, no GPU, thousands of hours of video,
  and queries answered while the person waits.

## Products and tools

| | What it does | Limits |
| --- | --- | --- |
| Apple Photos, iOS 18.1 ([Tom's Guide](https://www.tomsguide.com/phones/ios-18-1-lets-you-search-for-moments-in-videos-heres-how-it-works)) | natural-language search that jumps to the matching scene of a video, indexed on the device | Apple devices with Apple Intelligence only; closed |
| [Immich](https://immich.app/docs/features/searching) | self-hosted photo and video library with CLIP search | videos are searched through their thumbnail only; search inside videos has been requested since 2024 ([#10789](https://github.com/immich-app/immich/discussions/10789), [#25637](https://github.com/immich-app/immich/discussions/25637)) |
| [immich-moments](https://github.com/Booyaka101/immich-moments) (2026) | scene detection, frame embeddings with Immich's CLIP model, speech transcripts | no published accuracy; tested on short films and synthetic clips according to its README |
| [Twelve Labs](https://www.twelvelabs.io) | video search API | cloud, paid |

## Research

**Supervised moment retrieval.** Models trained on a benchmark's own
annotations: Moment-DETR and QVHighlights ([Lei et al., 2021](https://arxiv.org/abs/2107.09609)),
and the reproducible collection in [Lighthouse](https://arxiv.org/abs/2408.02901)
(six models, three feature sets, five datasets). They need features and a
trained head for each domain.

**Training-free grounding with multimodal language models.** A frozen
vision-language model is asked where the moment is:
[Moment-GPT](https://arxiv.org/abs/2501.07972), [VTG-GPT](https://arxiv.org/abs/2403.02076),
[GranAlign](https://arxiv.org/abs/2601.00584) (AAAI 2026),
[REZE](https://arxiv.org/abs/2608.04480) (reports 40.32 moment-retrieval mAP
on QVHighlights, up from 38.23 for the previous training-free best),
["Your VLM Already Knows When"](https://arxiv.org/abs/2608.08315) (56.8 %
R@0.5 on Charades-STA by asking yes-or-no questions), and
[MarkIt](https://arxiv.org/abs/2604.25886). These run a 3-7 billion parameter
model over the frames of the video for every query: accurate, and far beyond a
home server when the library holds thousands of hours.

**Long videos.** [CONE](https://arxiv.org/abs/2209.10918) and
[TimeScope](https://arxiv.org/abs/2509.26360) narrow the search coarse-to-fine.
[Point to Span](https://arxiv.org/abs/2512.10363) is the closest to whenframe:
training-free, CLIP frame similarities, spans grown from peaks of the similarity
signal, a small language model to decompose the query, under a second per query
on hour-long videos. It still searches one video at a time.

**Encoders.** [SigLIP 2](https://arxiv.org/abs/2502.14786) scores each
image-text pair with its own sigmoid, so scores are comparable across videos;
a tenth of its training pairs come from non-English pages.
[Perception Encoder](https://arxiv.org/abs/2504.13181) (Meta, 2025) reports
better zero-shot video results than InternVideo2 on most tasks with a simple
frame-level encoder.

**Cheaper encoders.** [MobileCLIP2](https://arxiv.org/abs/2508.20691) (Apple,
2025) and MobileCLIP report accuracy close to ViT-B/16 CLIP models at a fraction
of the latency, but Apple licenses the weights for non-commercial research only.
SigLIP 2 also comes as B/32 at 256 px (64 patches instead of 196), under Apache
2.0. Weight-only 4-bit and static int8 quantization of ViTs are the other usual
routes to a cheaper encoder; on this CPU the first gave no speed-up and the
second broke the embeddings ([PROTOCOL.md](PROTOCOL.md), second round).

**Hubs.** In a shared text-image space, a few gallery items are close to
almost every query and crowd the top of the rankings. Query-bank normalisation
([Bogolin et al., CVPR 2022](https://arxiv.org/abs/2112.12777)) rescales each
gallery item's scores by how it scores against a fixed bank of unrelated
queries, without retraining and without seeing the other test queries. The dual
softmax of [Cheng et al. (2021)](https://arxiv.org/abs/2109.04290) normalises
over the test queries themselves, which a tool answering one query at a time
cannot do.

**Video search systems.** [This 2025 system](https://arxiv.org/abs/2504.08384)
combines a coarse and a fine model, keeps only representative keyframes to save
storage, and re-ranks with neighbouring frames.

## The gap

I found no published method or measurement for the setting above: zero-shot
retrieval of moments across a whole library, on personal videos, with the cost
of indexing and searching measured on a CPU. The closest pieces exist
separately: training-free single-video grounding (accurate, expensive),
corpus moment retrieval (supervised, e.g. on TV shows), and home tools (no
measured accuracy).

## What whenframe does about it

- Searches the whole library at once and ranks moments across videos with
  SigLIP 2's calibrated scores.
- Marks down frames that match almost any description, with query-bank
  normalisation applied to video frames. The correction is computed once per
  frame, when the frame is indexed, so it adds nothing to a search.
  On DiDeMo it adds about two points of recall, by ranking the right video
  higher ([RESULTS.md](RESULTS.md)).
- Places moments as fixed windows around the best frames of each video. Spans
  grown from the score curve, as in Point to Span, were measured as well: they
  are more precise at IoU >= 0.7 and less often right at IoU >= 0.5, on a
  dataset whose moments are 5-second segments.
- Skips frames that repeat the previous one before encoding them.
- Offers a cheaper mode (SigLIP 2 B/32) and a more accurate one (B/16 and B/32
  combined), with their costs measured, and a search page that keeps the model
  loaded.
- Trains a small model of its own (1.45 million weights): a temporal
  adapter in the spirit of [CLIP4Clip](https://arxiv.org/abs/2104.08860)'s
  temporal transformer, which rewrites each frame's SigLIP 2 embedding with
  its neighbours', trained on the search's own scores, query bank included,
  leashed to SigLIP 2's embeddings, on DiDeMo's and MSR-VTT's training
  videos, and averaged over three runs (a model soup, [Wortsman et al.,
  2022](https://arxiv.org/abs/2203.05482)). It is on by default in the
  standard mode, having found more than SigLIP 2 alone on four benchmarks
  without losing on any.
- Compares its ways of embedding frames the same way on four benchmarks
  (DiDeMo, Charades-STA, YouCook2, UnAV-100), the costs measured alike:
  So400m finds the most, for 3.4 times the CPU time; whenframe's model gives
  part of that gain for almost none of the cost ([RESULTS.md](RESULTS.md),
  sixth round).
- Measures all of it on DiDeMo, personal Flickr videos, with the protocol fixed
  in advance ([PROTOCOL.md](PROTOCOL.md)), against a shot-based baseline in the
  spirit of current home tools, with the CPU cost of indexing and searching.

## Against published numbers

[figures/benchmark.png](figures/benchmark.png) sets whenframe beside published
methods only where the benchmark, the split and the metric are the same:
placing a described moment in a given Charades-STA test video, R1 at IoU 0.5,
with no training on Charades-STA's annotations. As their authors report them:
["Your VLM Already Knows When"](https://arxiv.org/abs/2608.08315) 56.8 %, with
a 7-billion-parameter vision-language model;
[TFVTG](https://arxiv.org/abs/2408.16219) (Zheng et al., ECCV 2024) 49.97 %,
with a language model and vision-language models;
[VTG-GPT](https://arxiv.org/abs/2403.02076) (Xu et al., 2024) 43.68 %;
[PSVL](https://arxiv.org/abs/2110.00428) (Nam et al., ICCV 2021) 31.29 %,
trained on unannotated videos. whenframe places 45.7 % (fixed windows), 46.1 %
with its own model (trained on DiDeMo and MSR-VTT, not on Charades), 47.2 %
(adaptive spans) and 50.4 % with SigLIP 2 So400m, with its DiDeMo parameters,
on a CPU, from frames it embedded once for every search; the first three run language or
vision-language models over the video for each sentence, and PSVL trains a
model on Charades videos without their annotations. The home-tool approaches of the same figure (one image per video, one
keyframe per shot, frames searched without the query bank) are run here, on
the same videos and descriptions as whenframe.

## What is not claimed

- No comparison is made with numbers from other benchmarks or settings than the
  one above.
- A frame-level encoder cannot see motion: descriptions like "the camera pans
  left" are out of its reach, and DiDeMo has many of them.
