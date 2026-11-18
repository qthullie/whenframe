# whenframe

<img src="https://raw.githubusercontent.com/qthullie/whenframe/main/site/logo.svg" alt="" width="56" align="left">

Find a moment in your own videos by describing it. whenframe indexes a folder
of videos (and photos) on your computer, with no GPU and nothing sent anywhere,
and answers "the dog jumps into the pool" with the video and the seconds where
it happens.

**[See it at work in your browser](https://qthullie.github.io/whenframe/)** on three open movies, or install it below.

<p align="center">
  <a href="https://qthullie.github.io/whenframe/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/site-dark.webp">
      <img src="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/site.webp" alt="The demo site: the three films; 'a girl hugging a baby dragon' finds three moments of Sintel, and the first, at 3:28, plays its ten seconds; then 'two people arguing on a bridge' finds the opening scene of Tears of Steel" width="800">
    </picture>
  </a>
  <br>
  <sub>The demo site, searching Big Buck Bunny, Sintel and Tears of Steel (© Blender Foundation, CC BY 3.0) in the browser.</sub>
</p>

<!-- table readme-headline -->
Searching 979 personal videos at once (DiDeMo test) for 3,902 descriptions written by people, whenframe puts the right moment among its first five results **29.2 %** of the time with its own model (27.6 % without it; 29.5 % with SigLIP 2 So400m, for 3.4 times the CPU time). Indexing one keyframe per shot, as some home video tools do, finds it 2.3 % of the time.
<!-- /table -->

## Use

```bash
pip install git+https://github.com/qthullie/whenframe   # Python 3.10+, and ffmpeg on the PATH
whenframe index ~/Videos                                # once, then again to add new videos
whenframe search "the dog jumps into the pool"
whenframe serve                                         # a search page, at http://127.0.0.1:8765
whenframe watch ~/Videos                                # or keep the index up to date on its own
```

```
 83%     1:12 - 1:22      /home/me/Videos/2024/summer/IMG_4120.MOV
 71%     0:03 - 0:13      /home/me/Videos/2023/garden.mp4
```

`whenframe serve` opens a search page on your own library, which plays each
video from the moment found:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/demo-1-dark.webp">
    <img src="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/demo-1.webp" alt="The local search page of whenframe serve: 'two people arguing on a bridge' finds the opening scene of Tears of Steel first, among three films" width="800">
  </picture>
</p>

The first run downloads the SigLIP 2 model (1.5 GB). A library is indexed in
one of four modes, chosen the first time (`--mode`); measured the same way on
four benchmarks (sixth round, below):

<!-- table readme-models -->
| `--mode` | right moment in the first five (DiDeMo) | mean of the four benchmarks | indexing, CPU min per hour of video (OpenVINO) |
|---|---|---|---|
| `standard`, with whenframe's model (the default) | 29.2 % | 38.65 | 7.3 |
| `standard --no-temporal` | 27.6 % | 38.07 | 7.2 |
| `economy` | 24.3 % | 36.80 | 4.3 |
| `large` | 29.5 % | 40.63 | 24.9 |
<!-- /table -->

The `precision` mode (B/16 and B/32 combined) found as many moments as
`standard` on DiDeMo in the second round and was not part of the sixth.

<!-- table readme-openvino -->
With OpenVINO (`pip install 'whenframe[fast]'`) on a processor that computes in bfloat16, as this one does, the standard mode indexes for 6.5 CPU minutes per hour of video instead of 24.9, and finds 27.5 % of the moments instead of 27.2 % (DiDeMo test).
<!-- /table -->

Photos in the same folders are indexed too, each as a moment of its own, so a
search returns photos and moments of videos together (`--no-photos` to leave
them out). `whenframe index --speech base` also transcribes what is said (Whisper, on the
CPU, `pip install 'whenframe[speech]'`), and a search then finds the words
said as well as what is seen. `whenframe index --sounds` also listens to the
sound tracks (CLAP), to find laughter, applause, a dog barking or a siren.
In a library of long videos, a moment lasts as long as the frames match
(`--moments`). In the standard mode, each video's frames go through
whenframe's own model, trained on DiDeMo and MSR-VTT (`--no-temporal` leaves
it out; its weights are for research and non-commercial use). `serve
--jellyfin` and `serve --immich` link each moment to the video in a Jellyfin
or Immich server.

Nothing else goes over the network. ONNX Runtime 1.29 sends usage telemetry to
Microsoft by default, and `import openvino` reports to Google Analytics;
whenframe turns off the first and keeps the second from loading (checked by
logging every outbound connection of an indexing and search run). More in
[docs/TUTORIAL.md](https://github.com/qthullie/whenframe/blob/main/docs/TUTORIAL.md): watching folders, Jellyfin, Immich, Docker.

## How it works

1. ffmpeg reads one frame per second; frames that barely differ from the last
   one kept are skipped.
2. [SigLIP 2](https://arxiv.org/abs/2502.14786) embeds the frames and the
   description (ONNX Runtime, CPU). Its sigmoid gives each frame an absolute
   match score, so frames from different videos compare directly. Then a
   small transformer of whenframe's own, trained on DiDeMo's and MSR-VTT's
   training videos, rewrites each frame's embedding with what the frames
   around it show (in the spirit of
   [CLIP4Clip](https://arxiv.org/abs/2104.08860)'s temporal transformer; the
   average of three runs, a model soup), for about 0.04 CPU minutes per hour
   of video.
3. Frames that match almost any description are marked down, using a bank of
   4,096 unrelated descriptions: query-bank normalisation
   ([Bogolin et al., CVPR 2022](https://arxiv.org/abs/2112.12777)), applied to
   video frames.
4. With `--speech`, the words of the description said around a frame add to
   its score (Whisper and a full-text index); with `--sounds`, so does what is
   heard around it ([CLAP](https://arxiv.org/abs/2211.06687), standardised and
   added). Both count as chosen with each mode's model on the tuning sets.
5. In each of the best videos, up to three 10-second moments are placed around
   the best frames, and all moments are ranked together. In precision mode the
   scores of two models are standardised and added first.

## Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/benchmark-dark.png">
    <img src="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/benchmark.png" alt="whenframe against what exists: on Charades-STA it places 45.7 to 50.4 % of the moments, between VTG-GPT (43.7 %) and a 7-billion-parameter vision-language model (56.8 %); on DiDeMo it finds the right video among the first five 46.4 % of the time against 28 to 31 % for one image per video, and the right moment 29.2 % with its own model (the default), 27.6 % without it, 29.5 % with SigLIP 2 So400m for 3.4 times the CPU time, against 2.3 % for one keyframe per shot; on YouCook2, speech and adaptive spans reach 26.0 to 28.5 %; on UnAV-100, frames and sound 53.2 to 54.1 %" width="800">
  </picture>
</p>

Beside published zero-shot methods, on the one benchmark and metric they
share, and beside the ways home tools index videos, run here on the same
videos ([docs/RESEARCH.md](https://github.com/qthullie/whenframe/blob/main/docs/RESEARCH.md) gives the sources).

DiDeMo test split, with the protocol fixed in advance and every change to it
listed ([docs/PROTOCOL.md](https://github.com/qthullie/whenframe/blob/main/docs/PROTOCOL.md)). The numbers on this page are
written by `scripts/tables.py` from `results/*.json`.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/results-dark.png">
    <img src="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/results.png" alt="Share of DiDeMo test descriptions whose moment is among the first five results, for each search method of the first round" width="800">
  </picture>
</p>

<!-- table didemo-criteria -->
| criterion (fixed in advance) | measured on test | met |
|---|---|---|
| 1. adaptive spans beat both baselines by at least 2 points | -1.8 points | no |
| 2. skipping repeats encodes at most 70 % of frames and loses at most 1 point | 91 % of frames, +0.2 points | no |
| 3. indexing takes at most 10 min of CPU time per hour of video | 24.9 min | no |
| 4. the query bank adds at least 1 point to adaptive spans | +1.6 points | yes |
| 5. the query bank adds at least 1 point to fixed windows | +2.2 points | yes |
<!-- /table -->

Second round (two more modes, French descriptions):

<!-- table round2-criteria -->
| criterion (fixed before the second run) | measured on test | met |
|---|---|---|
| 6. the economy mode loses at most 2 points | -2.7 points | no |
| 7. economy indexing takes at most 10 min of CPU time per hour of video | 8.4 min | yes |
| 8. the precision mode gains at least 1 point | +0.1 points | no |
<!-- /table -->

On Charades-STA, with the parameters chosen on DiDeMo and nothing trained or
tuned on Charades, it places the moment in a given video 46.1 % of the time at
IoU >= 0.5 with its own model (45.7 % without it, 50.4 % with SigLIP 2
So400m); a 7-billion-parameter vision-language model, run over the video's
frames for each sentence, is reported at 56.8 %.

<!-- table readme-youcook2 -->
On 252 YouCook2 cooking videos (5 minutes on average, searched all at once), the right step is among the first five moments 19.4 % of the time with adaptive spans (placed by default in a library of long videos), and 25.8 % when what is said is searched too (indexed with `--speech`).
<!-- /table -->

Third round (OpenVINO, long videos, what is said):

<!-- table round3-criteria -->
| criterion (fixed before the third run) | measured on test | met |
|---|---|---|
| 9. indexing with OpenVINO takes at most 10 min of CPU time per hour of video | 6.5 min | yes |
| 10. OpenVINO loses at most 1 point on the primary metric | +0.4 points | yes |
| 11. speech adds at least 2 points on YouCook2 | +6.5 points | yes |
| 12. transcription takes at most 15 min of CPU time per hour of video | 8.1 min | yes |
<!-- /table -->

Fourth round (So400m, speech with spans, what is heard, motion; each part fixed
before its test run):

<!-- table round4-criteria -->
| criterion (fixed before its test run) | measured on test | met |
|---|---|---|
| 13. speech with adaptive spans beats speech with fixed windows by at least 2 points (YouCook2) | +7.0 points | yes |
| 14. So400m beats the base model by at least 2 points (DiDeMo, both with OpenVINO) | +1.9 points | no |
| 15. indexing with So400m takes at most 30 min of CPU time per hour of video | 25.8 min | yes |
| 16. the motion grids add at least 1 point (DiDeMo, weight 0.25) | +0.4 points | no |
| 17. frames and sound together beat the frames alone by at least 4 points (UnAV-100, P@10) | +10.3 points | yes |
<!-- /table -->

Fifth round (whenframe's own model, trained on DiDeMo's training videos):

<!-- table round5-criteria -->
| criterion (fixed before its test run) | measured on test | met |
|---|---|---|
| 18. the adapted frames add at least 1 point on the DiDeMo test | +1.6 points | yes |
| 19. they lose at most 1 point on each other benchmark | Charades-STA, R1 at IoU >= 0.5 +0.9; YouCook2, R@5, speech + spans +0.4; UnAV-100, P@10, frames + sound -1.1 | no |
| 20. the adapter adds at most 1 CPU minute per hour of video | 0.04 min | yes |
<!-- /table -->

Sixth round (the encoders compared the same way on the four benchmarks, and
whenframe's model trained again on DiDeMo and MSR-VTT; the mean of the four
primary metrics, each step significant):

<!-- table round6-criteria -->
| criterion (fixed before its test run) | measured on test | met |
|---|---|---|
| 21. v2's mean is higher than SigLIP 2 B/16's, sign test p < 0.05 | +0.57 points, p = 6.5e-05 | yes |
| 22. v2 loses at most 1 point on each test against B/16 | DiDeMo +1.5; Charades-STA +0.4; YouCook2 +0.2; UnAV-100 +0.2 | yes |
| 23. v2's mean is higher than v1's | -0.78 points | no |
<!-- /table -->

All tables, the costs and the figures: [docs/RESULTS.md](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md).

## Limits

- It sees frames, not motion: "the camera pans left" is out of its reach, and
  grids of four successive frames did not help (DiDeMo test, +0.4 points).
- What is heard (`--sounds`) was measured on short clips cut around their
  events (UnAV-100); in long home videos, where most of the sound is talk or
  nothing, it was not.
- It was measured on short personal videos (DiDeMo, their first 30 seconds) and
  on cooking videos of about 5 minutes (YouCook2); recordings of an hour or more
  were not measured.
- Windows did better on DiDeMo's short clips and spans on YouCook2's long
  videos; the tool picks one for a whole library from its median video
  (`--moments auto`), so a library mixing both kinds gets one of them.
- It cannot tell apart videos that show the same thing: among Charades'
  1,334 look-alike videos of people at home, the right one is in the first five
  results 4 % of the time.
- Indexing is slow on a small machine without OpenVINO (see the modes above,
  measured on 4 vCPUs); OpenVINO's speed-up was measured on one processor only.
- Descriptions in French were measured through machine translation only.
- whenframe's own model learnt from DiDeMo's and MSR-VTT's descriptions; it
  gains most on personal videos like DiDeMo's (+1.5 points) and little
  elsewhere (+0.2 to +0.4), and was not measured on videos of an hour or more.
- Several criteria set in advance were not met; the results page says which and
  why.

## Documents

- [docs/RESEARCH.md](https://github.com/qthullie/whenframe/blob/main/docs/RESEARCH.md): what already exists, and what is new here
- [docs/PROTOCOL.md](https://github.com/qthullie/whenframe/blob/main/docs/PROTOCOL.md): the evaluation, written before each test run, and every change made to it
- [docs/RESULTS.md](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md): all the numbers
- [docs/TUTORIAL.md](https://github.com/qthullie/whenframe/blob/main/docs/TUTORIAL.md): installing, indexing, searching, the page, reproducing the benchmark

## Credits

- Evaluation data, used for measurements only; no video of them is
  redistributed. DiDeMo: Hendricks et al., *Localizing Moments in Video with
  Natural Language*, ICCV 2017. Charades-STA: Gao et al., *TALL*, ICCV 2017, on
  Charades: Sigurdsson et al., ECCV 2016. YouCook2: Zhou et al., *Towards
  Automatic Learning of Procedures from Web Instructional Videos*, AAAI 2018.
  UnAV-100: Geng et al., *Dense-Localizing Audio-Visual Events in Untrimmed
  Videos*, CVPR 2023 (CC BY-NC-SA 4.0).
- SigLIP 2: Tschannen et al., 2025. The weights are Google's, under the Apache
  2.0 licence. whenframe downloads the ONNX conversions published on the
  Hugging Face Hub by onnx-community (B/16 and B/32) and by Immich (So400m, for
  `--mode large`); these conversions state no licence of their own.
- CLAP (`--sounds`): Wu et al., *Large-scale Contrastive Language-Audio
  Pretraining*, ICASSP 2023; LAION's larger_clap_general weights (Apache 2.0),
  in Xenova's ONNX conversion.
- Whisper (`--speech`): Radford et al., *Robust Speech Recognition via
  Large-Scale Weak Supervision*, 2022 (OpenAI, MIT licence), run by
  faster-whisper (SYSTRAN, MIT) on CTranslate2.
- The query banks hold embeddings of DiDeMo training descriptions (BSD 2-Clause);
  see [whenframe/data/NOTICE](https://github.com/qthullie/whenframe/blob/main/whenframe/data/NOTICE).
- whenframe's own model was trained on DiDeMo's training videos (Flickr,
  Creative Commons licences, some non-commercial) and MSR-VTT's training clips
  (Xu et al., CVPR 2016; YouTube), both published for research; its weights
  are for research and non-commercial use (same NOTICE).
- The benchmark also measures MobileCLIP-S2 (Apple, research-only licence) and
  translates descriptions with Opus-MT (Helsinki-NLP, Apache 2.0); the tool uses
  neither.
- The demo uses Big Buck Bunny, Sintel and Tears of Steel, © Blender Foundation,
  CC BY 3.0.

## Licence

MIT, except the weights of whenframe's own model
(`whenframe/data/temporal-siglip2-base.npz`; `--no-temporal` does without
them): research and non-commercial use, as the videos it was trained on.
