# whenframe

<img src="site/logo.svg" alt="" width="56" align="left">

Find a moment in your own videos by describing it. whenframe indexes a folder
of videos (and photos) on your computer, with no GPU and nothing sent anywhere,
and answers "the dog jumps into the pool" with the video and the seconds where
it happens.

**[See it at work in your browser](https://qthullie.github.io/whenframe/)** on three open movies, or install it below.

<p align="center">
  <a href="https://qthullie.github.io/whenframe/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/figures/site-dark.webp">
      <img src="docs/figures/site.webp" alt="The demo site: the three films; 'a girl hugging a baby dragon' finds three moments of Sintel, and the first, at 3:28, plays its ten seconds; then 'two people arguing on a bridge' finds the opening scene of Tears of Steel" width="800">
    </picture>
  </a>
  <br>
  <sub>The demo site, searching Big Buck Bunny, Sintel and Tears of Steel (© Blender Foundation, CC BY 3.0) in the browser.</sub>
</p>

<!-- table readme-headline -->
Searching 979 personal videos at once (DiDeMo test) for 3,902 descriptions written by people, whenframe puts the right moment among its first five results **27.2 %** of the time. Indexing one keyframe per shot, as some home video tools do, finds it 2.3 % of the time, and the best of the two baselines (fixed windows) 25.0 %.
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
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/demo-1-dark.webp">
    <img src="docs/figures/demo-1.webp" alt="The local search page of whenframe serve: 'two people arguing on a bridge' finds the opening scene of Tears of Steel first, among three films" width="800">
  </picture>
</p>

The first run downloads the SigLIP 2 model (1.5 GB). A library is indexed in
one of three modes, chosen the first time (`--mode`):

<!-- table readme-modes -->
| `--mode` | right moment in the first five (DiDeMo) | indexing, CPU min per hour of video | search (ms) |
|---|---|---|---|
| `standard` | 27.2 % | 24.9 | 63 |
| `economy` | 24.5 % | 8.4 | 61 |
| `precision` | 27.2 % | 33.2 | 123 |
<!-- /table -->

<!-- table readme-openvino -->
With OpenVINO (`pip install 'whenframe[fast]'`) on a processor that computes in bfloat16, as this one does, the standard mode indexes for 6.5 CPU minutes per hour of video instead of 24.9, and finds 27.5 % of the moments instead of 27.2 % (DiDeMo test).
<!-- /table -->

Photos in the same folders are indexed too, each as a moment of its own, so a
search returns photos and moments of videos together (`--no-photos` to leave
them out). `whenframe index --speech base` also transcribes what is said (Whisper, on the
CPU, `pip install 'whenframe[speech]'`), and a search then finds the words
said as well as what is seen. `serve --jellyfin` links each moment to the
video in a Jellyfin server.

Nothing else goes over the network. ONNX Runtime 1.29 sends usage telemetry to
Microsoft by default, and `import openvino` reports to Google Analytics;
whenframe turns off the first and keeps the second from loading (checked by
logging every outbound connection of an indexing and search run). More in
[docs/TUTORIAL.md](docs/TUTORIAL.md): watching folders, Jellyfin, Docker.

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
   the best frames, and all moments are ranked together. In precision mode the
   scores of two models are standardised and added first.

## Results

DiDeMo test split, with the protocol fixed in advance and every change to it
listed ([docs/PROTOCOL.md](docs/PROTOCOL.md)). The numbers on this page are
written by `scripts/tables.py` from `results/*.json`.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/results-dark.png">
    <img src="docs/figures/results.png" alt="Share of DiDeMo test descriptions whose moment is among the first five results, for each search method of the first round" width="800">
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

On Charades-STA, with the parameters chosen on DiDeMo and nothing tuned, it
places the moment in a given video 45.7 % of the time at IoU >= 0.5; a
7-billion-parameter vision-language model, run over the video's frames for each
sentence, is reported at 56.8 %.

<!-- table readme-youcook2 -->
On 252 YouCook2 cooking videos (5 minutes on average, searched all at once), the right step is among the first five moments 12.3 % of the time, and 18.8 % when what is said is searched too (`--speech`).
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

All tables, the costs and the figures: [docs/RESULTS.md](docs/RESULTS.md).

## Limits

- It sees frames, not motion: "the camera pans left" is out of its reach. Of
  the sound it only hears speech, with `--speech`: "someone laughs" is not
  found.
- It was measured on short personal videos (DiDeMo, their first 30 seconds) and
  on cooking videos of about 5 minutes (YouCook2); recordings of an hour or more
  were not measured.
- On YouCook2's long recipe steps, adaptive spans did better than the default
  fixed windows, which were chosen on DiDeMo and stay the default.
- It cannot tell apart videos that show the same thing: among Charades'
  1,334 look-alike videos of people at home, the right one is in the first five
  results 4 % of the time.
- Indexing is slow on a small machine without OpenVINO (see the modes above,
  measured on 4 vCPUs); OpenVINO's speed-up was measured on one processor only.
- Descriptions in French were measured through machine translation only.
- Several criteria set in advance were not met; the results page says which and
  why.

## Documents

- [docs/RESEARCH.md](docs/RESEARCH.md): what already exists, and what is new here
- [docs/PROTOCOL.md](docs/PROTOCOL.md): the evaluation, written before each test run, and every change made to it
- [docs/RESULTS.md](docs/RESULTS.md): all the numbers
- [docs/TUTORIAL.md](docs/TUTORIAL.md): installing, indexing, searching, the page, reproducing the benchmark

## Credits

- DiDeMo: Hendricks et al., *Localizing Moments in Video with Natural
  Language*, ICCV 2017. Charades-STA: Gao et al., *TALL*, ICCV 2017, on
  Charades: Sigurdsson et al., ECCV 2016.
- SigLIP 2: Tschannen et al., 2025. The weights, downloaded at the first run,
  are Google's, under the Apache 2.0 licence.
- The query bank holds embeddings of DiDeMo training descriptions (BSD 2-Clause);
  see [whenframe/data/NOTICE](whenframe/data/NOTICE).
- The benchmark also measures MobileCLIP-S2 (Apple, research-only licence) and
  translates descriptions with Opus-MT (Helsinki-NLP, Apache 2.0); the tool uses
  neither.
- The demo uses Big Buck Bunny, Sintel and Tears of Steel, © Blender Foundation,
  CC BY 3.0.

## Licence

MIT.
