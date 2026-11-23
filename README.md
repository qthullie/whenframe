<p align="center">
  <img src="https://raw.githubusercontent.com/qthullie/whenframe/main/site/logo.svg" alt="" width="72">
</p>

<h1 align="center">whenframe</h1>

<p align="center">
  <b>Describe a moment, find it in your videos.</b><br>
  On your own computer: no GPU, nothing sent anywhere.<br>
  <a href="https://qthullie.github.io/whenframe/"><b>Try it in your browser</b></a>
</p>

<p align="center">
  <a href="https://qthullie.github.io/whenframe/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/site-dark.webp">
      <img src="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/site.webp" alt="The demo site: in someone's phone, 67 videos and 165 photos named VID_ and IMG_, 'a geyser shooting steam into the sky' finds the eruption in two videos, and the first plays its ten seconds; then 'a pipe band marching down the street' finds the parade, and a photo of it" width="800">
    </picture>
  </a>
  <br>
  <sub>Real videos and photos, shared on Flickr under CC BY 2.0 (YFCC100M; <a href="https://qthullie.github.io/whenframe/credits.html">credits</a>), searched in the browser.</sub>
</p>

```bash
pip install git+https://github.com/qthullie/whenframe   # Python 3.10+ and ffmpeg
whenframe index ~/Videos
whenframe search "the dog jumps into the pool"
whenframe search "le chien saute dans la piscine"   # French is translated on your computer
```

## Benchmark

<!-- table readme-compare -->
|  | whenframe, on a CPU | others |
|---|---|---|
| **Find the video** among 979 (DiDeMo, in the first 5) | **51.1 %** | 25.4 % EventFormer (2024), trained on DiDeMo, on a GPU · 30.7 % one image per video |
| **Find the moment** among 979 videos (DiDeMo, in the first 5) | **32.0 %** | 2.3 % one keyframe per shot, as home video tools index |
| **Place the moment** in a given video, never trained on it (Charades-STA, first answer) | **49.9 %** | 43.7 % VTG-GPT · 50.0 % TFVTG · 56.8 % a 7B vision-language model |
<!-- /table -->

Every number, how it was measured, and the protocol written before each test:
[results](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md) ·
[protocol](https://github.com/qthullie/whenframe/blob/main/docs/PROTOCOL.md) ·
[tutorial](https://github.com/qthullie/whenframe/blob/main/docs/TUTORIAL.md).

MIT licence. The weights of whenframe's own models are for research and
non-commercial use ([credits](https://github.com/qthullie/whenframe/blob/main/docs/CREDITS.md)).
