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
      <img src="https://raw.githubusercontent.com/qthullie/whenframe/main/docs/figures/site.webp" alt="The demo site: 'a girl hugging a baby dragon' finds three moments of Sintel, and the first, at 3:28, plays its ten seconds; then 'two people arguing on a bridge' finds the opening scene of Tears of Steel" width="800">
    </picture>
  </a>
  <br>
  <sub>Big Buck Bunny, Sintel and Tears of Steel (© Blender Foundation, CC BY 3.0), searched in the browser.</sub>
</p>

```bash
pip install git+https://github.com/qthullie/whenframe   # Python 3.10+ and ffmpeg
whenframe index ~/Videos
whenframe search "the dog jumps into the pool"
```

## Benchmark

<!-- table readme-compare -->
|  | whenframe, on a CPU | others |
|---|---|---|
| **Find the video** among 979 (DiDeMo, in the first 5) | **48.1 %** | 25.4 % EventFormer (2024), trained on DiDeMo, on a GPU · 30.7 % one image per video |
| **Find the moment** among 979 videos (DiDeMo, in the first 5) | **29.4 %** | 2.3 % one keyframe per shot, as home video tools index |
| **Place the moment** in a given video, never trained on it (Charades-STA, first answer) | **46.5 %** | 43.7 % VTG-GPT · 50.0 % TFVTG · 56.8 % a 7B vision-language model |
<!-- /table -->

Every number, how it was measured, and the protocol written before each test:
[results](https://github.com/qthullie/whenframe/blob/main/docs/RESULTS.md) ·
[protocol](https://github.com/qthullie/whenframe/blob/main/docs/PROTOCOL.md) ·
[tutorial](https://github.com/qthullie/whenframe/blob/main/docs/TUTORIAL.md).

MIT licence. The weights of whenframe's own model are for research and
non-commercial use ([credits](https://github.com/qthullie/whenframe/blob/main/docs/CREDITS.md)).
