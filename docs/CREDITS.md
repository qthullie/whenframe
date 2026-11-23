# Credits

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
  `--mode so400m`); these conversions state no licence of their own.
- Perception Encoder's PE core B/16 (`--mode large`): Bolya et al., *Perception
  Encoder*, 2025. The weights are Meta's, under the Apache 2.0 licence.
  whenframe downloads onnx-community's conversion of its text tower, timm's
  tokenizer, and its image tower as whenframe's release models-1 publishes it,
  exported again from timm's weights, unchanged, with Meta's licence and a
  NOTICE of what was changed in how it is written
  ([models/pe-core-b16](https://github.com/qthullie/whenframe/tree/main/models/pe-core-b16)).
- CLAP (`--sounds`): Wu et al., *Large-scale Contrastive Language-Audio
  Pretraining*, ICASSP 2023; LAION's larger_clap_general weights (Apache 2.0),
  in Xenova's ONNX conversion.
- Whisper (`--speech`): Radford et al., *Robust Speech Recognition via
  Large-Scale Weak Supervision*, 2022 (OpenAI, MIT licence), run by
  faster-whisper (SYSTRAN, MIT) on CTranslate2.
- The query banks hold embeddings of DiDeMo training descriptions (BSD 2-Clause);
  see [whenframe/data/NOTICE](https://github.com/qthullie/whenframe/blob/main/whenframe/data/NOTICE).
- whenframe's own models (v3 on SigLIP 2 B/16's frames, v4 on PE core B/16's)
  were trained on DiDeMo's training videos (Flickr,
  Creative Commons licences, some non-commercial) with TEMPO-HL's sentences
  (Hendricks et al., EMNLP 2018, BSD 2-Clause), MSR-VTT's training clips (Xu
  et al., CVPR 2016; YouTube), Charades-STA's training videos (Charades,
  licensed for non-commercial research by the Allen Institute for AI) and
  YouCook2's training videos (YouTube), all published for research; their
  weights are for research and non-commercial use (same NOTICE).
- The benchmark also measures MobileCLIP-S2 (Apple, research-only licence),
  describes frames with Florence-2-base (Microsoft, MIT licence) and translates
  descriptions with Opus-MT (Helsinki-NLP, Apache 2.0); the tool uses none of
  them.
- The demo uses Big Buck Bunny, Sintel and Tears of Steel, © Blender Foundation,
  CC BY 3.0.

## Licence

MIT, except the weights of whenframe's own models
(`whenframe/data/temporal-siglip2-base.npz` and `temporal-pe-core-b16.npz`;
`--no-temporal` does without them): research and non-commercial use, as the
videos they were trained on.
