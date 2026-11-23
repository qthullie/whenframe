---
title: whenframe demo text server
sdk: docker
app_port: 7860
license: mit
short_description: Embeds the descriptions typed on whenframe's demo site
---

# The demo site's text server

The [demo site](https://qthullie.github.io/whenframe/) searches a phone's videos
and photos in the browser, on frames whenframe embedded beforehand. A description
typed there is sent to this server, embedded by SigLIP 2's and PE core B/16's text
towers exactly as `whenframe search` embeds a query, and sent back: the browser
searches with it.

- Only the description comes here. Nothing is logged or stored.
- `GET /embed?q=TEXT` answers `{"vectors": {...}, "dims": {...}, "ms": ...}`:
  each model's unit vector as little-endian float32, in base64. `GET /health`
  answers `ok`.
- Only the demo site may read the answers from a browser (`--origin`, or
  `WHENFRAME_ORIGINS`, comma-separated). Descriptions are limited to 200
  characters, each address to 30 a minute.

It needs about 4 GB of memory and no GPU. The text towers (1.1 and 0.7 GB) are
downloaded when the image is built.

```bash
docker build -t whenframe-demo demo-server
docker run -p 7860:7860 whenframe-demo
```

Or without Docker, from the repository: `python demo-server/app.py --origin
http://127.0.0.1:8000`, then serve `site/` at that address and open
`http://127.0.0.1:8000/?server=http://127.0.0.1:7860`.

Once it runs at a public address, `site/config.json`'s `text_server` tells the
site where it is.
