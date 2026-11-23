"""The files of the demo site's library (site/library.json), from YFCC100M's public copy of the Flickr media
(multimedia-commons on AWS), as the site shows them:

    python scripts/site_media.py site/media [--originals DIR]

Videos: H.264 at most 640 pixels wide, without their sound or their metadata (where they were filmed). Photos: the
500-pixel copies of YFCC100M, turned upright and saved again without their metadata. --originals keeps the files as
downloaded there (scripts/site.py indexes those); otherwise they are downloaded into a temporary folder. The deploy
workflow (.github/workflows/pages.yml) runs this; the media stay out of git.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data"


def source(item: dict) -> str:
    h = item["hash"]
    if item["kind"] == "video":
        return f"{BUCKET}/videos/mp4/{h[:3]}/{h[3:6]}/{h}.mp4"
    return f"{BUCKET}/images/{h[:3]}/{h[3:6]}/{h}.jpg"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed host)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == 3:
                raise


def shown(item: dict, original: Path, dest: Path) -> None:
    """The file as the site shows it."""
    if dest.exists():
        return
    tmp = dest.with_name(dest.stem + ".part" + dest.suffix)
    if item["kind"] == "video":
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(original), "-an", "-map_metadata", "-1",
                        "-vf", "scale='min(640,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], check=True)  # fmt: skip
    else:
        from PIL import Image, ImageOps

        im = ImageOps.exif_transpose(Image.open(original)).convert("RGB")
        im.save(tmp, "JPEG", quality=85, optimize=True)  # no metadata kept
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="where the site's copies go (site/media)")
    ap.add_argument("--originals", help="keep the downloaded files there")
    a = ap.parse_args()
    items = json.loads((ROOT / "site" / "library.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        originals = Path(a.originals or tmp)
        originals.mkdir(parents=True, exist_ok=True)

        def one(item: dict) -> None:
            original = originals / item["file"]
            download(source(item), original)
            shown(item, original, out / item["file"])

        with cf.ThreadPoolExecutor(8) as ex:
            list(ex.map(one, items))
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"{len(items)} files in {out}, {size:.0f} MB")


if __name__ == "__main__":
    main()
