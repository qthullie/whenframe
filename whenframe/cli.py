"""whenframe index | search | serve | stats | compact"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .encoder import MODES
from .library import Library, Settings
from .search import SearchConfig, Searcher
from .video import find_videos


def default_library() -> Path:
    return Path(os.environ.get("WHENFRAME_LIBRARY") or Path.home() / ".local" / "share" / "whenframe" / "library")


def clock(t: float) -> str:
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def cmd_index(args) -> int:
    overrides = {k: v for k, v in (("fps", args.fps), ("dedup", args.dedup)) if v is not None}
    if args.mode:
        overrides["model"] = MODES[args.mode]
    lib = Library(args.library, Settings(**overrides) if overrides else None)
    paths = find_videos(args.paths)
    todo = [p for p in paths if lib.needs_indexing(p.resolve())]
    print(f"{len(paths)} videos found, {len(todo)} to index")
    t0 = time.time()
    failed = 0

    def report(path, video, err):
        nonlocal failed
        if err:
            failed += 1
            print(f"  skipped {path}: {err}", file=sys.stderr)
        elif video:
            print(f"  {path.name}  {clock(video.duration)}  {video.count} frames")

    lib.add_many(todo, report)
    print(f"done in {time.time() - t0:.0f} s" + (f", {failed} failed" if failed else ""))
    return 1 if failed and failed == len(todo) else 0


def cmd_search(args) -> int:
    lib = Library(args.library)
    hits = Searcher(lib, SearchConfig()).search(" ".join(args.query), k=args.k)
    if not hits:
        print("no video indexed yet: run `whenframe index <folder>` first")
        return 1
    for h in hits:
        print(f"{h.confidence:5.0%}  {clock(h.start):>7} - {clock(h.end):<7}  {h.path}")
    return 0


def cmd_stats(args) -> int:
    s = Library(args.library).stats()
    size = s["index_bytes"] / 1e6
    print(f"{s['videos']} videos, {s['hours']:.1f} h, {s['embeddings']} frames indexed, {size:.1f} MB on disk")
    print(f"settings: {s['settings']}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(Library(args.library), args.host, args.port)
    return 0


def cmd_compact(args) -> int:
    Library(args.library).compact()
    return cmd_stats(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="whenframe", description="Find the moment you are looking for in your videos.")
    ap.add_argument("--library", type=Path, default=default_library(), help="where the index lives")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="index the videos in folders or files (only new or changed ones)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--fps", type=float, default=None, help="frames looked at per second (default 1)")
    p.add_argument("--dedup", type=float, default=None, help="skip frames that barely changed (default 4, 0 = off)")
    p.add_argument("--mode", choices=list(MODES), default=None,
                   help="standard (default); economy: about 2.3 times less CPU, a little less accurate; "
                        "precision: both models, more accurate, about 40 %% slower. Fixed for a library.")  # fmt: skip
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("search", help="find moments matching a description")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=10, help="number of moments to show")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("serve", help="a search page in the browser, with the model kept loaded")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)
    sub.add_parser("stats", help="what the library contains").set_defaults(fn=cmd_stats)
    sub.add_parser("compact", help="reclaim space left by re-indexed or removed videos").set_defaults(fn=cmd_compact)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
