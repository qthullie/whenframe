"""whenframe index | watch | search | serve | stats | compact"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .encoder import BACKENDS, MODES
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


def open_library(args) -> Library:
    """The library, created with the indexing options the first time."""
    overrides = {k: v for k, v in (("fps", args.fps), ("dedup", args.dedup)) if v is not None}
    if args.mode:
        overrides["model"] = MODES[args.mode]
    lib = Library(args.library, Settings(**overrides) if overrides else None, backend=args.backend)
    if args.speech:
        lib.enable_speech(args.speech)
    return lib


def cmd_index(args) -> int:
    lib = open_library(args)
    paths = find_videos(args.paths)
    todo = [p for p in paths if lib.needs_indexing(p.resolve()) or lib.needs_transcript(p.resolve())]
    print(f"{len(paths)} videos found, {len(todo)} to index")
    if todo:
        enc = lib.encoder
        print(f"image encoder: {enc.name} on {enc.backend} ({enc.images.precision})")
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
        if h.said:
            print(f"{'':16}said: “{h.said}”")
    return 0


def cmd_stats(args) -> int:
    s = Library(args.library).stats()
    size = s["index_bytes"] / 1e6
    print(f"{s['videos']} videos, {s['hours']:.1f} h, {s['embeddings']} frames indexed, {size:.1f} MB on disk")
    print(f"settings: {s['settings']}")
    return 0


def print_event(event: str, path: Path, detail: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {event:<11} {path}" + (f"  ({detail})" if detail else ""), flush=True)


def cmd_watch(args) -> int:
    from .watch import Watcher

    watcher = Watcher(open_library(args), args.folders, settle=args.settle, report=print_event)
    if args.once:
        watcher.sync()
        return 0
    print(f"watching {', '.join(map(str, watcher.roots))} every {args.every:g} s (Ctrl-C to stop)", flush=True)
    try:
        watcher.run(args.every)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    lib = Library(args.library)
    if args.watch:
        import threading

        from .watch import Watcher

        watcher = Watcher(lib, args.watch, settle=args.settle, report=print_event)
        threading.Thread(target=watcher.run, args=(args.every,), daemon=True).start()
    links = []
    if args.jellyfin:
        from .jellyfin import Jellyfin, path_pairs

        key = os.environ.get("JELLYFIN_API_KEY")
        if not key:
            print("--jellyfin needs an API key in $JELLYFIN_API_KEY (Jellyfin: Dashboard, API Keys)", file=sys.stderr)
            return 2
        links.append(Jellyfin(args.jellyfin, key, path_pairs(args.jellyfin_path)))
    serve(lib, args.host, args.port, links=links)
    return 0


def cmd_compact(args) -> int:
    Library(args.library).compact()
    return cmd_stats(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="whenframe", description="Find the moment you are looking for in your videos.")
    ap.add_argument("--library", type=Path, default=default_library(), help="where the index lives")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def indexing_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--fps", type=float, default=None, help="frames looked at per second (default 1)")
        p.add_argument("--dedup", type=float, default=None, help="skip frames that barely changed (default 4, 0 = off)")
        p.add_argument("--mode", choices=list(MODES), default=None,
                       help="standard (default); economy: about 3 times less CPU, less accurate; precision: two "
                            "models, no gain measured on DiDeMo. Fixed for a library (docs/RESULTS.md).")  # fmt: skip
        p.add_argument("--backend", choices=BACKENDS, default=None,
                       help="runtime of the image encoder: OpenVINO if installed (auto, the default), else ONNX "
                            "Runtime; also $WHENFRAME_BACKEND")  # fmt: skip
        p.add_argument("--speech", choices=["base", "small"], default=None,
                       help="also transcribe what is said, with this Whisper model (pip install whenframe[speech]); "
                            "kept for the library from then on")  # fmt: skip

    def watching_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--every", type=float, default=300.0, help="seconds between two passes (default 300)")
        p.add_argument("--settle", type=float, default=60.0,
                       help="leave files modified in the last N seconds for the next pass (default 60)")  # fmt: skip

    p = sub.add_parser("index", help="index the videos in folders or files (only new or changed ones)")
    p.add_argument("paths", nargs="+")
    indexing_options(p)
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("watch", help="keep the library in step with folders: index new videos, forget removed ones")
    p.add_argument("folders", nargs="+")
    indexing_options(p)
    watching_options(p)
    p.add_argument("--once", action="store_true", help="one pass, then exit (for cron or a systemd timer)")
    p.set_defaults(fn=cmd_watch)
    p = sub.add_parser("search", help="find moments matching a description")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=10, help="number of moments to show")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("serve", help="a search page in the browser, with the model kept loaded")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--watch", nargs="+", metavar="FOLDER", help="also keep the library in step with these folders")
    watching_options(p)
    p.add_argument("--jellyfin", metavar="URL", help="link each moment to its video in this Jellyfin server "
                   "(API key in $JELLYFIN_API_KEY)")  # fmt: skip
    p.add_argument("--jellyfin-path", action="append", metavar="HERE=THERE",
                   help="a folder as whenframe sees it, and as Jellyfin does (repeatable)")  # fmt: skip
    p.set_defaults(fn=cmd_serve)
    sub.add_parser("stats", help="what the library contains").set_defaults(fn=cmd_stats)
    sub.add_parser("compact", help="reclaim space left by re-indexed or removed videos").set_defaults(fn=cmd_compact)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
