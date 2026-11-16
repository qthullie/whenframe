"""Keeping a library in step with folders: new and changed videos are indexed, removed ones forgotten.

`whenframe watch` runs a pass every few minutes; `whenframe serve --watch` runs the
same passes in the background while it serves the page. Folders are scanned rather
than followed with file-system events, which also works on network shares.

Two cases are left alone on purpose. A file modified less than `settle` seconds ago
is left for the next pass, so that a video still being copied is not indexed half
written. And when every video indexed under a folder has gone, the folder is taken
for an unmounted disk or share, and nothing under it is forgotten.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .library import Library
from .video import find_videos

SETTLE = 60.0  # seconds a file has to stay unchanged before it is indexed
COMPACT_ABOVE = 0.25  # share of dead rows in the index files that triggers a compaction


@dataclass
class Pass:
    indexed: list[Path] = field(default_factory=list)
    forgotten: list[Path] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)
    waiting: list[Path] = field(default_factory=list)  # modified too recently
    unavailable: list[Path] = field(default_factory=list)  # folders missing, or looking unmounted


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class Watcher:
    def __init__(self, library: Library, folders: Iterable[str | Path], settle: float = SETTLE,
                 report: Callable[[str, Path, str], None] | None = None):  # fmt: skip
        self.library = library
        self.roots = [Path(f).resolve() for f in folders]
        self.settle = settle
        self.report = report or (lambda event, path, detail: None)
        self._failed: dict[Path, tuple[int, float]] = {}  # tried and failed, as (size, mtime): retried once changed

    def sync(self) -> Pass:
        """One pass over the folders."""
        result = Pass()
        indexed = [Path(v.path) for v in self.library.videos()]
        for root in self.roots:
            mine = [p for p in indexed if _inside(p, root)]
            gone = [p for p in mine if not p.exists()]
            if not root.is_dir() or (mine and len(gone) == len(mine)):
                result.unavailable.append(root)
                self.report("unavailable", root, "skipped: missing, or every video in it is gone")
                continue
            for p in gone:
                self.library.remove(p)
                result.forgotten.append(p)
                self.report("forgotten", p, "")
            now = time.time()
            for p in find_videos([root]):
                p = p.resolve()
                try:
                    if not self.library.needs_indexing(p):
                        continue
                    st = p.stat()
                except OSError:  # gone since the folder was listed
                    continue
                if now - st.st_mtime < self.settle:
                    result.waiting.append(p)
                    continue
                if self._failed.get(p) == (st.st_size, st.st_mtime):
                    continue
                self.library.add_many([p], lambda path, video, err, st=st: self._done(result, path, video, err, st))
        self.library.compact_if_needed(COMPACT_ABOVE)
        return result

    def _done(self, result: Pass, path: Path, video, err, st) -> None:
        if err:
            self._failed[path] = (st.st_size, st.st_mtime)
            result.failed.append(path)
            self.report("failed", path, str(err))
        elif video:
            self._failed.pop(path, None)
            result.indexed.append(path)
            self.report("indexed", path, f"{video.count} frames")

    def run(self, every: float = 300.0, stop: threading.Event | None = None) -> None:
        """Passes every `every` seconds until `stop` is set (or forever)."""
        stop = stop or threading.Event()
        while True:
            self.sync()
            if stop.wait(every):
                return
