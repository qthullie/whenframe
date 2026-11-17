"""Links from whenframe's results to the same videos and photos in an Immich server.

`whenframe serve --immich URL`, with an API key (Immich: Account Settings, API
Keys, with the asset.read permission) in $IMMICH_API_KEY, adds an "Open in
Immich" link to each moment whose file Immich also has, in an external library
or its own uploads. Immich often sees the files under another path, inside its
container: `--immich-path /srv/videos=/mnt/media` says that /srv/videos/a.mp4
here is /mnt/media/a.mp4 there.

Each file is looked up once, by its path (POST /api/search/metadata with
originalPath, which Immich matches as a part of the path, so the answer is
checked for the exact one); a file Immich did not have is asked about again
after ten minutes. Written from Immich's API documentation (version 1.140) and
tested against a stand-in server that answers as it documents, not against a
running Immich.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request


class Immich:
    name = "Immich"

    def __init__(self, url: str, key: str, paths: list[tuple[str, str]] | None = None, retry: float = 600.0):
        self.url = url.rstrip("/")
        self.key = key
        self.paths = [(here.rstrip("/"), there.rstrip("/")) for here, there in paths or []]
        self.retry = retry
        self._ids: dict[str, tuple[str | None, float]] = {}  # their path -> (asset id or None, when asked)
        self._warned = False
        self._lock = threading.Lock()

    def _post(self, endpoint: str, body: dict) -> dict:
        req = urllib.request.Request(f"{self.url}{endpoint}", data=json.dumps(body).encode(), method="POST",
                                     headers={"x-api-key": self.key, "Content-Type": "application/json",
                                              "Accept": "application/json"})  # fmt: skip
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (the user's own server)
            return json.load(r)

    def their_path(self, path: str) -> str:
        """Where Immich sees a file whenframe indexed."""
        for here, there in self.paths:
            if path == here or path.startswith(here + "/"):
                return there + path[len(here) :]
        return path

    def _find(self, theirs: str) -> str | None:
        page = self._post("/api/search/metadata", {"originalPath": theirs, "size": 50, "withDeleted": False})
        items = (page.get("assets") or {}).get("items") or []
        return next((a["id"] for a in items if a.get("originalPath") == theirs), None)

    def link(self, path: str) -> str | None:
        """The page of this video or photo in Immich's web client, if Immich has it."""
        theirs = self.their_path(path)
        with self._lock:
            found, when = self._ids.get(theirs, (None, float("-inf")))
            if found is None and time.monotonic() - when > self.retry:
                try:
                    found = self._find(theirs)
                except (OSError, ValueError, KeyError) as e:  # unreachable or wrong key: no links, and say it once
                    if not self._warned:
                        print(f"whenframe: no links to Immich at {self.url}: {e}", file=sys.stderr, flush=True)
                        self._warned = True
                self._ids[theirs] = (found, time.monotonic())
        return f"{self.url}/photos/{found}" if found else None
