"""Links from whenframe's results to the same videos in a Jellyfin server.

`whenframe serve --jellyfin URL`, with an API key (Jellyfin: Dashboard, API Keys)
in $JELLYFIN_API_KEY, adds an "Open in Jellyfin" link to each moment whose video
Jellyfin also has. Jellyfin often sees the files under another path, inside its
container: `--jellyfin-path /srv/videos=/media` says that /srv/videos/a.mp4 here
is /media/a.mp4 there.

The list of Jellyfin's videos is fetched at the first search and again every ten
minutes, or a minute after a search finds a video Jellyfin did not list yet.
Written and tested against Jellyfin 10.11.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.parse
import urllib.request

KINDS = "Movie,Episode,Video,MusicVideo"


class Jellyfin:
    name = "Jellyfin"

    def __init__(self, url: str, key: str, paths: list[tuple[str, str]] | None = None, refresh: float = 600.0):
        self.url = url.rstrip("/")
        self.key = key
        self.paths = [(here.rstrip("/"), there.rstrip("/")) for here, there in paths or []]
        self.refresh = refresh
        self._ids: dict[str, str] = {}
        self._server = ""
        self._loaded = float("-inf")
        self._warned = False
        self._lock = threading.Lock()

    def _get(self, endpoint: str, **params) -> dict:
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{self.url}{endpoint}?{query}",
                                     headers={"Authorization": f'MediaBrowser Token="{self.key}"'})  # fmt: skip
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (the user's own server)
            return json.load(r)

    def _load(self) -> None:
        ids, server, start = {}, "", 0
        while True:
            page = self._get("/Items", Recursive="true", IncludeItemTypes=KINDS, Fields="Path", EnableImages="false",
                             EnableUserData="false", StartIndex=start, Limit=1000)  # fmt: skip
            items = page.get("Items", [])
            for item in items:
                if item.get("Path"):
                    ids[item["Path"]] = item["Id"]
                    server = item.get("ServerId", server)
            start += len(items)
            if not items or start >= page.get("TotalRecordCount", 0):
                break
        self._ids, self._server, self._loaded = ids, server, time.monotonic()

    def their_path(self, path: str) -> str:
        """Where Jellyfin sees a file whenframe indexed."""
        for here, there in self.paths:
            if path == here or path.startswith(here + "/"):
                return there + path[len(here) :]
        return path

    def link(self, path: str) -> str | None:
        """The page of this video in Jellyfin's web client, if Jellyfin has it."""
        theirs = self.their_path(path)
        with self._lock:
            age = time.monotonic() - self._loaded
            if age > self.refresh or (theirs not in self._ids and age > 60):
                try:
                    self._load()
                except (OSError, ValueError) as e:  # unreachable or wrong key: no links, and say it once
                    self._loaded = time.monotonic()
                    if not self._warned:
                        print(f"whenframe: no links to Jellyfin at {self.url}: {e}", file=sys.stderr, flush=True)
                        self._warned = True
            item = self._ids.get(theirs)
        if not item:
            return None
        return f"{self.url}/web/#/details?id={item}" + (f"&serverId={self._server}" if self._server else "")


def path_pairs(specs: list[str] | None) -> list[tuple[str, str]]:
    """["/srv/videos=/media", ...] -> [("/srv/videos", "/media"), ...]"""
    pairs = []
    for spec in specs or []:
        here, sep, there = spec.partition("=")
        if not sep or not here or not there:
            raise ValueError(f"expected HERE=THERE, got {spec!r}")
        pairs.append((here, there))
    return pairs
