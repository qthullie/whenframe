"""Links to Jellyfin, against a stand-in server answering as Jellyfin 10.11 does."""

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from whenframe.jellyfin import Jellyfin, path_pairs

ITEMS = [  # as Jellyfin 10.11.3 lists two home videos (trimmed)
    {"Name": "garden", "ServerId": "d35c4444", "Id": "4e600627", "Path": "/media/garden.mp4", "Type": "Video"},
    {"Name": "beach", "ServerId": "d35c4444", "Id": "f0c5f33e", "Path": "/media/holidays/beach.mp4", "Type": "Video"},
]


@pytest.fixture
def jellyfin():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            query = dict(urllib.parse.parse_qsl(url.query))
            seen.append((url.path, query, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != 'MediaBrowser Token="secret"':
                self.send_error(401)
                return
            start = int(query.get("StartIndex", 0))  # one item a page, to exercise the paging
            body = json.dumps({"Items": ITEMS[start : start + 1], "TotalRecordCount": len(ITEMS), "StartIndex": start})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()
    httpd.server_close()


def test_links_through_the_path_mapping(jellyfin):
    url, seen = jellyfin
    j = Jellyfin(url + "/", "secret", [("/srv/videos/", "/media")])
    assert j.link("/srv/videos/holidays/beach.mp4") == f"{url}/web/#/details?id=f0c5f33e&serverId=d35c4444"
    assert j.link("/srv/videos/garden.mp4").endswith("id=4e600627&serverId=d35c4444")
    assert j.link("/elsewhere/garden.mp4") is None
    paths = [p for p, _, _ in seen]
    assert paths == ["/Items", "/Items"]  # two pages, then the list is reused
    assert seen[0][1]["Fields"] == "Path" and seen[0][1]["Recursive"] == "true"


def test_a_wrong_key_or_no_server_gives_no_link(jellyfin, capsys):
    url, _ = jellyfin
    assert Jellyfin(url, "wrong").link("/media/garden.mp4") is None
    assert "no links to Jellyfin" in capsys.readouterr().err
    assert Jellyfin("http://127.0.0.1:9", "secret").link("/media/garden.mp4") is None


def test_path_pairs():
    assert path_pairs(["/srv/videos=/media", "/mnt/a=/b"]) == [("/srv/videos", "/media"), ("/mnt/a", "/b")]
    with pytest.raises(ValueError):
        path_pairs(["/srv/videos"])
