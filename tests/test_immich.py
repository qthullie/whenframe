"""Links to Immich, against a stand-in server answering POST /api/search/metadata as Immich documents it."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from whenframe.immich import Immich

ASSETS = [  # Immich matches originalPath as a part of the path: both come back for "/mnt/media/a.mp4"
    {"id": "1f1a2b3c", "originalPath": "/mnt/media/a.mp4", "type": "VIDEO"},
    {"id": "9e8d7c6b", "originalPath": "/mnt/media/a.mp4.old/a.mp4", "type": "VIDEO"},
]


@pytest.fixture
def immich():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, self.headers.get("x-api-key")))
            if self.headers.get("x-api-key") != "secret":
                self.send_error(401)
                return
            items = [a for a in ASSETS if body["originalPath"].lower() in a["originalPath"].lower()]
            out = json.dumps(
                {
                    "albums": {"items": [], "total": 0},
                    "assets": {"items": items, "total": len(items), "count": len(items), "nextPage": None},
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()
    httpd.server_close()


def test_a_file_immich_has_gets_its_page(immich):
    url, seen = immich
    links = Immich(url + "/", "secret", [("/srv/videos", "/mnt/media")])
    assert links.link("/srv/videos/a.mp4") == f"{url}/photos/1f1a2b3c"  # the exact path, not the other match
    assert links.link("/srv/videos/a.mp4") == f"{url}/photos/1f1a2b3c"
    assert len(seen) == 1 and seen[0][0] == "/api/search/metadata"  # asked once
    assert seen[0][1]["originalPath"] == "/mnt/media/a.mp4"
    assert links.link("/srv/videos/b.mp4") is None
    assert links.link("/srv/videos/b.mp4") is None and len(seen) == 2  # not asked again before ten minutes


def test_a_wrong_key_gives_no_links(immich, capsys):
    url, _ = immich
    links = Immich(url, "wrong")
    assert links.link("/mnt/media/a.mp4") is None
    assert "no links to Immich" in capsys.readouterr().err
