"""The training data's readers, on a local server: a connection that ends early is read again where it stopped,
and a reading that fails makes its stage fail instead of assembling a part of the videos."""

import http.server
import io
import socket
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import moments_data  # noqa: E402
from train_data import Archive  # noqa: E402

PARTS = {"a.part": bytes(range(256)) * 40, "b.part": bytes(reversed(range(256))) * 30}
WHOLE = b"".join(PARTS.values())


class Parts(http.server.BaseHTTPRequestHandler):
    """PARTS by byte ranges. The first `early` responses stop halfway through their body, the connection closed;
    the first `reset` ones, the connection reset."""

    early = reset = 0

    def do_GET(self):  # noqa: N802 - http.server's name
        start = int(self.headers["Range"].removeprefix("bytes=").split("-")[0])
        body = PARTS[self.path.lstrip("/")][start:]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).reset > 0:
            type(self).reset -= 1
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()  # a reset: the reader's next read raises
            return
        if type(self).early > 0:
            type(self).early -= 1
            body = body[: len(body) // 2]
        self.wfile.write(body)

    def finish(self):
        try:
            super().finish()
        except OSError:  # the connection reset above
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Parts)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


def archive(url: str, start: int = 0, retries: int = 8) -> io.BufferedReader:
    parts = [(name, len(data)) for name, data in PARTS.items()]
    return io.BufferedReader(Archive(start, url=url, parts=parts, retries=retries, pause=0), buffer_size=1000)


def test_the_archive_is_its_parts_one_after_the_other(url):
    Parts.early = Parts.reset = 0
    assert archive(url).read() == WHOLE
    start = len(PARTS["a.part"]) - 100  # from a byte of the whole, across the parts' boundary
    assert archive(url, start).read() == WHOLE[start:]


def test_a_connection_that_ends_early_is_read_again_where_it_stopped(url):
    Parts.early, Parts.reset = 3, 0  # http.client reads 0 bytes there, as at the end of a part
    assert archive(url).read() == WHOLE
    assert Parts.early == 0


def test_a_connection_reset_is_read_again_where_it_stopped(url):
    Parts.early, Parts.reset = 0, 3
    assert archive(url).read() == WHOLE
    assert Parts.reset == 0


def test_the_archive_gives_up_after_its_retries_without_a_byte(url):
    Parts.early, Parts.reset = 10**6, 0  # each response half of what is left: down to nothing
    with pytest.raises(OSError, match="the archive stopped at byte"):
        archive(url, retries=2).read()


def test_a_reading_that_fails_fails_the_stage_once_the_videos_read_are_handled(tmp_path, monkeypatch):
    monkeypatch.setattr(moments_data, "probe", lambda path: SimpleNamespace(duration=3.0))
    monkeypatch.setattr(moments_data, "sample_frames", lambda path, fps, size, seconds=None: (
        np.arange(3.0), np.zeros((3, 4, 4, 3), np.uint8)))  # fmt: skip

    def videos():
        for i in range(4):
            path = tmp_path / f"v{i}.mp4"
            path.write_bytes(b"")
            yield f"v{i}", path
        raise OSError("the archive stopped")

    handled = []
    with pytest.raises(RuntimeError, match="after 4 this run"):
        moments_data.pipeline(videos(), lambda name, t, frames, duration: handled.append(name))
    assert sorted(handled) == ["v0", "v1", "v2", "v3"]
    assert not list(tmp_path.iterdir())  # each video deleted once decoded
