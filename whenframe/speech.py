"""What is said in the videos: transcribed on the CPU, then searched by its words.

Optional: `pip install whenframe[speech]`, then `whenframe index --speech base`.
Whisper runs through faster-whisper (CTranslate2, int8 weights) on audio that
ffmpeg decodes at 16 kHz, ten minutes at a time. Each segment of speech and its
times go into an SQLite FTS5 table next to the videos.

A search matches the description's words against the segments with BM25, and a
frame's score becomes its image score plus `weight` times the best score of the
segments said around it (from `pad` seconds before it starts to `pad` seconds
after it ends). The weight and the pad were chosen on YouCook2 tuning videos
(docs/PROTOCOL.md, third round).
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from dataclasses import dataclass

import numpy as np

from .encoder import fetch_files

WHISPER = {"base": "Systran/faster-whisper-base", "small": "Systran/faster-whisper-small"}
WHISPER_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
CHUNK = 600.0  # seconds of audio transcribed at a time
RATE = 16000

SCHEMA = """
create table if not exists speech (
    id integer primary key,
    video_id integer not null,
    start real not null,
    end real not null,
    text text not null
);
create index if not exists speech_video on speech(video_id);
create virtual table if not exists speech_fts using fts5(
    text, content='speech', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2'
);
create trigger if not exists speech_added after insert on speech begin
    insert into speech_fts(rowid, text) values (new.id, new.text);
end;
create trigger if not exists speech_removed after delete on speech begin
    insert into speech_fts(speech_fts, rowid, text) values ('delete', old.id, old.text);
end;
create table if not exists transcribed (video_id integer primary key, model text not null, at real not null);
"""

# words that say nothing about a moment, in the two languages the tool is used in most
STOP = set(
    """
a an and are as at be been but by can could did do does for from had has have he her here him his how i if in into is
it its just me my no not now of on or our out over she so some that the their them then there these they this those to
too up us was we were what when where which while who why will with would you your
au aux avec ce ces cet cette dans de des du elle en est et eux il ils je la le les leur lui ma mais me mes moi mon ne
nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y
""".split()
)
WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SpeechConfig:
    # chosen on the YouCook2 tuning videos among 24 combinations (results/youcook2_tuned.json)
    weight: float = 0.005
    pad: float = 0.0  # seconds
    normalise: str = "raw"  # "raw": BM25 as it is; "best": divided by the query's best segment


def audio(path: str, start: float, seconds: float) -> np.ndarray:
    """The sound of `seconds` from `start`, mono, 16 kHz, float32; empty when there is none."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(run.stdout, dtype=np.float32) if run.returncode == 0 else np.zeros(0, np.float32)


class Transcriber:
    def __init__(self, size: str = "base", threads: int | None = None, folder: str | None = None):
        from faster_whisper import WhisperModel  # optional dependency: whenframe[speech]

        self.size = size
        folder = folder or str(fetch_files(WHISPER[size], WHISPER_FILES))
        self.model = WhisperModel(folder, device="cpu", compute_type="int8", cpu_threads=threads or 0)

    def __call__(self, path: str, duration: float) -> list[tuple[float, float, str]]:
        """(start, end, text) of every stretch of speech in the video."""
        out, t = [], 0.0
        while not duration or t < duration:
            sound = audio(path, t, CHUNK)
            if len(sound) < RATE // 10:  # no sound track, or its end
                break
            segments, _ = self.model.transcribe(sound, vad_filter=True, beam_size=1)
            out += [(t + s.start, t + s.end, s.text.strip()) for s in segments if s.text.strip()]
            t += CHUNK
        return out


def match_expression(query: str) -> str | None:
    """The FTS5 query for a description: its words but the stop words, any of them."""
    words = [w for w in dict.fromkeys(WORD.findall(query.lower())) if w not in STOP and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words) or None


def segment_scores(db: sqlite3.Connection, query: str) -> list[tuple[int, float, float, float, str]]:
    """(video_id, start, end, score, text) of the segments matching the query; higher scores match better."""
    expr = match_expression(query)
    if expr is None:
        return []
    rows = db.execute(
        "select s.video_id, s.start, s.end, -bm25(speech_fts), s.text from speech_fts"
        " join speech s on s.id = speech_fts.rowid where speech_fts match ?",
        (expr,),
    ).fetchall()
    return [(int(v), float(a), float(b), float(score), text) for v, a, b, score, text in rows]


def frame_scores(times: np.ndarray, ranges: dict[int, tuple[int, int]], hits, config: SpeechConfig) -> np.ndarray:
    """For every row (frame), the best score of the segments of its video said around it.

    A kept frame stands for the time until the next one kept (frames in between were
    skipped as repeats); a segment counts for it when they overlap, once both are
    widened by `pad` seconds. `ranges` maps a video id to its rows (start, count);
    `hits` are segment_scores().
    """
    out = np.zeros(len(times), dtype=np.float32)
    if not hits:
        return out
    top = max(h[3] for h in hits) if config.normalise == "best" else 1.0
    for video, start, end, score, _ in hits:
        if video not in ranges:
            continue
        first, count = ranges[video]
        t = times[first : first + count]
        until = np.r_[t[1:], t[-1] + 1.0]
        near = (start < until + config.pad) & (end > t - config.pad)
        rows = out[first : first + count]
        rows[near] = np.maximum(rows[near], score / top)
    return out
