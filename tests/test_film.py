"""Montages: the shots chosen for a story or a description, and the film ffmpeg makes of them."""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pytest
from test_library import COLORS, ToyEncoder, make_video

from whenframe import Library, Searcher, Settings, cli, film
from whenframe.film import Shot

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def with_sound(src, dest, when=None):
    """A copy of a video with a tone as its sound, and a creation time if given."""
    meta = ["-metadata", f"creation_time={when}"] if when else []
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "lavfi", "-i", "sine=f=440", "-shortest",
           "-c:v", "copy", "-c:a", "aac", *meta, str(dest)]  # fmt: skip
    subprocess.run(cmd, check=True)


@pytest.fixture
def library(tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    make_video(str(tmp_path / "a.mp4"), [("grey", 6), ("red", 3), ("grey", 6)])
    with_sound(tmp_path / "a.mp4", folder / "a.mp4", "2022-07-01T10:00:00Z")
    make_video(str(folder / "b.mp4"), [("green", 4), ("grey", 8)])
    make_video(str(folder / "c.mp4"), [("grey", 5), ("blue", 5)])
    # a dark red photo, standing upright: the second best red moment
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0xc80000:s=48x64", "-frames:v", "1",
                    str(folder / "red.jpg")], check=True)  # fmt: skip
    lib = Library(tmp_path / "lib", Settings(model="toy", dedup=4.0), encoder=ToyEncoder())
    for name in ("a.mp4", "b.mp4", "c.mp4", "red.jpg"):
        lib.add(folder / name)
    return lib


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,width,height",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)  # fmt: skip
    return {s["codec_type"]: s for s in json.loads(out.stdout)["streams"]}


def colour_at(path, t):
    """The mean colour of the top half of the film's frame at t seconds."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1", "-vf", "scale=16:16",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]  # fmt: skip
    frame = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(16, 16, 3)
    return frame[:8].reshape(-1, 3).mean(axis=0)


def nearest(colour):
    return min(COLORS, key=lambda name: np.abs(np.array(COLORS[name]) - colour).sum())


def test_a_story_is_its_sentences_or_the_lines_of_a_text_file(tmp_path):
    (tmp_path / "story.txt").write_text("# my holidays\na dog on the beach\n\n  the sunset #2 \n", encoding="utf-8")
    assert film.story([str(tmp_path / "story.txt")]) == ["a dog on the beach", "the sunset #2"]
    assert film.story(["a dog", " the sea "]) == ["a dog", "the sea"]


def test_a_shot_starts_on_the_best_frame_when_its_picture_lasts_and_is_centred_on_it_otherwise():
    assert film.cut(6.0, 9.0, 15.0, False, 3.0) == (6.0, 3.0)  # still from 6 to 9: the shot is that
    assert film.cut(6.0, 7.0, 15.0, False, 3.0) == (5.0, 3.0)  # one second of it: one more on each side
    assert film.cut(0.0, 1.0, 15.0, False, 3.0) == (0.0, 3.0)  # inside the video
    assert film.cut(14.0, 15.0, 15.0, False, 3.0) == (12.0, 3.0)
    assert film.cut(1.0, 2.0, 2.0, False, 3.0) == (0.0, 2.0)  # a video shorter than a shot: all of it
    assert film.cut(0.0, 0.0, 0.0, True, 3.0) == (0.0, 3.0)  # a photo, held


def test_a_film_takes_one_shot_per_sentence_in_its_order_never_the_same_moment_twice(library):
    searcher = Searcher(library)
    shots, missing = film.plan_film(searcher, ["red", "blue", "green", "red"])
    assert [os.path.basename(s.path) for s in shots] == ["a.mp4", "c.mp4", "b.mp4", "red.jpg"]
    assert [s.caption for s in shots] == ["red", "blue", "green", "red"]
    assert (shots[0].start, shots[0].length) == (6.0, 3.0)  # the red seconds of a.mp4
    assert shots[1].start == 5.0 and shots[2].start == 0.0 and shots[3].photo
    assert missing == []
    shots, missing = film.plan_film(searcher, ["red", "red", "red"])  # two red moments in the library, not three
    assert len(shots) == 2 and missing == ["red"]


def test_a_sentence_ending_with_a_number_takes_that_moment(library):
    searcher = Searcher(library)
    first, _ = film.plan_film(searcher, ["red"])
    second, _ = film.plan_film(searcher, ["red #2"])
    assert second[0].caption == "red" and second[0].path != first[0].path
    assert film.plan_film(searcher, ["red #3"]) == ([], ["red #3"])


def test_a_supercut_puts_the_moments_in_the_order_they_were_filmed(library, tmp_path):
    searcher = Searcher(library)
    shots = film.plan_supercut(searcher, "red")
    assert {os.path.basename(s.path) for s in shots} == {"a.mp4", "red.jpg"}
    # a.mp4 says it was filmed in 2022; the photo says nothing, and was modified now
    assert [os.path.basename(s.path) for s in shots] == ["a.mp4", "red.jpg"]
    assert shots[0].caption == "a.mp4 · 0:06" and shots[1].caption == "red.jpg"
    os.utime(tmp_path / "media" / "red.jpg", (1e9, 1e9))  # 2001
    assert [os.path.basename(s.path) for s in film.plan_supercut(searcher, "red")] == ["red.jpg", "a.mp4"]
    assert len(film.plan_supercut(searcher, "red", most=1)) == 1


def test_when_a_file_was_taken(tmp_path):
    make_video(str(tmp_path / "plain.mp4"), [("grey", 1)])
    with_sound(tmp_path / "plain.mp4", tmp_path / "dated.mp4", "2021-03-04T05:06:07Z")
    assert film.taken_at(tmp_path / "dated.mp4") == datetime(2021, 3, 4, 5, 6, 7, tzinfo=timezone.utc).timestamp()
    shutil.copy(tmp_path / "plain.mp4", tmp_path / "VID_20200102_030405.mp4")
    assert film.taken_at(tmp_path / "VID_20200102_030405.mp4") == datetime(2020, 1, 2, 3, 4, 5).timestamp()
    os.utime(tmp_path / "plain.mp4", (1e9, 1e9))
    assert film.taken_at(tmp_path / "plain.mp4") == 1e9


def test_captions_last_from_the_middle_of_one_cross_fade_to_the_middle_of_the_next():
    times = film.timeline([3.0, 3.0, 2.0], 0.5)
    assert times == [(0.0, 2.75), (2.75, 5.25), (5.25, 7.0)]
    text = film.srt(["un chien", "the sea", "end"], times)
    assert text.startswith("1\n00:00:00,000 --> 00:00:02,750\nun chien\n\n2\n00:00:02,750 --> 00:00:05,250\n")
    assert "Dialogue: 0,0:00:02.75,0:00:05.25,Default,,0,0,0,,{\\fad(250,250)}the sea" in film.ass(
        ["un chien", "the sea", "end"], times, 1280, 720
    )


def test_a_film_is_its_shots_joined_with_their_sound_and_captions(library, tmp_path):
    shots, _ = film.plan_film(Searcher(library), ["red", "blue", "green"])
    out = tmp_path / "out" / "film.mp4"
    seen = []
    seconds, drawn = film.render(shots, out, size=(160, 96), fps=10, report=lambda i, s: seen.append(i))
    assert seen == [0, 1, 2, 3] and seconds == pytest.approx(9.0 - 2 * 0.5)
    streams = probe(out)
    assert (streams["video"]["width"], streams["video"]["height"]) == (160, 96)
    assert float(streams["video"]["duration"]) == pytest.approx(8.0, abs=0.05)
    assert float(streams["audio"]["duration"]) == pytest.approx(8.0, abs=0.1)
    assert [nearest(colour_at(out, t)) for t in (1.5, 4.0, 6.5)] == ["red", "blue", "green"]
    assert out.with_suffix(".srt").read_text(encoding="utf-8").count(" --> ") == 3
    assert drawn == film.can_draw_captions()
    assert not list(out.parent.glob(".whenframe-film-*"))  # the shots rendered on their own are gone


def test_a_film_with_music_and_photos(library, tmp_path):
    music = tmp_path / "music.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=1", str(music)], check=True)
    shots = [Shot(str(tmp_path / "media" / "red.jpg"), 0.0, 2.0, "a photo", photo=True),
             Shot(str(tmp_path / "media" / "c.mp4"), 5.0, 2.0, "blue")]  # fmt: skip
    out = tmp_path / "with music.mp4"
    seconds, _ = film.render(shots, out, music=music, captions=False, size=(160, 96), fps=10)
    streams = probe(out)
    assert seconds == pytest.approx(3.5) and float(streams["audio"]["duration"]) == pytest.approx(3.5, abs=0.1)
    assert [nearest(colour_at(out, t)) for t in (0.8, 2.5)] == ["red", "blue"]


def test_the_film_command(library, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Library", lambda folder: library)
    out = tmp_path / "story.mp4"
    assert cli.main(["film", "red", "blue", "--plan", "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "1. “red”" in printed and "a.mp4" in printed and "c.mp4" in printed and not out.exists()
    assert cli.main(["film", "red", "red", "red", "-o", str(out), "--seconds", "2"]) == 0
    printed = capsys.readouterr().out
    assert "left out, nothing else in the library matches it well enough: “red”" in printed
    assert float(probe(out)["video"]["duration"]) == pytest.approx(2 * 2.0 - 0.5, abs=0.05)
    assert cli.main(["supercut", "blue", "-o", str(tmp_path / "blue.mp4")]) == 0
    assert (tmp_path / "blue.mp4").exists() and "c.mp4 · 0:05" in (tmp_path / "blue.srt").read_text()
    assert cli.main(["film", "--plan", "blue", "#1"]) == 0  # a line that is only a comment is no sentence
    assert cli.main(["film", "blue", "--seconds", "0.5"]) == 2
    assert cli.main(["supercut", "blue", "--music", str(tmp_path / "none.mp3")]) == 2
