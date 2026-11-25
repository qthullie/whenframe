"""Montages of the moments a search finds: a film written sentence by sentence, and a supercut of one description.

    whenframe film story.txt -o film.mp4             # one shot per sentence of the story, in its order
    whenframe supercut "the dog jumps into the pool"   # every moment it happens, in the order it happened

Each shot is cut around the best frame of the moment found (Hit.at), a few seconds long; a photo is held on screen
with a slow zoom. A film never uses the same moment twice: a sentence takes the best moment not already in the film,
and a sentence ending with #2 takes its second best. ffmpeg renders each shot at the film's size (a video of another
shape over a blurred copy of itself), then joins them with cross-fades, the captions drawn in the picture (libass)
and written beside the film (.srt), with the shots' own sound or a music given.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path

from .search import Searcher, worth_showing

SECONDS = 3.0  # a shot's length on screen
SHORTEST = 1.0  # a video shorter than this is not a shot
FADE = 0.5  # the cross-fade between two shots, at most a third of the shortest shot
ZOOM = 0.08  # how much a photo is zoomed into while it is on screen
SIZE = (1280, 720)
FPS = 30
PICK = re.compile(r"\s+#(\d+)\s*$")  # "a sentence #2": its second best moment


class FilmError(RuntimeError):
    pass


@dataclass(frozen=True)
class Shot:
    path: str
    start: float  # where the cut starts in the video, in seconds (0 for a photo)
    length: float  # seconds on screen
    caption: str  # the story's sentence; in a supercut, where the moment comes from
    photo: bool = False
    confidence: float = 0.0
    searched_as: str = ""  # the sentence as it was searched, when it was translated
    at: float = 0.0  # the best frame of the moment


def story(args: list[str]) -> list[str]:
    """The sentences of a story: the arguments, or the lines of a text file named instead (empty lines and lines
    starting with # left out)."""
    lines = []
    for arg in args:
        path = Path(arg)
        if path.suffix.lower() in (".txt", ".md") and path.is_file():
            lines += [s.strip() for s in path.read_text(encoding="utf-8").splitlines()]
        else:
            lines.append(arg.strip())
    return [s for s in lines if s and not s.startswith("#")]


def cut(at: float, until: float, duration: float, photo: bool, seconds: float) -> tuple[float, float]:
    """(start, length) of a shot showing the instant `at`, whose picture lasts until `until` (the next frame the
    index kept: it leaves out frames that barely changed). The shot starts at `at` when that much covers it, and is
    centred on [at, until) otherwise; it stays inside the video."""
    if photo:
        return 0.0, seconds
    length = min(seconds, duration)
    lead = max(length - (until - at), 0.0) / 2
    return min(max(at - lead, 0.0), max(duration - length, 0.0)), length


def _candidates(searcher: Searcher, text: str, seconds: float, k: int, used: list[tuple[int, float, float]]):
    """The moments worth showing for a description, best first, as (hit, start, length), leaving out those that
    overlap a shot already used and videos no longer where they were indexed."""
    videos, _, times = searcher.library.snapshot()
    by_id = {v.id: v for v in videos}
    for h in searcher.search(text, k=k):
        v = by_id.get(h.video_id)
        if not worth_showing(h) or v is None or not Path(h.path).exists():
            continue
        t = times[v.start : v.start + v.count]
        later = t[t > h.at]
        start, length = cut(h.at, float(later[0]) if len(later) else v.duration, v.duration, h.photo, seconds)
        if not h.photo and length < SHORTEST:
            continue
        if any(i == h.video_id and s < start + length and start < e for i, s, e in used):
            continue
        yield h, start, length


def plan_film(searcher: Searcher, lines: list[str], seconds: float = SECONDS, k: int = 30):
    """One shot per sentence, in the story's order, each the best moment for its sentence that the film does not
    already use (or its n-th best, for a sentence ending with #n). Returns (shots, sentences left out)."""
    shots, missing, used = [], [], []
    for line in lines:
        m = PICK.search(line)
        text, n = (line[: m.start()], int(m.group(1))) if m else (line, 1)
        found = list(_candidates(searcher, text, seconds, k, used))
        if n < 1 or len(found) < n:
            missing.append(line)
            continue
        h, start, length = found[n - 1]
        used.append((h.video_id, start, start + length))
        searched = searcher.searched_as(text)
        shots.append(Shot(h.path, start, length, text, h.photo, h.confidence,
                          searched if searched != text else "", h.at))  # fmt: skip
    return shots, missing


def clock(t: float) -> str:
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def plan_supercut(searcher: Searcher, description: str, seconds: float = SECONDS, most: int = 12) -> list[Shot]:
    """The `most` best moments of one description, at most one shot per moment, in the order they were filmed
    (taken_at), each captioned with where it comes from."""
    picked, used = [], []
    for h, start, length in _candidates(searcher, description, seconds, 3 * most, used):
        if len(picked) == most:
            break
        used.append((h.video_id, start, start + length))
        where = Path(h.path).name if h.photo else f"{Path(h.path).name} · {clock(start)}"
        picked.append(Shot(h.path, start, length, where, h.photo, h.confidence, at=h.at))
    when = {s.path: taken_at(s.path) for s in picked}
    return sorted(picked, key=lambda s: (when[s.path], s.path, s.start))


# a date in a file's name, with its time if there is one: IMG_20230812_153012.jpg, PXL_20230812_153012345.mp4,
# 2023-08-12 15.30.12.jpg, WhatsApp Video 2023-08-12 at 15.30.12.mp4
DATE_IN_NAME = re.compile(r"(?<!\d)((?:19|20)\d\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])"
                          r"(?:\D{1,4}([01]\d|2[0-3])[-_.:h]?([0-5]\d)[-_.:m]?([0-5]\d))?")  # fmt: skip


def taken_at(path: str | Path) -> float:
    """When a video or photo was taken, as a timestamp: the creation time its file records (phones and cameras
    write it in their videos), else a date in its name, else when the file was last modified."""
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format_tags=creation_time,com.apple.quicktime.creationdate", "-of", "default=nw=1", str(path)]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for line in out.splitlines():
        value = line.split("=", 1)[-1].strip().replace("Z", "+00:00")
        try:
            when = datetime.fromisoformat(value)
        except ValueError:
            continue
        if when.year >= 1990:  # not a camera's clock left unset
            return when.timestamp()
    m = DATE_IN_NAME.search(Path(path).name)
    if m:
        try:
            return datetime(*(int(g) for g in m.groups() if g is not None)).timestamp()
        except ValueError:
            pass
    return Path(path).stat().st_mtime


@cache
def _ffmpeg_has() -> tuple[set[str], set[str]]:
    """The filters and the encoders of the ffmpeg installed."""

    def names(kind: str) -> set[str]:
        out = subprocess.run(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True, text=True).stdout
        return {parts[1] for parts in (line.split() for line in out.splitlines()) if len(parts) > 2}

    return names("filters"), names("encoders")


def _video_codec(final: bool) -> list[str]:
    encoders = _ffmpeg_has()[1]
    if "libx264" in encoders:
        return ["-c:v", "libx264", "-preset", "fast" if final else "veryfast", "-crf", "20" if final else "16"]
    if "libopenh264" in encoders:
        return ["-c:v", "libopenh264", "-b:v", "8M" if final else "16M"]
    return ["-c:v", "mpeg4", "-q:v", "3" if final else "2"]


def can_draw_captions() -> bool:
    """Whether ffmpeg can draw captions in the picture (built with libass)."""
    return "ass" in _ffmpeg_has()[0]


def has_audio(path: str | Path) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
           str(path)]  # fmt: skip
    return bool(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    run = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *cmd], capture_output=True, text=True, cwd=cwd)
    if run.returncode != 0:
        raise FilmError(" / ".join(run.stderr.strip().splitlines()[-3:]) or "ffmpeg failed")


def _blurred(width: int, height: int) -> str:
    """A frame filling width x height, blurred and darkened: the background of a picture of another shape (blurred
    at a quarter of the size, which looks the same and costs a sixteenth)."""
    w, h = width // 4, height // 4
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},gblur=sigma={max(h // 30, 2)},"
            f"scale={width}:{height},lutyuv=y=val*0.7")  # fmt: skip


def _render_shot(shot: Shot, dest: Path, width: int, height: int, fps: int) -> None:
    """One shot at the film's size and rate, with its sound (silence for a photo or a silent video), exactly
    round(length * fps) frames long."""
    frames = round(shot.length * fps)
    if shot.photo:
        # composed at twice the size, then zoomed into: zoompan moves by whole pixels, which a larger picture smooths
        w, h = 2 * width, 2 * height
        picture = (f"[0:v]split[b][f];[b]{_blurred(w, h)}[bb];"
                   f"[f]scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2[ff];"
                   f"[bb][ff]overlay=(W-w)/2:(H-h)/2,zoompan=z=1+{ZOOM}*on/{frames}:x=iw/2-iw/zoom/2:"
                   f"y=ih/2-ih/zoom/2:d={frames}:s={width}x{height}:fps={fps}")  # fmt: skip
        inputs = ["-i", shot.path]
    else:
        picture = (f"[0:v]fps={fps},split[b][f];[b]{_blurred(width, height)}[bb];"
                   f"[f]scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2[ff];"
                   f"[bb][ff]overlay=(W-w)/2:(H-h)/2,tpad=stop_mode=clone:stop_duration={shot.length:.3f},"
                   f"trim=end_frame={frames}")  # fmt: skip
        inputs = ["-ss", f"{shot.start:.3f}", "-t", f"{shot.length:.3f}", "-i", shot.path]
    seconds = frames / fps
    if not shot.photo and has_audio(shot.path):
        sound = f"[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=duration={seconds}"
    else:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        sound = f"[1:a]atrim=duration={seconds}"
    graph = f"{picture},format=yuv420p,setsar=1[v];{sound}[a]"
    _run([*inputs, "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-frames:v", str(frames),
          *_video_codec(final=False), "-c:a", "aac", "-b:a", "192k", str(dest)])  # fmt: skip


def timeline(lengths: list[float], fade: float) -> list[tuple[float, float]]:
    """When each caption is on screen: from the middle of the cross-fade that brings its shot in to the middle of
    the one that takes it away (from the start of the film, to its end)."""
    out, begin = [], 0.0
    for i, length in enumerate(lengths):
        first, last = i == 0, i == len(lengths) - 1
        out.append((begin + (0 if first else fade / 2), begin + length - (0 if last else fade / 2)))
        begin += length - fade
    return out


def _stamp(t: float, sep: str, digits: int) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    whole = int(s)
    frac = round((s - whole) * 10**digits)
    if frac == 10**digits:
        whole, frac = whole + 1, 0
    hours = f"{int(h):02d}" if digits == 3 else f"{int(h)}"
    return f"{hours}:{int(m):02d}:{whole:02d}{sep}{frac:0{digits}d}"


def srt(captions: list[str], times: list[tuple[float, float]]) -> str:
    blocks = [f"{i}\n{_stamp(a, ',', 3)} --> {_stamp(b, ',', 3)}\n{text}\n"
              for i, (text, (a, b)) in enumerate(zip(captions, times, strict=True), 1)]  # fmt: skip
    return "\n".join(blocks)


def ass(captions: list[str], times: list[tuple[float, float]], width: int, height: int) -> str:
    """The captions as libass draws them: white, outlined, at the bottom, faded in and out, wrapped when long."""
    size, outline, margin = round(height * 0.058), max(round(height * 0.0042), 1), round(height * 0.07)
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Default,Arial,{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},"
        f"{max(outline // 2, 1)},2,{round(width * 0.06)},{round(width * 0.06)},{margin},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]  # fmt: skip
    for text, (a, b) in zip(captions, times, strict=True):
        clean = text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")
        head.append(f"Dialogue: 0,{_stamp(a, '.', 2)},{_stamp(b, '.', 2)},Default,,0,0,0,,{{\\fad(250,250)}}{clean}")
    return "\n".join(head) + "\n"


def render(shots: list[Shot], out: str | Path, music: str | Path | None = None, captions: bool = True,
           size: tuple[int, int] = SIZE, fps: int = FPS,
           report: Callable[[int, Shot | None], None] | None = None) -> tuple[float, bool]:  # fmt: skip
    """The film of these shots at `out` (an MP4 file), its captions beside it (.srt). Each shot is rendered on its
    own (`report(i, shot)` before each, `report(len(shots), None)` before they are joined), then all are joined
    with cross-fades, the film fading in from black and out to it, with the shots' sound or `music` (looped if
    shorter, faded out at the end). Returns the film's length in seconds, and whether the captions are drawn in the
    picture: asked for, and ffmpeg can (libass)."""
    if not shots:
        raise FilmError("no shot to put in the film")
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    lengths = [round(s.length * fps) / fps for s in shots]
    fade = round(min(FADE, min(lengths) / 3) * fps) / fps
    times = timeline(lengths, fade)
    total = sum(lengths) - fade * (len(shots) - 1)
    out.with_suffix(".srt").write_text(srt([s.caption for s in shots], times), encoding="utf-8")
    burn = captions and can_draw_captions()
    with tempfile.TemporaryDirectory(prefix=".whenframe-film-", dir=out.parent) as tmp:
        tmp = Path(tmp)
        for i, shot in enumerate(shots):
            if report:
                report(i, shot)
            _render_shot(shot, tmp / f"{i:04d}.mp4", width, height, fps)
        if report:
            report(len(shots), None)
        inputs, graph = [], []
        for i in range(len(shots)):
            inputs += ["-i", f"{i:04d}.mp4"]
            graph.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS[v{i}]")
            if music is None:
                graph.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
        video, sound, begin = "[v0]", "[a0]", lengths[0]
        for i in range(1, len(shots)):
            graph.append(f"{video}[v{i}]xfade=transition=fade:duration={fade}:offset={begin - fade:.4f}[x{i}]")
            if music is None:
                graph.append(f"{sound}[a{i}]acrossfade=d={fade}[y{i}]")
            video, sound, begin = f"[x{i}]", f"[y{i}]", begin + lengths[i] - fade
        end = min(0.8, total / 4)
        picture = f"{video}fade=t=in:st=0:d={min(0.4, total / 4):.3f},fade=t=out:st={total - end:.4f}:d={end:.3f}"
        if burn:
            (tmp / "captions.ass").write_text(ass([s.caption for s in shots], times, width, height), encoding="utf-8")
            picture += ",ass=captions.ass"  # read from the folder ffmpeg runs in: no path to escape
        graph.append(f"{picture}[vout]")
        if music is not None:
            inputs += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
            out_fade = min(2.0, total / 4)
            graph.append(f"[{len(shots)}:a]atrim=duration={total:.4f},asetpts=PTS-STARTPTS,afade=t=in:d=0.5,"
                         f"afade=t=out:st={total - out_fade:.4f}:d={out_fade:.3f}[aout]")  # fmt: skip
        else:
            graph.append(f"{sound}afade=t=out:st={total - end:.4f}:d={end:.3f}[aout]")
        _run([*inputs, "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "[aout]", "-t", f"{total:.4f}",
              *_video_codec(final=True), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
              "-movflags", "+faststart", "-f", "mp4", str(out)], cwd=tmp)  # fmt: skip
    return total, burn
