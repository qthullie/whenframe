// whenframe in the browser: the frames of three films, embedded once with SigLIP 2 (scripts/site.py), searched here
// the way the tool searches: cosine minus the query-bank offset, 10-second windows around the best frames.
// The twelve descriptions were embedded by the tool too; the films play from small copies (scripts/site_videos.sh).
const DATA = "data/";
const HIDDEN_BELOW = 0.005;  // moments shown as "0 %" sure say nothing: left out
const $ = (id) => document.getElementById(id);

let meta, frames, offsets, suggestions, vectors;
let playing = null;  // the moment being played, which stops at its end; null for a whole film

// float16 -> float32, through a table of the 65,536 values
const F16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  F16[h] = e === 0 ? sign * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : sign * Infinity) : sign * 2 ** (e - 15) * (1 + f / 1024);
}

const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

function tile(row) {
  // a div showing frame `row` from the thumbnail sheets
  const div = el("div", "shot");
  const k = row % meta.per_sheet, rows = meta.per_sheet / meta.cols;
  div.style.backgroundImage = `url(${DATA}sheets/${Math.floor(row / meta.per_sheet)}.webp)`;
  div.style.backgroundPosition = `${((k % meta.cols) / (meta.cols - 1)) * 100}% ${(Math.floor(k / meta.cols) / (rows - 1)) * 100}%`;
  return div;
}

function source(film) {
  // H.264 where the browser has it, VP9 elsewhere
  const mp4 = document.createElement("video").canPlayType('video/mp4; codecs="avc1.4D401E, mp4a.40.2"');
  return `${film.video}.${mp4 ? "mp4" : "webm"}`;
}

function search(q) {
  const n = meta.count, d = meta.dim, raw = new Float32Array(n), score = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0, o = i * d; k < d; k++) s += frames[o + k] * q[k];
    raw[i] = s;
    score[i] = s - offsets[i];
  }
  const out = [], W = meta.window;
  meta.films.forEach((film, f) => {
    const rows = [];
    for (let i = 0; i < n; i++) if (meta.film[i] === f) rows.push(i);
    const left = rows.map((i) => score[i]);
    for (let m = 0; m < meta.per_video; m++) {
      let best = -1;
      left.forEach((v, j) => { if (v > -Infinity && (best < 0 || v > left[best])) best = j; });
      if (best < 0) break;
      const row = rows[best], t = meta.times[row];
      const start = Math.min(Math.max(0, t + 0.5 - W / 2), Math.max(0, film.duration - W));
      const end = Math.min(film.duration, start + W);
      let top = -Infinity;
      for (const i of rows) if (meta.times[i] >= start && meta.times[i] < end) top = Math.max(top, raw[i]);
      out.push({ film: f, start, end, row, score: score[row], confidence: 1 / (1 + Math.exp(-(meta.scale * top + meta.bias))) });
      rows.forEach((i, j) => { if (Math.abs(meta.times[i] - t) < W) left[j] = -Infinity; });
    }
  });
  return out.sort((a, b) => b.score - a.score);
}

function showFilms() {
  // no description picked: the films themselves, to watch from the start
  $("status").textContent = "The three films the search runs on. Pick a description above to find a moment in them.";
  const box = $("results");
  box.replaceChildren();
  meta.films.forEach((film, f) => {
    const rows = [];
    for (let i = 0; i < meta.count; i++) if (meta.film[i] === f) rows.push(i);
    const card = el("button", "card");
    card.type = "button";
    const shot = tile(rows[Math.floor(rows.length / 3)]);
    shot.append(el("span", "when", clock(film.duration)));
    const info = el("div", "meta");
    info.append(el("div", "title", film.title), el("div", "conf", `${rows.length.toLocaleString("en")} frames indexed`));
    card.append(shot, info);
    card.addEventListener("click", () => open(film, null));
    box.append(card);
  });
}

function showMoments(query, vector, ms) {
  const t0 = performance.now();
  const hits = search(vector);
  const shown = hits.filter((h) => h.confidence >= HIDDEN_BELOW);
  const took = ms + performance.now() - t0;
  const hidden = hits.length - shown.length;
  $("status").textContent = `${shown.length} moments in ${Math.round(took)} ms, for “${query}”` +
    (hidden ? ` (${hidden} more at 0 % left out)` : "");
  const box = $("results");
  box.replaceChildren();
  shown.forEach((h, i) => {
    const card = el("button", h.confidence < 0.05 ? "card unlikely" : "card");
    card.type = "button";
    const shot = tile(h.row);
    shot.append(el("span", "rank", String(i + 1)), el("span", "when", `${clock(h.start)} – ${clock(h.end)}`));
    const info = el("div", "meta");
    info.append(el("div", "title", meta.films[h.film].title),
                el("div", "conf", `${Math.round(100 * h.confidence)} % sure the best frame matches`));
    card.append(shot, info);
    card.addEventListener("click", () => open(meta.films[h.film], h));
    box.append(card);
  });
}

function open(film, h) {
  // a moment plays its ten seconds and stops; a film plays from the start
  playing = h;
  const v = $("m-video");
  $("m-title").textContent = film.title;
  $("m-when").textContent = h ? `${clock(h.start)} – ${clock(h.end)}` : clock(film.duration);
  $("m-again").hidden = !h;
  const board = $("m-board");
  board.replaceChildren();
  if (h) {
    for (let i = 0; i < meta.count; i++) {
      if (meta.films[meta.film[i]] !== film || meta.times[i] < h.start || meta.times[i] >= h.end) continue;
      const shot = tile(i);
      if (i === h.row) shot.classList.add("best");
      shot.append(el("span", "", clock(meta.times[i])));
      shot.title = "Play from here";
      const t = meta.times[i];
      shot.addEventListener("click", () => { v.currentTime = t; v.play().catch(() => {}); });
      board.append(shot);
    }
  }
  const caption = $("m-caption");
  caption.replaceChildren(h ? "The frames of the moment as the search saw them; the outlined one matches best. "
                            : "The whole film. ", Object.assign(el("a", "", "About the film"),
                                                                { href: film.url, target: "_blank", rel: "noopener" }));
  v.src = source(film);
  v.onloadedmetadata = () => { v.currentTime = h ? h.start : 0; v.play().catch(() => {}); };
  v.onerror = () => { caption.prepend("This browser could not load the film. "); };
  $("moment").showModal();
}

$("m-video").addEventListener("timeupdate", () => {
  const v = $("m-video");
  if (playing && v.currentTime >= playing.end && !v.paused) v.pause();
});
$("m-again").addEventListener("click", () => {
  const v = $("m-video");
  if (playing) { v.currentTime = playing.start; v.play().catch(() => {}); }
});
$("m-close").addEventListener("click", () => $("moment").close());
$("moment").addEventListener("close", () => {
  const v = $("m-video");
  v.pause();
  v.removeAttribute("src");
  v.load();
});

function pick(query) {
  // a description from the list, or none: the films
  const known = suggestions.findIndex((s) => s === query);
  history.replaceState(null, "", known >= 0 ? `?q=${encodeURIComponent(query)}` : location.pathname);
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.textContent === query));
  if (known < 0) showFilms();
  else showMoments(query, vectors.subarray(known * meta.dim, (known + 1) * meta.dim), 0);
}

async function start() {
  const get = (name, kind) => fetch(DATA + name).then((r) => (kind === "json" ? r.json() : r.arrayBuffer()));
  const [m, f, o, s, v] = await Promise.all([get("frames.json", "json"), get("frames.bin"), get("offsets.bin"),
                                             get("queries.json", "json"), get("queries.bin")]);
  meta = m;
  const half = new Uint16Array(f);
  frames = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) frames[i] = F16[half[i]];
  offsets = new Float32Array(o);
  suggestions = s;
  vectors = new Float32Array(v);
  const minutes = Math.round(meta.films.reduce((t, x) => t + x.duration, 0) / 60);
  $("what").textContent = `Three open movies: ${meta.count.toLocaleString("en")} frames, ${minutes} minutes`;
  for (const text of suggestions) {
    const chip = el("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => pick(chip.classList.contains("on") ? null : text));
    $("chips").append(chip);
  }
  fetch(DATA + "summary.json").then((r) => r.json()).then((sum) => {
    $("n-default").textContent = `${(100 * sum.default).toFixed(1)} %`;
    $("n-videos").textContent = sum.videos.toLocaleString("en");
    $("n-shots").textContent = `${(100 * sum.shots).toFixed(1)} %`;
  }).catch(() => {});
  pick(new URLSearchParams(location.search).get("q"));
}

$("all").addEventListener("click", () => pick(null));
start();
