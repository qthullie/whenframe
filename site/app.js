// whenframe in the browser: someone's phone library (library.json), indexed once by the tool (scripts/site.py) and
// searched here the way the tool searches it: each model's cosine minus its query-bank offset, the two models' scores
// standardised over the library and added (PE's with its weight), put back on SigLIP 2's scale; ten-second windows
// around a video's best frames, a photo as its own moment. Films are planned as `whenframe film` plans them
// (whenframe/film.py): one shot per sentence, the best moment the film does not use yet, cut from its best frame.
// The sentences were embedded by the tool; a description of the visitor's own is embedded by the demo's text server
// (demo-server/, config.json), with the same text models.
const DATA = "data/", MEDIA = "media/";
const HIDDEN_BELOW = 0.005;  // moments shown as "0 %" sure say nothing: left out, as the tool leaves them out
const SHORTLIST = 64;  // files examined in detail, as the tool's search does
const SHOWN = 24, GALLERY = 30;
const MOST = 8;  // sentences in a film of one's own
const $ = (id) => document.getElementById(id);
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

let meta, models, Q, FILM, rowsOf;
let server = "";  // the text server's address, when there is one
let playing = null;  // the moment being played in the dialog, which stops at its end; null for a whole video
let asked = 0;  // the latest description sent to the server: an older answer arriving late is dropped
let story = 0;  // the film shown: an index of FILM.stories, or -1 for the visitor's own
const mine = [];  // the visitor's sentences

// float16 -> float32, through a table of the 65,536 values
const F16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  F16[h] = e === 0 ? sign * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : sign * Infinity) : sign * 2 ** (e - 15) * (1 + f / 1024);
}
const half = (buffer) => {
  const h = new Uint16Array(buffer), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = F16[h[i]];
  return out;
};

const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const pct = (x) => `${(100 * x).toFixed(1)} %`;
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const sure = (cos) => 1 / (1 + Math.exp(-(meta.scale * cos + meta.bias)));  // the first model's, as the tool says it

function tile(row) {
  // a div showing frame `row` (a photo is one frame) from the thumbnail sheets
  const div = el("div", "shot");
  const k = row % meta.per_sheet, rows = meta.per_sheet / meta.cols;
  div.style.backgroundImage = `url(${DATA}sheets/${Math.floor(row / meta.per_sheet)}.webp)`;
  div.style.backgroundPosition = `${((k % meta.cols) / (meta.cols - 1)) * 100}% ${(Math.floor(k / meta.cols) / (rows - 1)) * 100}%`;
  return div;
}

function standardised(x) {
  let mean = 0, sq = 0;
  for (const v of x) mean += v;
  mean /= x.length;
  for (const v of x) sq += (v - mean) * (v - mean);
  const sd = Math.max(Math.sqrt(sq / x.length), 1e-12);
  return { z: x.map((v) => (v - mean) / sd), mean, sd };
}

function search(qs) {
  // qs: the description embedded by each model, in meta.models' order
  const n = meta.count, raw = [], corrected = [];
  models.forEach((m, k) => {
    const d = m.dim, r = new Float64Array(n), c = new Float64Array(n), q = qs[k];
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0, o = i * d; j < d; j++) s += m.frames[o + j] * q[j];
      r[i] = s;
      c[i] = s - m.offsets[i];
    }
    raw.push(r);
    corrected.push(c);
  });
  // the models together, as the tool adds them (encoder.FUSED), on the first model's scale
  const sum = new Float64Array(n);
  models.forEach((m, k) => standardised(corrected[k]).z.forEach((v, i) => { sum[i] += m.weight * v; }));
  const first = standardised(corrected[0]), z = standardised(sum).z;
  const ranking = z.map((v) => first.mean + first.sd * v);
  const best = meta.items.map((_, f) => Math.max(...rowsOf[f].map((i) => ranking[i])));
  const order = best.map((_, f) => f).sort((a, b) => best[b] - best[a]).slice(0, SHORTLIST);
  const out = [], W = meta.window;
  for (const f of order) {
    const item = meta.items[f], rows = rowsOf[f];
    if (item.kind === "photo") {
      const row = rows[0];
      out.push({ item: f, start: 0, end: 0, row, score: ranking[row], confidence: sure(raw[0][row]) });
      continue;
    }
    const left = rows.map((i) => ranking[i]);
    for (let m = 0; m < meta.per_video; m++) {
      let b = -1;
      left.forEach((v, j) => { if (v > -Infinity && (b < 0 || v > left[b])) b = j; });
      if (b < 0) break;
      const t = meta.times[rows[b]];
      const start = Math.min(Math.max(0, t + 0.5 - W / 2), Math.max(0, item.duration - W));
      const end = Math.min(item.duration, start + W);
      let inside = rows.filter((i) => meta.times[i] >= start && meta.times[i] < end);
      if (!inside.length) inside = rows;
      const top = Math.max(...inside.map((i) => raw[0][i]));
      const row = inside.reduce((a, i) => (ranking[i] > ranking[a] ? i : a), inside[0]);
      out.push({ item: f, start, end, row, score: left[b], confidence: sure(top) });
      rows.forEach((i, j) => { if (Math.abs(meta.times[i] - t) < W) left[j] = -Infinity; });
    }
  }
  return out.sort((a, b) => b.score - a.score);
}

// ---- films ----

const queriesOf = (k) => models.map((m) => m.queries.subarray(k * m.dim, (k + 1) * m.dim));

function cut(at, until, duration, photo, seconds) {
  // whenframe/film.py's cut: from the best frame when its picture lasts the shot, centred on it otherwise
  if (photo) return [0, seconds];
  const length = Math.min(seconds, duration);
  const lead = Math.max(length - (until - at), 0) / 2;
  return [Math.min(Math.max(at - lead, 0), Math.max(duration - length, 0)), length];
}

function plan(lines) {
  // as `whenframe film` plans a film: for each sentence, the best moment worth showing that the film does not use
  const used = [];
  return lines.map((line) => {
    const k = Q.texts.indexOf(line);
    const hits = search(queriesOf(k)).slice(0, FILM.candidates).filter((h) => h.confidence >= HIDDEN_BELOW);
    for (const h of hits) {
      const item = meta.items[h.item], photo = item.kind === "photo", at = meta.times[h.row];
      const next = rowsOf[h.item].find((i) => meta.times[i] > at);
      const [start, length] = cut(at, next === undefined ? item.duration : meta.times[next], item.duration, photo,
                                  FILM.seconds);
      if (!photo && length < FILM.shortest) continue;
      if (used.some((u) => u.item === h.item && u.start < start + length && start < u.end)) continue;
      used.push({ item: h.item, start, end: start + length });
      return { line, k, item: h.item, start, length, row: h.row, sure: h.confidence };
    }
    return { line, k, missing: true };
  });
}

// the player: two layers, the next shot sought in the hidden one while the other plays, then faded in over it
const P = { shots: [], begins: [], total: 0, i: -1, t: 0, since: 0, running: false, ended: false, raf: 0,
            layers: [], front: null, next: null, auto: false, cards: [] };
const music = $("music");
let musicOn = false;

function setupLayers() {
  P.layers = [...document.querySelectorAll(".layer")].map((e) => {
    const canvas = e.querySelector("canvas");
    return { el: e, canvas, ctx: canvas.getContext("2d"), video: e.querySelector("video"), img: e.querySelector("img"),
             shot: null, ready: false, fallback: null };
  });
}

function paint(L) {
  // the blurred background: the middle of the picture at the stage's shape, drawn small and blurred by CSS
  const src = L.video.hidden ? L.img : L.video;
  const w = src.videoWidth || src.naturalWidth, h = src.videoHeight || src.naturalHeight;
  if (!w || !h || L.fallback) return;
  let sw = w, sh = (w * 9) / 16;
  if (sh > h) { sh = h; sw = (h * 16) / 9; }
  try { L.ctx.drawImage(src, (w - sw) / 2, (h - sh) / 2, sw, sh, 0, 0, L.canvas.width, L.canvas.height); } catch { /* not decoded yet */ }
}

function fail(L, shot) {
  // a file this browser cannot play: its frame from the thumbnails stands in for it
  if (L.shot !== shot || L.fallback) return;
  L.video.hidden = true;
  L.img.hidden = true;
  L.fallback = tile(shot.row);
  L.el.append(L.fallback);
  L.ready = true;
}

function load(L, shot) {
  // a shot put in a layer, still: a video sought to the cut, a photo loaded
  if (L.shot === shot) return;
  L.shot = shot;
  L.ready = false;
  if (L.fallback) { L.fallback.remove(); L.fallback = null; }
  const item = meta.items[shot.item];
  L.img.classList.remove("zoom");
  if (item.kind === "photo") {
    L.video.pause();
    L.video.hidden = true;
    L.img.hidden = false;
    L.img.onload = () => { if (L.shot === shot) { L.ready = true; paint(L); } };
    L.img.onerror = () => fail(L, shot);
    L.img.src = MEDIA + item.file;
    if (L.img.complete && L.img.naturalWidth && L.img.src.endsWith(item.file)) { L.ready = true; paint(L); }
    return;
  }
  L.img.hidden = true;
  L.video.hidden = false;
  const v = L.video, url = new URL(MEDIA + item.file, location.href).href;
  v.pause();
  v.onseeked = () => { if (L.shot === shot) { L.ready = true; paint(L); } };
  v.onerror = () => fail(L, shot);
  const seek = () => { if (L.shot === shot) v.currentTime = shot.start + 0.01; };
  if (v.src === url && v.readyState >= 1) seek();
  else {
    v.onloadedmetadata = seek;
    v.src = url;
  }
}

function enter(i) {
  // shot i comes on screen: cut in if nothing is shown, faded in over the shot before otherwise
  const L = P.next, other = P.layers.find((x) => x !== L), shot = P.shots[i];
  const first = !P.layers.some((x) => x.el.classList.contains("on"));
  P.i = i;
  P.front = L;
  P.next = other;
  L.el.classList.remove("in");
  void L.el.offsetWidth;  // the fade starts again
  L.el.classList.add("on");
  if (!first) L.el.classList.add("in");
  if (!L.fallback && !L.video.hidden) L.video.play().catch(() => {});
  if (!L.img.hidden) {
    L.img.style.setProperty("--shot", `${shot.length}s`);
    L.img.classList.add("zoom");
  }
  setTimeout(() => {
    if (P.front !== L) return;
    other.el.classList.remove("on", "in");
    other.video.pause();
    L.el.classList.remove("in");
    if (P.shots[i + 1]) load(other, P.shots[i + 1]);
  }, first ? 0 : FILM.fade * 1000 + 60);
  caption(i, first);
  P.cards.forEach((c, n) => c.classList.toggle("now", n === i));
}

function caption(i, first) {
  const s = P.shots[i], item = meta.items[s.item], sub = $("subtitle"), tag = $("tag");
  const apply = () => {
    $("line").textContent = s.line;
    const as = Q.searched_as[s.k];
    $("searched").textContent = as !== s.line ? `searched as “${as}”` : "";
    $("tag-file").textContent = item.file;
    $("tag-when").textContent = item.kind === "photo" ? "· photo" : `· ${clock(s.start)}–${clock(s.start + s.length)}`;
    $("tag-sure").textContent = `${Math.round(100 * s.sure)} % sure it matches`;
    sub.style.opacity = 1;
    tag.style.opacity = 1;
  };
  if (first) apply();
  else {
    sub.style.opacity = 0;
    tag.style.opacity = 0;
    setTimeout(() => { if (P.i === i) apply(); }, FILM.fade * 500);
  }
}

function progress(t) {
  const bars = $("bars").children;
  P.shots.forEach((_, n) => {
    const a = P.begins[n], b = n + 1 < P.shots.length ? P.begins[n + 1] : P.total;
    bars[n].firstChild.style.width = `${Math.min(Math.max((t - a) / (b - a), 0), 1) * 100}%`;
  });
}

function tick(now) {
  if (!P.running) return;
  let t = P.t + (now - P.since) / 1000;
  const n = P.i + 1;
  if (n < P.shots.length && t >= P.begins[n]) {
    if (!P.next.ready) {  // the next shot is still loading: the film waits for it
      P.t = P.begins[n];
      P.since = now;
      t = P.t;
    } else enter(n);
  }
  if (t >= P.total) return end();
  progress(t);
  for (const L of P.layers) if (L.el.classList.contains("on") && !L.video.hidden) paint(L);
  P.raf = requestAnimationFrame(tick);
}

function filmTime() {
  return P.running ? P.t + (performance.now() - P.since) / 1000 : P.t;
}

function play() {
  if (!P.shots.length) return;
  if (P.ended) return replay();
  P.running = true;
  P.since = performance.now();
  $("play").textContent = "Pause";
  $("big").hidden = true;
  if (P.front && P.front.el.classList.contains("on") && !P.front.video.hidden) P.front.video.play().catch(() => {});
  syncMusic();
  cancelAnimationFrame(P.raf);
  P.raf = requestAnimationFrame(tick);
}

function pause() {
  if (!P.running) return;
  P.t = filmTime();
  P.running = false;
  cancelAnimationFrame(P.raf);
  P.layers.forEach((L) => L.video.pause());
  $("play").textContent = "Play";
  syncMusic();
}

function end() {
  P.running = false;
  P.ended = true;
  P.t = P.total;
  progress(P.total);
  $("curtain").classList.add("on");
  $("play").textContent = "Play";
  $("big").textContent = "Play again";
  $("big").hidden = false;
  fadeMusic();
}

function reset(from = 0) {
  // the film ready to play from shot `from`, nothing on screen
  P.running = false;
  cancelAnimationFrame(P.raf);
  P.ended = false;
  P.i = from - 1;
  P.t = P.begins[from] || 0;
  $("curtain").classList.remove("on");
  for (const L of P.layers) {
    L.el.classList.remove("on", "in");
    L.video.pause();
  }
  P.next = P.layers[0];
  load(P.layers[0], P.shots[from]);
  if (P.shots[from + 1]) load(P.layers[1], P.shots[from + 1]);
  progress(P.t);
}

function replay() {
  if (!P.shots.length) return;
  reset(0);
  play();
}

function jump(i) {
  reset(i);
  play();
}

function startFilm(plan, title) {
  // a planned film, on the stage and in the strip beneath it, played from its start
  pause();
  P.shots = plan.filter((s) => !s.missing);
  P.begins = [];
  let b = 0;
  for (const s of P.shots) {
    P.begins.push(b);
    b += s.length - FILM.fade;
  }
  P.total = b + FILM.fade;
  $("bars").replaceChildren(...P.shots.map(() => { const s = el("span"); s.append(el("i")); return s; }));
  strip(plan);
  const name = (title || "film").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "film";
  $("cli").textContent = `$ whenframe film ${plan.map((s) => `"${s.line.replaceAll('"', '\\"')}"`).join(" ")} -o ${name}.mp4`;
  if (!P.shots.length) {
    $("line").textContent = "Nothing in the library matches these sentences well enough.";
    $("searched").textContent = "";
    return;
  }
  reset(0);
  if (still) {  // reduced motion: the film waits to be asked for
    $("big").textContent = "Play the film";
    $("big").hidden = false;
    return;
  }
  play();
}

function strip(plan) {
  const box = $("strip");
  box.replaceChildren();
  P.cards = [];
  plan.forEach((s, n) => {
    if (s.missing) {
      box.append(el("div", "gone", `${n + 1}. “${s.line}”: nothing in the library matches it well enough, left out`));
      return;
    }
    const item = meta.items[s.item], b = el("button");
    b.type = "button";
    Object.assign(b.dataset, { file: item.file, start: s.start.toFixed(2), length: s.length.toFixed(2) });
    const th = tile(s.row);
    th.append(el("span", "when", item.kind === "photo" ? "photo" : `${clock(s.start)}–${clock(s.start + s.length)}`));
    const src = el("div", "src");
    src.append(el("code", "", item.file));
    b.append(th, el("div", "say", `${n + 1}. ${s.line}`), src);
    const i = P.cards.length;
    b.addEventListener("click", () => jump(i));
    P.cards.push(b);
    box.append(b);
  });
}

function show(which) {
  // a story of the page, or the visitor's own (-1), planned by whoever picks the shots now
  story = which;
  const lines = which < 0 ? mine : FILM.stories[which].lines;
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(Number(t.dataset.story) === which)));
  startFilm(plan(lines), which < 0 ? "my film" : FILM.stories[which].title);
}

function syncMusic() {
  if (!musicOn || !P.running) {
    music.pause();
    return;
  }
  // the music starts with the film: a bar lasts a shot, so each shot comes in on a bar
  if (music.duration) music.currentTime = filmTime() % music.duration;
  music.volume = 0.7;
  music.play().catch(() => {});
}

function fadeMusic() {
  if (music.paused) return;
  const from = music.volume, t0 = performance.now();
  const step = (now) => {
    const k = Math.min((now - t0) / 1200, 1);
    music.volume = from * (1 - k);
    if (k < 1 && !P.running) requestAnimationFrame(step);
    else if (!P.running) music.pause();
  };
  requestAnimationFrame(step);
}

// ---- writing one's own ----

function composer() {
  const pool = $("pool");
  for (const text of Q.texts) {
    const b = el("button", "pick", text);
    b.type = "button";
    b.addEventListener("click", () => {
      if (mine.length >= MOST) return;
      mine.push(text);
      drawMine();
    });
    pool.append(b);
  }
  $("play-mine").addEventListener("click", () => {
    show(-1);
    $("stage").scrollIntoView({ behavior: still ? "auto" : "smooth", block: "center" });
  });
  $("clear").addEventListener("click", () => {
    mine.length = 0;
    drawMine();
  });
}

function drawMine() {
  const list = $("mine");
  list.replaceChildren(...mine.map((text, n) => {
    const li = el("li", "", text), x = el("button", "", "×");
    x.type = "button";
    x.setAttribute("aria-label", `Take out “${text}”`);
    x.addEventListener("click", () => {
      mine.splice(n, 1);
      drawMine();
    });
    li.append(x);
    return li;
  }));
  $("play-mine").disabled = !mine.length;
  $("clear").disabled = !mine.length;
  $("mine-tab").disabled = !mine.length;
  document.querySelectorAll(".pick").forEach((b) => { b.disabled = mine.length >= MOST; });
}

function tabs() {
  const box = $("tabs");
  FILM.stories.forEach((s, n) => {
    const t = el("button", "tab", s.title);
    t.type = "button";
    t.setAttribute("role", "tab");
    t.dataset.story = String(n);
    if (s.lines.some((line) => Q.searched_as[Q.texts.indexOf(line)] !== line)) t.append(" ", el("small", "", "written in French"));
    t.addEventListener("click", () => show(n));
    box.append(t);
  });
  const own = el("button", "tab", "Your film");
  own.type = "button";
  own.id = "mine-tab";
  own.setAttribute("role", "tab");
  own.dataset.story = "-1";
  own.disabled = true;
  own.addEventListener("click", () => show(-1));
  box.append(own);
}

// ---- the search ----

function card(row, item, badge, extra) {
  const c = el("button", "card");
  c.type = "button";
  const shot = tile(row);
  if (badge) shot.append(el("span", "when", badge));
  const info = el("div", "meta");
  info.append(el("div", "title", item.file), el("div", "conf", extra));
  c.append(shot, info);
  return c;
}

function showLibrary(all = false) {
  // no description picked: the library itself, newest first, as a phone's gallery shows it
  const videos = meta.items.filter((x) => x.kind === "video").length;
  $("status").textContent = `${meta.items.length} files: ${videos} videos and ${meta.items.length - videos} photos. ` +
    "Pick a description above to find a moment in them.";
  const box = $("results");
  box.replaceChildren();
  const order = meta.items.map((_, f) => f).reverse();
  for (const f of all ? order : order.slice(0, GALLERY)) {
    const item = meta.items[f], rows = rowsOf[f];
    const c = card(rows[Math.floor(rows.length / 3)], item, item.kind === "video" ? clock(item.duration) : "photo",
                   item.taken.slice(0, 10));
    c.addEventListener("click", () => open(f, null));
    box.append(c);
  }
  if (!all) {
    const more = el("button", "more-files", `Show all ${meta.items.length} files`);
    more.type = "button";
    more.addEventListener("click", () => showLibrary(true));
    box.append(more);
  }
}

function showMoments(query, qs, embedded) {
  // embedded: the milliseconds the text server took to embed the description, or null for one embedded beforehand
  const t0 = performance.now();
  const box = $("results");
  box.replaceChildren();
  const by = embedded === null ? "" : `; embedded in ${embedded} ms by the demo's text server`;
  const hits = search(qs);
  const took = performance.now() - t0;
  const shown = hits.filter((h) => h.confidence >= HIDDEN_BELOW).slice(0, SHOWN);
  $("status").textContent = `${shown.length} moments for “${query}”: ${meta.count.toLocaleString("en")} frames and ` +
    `photos searched in ${Math.round(took)} ms, in your browser${by}.`;
  shown.forEach((h, i) => {
    const item = meta.items[h.item];
    const c = card(h.row, item, item.kind === "video" ? `${clock(h.start)} – ${clock(h.end)}` : "photo",
                   `${Math.round(100 * h.confidence)} % sure it matches`);
    if (h.confidence < 0.05) c.classList.add("unlikely");
    c.querySelector(".shot").append(el("span", "rank", String(i + 1)));
    c.addEventListener("click", () => open(h.item, h));
    box.append(c);
  });
}

function open(f, h) {
  // a video's moment plays its ten seconds and stops; a video picked in the gallery plays from the start
  pause();
  const item = meta.items[f];
  playing = h && item.kind === "video" ? h : null;
  const v = $("m-video"), img = $("m-photo");
  $("m-title").textContent = item.file;
  $("m-when").textContent = item.kind === "photo" ? item.taken.slice(0, 10) : h ? `${clock(h.start)} – ${clock(h.end)}`
    : clock(item.duration);
  $("m-again").hidden = !playing;
  const board = $("m-board");
  board.replaceChildren();
  if (playing) {
    for (const i of rowsOf[f]) {
      if (meta.times[i] < h.start || meta.times[i] >= h.end) continue;
      const shot = tile(i);
      if (i === h.row) shot.classList.add("best");
      shot.append(el("span", "", clock(meta.times[i])));
      shot.title = "Play from here";
      const t = meta.times[i];
      shot.addEventListener("click", () => { v.currentTime = t; v.play().catch(() => {}); });
      board.append(shot);
    }
  }
  const credit = $("m-caption");
  const title = Object.assign(el("a", "", item.title || "untitled"), { href: item.page, target: "_blank", rel: "noopener" });
  const licence = Object.assign(el("a", "", "CC BY 2.0"), { href: "https://creativecommons.org/licenses/by/2.0/",
                                                            target: "_blank", rel: "noopener" });
  credit.replaceChildren(playing ? "The frames of the moment as the search saw them; the outlined one matches best. " : "",
                         "“", title, `” by ${item.author} on Flickr, `, licence,
                         item.kind === "video" ? ", shown smaller and without its sound." : ".");
  v.hidden = item.kind !== "video";
  img.hidden = item.kind !== "photo";
  if (item.kind === "video") {
    v.src = MEDIA + item.file;
    v.onloadedmetadata = () => { v.currentTime = playing ? playing.start : 0; v.play().catch(() => {}); };
    v.onerror = () => { credit.prepend("This browser could not load the video. "); };
  } else {
    img.src = MEDIA + item.file;
    img.alt = item.title || item.file;
  }
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
  $("m-photo").removeAttribute("src");
});

function pick(query) {
  // a description from the list, one of the visitor's own (with a text server), or none: the library
  query = (query || "").trim().slice(0, 200);
  const known = Q.texts.slice(0, Q.suggestions).findIndex((s) => s === query);
  const asking = known >= 0 || (query && server);
  history.replaceState(null, "", asking ? `?q=${encodeURIComponent(query)}${keep()}` : location.pathname + keep("?"));
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.textContent === query));
  if (server) $("q").value = query;
  asked++;
  if (known >= 0) showMoments(query, queriesOf(known), null);
  else if (asking) ask(query, asked);
  else showLibrary();
}

function keep(first = "&") {
  // a server given in the address (a local one, to try the page with it) stays in the address
  const given = new URLSearchParams(location.search).get("server");
  return given ? `${first}server=${encodeURIComponent(given)}` : "";
}

async function ask(query, turn) {
  // the visitor's own description: embedded by the text server, then searched here like the others
  $("status").textContent = `Embedding “${query}” on the demo's text server…`;
  $("results").replaceChildren();
  let why = "";  // what the server said, when it said why
  try {
    const r = await fetch(`${server}/embed?q=${encodeURIComponent(query)}`, { signal: AbortSignal.timeout(60000) });
    const body = await r.json();
    if (turn !== asked) return;  // the visitor has moved on
    if (!r.ok) throw new Error((why = body.error || ""));
    const qs = models.map((m) => {
      const b64 = (body.vectors || {})[m.name];
      if (!b64) throw new Error((why = "it answered with other models"));
      const bin = atob(b64), bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const v = new Float32Array(bytes.buffer);
      if (v.length !== m.dim) throw new Error((why = "it answered with other models"));
      return v;
    });
    showMoments(query, qs, body.ms);
  } catch {
    if (turn !== asked) return;
    $("status").textContent = `The demo's text server did not answer${why ? `: ${why}` : ""}. ` +
      "The descriptions below work without it.";
  }
}

function textServer(config) {
  // the server the site was deployed with (config.json), or a local one given in the address (?server=)
  const given = new URLSearchParams(location.search).get("server");
  if (given && /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?\/?$/.test(given)) return given.replace(/\/$/, "");
  return (config.text_server || "").replace(/\/$/, "");
}

// ---- the model's numbers ----

function numbers(sum) {
  $("t-video").textContent = pct(sum.video);
  $("t-moment").textContent = pct(sum.moment);
  const panels = [
    { title: "Find the video among 979 personal videos", sub: "DiDeMo test: the right video in the first five",
      rows: [["whenframe", sum.video, "on a CPU", true],
             ["One image per video", sum.one_image, "its middle frame, SigLIP 2 B/16, on a CPU"],
             [sum.published.name, sum.published["R@5"], "trained on DiDeMo, on a GPU, as its authors report"]] },
    { title: "Find the moment among 979 videos", sub: "DiDeMo test: the right seconds in the first five, all videos at once",
      rows: [["whenframe", sum.moment, "on a CPU", true],
             ["One keyframe per shot", sum.keyframe, "as home video tools index videos, on a CPU"]] },
    { title: "Place the moment in a given video", sub: "Charades-STA, never trained on it: the first answer right",
      rows: [["whenframe", sum.charades, "on a CPU", true],
             ...sum.charades_others.map((x) => [x.name, x.R1, "as its authors report"])] },
  ];
  const top = Math.max(...panels.flatMap((p) => p.rows.map((r) => r[1])));
  const scale = Math.ceil(top * 10) / 10;  // the same scale in every panel
  const tip = $("tip"), figure = $("chart"), table = $("chart-table");
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", "", "test"), el("th", "", "system"), el("th", "", "share right"));
  table.append(head);
  $("panels").replaceChildren(...panels.map((p) => {
    const box = el("div", "panel");
    box.append(el("h3", "", p.title), el("p", "", p.sub));
    for (const [name, value, how, me] of [...p.rows].sort((a, b) => b[1] - a[1])) {
      const row = el("div", me ? "bar-row me" : "bar-row");
      row.tabIndex = 0;
      row.setAttribute("aria-label", `${name}: ${pct(value)}, ${how}`);
      const track = el("div", "track"), bar = el("div", "bar");
      bar.style.width = `calc((100% - 64px) * ${value / scale})`;
      track.append(bar, el("span", "v", pct(value)));
      row.append(el("div", "name", name), track);
      const showTip = () => {
        tip.replaceChildren(el("b", "", pct(value)), el("span", "", `${name}, ${how}`));
        tip.hidden = false;
        const f = figure.getBoundingClientRect(), r = bar.getBoundingClientRect();
        tip.style.left = `${Math.min(r.right - f.left + 8, f.width - tip.offsetWidth - 8)}px`;
        tip.style.top = `${r.top - f.top - tip.offsetHeight - 6}px`;
      };
      row.addEventListener("pointerenter", showTip);
      row.addEventListener("focus", showTip);
      row.addEventListener("pointerleave", () => { tip.hidden = true; });
      row.addEventListener("blur", () => { tip.hidden = true; });
      box.append(row);
      const tr = el("tr");
      tr.append(el("td", "", p.title), el("td", "", `${name} (${how})`), el("td", "", pct(value)));
      table.append(tr);
    }
    return box;
  }));
}

// ---- start ----

async function start() {
  const get = (name, kind) => fetch(DATA + name).then((r) => (kind === "json" ? r.json() : r.arrayBuffer()));
  const config = fetch("config.json").then((r) => r.json()).catch(() => ({}));
  $("status").textContent = "Loading the library's index…";
  [meta, Q, FILM] = await Promise.all([get("frames.json", "json"), get("queries.json", "json"), get("film.json", "json")]);
  models = await Promise.all(meta.models.map(async (m) => {
    const [f, o, q] = await Promise.all([get(`frames.${m.name}.bin`), get(`offsets.${m.name}.bin`),
                                         get(`queries.${m.name}.bin`)]);
    return { ...m, frames: half(f), offsets: new Float32Array(o), queries: new Float32Array(q) };
  }));
  rowsOf = meta.items.map(() => []);
  meta.item.forEach((f, i) => rowsOf[f].push(i));
  server = textServer(await config);
  if (server) {
    $("ask").hidden = false;
    $("privacy").hidden = false;
    $("how").textContent = "Describe a moment in your own words, or pick one below:";
    fetch(`${server}/health`).catch(() => {});  // wakes a server that sleeps while the visitor reads
  }
  const videos = meta.items.filter((x) => x.kind === "video");
  const minutes = Math.round(videos.reduce((t, x) => t + x.duration, 0) / 60);
  const years = meta.items.map((x) => x.taken.slice(0, 4)).sort();
  $("what").textContent = `Someone's phone, ${years[0]}–${years[years.length - 1]}: ${videos.length} videos ` +
    `(${minutes} minutes) and ${meta.items.length - videos.length} photos`;
  $("n-files").textContent = String(meta.items.length);
  for (const text of Q.texts.slice(0, Q.suggestions)) {
    const chip = el("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => pick(chip.classList.contains("on") ? null : text));
    $("chips").append(chip);
  }
  setupLayers();
  tabs();
  composer();
  fetch(DATA + "summary.json").then((r) => r.json()).then(numbers).catch(() => {});
  show(0);
  pick(new URLSearchParams(location.search).get("q"));
}

$("all").addEventListener("click", () => pick(null));
$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  pick($("q").value);
});
$("play").addEventListener("click", () => (P.running ? pause() : play()));
$("replay").addEventListener("click", replay);
$("big").addEventListener("click", () => (P.ended ? replay() : play()));
$("sound").addEventListener("click", () => {
  musicOn = !musicOn;
  $("sound").textContent = `♪ Music: ${musicOn ? "on" : "off"}`;
  $("sound").setAttribute("aria-pressed", String(musicOn));
  if (musicOn && music.preload === "none") music.preload = "auto";
  syncMusic();
});
// the film pauses while it is out of sight, and goes on when it is back
new IntersectionObserver(([e]) => {
  if (!e.isIntersecting && P.running) {
    pause();
    P.auto = true;
  } else if (e.isIntersecting && P.auto) {
    P.auto = false;
    play();
  }
}, { threshold: 0.25 }).observe($("stage"));
document.addEventListener("visibilitychange", () => {
  if (document.hidden && P.running) {
    pause();
    P.auto = true;
  } else if (!document.hidden && P.auto) {
    P.auto = false;
    play();
  }
});
start();
