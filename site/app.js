// whenframe in the browser: someone's phone library (library.json), indexed once by the tool (scripts/site.py) and
// searched here the way the tool searches it: each model's cosine minus its query-bank offset, the two models' scores
// standardised over the library and added (PE's with its weight), put back on SigLIP 2's scale; ten-second windows
// around a video's best frames, a photo as its own moment. The suggestions were embedded by the tool; a description of
// the visitor's own is embedded by the demo's text server (demo-server/, config.json), with the same text models.
const DATA = "data/", MEDIA = "media/";
const HIDDEN_BELOW = 0.005;  // moments shown as "0 %" sure say nothing: left out, as the tool leaves them out
const SHORTLIST = 64;  // files examined in detail, as the tool's search does
const SHOWN = 24, GALLERY = 30;
const $ = (id) => document.getElementById(id);

let meta, models, suggestions, rowsOf;
let server = "";  // the text server's address, when there is one
let playing = null;  // the moment being played, which stops at its end; null for a whole video
let asked = 0;  // the latest description sent to the server: an older answer arriving late is dropped

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
    const sure = (top) => 1 / (1 + Math.exp(-(meta.scale * top + meta.bias)));
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
  const hits = search(qs);
  const took = performance.now() - t0;
  const shown = hits.filter((h) => h.confidence >= HIDDEN_BELOW).slice(0, SHOWN);
  $("status").textContent = `${shown.length} moments for “${query}”: ${meta.count.toLocaleString("en")} frames and ` +
    `photos searched in ${Math.round(took)} ms, in your browser` +
    (embedded === null ? "" : `; embedded in ${embedded} ms by the demo's text server`) + ".";
  const box = $("results");
  box.replaceChildren();
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
  const known = suggestions.findIndex((s) => s === query);
  const searched = known >= 0 || (query && server);
  history.replaceState(null, "", searched ? `?q=${encodeURIComponent(query)}${keep()}` : location.pathname + keep("?"));
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.textContent === query));
  if (server) $("q").value = query;
  asked++;
  if (known >= 0) showMoments(query, models.map((m) => m.queries.subarray(known * m.dim, (known + 1) * m.dim)), null);
  else if (searched) ask(query, asked);
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

async function start() {
  const get = (name, kind) => fetch(DATA + name).then((r) => (kind === "json" ? r.json() : r.arrayBuffer()));
  const config = fetch("config.json").then((r) => r.json()).catch(() => ({}));
  $("status").textContent = "Loading the library's index…";
  [meta, suggestions] = await Promise.all([get("frames.json", "json"), get("queries.json", "json")]);
  models = await Promise.all(meta.models.map(async (m) => {
    const [f, o, q] = await Promise.all([get(`frames.${m.name}.bin`), get(`offsets.${m.name}.bin`),
                                         get(`queries.${m.name}.bin`)]);
    const half = new Uint16Array(f), frames = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) frames[i] = F16[half[i]];
    return { ...m, frames, offsets: new Float32Array(o), queries: new Float32Array(q) };
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
  for (const text of suggestions) {
    const chip = el("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => pick(chip.classList.contains("on") ? null : text));
    $("chips").append(chip);
  }
  fetch(DATA + "summary.json").then((r) => r.json()).then((sum) => {
    $("n-video").textContent = `${(100 * sum.video).toFixed(1)} %`;
    $("n-moment").textContent = `${(100 * sum.moment).toFixed(1)} %`;
  }).catch(() => {});
  pick(new URLSearchParams(location.search).get("q"));
}

$("all").addEventListener("click", () => pick(null));
$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  pick($("q").value);
});
start();
