// whenframe in the browser: the frames of three films, embedded once with SigLIP 2 (scripts/site.py), searched here
// the way the tool searches: cosine minus the query-bank offset, 10-second windows around the best frames.
const DATA = "data/";
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";
const MODEL = "onnx-community/siglip2-base-patch16-224-ONNX";
const $ = (id) => document.getElementById(id);

let meta, frames, offsets, suggestions, vectors;
let encode = null;

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
const unit = (v) => {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n);
  return v.map((x) => x / n);
};

function tile(row) {
  // a div showing frame `row` from the thumbnail sheets
  const div = el("div", "shot");
  const k = row % meta.per_sheet, rows = meta.per_sheet / meta.cols;
  div.style.backgroundImage = `url(${DATA}sheets/${Math.floor(row / meta.per_sheet)}.webp)`;
  div.style.backgroundPosition = `${((k % meta.cols) / (meta.cols - 1)) * 100}% ${(Math.floor(k / meta.cols) / (rows - 1)) * 100}%`;
  return div;
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

function render(query, vector, ms) {
  const t0 = performance.now();
  const hits = search(vector);
  const took = ms + performance.now() - t0;
  $("status").textContent = `${hits.length} moments in ${Math.round(took)} ms, for “${query}”`;
  const box = $("results");
  box.replaceChildren();
  hits.forEach((h, i) => {
    const card = el("button", h.confidence < 0.05 ? "card unlikely" : "card");
    card.type = "button";
    const shot = tile(h.row);
    shot.append(el("span", "rank", String(i + 1)), el("span", "when", `${clock(h.start)} – ${clock(h.end)}`));
    const info = el("div", "meta");
    info.append(el("div", "title", meta.films[h.film].title),
                el("div", "conf", `${Math.round(100 * h.confidence)} % sure the best frame matches`));
    card.append(shot, info);
    card.addEventListener("click", () => storyboard(h));
    box.append(card);
  });
}

function storyboard(h) {
  const film = meta.films[h.film];
  $("m-title").textContent = film.title;
  $("m-when").textContent = `${clock(h.start)} – ${clock(h.end)}`;
  $("m-link").href = film.url;
  const board = $("m-board");
  board.replaceChildren();
  for (let i = 0; i < meta.count; i++) {
    if (meta.film[i] !== h.film || meta.times[i] < h.start || meta.times[i] >= h.end) continue;
    const shot = tile(i);
    if (i === h.row) shot.classList.add("best");
    shot.append(el("span", "", clock(meta.times[i])));
    board.append(shot);
  }
  $("moment").showModal();
}

async function loadEncoder() {
  if (encode) return encode;
  $("load").disabled = true;
  const progress = $("progress");
  progress.hidden = false;
  const { AutoTokenizer, SiglipTextModel, env } = await import(TRANSFORMERS);
  env.allowLocalModels = false;
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
  const model = await SiglipTextModel.from_pretrained(MODEL, {
    dtype: "q8",
    progress_callback: (p) => { if (p.status === "progress" && p.file.endsWith(".onnx")) progress.value = p.progress; },
  });
  encode = async (text) => {
    const inputs = tokenizer([text.toLowerCase()], { padding: "max_length", max_length: meta.text_length, truncation: true });
    const { pooler_output } = await model(inputs);
    return unit(Float32Array.from(pooler_output.data));
  };
  $("model").classList.remove("show");
  return encode;
}

async function run(query) {
  query = query.trim();
  history.replaceState(null, "", query ? `?q=${encodeURIComponent(query)}` : location.pathname);
  if (!query) return;
  const known = suggestions.findIndex((s) => s.toLowerCase() === query.toLowerCase());
  if (known >= 0) {
    render(query, vectors.subarray(known * meta.dim, (known + 1) * meta.dim), 0);
    return;
  }
  if (!encode) {
    $("model").classList.add("show");
    $("status").textContent = "";
    return;
  }
  const t0 = performance.now();
  render(query, await encode(query), performance.now() - t0);
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
  $("what").textContent = `Try it on three open movies: ${meta.count.toLocaleString("en")} frames, ${minutes} minutes`;
  for (const text of suggestions) {
    const chip = el("button", "chip", text);
    chip.type = "button";
    chip.addEventListener("click", () => { $("q").value = text; run(text); });
    $("chips").append(chip);
  }
  fetch(DATA + "summary.json").then((r) => r.json()).then((sum) => {
    $("n-default").textContent = `${(100 * sum.default).toFixed(1)} %`;
    $("n-videos").textContent = sum.videos.toLocaleString("en");
    $("n-shots").textContent = `${(100 * sum.shots).toFixed(1)} %`;
  }).catch(() => {});
  const initial = new URLSearchParams(location.search).get("q") || suggestions[0];
  $("q").value = initial;
  run(initial);
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); run($("q").value); });
$("load").addEventListener("click", async () => {
  try {
    await loadEncoder();
    run($("q").value);
  } catch (err) {
    $("model-text").textContent = `The model could not be loaded here (${err.message}). The suggested searches still work.`;
  }
});
$("m-close").addEventListener("click", () => $("moment").close());
start();
