// The small model's text tower in a browser (docs/PROTOCOL.md, thirteenth round; reported beside the test, deciding
// nothing): site/clip.js in headless Chromium, onnxruntime-web's WebAssembly backend on one thread, as a page of
// GitHub Pages runs it (no cross-origin isolation, so no threads). For each description of scripts/small.py web's
// web_check.json: the token ids must be the tool's, the embedding at a cosine of at least 0.999 to the tool's; the
// time to embed it, after one warm-up, and the median. Writes web_browser.json beside them.
//     node scripts/web_text.mjs FOLDER ORT_DIST    # FOLDER: small.py web's; ORT_DIST: onnxruntime-web's dist/
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createServer } from "node:http";
import { join, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [folder, dist] = process.argv.slice(2);
const site = join(dirname(fileURLToPath(import.meta.url)), "..", "site");
const MIN_COSINE = 0.999;

const page_ = `<!doctype html><meta charset="utf-8"><title>text tower</title><script type="module">
import * as ort from "/ort/ort.wasm.min.mjs";
import { TextTower } from "/clip.js";
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "/ort/";
const check = await (await fetch("/check.json")).json();
const t0 = performance.now();
const tower = await TextTower.load(ort, "/model.onnx", "/merges.txt");
const load = performance.now() - t0;
await tower.embed("a warm-up");
const results = [];
for (const text of check.texts) {
  const ids = Array.from(tower.tokenizer.encode(text), Number);
  const s = performance.now();
  const v = await tower.embed(text);
  results.push({ ids, v: Array.from(v), ms: performance.now() - s });
}
window.done = { load, results, agent: navigator.userAgent, isolated: crossOriginIsolated };
</script>`;

const files = {
  "/clip.js": join(site, "clip.js"),
  "/model.onnx": join(folder, "onnx", "text_model_web.onnx"),
  "/merges.txt": join(folder, "merges.txt"),
  "/check.json": join(folder, "web_check.json"),
};
const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const server = createServer((req, res) => {
  const url = req.url.split("?")[0];
  if (url === "/") {
    res.writeHead(200, { "content-type": "text/html" });
    return res.end(page_);
  }
  const path = files[url] ?? (url.startsWith("/ort/") ? join(dist, url.slice(5)) : null);
  if (!path || !existsSync(path)) {
    res.writeHead(404);
    return res.end();
  }
  res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" });
  res.end(readFileSync(path));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const url = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
await page.goto(url);
await page.waitForFunction(() => window.done || null, null, { timeout: 600000 });
const done = await page.evaluate(() => window.done);
await browser.close();
server.close();

const check = JSON.parse(readFileSync(files["/check.json"], "utf8"));
const cos = (a, b) => {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return ab / Math.sqrt(aa * bb);
};
const differ = [], cosines = [];
done.results.forEach((r, i) => {
  if (r.ids.join() !== check.ids[i].join()) differ.push({ text: check.texts[i], page: r.ids, tool: check.ids[i] });
  cosines.push(cos(r.v, check.tool[i]));
});
const ms = done.results.map((r) => r.ms).sort((a, b) => a - b);
const median = ms.length % 2 ? ms[(ms.length - 1) / 2] : (ms[ms.length / 2 - 1] + ms[ms.length / 2]) / 2;
const result = {
  texts: check.texts.length, ids_equal: differ.length === 0, ids_differ: differ.slice(0, 5),
  cosine_to_tool_min: Math.min(...cosines), median_ms: median, max_ms: ms[ms.length - 1], load_ms: done.load,
  threads: 1, cross_origin_isolated: done.isolated, browser: done.agent, errors,
};
result.passed = result.ids_equal && result.cosine_to_tool_min >= MIN_COSINE && errors.length === 0;
writeFileSync(join(folder, "web_browser.json"), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result, null, 1));
process.exit(result.passed ? 0 : 1);
