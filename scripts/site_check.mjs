// The demo site searches and plans films as the tool does: for each suggestion, the moments the page shows, in order,
// against those whenframe's own search found when the data was built; for each film of the page's opening, the
// shots the page plans with Déclic and with one image per video, against those whenframe/film.py and scripts/site.py
// planned (site/data/expected.json, scripts/site.py).
//     python -m http.server --directory site 8123 &  node scripts/site_check.mjs http://127.0.0.1:8123
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [url] = process.argv.slice(2);
const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(url);
await page.waitForFunction(() => document.querySelectorAll(".chip").length > 0 && document.querySelectorAll("#strip button").length > 0);
const expected = await (await fetch(new URL("data/expected.json", url))).json();
let failed = 0;
for (const [query, hits] of Object.entries(expected.searches)) {
  await page.click(`.chip >> text="${query}"`);
  await page.waitForFunction((q) => document.querySelector("#status").textContent.includes(q), query);
  const shown = await page.$$eval(".card", (cards) => cards.map((c) => [c.querySelector(".title").textContent,
                                                                         c.querySelector(".when").textContent]));
  const want = hits.filter((h) => h.confidence >= 0.005)
    .map((h) => [h.file, h.end > 0 ? `${clock(h.start)} – ${clock(h.end)}` : "photo"]);
  const n = Math.min(want.length, shown.length);
  const same = n > 0 && want.slice(0, n).every((w, i) => w[0] === shown[i][0] && w[1] === shown[i][1]);
  if (!same) failed++;
  console.log(`${same ? "same" : "DIFFERENT"}  ${query}: ${n} moments compared`);
  if (!same) console.log("  tool:", JSON.stringify(want.slice(0, n)), "\n  page:", JSON.stringify(shown.slice(0, n)));
  await page.click(`.chip >> text="${query}"`);  // back to the library
}
for (const [title, plans] of Object.entries(expected.films)) {
  for (const [which, shots] of Object.entries(plans)) {
    await page.click(`.controls [data-declic="${which === "declic" ? 1 : 0}"]`);
    await page.click(`.tab:has-text("${title}")`);
    const shown = await page.$$eval("#strip button", (b) => b.map((x) => [x.dataset.file, Number(x.dataset.start),
                                                                           Number(x.dataset.length)]));
    const same = shown.length === shots.length && shots.every((s, i) => s.file === shown[i][0] &&
      Math.abs(s.start - shown[i][1]) < 0.02 && Math.abs(s.length - shown[i][2]) < 0.02);
    if (!same) failed++;
    console.log(`${same ? "same" : "DIFFERENT"}  film “${title}”, ${which}: ${shots.length} shots compared`);
    if (!same) {
      console.log("  tool:", JSON.stringify(shots.map((s) => [s.file, s.start, s.length])),
                  "\n  page:", JSON.stringify(shown));
    }
  }
}
await browser.close();
if (errors.length) console.log("page errors:", errors);
process.exit(failed || errors.length ? 1 : 0);
