// Frames of the site's opening film as it plays, for the README's animation:
//     node scripts/site_preview.mjs URL OUTDIR [light|dark]
// Prints one line per frame, its file and how long it stays (milliseconds, the time until the next one was taken).
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [url, out, scheme = "light"] = process.argv.slice(2);

const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 820 }, deviceScaleFactor: 2, colorScheme: scheme });
await page.goto(url);
await page.waitForFunction(() => document.querySelector("#line").textContent.length > 0);
// the story tabs, the film and its controls, in view
await page.evaluate(() => window.scrollTo(0, document.querySelector("#tabs").getBoundingClientRect().top + scrollY - 12));
const clip = await page.evaluate(() => {
  const a = document.querySelector("#tabs").getBoundingClientRect(), b = document.querySelector(".controls").getBoundingClientRect();
  return { x: a.left - 12, y: a.top - 12, width: a.width + 24, height: b.bottom - a.top + 24 };
});
await page.click("#replay");
const frames = [], t0 = Date.now();
for (let n = 0; ; n++) {
  const file = path.join(out, `frame-${String(n).padStart(3, "0")}.jpg`);
  await page.screenshot({ path: file, clip, type: "jpeg", quality: 90 });  // faster than PNG: more frames a second
  frames.push([file, Date.now()]);
  const ended = await page.evaluate(() => document.querySelector("#curtain").classList.contains("on"));
  if (ended || Date.now() - t0 > 30000) break;
}
frames.forEach(([file, at], i) => console.log(file, (i + 1 < frames.length ? frames[i + 1][1] : at + 1200) - at));
await browser.close();
