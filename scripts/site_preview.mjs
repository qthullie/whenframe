// Frames of the site in use, for the README's animation: node scripts/site_preview.mjs URL OUTDIR [light|dark]
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [url, out, scheme = "light"] = process.argv.slice(2);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 760 }, deviceScaleFactor: 2, colorScheme: scheme });
let n = 0;
const snap = async (repeat = 1) => {
  const file = path.join(out, `frame-${String(n).padStart(3, "0")}.png`);
  await page.screenshot({ path: file });
  for (let i = 1; i < repeat; i++) console.log(file);  // the assembler repeats a frame once per line
  console.log(file);
  n += 1;
};

await page.goto(url);
await page.waitForFunction(() => [...document.querySelectorAll(".card .shot")].length > 0);
await page.evaluate(() => window.scrollTo(0, document.querySelector(".demo").offsetTop - 12));
await page.waitForTimeout(300);
await snap(8);  // the library, before any description is picked
await page.click('.chip >> text="a geyser shooting steam into the sky"');
await page.waitForFunction(() => document.querySelector("#status").textContent.includes("geyser"));
await page.waitForTimeout(400);
await snap(10);
await page.click(".card");  // the first moment plays its ten seconds
await page.waitForFunction(() => { const v = document.querySelector("#m-video"); return v.currentTime > 0 && !v.paused; });
for (let i = 0; i < 18; i++) {
  await snap();
  await page.waitForTimeout(110);
}
await page.click("#m-close");
await page.click('.chip >> text="a pipe band marching down the street"');
await page.waitForFunction(() => document.querySelector("#status").textContent.includes("pipe band"));
await page.waitForTimeout(400);
await snap(12);
await browser.close();
