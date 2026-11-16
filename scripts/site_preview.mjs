// Frames of the site in use, for the README's animation: node scripts/site_preview.mjs URL OUTDIR
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [url, out] = process.argv.slice(2);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 760 }, deviceScaleFactor: 1 });
let n = 0;
const snap = async (repeat = 1) => {
  const file = path.join(out, `frame-${String(n).padStart(3, "0")}.png`);
  await page.screenshot({ path: file });
  for (let i = 1; i < repeat; i++) console.log(file);  // the assembler repeats a frame once per line
  console.log(file);
  n += 1;
};
const settle = () => page.waitForFunction(() => [...document.querySelectorAll(".card .shot")].length > 0);

await page.goto(`${url}/?q=${encodeURIComponent("a girl hugging a baby dragon")}`);
await settle();
await page.evaluate(() => {
  document.querySelector("#q").value = "";
  document.querySelector("#results").replaceChildren();
  document.querySelector("#status").textContent = "";
  window.scrollTo(0, 120);
});
await snap(4);
const query = "a girl hugging a baby dragon";
for (let i = 1; i <= query.length; i += 2) {
  await page.fill("#q", query.slice(0, i));
  await snap();
}
await page.fill("#q", query);
await page.press("#q", "Enter");
await page.waitForFunction(() => document.querySelector("#status").textContent.includes("dragon"));
await page.waitForTimeout(400);
await snap(10);
await page.click(".card");
await page.waitForSelector("#m-board .shot");
await page.waitForTimeout(400);
await snap(14);
await page.click("#m-close");
await page.click(".chip");
await page.waitForFunction(() => document.querySelector("#status").textContent.includes("bridge"));
await page.waitForTimeout(400);
await snap(12);
await browser.close();
