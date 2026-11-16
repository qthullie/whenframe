// Screenshots of the search page for the docs, light and dark: node scripts/demo.mjs URL OUTDIR QUERY...
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const [url, out, ...queries] = process.argv.slice(2);

const browser = await chromium.launch();
for (const scheme of ["light", "dark"]) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 672 }, deviceScaleFactor: 2, colorScheme: scheme });
  for (const [i, query] of queries.entries()) {
    await page.goto(`${url}/?q=${encodeURIComponent(query)}`);
    await page.waitForSelector(".card img");
    await page.waitForFunction(() => [...document.querySelectorAll(".card img")].every((img) => img.complete));
    const suffix = scheme === "dark" ? "-dark" : "";
    await page.screenshot({ path: path.join(out, `demo-${i + 1}${suffix}.png`) });
    console.log(`demo-${i + 1}${suffix}.png: ${query}`);
  }
  await page.close();
}
await browser.close();
