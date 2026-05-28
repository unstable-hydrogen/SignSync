import { chromium } from "@playwright/test";
import path from "path";

const browser = await chromium.launch({
  args: ["--no-sandbox", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
});
const context = await browser.newContext({
  permissions: ["camera"],
  viewport: { width: 1280, height: 800 },
});
const page = await context.newPage();

const errors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});
page.on("pageerror", (err) => errors.push(err.message));

await page.goto("http://localhost:5173");
// Wait for the page to paint something, then screenshot whatever is there
await page.waitForTimeout(8000);
await page.screenshot({ path: "screenshot.png", fullPage: true });

console.log("Screenshot saved.");
if (errors.length) {
  console.log("Console errors:");
  errors.forEach((e) => console.log(" -", e));
} else {
  console.log("No console errors.");
}

await browser.close();
