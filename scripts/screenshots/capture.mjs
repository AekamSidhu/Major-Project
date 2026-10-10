// Renders the screenshot harness in headless Chromium (Edge or Chrome) and
// writes docs/screenshots/*.png. Drives the browser over the DevTools
// protocol so each capture waits until the page reports it is ready. Uses a
// throwaway browser profile; needs Node 22+ (built-in WebSocket).
//
//   node scripts/screenshots/capture.mjs
//   OPENMEET_BROWSER=/path/to/chrome node scripts/screenshots/capture.mjs

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { build, preview } from "vite";

const here = fileURLToPath(new URL(".", import.meta.url));
const repo = resolve(here, "../..");
const out = join(repo, "docs", "screenshots");
const PORT = 4179;
const DEBUG_PORT = 9339;

const SCENES = [
  { name: "overlay-candidate", width: 1100, height: 860 },
  { name: "overlay-interviewer", width: 1100, height: 640 },
  { name: "main-live", width: 1200, height: 800 },
  { name: "main-privacy", width: 1200, height: 800 },
];

function findBrowser() {
  const candidates = [
    process.env.OPENMEET_BROWSER,
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ];
  const found = candidates.find((p) => p && existsSync(p));
  if (!found) throw new Error("No Chromium-based browser found; set OPENMEET_BROWSER.");
  return found;
}

/** Minimal DevTools protocol client for one page target. */
async function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  await new Promise((ok, fail) => {
    ws.addEventListener("open", ok, { once: true });
    ws.addEventListener("error", fail, { once: true });
  });
  let seq = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(String(ev.data));
    const p = msg.id !== undefined ? pending.get(msg.id) : undefined;
    if (!p) return;
    pending.delete(msg.id);
    if (msg.error) p.fail(new Error(msg.error.message));
    else p.ok(msg.result);
  });
  return {
    send(method, params = {}) {
      const id = ++seq;
      ws.send(JSON.stringify({ id, method, params }));
      return new Promise((ok, fail) => pending.set(id, { ok, fail }));
    },
    close() {
      ws.close();
    },
  };
}

async function pageTarget() {
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
      const page = (await res.json()).find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch {
      // browser still starting
    }
    await sleep(100);
  }
  throw new Error("The browser did not open a debuggable page.");
}

const browserPath = findBrowser();
const dist = mkdtempSync(join(tmpdir(), "openmeet-shots-"));
const profile = mkdtempSync(join(tmpdir(), "openmeet-shots-profile-"));
const shared = {
  configFile: false,
  root: here,
  base: "./",
  logLevel: "warn",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": join(repo, "src") } },
};

await build({ ...shared, build: { outDir: dist, emptyOutDir: true } });
const server = await preview({
  ...shared,
  build: { outDir: dist },
  preview: { host: "127.0.0.1", port: PORT, strictPort: true },
});
const browser = spawn(
  browserPath,
  [
    "--headless=new",
    "--no-first-run",
    "--disable-extensions",
    "--hide-scrollbars",
    `--remote-debugging-port=${DEBUG_PORT}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { stdio: "ignore" },
);

mkdirSync(out, { recursive: true });
let cdp;
try {
  cdp = await connect(await pageTarget());
  await cdp.send("Page.enable");
  for (const s of SCENES) {
    await cdp.send("Emulation.setDeviceMetricsOverride", {
      width: s.width,
      height: s.height,
      deviceScaleFactor: 2,
      mobile: false,
    });
    await cdp.send("Page.navigate", { url: `http://127.0.0.1:${PORT}/index.html?scene=${s.name}` });
    let ready = false;
    for (let i = 0; i < 150 && !ready; i++) {
      await sleep(100);
      const r = await cdp.send("Runtime.evaluate", { expression: "window.__shotReady === true" });
      ready = r.result.value === true;
    }
    if (!ready) throw new Error(`${s.name}: the page never became ready`);
    const shot = await cdp.send("Page.captureScreenshot", { format: "png" });
    const file = join(out, `${s.name}.png`);
    writeFileSync(file, Buffer.from(shot.data, "base64"));
    console.warn(`wrote ${file}`);
  }
} finally {
  cdp?.close();
  browser.kill();
  server.httpServer.close();
  await sleep(500);
  rmSync(dist, { recursive: true, force: true });
  rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
