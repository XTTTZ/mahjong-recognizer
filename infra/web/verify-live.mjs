/** Verify deployed phone flow. GPU work requires --recognize or --warmup. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from '@playwright/test';

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
};
const site = option('--url', 'https://mahjong.avalonxty.site');
const image = option('--image');
const recognize = args.includes('--recognize');
const warmup = recognize || args.includes('--warmup');
const waitForWarmup = args.includes('--wait-warmup');
const proxy = option('--proxy');
const expectedModel = option('--expect-model');
const expectedBackend = option('--expect-backend');
const out = path.resolve(option('--out', 'artifacts/web/live'));
if (recognize && !image) throw new Error('--recognize requires a local --image');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(proxy ? { proxy: { server: proxy } } : {}) });
const report = { site, observedAt: new Date().toISOString(), gpuRequested: recognize || warmup,
  recognitionRequested: recognize, pageWarmupAllowed: warmup, proxyUsed: Boolean(proxy),
  screens: [], browserErrors: [], consoleErrors: [], externalScripts: [], cspViolations: [], pageChecks: [], sourceOffers: [] };
try {
  const context = await browser.newContext({ isMobile: true, hasTouch: true, deviceScaleFactor: 1 });
  context.setDefaultTimeout(20_000);
  context.setDefaultNavigationTimeout(30_000);
  if (!warmup) await context.route('**/api/warmup', route => route.fulfill({ json: { ready: false, skippedByReadOnlyProbe: true } }));
  const healthResponse = await context.request.get(`${site}/api/health`);
  assert.equal(healthResponse.status(), 200);
  report.health = await healthResponse.json();
  assert.equal(report.health.inferenceProbed, false);
  if (recognize) assert.equal(report.health.configured, true, 'Gateway is not configured');

  for (const width of [320, 390, 430]) {
    const page = await context.newPage();
    await page.setViewportSize({ width, height: 844 });
    page.on('pageerror', error => report.browserErrors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error') report.consoleErrors.push({ width, message: message.text().slice(0, 1000),
        source: message.location().url });
    });
    await page.addInitScript(() => {
      window.__mahjongCspViolations = [];
      document.addEventListener('securitypolicyviolation', event => {
        window.__mahjongCspViolations.push({ blockedURI: event.blockedURI, directive: event.effectiveDirective });
      });
    });
    const browserHealth = page.waitForResponse(response => new URL(response.url()).pathname === '/api/health');
    const warmStarted = Date.now();
    const pageWarmup = warmup && waitForWarmup && width === 320
      ? page.waitForResponse(response => new URL(response.url()).pathname === '/api/warmup', { timeout: 70_000 }) : null;
    const navigation = await page.goto(site);
    assert.equal(navigation.status(), 200);
    assert.match(navigation.headers()['content-security-policy'] ?? '', /frame-ancestors 'none'/);
    report.pageChecks.push({ width, noTransform: /\bno-transform\b/.test(navigation.headers()['cache-control'] ?? '') });
    await page.getByRole('button', { name: '拍一张', exact: true }).waitFor();
    assert.equal((await browserHealth).status(), 200);
    if (pageWarmup) {
      const response = await pageWarmup;
      report.pageWarmup = { status: response.status(), elapsedMs: Date.now() - warmStarted, ...await response.json() };
    }
    await page.getByText('试用模型 · 结果请逐张核对', { exact: true }).waitFor();
    if (!report.health.configured) {
      await page.getByText('识别服务准备中，暂可拍照、选图和裁剪。', { exact: true }).waitFor();
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    const homeFile = `home-${width}.png`;
    await page.screenshot({ path: path.join(out, homeFile), fullPage: true });
    report.screens.push(homeFile);
    if (width === 390 && image) {
      await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles(path.resolve(image));
      await page.getByRole('button', { name: '选取要识别的牌面区域' }).waitFor();
      if (recognize) {
        let calls = 0;
        page.on('request', request => { if (new URL(request.url()).pathname === '/api/recognize') calls++; });
        const started = Date.now();
        const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/recognize', { timeout: 105_000 });
        await page.getByRole('button', { name: '开始识别', exact: true }).click();
        const response = await responsePromise;
        report.recognitionHttpStatus = response.status();
        assert.equal(response.status(), 200, `Recognition returned ${response.status()}: ${await response.text()}`);
        const value = await response.json();
        assert.equal(typeof value.modelVersion, 'string');
        if (expectedModel) assert.equal(value.modelVersion, expectedModel, 'The gateway must serve the intended released model');
        if (expectedBackend) assert.equal(value.runtimeBackend, expectedBackend, 'The gateway must serve the verified runtime');
        assert.equal(Array.isArray(value.tiles), true);
        assert.equal(typeof value.image.width, 'number');
        await page.getByRole('region', { name: '识别结果' }).waitFor();
        assert.equal(calls, 1, 'A single click must send a single recognition request');
        report.recognition = { requestId: value.requestId, modelVersion: value.modelVersion,
          runtimeBackend: value.runtimeBackend, tiles: value.tiles.length,
          image: value.image, quality: value.quality, serviceMs: value.timings?.totalMs,
          elapsedMs: Date.now() - started, automaticRequests: calls };
        const originalScreen = 'recognition-original-390.png';
        await page.screenshot({ path: path.join(out, originalScreen), fullPage: true });
        report.screens.push(originalScreen);
        if (value.tiles.length) {
          const shownLabels = await page.locator('.tile-label').allTextContents();
          assert.equal(shownLabels.every(label => !/^(?:[1-9][mps]|[1-8]h|[1-7]z|back)$/.test(label)), true,
            'Known tile classes should display their Chinese names');
          report.chineseLabelsVerified = true;
          const previousLabel = shownLabels[0];
          const replacement = previousLabel === '一万' ? '二万' : '一万';
          await page.locator('.tile-button').first().click();
          await page.getByRole('button', { name: replacement, exact: true }).click();
          assert.equal(await page.locator('.tile-button.corrected').count(), 1);
          assert.equal(await page.locator('.tile-button.corrected .tile-label').textContent(), replacement);
          report.manualCorrectionVerified = true;
          report.manualCorrection = { previousLabel, replacement };
        }
      }
      const file = recognize ? 'recognition-390.png' : 'upload-390.png';
      await page.screenshot({ path: path.join(out, file), fullPage: true });
      report.screens.push(file);
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    const scripts = await page.locator('script[src]').evaluateAll(nodes => nodes.map(node => node.src));
    report.externalScripts.push(...scripts.filter(src => new URL(src).origin !== new URL(site).origin));
    report.cspViolations.push(...await page.evaluate(() => window.__mahjongCspViolations));
    report.sourceOffers.push({ width,
      source: await page.locator('footer a[href="https://github.com/XTTTZ/mahjong-recognizer"]').count() === 1,
      model: await page.locator('footer a[href="https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027"]').count() === 1 });
    await page.close();
  }
  assert.deepEqual(report.browserErrors, []);
  assert.deepEqual(report.consoleErrors, []);
  assert.deepEqual(report.cspViolations, []);
  assert.deepEqual(report.externalScripts, []);
  assert.equal(report.pageChecks.every(page => page.noTransform), true, 'Every page must opt out of script injection');
  assert.equal(report.sourceOffers.every(page => page.source && page.model), true, 'Every page must show the source/model offer');
  report.passed = true;
} catch (error) {
  report.passed = false;
  report.error = error.message;
  process.exitCode = 1;
} finally {
  await browser.close();
  await fs.writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
}
