/** Live assets + local scoring verification. Recognition/warmup are ALWAYS MOCKED; no GPU calls. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium, expect } from '@playwright/test';

const args = process.argv.slice(2);
const option = (name, fallback) => { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; };
const site = option('--url', 'https://mahjong.avalonxty.site');
const out = path.resolve(option('--out', 'artifacts/web/live-scoring-mocked'));
const proxy = option('--proxy');
const expectedVersion = option('--expect-version', '2026-10-29-scoring-warmup');
const ordinary = ['1m', '1m', '1m', '2m', '3m', '4m', '4p', '5p', '6p', '7s', '8s', '9s', '5z', '5z'];
await fs.mkdir(out, { recursive: true });
assert.equal(await fs.stat(path.join(out, 'report.json')).then(() => true).catch(() => false), false,
  'Choose a fresh output directory to preserve previous measurements');
const report = { site, observedAt: new Date().toISOString(), gpuRequested: false,
  recognitionResponsesMocked: true, warmupResponsesMocked: true,
  purpose: 'Verify deployed mobile assets and local hand editing/scoring; not model accuracy or GPU connectivity.',
  browserErrors: [], consoleErrors: [], cspViolations: [], unexpectedGpuRequests: [], screens: [], widths: [] };
const browser = await chromium.launch({ headless: true, ...(proxy ? { proxy: { server: proxy } } : {}) });
try {
  const context = await browser.newContext({ isMobile: true, hasTouch: true, deviceScaleFactor: 1 });
  await context.route('**/*.modal.run/**', route => {
    report.unexpectedGpuRequests.push(route.request().url());
    return route.abort();
  });
  const health = await context.request.get(`${site}/api/health`);
  assert.equal(health.status(), 200);
  report.health = await health.json();
  assert.equal(report.health.inferenceProbed, false);
  assert.equal(report.health.version, expectedVersion, 'The deployed gateway must match the expected release');
  for (const width of [320, 390, 430]) {
    const page = await context.newPage();
    await page.setViewportSize({ width, height: 844 });
    page.on('pageerror', error => report.browserErrors.push({ width, message: error.message }));
    page.on('console', message => {
      if (message.type() === 'error') report.consoleErrors.push({ width, message: message.text(), source: message.location().url });
    });
    await page.addInitScript(() => {
      window.__scoringCspViolations = [];
      document.addEventListener('securitypolicyviolation', event => {
        window.__scoringCspViolations.push({ blockedURI: event.blockedURI, directive: event.effectiveDirective });
      });
    });
    let recognitionCalls = 0, warmupCalls = 0;
    await page.route('**/api/warmup', route => {
      warmupCalls++;
      return route.fulfill({ json: { ready: true, mockedForLocalScoringVerification: true } });
    });
    await page.route('**/api/recognize', route => {
      recognitionCalls++;
      assert.equal(route.request().method(), 'POST');
      return route.fulfill({ json: {
        requestId: `live-scoring-mock-${width}`, modelVersion: 'MOCK-local-scoring-only-no-GPU',
        image: { width: 1000, height: 600 }, quality: { status: 'ok', reasons: [] }, timings: { totalMs: 0 },
        tiles: ordinary.map((classId, i) => ({ id: `fixture-${i}`, classId, label: classId, score: .99,
          uncertain: false, bbox: [10 + (i % 7) * 140, 10 + Math.floor(i / 7) * 180, 100, 150] })),
      } });
    });
    const navigation = await page.goto(site);
    assert.equal(navigation.status(), 200);
    assert.match(navigation.headers()['content-security-policy'] ?? '', /script-src 'self'/);
    assert.match(navigation.headers()['cache-control'] ?? '', /no-transform/);
    await expect(page.getByRole('button', { name: '拍一张', exact: true })).toBeVisible();
    const assets = await page.locator('script[src]').evaluateAll(nodes => nodes.map(node => node.src));
    assert.equal(assets.every(url => new URL(url).origin === new URL(site).origin), true);
    if (!report.assets) {
      report.assets = [];
      for (const url of assets) {
        const asset = await context.request.get(url);
        assert.equal(asset.status(), 200);
        report.assets.push({ url, sha256: createHash('sha256').update(await asset.body()).digest('hex') });
      }
    }
    const jpeg = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width = 1000; canvas.height = 600;
      const ctx = canvas.getContext('2d'); ctx.fillStyle = '#356952'; ctx.fillRect(0, 0, 1000, 600);
      return canvas.toDataURL('image/jpeg').split(',')[1];
    });
    await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles({
      name: 'MOCK-scoring-only.jpg', mimeType: 'image/jpeg', buffer: Buffer.from(jpeg, 'base64'),
    });
    await page.getByRole('button', { name: '开始识别', exact: true }).click();
    const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
    await expect(panel).toBeVisible();
    await expect(panel.getByTestId('score-total')).toHaveText('0');
    await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
    await panel.getByLabel('和牌方式', { exact: true }).selectOption('self-draw');
    await expect(panel.getByTestId('score-total')).toHaveText('0');
    await panel.getByRole('checkbox', { name: '门清', exact: true }).check();
    await expect(panel.getByTestId('score-total')).toHaveText('3');
    await panel.locator('summary').filter({ hasText: /^中码/ }).click();
    await panel.getByLabel('中码数', { exact: true }).fill('2');
    await expect(panel.getByTestId('score-total')).toHaveText('9');

    // Break and restore the pair, then delete and supplement it; all scoring stays local.
    await page.locator('.tile-button').last().click();
    await page.getByRole('dialog').getByRole('button', { name: '东', exact: true }).click();
    await expect(panel.getByTestId('score-total')).toHaveText('—');
    await expect(panel.getByText('当前牌面暂未组成胡牌', { exact: true })).toBeVisible();
    await page.locator('.tile-button').last().click();
    await page.getByRole('dialog').getByRole('button', { name: '白板', exact: true }).click();
    await expect(panel.getByTestId('score-total')).toHaveText('9');
    await page.locator('.tile-button').last().click();
    await page.getByRole('dialog').getByRole('button', { name: '删除这张牌', exact: true }).click();
    await expect(page.getByRole('heading', { name: '13 张麻将', exact: true })).toBeVisible();
    await expect(panel.getByTestId('score-total')).toHaveText('—');
    await page.getByRole('button', { name: '＋ 添加麻将牌', exact: true }).click();
    await page.getByRole('dialog').getByRole('button', { name: '白板', exact: true }).click();
    await page.getByRole('button', { name: '完成补牌', exact: true }).click();
    await expect(page.getByRole('heading', { name: '14 张麻将', exact: true })).toBeVisible();
    await expect(panel.getByTestId('score-total')).toHaveText('9');
    await expect(page.locator('.annotation-layer g')).toHaveCount(13);
    await expect(page.locator('.manual-added')).toHaveCount(1);
    assert.equal(recognitionCalls, 1, 'Local editing/scoring must not send another recognition');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    const touchHeights = await panel.locator('button:visible,select:visible,input[type="number"]:visible,summary:visible,.scoring-choice:visible')
      .evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
    assert.equal(touchHeights.every(height => height >= 44), true);
    await expect(page.locator('footer a[href="https://github.com/XTTTZ/mahjong-recognizer"]')).toBeVisible();
    await expect(page.locator('footer a[href="https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027"]')).toBeVisible();
    const screenshot = `scoring-mocked-${width}.png`;
    await page.screenshot({ path: path.join(out, screenshot), fullPage: true });
    report.screens.push(screenshot);
    report.cspViolations.push(...await page.evaluate(() => window.__scoringCspViolations));
    report.widths.push({ width, mockRecognitionCalls: recognitionCalls, mockWarmupCalls: warmupCalls,
      ordinarySelfDraw: 0, menqing: 3, twoCodeHits: 9, correctionRecomputed: true,
      deletionRecomputed: true, supplementRecomputed: true, noFakeManualBbox: true,
      noHorizontalOverflow: true, minimumTouchHeight: Math.min(...touchHeights), sourceOfferVisible: true });
    await page.close();
  }
  for (const field of ['browserErrors', 'consoleErrors', 'cspViolations', 'unexpectedGpuRequests']) assert.deepEqual(report[field], []);
  report.passed = true;
} catch (error) {
  report.passed = false; report.error = error.message; process.exitCode = 1;
} finally {
  await browser.close();
  await fs.writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
}
