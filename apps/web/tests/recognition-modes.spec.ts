import { expect, test, type Page } from '@playwright/test';

const result = (requestId: string, modelVersion: string) => ({
  requestId, modelVersion, image: { width: 100, height: 100 },
  quality: { status: 'ok', reasons: [] }, timings: { totalMs: 125 },
  tiles: [{ id: '1', classId: '1m', label: '一万', bbox: [10, 10, 50, 70], score: .9, uncertain: false }],
});
async function upload(page: Page) {
  const pixels = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 100; canvas.height = 100;
    canvas.getContext('2d')!.fillRect(0, 0, 100, 100);
    return canvas.toDataURL('image/jpeg').split(',')[1];
  });
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles({ name: 'fixture.jpg', mimeType: 'image/jpeg', buffer: Buffer.from(pixels, 'base64') });
  await expect(page.getByRole('button', { name: '开始识别' })).toBeVisible();
}

test('discovers configuration before warming only the default fast service', async ({ page }) => {
  const warmups: string[] = [], resources: string[] = [];
  let resolveHealth!: () => void;
  const health = new Promise<void>(resolve => { resolveHealth = resolve; });
  await page.route('**/api/health', async route => { await health; await route.fulfill({ json: { configured: true, fastConfigured: true } }); });
  await page.route('**/api/warmup*', route => { warmups.push(route.request().url()); return route.fulfill({ json: { ready: true } }); });
  page.on('request', request => { if (/\/models\/|\/ort\/|local\.worker/.test(request.url())) resources.push(request.url()); });
  await page.setViewportSize({ width: 320, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: '快速识别', exact: true })).toBeVisible();
  expect(warmups).toHaveLength(0);
  resolveHealth();
  await expect.poll(() => warmups.length).toBe(1);
  expect(new URL(warmups[0]).search).toBe('?mode=fast');
  await expect(page.getByRole('button', { name: '快速识别', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect((await page.getByRole('button', { name: '快速识别', exact: true }).boundingBox())?.height).toBeGreaterThanOrEqual(44);
  await expect(page.getByRole('link', { name: '本机测试', exact: true })).toHaveAttribute('href', '/benchmark');
  expect(resources).toHaveLength(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('fast failure keeps the photo and requires a manual precision upload', async ({ page }) => {
  const calls: string[] = [];
  await page.route('**/api/health', route => route.fulfill({ json: { configured: true, fastConfigured: true } }));
  await page.route('**/api/warmup*', route => route.fulfill({ json: { ready: true } }));
  await page.route('**/api/recognize*', async route => {
    calls.push(route.request().url());
    if (new URL(route.request().url()).searchParams.get('mode') === 'fast') {
      await route.fulfill({ status: 503, json: { error: { code: 'UNAVAILABLE', message: '快速识别暂不可用，可手动切换为精细识别。' } } });
    } else await route.fulfill({ json: result('precision-result', 'original-precision-fixture') });
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: '快速识别', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await upload(page);
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('alert')).toContainText('手动切换为精细识别');
  expect(calls).toHaveLength(1);
  await page.getByRole('button', { name: '精细识别', exact: true }).click();
  expect(calls).toHaveLength(1);
  await expect(page.getByAltText('本次识别的麻将照片')).toBeVisible();
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: '1 张麻将', exact: true })).toBeVisible();
  expect(calls).toHaveLength(2); expect(new URL(calls[1]).search).toBe('');
  await page.locator('.result-details > summary').click();
  await expect(page.locator('.result-details')).toContainText('精细识别');
  await expect(page.locator('.result-details')).toContainText('original-precision-fixture');
});

test('busy requests cannot change mode and cancelled results cannot replace the next mode result', async ({ page }) => {
  let release!: () => void;
  let requestStarted!: () => void;
  const started = new Promise<void>(resolve => { requestStarted = resolve; });
  const blocked = new Promise<void>(resolve => { release = resolve; });
  const calls: string[] = [];
  await page.route('**/api/health', route => route.fulfill({ json: { configured: true, fastConfigured: true } }));
  await page.route('**/api/warmup*', route => route.fulfill({ json: { ready: true } }));
  await page.route('**/api/recognize*', async route => {
    calls.push(route.request().url());
    if (new URL(route.request().url()).searchParams.get('mode') === 'fast') {
      requestStarted(); await blocked;
      await route.fulfill({ json: result('stale-fast', 'stale-fast-version') }).catch(() => {});
    } else await route.fulfill({ json: result('new-precision', 'new-precision-version') });
  });
  await page.goto('/'); await upload(page);
  await page.getByRole('button', { name: '开始识别' }).click(); await started;
  await expect(page.getByRole('button', { name: '快速识别', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: '精细识别', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: '取消', exact: true }).click();
  await page.getByRole('button', { name: '精细识别', exact: true }).click();
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: '1 张麻将', exact: true })).toBeVisible();
  release();
  await page.locator('.result-details > summary').click();
  await expect(page.locator('.result-details')).toContainText('new-precision-version');
  await expect(page.locator('.result-details')).not.toContainText('stale-fast-version');
  expect(calls).toHaveLength(2);
});
