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

test('warms only the default fast service while configuration is pending', async ({ page }) => {
  const warmups: string[] = [], resources: string[] = [];
  let resolveHealth!: () => void;
  const health = new Promise<void>(resolve => { resolveHealth = resolve; });
  await page.route('**/api/health', async route => { await health; await route.fulfill({ json: { configured: true, fastConfigured: true } }); });
  await page.route('**/api/warmup*', route => { warmups.push(route.request().url()); return route.fulfill({ json: { ready: true } }); });
  page.on('request', request => { if (/\/models\/|\/ort\/|local\.worker/.test(request.url())) resources.push(request.url()); });
  await page.setViewportSize({ width: 320, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: '快速识别', exact: true })).toBeVisible();
  await expect.poll(() => warmups.length).toBe(1);
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

test('local opt-in persists, reuses a compact worker and feeds the shared review flow without cloud requests', async ({ page }) => {
  const cloud: string[] = [];
  await page.route('**/api/**', route => {
    cloud.push(new URL(route.request().url()).pathname);
    return route.fulfill({ json: { configured: true, fastConfigured: true, ready: true } });
  });
  await page.addInitScript(() => {
    class FixtureWorker {
      onmessage?: (event: { data: unknown }) => void;
      postMessage(message: { type: string; profileId?: string; provider?: string; file?: File }) {
        if (message.type === 'load') {
          if (message.profileId !== 'compact-v10' || message.provider !== 'wasm') throw new Error('wrong model');
          document.documentElement.dataset.loads = String(Number(document.documentElement.dataset.loads ?? 0) + 1);
          queueMicrotask(() => this.onmessage?.({ data: { type: 'ready' } }));
        } else {
          if (!(message.file instanceof File)) throw new Error('expected local file');
          const tiles = ['1m', '1m', '1m', '2m', '3m', '4m', '2p', '3p', '4p', '6s', '7s', '8s', '1z', '1z'].map((classId, i) => ({
            classId, bbox: [i * 6, 1, 5, 20], score: .9,
          }));
          this.onmessage?.({ data: { type: 'result', value: { profileId: 'compact-v10', modelVersion: 'local-fixture',
            inputImage: { width: 100, height: 100 }, decodeMs: 1, runs: [{ totalMs: 350, tiles }] } } });
        }
      }
      terminate() {}
    }
    Object.defineProperty(window, 'Worker', { value: FixtureWorker });
  });
  await page.goto('/');
  await page.getByRole('button', { name: '本机识别', exact: true }).click();
  await expect(page.getByText(/首次准备约 41.8 MB/)).toBeVisible();
  await upload(page);
  await expect(page.getByRole('button', { name: '开始识别' })).toBeDisabled();
  await page.getByRole('button', { name: '启用本机识别' }).click();
  await expect(page.getByText(/本机模型已就绪/)).toBeVisible();
  cloud.length = 0;
  await page.reload();
  await expect(page.getByRole('button', { name: '本机识别', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await upload(page);
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: '14 张麻将' })).toBeVisible();
  await page.locator('.tile-button').last().click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  await expect(page.getByRole('heading', { name: '13 张麻将' })).toBeVisible();
  await expect(page.locator('#hand-score')).toContainText('听牌');
  await upload(page);
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: '14 张麻将' })).toBeVisible();
  expect(await page.locator('html').getAttribute('data-loads')).toBe('1');
  expect(cloud).toEqual([]);
});

test('cancelling local load releases its worker and a manual switch retains the photo', async ({ page }) => {
  const uploads: string[] = [];
  await page.route('**/api/recognize*', route => { uploads.push(route.request().url()); return route.abort(); });
  await page.route('**/api/health', route => route.fulfill({ json: { configured: true, fastConfigured: true } }));
  await page.route('**/api/warmup*', route => route.fulfill({ json: { ready: true } }));
  await page.addInitScript(() => {
    localStorage.setItem('mahjong:preferred-local:compact-v10', 'yes');
    class SlowWorker {
      onmessage?: (event: { data: unknown }) => void;
      postMessage() { this.onmessage?.({ data: { type: 'status', value: '下载定位模型 10%' } }); }
      terminate() { document.documentElement.dataset.released = 'yes'; }
    }
    Object.defineProperty(window, 'Worker', { value: SlowWorker });
  });
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('下载定位模型 10%');
  await upload(page);
  await expect(page.getByRole('button', { name: '开始识别' })).toBeDisabled();
  await page.getByRole('button', { name: '取消准备', exact: true }).click();
  expect(await page.locator('html').getAttribute('data-released')).toBe('yes');
  await page.getByRole('button', { name: '快速识别', exact: true }).click();
  await expect(page.getByAltText('本次识别的麻将照片')).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem('mahjong:preferred-local:compact-v10'))).toBeNull();
  expect(uploads).toEqual([]);
});

test('background preparation leaves cloud recognition available and uses no photo until local recognition is chosen', async ({ page }) => {
  const uploads: string[] = [];
  await page.route('**/api/**', route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/recognize') {
      uploads.push(url.search);
      return route.fulfill({ json: result('cloud-during-load', 'fast-fixture') });
    }
    return route.fulfill({ json: { configured: true, fastConfigured: true, ready: true } });
  });
  await page.addInitScript(() => {
    class HeldWorker {
      onmessage?: (event: { data: unknown }) => void;
      postMessage(command: { type: string; file?: File }) {
        if (command.type !== 'load' || command.file) throw new Error('background preparation must never receive photos');
        window.addEventListener('fixture-ready', () => this.onmessage?.({ data: { type: 'ready' } }), { once: true });
        this.onmessage?.({ data: { type: 'status', value: '下载分类模型 30%' } });
      }
      terminate() {}
    }
    Object.defineProperty(window, 'Worker', { value: HeldWorker });
  });
  await page.goto('/');
  await page.getByRole('button', { name: '后台准备本机识别' }).click();
  await expect(page.getByRole('status')).toContainText('可继续云端识别');
  await upload(page);
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: '1 张麻将' })).toBeVisible();
  await page.evaluate(() => window.dispatchEvent(new Event('fixture-ready')));
  await expect(page.getByText('本机模型已就绪，当前仍使用云端识别。')).toBeVisible();
  expect(uploads).toEqual(['?mode=fast']);
  await page.getByRole('button', { name: '使用本机识别' }).click();
  await expect(page.getByRole('button', { name: '本机识别', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: '开始识别' })).toBeEnabled();
  expect(uploads).toHaveLength(1);
});

test('a failed local preload can be retried and never forwards the selected photo', async ({ page }) => {
  const requests: string[] = [];
  await page.route('**/api/**', route => { requests.push(route.request().url()); return route.abort(); });
  await page.addInitScript(() => {
    localStorage.setItem('mahjong:preferred-local:compact-v10', 'yes');
    let attempts = 0;
    class FailsOnce {
      onmessage?: (event: { data: unknown }) => void;
      postMessage() {
        this.onmessage?.({ data: ++attempts === 1 ? { type: 'error', value: '模型校验失败' } : { type: 'ready' } });
      }
      terminate() {}
    }
    Object.defineProperty(window, 'Worker', { value: FailsOnce });
  });
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('模型校验失败');
  await upload(page);
  await expect(page.getByRole('button', { name: '开始识别' })).toBeDisabled();
  await page.getByRole('button', { name: '重新准备' }).click();
  await expect(page.getByRole('button', { name: '开始识别' })).toBeEnabled();
  expect(requests).toEqual([]);
});
