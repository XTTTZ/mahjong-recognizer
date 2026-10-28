import { expect, test, type Page } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/api/warmup', route => route.fulfill({ json: { ready: true } }));
});
import fs from 'node:fs/promises';

const classes = ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', '1z', '2z', '3z', '4z', '5z'];
const labels = ['一万', '二万', '三万', '四筒', '五筒', '六筒', '七条', '八条', '九条', '东', '南', '西', '北', '白'];
const response = {
  requestId: 'browser-contract-test', modelVersion: 'contract-test-only',
  image: { width: 1400, height: 700 }, quality: { status: 'needs_retake', reasons: ['第 7 张牌面不够清楚，请核对。'] }, timings: { totalMs: 1234 },
  tiles: classes.map((classId, i) => ({ id: `tile-${i}`, classId, label: labels[i], bbox: [60 + (i % 7) * 185, 110 + Math.floor(i / 7) * 260, 140, 200], score: i === 6 ? 0.5 : 0.95, uncertain: i === 6 })),
};

test.beforeEach(async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { configured: true, inferenceProbed: false } }));
});

async function fixture(page: Page) {
  const data = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1400; canvas.height = 700;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#356952'; ctx.fillRect(0, 0, 1400, 700);
    ctx.font = 'bold 70px serif'; ctx.textAlign = 'center';
    const text = ['一万', '二万', '三万', '四筒', '五筒', '六筒', '七条', '八条', '九条', '东', '南', '西', '北', '白'];
    for (let i = 0; i < 14; i++) {
      const x = 60 + (i % 7) * 185, y = 110 + Math.floor(i / 7) * 260;
      ctx.fillStyle = '#c3cda3'; ctx.fillRect(x, y, 140, 209);
      ctx.fillStyle = '#fffef1'; ctx.fillRect(x, y, 140, 200);
      ctx.fillStyle = i < 3 ? '#a74f3c' : '#234a3c';
      ctx.fillText(text[i], x + 70, y + 125, 128);
    }
    return canvas.toDataURL('image/jpeg', 0.95).split(',')[1];
  });
  return { name: 'contract-fixture.jpg', mimeType: 'image/jpeg', buffer: Buffer.from(data, 'base64') };
}

async function upload(page: Page) {
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles(await fixture(page));
  await expect(page.getByRole('button', { name: '开始识别' })).toBeVisible();
}

test('preparing service keeps photo tools available and disables recognition', async ({ page }) => {
  let inferenceCalls = 0;
  await page.route('**/api/health', route => route.fulfill({ json: { configured: false, inferenceProbed: false } }));
  await page.route('**/api/recognize', route => { inferenceCalls++; return route.abort(); });
  await page.goto('/');
  await expect(page.getByText('试用模型 · 结果请逐张核对')).toBeVisible();
  await expect(page.getByText('识别服务准备中，暂可拍照、选图和裁剪。')).toBeVisible();
  await upload(page);
  await expect(page.getByRole('button', { name: '开始识别' })).toBeDisabled();
  await expect(page.getByRole('button', { name: '选取要识别的牌面区域' })).toBeEnabled();
  expect(inferenceCalls).toBe(0);
});

test('known class tokens display reviewed Chinese names when model labels are absent', async ({ page }) => {
  await page.goto('/');
  await upload(page);
  await page.route('**/api/recognize', route => route.fulfill({ json: {
    ...response, tiles: response.tiles.map(tile => ({ ...tile, label: tile.classId })),
  } }));
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('button', { name: '第 1 张 一万', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '第 14 张 白板', exact: true })).toBeVisible();
  await expect(page.locator('.tile-label').first()).toHaveText('一万');
});

test('source and model download offers are visible and usable at 320px', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 });
  await page.goto('/');
  const source = page.getByRole('link', { name: '完整源码 · AGPL-3.0' });
  const model = page.getByRole('link', { name: '模型下载' });
  await expect(source).toHaveAttribute('href', 'https://github.com/XTTTZ/mahjong-recognizer');
  await expect(model).toHaveAttribute('href', 'https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027');
  await expect(source).toBeVisible();
  await expect(model).toBeVisible();
  for (const link of [source, model]) {
    expect((await link.boundingBox())?.height).toBeGreaterThanOrEqual(44);
    await expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

for (const width of [320, 390, 430]) {
  test(`mobile ${width}: upload, result, corrections and layout`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('/');
    await expect(page.getByRole('button', { name: '拍一张' })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await fs.mkdir('artifacts/web/screenshots', { recursive: true });
    if (width === 390) await page.screenshot({ path: 'artifacts/web/screenshots/home-390.png', fullPage: true });
    await upload(page);
    let count = 0;
    await page.route('**/api/recognize', async route => {
      count++;
      expect(route.request().headers()['content-type']).toContain('multipart/form-data');
      await route.fulfill({ json: response });
    });
    await page.getByRole('button', { name: '开始识别' }).click();
    await expect(page.getByRole('heading', { name: '14 张麻将' })).toBeVisible();
    await expect(page.getByText('1 张待核对', { exact: true })).toBeVisible();
    expect(count).toBe(1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    const targets = await page.locator('.tile-button').evaluateAll(elements => elements.map(element => ({ width: element.getBoundingClientRect().width, height: element.getBoundingClientRect().height })));
    expect(targets.every(target => target.width >= 44 && target.height >= 44)).toBe(true);
    await page.getByRole('button', { name: '第 7 张 七条，待核对', exact: true }).click();
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.getByRole('button', { name: '八条', exact: true }).click();
    await expect(page.getByRole('button', { name: '第 7 张 八条，已手动修改', exact: true })).toBeVisible();
    await expect(page.getByText('点击牌面可修改 · 1 处人工调整')).toBeVisible();
    await page.screenshot({ path: `artifacts/web/screenshots/result-${width}.png`, fullPage: true });
    await page.getByRole('button', { name: '第 1 张 一万', exact: true }).click();
    await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
    await expect(page.getByRole('heading', { name: '13 张麻将' })).toBeVisible();
    await page.getByRole('button', { name: '向右旋转照片' }).click();
    await expect(page.getByRole('button', { name: '开始识别' })).toBeVisible();
    const dimensions = await page.locator('.source-photo').evaluate((img: HTMLImageElement) => ({ width: img.naturalWidth, height: img.naturalHeight }));
    expect(dimensions).toEqual({ width: 700, height: 1400 });
  });
}

test('cancelled or replaced requests cannot overwrite the next photo', async ({ page }) => {
  await page.goto('/');
  await upload(page);
  let release: () => void = () => {};
  let calls = 0;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/recognize', async route => {
    calls++;
    await gate;
    await route.fulfill({ json: response }).catch(() => {});
  });
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByText('正在仔细看牌…')).toBeVisible();
  await page.getByRole('button', { name: '取消', exact: true }).click();
  await upload(page);
  release();
  await expect(page.getByRole('button', { name: '开始识别' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '14 张麻将' })).toHaveCount(0);
  await page.waitForTimeout(400);
  expect(calls).toBe(1);
});

test('rate limit is actionable and does not automatically send another paid request', async ({ page }) => {
  await page.goto('/');
  await upload(page);
  let calls = 0;
  await page.route('**/api/recognize', async route => { calls++; await route.fulfill({ status: 429, json: { error: { code: 'RATE_LIMITED', message: '识别请求较多，请稍等片刻再试。' } } }); });
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('alert')).toContainText('识别请求较多');
  await page.waitForTimeout(300);
  expect(calls).toBe(1);
  await expect(page.getByRole('button', { name: '开始识别' })).toBeEnabled();
});

test('a slow GPU cold start can finish after the former client timeout without a duplicate request', async ({ page }) => {
  await page.clock.install();
  await page.goto('/');
  await upload(page);
  let release: () => void = () => {};
  let calls = 0;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/recognize', async route => {
    calls++;
    await gate;
    await route.fulfill({ json: response });
  });
  await page.getByRole('button', { name: '开始识别' }).click();
  await page.clock.fastForward(75_000);
  await expect(page.getByText('正在仔细看牌…')).toBeVisible();
  await expect(page.getByText('服务正在启动，可能需要多等一会儿')).toBeVisible();
  release();
  await expect(page.getByRole('heading', { name: '14 张麻将' })).toBeVisible();
  expect(calls).toBe(1);
});

test('unsupported files and offline states have clear recovery', async ({ page, context }) => {
  await page.goto('/');
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles({ name: 'not-an-image.txt', mimeType: 'text/plain', buffer: Buffer.from('no image') });
  await expect(page.getByRole('alert')).toContainText('请选择');
  await upload(page);
  await context.setOffline(true);
  await expect(page.getByRole('button', { name: '开始识别' })).toBeDisabled();
  await context.setOffline(false);
  await expect(page.getByRole('button', { name: '开始识别' })).toBeEnabled();
});

async function setCropSlider(page: Page, name: string, value: number) {
  await page.getByRole('slider', { name }).evaluate((element, next) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
    setter.call(element, String(next));
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function photoSize(page: Page) {
  return page.locator('.source-photo').evaluate((image: HTMLImageElement) => ({ width: image.naturalWidth, height: image.naturalHeight }));
}

async function colorFixture(page: Page, width = 1400, height = 700) {
  const base64 = await page.evaluate(({ width, height }) => {
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#2453c1'; context.fillRect(0, 0, width, height / 2);
    context.fillStyle = '#c16c24'; context.fillRect(0, height / 2, width, height / 2);
    return canvas.toDataURL('image/jpeg', .98).split(',')[1];
  }, { width, height });
  return { name: 'manual-crop-fixture.jpg', mimeType: 'image/jpeg', buffer: Buffer.from(base64, 'base64') };
}

test('manual ROI sends a real cropped JPEG, preserves original, and keeps 320px controls usable', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 760 });
  await page.goto('/');
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles(await colorFixture(page));
  await expect(page.getByRole('button', { name: '开始识别' })).toBeVisible();
  const crop = page.getByRole('dialog', { name: '要识别哪一组牌？' });
  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await expect(crop).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  const targets = await crop.locator('input[type=range],button').evaluateAll(elements => elements.map(element => ({ width: element.getBoundingClientRect().width, height: element.getBoundingClientRect().height })));
  expect(targets.every(target => target.width >= 44 && target.height >= 44)).toBe(true);
  await setCropSlider(page, '裁剪上边', 50);
  await page.screenshot({ path: 'artifacts/web/screenshots/crop-320.png', fullPage: true });
  await crop.getByRole('button', { name: '应用区域' }).click();
  await expect(crop).not.toBeVisible();
  await expect.poll(() => photoSize(page)).toEqual({ width: 1400, height: 350 });

  // Reopening edits against the retained original, while cancellation leaves the
  // accepted pixel region intact rather than progressively cropping the crop.
  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await expect(page.getByRole('slider', { name: '裁剪上边' })).toHaveValue('50');
  await setCropSlider(page, '裁剪左边', 20);
  await crop.getByRole('button', { name: '取消', exact: true }).click();
  await expect.poll(() => photoSize(page)).toEqual({ width: 1400, height: 350 });
  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await expect(page.getByRole('slider', { name: '裁剪左边' })).toHaveValue('0');
  await crop.getByRole('button', { name: '重置整图' }).click();
  await crop.getByRole('button', { name: '应用区域' }).click();
  await expect.poll(() => photoSize(page)).toEqual({ width: 1400, height: 700 });

  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await setCropSlider(page, '裁剪上边', 50);
  await crop.getByRole('button', { name: '应用区域' }).click();
  let uploaded: Buffer | undefined;
  await page.route('**/api/recognize', async route => {
    const bytes = new Uint8Array(route.request().postDataBuffer()!).buffer;
    const form = await new Request('https://contract.test/recognize', { method: 'POST', body: bytes,
      headers: { 'Content-Type': route.request().headers()['content-type'] } }).formData();
    const image = form.get('image') as File;
    uploaded = Buffer.from(await image.arrayBuffer());
    await route.fulfill({ json: { requestId: 'manual-crop-contract', modelVersion: 'contract-only',
      image: { width: 1400, height: 350 }, tiles: [], quality: { status: 'needs_retake', reasons: ['合同测试图片'] }, timings: { totalMs: 1 } } });
  });
  await page.getByRole('button', { name: '开始识别' }).click();
  await expect(page.getByRole('heading', { name: '暂未找到麻将' })).toBeVisible();
  expect(uploaded).toBeDefined();
  const encoded = await page.evaluate(async data => {
    const bytes = Uint8Array.from(atob(data), char => char.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const context = canvas.getContext('2d')!;
    context.drawImage(bitmap, 0, 0);
    const pixel = Array.from(context.getImageData(16, 16, 1, 1).data);
    const value = { width: bitmap.width, height: bitmap.height, pixel };
    bitmap.close(); return value;
  }, uploaded!.toString('base64'));
  expect(encoded.width).toBe(1400);
  expect(encoded.height).toBe(350);
  expect(encoded.pixel[0]).toBeGreaterThan(150);
  expect(encoded.pixel[2]).toBeLessThan(80);

  await page.getByRole('button', { name: '向右旋转照片' }).click();
  await expect.poll(() => photoSize(page)).toEqual({ width: 350, height: 1400 });
  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await expect(page.getByRole('slider', { name: '裁剪左边' })).toHaveValue('0');
  await expect(page.getByRole('slider', { name: '裁剪右边' })).toHaveValue('50');
  await expect(page.getByRole('slider', { name: '裁剪上边' })).toHaveValue('0');
  await crop.getByRole('button', { name: '取消', exact: true }).click();
  await page.getByRole('button', { name: '恢复整图' }).click();
  await expect.poll(() => photoSize(page)).toEqual({ width: 700, height: 1400 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('a late crop operation cannot replace a new photo or close its new region dialog', async ({ page }) => {
  await page.goto('/');
  await upload(page);
  await page.evaluate(() => {
    const original = HTMLCanvasElement.prototype.toBlob;
    let calls = 0;
    HTMLCanvasElement.prototype.toBlob = function(callback, ...args) {
      if (calls++ === 0) original.call(this, blob => {
        (window as unknown as { releaseCrop: () => void }).releaseCrop = () => callback(blob);
      }, ...args);
      else original.call(this, callback, ...args);
    };
  });
  const crop = page.getByRole('dialog', { name: '要识别哪一组牌？' });
  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await setCropSlider(page, '裁剪上边', 50);
  await crop.getByRole('button', { name: '应用区域' }).click();
  await page.waitForFunction(() => 'releaseCrop' in window);
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles(await colorFixture(page, 800, 800));
  await expect(page.getByRole('button', { name: '开始识别' })).toBeVisible();
  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await expect(crop).toBeVisible();
  await page.evaluate(() => (window as unknown as { releaseCrop: () => void }).releaseCrop());
  await page.waitForTimeout(150);
  await expect(crop).toBeVisible();
  await expect(page.getByRole('slider', { name: '裁剪上边' })).toHaveValue('0');
  await crop.getByRole('button', { name: '取消', exact: true }).click();
  await expect.poll(() => photoSize(page)).toEqual({ width: 800, height: 800 });
});
