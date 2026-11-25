import { expect, test, type Page } from '@playwright/test';

type Size = [number, number];
type Box = [number, number, number, number];

async function selectFixture(page: Page, [width, height]: Size) {
  const data = await page.evaluate(({ width, height }) => {
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#25405e'; ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#e92413'; ctx.fillRect(width / 4, height / 4, width / 4, height / 2);
    return canvas.toDataURL('image/png').split(',')[1];
  }, { width, height });
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles({
    name: 'red-tile.png', mimeType: 'image/png', buffer: Buffer.from(data, 'base64'),
  });
}

async function captureUploads(page: Page) {
  const received: { size: Size; mode: string | null }[] = [];
  await page.route('**/api/health', route => route.fulfill({ json: { configured: true, fastConfigured: true } }));
  await page.route('**/api/warmup*', route => route.fulfill({ json: { ready: true } }));
  await page.route('**/api/recognize*', async route => {
    const request = route.request();
    const form = await new Response(new Uint8Array(request.postDataBuffer()!), {
      headers: { 'content-type': request.headers()['content-type'] },
    }).formData();
    const file = form.get('image') as File;
    const detection = await page.evaluate(async base64 => {
      const bitmap = await createImageBitmap(new Blob([Uint8Array.from(atob(base64), c => c.charCodeAt(0))]));
      const { width, height } = bitmap;
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      const pixels = ctx.getImageData(0, 0, width, height).data;
      let left = width, top = height, right = 0, bottom = 0;
      for (let y = 0; y < height; y += 2) for (let x = 0; x < width; x += 2) {
        const idx = (y * width + x) * 4;
        if (pixels[idx] < 160 || pixels[idx + 2] > 70) continue;
        left = Math.min(x, left); right = x > right ? x : right;
        top = Math.min(y, top); bottom = y > bottom ? y : bottom;
      }
      return { width, height, bbox: [left, top, right + 1 - left, bottom + 1 - top] };
    }, Buffer.from(await file.arrayBuffer()).toString('base64'));
    received.push({ size: [detection.width, detection.height], mode: new URL(request.url()).searchParams.get('mode') });
    await route.fulfill({ json: {
      requestId: 'upload-' + received.length, modelVersion: 'red-marker',
      image: { width: detection.width, height: detection.height }, timings: { totalMs: 1 },
      quality: { status: 'ok', reasons: [] },
      tiles: [{ id: '1', classId: '1m', label: '一万', score: 1, uncertain: false, bbox: detection.bbox }],
    } });
  });
  return received;
}

async function verifyDisplayedMarker(page: Page, size: Size, marker: Box) {
  const image = page.getByAltText('本次识别的麻将照片');
  const dimensions = await image.evaluate((el: HTMLImageElement) => [el.naturalWidth, el.naturalHeight]);
  expect(dimensions).toEqual(size);
  const rect = (await image.boundingBox())!;
  const scale = Math.min(rect.width / size[0], rect.height / size[1]);
  const width = size[0] * scale, height = size[1] * scale;
  const left = rect.x + (rect.width - width) / 2, top = rect.y + (rect.height - height) / 2;
  const drawn = (await page.locator('.annotation rect').first().boundingBox())!;
  const expected = [left + marker[0] * width, top + marker[1] * height, marker[2] * width, marker[3] * height];
  [drawn.x, drawn.y, drawn.width, drawn.height].forEach((v, index) => {
    expect(Math.abs(v - expected[index])).toBeLessThan(2);
  });
}

test('cloud upload sizes and box alignment follow mode, rotation and crop', async ({ page }) => {
  const uploads = await captureUploads(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await selectFixture(page, [3200, 1600]);
  async function check(mode: '快速识别' | '精细识别', posted: Size, preview: Size, rect: Box) {
    await page.getByRole('button', { name: mode, exact: true }).click();
    await page.getByRole('button', { name: '开始识别', exact: true }).click();
    await expect(page.locator('.annotation rect')).toHaveCount(1);
    expect(uploads.at(-1)).toEqual({ size: posted, mode: mode === '快速识别' ? 'fast' : null });
    await verifyDisplayedMarker(page, preview, rect);
  }
  await check('快速识别', [1600, 800], [2560, 1280], [.25, .25, .25, .5]);
  await check('精细识别', [2560, 1280], [2560, 1280], [.25, .25, .25, .5]);
  await page.getByRole('button', { name: '向右旋转照片' }).click();
  await check('快速识别', [800, 1600], [1280, 2560], [.25, .25, .5, .25]);
  await page.getByRole('button', { name: '选取要识别的牌面区域' }).click();
  await page.getByRole('slider', { name: '裁剪下边' }).evaluate(el => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(el, '75');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.getByRole('button', { name: '应用区域' }).click();
  await check('精细识别', [1280, 1920], [1280, 1920], [.25, 1 / 3, .5, 1 / 3]);
  await check('快速识别', [1067, 1600], [1280, 1920], [.25, 1 / 3, .5, 1 / 3]);

  await selectFixture(page, [640, 480]);
  await check('快速识别', [640, 480], [640, 480], [.25, .25, .25, .5]);
  await check('精细识别', [640, 480], [640, 480], [.25, .25, .25, .5]);
  expect(uploads).toHaveLength(7);
});
