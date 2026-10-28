import { expect, test, type Page } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/api/warmup', route => route.fulfill({ json: { ready: true } }));
});

const fullHand = ['1m','1m','1m','2m','2m','2m','3m','3m','3m','4m','4m','4m','5m','5m'];
async function recognizeFixture(page: Page, tiles: string[]) {
  let calls = 0;
  await page.route('**/api/health', route => route.fulfill({ json: { configured: true, inferenceProbed: false } }));
  await page.route('**/api/recognize', route => {
    calls++;
    return route.fulfill({ json: {
      requestId: 'hand-editing-fixture', modelVersion: 'contract-only', image: { width: 700, height: 300 },
      quality: { status: tiles.length ? 'ok' : 'needs_retake', reasons: [] }, timings: { totalMs: 10 },
      tiles: tiles.map((classId, i) => ({ id: `detected-${i}`, classId, label: classId,
        bbox: [5 + i * 40, 30, 32, 90], score: .99, uncertain: false })),
    } });
  });
  await page.goto('/');
  const data = await page.evaluate(() => {
    const c = document.createElement('canvas'); c.width = 700; c.height = 300;
    c.getContext('2d')!.fillRect(0, 0, 700, 300);
    return c.toDataURL('image/jpeg').split(',')[1];
  });
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles({ name: 'synthetic.jpg', mimeType: 'image/jpeg', buffer: Buffer.from(data, 'base64') });
  await page.getByRole('button', { name: '开始识别', exact: true }).click();
  await expect(page.getByRole('region', { name: '识别结果', exact: true })).toBeVisible();
  return () => calls;
}

test('320px: supplement missing tile, delete, correct and undo recompute without another GPU request', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 });
  const calls = await recognizeFixture(page, fullHand.slice(0, 13));
  await expect(page.getByRole('heading', { name: '13 张麻将', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '＋ 添加麻将牌', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '五万', exact: true }).click();
  await page.getByRole('button', { name: '完成补牌', exact: true }).click();
  await expect(page.getByRole('heading', { name: '14 张麻将', exact: true })).toBeVisible();
  await expect(page.getByTestId('score-total')).toHaveText('9');
  await expect(page.locator('.annotation-layer g')).toHaveCount(13);
  await expect(page.locator('.manual-added')).toHaveCount(1);
  await page.getByRole('button', { name: '第 14 张 五万，手动补入', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '东', exact: true }).click();
  await expect(page.getByText('当前牌面暂未组成胡牌', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '第 14 张 东，手动补入', exact: true }).click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  await expect(page.getByRole('heading', { name: '13 张麻将', exact: true })).toBeVisible();
  await page.locator('.tile-button').first().click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  await expect(page.getByRole('heading', { name: '12 张麻将', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '撤销调整', exact: true }).click();
  await expect(page.getByRole('heading', { name: '13 张麻将', exact: true })).toBeVisible();
  await expect(page.locator('.manual-added')).toHaveCount(0);
  expect(calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('390px: empty recognition supports consecutive additions, four-copy limit and manual deletion without fake boxes', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const calls = await recognizeFixture(page, []);
  await page.getByRole('button', { name: '＋ 添加麻将牌', exact: true }).click();
  const one = page.getByRole('dialog').getByRole('button', { name: '一万', exact: true });
  for (let i = 0; i < 4; i++) await one.click();
  await expect(one).toBeDisabled();
  await expect(page.getByRole('status').filter({ hasText: '已添加一万，当前 4 张' })).toBeVisible();
  await page.getByRole('button', { name: '完成补牌', exact: true }).click();
  await expect(page.locator('.manual-added')).toHaveCount(4);
  await expect(page.locator('.annotation-layer g')).toHaveCount(0);
  await page.locator('.manual-added').first().click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  await expect(page.getByRole('heading', { name: '3 张麻将', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '撤销调整', exact: true }).click();
  await expect(page.locator('.tile-button')).toHaveCount(0);
  expect(calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
