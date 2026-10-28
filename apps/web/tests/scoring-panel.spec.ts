import { expect, test, type Page } from '@playwright/test';
import fs from 'node:fs/promises';

// Deliberate response fixtures test the local scoring flow, never a GPU endpoint.
const ordinary = ['1m', '1m', '1m', '2m', '3m', '4m', '4p', '5p', '6p', '7s', '8s', '9s', '5z', '5z'];

test.beforeEach(async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { configured: true, inferenceProbed: false } }));
  await page.route('**/api/warmup', route => route.fulfill({ json: { accepted: true } }));
});

async function fixturePhoto(page: Page) {
  const data = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1000; canvas.height = 600;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#356952'; context.fillRect(0, 0, 1000, 600);
    return canvas.toDataURL('image/jpeg').split(',')[1];
  });
  return { name: 'scoring-contract-fixture.jpg', mimeType: 'image/jpeg', buffer: Buffer.from(data, 'base64') };
}

async function upload(page: Page) {
  await page.getByLabel('上传麻将照片', { exact: true }).setInputFiles(await fixturePhoto(page));
  await page.getByRole('button', { name: '开始识别', exact: true }).click();
  await expect(page.getByRole('region', { name: '胡牌计倍', exact: true })).toBeVisible();
}

async function mockHand(page: Page, initial = ordinary) {
  let cards = [...initial];
  let calls = 0;
  await page.route('**/api/recognize', route => {
    calls++;
    return route.fulfill({ json: {
      requestId: `score-contract-${calls}`, modelVersion: 'local-scoring-contract-only',
      image: { width: 1000, height: 600 }, quality: { status: 'ok', reasons: [] }, timings: { totalMs: 10 },
      tiles: cards.map((classId, i) => ({ id: `score-tile-${i}`, classId, label: classId,
        bbox: [10 + (i % 7) * 140, 10 + Math.floor(i / 7) * 180, 100, 150], score: .99, uncertain: false })),
    } });
  });
  return { calls: () => calls, setCards: (next: string[]) => { cards = next; } };
}

test('320px: manual menqing reaches three, codes multiply locally, ordinary self draw adds nothing', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 });
  const api = await mockHand(page);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await expect(panel.getByText('3倍起胡', { exact: true })).toBeVisible();
  await expect(panel.getByTestId('score-total')).toHaveText('0');
  await expect(panel.getByText('不足3倍，暂不能胡', { exact: true })).toBeVisible();
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  await panel.getByLabel('和牌方式', { exact: true }).selectOption('self-draw');
  await expect(panel.getByTestId('score-total')).toHaveText('0');
  await panel.getByRole('checkbox', { name: '门清', exact: true }).check();
  await expect(panel.getByTestId('score-total')).toHaveText('3');
  await panel.locator('summary').filter({ hasText: /^中码/ }).click();
  await panel.getByLabel('中码数', { exact: true }).fill('2');
  await expect(panel.getByTestId('score-total')).toHaveText('9');
  await expect(panel.getByText('达到起胡门槛', { exact: true })).toBeVisible();
  await panel.locator('summary').filter({ hasText: /^碰杠/ }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  const heights = await panel.locator('button:visible, select:visible, input[type="number"]:visible, summary:visible, .scoring-choice:visible')
    .evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
  expect(heights.every(height => height >= 44)).toBe(true);
  expect(api.calls()).toBe(1);
  await fs.mkdir('artifacts/web/scoring', { recursive: true });
  await panel.screenshot({ path: 'artifacts/web/scoring/panel-320.png' });
});

test('390px: meld declarations consume existing physical tiles and never invent missing kong tiles', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const api = await mockHand(page);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^碰杠/ }).click();
  await panel.getByRole('button', { name: '明杠', exact: true }).click();
  await panel.getByLabel('分组牌面', { exact: true }).selectOption('5z');
  await panel.getByRole('button', { name: '声明明杠', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('需要4张白板');
  await expect(panel.getByRole('alert')).toContainText('只有2张');
  await expect(panel.getByRole('list', { name: '已声明碰杠' })).toHaveCount(0);
  await panel.getByRole('button', { name: '碰', exact: true }).click();
  await panel.getByLabel('分组牌面', { exact: true }).selectOption('1m');
  await panel.getByRole('button', { name: '声明碰', exact: true }).click();
  await expect(panel.getByRole('list', { name: '已声明碰杠' })).toContainText('碰 · 一万 ×3');
  await expect(page.locator('.tile-button')).toHaveCount(14);
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  await expect(panel.getByRole('checkbox', { name: '门清', exact: true })).toBeDisabled();
  await panel.getByRole('button', { name: '删除第1组碰', exact: true }).click();
  await expect(panel.getByRole('checkbox', { name: '门清', exact: true })).toBeEnabled();
  expect(api.calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

for (const invalid of [
  { label: 'five identical tiles', cards: ['1m', '1m', '1m', '1m', '1m', '2m', '3m', '4m', '4p', '5p', '6p', '7s', '8s', '9s'] },
  { label: 'a flower in a 136-tile hand', cards: [...ordinary.slice(0, 13), '1h'] },
]) {
  test(`invalid ${invalid.label} has no valid total multiplier`, async ({ page }) => {
    await mockHand(page, invalid.cards);
    await page.goto('/');
    await upload(page);
    const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
    await expect(panel.getByTestId('score-total')).toHaveText('—');
    await expect(panel.getByRole('alert')).toBeVisible();
    await expect(panel.getByText('达到起胡门槛', { exact: true })).toHaveCount(0);
  });
}

test('430px: replacing a recognition resets all manual conditions without sending scoring requests', async ({ page }) => {
  await page.setViewportSize({ width: 430, height: 844 });
  const api = await mockHand(page);
  await page.goto('/');
  await upload(page);
  let panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  await panel.getByRole('checkbox', { name: '门清', exact: true }).check();
  await panel.locator('summary').filter({ hasText: /^中码/ }).click();
  await panel.getByLabel('中码数', { exact: true }).fill('2');
  await expect(panel.getByTestId('score-total')).toHaveText('9');
  await upload(page);
  panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await expect(panel.getByTestId('score-total')).toHaveText('0');
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  await expect(panel.getByRole('checkbox', { name: '门清', exact: true })).not.toBeChecked();
  await panel.locator('summary').filter({ hasText: /^中码/ }).click();
  await expect(panel.getByLabel('中码数', { exact: true })).toHaveValue('0');
  expect(api.calls()).toBe(2);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
