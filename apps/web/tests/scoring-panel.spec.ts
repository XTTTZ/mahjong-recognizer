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

for (const width of [320, 390, 430]) {
  test(`${width}px: compact photo expands, result navigation moves focus, and touch controls stay usable`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    const api = await mockHand(page);
    await page.goto('/');
    await upload(page);
    const review = page.locator('#hand-review');
    const score = page.locator('#hand-score');
    await expect(review).toBeFocused();
    const photo = page.getByRole('region', { name: '手牌照片', exact: true });
    await expect(photo).toHaveClass(/compact-photo/);
    const compactHeight = (await photo.locator('.photo-canvas').boundingBox())!.height;
    await expect(photo.getByRole('button', { name: '展开', exact: true })).toHaveAttribute('aria-expanded', 'false');
    await photo.getByRole('button', { name: '展开', exact: true }).click();
    await expect(photo).not.toHaveClass(/compact-photo/);
    await expect(photo.getByRole('button', { name: '收起', exact: true })).toHaveAttribute('aria-expanded', 'true');
    expect((await photo.locator('.photo-canvas').boundingBox())!.height).toBeGreaterThan(compactHeight);
    await expect(photo.getByRole('button', { name: '选取要识别的牌面区域' })).toBeEnabled();
    await expect(photo.getByRole('button', { name: '向右旋转照片' })).toBeEnabled();
    await photo.getByRole('button', { name: '收起', exact: true }).click();
    await expect(photo).toHaveClass(/compact-photo/);
    await page.getByRole('link', { name: '胡牌 / 听牌', exact: true }).click();
    await expect(score).toBeFocused();
    const navBottom = (await page.getByRole('navigation', { name: '结果快速导航' }).boundingBox())!.y
      + (await page.getByRole('navigation', { name: '结果快速导航' }).boundingBox())!.height;
    expect((await score.boundingBox())!.y).toBeGreaterThanOrEqual(navBottom);
    expect((await score.boundingBox())!.y).toBeLessThan(844);
    await page.getByRole('link', { name: '核对 14 张牌', exact: true }).click();
    await expect(review).toBeFocused();
    await page.locator('.tile-button').first().click();
    await page.getByRole('dialog').getByRole('button', { name: '二万', exact: true }).click();
    await expect(page.locator('.tile-button').first()).toBeFocused();
    // A correction keeps the user's place; only a new request focuses the results section.
    await expect(review).not.toBeFocused();
    const targets = await page.locator('button:visible, a:visible, select:visible, input[type="number"]:visible, summary:visible')
      .evaluateAll(nodes => nodes.map(node => ({ label: node.getAttribute('aria-label') ?? node.textContent?.trim(),
        width: node.getBoundingClientRect().width, height: node.getBoundingClientRect().height })));
    expect(targets.filter(target => target.width < 44 || target.height < 44)).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(api.calls()).toBe(1);
  });
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

test('390px: only sufficient ungrouped faces can be declared and used tiles disappear', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const api = await mockHand(page);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^碰杠/ }).click();
  const faces = panel.getByLabel('分组牌面', { exact: true });
  await expect(faces.locator('option')).toHaveText(['一万（未分组3张）']);
  await expect(faces.locator('option[value="5z"]')).toHaveCount(0);
  await panel.getByRole('button', { name: '明杠', exact: true }).click();
  await expect(faces).toBeDisabled();
  await expect(faces).toHaveValue('');
  await expect(faces.locator('option')).toHaveText(['暂无可声明牌面']);
  await expect(panel.getByRole('button', { name: '声明明杠', exact: true })).toBeDisabled();
  await expect(panel.getByText('当前没有未分组满4张的牌，不能声明明杠。', { exact: true })).toBeVisible();
  await expect(panel.getByRole('list', { name: '已声明碰杠' })).toHaveCount(0);
  await panel.getByRole('button', { name: '碰', exact: true }).click();
  await panel.getByLabel('分组牌面', { exact: true }).selectOption('1m');
  await panel.getByRole('button', { name: '声明碰', exact: true }).click();
  await expect(panel.getByRole('list', { name: '已声明碰杠' })).toContainText('碰 · 一万 ×3');
  await expect(faces.locator('option[value="1m"]')).toHaveCount(0);
  await expect(panel.getByRole('button', { name: '声明碰', exact: true })).toBeDisabled();
  await expect(page.locator('.tile-button')).toHaveCount(14);
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  await expect(panel.getByRole('checkbox', { name: '门清', exact: true })).toBeDisabled();
  await panel.getByRole('button', { name: '删除第1组碰', exact: true }).click();
  await expect(panel.getByRole('checkbox', { name: '门清', exact: true })).toBeEnabled();
  await expect(faces.locator('option')).toHaveText(['一万（未分组3张）']);
  expect(api.calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('320px: live one-away candidates keep a zero-base wait conditional despite historical selections and code hits', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 844 });
  const api = await mockHand(page);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  await panel.getByRole('checkbox', { name: '天胡', exact: true }).check();
  await panel.locator('summary').filter({ hasText: /^中码/ }).click();
  await panel.getByLabel('中码数', { exact: true }).fill('2');
  await expect(panel.getByTestId('score-total')).toHaveText('150');
  await page.locator('.tile-button').last().click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  const waits = panel.getByRole('region', { name: '听牌候选', exact: true });
  await expect(waits).toBeVisible();
  await expect(panel.getByRole('alert')).toHaveCount(0);
  await expect(waits.getByRole('list', { name: '基础已满3倍', exact: true })).toHaveCount(0);
  const conditional = waits.getByRole('list', { name: '需特殊条件，基础不足3倍', exact: true });
  await expect(conditional.locator('[data-tile="5z"]')).toContainText('白板');
  await expect(conditional.locator('[data-tile="5z"]')).toContainText('基础0倍');
  await expect(waits.getByText('不足3倍的候选不能直接胡。', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: '＋ 添加麻将牌', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '白板', exact: true }).click();
  await page.getByRole('button', { name: '完成补牌', exact: true }).click();
  await expect(waits).toHaveCount(0);
  expect(api.calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await fs.mkdir('artifacts/web/scoring', { recursive: true });
  // Return to 13 tiles for a readable conditional-wait screenshot.
  await page.locator('.tile-button').last().click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  await panel.screenshot({ path: 'artifacts/web/scoring/waits-conditional-320.png' });
});

test('390px: a declared kong uses physical tiles and produces 13-plus-kong base-qualified waits', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const cards = ['1m', '1m', '1m', '1m', '2p', '2p', '2p', '3s', '3s', '3s', '4m', '4m', '4m', '5z'];
  const api = await mockHand(page, cards);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^碰杠/ }).click();
  await panel.getByRole('button', { name: '暗杠', exact: true }).click();
  await expect(panel.getByLabel('分组牌面', { exact: true }).locator('option')).toHaveText(['一万（未分组4张）']);
  await panel.getByRole('button', { name: '声明暗杠', exact: true }).click();
  const waits = panel.getByRole('region', { name: '听牌候选', exact: true });
  await expect(waits.getByText('当前14张 · 再进1张', { exact: true })).toBeVisible();
  const qualified = waits.getByRole('list', { name: '基础已满3倍', exact: true });
  await expect(qualified.locator('[data-tile="5z"]')).toContainText('白板');
  await expect(qualified.locator('[data-tile="5z"]')).toContainText('基础3倍');
  await expect(qualified.locator('[data-tile="5z"]')).toContainText('碰碰胡');
  await expect(waits.locator('[data-tile="1m"]')).toHaveCount(0);
  await expect(panel.getByLabel('分组牌面', { exact: true }).locator('option[value="1m"]')).toHaveCount(0);
  await expect(page.locator('.tile-button')).toHaveCount(14);
  expect(api.calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await panel.screenshot({ path: 'artifacts/web/scoring/waits-kong-390.png' });
});

test('430px: seven-pairs interpretation permits real meld correction, while incompatible bonuses cannot be reselected', async ({ page }) => {
  await page.setViewportSize({ width: 430, height: 844 });
  const luxury = ['1m', '1m', '1m', '1m', '2m', '2m', '3m', '3m', '4m', '4m', '5m', '5m', '6m', '6m'];
  const api = await mockHand(page, luxury);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  await expect(panel.getByRole('checkbox', { name: '门清', exact: true })).toBeDisabled();
  await panel.locator('summary').filter({ hasText: /^碰杠/ }).click();
  await panel.getByRole('button', { name: '暗杠', exact: true }).click();
  await expect(panel.getByRole('button', { name: '声明暗杠', exact: true })).toBeEnabled();
  await expect(panel.locator('p.scoring-hint').filter({ hasText: /七对.*不能有任何/ })).toBeVisible();
  await panel.getByRole('button', { name: '声明暗杠', exact: true }).click();
  await expect(panel.locator('.scoring-breakdown')).toHaveCount(0);
  await expect(panel.getByRole('checkbox', { name: '门清', exact: true })).toBeDisabled();
  await expect(panel.getByRole('region', { name: '听牌候选', exact: true })).toBeVisible();
  await panel.getByRole('button', { name: '删除第1组暗杠', exact: true }).click();
  await expect(panel.locator('.scoring-breakdown')).toContainText('豪华七对');
  expect(api.calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('430px: thirteen-orphans waits draw tile faces and fold the long list without overflow', async ({ page }) => {
  await page.setViewportSize({ width: 430, height: 844 });
  const orphanTiles = ['1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z', '4z', '5z', '6z', '7z'];
  const api = await mockHand(page, orphanTiles);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  const waits = panel.getByRole('region', { name: '听牌候选', exact: true });
  await expect(waits.getByRole('list', { name: '基础已满3倍', exact: true }).locator('li')).toHaveCount(6);
  await expect(waits.locator('[data-tile="1m"]')).toContainText('十三幺');
  await expect(waits.locator('[data-tile="1m"] .tile-face')).toHaveCount(1);
  await waits.getByText('查看其余7种候选', { exact: true }).click();
  await expect(waits.locator('li:visible')).toHaveCount(13);
  await expect(waits.getByRole('list', { name: '基础已满3倍 · 更多', exact: true })).toBeVisible();
  expect(api.calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await panel.screenshot({ path: 'artifacts/web/scoring/waits-orphans-430.png' });
});

test('430px: a hand with both seven-pair and standard routes can declare an actual pung and stop scoring seven pairs', async ({ page }) => {
  await page.setViewportSize({ width: 430, height: 844 });
  const dualRoute = ['1m', '1m', '1m', '1m', '2m', '2m', '2m', '2m', '3m', '3m', '3m', '3m', '4m', '4m'];
  const api = await mockHand(page, dualRoute);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await expect(panel.locator('.scoring-breakdown')).toContainText('七对');
  await panel.locator('summary').filter({ hasText: /^碰杠/ }).click();
  await panel.getByLabel('分组牌面', { exact: true }).selectOption('1m');
  await expect(panel.getByRole('button', { name: '声明碰', exact: true })).toBeEnabled();
  await panel.getByRole('button', { name: '声明碰', exact: true }).click();
  await expect(panel.getByTestId('score-total')).toHaveText('6');
  await expect(panel.locator('.scoring-breakdown')).not.toContainText('七对');
  await expect(panel.locator('.scoring-breakdown')).toContainText('清一色');
  await panel.getByRole('button', { name: '删除第1组碰', exact: true }).click();
  await expect(panel.locator('.scoring-breakdown')).toContainText('七对');
  expect(api.calls()).toBe(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('390px: four concealed triplets and menqing require cancelling the conflicting prior selection', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const allPungs = ['1m', '1m', '1m', '2p', '2p', '2p', '3s', '3s', '3s', '4m', '4m', '4m', '5z', '5z'];
  const api = await mockHand(page, allPungs);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  const menqing = panel.getByRole('checkbox', { name: '门清', exact: true });
  const fourConcealed = panel.getByRole('checkbox', { name: '四暗刻', exact: true });
  await menqing.check();
  await expect(fourConcealed).toBeDisabled();
  await expect(panel.getByText('四暗刻与门清不叠加；请先取消门清，再按实际牌局确认四暗刻。', { exact: true }).first()).toBeVisible();
  await menqing.uncheck();
  await expect(fourConcealed).toBeEnabled();
  await fourConcealed.check();
  await expect(panel.getByTestId('score-total')).toHaveText('10');
  await expect(menqing).toBeDisabled();
  await expect(panel.getByText('表内：四暗刻不计门清。', { exact: true })).toBeVisible();
  expect(api.calls()).toBe(1);
});

test('390px: changing/deleting faces clears unavailable select values and selected invalid bonuses can only be removed', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const api = await mockHand(page);
  await page.goto('/');
  await upload(page);
  const panel = page.getByRole('region', { name: '胡牌计倍', exact: true });
  await panel.locator('summary').filter({ hasText: /^特殊情况/ }).click();
  const menqing = panel.getByRole('checkbox', { name: '门清', exact: true });
  await menqing.check();
  await panel.getByLabel('和牌张', { exact: true }).selectOption('5z');
  await panel.locator('summary').filter({ hasText: /^碰杠/ }).click();
  await panel.getByRole('button', { name: '声明碰', exact: true }).click();
  await expect(panel.getByText('门清已不适用于当前牌面，不计入倍数；可取消原选择。', { exact: true })).toBeVisible();
  await menqing.uncheck();
  await expect(menqing).toBeDisabled();
  await expect(panel.getByLabel('分组牌面', { exact: true })).toHaveValue('');
  await page.locator('.tile-button').last().click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  await page.locator('.tile-button').last().click();
  await page.getByRole('button', { name: '删除这张牌', exact: true }).click();
  await expect(panel.getByLabel('和牌张', { exact: true })).toHaveValue('');
  await expect(panel.getByLabel('和牌张', { exact: true }).locator('option[value="5z"]')).toHaveCount(0);
  await page.getByRole('button', { name: '＋ 添加麻将牌', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: '白板', exact: true }).click();
  await page.getByRole('button', { name: '完成补牌', exact: true }).click();
  await expect(panel.getByLabel('和牌张', { exact: true }).locator('option[value="5z"]')).toHaveCount(1);
  await expect(panel.getByLabel('和牌张', { exact: true })).toHaveValue('');
  expect(api.calls()).toBe(1);
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
