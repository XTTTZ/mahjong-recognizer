import { expect, test } from '@playwright/test';

// Cold-route checks intentionally never opt into model loading or inference.
for (const pathname of ['/', '/benchmark']) {
  test(`${pathname}: local inference resources stay unloaded before explicit opt-in`, async ({ page }) => {
    const localResources: string[] = [];
    const gpuRequests: string[] = [];
    let warmups = 0;
    page.on('request', request => {
      const url = new URL(request.url());
      if (/\/models\/|\/ort\/|local[._-]worker|onnxruntime[-_]web/i.test(url.pathname)) localResources.push(url.pathname);
      if (url.pathname === '/api/recognize' || url.hostname.endsWith('.modal.run')) gpuRequests.push(request.url());
    });
    await page.route('**/api/warmup', route => { warmups++; return route.fulfill({ json: { ready: true } }); });
    await page.route('**/api/health', route => route.fulfill({ json: { configured: true, inferenceProbed: false } }));
    // These guards also prevent accidental heavy downloads if lazy loading regresses.
    await page.route('**/models/**', route => route.abort());
    await page.route('**/ort/**', route => route.abort());
    await page.route('**/api/recognize', route => route.abort());
    await page.route('https://**.modal.run/**', route => route.abort());
    await page.goto(pathname);
    if (pathname === '/') {
      await expect(page.getByRole('button', { name: '拍一张', exact: true })).toBeVisible();
      await expect(page.getByRole('link', { name: '设备性能测试', exact: true })).toHaveAttribute('href', '/benchmark');
    } else {
      await expect(page.getByRole('button', { name: '下载并加载模型', exact: true })).toBeEnabled();
      await expect(page.getByRole('button', { name: '开始本地测试', exact: true })).toBeDisabled();
    }
    await page.waitForLoadState('networkidle');
    expect(localResources).toEqual([]);
    expect(gpuRequests).toEqual([]);
    if (pathname === '/benchmark') expect(warmups).toBe(0);
  });
}
