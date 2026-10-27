import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './apps/web/tests',
  testMatch: '**/*.spec.ts',
  fullyParallel: true,
  use: { baseURL: 'http://127.0.0.1:5177', browserName: 'chromium', headless: true },
  webServer: { command: 'npm run dev -- --port 5177', port: 5177, reuseExistingServer: true },
  reporter: [['list']],
  outputDir: 'artifacts/web/test-output',
});
