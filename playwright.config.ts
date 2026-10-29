import { defineConfig } from '@playwright/test';
const deployedBaseURL = process.env.PLAYWRIGHT_BASE_URL;
export default defineConfig({
  testDir: './apps/web/tests',
  testMatch: '**/*.spec.ts',
  fullyParallel: true,
  use: { baseURL: deployedBaseURL ?? 'http://127.0.0.1:5177', browserName: 'chromium', headless: true },
  webServer: deployedBaseURL ? undefined : { command: 'npm run dev -- --port 5177', port: 5177, reuseExistingServer: true },
  reporter: [['list']],
  outputDir: 'artifacts/web/test-output',
});
