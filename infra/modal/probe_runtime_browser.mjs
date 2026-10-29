/** Protected, no-retry Modal probe using Playwright's HTTP client and TLS verification. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { request } from '@playwright/test';

const args = process.argv.slice(2);
const option = (name, fallback) => { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1]; };
const endpoint = option('--endpoint');
const imagePath = option('--image');
const output = option('--out');
const recognitions = Number(option('--recognitions', '1'));
assert.ok(endpoint && imagePath && output, '--endpoint, --image and a fresh --out are required');
assert.ok(Number.isInteger(recognitions) && recognitions >= 1 && recognitions <= 2);
const base = endpoint.replace(/\/recognize\/?$/, '').replace(/\/$/, '');
assert.equal(new URL(base).protocol, 'https:');
assert.ok(new URL(base).hostname.endsWith('.modal.run'));
assert.equal(new URL(base).search, '');
assert.equal(new URL(base).pathname, '/');
assert.equal(await fs.stat(output).then(() => true).catch(() => false), false, 'Preserve prior evidence: choose a fresh --out');
const token = (await fs.readFile(option('--token-file', '.secrets/modal-proxy-token'), 'utf8')).trim();
assert.ok(token, 'Empty proxy token file');
const photo = await fs.readFile(imagePath);
const report = { endpoint: base, observedAt: new Date().toISOString(), client: 'Playwright APIRequestContext',
  httpsCertificateValidation: true, automaticRetries: 0, maxRedirects: 0, modelTraining: false,
  plannedRecognitions: recognitions, complete: false, probes: [] };
const context = await request.newContext({ ignoreHTTPSErrors: false, timeout: 110_000,
  extraHTTPHeaders: { Authorization: `Bearer ${token}` } });
await fs.mkdir(path.dirname(output), { recursive: true });
const persist = () => fs.writeFile(output, JSON.stringify(report, null, 2) + '\n');
await persist();
try {
  for (let i = -1; i < recognitions; i++) {
    const urlPath = i < 0 ? '/health' : '/recognize';
    const started = performance.now();
    const response = i < 0
      ? await context.get(base + urlPath, { maxRedirects: 0, maxRetries: 0 })
      : await context.post(base + urlPath, { maxRedirects: 0, maxRetries: 0,
        multipart: { image: { name: 'authorized-validation.jpg', mimeType: 'image/jpeg', buffer: photo } } });
    const value = await response.json().catch(() => ({}));
    const row = { path: urlPath, status: response.status(), wallMs: Math.round((performance.now() - started) * 10) / 10 };
    if (i < 0) row.health = value;
    else Object.assign(row, { repeat: i, modelVersion: value.modelVersion, tiles: value.tiles?.length ?? 0,
      timings: value.timings, quality: value.quality, classIds: (value.tiles ?? []).map(tile => tile.classId) });
    report.probes.push(row);
    await persist();
    console.log(JSON.stringify(row));
    assert.equal(row.status, 200, 'Non-200 response: no further requests submitted');
  }
  report.complete = true;
} catch (error) {
  // Playwright's full call log may contain request headers. Never print/store it.
  report.error = { name: error.name, summary: String(error.message).split('\n')[0].replaceAll(token, '[REDACTED]') };
  report.failedAfterSuccessfulResponses = report.probes.length;
  process.exitCode = 1;
} finally {
  await context.dispose();
  await persist();
  console.log(JSON.stringify({ output, complete: report.complete, probes: report.probes.length, ...(report.error ? { error: report.error } : {}) }));
}
