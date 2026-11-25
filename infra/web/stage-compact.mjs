// Stage the exact public release alongside the web build for Cloudflare Static Assets.
import { createHash } from 'node:crypto';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const cache = path.join(root, 'artifacts/models/releases/v10-nano800-kd224-fp32-20261109');
const out = path.join(root, 'dist/models/compact-v10');
const release = 'https://github.com/XTTTZ/mahjong-recognizer/releases/download/runtime-compact-fp32-20261109/';
const files = [
  ['manifest.json', 5699, '6525882d3aa05a35dd0d4d6070dc47c727701973c3b871e5700c8a6c899104b2'],
  ['detector.onnx', 12840290, 'ffa2ae66b67fb385cb5810fabab893dc903e5d77346998eb48a6952c679388d5'],
  ['classifier.onnx', 17013455, 'ea022a0ee9eab89fd82a4b8c1ce5c0ceedca0e0619376e8a9d70865e055684e3'],
];
await stat(path.join(root, 'dist/index.html'));
await mkdir(out, { recursive: true });
for (const [name, size, sha256] of files) {
  let bytes;
  try {
    bytes = await readFile(path.join(cache, name));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const response = await fetch(release + name, { signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw new Error('Cannot fetch public release file ' + name);
    bytes = Buffer.from(await response.arrayBuffer());
  }
  if (bytes.length !== size || createHash('sha256').update(bytes).digest('hex') !== sha256) {
    throw new Error('Invalid public release file ' + name);
  }
  await writeFile(path.join(out, name), bytes);
}
console.log('Verified and staged the compact model for static delivery.');
