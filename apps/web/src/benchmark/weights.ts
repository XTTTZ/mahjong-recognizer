import { MODEL_FILES, profilePath, type Manifest, type ModelFile, type ProfileId } from './profiles';

type Weights = Uint8Array<ArrayBuffer>;
type TransferManifest = Pick<Manifest, 'filesBytes' | 'filesSha256'>;
const label = (name: ModelFile) => name === 'detector.onnx' ? '定位模型' : '分类模型';

async function digestMatches(bytes: Weights, expected: string) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('') === expected;
}

async function download(profile: ProfileId, name: ModelFile, model: TransferManifest, signal: AbortSignal,
  advance: (bytes: number) => void): Promise<Weights> {
  const url = profilePath(profile, name);
  const expected = model.filesBytes[name];
  const hash = model.filesSha256[name];
  const key = new URL(url, self.location.origin);
  key.searchParams.set('sha256', hash);
  let cache: Cache | undefined;
  if (profile === 'compact-v10') {
    try {
      cache = await caches.open('mahjong-compact-weights-v1');
      const stored = await cache.match(key.href);
      if (stored) {
        const bytes = new Uint8Array(await stored.arrayBuffer());
        if (bytes.byteLength === expected && await digestMatches(bytes, hash)) {
          advance(expected);
          return bytes;
        }
        await cache.delete(key.href);
      }
    } catch { /* Storage is optional on devices that restrict it. */ }
  }

  signal.throwIfAborted();
  const response = await fetch(url, { credentials: 'omit', signal }).catch(() => {
    throw new Error(`${label(name)}下载中断，请检查网络后重试`);
  });
  if (!response.ok || !response.body) throw new Error(`${label(name)}下载失败，请稍后再试`);
  const bytes = new Uint8Array(expected);
  const reader = response.body.getReader();
  let offset = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (offset + value.length > expected) throw new Error(`${label(name)}文件大小不符`);
      bytes.set(value, offset);
      offset += value.length;
      advance(value.length);
    }
  } catch (error) {
    if (error instanceof TypeError) throw new Error(`${label(name)}下载中断，请检查网络后重试`);
    throw error;
  } finally { reader.releaseLock(); }
  if (offset !== expected || !await digestMatches(bytes, hash)) throw new Error(`${label(name)}校验失败，请重新准备`);
  signal.throwIfAborted();
  if (cache) {
    try { await cache.put(key.href, new Response(bytes)); } catch { /* In-memory inference remains available. */ }
  }
  return bytes;
}

export async function loadWeights(profile: ProfileId, model: TransferManifest, status: (text: string) => void,
  createSession: (name: ModelFile, bytes: Weights) => Promise<void>) {
  const controller = new AbortController();
  const total = MODEL_FILES.reduce((sum, name) => sum + model.filesBytes[name], 0);
  let received = 0, previousPercent = -1, downloadAndHashMs = 0, sessionCreateMs = 0;
  function advance(count: number) {
    received += count;
    const percent = Math.floor(100 * received / total);
    if (percent === previousPercent || controller.signal.aborted) return;
    previousPercent = percent;
    status(percent === 100 ? '正在校验模型文件…' : `正在读取模型文件 ${percent}%`);
  }

  // The 119 MB comparison profile keeps one weight buffer live at a time.
  const groups = profile === 'compact-v10' ? [MODEL_FILES] : MODEL_FILES.map(name => [name]);
  try {
    for (const names of groups) {
      previousPercent = -1;
      advance(0);
      const start = performance.now();
      const buffers = await Promise.all(names.map(name => download(profile, name, model, controller.signal, advance)));
      downloadAndHashMs += performance.now() - start;
      for (const [index, name] of names.entries()) {
        status(`正在初始化${label(name)}…`);
        const started = performance.now();
        await createSession(name, buffers[index]);
        sessionCreateMs += performance.now() - started;
      }
    }
  } catch (error) {
    controller.abort();
    throw error;
  }
  return { downloadAndHashMs, sessionCreateMs };
}
