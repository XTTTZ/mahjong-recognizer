import { createHash } from 'node:crypto';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { loadWeights } from './weights';
import { type ModelFile } from './profiles';

const detector = 'detector.onnx', classifier = 'classifier.onnx';
const content = { [detector]: '1234', [classifier]: 'abcdef' };
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const manifest = {
  filesBytes: { [detector]: 4, [classifier]: 6 },
  filesSha256: { [detector]: hash(content[detector]), [classifier]: hash(content[classifier]) },
};
beforeEach(() => vi.stubGlobal('self', { location: { origin: 'https://example.test' } }));
afterEach(() => vi.unstubAllGlobals());

it('starts both compact reads together, reports combined progress, then initializes in order', async () => {
  const calls: ModelFile[] = [], events: string[] = [], statuses: string[] = [];
  const replies = new Map<ModelFile, (value: Response) => void>();
  vi.stubGlobal('fetch', (url: string) => {
    const name = url.split('/').pop() as ModelFile;
    calls.push(name);
    return new Promise<Response>(resolve => replies.set(name, resolve));
  });
  const loading = loadWeights('compact-v10', manifest, text => statuses.push(text), async name => {
    events.push('start ' + name);
    await Promise.resolve();
    events.push('done ' + name);
  });
  await vi.waitFor(() => expect(calls).toEqual([detector, classifier]));
  replies.get(classifier)!(new Response(content[classifier]));
  await vi.waitFor(() => expect(statuses).toContain('正在读取模型文件 60%'));
  expect(events).toEqual([]);
  replies.get(detector)!(new Response(content[detector]));
  await loading;
  expect(events).toEqual(['start ' + detector, 'done ' + detector, 'start ' + classifier, 'done ' + classifier]);
});

it('aborts the other compact request if a file fails SHA verification', async () => {
  let siblingSignal: AbortSignal | undefined;
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith(detector)) return new Response('xxxx');
    siblingSignal = init.signal!;
    return new Promise<Response>((_, reject) => siblingSignal!.addEventListener('abort', () => reject(siblingSignal!.reason)));
  });
  const create = vi.fn();
  await expect(loadWeights('compact-v10', manifest, () => {}, create)).rejects.toThrow('定位模型校验失败');
  expect(siblingSignal?.aborted).toBe(true);
  expect(create).not.toHaveBeenCalled();
});

it('finishes creating the first large reference session before fetching the second file', async () => {
  const events: string[] = [];
  vi.stubGlobal('fetch', async (url: string) => {
    const name = url.split('/').pop() as ModelFile;
    events.push('fetch ' + name);
    return new Response(content[name]);
  });
  await loadWeights('portable-v1', manifest, () => {}, async name => { events.push('create ' + name); });
  expect(events).toEqual(['fetch ' + detector, 'create ' + detector, 'fetch ' + classifier, 'create ' + classifier]);
});
