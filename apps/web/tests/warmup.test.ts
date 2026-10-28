import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class FakeStorage {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

function browser(initialVisibility = 'visible', online = true, storageBlocked = false) {
  const document = Object.assign(new EventTarget(), { visibilityState: initialVisibility });
  const window = new EventTarget();
  const storage = new FakeStorage();
  const navigator = { onLine: online };
  Object.defineProperty(window, 'sessionStorage', { get: () => {
    if (storageBlocked) throw new Error('storage-disabled');
    return storage;
  } });
  vi.stubGlobal('document', document);
  vi.stubGlobal('window', window);
  vi.stubGlobal('navigator', navigator);
  return { document, window, navigator, storage };
}

beforeEach(() => {
  vi.resetModules();
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-10-29T00:00:00Z'));
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('optional visible-page GPU warmup', () => {
  it('waits for first visibility and sends only one empty warmup POST', async () => {
    const env = browser('hidden');
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 202 }));
    vi.stubGlobal('fetch', fetch);
    const { warmupWhenVisible } = await import('../src/warmup');
    const cleanup = warmupWhenVisible();
    expect(fetch).not.toHaveBeenCalled();
    env.document.visibilityState = 'visible';
    env.document.dispatchEvent(new Event('visibilitychange'));
    env.document.dispatchEvent(new Event('visibilitychange'));
    await vi.advanceTimersByTimeAsync(0);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0]).toEqual(['/api/warmup', expect.objectContaining({ method: 'POST', cache: 'no-store', credentials: 'same-origin' })]);
    expect(fetch.mock.calls[0][1]).not.toHaveProperty('body');
    cleanup();
  });

  it('deduplicates remounts/reloads in this session and allows a later explicit page visit', async () => {
    browser();
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 202 }));
    vi.stubGlobal('fetch', fetch);
    const { warmupWhenVisible, WARMUP_COOLDOWN_MS } = await import('../src/warmup');
    warmupWhenVisible()();
    warmupWhenVisible()();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetch).toHaveBeenCalledTimes(1);
    vi.resetModules(); // Same sessionStorage survives a new page module instance.
    const next = await import('../src/warmup');
    next.warmupWhenVisible()();
    expect(fetch).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(WARMUP_COOLDOWN_MS + 1);
    next.warmupWhenVisible()();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('waits for connectivity and cleanup prevents a hidden abandoned page from warming', async () => {
    const env = browser('hidden', false);
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 202 }));
    vi.stubGlobal('fetch', fetch);
    const { warmupWhenVisible } = await import('../src/warmup');
    const cleanup = warmupWhenVisible();
    env.document.visibilityState = 'visible';
    env.document.dispatchEvent(new Event('visibilitychange'));
    expect(fetch).not.toHaveBeenCalled();
    cleanup();
    env.navigator.onLine = true;
    env.window.dispatchEvent(new Event('online'));
    expect(fetch).not.toHaveBeenCalled();
    warmupWhenVisible();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('falls back to memory when session metadata storage is unavailable', async () => {
    browser('visible', true, true);
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 202 }));
    vi.stubGlobal('fetch', fetch);
    const { warmupWhenVisible } = await import('../src/warmup');
    expect(() => warmupWhenVisible()).not.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    warmupWhenVisible();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('swallows warmup failures without retrying or affecting an explicit recognition request', async () => {
    browser();
    const fetch = vi.fn().mockRejectedValueOnce(new TypeError('network offline'));
    vi.stubGlobal('fetch', fetch);
    const { warmupWhenVisible } = await import('../src/warmup');
    expect(() => warmupWhenVisible()).not.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    warmupWhenVisible();
    expect(fetch).toHaveBeenCalledTimes(1);
    fetch.mockResolvedValueOnce(Response.json({ requestId: 'explicit-request', modelVersion: 'fixture',
      image: { width: 100, height: 100 }, tiles: [], quality: { status: 'needs_retake', reasons: [] }, timings: { totalMs: 10 } }));
    const { recognize } = await import('../src/api');
    await expect(recognize(new File(['image'], 'hand.jpg', { type: 'image/jpeg' }), new AbortController().signal))
      .resolves.toMatchObject({ requestId: 'explicit-request' });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls.map(call => call[0])).toEqual(['/api/warmup', '/api/recognize']);
  });

  it('bounds a stalled warmup and never turns it into an automatic recognition or retry', async () => {
    const env = browser();
    const fetch = vi.fn((_url: string, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
    }));
    vi.stubGlobal('fetch', fetch);
    const { warmupWhenVisible, WARMUP_TIMEOUT_MS, WARMUP_COOLDOWN_MS } = await import('../src/warmup');
    warmupWhenVisible();
    await vi.advanceTimersByTimeAsync(WARMUP_COOLDOWN_MS - 1);
    warmupWhenVisible();
    await vi.advanceTimersByTimeAsync(WARMUP_TIMEOUT_MS - WARMUP_COOLDOWN_MS + 2);
    expect(fetch.mock.calls[0][1].signal?.aborted).toBe(true);
    env.document.dispatchEvent(new Event('visibilitychange'));
    env.window.dispatchEvent(new Event('online'));
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
});
