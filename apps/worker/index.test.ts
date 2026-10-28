import { afterEach, describe, expect, it, vi } from 'vitest';
import worker, { recognize, warmup } from './index';

function environment(overrides: Record<string, unknown> = {}): Env {
  return {
    ALLOWED_ORIGINS: 'https://mahjong.example', INFERENCE_ENABLED: 'true',
    MODAL_URL: 'https://example--recognize.modal.run/', MODAL_PROXY_TOKEN: 'test-proxy-token',
    USER_LIMIT: { limit: async () => ({ success: true }) },
    SITE_LIMIT: { limit: async () => ({ success: true }) },
    WARM_USER_LIMIT: { limit: async () => ({ success: true }) },
    WARM_SITE_LIMIT: { limit: async () => ({ success: true }) },
    ...overrides,
  } as Env;
}

function upload(headers: Record<string, string> = {}): Request {
  return new Request('https://mahjong.example/api/recognize', { method: 'POST', body: 'test-image',
    headers: { 'Content-Type': 'multipart/form-data; boundary=fixture', ...headers } });
}

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
describe('upload gateway', () => {
  it.each([{ readyAt: 75_000, status: 200 }, { readyAt: 100_000, status: 504 }])(
    'waits for a slow cold start but bounds the request: $readyAt ms', async ({ readyAt, status }) => {
      vi.useFakeTimers();
      const fetch = vi.fn((_url: URL, init: RequestInit) => new Promise<Response>((resolve, reject) => {
        const timer = setTimeout(() => resolve(Response.json({ tiles: [] })), readyAt);
        init.signal?.addEventListener('abort', () => {
          clearTimeout(timer);
          reject(new DOMException('Aborted', 'AbortError'));
        }, { once: true });
      }));
      vi.stubGlobal('fetch', fetch);
      const pending = recognize(upload(), environment());
      await vi.advanceTimersByTimeAsync(91_000);
      expect((await pending).status).toBe(status);
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  it('opts its asset responses out of analytics injection while preserving strict script sources', async () => {
    const env = environment({ ASSETS: { fetch: async () => new Response('<!doctype html><title>麻将</title>',
      { headers: { 'Content-Type': 'text/html' } }) } });
    const response = await worker.fetch(new Request('https://mahjong.example/'), env);
    expect(response.headers.get('Cache-Control')).toContain('no-transform');
    expect(response.headers.get('Content-Security-Policy')).toContain("script-src 'self';");
    expect(response.headers.get('Content-Security-Policy')).not.toContain('cloudflareinsights.com');
  });
  it('does not invoke GPU for a request cancelled while waiting for admission', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    const controller = new AbortController();
    const request = new Request(upload(), { signal: controller.signal });
    const env = environment({ USER_LIMIT: { limit: async () => { controller.abort(); return { success: true }; } } });
    expect((await recognize(request, env)).status).toBe(499);
    expect(fetch).not.toHaveBeenCalled();
  });
  it('rejects cross-origin requests before invoking GPU', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    expect((await recognize(upload({ Origin: 'https://unrelated.example' }), environment())).status).toBe(403);
    expect(fetch).not.toHaveBeenCalled();
  });
  it('rejects oversized uploads, disabled service and rate limit before GPU', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    expect((await recognize(upload({ 'Content-Length': '13000000' }), environment())).status).toBe(413);
    expect((await recognize(upload(), environment({ INFERENCE_ENABLED: 'false' }))).status).toBe(503);
    expect((await recognize(upload(), environment({ USER_LIMIT: { limit: async () => ({ success: false }) } }))).status).toBe(429);
    expect(fetch).not.toHaveBeenCalled();
  });
  it('returns upstream tiles with gateway request id and keeps credentials server-side', async () => {
    const fetch = vi.fn(async (_url: URL, init: RequestInit) => {
      await new Response(init.body).text();
      return Response.json({ requestId: 'not-trusted', modelVersion: 'fixture', tiles: [] });
    }); vi.stubGlobal('fetch', fetch);
    const result = await recognize(upload(), environment());
    const text = await result.text();
    expect(result.status).toBe(200);
    expect(text).not.toContain('test-proxy-token');
    expect(text).not.toContain('not-trusted');
    expect(fetch.mock.calls[0][1].headers).toMatchObject({ Authorization: 'Bearer test-proxy-token' });
    expect(result.headers.get('Cache-Control')).toBe('no-store');
  });
  it('does not expose upstream authentication or traceback messages', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('private traceback secret', { status: 401 })));
    const result = await recognize(upload(), environment());
    expect(result.status).toBe(503);
    expect(await result.text()).not.toContain('private');
  });
  it('uses the Workers-supported manual redirect policy and never forwards credentials to a redirect target', async () => {
    const fetch = vi.fn(async (_url: URL, _init: RequestInit) => new Response('redirect', { status: 307,
      headers: { Location: 'https://unrelated.example/collect' } }));
    vi.stubGlobal('fetch', fetch);
    const result = await recognize(upload(), environment());
    expect(result.status).toBe(503);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][1]).toMatchObject({ redirect: 'manual' });
    expect(await result.text()).not.toContain('unrelated.example');
  });
  it('caps streamed upload when Content-Length is missing', async () => {
    const headers = { 'Content-Type': 'multipart/form-data; boundary=fixture' };
    vi.stubGlobal('fetch', vi.fn(async (_url: URL, init: RequestInit) => {
      await new Response(init.body).arrayBuffer();
      return Response.json({ tiles: [] });
    }));
    const request = new Request('https://mahjong.example/api/recognize', {
      method: 'POST', headers, body: new Uint8Array(12 * 1024 * 1024 + 64 * 1024 + 1),
    });
    expect((await recognize(request, environment())).status).toBe(413);
  });
});

describe('page-open warmup admission', () => {
  const request = (headers: Record<string, string> = {}) => new Request('https://mahjong.example/api/warmup', {
    method: 'POST', headers: { Origin: 'https://mahjong.example', ...headers },
  });
  it('loads only the authenticated model health endpoint, strips query, and returns no secret', async () => {
    const fetch = vi.fn(async (_url: URL, _init: RequestInit) => Response.json({ loaded: true }));
    vi.stubGlobal('fetch', fetch);
    const response = await warmup(request(), environment({ MODAL_URL: 'https://example.modal.run/recognize?private=1' }));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ready: true });
    expect(fetch.mock.calls[0][0].href).toBe('https://example.modal.run/health');
    expect(fetch.mock.calls[0][1]).toMatchObject({ method: 'GET', redirect: 'manual', headers: { Authorization: 'Bearer test-proxy-token' } });
  });
  it('rejects cross-site calls, disabled service and a separate warmup quota before GPU', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    expect((await warmup(request({ Origin: 'https://bad.example' }), environment())).status).toBe(403);
    expect((await warmup(request({ 'Sec-Fetch-Site': 'cross-site' }), environment())).status).toBe(403);
    expect((await warmup(request(), environment({ INFERENCE_ENABLED: 'false' }))).status).toBe(503);
    expect(await (await warmup(request(), environment({ WARM_SITE_LIMIT: { limit: async () => ({ success: false }) } }))).json()).toEqual({ ready: false, deferred: true });
    expect(fetch).not.toHaveBeenCalled();
  });
  it('never follows an upstream redirect and does not spend the recognition quota', async () => {
    const fetch = vi.fn(async () => new Response('', { status: 307, headers: { Location: 'https://bad.example' } }));
    const limit = vi.fn();
    vi.stubGlobal('fetch', fetch);
    const response = await warmup(request(), environment({ USER_LIMIT: { limit }, SITE_LIMIT: { limit } }));
    expect(response.status).toBe(503); expect(fetch).toHaveBeenCalledTimes(1); expect(limit).not.toHaveBeenCalled();
  });
  it('bounds a failed cold warmup without retrying', async () => {
    vi.useFakeTimers();
    const fetch = vi.fn((_url: URL, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(new Error('timeout')), { once: true });
    })); vi.stubGlobal('fetch', fetch);
    const pending = warmup(request(), environment());
    await vi.advanceTimersByTimeAsync(60_001);
    expect((await pending).status).toBe(503); expect(fetch).toHaveBeenCalledTimes(1);
  });
});
