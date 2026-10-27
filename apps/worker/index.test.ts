import { afterEach, describe, expect, it, vi } from 'vitest';
import worker, { recognize } from './index';

function environment(overrides: Record<string, unknown> = {}): Env {
  return {
    ALLOWED_ORIGINS: 'https://mahjong.example', INFERENCE_ENABLED: 'true',
    MODAL_URL: 'https://example--recognize.modal.run/', MODAL_PROXY_TOKEN: 'test-proxy-token',
    USER_LIMIT: { limit: async () => ({ success: true }) },
    SITE_LIMIT: { limit: async () => ({ success: true }) },
    ...overrides,
  } as Env;
}

function upload(headers: Record<string, string> = {}): Request {
  return new Request('https://mahjong.example/api/recognize', { method: 'POST', body: 'test-image',
    headers: { 'Content-Type': 'multipart/form-data; boundary=fixture', ...headers } });
}

afterEach(() => vi.unstubAllGlobals());
describe('upload gateway', () => {
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
