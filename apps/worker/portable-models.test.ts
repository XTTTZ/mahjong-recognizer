import { afterEach, describe, expect, it, vi } from 'vitest';
import { portableModel } from './portable-models';
import worker from './index';

afterEach(() => vi.unstubAllGlobals());
const request = (path = 'detector.onnx', init?: RequestInit) => new Request(`https://mahjong.example/models/portable-v1/${path}`, init);

describe('opt-in public portable models', () => {
  it('serves only exact assets and read-only methods', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    expect((await portableModel(request('private.pt'))).status).toBe(404);
    expect((await portableModel(request('classifier.onnx', { method: 'POST' }))).status).toBe(405);
    expect(fetch).not.toHaveBeenCalled();
  });
  it('streams fixed public weights without forwarding cookies, authorization, query or upstream cookies', async () => {
    const fetch = vi.fn(async () => new Response(new Uint8Array([1, 2, 3]), {
      headers: { 'Content-Length': '3', 'Set-Cookie': 'private=unused' },
    })); vi.stubGlobal('fetch', fetch);
    const response = await portableModel(request('detector.onnx?url=https://private.example', {
      headers: { Authorization: 'private', Cookie: 'private' },
    }));
    expect([...new Uint8Array(await response.arrayBuffer())]).toEqual([1, 2, 3]);
    expect(response.headers.get('Set-Cookie')).toBeNull();
    expect(response.headers.get('Cache-Control')).toContain('immutable');
    const [url, options] = fetch.mock.calls[0] as unknown as [URL, RequestInit];
    expect(url.href).toBe('https://github.com/XTTTZ/mahjong-recognizer/releases/download/runtime-onnx-fp32-20261030/detector.onnx');
    expect(JSON.stringify(options.headers)).not.toContain('private');
  });
  it('accepts GitHub asset redirects but refuses other hosts', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(new Response(null, { status: 302,
      headers: { Location: 'https://release-assets.githubusercontent.com/release/file' } }))
      .mockResolvedValueOnce(new Response('model'));
    vi.stubGlobal('fetch', fetch);
    expect((await portableModel(request())).status).toBe(200);
    expect(fetch).toHaveBeenCalledTimes(2);
    fetch.mockReset().mockResolvedValue(new Response(null, { status: 302,
      headers: { Location: 'https://unrelated.example/private' } }));
    expect((await portableModel(request())).status).toBe(503);
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it('does not cache errors or return oversized metadata', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('bad', { headers: { 'Content-Length': '9999999' } })));
    const response = await portableModel(request('manifest.json'));
    expect(response.status).toBe(503);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
  });
  it('allows WASM only for the isolated benchmark document, without touching the GPU', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    const env = { ASSETS: { fetch: async () => new Response('<html>test</html>') } } as Env;
    const main = await worker.fetch(new Request('https://mahjong.example/'), env);
    const bench = await worker.fetch(new Request('https://mahjong.example/benchmark'), env);
    const script = await worker.fetch(new Request('https://mahjong.example/assets/local.worker-fixture123.js'), env);
    expect(main.headers.get('Content-Security-Policy')).not.toContain('wasm-unsafe-eval');
    expect(bench.headers.get('Content-Security-Policy')).toContain("script-src 'self' 'wasm-unsafe-eval'");
    expect(bench.headers.get('Content-Security-Policy')).not.toContain("'unsafe-eval'");
    expect(bench.headers.get('Cross-Origin-Embedder-Policy')).toBe('require-corp');
    expect(script.headers.get('Cross-Origin-Embedder-Policy')).toBe('require-corp');
    expect(script.headers.get('Content-Security-Policy')).toContain("'wasm-unsafe-eval'");
    expect(fetch).not.toHaveBeenCalled();
  });
});
