/** Public, immutable model assets; this gateway never handles photos or credentials. */
const RELEASE = 'https://github.com/XTTTZ/mahjong-recognizer/releases/download/runtime-onnx-fp32-20261030/';
const FILES = new Map([
  ['/models/portable-v1/manifest.json', 'manifest.json'],
  ['/models/portable-v1/detector.onnx', 'detector.onnx'],
  ['/models/portable-v1/classifier.onnx', 'classifier.onnx'],
]);
const HOSTS = new Set(['github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com']);

export async function portableModel(request: Request): Promise<Response> {
  const name = FILES.get(new URL(request.url).pathname);
  if (!name) return new Response('Model asset not found', { status: 404 });
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response('Method not allowed', { status: 405, headers: { Allow: 'GET, HEAD' } });
  }
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (request.signal.aborted) return new Response(null, { status: 499 });
  request.signal.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(cancel, 45_000);
  try {
    let url = new URL(name, RELEASE);
    for (let redirect = 0; redirect <= 3; redirect += 1) {
      // Construct fresh headers: never send the caller's cookies/authorization.
      const upstream = await fetch(url, { method: request.method, redirect: 'manual', signal: controller.signal,
        headers: { Accept: 'application/octet-stream', 'User-Agent': 'Mahjong-Public-Model-Gateway' },
        cf: { cacheEverything: true, cacheTtlByStatus: { '200-299': 86400, '400-599': 0 } } });
      if (upstream.status >= 300 && upstream.status < 400) {
        const location = upstream.headers.get('Location');
        await upstream.body?.cancel();
        if (!location) break;
        const next = new URL(location, url);
        if (next.protocol !== 'https:' || !HOSTS.has(next.hostname) || next.username || next.password) break;
        url = next;
        continue;
      }
      if (upstream.status !== 200) { await upstream.body?.cancel(); break; }
      const length = upstream.headers.get('Content-Length');
      // Bounds catch accidental release payload mistakes without buffering weights.
      const cap = name.endsWith('.json') ? 128 * 1024 : 160 * 1024 * 1024;
      if (length && (!/^\d+$/.test(length) || Number(length) > cap)) { await upstream.body?.cancel(); break; }
      const headers = new Headers({
        'Content-Type': name.endsWith('.json') ? 'application/json; charset=utf-8' : 'application/octet-stream',
        'Cache-Control': 'public, max-age=31536000, immutable, no-transform',
        'X-Content-Type-Options': 'nosniff', 'Cross-Origin-Resource-Policy': 'same-origin',
      });
      if (length) headers.set('Content-Length', length);
      if (request.method === 'HEAD') { await upstream.body?.cancel(); return new Response(null, { headers }); }
      // Pass through the stream; no full-model copy in the Worker's memory.
      return new Response(upstream.body, { headers });
    }
  } catch { /* Keep upstream details out of user-facing responses. */ }
  finally { clearTimeout(timer); request.signal.removeEventListener('abort', cancel); }
  return new Response('Model download temporarily unavailable. Please retry later.', {
    status: 503, headers: { 'Cache-Control': 'no-store' },
  });
}
