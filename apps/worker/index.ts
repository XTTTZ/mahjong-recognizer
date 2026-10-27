const MAX_UPLOAD = 12 * 1024 * 1024 + 64 * 1024; // multipart overhead; image itself remains capped by inference
const MAX_RESULT = 1024 * 1024;
const RESPONSE_HEADERS = {
  'Cache-Control': 'no-store',
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
};

function json(value: unknown, status = 200): Response {
  return Response.json(value, { status, headers: RESPONSE_HEADERS });
}

function error(requestId: string, status: number, code: string, message: string): Response {
  return json({ requestId, error: { code, message } }, status);
}

export async function recognize(request: Request, env: Env): Promise<Response> {
  const requestId = crypto.randomUUID();
  if (request.method !== 'POST') return error(requestId, 405, 'METHOD', '请使用上传入口。');
  const origin = request.headers.get('Origin');
  const allowed = env.ALLOWED_ORIGINS.split(',').map(v => v.trim()).filter(Boolean);
  if (origin && origin !== new URL(request.url).origin && !allowed.includes(origin)) {
    return error(requestId, 403, 'ORIGIN', '请从本站上传照片。');
  }
  const type = request.headers.get('Content-Type') ?? '';
  if (!/^multipart\/form-data;.*boundary=/i.test(type)) {
    return error(requestId, 415, 'CONTENT_TYPE', '请上传一张图片。');
  }
  const length = request.headers.get('Content-Length');
  if (length && (!/^\d+$/.test(length) || Number(length) > MAX_UPLOAD)) {
    return error(requestId, 413, 'TOO_LARGE', '图片过大，请选择小于 12 MB 的照片。');
  }
  if (!request.body) return error(requestId, 400, 'EMPTY', '尚未选择照片。');
  if (env.INFERENCE_ENABLED !== 'true' || !env.MODAL_URL || !env.MODAL_PROXY_TOKEN) {
    return error(requestId, 503, 'UNAVAILABLE', '识别服务正在准备，请稍后再试。');
  }
  let upstream: URL;
  try { upstream = new URL(env.MODAL_URL); }
  catch { return error(requestId, 503, 'CONFIGURATION', '识别服务暂不可用。'); }
  if (upstream.protocol !== 'https:' || !upstream.hostname.endsWith('.modal.run')) {
    return error(requestId, 503, 'CONFIGURATION', '识别服务暂不可用。');
  }
  const ip = request.headers.get('CF-Connecting-IP') ?? 'local';
  const [userLimit, siteLimit] = await Promise.all([
    env.USER_LIMIT.limit({ key: ip }), env.SITE_LIMIT.limit({ key: 'recognize' }),
  ]);
  if (!userLimit.success || !siteLimit.success) {
    const response = error(requestId, 429, 'RATE_LIMIT', '识别请求较多，请稍后再试。');
    response.headers.set('Retry-After', '60');
    return response;
  }
  if (request.signal.aborted) return error(requestId, 499, 'CANCELLED', '识别请求已取消。');
  const abort = new AbortController();
  const onAbort = () => abort.abort();
  request.signal.addEventListener('abort', onAbort, { once: true });
  const timer = setTimeout(() => abort.abort(), 90_000);
  let bytes = 0;
  let oversized = false;
  const bounded = request.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      bytes += chunk.byteLength;
      if (bytes > MAX_UPLOAD) {
        oversized = true;
        controller.error(new Error('upload-limit'));
        abort.abort();
      } else { controller.enqueue(chunk); }
    },
  }));
  const started = Date.now();
  let phase = 'request';
  try {
    const response = await fetch(upstream, {
      method: 'POST', body: bounded, redirect: 'manual', signal: abort.signal,
      headers: { 'Content-Type': type, 'Authorization': `Bearer ${env.MODAL_PROXY_TOKEN}`,
        'X-Request-ID': requestId },
    });
    phase = 'response';
    if (response.status >= 300 && response.status < 400) {
      await response.body?.cancel();
      return error(requestId, 503, 'UPSTREAM_REDIRECT', '识别服务地址暂不可用，请稍后再试。');
    }
    if (!response.body) return error(requestId, 502, 'UPSTREAM', '识别结果未返回，请稍后再试。');
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let resultBytes = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      resultBytes += value.byteLength;
      if (resultBytes > MAX_RESULT) {
        await reader.cancel();
        return error(requestId, 502, 'UPSTREAM', '识别结果异常，请稍后再试。');
      }
      chunks.push(value);
    }
    const buffer = new Uint8Array(resultBytes);
    let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
    if (!response.ok) {
      const safeCodes = [400, 413, 415, 422, 429];
      const status = safeCodes.includes(response.status) ? response.status : 503;
      const messages: Record<number, string> = {
        400: '无法读取照片，请重新选择。', 413: '图片过大，请缩小后重试。',
        415: '此图片格式暂不支持，请使用 JPEG、PNG 或 WebP。',
        422: '图片内容无法处理，请重新拍摄。', 429: '识别服务繁忙，请稍后再试。',
      };
      return error(requestId, status, 'INFERENCE', messages[status] ?? '识别服务暂不可用，请稍后再试。');
    }
    phase = 'decode';
    const result: unknown = JSON.parse(new TextDecoder().decode(buffer));
    if (!result || typeof result !== 'object' || !('tiles' in result) || !Array.isArray(result.tiles)) {
      return error(requestId, 502, 'INVALID_RESULT', '识别结果异常，请重新拍摄。');
    }
    console.log(JSON.stringify({ event: 'recognize', requestId, bytes, elapsedMs: Date.now() - started, status: 200 }));
    return json({ ...result, requestId });
  } catch (cause) {
    const status = oversized ? 413 : abort.signal.aborted ? 504 : 503;
    // Fetch runtime messages describe transport errors. Never log response
    // parsing errors, which could quote upstream content, or request headers.
    const transportError = phase === 'request' && cause instanceof Error
      ? cause.message.replaceAll(env.MODAL_PROXY_TOKEN, '[redacted]').slice(0, 200) : undefined;
    console.warn(JSON.stringify({ event: 'recognize', requestId, bytes, elapsedMs: Date.now() - started, status,
      phase, errorType: cause instanceof Error ? cause.name : 'unknown', transportError }));
    return error(requestId, status, oversized ? 'TOO_LARGE' : status === 504 ? 'TIMEOUT' : 'UNAVAILABLE',
      oversized ? '图片过大，请缩小后重试。' : status === 504 ? '识别等待超时，原请求可能仍在处理。请稍后手动重试。' : '暂时无法连接识别服务，请稍后再试。');
  } finally {
    clearTimeout(timer);
    request.signal.removeEventListener('abort', onAbort);
  }
}

export default {
  async fetch(request, env): Promise<Response> {
    const path = new URL(request.url).pathname;
    if (path === '/api/health') return json({ service: 'mahjong', version: env.APP_VERSION,
      configured: env.INFERENCE_ENABLED === 'true' && Boolean(env.MODAL_URL && env.MODAL_PROXY_TOKEN),
      inferenceProbed: false });
    if (path === '/api/recognize') {
      try { return await recognize(request, env); }
      catch { return error(crypto.randomUUID(), 503, 'UNAVAILABLE', '识别服务暂不可用。'); }
    }
    if (path.startsWith('/api/')) return error(crypto.randomUUID(), 404, 'NOT_FOUND', '接口不存在。');
    const asset = await env.ASSETS.fetch(request);
    const response = new Response(asset.body, asset);
    // Opt this app out of automatic HTML/beacon injection without changing the
    // shared Cloudflare zone or permitting third-party scripts in the CSP.
    response.headers.set('Cache-Control', 'public, max-age=0, must-revalidate, no-transform');
    response.headers.set('X-Content-Type-Options', 'nosniff');
    response.headers.set('Referrer-Policy', 'no-referrer');
    response.headers.set('Permissions-Policy', 'camera=(self), microphone=(), geolocation=()');
    response.headers.set('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'");
    return response;
  },
} satisfies ExportedHandler<Env>;
