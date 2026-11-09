import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseResult, recognize, serviceAvailability } from '../src/api';

const validResult = () => ({
  requestId: 'request-1', modelVersion: 'test-model', image: { width: 1600, height: 900 },
  quality: { status: 'ok', reasons: [] }, timings: { totalMs: 205 },
  tiles: [{ id: 'tile-1', classId: '1m', label: '一万', bbox: [20, 50, 80, 120], score: 0.9, uncertain: false }],
});

afterEach(() => vi.unstubAllGlobals());

describe('inference response contract', () => {
  it('validates an optional recognition mode without requiring it for older responses', () => {
    expect(parseResult({ ...validResult(), recognitionMode: 'fast' }).recognitionMode).toBe('fast');
    expect(() => parseResult({ ...validResult(), recognitionMode: 'unknown' })).toThrow('识别结果暂时无法读取');
  });
  it('retains the server result without fabricating a hand score', () => {
    expect(parseResult(validResult())).toEqual(validResult());
  });
  it('rejects invalid coordinates before drawing a misleading overlay', () => {
    const value = validResult();
    value.tiles[0].bbox = [1580, 0, 80, 120];
    expect(() => parseResult(value)).toThrow('识别结果暂时无法读取');
  });
  it('rejects duplicate tile identities that could make corrections target the wrong tile', () => {
    const value = validResult();
    value.tiles.push({ ...value.tiles[0], bbox: [200, 50, 80, 120] });
    expect(() => parseResult(value)).toThrow('识别结果暂时无法读取');
  });
  it('rejects undocumented response shapes', () => {
    expect(() => parseResult({ predictions: [] })).toThrow('识别结果暂时无法读取');
  });
});

describe('recognition request', () => {
  it('discovers fast mode without starting model loading and preserves legacy health responses', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(Response.json({ configured: true, fastConfigured: true }))
      .mockResolvedValueOnce(Response.json({ configured: true }));
    vi.stubGlobal('fetch', fetch);
    expect(await serviceAvailability(new AbortController().signal)).toEqual({ configured: true, fastConfigured: true });
    expect(await serviceAvailability(new AbortController().signal)).toEqual({ configured: true, fastConfigured: false });
    expect(fetch.mock.calls.map(call => call[0])).toEqual(['/api/health', '/api/health']);
  });
  it('uploads once to the selected fast route without automatic fallback after failure', async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json({ error: { code: 'UNAVAILABLE', message: '稍后再试' } }, { status: 503 }));
    vi.stubGlobal('fetch', fetch);
    await expect(recognize(new File(['bytes'], 'hand.jpg'), new AbortController().signal, 'fast')).rejects.toMatchObject({ code: 'UNAVAILABLE' });
    expect(fetch).toHaveBeenCalledTimes(1); expect(fetch.mock.calls[0][0]).toBe('/api/recognize?mode=fast');
  });
  it('sends the uploaded image once and respects service rate limits', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { code: 'RATE_LIMITED', message: '请稍后再试' } }), { status: 429 }));
    vi.stubGlobal('fetch', fetch);
    const file = new File(['bytes'], 'hand.jpg', { type: 'image/jpeg' });
    await expect(recognize(file, new AbortController().signal)).rejects.toMatchObject({ code: 'RATE_LIMITED', message: '请稍后再试' });
    expect(fetch).toHaveBeenCalledTimes(1);
    const [, options] = fetch.mock.calls[0];
    expect((options.body as FormData).get('image')).toBe(file);
  });
});
