import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseResult, recognize } from '../src/api';

const validResult = () => ({
  requestId: 'request-1', modelVersion: 'test-model', image: { width: 1600, height: 900 },
  quality: { status: 'ok', reasons: [] }, timings: { totalMs: 205 },
  tiles: [{ id: 'tile-1', classId: '1m', label: '一万', bbox: [20, 50, 80, 120], score: 0.9, uncertain: false }],
});

afterEach(() => vi.unstubAllGlobals());

describe('inference response contract', () => {
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
