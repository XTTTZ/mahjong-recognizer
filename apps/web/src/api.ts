import type { RecognitionResult } from './types';

export class RecognitionError extends Error {
  constructor(message: string, readonly code: string, readonly requestId?: string) {
    super(message);
    this.name = 'RecognitionError';
  }
}

/** This only checks gateway configuration; it never starts the inference GPU. */
export async function serviceConfigured(signal: AbortSignal): Promise<boolean | null> {
  try {
    const response = await fetch('/api/health', { signal, cache: 'no-store', credentials: 'same-origin' });
    if (!response.ok) return null;
    const value: unknown = await response.json();
    return isRecord(value) && typeof value.configured === 'boolean' ? value.configured : null;
  } catch { return null; }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Validate the contract before drawing coordinates supplied by the service. */
export function parseResult(value: unknown): RecognitionResult {
  if (!isRecord(value) || !isRecord(value.image) || !isRecord(value.quality)
    || !isRecord(value.timings) || typeof value.requestId !== 'string'
    || typeof value.modelVersion !== 'string' || !Array.isArray(value.tiles)
    || value.tiles.length > 200 || !Number.isFinite(value.image.width)
    || !Number.isFinite(value.image.height) || Number(value.image.width) <= 0
    || Number(value.image.height) <= 0 || !Number.isFinite(value.timings.totalMs)
    || !['ok', 'needs_retake'].includes(String(value.quality.status))
    || !Array.isArray(value.quality.reasons) || !value.quality.reasons.every(x => typeof x === 'string')) {
    throw new RecognitionError('识别结果暂时无法读取，请稍后重试。', 'INVALID_RESPONSE');
  }
  const ids = new Set<string>();
  for (const tile of value.tiles) {
    if (!isRecord(tile) || typeof tile.id !== 'string' || typeof tile.classId !== 'string'
      || typeof tile.label !== 'string' || typeof tile.uncertain !== 'boolean'
      || !Number.isFinite(tile.score) || Number(tile.score) < 0 || Number(tile.score) > 1
      || !Array.isArray(tile.bbox) || tile.bbox.length !== 4 || !tile.bbox.every(Number.isFinite)
      || tile.bbox[0] < 0 || tile.bbox[1] < 0 || tile.bbox[2] <= 0 || tile.bbox[3] <= 0
      || tile.bbox[0] + tile.bbox[2] > Number(value.image.width) + 1
      || tile.bbox[1] + tile.bbox[3] > Number(value.image.height) + 1) {
      throw new RecognitionError('识别结果暂时无法读取，请稍后重试。', 'INVALID_RESPONSE');
    }
    if (ids.has(tile.id)) throw new RecognitionError('识别结果暂时无法读取，请稍后重试。', 'INVALID_RESPONSE');
    ids.add(tile.id);
  }
  return value as unknown as RecognitionResult;
}

export async function recognize(file: File, signal: AbortSignal): Promise<RecognitionResult> {
  const body = new FormData();
  body.append('image', file);
  const response = await fetch('/api/recognize', {
    method: 'POST', body, signal, credentials: 'same-origin', cache: 'no-store',
    headers: { Accept: 'application/json' },
  });
  const value: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = isRecord(value) && isRecord(value.error) ? value.error : undefined;
    const messages: Record<number, string> = {
      413: '照片太大，请裁剪后重新选择。',
      415: '暂时无法读取这张照片，请尝试 JPEG、PNG 或 WebP。',
      429: '识别请求较多，请稍等片刻再试。',
      503: '识别服务暂时不可用，请稍后重试。',
      504: '这次等待时间较长，请稍后再试。',
    };
    throw new RecognitionError(
      typeof error?.message === 'string' ? error.message : messages[response.status] ?? '识别未完成，请稍后重试。',
      typeof error?.code === 'string' ? error.code : `HTTP_${response.status}`,
      isRecord(value) && typeof value.requestId === 'string' ? value.requestId : undefined,
    );
  }
  return parseResult(value);
}
