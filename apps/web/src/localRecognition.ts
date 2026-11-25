import { parseResult } from './api';
import { tileLabel } from './benchmark/labels';
import type { RecognitionResult, TilePrediction } from './types';

type LocalTile = Pick<TilePrediction, 'score' | 'bbox' | 'classId'>;
interface LocalReport {
  profileId: string;
  modelVersion: string;
  decodeMs: number;
  inputImage: { width: number; height: number };
  runs: { totalMs: number; tiles: LocalTile[] }[];
}

const centerY = (tile: LocalTile) => tile.bbox[1] + tile.bbox[3] / 2;

/** Match the cloud review order: group nearby vertical centers, then read each row left to right. */
export function localReviewResult(report: LocalReport): RecognitionResult {
  if (report.profileId !== 'compact-v10' || report.runs.length !== 1) throw new Error('本机模型结果不匹配');
  const run = report.runs[0];
  const rows: LocalTile[][] = [];
  for (const tile of [...run.tiles].sort((a, b) => centerY(a) - centerY(b))) {
    const row = rows.find(items => {
      const meanY = items.reduce((sum, item) => sum + centerY(item), 0) / items.length;
      const meanHeight = items.reduce((sum, item) => sum + item.bbox[3], 0) / items.length;
      return Math.abs(centerY(tile) - meanY) < 0.55 * Math.min(tile.bbox[3], meanHeight);
    });
    if (row) row.push(tile);
    else rows.push([tile]);
  }
  const tiles = rows.flatMap(row => row.sort((a, b) => a.bbox[0] - b.bbox[0])).map((tile, index) => ({
    ...tile, id: String(index + 1), label: tileLabel(tile.classId), uncertain: tile.score < 0.7,
  }));
  const reasons: string[] = [];
  if (!tiles.length) reasons.push('没有找到清晰的牌面，请靠近并重新拍摄。');
  if (tiles.some(tile => tile.uncertain)) reasons.push('部分牌面不够确定，请核对标记的牌或重新拍摄。');
  return { ...parseResult({
    requestId: crypto.randomUUID(), modelVersion: report.modelVersion,
    image: { width: report.inputImage.width, height: report.inputImage.height }, tiles,
    quality: { status: reasons.length ? 'needs_retake' : 'ok', reasons },
    timings: { totalMs: report.decodeMs + run.totalMs },
  }), recognitionMode: 'local' };
}

type WorkerCommand = { type: 'load'; provider: 'wasm'; profileId: 'compact-v10' }
  | { type: 'run'; file: File; repeats: 1 };

export class LocalRecognizer {
  private worker: Worker | null = null;

  release() {
    this.worker?.terminate();
    this.worker = null;
  }

  async prepare(signal: AbortSignal, status: (message: string) => void) {
    signal.throwIfAborted();
    this.worker = new Worker(new URL('./benchmark/local.worker.ts', import.meta.url), { type: 'module' });
    await this.execute({ type: 'load', provider: 'wasm', profileId: 'compact-v10' }, 'ready', signal, status);
  }

  async recognize(file: File, signal: AbortSignal, status: (value: string) => void): Promise<RecognitionResult> {
    const report = await this.execute({ type: 'run', file, repeats: 1 }, 'result', signal, status);
    return localReviewResult(report as LocalReport);
  }

  private execute(command: WorkerCommand, done: 'ready' | 'result', signal: AbortSignal,
    status: (value: string) => void): Promise<unknown> {
    signal.throwIfAborted();
    const worker = this.worker;
    if (!worker) throw new Error('请先准备本机模型');
    return new Promise((resolve, reject) => {
      const detach = () => {
        signal.removeEventListener('abort', onAbort);
        worker.onmessage = null;
        worker.onerror = null;
        worker.onmessageerror = null;
      };
      const fail = (error: Error) => {
        detach();
        this.release();
        reject(error);
      };
      const onAbort = () => fail(new DOMException('已取消', 'AbortError'));
      signal.addEventListener('abort', onAbort, { once: true });
      worker.onmessage = event => {
        const { type, value } = event.data;
        if (type === 'status') status(String(value));
        else if (type === done) {
          detach();
          resolve(value);
        } else if (type === 'error') fail(new Error(String(value)));
      };
      worker.onerror = worker.onmessageerror = () => fail(new Error('浏览器无法运行本机模型'));
      try { worker.postMessage(command); }
      catch { fail(new Error('浏览器无法启动本机模型')); }
    });
  }
}
