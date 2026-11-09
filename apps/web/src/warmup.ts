import type { RecognitionMode } from './types';

/** One optional, independent warmup per selected service when the page becomes visible. */
export const WARMUP_COOLDOWN_MS = 60_000;
export const WARMUP_TIMEOUT_MS = 65_000;
const SESSION_KEY = 'mahjong:warmup-at:v1';
const memoryAttempts = new Map<RecognitionMode, number>();
const inFlight = new Set<RecognitionMode>();

function previousAttempt(now: number, mode: RecognitionMode): number | null {
  let stored: number | null = null;
  try {
    const raw = window.sessionStorage.getItem(mode === 'precision' ? SESSION_KEY : `${SESSION_KEY}:fast`);
    const value = raw === null ? Number.NaN : Number(raw);
    if (Number.isFinite(value) && value >= 0 && value <= now) stored = value;
  } catch { /* Private modes may block sessionStorage; memory still deduplicates. */ }
  const attempt = memoryAttempts.get(mode);
  const memory = attempt !== undefined && attempt <= now ? attempt : null;
  return stored === null ? memory : memory === null ? stored : Math.max(stored, memory);
}

/**
 * Call from an effect and return its cleanup. Cleanup only removes visibility/
 * online listeners; it does not cancel an already-started optional warmup.
 * This makes React StrictMode remounts safe. No recognition request is issued.
 */
export function warmupWhenVisible(mode: RecognitionMode = 'precision'): () => void {
  if (typeof window === 'undefined' || typeof document === 'undefined') return () => {};
  let done = false;
  const removeListeners = () => {
    document.removeEventListener('visibilitychange', maybeWarm);
    window.removeEventListener('online', maybeWarm);
  };
  function maybeWarm() {
    if (done || document.visibilityState !== 'visible' || (typeof navigator !== 'undefined' && navigator.onLine === false)) return;
    done = true;
    removeListeners();
    const now = Date.now();
    const previous = previousAttempt(now, mode);
    if (inFlight.has(mode) || (previous !== null && now - previous < WARMUP_COOLDOWN_MS)) return;
    memoryAttempts.set(mode, now);
    try { window.sessionStorage.setItem(mode === 'precision' ? SESSION_KEY : `${SESSION_KEY}:fast`, String(now)); } catch { /* Best effort metadata only. */ }
    inFlight.add(mode);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), WARMUP_TIMEOUT_MS);
    const finish = () => { clearTimeout(timer); inFlight.delete(mode); };
    try {
      void Promise.resolve(fetch(mode === 'fast' ? '/api/warmup?mode=fast' : '/api/warmup', {
        method: 'POST', credentials: 'same-origin', cache: 'no-store', signal: controller.signal,
        headers: { Accept: 'application/json' },
      // Drain the small same-origin JSON response before clearing the timeout
      // and in-flight flag; headers alone do not finish the response lifecycle.
      })).then(response => response.text()).catch(() => undefined).finally(finish);
    } catch { finish(); }
  }
  document.addEventListener('visibilitychange', maybeWarm);
  window.addEventListener('online', maybeWarm);
  maybeWarm();
  return () => { done = true; removeListeners(); };
}
