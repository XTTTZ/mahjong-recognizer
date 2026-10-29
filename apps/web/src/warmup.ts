/** One optional, independent warmup when this page first becomes visible. */
export const WARMUP_COOLDOWN_MS = 60_000;
export const WARMUP_TIMEOUT_MS = 65_000;
const SESSION_KEY = 'mahjong:warmup-at:v1';
let memoryAttemptAt: number | null = null;
let inFlight = false;

function previousAttempt(now: number): number | null {
  let stored: number | null = null;
  try {
    const raw = window.sessionStorage.getItem(SESSION_KEY);
    const value = raw === null ? Number.NaN : Number(raw);
    if (Number.isFinite(value) && value >= 0 && value <= now) stored = value;
  } catch { /* Private modes may block sessionStorage; memory still deduplicates. */ }
  const memory = memoryAttemptAt !== null && memoryAttemptAt <= now ? memoryAttemptAt : null;
  return stored === null ? memory : memory === null ? stored : Math.max(stored, memory);
}

/**
 * Call from an effect and return its cleanup. Cleanup only removes visibility/
 * online listeners; it does not cancel an already-started optional warmup.
 * This makes React StrictMode remounts safe. No recognition request is issued.
 */
export function warmupWhenVisible(): () => void {
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
    const previous = previousAttempt(now);
    if (inFlight || (previous !== null && now - previous < WARMUP_COOLDOWN_MS)) return;
    memoryAttemptAt = now;
    try { window.sessionStorage.setItem(SESSION_KEY, String(now)); } catch { /* Best effort metadata only. */ }
    inFlight = true;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), WARMUP_TIMEOUT_MS);
    const finish = () => { clearTimeout(timer); inFlight = false; };
    try {
      void Promise.resolve(fetch('/api/warmup', {
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
