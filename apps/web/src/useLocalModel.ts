import { useEffect, useRef, useState } from 'react';
import { LocalRecognizer } from './localRecognition';

const localModeKey = 'mahjong:preferred-local:compact-v10';
const prepareKey = 'mahjong:preload:compact-v10';

function readPreference(key: string) {
  try { return localStorage.getItem(key); }
  catch { return null; }
}

function savePreference(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* The preference remains available in this tab. */ }
}

export const prefersLocal = () => readPreference(localModeKey) === 'yes';
export function rememberLocal(enabled: boolean) {
  // New selections alone never grant download permission. Existing local users
  // had already consented when the older mode-only preference was recorded.
  if (enabled && readPreference(prepareKey) === null && !prefersLocal()) savePreference(prepareKey, 'no');
  savePreference(localModeKey, enabled ? 'yes' : null);
}

export function useLocalModel() {
  // The original local-mode preference was set only after explicit model consent.
  const [attempt, setAttempt] = useState<number | null>(() => {
    const preference = readPreference(prepareKey);
    return (preference === 'yes' || (preference === null && prefersLocal())) ? 0 : null;
  });
  const [phase, setPhase] = useState<'idle' | 'loading' | 'ready' | 'failed'>('idle');
  const [message, setMessage] = useState('');
  const recognizer = useRef<LocalRecognizer | null>(null);

  useEffect(() => {
    if (attempt === null) return;
    const controller = new AbortController();
    const model = new LocalRecognizer();
    recognizer.current = model;
    setPhase('loading');
    setMessage('正在准备本机模型…');
    let disposed = false;
    const timer = window.setTimeout(() => controller.abort(), 480_000);
    // StrictMode can dispose the first effect before the worker starts downloading.
    queueMicrotask(async () => {
      if (disposed) return;
      try {
        await model.prepare(controller.signal, text => { if (!disposed) setMessage(text); });
        if (!disposed) setPhase('ready');
      } catch (cause) {
        if (disposed) return;
        setPhase('failed');
        setMessage(controller.signal.aborted ? '准备超时。' : cause instanceof Error ? cause.message : '模型准备失败。');
      } finally { window.clearTimeout(timer); }
    });
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      controller.abort();
      model.release();
      recognizer.current = null;
    };
  }, [attempt]);

  function prepare() {
    savePreference(prepareKey, 'yes');
    setAttempt(current => (current ?? 0) + 1);
  }

  function stop() {
    savePreference(prepareKey, 'no');
    setAttempt(null);
    setPhase('idle');
    setMessage('');
  }

  return { phase, message, recognizer, prepare, stop };
}
