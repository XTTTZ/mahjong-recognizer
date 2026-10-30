import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const root = createRoot(document.getElementById('root')!);
// Separate entry avoids mounting cloud prewarm or downloading inference code on
// normal visits. The large runtime is loaded only after explicit opt-in.
const page = /^\/benchmark\/?$/.test(location.pathname)
  ? import('./benchmark/BrowserBenchmark') : import('./App');
void page.then(({ default: Page }) => root.render(<React.StrictMode><Page /></React.StrictMode>));
