import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';
export default defineConfig({
  root: 'apps/web',
  plugins: [react(), {
    name: 'local-ort-runtime',
    generateBundle() {
      for (const ext of ['mjs', 'wasm']) {
        // The pinned 1.23.2 WebGPU entry uses asyncify (not the older JSEP
        // binary). Keep both files from this exact npm build.
        const name = `ort-wasm-simd-threaded.asyncify.${ext}`;
        this.emitFile({ type: 'asset', fileName: `ort/${name}`, source: readFileSync(`node_modules/onnxruntime-web/dist/${name}`) });
      }
    },
  }],
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm'] },
  worker: { format: 'es' },
  build: { outDir: '../../dist', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8787' } },
});
