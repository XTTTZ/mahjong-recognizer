# Optional local browser benchmark

`/benchmark` is a separate lazy route. Opening it creates no inference worker,
model download, cloud warmup, API request, or photo upload. The user explicitly
loads the selected profile plus a 25.5 MB WASM runtime. The compact profile is
preferred when configured; the original 119 MB profile remains an explicit control.
Exact selected model bytes are shown from the verified manifest after opt-in. A dedicated
module worker performs inference; cancel/release/unmount terminates that worker
and its model memory. One job at a time; runs are bounded to 1–3 repetitions,
with load/run timeouts. WASM uses one thread; WebGPU reports adapter information
and explicitly notes potential unsupported-operator fallback.

The fixed same-origin manifest and both model files are fetched from
`/models/compact-v10/` or `/models/portable-v1/`. Each model's exact bytes and SHA-256 are checked before
session creation; the reviewed geometry, class ontology, thresholds and tensor
shapes are validated. ONNX Runtime Web 1.23.2's WebGPU entry actually requests the
matching `ort-wasm-simd-threaded.asyncify.{mjs,wasm}` files. The Vite external-WASM
condition keeps these self-hosted under `/ort/` rather than baking the runtime
into the worker or requesting a CDN. Do not substitute another version's binary.

The `/benchmark` document **and** `/assets/local.worker-*.js` must return the
benchmark CSP (`script-src 'self' 'wasm-unsafe-eval'`, `worker-src 'self'`,
`connect-src 'self'`) and `Cross-Origin-Embedder-Policy: require-corp`.
`Cross-Origin-Opener-Policy: same-origin` is set on the document; resources are
same-origin with CORP. The normal application's CSP need not be relaxed. The
worker itself needs its compatible COEP response header; omitting it was verified
to fail with `coep-frame-resource-needs-coep-header` before model loading.

The same `preparePhoto` as the main app applies longest-edge 2560/JPEG .94.
Detector inverse scaling uses the actual rounded per-axis gains from the pinned
Ultralytics geometry. NMS .6/max300 precedes the >200 crowd guard. Every remaining
candidate is classified in batches of up to four, with complete crop/context .08
and ImageNet normalization; fused scores must be strictly >.35. The JSON includes
each tile's class/bbox/fused score and whether all retained tiles reach .7. It
contains timings and browser/device metadata, without image bytes or filename.
**Canvas interpolation is not bit-identical to OpenCV/PIL.** This is an
experimental runtime; one smoke image does not establish validation accuracy.

Checks: `npx vitest run apps/web/src/benchmark/pipeline.test.ts`, `npm run build`.
Loopback production preview: `python infra/portable/serve_benchmark.py --models
artifacts/models/releases/mahjong-onnx-fp32-public-20261030 --compact-models
/path/to/downloaded-compact-profile`. Desktop Chrome
hardware smoke evidence is saved under `artifacts/runs/portable-browser-*-20261030.json`.
It is not phone hardware evidence. Loopback model-load times are not Internet
download estimates. Repeated runs may reuse the browser/driver shader cache.

Reproducible desktop smoke (requires the project's Playwright browser install;
`--browser chrome` uses an installed Google Chrome instead):

```sh
node infra/portable/probe_browser.mjs --base-url http://127.0.0.1:4174 \
  --image /path/to/hand.jpg --provider webgpu --browser chrome --headed \
  --out artifacts/runs/my-desktop-webgpu.json
```

The default provider is `wasm` and profile is `compact-v10`; use
`--profile portable-v1` for the original control, with a different output file.
`--repeats 1` is available for a quick check; the default and maximum is three.
The CLI always navigates to `/benchmark`, refuses existing output files, verifies
that no model/runtime is fetched before the opt-in click, blocks API/POST/external
requests, and saves an adjacent `.probe.json` network/release receipt. It does not
test the normal app route or upload the selected photo. For live testing, change
only `--base-url` to the published HTTPS origin.

For a phone measurement, use that physical phone's browser to open the published
HTTPS `/benchmark` page, explicitly load models, choose a photo, then save its JSON.
Mobile-sized desktop screenshots are layout checks only. The JSON intentionally
does not self-certify physical phone hardware; confirm the actual device/browser
separately when interpreting a submitted result.

Official references: [WebGPU import and execution provider](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html),
[matching WASM files and environment flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html),
[self-hosted deployment and CSP](https://onnxruntime.ai/docs/tutorials/web/deploy.html).

## Compact comparison profile (V10)

The explicit model selector defaults to the configured `compact-v10` profile.
The original `portable-v1` remains available as a manual comparison; a failed
compact download never silently downloads the larger original. `profiles.ts`
contains the finite enabled profile list. Only enable compact in a deployment
that serves its three fixed, same-origin files under `/models/compact-v10/`.
No manifest or runtime request happens before the load button is pressed.

The compact manifest must use `research-onnx-fp32-cascade`, `researchOnly:true`,
`precision:float32`, opset17,42 unique reviewed class tokens, detector640/800/960,
and classifier MobileNetV3-Large192/224 or EfficientNetV2-S320. Context remains.08,
with unchanged detector.05/fused>.35/accept.7, NMS.6 and safety bounds. The existing
candidate exporter emits this format. Both stages require source run/checkpoint/
signature hashes and `sourceEvidence.testOpened:false`. Manifest bytes, SHA256,
IO contracts and the loaded sessions' actual metadata are checked. Arbitrary
profile IDs, resource names and model URLs are rejected.

The UI receives exact model bytes from the verified manifest before model
transfer and saves the selected profile, version, hashes and geometry with each
result. `load.pageObservedLoadMs` measures the button-to-ready time including
worker/runtime bootstrap; `load.totalLoadMs` retains the worker's model loading
measurement. Neither should be conflated with warm inference time. ROI buffers
are normalized directly into bounded contiguous batches of four, including the
last partial batch. OffscreenCanvas buffers are reused within the worker and are
released when cancel/release destroys it.

The desktop probe supports `--profile compact-v10` (default) or
`--profile portable-v1`. It only permits the selected profile's fixed asset paths;
requests for API endpoints, POSTs, other model profiles or external resources fail.
For physical phone measurements, choose the same profile and engine on that phone.
Desktop results continue to set `physicalPhoneVerified:false`.
