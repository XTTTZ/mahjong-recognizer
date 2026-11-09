# V10 compact CPU/browser runtime

The user explicitly permits a small accuracy decrease in exchange for practical CPU or phone-local inference. Historical strict per-image non-regression evidence remains visible; it no longer automatically blocks all compact candidates.

## Selected candidate and measurements

Candidate: YOLOv8n800 generic detector + MobileNetV3-Large224, light distillation alpha.2/T2, context.08. Detector is V9 Nano800 last; classifier is V9 KD224 best (37complete epochs under its budget). Exact source SHA256, ONNX graph hashes/bytes and42-class mapping are in the distributed manifest. No new training, pseudo-labeling or sealed-test tuning was performed for this runtime.

FP32 ONNX total29,853,745bytes vs original119,092,864bytes, about75% smaller. Local nativeCPU frozen150-image evaluation:137/150 automatic exact photos,74/82 multi-tile,44/45 containing13/14 tiles; original135/150,72/82,43/45. This aggregate improvement includes per-image regressions, so it is not proof of non-inferiority or new-room accuracy. The one-time original sealed test is not reopened.

The provisional engineering tolerance is at most3/150,2/82,1/45 fewer automatic exact photos and at most1additional wrong accept in each cohort, zero inference failures; warm normal-hand p95≤2seconds, plus meaningful paired latency or model-size benefit. `infra/portable/tradeoff_policy.py` reports both aggregate qualification and old per-image losses. It never automatically deploys anything.

Real desktop Chrome smoke (Apple M4 Pro, one14-tile photo) found the initially screened compact CE224 profile about0.47–0.51seconds WASM vs4.3seconds original; WebGPU first0.355seconds, later0.046–0.060seconds. These desktop single-photo measurements are not phone evidence, not the final KD model's whole-set browser accuracy, and exclude Internet model download. Target Modal CPU and final-profile browser results are kept in the local run receipts and must be checked separately.

## Reproduce

1. Fetch signed run/checkpoint/terminal receipts from the configured training host using `infra/portable/prepare_candidates_v10.py`; transport mapping preserves original bytes and binds local paths to explicit remote source paths. `candidate_export.py` verifies the data/source/checkpoint contract before export. Source maps and raw logs stay private.
2. Export CPU-only, FP32 opset17, using `candidate_export`; reusing an ONNX stage requires exact source/config/I/O equivalence, not a matching filename. Three-photo native/ORT parity is a smoke check; whole150 validation is required separately.
3. `profile_matrix.py --profiles ... --valid-dir .../valid --out ...` defaults to validation-only. `--run-local` runs locally. `--run-cloud` runs one bounded CPU matrix. Large SDK uploads may fail on some connections; `stream_benchmark.py` downloads hash-pinned public model files inside its image and sends one private validation photo per call. It has a25-minute client budget,90-second method timeout andonecontainer, noGPU.
4. `MAHJONG_STREAM_CORES=1|2 python -m infra.portable.stream_benchmark --valid-dir DATA/valid --out FRESH.json --run-cloud` compares original and compact in the same warm container, alternating order. Set `MAHJONG_STREAM_CANDIDATE` to the selected local manifest/weights directory for SHA pinning. Network/load/coldstart are excluded from pipeline times and need endpoint probes.
5. For the CPU endpoint: `MAHJONG_CPU_MODEL_DIR=... MAHJONG_CPU_PUBLIC_BASE_URL=https://github.com/XTTTZ/mahjong-recognizer/releases/download/runtime-compact-fp32-20261109/ python -m modal deploy infra/modal/cpu_app.py`. Local files are verified; build downloads the three exact public assets with streamed SHA256/byte checks. This avoids SDK weight uploads. Without the public URL, the existing4MiB chunk uploader remains available.

CPU service defaults1physicalcore,2048MiB, batch4, CPU memorysnapshot, min0/max1, idle10s, requesttimeout90s, mandatory Modal proxy authentication. The GPU service is independent. The gateway's fast mode uses only the configured CPU URL; it does not silently retry or upload to the other service after failure. Legacy no-mode clients retain the original GPU endpoint. Page-open warmup follows the selected available mode.

The app offers quick/precision choices after configuration discovery. Local inference is an explicit `/benchmark` opt-in with fixed same-origin `/models/compact-v10/` assets. It verifies SHA256/bytes/source hashes/actual tensor I/O, classifies all detections in bounded batches and allows cancellation/release. Images never leave the browser on that page. Existing tile artwork, corrections, add/delete/undo, scoring and visible waits are preserved on normal recognition.

## Publication and phone checks

The compact public release contains sanitized FP32 model graphs, manifest, license and attribution. Corresponding source includes the training/evaluation/export/runtime code. No credentials, raw photographs, private transport maps or experiment logs are published. Public retrieval uses only fixed approved GitHub release paths; no caller URL, cookies or authorization are forwarded.

To test a physical phone, open the deployed `/benchmark`, choose compact and WASM or WebGPU, explicitly load the models, select a hand photo and save the JSON. Record actual phone/browser, first download/load, first inference, cached later runs and warm-device behavior. Mobile viewport emulation does not count as hardware evidence. CPU performance and cached browser performance do not promise a universal total response time over mobile networks.
