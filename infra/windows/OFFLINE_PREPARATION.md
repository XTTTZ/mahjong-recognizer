# Offline preparation while the Windows machine is off

This section records the preparation completed while Windows was off. Later on 2026-10-26 the user powered it on and authorized training; connectivity, environment and curated-data hashes were rechecked before starting a fresh run. See ../../PROGRESS.md and ../../docs/models/training-run-v3.json for that later state. The offline tools themselves do not perform SSH or launch training.

## Historical checkpoint and limitations

The raw-v2 run `rfdetr-small-640-full-v2` was intentionally stopped after its first complete epoch, rather than completing its proposed 30 epochs. Before shutdown, `checkpoint_0.ckpt` was fully read on CPU and contained model/optimizer/scheduler state, zero-based epoch 0 and global step 590. Its recorded SHA256 is `7cc131b59042446d9d7e5805f67b44705c6b3c458220b249e531c59cb9dd5866`, size 515,448,337 bytes. The last known Windows path is `D:\mahjong\runs\rfdetr-small-640-full-v2\checkpoint_0.ckpt`; no fresh verification has been performed while the machine is off, and no local checkpoint copy has been verified.

The owned process tree was terminated after validation of that checkpoint. The training runner therefore reported `failed`, while the intentional-stop observer completed successfully. This is documented in `verified/stopped-v2.json`; it is neither a completed experiment nor evidence of a training-code crash.

Some source broadcast frames annotate only the bottom hand while other visible tiles remain unlabeled. Its validation mAP values are raw-v2 research diagnostics only. No exact-hand, user-cropped, full-photo or real-room accuracy was established. Do not promote this checkpoint or resume it on curated-v3: start each architecture from its separately pinned public pretrained source.

## Small snapshot, separate large inputs

`offline_bundle.py` packages only training/data/evaluation/inference code, Windows helpers, version locks, asset hashes and dataset metadata. It excludes credentials, environments, frontend/node dependencies, dataset images/ZIPs, model weights and launchable historical v2 job specs. A transported bundle includes a per-file byte/hash manifest, a trusted expected bundle SHA256, a small standalone verifier and a receipt. Unpacking requires a fresh destination; it cannot silently mix new source into an old training checkout.

The raw archive remains the existing immutable 4,447,371,742-byte v2 COCO ZIP, SHA256 `14640fb5a47707b35d8f0025260883aff4936a2893f2eba9e07758b063df1268`. The new tracked v3 selection is a small JSONL. Oracle ROI derivatives are generated from selected members of that ZIP, without transferring or extracting a second complete 4 GB copy. The materializer must reproduce its frozen COCO and ROI PNG hashes; an encoding/environment mismatch is a preparation failure, not an excuse to ignore hashes.

After the data reviewer freezes the v3 files, build locally:

```bash
python -m infra.windows.experiment_plans \
  --metadata docs/data/final-dataset-v3.json \
  --split-manifest docs/data/split-v3.jsonl \
  --out infra/windows/specs/curated-v3
python -m infra.windows.offline_bundle build \
  --dataset-metadata docs/data/final-dataset-v3.json \
  --split-manifest docs/data/split-v3.jsonl \
  --out artifacts/runs/offline-prep/training-code-v3.zip
```

These commands do not contact the Windows machine. They fail if the frozen metadata/selection files are missing or hashes do not match. Regenerate plans and snapshot when source or dataset files change. The receipt's Git commit is context; the per-file SHA256 manifest defines the exact copied working-tree snapshot, including uncommitted files.

## Next boot: verification before training

Only after the machine has been powered on and work resumed, copy the small `.zip`, `.verify.py` and receipt from the trusted local output. Use the existing `mahjong-train` Python to verify/unpack with the exact bundle SHA256 printed in the receipt:

```powershell
& "$env:USERPROFILE\miniforge3\envs\mahjong-train\python.exe" D:\mahjong\incoming\training-code-v3.verify.py unpack `
  --bundle D:\mahjong\incoming\training-code-v3.zip `
  --expected-sha256 <trusted-receipt-SHA256> `
  --destination D:\mahjong\code-curated-v3
```

From that fresh checkout, preflight the data first. The default is read-only:

```powershell
python scripts/prepare_windows_data.py `
  --root D:\mahjong\data `
  --archive D:\mahjong\data\roboflow-v2-coco.zip `
  --metadata bundle-inputs/dataset-metadata.json `
  --split-manifest bundle-inputs/split-manifest.jsonl `
  --prepared-name curated-v3-roi --dry-run
```

Then repeat with `--apply` to generate a new, hash-verified dataset directory. The helper verifies archive/selection SHA256, source IDs, split counts, source group separation and selected source bytes. It checks ZIP members for POSIX/Windows path escapes, drive paths, NTFS streams, symlinks and case aliases before creating files. It verifies each prepared COCO and selected image after materialization; an existing `_ready.json` is rechecked, not trusted alone. New output is staged and renamed only after validation.

`specs/curated-v3/rfdetr-small.plan.json` and `rtdetr-r18.plan.json` keep the two experiments separate. They are **not launchable job specs**, and both runner entry points explicitly reject them. Their CPU `*PreflightArgv` commands include `--preflight-only --verify-images` and the frozen data manifest. After those pass, verify current CUDA/environment and run the proposed serial GPU smoke. Create new launchable run specs only after these technical checks; do not add `--resume` or reuse the old v2 output directory.

The RF plan uses the already verified public Small cache at `D:\mahjong\models\rf-detr-small.pth`. The RT plan references `D:\mahjong\pretrained\rtdetr-v2-r18vd`; its separately verified three pretrained files may need to be copied on the next boot. Weight files are intentionally outside the code bundle. Read the historical `verified/environment.json` and exact package lock, then recheck the actual machine rather than claiming the old snapshot is current.

The v3 validation scope is **oracle ROI auxiliary evaluation**. Its crop uses annotation-derived boundaries and does not establish realistic user framing or end-to-end phone performance. All generated plans remain `productionValidated=false`; the sealed test split must not be used to tune architecture, thresholds or exports.

## Local checks

```bash
python -m unittest infra.windows.tests.test_offline_preparation -v
python -m py_compile scripts/prepare_windows_data.py infra/windows/offline_bundle.py infra/windows/experiment_plans.py
```

Synthetic fixtures exercise dry-run non-mutation, atomic preparation/reuse, image tampering, group leakage, old-v2 rejection, Windows/POSIX path escapes, archive symlinks/case aliases, bundle tampering and fresh-directory unpacking. These are local preparation tests, not GPU training or model-accuracy tests.
