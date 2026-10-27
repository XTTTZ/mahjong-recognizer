# Immutable data audit and grouped COCO preparation

The Roboflow source is `mahjong-anaun/mahjong-lfhxf`, frozen version **2**, COCO format. Original annotations are never edited. `artifacts/data/` is ignored so photos, archives and signed download URLs do not enter Git.

Run with Python 3.11+ and Pillow/numpy:

```sh
python ml/data/audit.py --dataset artifacts/data/roboflow/mahjong-lfhxf-v2/original --out artifacts/data/roboflow/mahjong-lfhxf-v2/audit
python ml/data/materialize.py --root artifacts/data/roboflow/mahjong-lfhxf-v2/original --manifest artifacts/data/roboflow/mahjong-lfhxf-v2/audit/manifest.jsonl --out artifacts/data/roboflow/mahjong-lfhxf-v2/prepared
```

The audit writes image/annotation hashes, class and size/count statistics, malformed-box checks, duplicate groups, visual contact sheets, and a deterministic group-level 80/10/10 research split. Exact byte duplicates are deduplicated; exact duplicate images with conflicting annotations are excluded pending review. Invalid/unreadable images and invalid/out-of-bounds boxes are excluded pending review, not silently repaired.

Grouping combines the pre-export filename, image SHA256 and conservative dHash+aspect-ratio+color similarity. The manifest keeps the original split and proposed research split. The near-duplicate rules are an engineering heuristic; they do not identify all same-room/session/physical-tile sources. The prepared holdout is consequently **not** an independently collected real-world phone benchmark.

Materialization uses hardlinks where supported and copies otherwise. Each prepared split has its own `_annotations.coco.json` and image files, suitable for standard COCO loaders. Original category IDs are preserved; a training adapter should explicitly map these to contiguous model outputs and save that mapping with the checkpoint.

A failed export/download can be requested again with the Roboflow `versions_export` tool. Use the existing version; do not create a version, modify the shared annotation group, or enable paid cloud training. Do not commit the temporary download URL/key. Check the archive SHA256 and `zipfile` integrity before reusing it. Confirm the project attribution and CC BY 4.0 license in the downloaded README/license before distribution.
