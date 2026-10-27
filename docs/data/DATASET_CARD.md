# Mahjong v2 data snapshot

Snapshot source: [mahjong-anaun/mahjong-lfhxf version 2](https://universe.roboflow.com/mahjong-anaun/mahjong-lfhxf/dataset/2), exported 2026-10-25. Project attribution: a Roboflow user in workspace `mahjong`; see the [source project](https://universe.roboflow.com/mahjong-anaun/mahjong-lfhxf). License stated by the project and downloaded README: **CC BY 4.0**. This local work adds audit metadata, excludes known-invalid/duplicate records when preparing research splits, and may change the split assignment; original images and annotations remain unchanged.

## Immutable source

- Local archive: `artifacts/data/roboflow/mahjong-lfhxf-v2/coco.zip` (ignored from Git).
- Archive bytes: **4,447,371,742**.
- SHA256: `14640fb5a47707b35d8f0025260883aff4936a2893f2eba9e07758b063df1268`.
- Raw extracted COCO: `artifacts/data/roboflow/mahjong-lfhxf-v2/original/`.
- Raw splits: train **10,777 images / 92,908 boxes**; valid **718 / 8,514**; test **426 / 4,719**.
- Total: **11,921 images / 106,141 boxes**. This resolves the earlier ambiguity in the platform's class counters: for this snapshot, they match the COCO annotation-instance total.
- The export metadata has `preprocessing={}` and `augmentation={}`; its README says no augmentation was applied. The preprocessing list is blank. This describes version generation, not the unknown processing history before source images were imported.

## Classes

There are **43 annotated classes**: 27 suit tiles, 7 honor tiles, 8 flower/season tiles, and one `back` class. COCO `categories` also has a parent category `id=0, name=mahjong-yolo` with no instances. Do not train that parent category as a tile. Actual original category IDs are 1–43; a model adapter must explicitly save and validate its own contiguous output mapping.

The tracked [class-map.json](./class-map.json) stores original COCO ID, proposed zero-based model index, token, Chinese label, and four example source boxes per class. All 43 tokens were visually checked against these four crops per class. This verifies token meaning, not every label in the dataset.

- `1z–7z`: 东、南、西、北、白板、发、中.
- `1h–8h`: 春、夏、秋、冬、梅、兰、菊、竹. In particular, **7h=菊 and 8h=竹**.
- `m/p/s`: 万、筒、条; `back`: 牌背. White dragon includes both a blue-frame design and plain-white faces; bamboo one has bird-like designs.

## Suitability and known coverage gaps

- **4,821 images (40.4%) contain only one annotated tile.** Images with 14 tiles: **3,342**; with 13 tiles: **432**. Per-image success over the whole mixed dataset can therefore obscure the target whole-hand performance. Report the 13/14-tile and broader multi-tile subsets separately.
- There are **no zero-annotation negative images** in this snapshot. Empty tables, ordinary objects, and non-mahjong phone photos need separate negatives/robustness testing before claiming false-positive safety.
- Most frequent resolutions: 1024² (**6,844**), 1080×1920 (**1,520**), 4032×3024 (**855**), 3024×4032 (**557**). The maximum edge is **5,712** pixels; original-size detail is available for at least part of the data.
- The crop exemplars include multiple tile designs, partial/occluded faces, oblique views and various orientations. A systematic room/device/session/style inventory has not yet been established.
- The original workspace search can cross shared projects' split metadata. Only the immutable export's own split manifest is authoritative for this snapshot.
- Source capture-session identity is unavailable. Even a deduplicated grouped local holdout **does not replace** an independently collected set of real mobile-phone hands from unfamiliar rooms and tile sets.
- A 300–500-photo independent real-world hand benchmark remains a required follow-up if the user wants evidence for the production accuracy target. It must not be fabricated from synthetic augmentation of this training source.

## Audit and research preparation

The reproducible tools are in `ml/data/`. `audit.py` checks decoding/dimensions, box validity, exact duplicate/conflicting annotations, source-name groups and conservative near-duplicate candidates, then builds a deterministic group-level research split. `materialize.py` creates standard COCO directories by hardlink/copy, preserving raw files. Invalid/conflicting examples are excluded for review rather than silently relabeled.

Full audit results and the 240-image contact-sheet sample will be documented below after the pipeline completes. Near-duplicate grouping is heuristic and may both miss same-session images and overgroup similar scenes; the report preserves candidate groups and limitations.
