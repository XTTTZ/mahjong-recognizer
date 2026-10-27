# Visual audit and conservative v3 curation

Date: 2026-10-26. No Windows connection or cloud training was used for this offline curation.

## What was actually reviewed

All 43 class tokens were checked against four crop examples each (172 crops). The initial raw contact-sheet sample exposed dimension errors and non-exhaustive full-table annotation. A narrower source-family pass then selected **353 canonical candidates** from controlled blue-cloth phone photos and keyboard/tabletop photos. Every candidate's annotation-derived ROI was reviewed on **12 contact sheets**, with full-resolution spot checks of important ambiguous examples. This is a visual **scope** review, not an independent per-tile class reannotation or proof every accepted label is correct.

Concrete source IDs, contact indexes and notes are in [visual-audit.json](./visual-audit.json). The complete 353 candidate decisions are in [roi-reviews-v3.json](./roi-reviews-v3.json).

## Why the initial full dataset is not a trustworthy all-frame gold set

- The raw sample `hand_155...` (initial sample index 58) labels the lower hand while visible front faces on the center table remain unannotated. Other broadcast and full-table sources use different scopes. These frames cannot establish whole-frame all-face accuracy without relabeling.
- With the current **43-class ontology**, `back` is a target. Many otherwise useful phone photos label front faces but omit separate yellow/blue/green-backed walls and discarded backs. These omitted backs must not silently be treated as verified negatives. Examples include ROI candidates 197, 206 and 350.
- A future **42-face-only ontology** could legitimately treat backs as background and rescue many such raw photos, eliminating oracle-crop bias. It must be a new version with a new explicit class map and fresh experiments. Broadcast frames with unannotated **fronts** still remain invalid. v3 keeps the existing 43-class contract and does not silently apply this change.
- Blue borders can belong to an annotated tile itself; candidate 198 was checked at full resolution to avoid falsely calling every blue edge a missing back. Uncertain edge cases were quarantined rather than presented as confirmed errors.
- There are 130 clear decoded-size/COCO dimension disagreements and 3 exact byte duplicates inherited from the structural v2 audit. Original bytes/labels remain intact. No geometry is guessed or silently repaired.

## Source quantity is much smaller than exported image count

The `S__` blue-cloth family contains **6,663 exported images but only 298 pre-export filename stems**. Each stem commonly has 21–26 different imported variants. The version export's `augmentation={}` therefore does not mean there are 6,663 independent original phone captures; transformations existed before import. Two stems containing broad 34-tile layouts were kept out of this initial hand-oriented whitelist. The v3 candidate pool contains **296 blue-cloth stems plus 57 canonical keyboard/tabletop sources** (one representative per original numeric source identity across pre-rotated variants).

Broadcast screenshots, web compilations, unknown camera families and noncanonical variants are **heuristically quarantined**, not asserted to be uniformly bad. Only specific visually supported problems are marked `manual_confirmed_scope_problem`.

## Frozen v3: oracle target-ROI research only

The crop is the union of source annotation boxes plus **10% of the median box shorter side**, floored/ceiled and clamped to raw image bounds. This uses ground truth: it is an **oracle**, not a user selection or learned localization result. The crop does not itself prove completeness; explicit visual scope review is still required. Any strictly out-of-bounds raw box is quarantined rather than clipped, even if the cause is a 0.01-pixel export rounding artifact.

Of the 353 candidates, **224 were accepted**, **59 rejected for visibly unannotated background backs**, and **70 quarantined because background/cutoff/context or review-resolution ambiguity remained**. Each decision is tracked. Accepted crop PNGs were encoded using Pillow **11.3.0**, with source SHA256, PNG SHA256 and decoded RGB pixel SHA256 recorded. The reviewed candidate pixels were checked equal under the pinned final encoder/runtime.

| Split | Images | Boxes | Images with at least 10 boxes | Groups |
| --- | ---: | ---: | ---: | ---: |
| train | 207 | 892 | 51 | 90 |
| valid | 11 | 49 | 3 | 10 |
| test | 6 | 56 | 4 | 6 |

The largest retained training group has **109 images**, mainly related single-tile views. The validation set is much too small for architecture selection or 95%/99% production claims. It is not an independently collected real-room/user-selected-ROI benchmark. Controlled surfaces and a small number of physical sets dominate. The `back` class has zero selected instances in every split; all other 42 classes have at least one training box, but several classes have no validation/test coverage. Full per-class counts and zero-coverage lists are in [final-dataset-v3.json](./final-dataset-v3.json).

## Reproduction

Reuse the immutable raw v2 archive; no second multi-gigabyte archive is required. The tracked [split-v3.jsonl](./split-v3.jsonl) records the final source selection, original group/split, crop geometry, decision type and hashes. No v2 group or sealed-test image is moved into another split. Nonselected variants are excluded, so choosing one canonical numeric source identity does not leave its rotated peers in another retained split.

```sh
.venv-tools/bin/python ml/data/materialize_v3.py \
  --archive artifacts/data/roboflow/mahjong-lfhxf-v2/coco.zip \
  --split-manifest docs/data/split-v3.jsonl \
  --out artifacts/data/roboflow/mahjong-lfhxf-v2/curated-v3-roi
```

Use a new empty output directory. The materializer verifies source bytes and dimensions, all 43 category IDs, exact bbox validity, integer crop bounds, RGB pixels and PNG bytes. The resulting COCO contains source IDs, group IDs and crop provenance. A changed decoder/encoder that changes content fails verification and requires an explicitly versioned rebuild. v3 experiments must start fresh from official pretrained weights, not resume the old v2 experiment. The research-only status remains until adequate independent gold data and actual end-to-end tests exist.
