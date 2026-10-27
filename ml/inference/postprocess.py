"""Pure output normalization shared by training validation and HTTP inference."""
import math

MAX_DETECTIONS = 200


def normalize_tiles(tiles: list[dict], width: int, height: int, accept_threshold: float) -> tuple[list[dict], list[str]]:
    normalized = []
    reasons = []
    for t in tiles:
        x, y, w, h = map(float, t["bbox"])
        score = float(t["score"])
        if not all(math.isfinite(v) for v in (x, y, w, h, score)) or not 0 <= score <= 1:
            raise ValueError("non-finite model output")
        x2, y2 = min(width, x + w), min(height, y + h)
        x, y = max(0, x), max(0, y)
        if x2 <= x or y2 <= y:
            continue
        uncertain = score < accept_threshold or t["classId"] == "back"
        box = [round(v, 2) for v in (x, y, x2 - x, y2 - y)]
        if box[2] <= 0 or box[3] <= 0:
            continue
        normalized.append({
            "id": "", "classId": str(t["classId"]), "label": str(t.get("label", t["classId"])),
            "bbox": box,
            "score": round(score, 5), "uncertain": uncertain,
        })
    # Cluster neighboring centers into rows. Ordering is a UI convenience, not a hand-size constraint.
    rows: list[list[dict]] = []
    for tile in sorted(normalized, key=lambda t: t["bbox"][1] + t["bbox"][3] / 2):
        cy = tile["bbox"][1] + tile["bbox"][3] / 2
        for row in rows:
            row_cy = sum(t["bbox"][1] + t["bbox"][3] / 2 for t in row) / len(row)
            row_h = sum(t["bbox"][3] for t in row) / len(row)
            if abs(cy - row_cy) < 0.55 * min(row_h, tile["bbox"][3]):
                row.append(tile)
                break
        else:
            rows.append([tile])
    ordered = [t for row in rows for t in sorted(row, key=lambda t: t["bbox"][0])]
    for index, tile in enumerate(ordered):
        tile["id"] = str(index + 1)
    if not ordered:
        reasons.append("没有找到清晰的牌面，请靠近并重新拍摄。")
    if any(t["uncertain"] for t in ordered):
        reasons.append("部分牌面不够确定，请核对标记的牌或重新拍摄。")
    if any(t["classId"] == "back" for t in ordered):
        reasons.append("牌背无法判断正面，请将牌面朝向镜头。")
    return ordered, reasons


