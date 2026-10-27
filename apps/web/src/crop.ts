export interface CropEdges { left: number; top: number; right: number; bottom: number }
export interface CropRect { x: number; y: number; width: number; height: number }
export const FULL_CROP: CropEdges = { left: 0, top: 0, right: 1, bottom: 1 };
export const MIN_CROP_PIXELS = 32;

export function cropRect(edges: CropEdges, width: number, height: number): CropRect {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < MIN_CROP_PIXELS || height < MIN_CROP_PIXELS
    || ![edges.left, edges.top, edges.right, edges.bottom].every(value => Number.isFinite(value) && value >= 0 && value <= 1)
    || edges.left >= edges.right || edges.top >= edges.bottom) throw new Error('请选择有效的牌面区域。');
  const x = Math.round(edges.left * width), y = Math.round(edges.top * height);
  const w = Math.round(edges.right * width) - x, h = Math.round(edges.bottom * height) - y;
  if (w < MIN_CROP_PIXELS || h < MIN_CROP_PIXELS) throw new Error('选取区域太小，请保留完整、清楚的牌面。');
  return { x, y, width: w, height: h };
}

/** Save the selected integer pixel boundaries, so later rotation cannot drift. */
export function canonicalCropEdges(edges: CropEdges, width: number, height: number): CropEdges {
  const rect = cropRect(edges, width, height);
  return { left: rect.x / width, top: rect.y / height,
    right: (rect.x + rect.width) / width, bottom: (rect.y + rect.height) / height };
}

export function rotateCropClockwise(edges: CropEdges): CropEdges {
  return { left: 1 - edges.bottom, top: edges.left, right: 1 - edges.top, bottom: edges.right };
}

export function isFullCrop(edges: CropEdges) {
  return edges.left === 0 && edges.top === 0 && edges.right === 1 && edges.bottom === 1;
}
