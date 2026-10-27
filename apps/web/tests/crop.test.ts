import { describe, expect, it } from 'vitest';
import { canonicalCropEdges, cropRect, FULL_CROP, rotateCropClockwise, type CropEdges } from '../src/crop';

describe('explicit image-region geometry', () => {
  it('keeps the full image by default and converts user edges to source pixels', () => {
    expect(cropRect(FULL_CROP, 1400, 700)).toEqual({ x: 0, y: 0, width: 1400, height: 700 });
    expect(cropRect({ left: .1, top: .2, right: .9, bottom: .8 }, 1000, 500)).toEqual({ x: 100, y: 100, width: 800, height: 300 });
  });

  it('rounds independent edges rather than extending a crop beyond the image', () => {
    const result = cropRect({ left: .125, top: .125, right: 1, bottom: 1 }, 333, 151);
    expect(result).toEqual({ x: 42, y: 19, width: 291, height: 132 });
    expect(result.x + result.width).toBe(333);
    expect(result.y + result.height).toBe(151);
  });

  it('rejects inverted, non-finite, out-of-bounds and unreadably tiny selections', () => {
    const invalid: CropEdges[] = [
      { ...FULL_CROP, left: 1 }, { ...FULL_CROP, right: -1 }, { ...FULL_CROP, bottom: 2 },
      { ...FULL_CROP, top: Number.NaN }, { ...FULL_CROP, bottom: Number.POSITIVE_INFINITY },
      { left: .4, right: .42, top: 0, bottom: 1 }, { left: 0, right: 1, top: .1, bottom: .13 },
    ];
    for (const region of invalid) expect(() => cropRect(region, 1000, 500)).toThrow();
    expect(() => cropRect(FULL_CROP, 31, 500)).toThrow();
    expect(() => cropRect(FULL_CROP, 1000.5, 500)).toThrow();
  });

  it('preserves the exact selected pixels after clockwise source rotation', () => {
    for (const [width, height] of [[333, 151], [1400, 700], [2560, 1920]]) {
      const original = canonicalCropEdges({ left: .125, top: .25, right: .875, bottom: .9 }, width, height);
      const before = cropRect(original, width, height);
      const after = cropRect(rotateCropClockwise(original), height, width);
      expect(after).toEqual({ x: height - before.y - before.height, y: before.x, width: before.height, height: before.width });
    }
  });

  it('returns to the same integer source pixels after four rotations', () => {
    const initial = canonicalCropEdges({ left: .125, top: .25, right: .875, bottom: .9 }, 333, 151);
    let rotated = initial;
    for (let i = 0; i < 4; i++) rotated = rotateCropClockwise(rotated);
    expect(cropRect(rotated, 333, 151)).toEqual(cropRect(initial, 333, 151));
  });
});
