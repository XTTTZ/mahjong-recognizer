import type { PreparedPhoto } from './types';
import { cropRect, type CropEdges } from './crop';

export const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;
const MAX_SOURCE_BYTES = 40 * 1024 * 1024;
const MAX_PIXELS = 40_000_000;
const LONG_EDGE = 2560;

interface DecodedImage {
  source: CanvasImageSource;
  width: number;
  height: number;
  close: () => void;
}

async function decode(file: Blob): Promise<DecodedImage> {
  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      return { source: bitmap, width: bitmap.width, height: bitmap.height, close: () => bitmap.close() };
    } catch { /* Safari may decode camera images only through HTMLImageElement. */ }
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return { source: image, width: image.naturalWidth, height: image.naturalHeight, close: () => URL.revokeObjectURL(url) };
  } catch {
    URL.revokeObjectURL(url);
    throw new Error('无法打开这张照片。HEIC 照片可先在相册中转换为 JPEG，或直接拍照重试。');
  }
}

async function encode(source: CanvasImageSource, width: number, height: number, turns = 0): Promise<PreparedPhoto> {
  if (width * height > MAX_PIXELS) throw new Error('照片分辨率太高，请先裁剪到需要识别的手牌。');
  if (Math.min(width, height) < 32) throw new Error('照片太小，请选择能看清完整牌面的照片。');
  const scale = Math.min(1, LONG_EDGE / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = turns % 2 ? h : w;
  canvas.height = turns % 2 ? w : h;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('当前浏览器无法处理照片，请换一个浏览器重试。');
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.translate(canvas.width / 2, canvas.height / 2);
  context.rotate(turns * Math.PI / 2);
  context.drawImage(source, -w / 2, -h / 2, w, h);
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(
    value => value ? resolve(value) : reject(new Error('照片处理失败，请重新选择。')), 'image/jpeg', 0.94,
  ));
  if (blob.size > MAX_UPLOAD_BYTES) throw new Error('照片仍然太大，请裁剪后重新选择。');
  const file = new File([blob], 'mahjong-photo.jpg', { type: 'image/jpeg' });
  return { file, url: URL.createObjectURL(file), width: canvas.width, height: canvas.height };
}

export async function preparePhoto(file: File): Promise<PreparedPhoto> {
  if (!file.size) throw new Error('这张照片是空文件，请重新选择。');
  if (file.size > MAX_SOURCE_BYTES) throw new Error('照片文件过大，请先裁剪到需要识别的手牌。');
  const allowed = /\.(jpe?g|png|webp|heic|heif)$/i.test(file.name)
    || ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif'].includes(file.type);
  if (!allowed) throw new Error('请选择 JPEG、PNG、WebP 或手机拍摄的照片。');
  const image = await decode(file);
  try { return await encode(image.source, image.width, image.height); }
  finally { image.close(); }
}

export async function rotatePhoto(photo: PreparedPhoto): Promise<PreparedPhoto> {
  const image = await decode(photo.file);
  try { return await encode(image.source, image.width, image.height, 1); }
  finally { image.close(); }
}

export async function cropPhoto(photo: PreparedPhoto, edges: CropEdges): Promise<PreparedPhoto> {
  const rect = cropRect(edges, photo.width, photo.height);
  const image = await decode(photo.file);
  try {
    if (image.width !== photo.width || image.height !== photo.height) throw new Error('照片尺寸已改变，请重新选择照片。');
    const canvas = document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('当前浏览器无法裁剪照片，请换一个浏览器重试。');
    context.drawImage(image.source, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
    return await encode(canvas, rect.width, rect.height);
  } finally { image.close(); }
}
