/**
 * Client-side image utilities for Clinical Curator.
 * Aspect-ratio-preserving resize for display (actual preprocessing
 * is handled server-side with letterboxing + ImageNet normalisation).
 */

export interface ImageMeta {
  width: number;
  height: number;
  size: number;
  name: string;
  dataUrl: string;
}

/** Load a File and return metadata + data URL. */
export async function readImageFile(file: File): Promise<ImageMeta> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const img = new Image();
      img.onload = () =>
        resolve({
          width: img.naturalWidth,
          height: img.naturalHeight,
          size: file.size,
          name: file.name,
          dataUrl,
        });
      img.onerror = reject;
      img.src = dataUrl;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/** Validate an uploaded image file. */
export type UploadError =
  | 'wrong_format'
  | 'too_small'
  | 'too_large'
  | 'generic';

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/dicom', 'application/dicom']);
const ALLOWED_EXTS  = new Set(['.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.nii', '.nii.gz']);

export function validateUpload(file: File, meta?: ImageMeta): UploadError | null {
  const ext = '.' + file.name.split('.').pop()!.toLowerCase();
  if (!ALLOWED_TYPES.has(file.type) && !ALLOWED_EXTS.has(ext)) {
    return 'wrong_format';
  }
  if (file.size > 50 * 1024 * 1024) return 'too_large';
  if (meta && (meta.width < 128 || meta.height < 128)) return 'too_small';
  return null;
}

/** Overlay a base64 GradCAM PNG on top of an HTMLCanvasElement. */
export function overlayGradCAM(
  canvas: HTMLCanvasElement,
  imageEl: HTMLImageElement,
  gradCamB64: string,
  opacity: number,  // 0–100
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(imageEl, 0, 0, width, height);
  if (!gradCamB64 || opacity === 0) return;
  const overlay = new Image();
  overlay.onload = () => {
    ctx.globalAlpha = opacity / 100;
    ctx.drawImage(overlay, 0, 0, width, height);
    ctx.globalAlpha = 1;
  };
  overlay.src = `data:image/png;base64,${gradCamB64}`;
}
