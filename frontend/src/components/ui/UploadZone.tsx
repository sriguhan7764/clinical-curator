import { useRef, useState, DragEvent, KeyboardEvent } from 'react';
import { readImageFile, validateUpload } from '../../utils/imageProcessing';
import type { ImageMeta, UploadError } from '../../utils/imageProcessing';
import { formatFileSize } from '../../utils/format';

const ERROR_MESSAGES: Record<UploadError, string> = {
  wrong_format: 'Unsupported format. Use DICOM, JPG, or PNG.',
  too_small:    'Image too small (min 128×128px required for analysis).',
  too_large:    'File too large. Maximum size is 50MB.',
  generic:      'Upload failed. Please try again.',
};

const ERROR_BORDER: Record<UploadError, string> = {
  wrong_format: '#D32F2F',
  too_small:    '#E65100',
  too_large:    '#E65100',
  generic:      '#D32F2F',
};

interface Props {
  onFile: (file: File, meta: ImageMeta) => void;
}

export default function UploadZone({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [state, setState]   = useState<'idle' | 'dragover' | 'uploading' | 'loaded' | 'error'>('idle');
  const [meta, setMeta]     = useState<ImageMeta | null>(null);
  const [uploadError, setUploadError] = useState<UploadError | null>(null);

  const handleFile = async (file: File) => {
    setState('uploading');
    setUploadError(null);
    try {
      const m = await readImageFile(file);
      const err = validateUpload(file, m);
      if (err) {
        setUploadError(err);
        setState('error');
        return;
      }
      setMeta(m);
      setState('loaded');
      onFile(file, m);
    } catch {
      setUploadError('generic');
      setState('error');
    }
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
  };

  /* ── border / background based on state ── */
  let borderColor = 'rgba(26,95,168,0.4)';
  let borderStyle = '2px dashed';
  let bg          = 'transparent';

  if (state === 'dragover')  { borderColor = 'var(--cc-brand)'; borderStyle = '2px solid'; bg = 'var(--cc-brand-light)'; }
  if (state === 'error' && uploadError) { borderColor = ERROR_BORDER[uploadError]; borderStyle = '2px solid'; }
  if (state === 'loaded')    { borderColor = 'var(--cc-brand)'; borderStyle = '2px solid'; }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload X-ray image. Press Enter or Space to open file picker."
      onDragOver={(e) => { e.preventDefault(); setState('dragover'); }}
      onDragLeave={() => setState(meta ? 'loaded' : 'idle')}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      onKeyDown={onKeyDown}
      style={{
        minHeight: 280,
        border: `${borderStyle} ${borderColor}`,
        borderRadius: 'var(--cc-radius-card)',
        background: bg,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        transition: 'all 0.2s',
        padding: 24,
        position: 'relative',
        outline: 'none',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.dcm,.dicom,.nii,.nii.gz"
        style={{ display: 'none' }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />

      {/* ── LOADED STATE ── */}
      {state === 'loaded' && meta ? (
        <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ position: 'relative' }}>
            <img
              src={meta.dataUrl}
              alt={`Loaded: ${meta.name}`}
              style={{
                width: '100%',
                maxHeight: 200,
                objectFit: 'contain',
                borderRadius: 8,
                background: '#0a0a0a',
              }}
            />
            <span
              style={{
                position: 'absolute',
                top: 8,
                left: 8,
                background: 'rgba(26,95,168,0.9)',
                color: '#fff',
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                padding: '2px 8px',
                borderRadius: 4,
              }}
            >
              LENS SCAN ACTIVE
            </span>
          </div>
          <div style={{ fontSize: 13, color: '#506071' }}>
            <strong style={{ color: '#191c1e' }}>{meta.name}</strong>
            &nbsp;·&nbsp;{formatFileSize(meta.size)}
            &nbsp;·&nbsp;{meta.width}×{meta.height}px
          </div>
        </div>
      ) : state === 'error' && uploadError ? (
        /* ── ERROR STATE ── */
        <>
          <span className="material-symbols-outlined" style={{ fontSize: 40, color: ERROR_BORDER[uploadError], marginBottom: 12 }}>
            error_outline
          </span>
          <p style={{ color: ERROR_BORDER[uploadError], fontWeight: 600, textAlign: 'center' }}>
            {ERROR_MESSAGES[uploadError]}
          </p>
          <button
            onClick={(e) => { e.stopPropagation(); setState('idle'); setUploadError(null); }}
            style={{
              marginTop: 12,
              padding: '6px 16px',
              border: `1px solid ${ERROR_BORDER[uploadError]}`,
              borderRadius: 6,
              background: 'transparent',
              color: ERROR_BORDER[uploadError],
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Try again
          </button>
        </>
      ) : state === 'uploading' ? (
        /* ── UPLOADING STATE ── */
        <>
          <span className="material-symbols-outlined" style={{ fontSize: 40, color: 'var(--cc-brand)', marginBottom: 12, animation: 'spin 1s linear infinite' }}>
            progress_activity
          </span>
          <p style={{ color: '#506071', fontWeight: 500 }}>Processing image…</p>
        </>
      ) : (
        /* ── IDLE STATE ── */
        <>
          <span className="material-symbols-outlined" style={{ fontSize: 48, color: 'var(--cc-brand)', marginBottom: 16 }}>
            upload_file
          </span>
          <p style={{ fontSize: 16, fontWeight: 600, color: '#191c1e', marginBottom: 6 }}>
            Drag and drop DICOM series
          </p>
          <p style={{ fontSize: 14, color: '#506071', textAlign: 'center' }}>
            or{' '}
            <span style={{ color: 'var(--cc-brand)', fontWeight: 600 }}>
              browse files from workstation
            </span>
          </p>
          <p style={{ marginTop: 24, fontSize: 11, fontWeight: 600, color: '#b0b8c1', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            SUPPORTED: DICOM, JPG, PNG, NIFTI
          </p>
        </>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
