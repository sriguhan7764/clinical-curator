interface Props {
  imageUrl: string;
  gradCamBase64?: string;
  opacity?: number;        // 0–100
  heatmapEnabled?: boolean;
  pathologicalFocus?: string;
  altText?: string;
  height?: number;
}

export default function XrayViewer({
  imageUrl,
  gradCamBase64 = '',
  opacity = 65,
  heatmapEnabled = true,
  pathologicalFocus,
  altText = 'Chest X-ray image',
  height = 380,
}: Props) {
  return (
    <div
      role="img"
      aria-label={altText}
      style={{
        position: 'relative',
        width: '100%',
        height,
        background: '#050810',
        overflow: 'hidden',
        flexShrink: 0,
      }}
    >
      {/* ── Base X-ray — always visible, never stretched ── */}
      <img
        src={imageUrl}
        alt={altText}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          display: 'block',
        }}
      />

      {/* ── GradCAM heatmap overlay — controlled by opacity + toggle ── */}
      {gradCamBase64 && (
        <img
          src={`data:image/png;base64,${gradCamBase64}`}
          aria-hidden="true"
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            opacity: heatmapEnabled ? opacity / 100 : 0,
            mixBlendMode: 'screen',
            transition: 'opacity 0.3s ease',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* ── LIVE AI PROCESSING pill — top right, compact ── */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          background: 'rgba(26,95,168,0.82)',
          backdropFilter: 'blur(6px)',
          color: '#fff',
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          padding: '3px 9px',
          borderRadius: 999,
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          whiteSpace: 'nowrap',
          lineHeight: 1.6,
          zIndex: 2,
        }}
      >
        <span
          style={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: '#4ade80',
            flexShrink: 0,
            animation: 'xray-pulse 1.5s ease-in-out infinite',
          }}
        />
        LIVE AI PROCESSING
      </div>

      {/* ── PATHOLOGICAL FOCUS — bottom left, not center ── */}
      {pathologicalFocus && (
        <div
          style={{
            position: 'absolute',
            bottom: 14,
            left: 14,
            background: 'rgba(211,47,47,0.86)',
            backdropFilter: 'blur(4px)',
            color: '#fff',
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            padding: '3px 9px',
            borderRadius: 4,
            zIndex: 2,
            whiteSpace: 'nowrap',
            lineHeight: 1.6,
          }}
        >
          PATHOLOGICAL FOCUS: {pathologicalFocus}
        </div>
      )}

      <style>{`
        @keyframes xray-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
      `}</style>
    </div>
  );
}
