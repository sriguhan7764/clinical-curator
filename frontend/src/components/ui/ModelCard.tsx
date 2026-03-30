import type { ModelInfo } from '../../types/api';
import { formatAUC } from '../../utils/format';

interface Props {
  model: ModelInfo;
  selected?: boolean;
  onSelect?: (id: string) => void;
}

export default function ModelCard({ model, selected = false, onSelect }: Props) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSelect?.(model.id)}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect?.(model.id); }}
      style={{
        background: 'var(--cc-card-bg)',
        border: selected ? '2px solid var(--cc-brand)' : '1px solid var(--cc-border)',
        borderRadius: 'var(--cc-radius-card)',
        padding: '16px 20px',
        cursor: 'pointer',
        transition: 'all 0.15s',
        outline: 'none',
        boxShadow: selected ? '0 0 0 3px rgba(26,95,168,0.15)' : 'var(--cc-shadow-card)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <span
            style={{
              display: 'inline-block',
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: 'var(--cc-brand)',
              background: 'var(--cc-brand-light)',
              padding: '2px 6px',
              borderRadius: 3,
              marginBottom: 4,
            }}
          >
            {model.review}
          </span>
          <p style={{ fontSize: 15, fontWeight: 600, color: '#191c1e' }}>{model.name}</p>
          <p style={{ fontSize: 12, color: '#767676', marginTop: 2 }}>{model.description}</p>
        </div>
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: '#506071',
            background: 'rgba(0,0,0,0.05)',
            padding: '2px 8px',
            borderRadius: 3,
            whiteSpace: 'nowrap',
          }}
        >
          {model.version_tag}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 12 }}>
        <div>
          <p style={{ fontSize: 11, fontWeight: 500, color: '#767676', textTransform: 'uppercase', letterSpacing: '0.04em' }}>AUC</p>
          <p style={{ fontSize: 16, fontWeight: 700, color: 'var(--cc-brand)' }}>{formatAUC(model.auc_score)}</p>
        </div>
        <div>
          <p style={{ fontSize: 11, fontWeight: 500, color: '#767676', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Inference</p>
          <p style={{ fontSize: 16, fontWeight: 700, color: '#191c1e' }}>{model.inference_ms}ms</p>
        </div>
        {model.is_temporal && (
          <span
            style={{
              alignSelf: 'flex-end',
              fontSize: 10,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--cc-brand)',
              border: '1px solid var(--cc-brand)',
              padding: '2px 6px',
              borderRadius: 3,
            }}
          >
            Temporal
          </span>
        )}
      </div>
    </div>
  );
}
