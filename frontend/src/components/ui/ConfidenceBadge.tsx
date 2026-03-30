import { getRiskColors, getRiskLevel } from '../../utils/format';
import type { RiskLevel } from '../../types/api';

interface Props {
  /** Raw 0–1 float OR integer 0–100 (detected by value > 1) */
  confidence: number;
  /** If already an integer pct, pass directly and set raw=false */
  raw?: boolean;
}

export default function ConfidenceBadge({ confidence, raw = true }: Props) {
  const pct: number = raw ? Math.round(confidence * 100) : confidence;
  const level: RiskLevel = getRiskLevel(pct);
  const { bg, text } = getRiskColors(level);

  return (
    <span
      role="status"
      aria-label={`${level}: ${pct}% confidence`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        padding: '2px 8px',
        borderRadius: 'var(--cc-radius-badge)',
        background: bg,
        color: text,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
        whiteSpace: 'nowrap',
      }}
    >
      {pct}% · {level}
    </span>
  );
}
