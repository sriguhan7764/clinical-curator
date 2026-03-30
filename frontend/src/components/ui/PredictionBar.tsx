import { getRiskColors, getRiskLevel } from '../../utils/format';
import type { RiskLevel } from '../../types/api';

interface Props {
  label: string;
  /** Integer 0–100 */
  confidencePct: number;
  /** Hide the % text label — use when a ConfidenceBadge already shows it */
  hideValue?: boolean;
  dimThreshold?: number;
}

export default function PredictionBar({
  label,
  confidencePct,
  hideValue = false,
  dimThreshold = 15,
}: Props) {
  const level: RiskLevel = getRiskLevel(confidencePct);
  const { text: barColor } = getRiskColors(level);
  const isDim = confidencePct < dimThreshold;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, opacity: isDim ? 0.5 : 1 }}>
      {/* Only render the label row when we have a label or a value to show */}
      {(label || !hideValue) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {label && (
            <span style={{ fontSize: 13, fontWeight: 500, color: isDim ? '#767676' : '#191c1e' }}>
              {label}
            </span>
          )}
          {!hideValue && (
            <span style={{ fontSize: 12, fontWeight: 600, color: barColor, minWidth: 36, textAlign: 'right' }}>
              {confidencePct}%
            </span>
          )}
        </div>
      )}
      <div style={{ height: 6, borderRadius: 99, background: 'rgba(0,0,0,0.07)', overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${confidencePct}%`,
            borderRadius: 99,
            background: barColor,
            transition: 'width 0.6s ease',
          }}
        />
      </div>
    </div>
  );
}
