import type { RiskLevel } from '../types/api';

/**
 * Convert raw 0–1 float → integer 0–100.
 * Never produces leading zeros (4 not "04").
 */
export function formatConfidence(raw: number): number {
  return Math.round(raw * 100);
}

/** Determine risk level from an integer confidence percentage. */
export function getRiskLevel(pct: number): RiskLevel {
  if (pct >= 70) return 'CRITICAL';
  if (pct >= 40) return 'CAUTION';
  return 'NORMAL';
}

/** Return CSS background + text colors for a given risk level. */
export function getRiskColors(level: RiskLevel): { bg: string; text: string } {
  const map: Record<RiskLevel, { bg: string; text: string }> = {
    CRITICAL: { bg: 'var(--cc-critical-bg)', text: 'var(--cc-critical-fg)' },
    CAUTION:  { bg: 'var(--cc-caution-bg)',  text: 'var(--cc-caution-fg)'  },
    NORMAL:   { bg: 'var(--cc-normal-bg)',   text: 'var(--cc-normal-fg)'   },
  };
  return map[level] ?? map.NORMAL;
}

/** Format a file size in bytes to a human-readable string. */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Format AUC score to 2 decimal places. */
export function formatAUC(score: number): string {
  return score.toFixed(2);
}

/** Format a date/ISO string into a readable label. */
export function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
    });
  } catch {
    return iso;
  }
}
