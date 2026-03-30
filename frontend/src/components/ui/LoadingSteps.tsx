import { useEffect, useState } from 'react';

const STEPS = [
  'Preprocessing image (letterbox + normalise)',
  'Running forward pass',
  'Computing GradCAM heatmap',
  'Ranking predictions',
  'Generating clinical insight',
];

export default function LoadingSteps() {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    if (current >= STEPS.length) return;
    const t = setTimeout(() => setCurrent((c) => c + 1), 900);
    return () => clearTimeout(t);
  }, [current]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '24px 0' }}>
      {STEPS.map((step, i) => (
        <div
          key={step}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            opacity: i > current ? 0.35 : 1,
            transition: 'opacity 0.4s',
          }}
        >
          {i < current ? (
            <span className="material-symbols-outlined" style={{ color: 'var(--cc-normal-fg)', fontSize: 20 }}>
              check_circle
            </span>
          ) : i === current ? (
            <span
              className="material-symbols-outlined"
              style={{ color: 'var(--cc-brand)', fontSize: 20, animation: 'spin 1s linear infinite' }}
            >
              progress_activity
            </span>
          ) : (
            <span className="material-symbols-outlined" style={{ color: '#c0c8d0', fontSize: 20 }}>
              radio_button_unchecked
            </span>
          )}
          <span style={{ fontSize: 14, fontWeight: i <= current ? 500 : 400, color: i <= current ? '#191c1e' : '#b0b8c1' }}>
            {step}
          </span>
        </div>
      ))}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
