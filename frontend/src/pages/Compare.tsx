import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Layout from '../components/layout/Layout';
import XrayViewer from '../components/ui/XrayViewer';
import PredictionBar from '../components/ui/PredictionBar';
import { usePrediction } from '../hooks/usePrediction';
import { getRiskColors, getRiskLevel } from '../utils/format';
import type { BatchPredictItem, PredictResponse } from '../types/api';

interface Props { darkMode: boolean; onToggleDark: () => void; }

const COMPARE_MODELS = ['mlp_r1', 'cnn_r1', 'densenet121_r4'];

const PLACEHOLDER_IMG = 'data:image/svg+xml,' + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224"><rect width="224" height="224" fill="#0a0a0a"/><text x="112" y="120" text-anchor="middle" fill="rgba(255,255,255,0.15)" font-size="12" font-family="sans-serif">X-Ray</text></svg>'
);

export default function Compare({ darkMode, onToggleDark }: Props) {
  const { state }    = useLocation();
  const navigate     = useNavigate();
  const singleResult = state?.result as PredictResponse | undefined;
  const preloadedBatch = state?.batchResult as BatchPredictItem[] | undefined;

  const { status, batchResult: fetchedBatch, predictBatch, error } = usePrediction();
  const [file, setFile] = useState<File | null>(state?.file instanceof File ? state.file : null);
  // Stable object URL for the uploaded/sample file — recreated only when file changes
  const [imageUrl, setImageUrl] = useState<string>(() =>
    state?.file instanceof File ? URL.createObjectURL(state.file) : PLACEHOLDER_IMG
  );

  // If we came from Results/Upload with a file but no preloaded batch, run batch
  useEffect(() => {
    if (!preloadedBatch && state?.file instanceof File) {
      setFile(state.file);
      setImageUrl(URL.createObjectURL(state.file));
      predictBatch(state.file, COMPARE_MODELS);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const batchResult = preloadedBatch ?? fetchedBatch;

  const results: BatchPredictItem[] = batchResult ?? [];

  // Determine best model (highest top prediction confidence)
  const bestModelName = results.reduce<string | null>((best, r) => {
    const topConf = r.predictions[0]?.confidence_pct ?? 0;
    if (!best) return r.model_name;
    const bestConf = results.find(x => x.model_name === best)?.predictions[0]?.confidence_pct ?? 0;
    return topConf > bestConf ? r.model_name : best;
  }, null);

  // Agreement score: std dev of top confidences (lower = more agreement)
  const topConfs = results.map(r => r.predictions[0]?.confidence_pct ?? 0);
  const mean     = topConfs.length ? topConfs.reduce((a, b) => a + b, 0) / topConfs.length : 0;
  const variance = topConfs.length ? topConfs.reduce((s, v) => s + (v - mean) ** 2, 0) / topConfs.length : 0;
  const agreementScore = topConfs.length ? Math.max(0, 100 - Math.sqrt(variance)).toFixed(2) : '—';
  const ensembleConf   = topConfs.length ? (mean).toFixed(2) : '—';

  return (
    <Layout darkMode={darkMode} onToggleDark={onToggleDark}>
      <div style={{ maxWidth: 1400, margin: '0 auto', padding: '28px 24px 40px' }}>
        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: '#767676',
              display: 'block',
              marginBottom: 6,
            }}
          >
            DIAGNOSTIC PROTOCOL
          </span>
          <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 12 }}>
            Multi-Architecture Performance
          </h1>
          <div style={{ display: 'flex', gap: 8 }}>
            {['MODALITY: CHEST X-RAY', 'CONTRAST: STANDARD'].map((chip) => (
              <span
                key={chip}
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                  padding: '4px 12px',
                  borderRadius: 20,
                  background: 'var(--cc-card-bg)',
                  border: '1px solid var(--cc-border)',
                  color: '#506071',
                }}
              >
                {chip}
              </span>
            ))}
          </div>
        </div>

        {/* Upload if no file */}
        {!file && !batchResult && (
          <div
            style={{
              background: 'var(--cc-card-bg)',
              border: '1px solid var(--cc-border)',
              borderRadius: 'var(--cc-radius-card)',
              padding: 32,
              textAlign: 'center',
              marginBottom: 24,
            }}
          >
            <p style={{ fontSize: 15, color: '#767676', marginBottom: 16 }}>
              Upload a chest X-ray to compare all models.
            </p>
            <label
              style={{
                padding: '10px 24px',
                background: 'var(--cc-brand)',
                color: '#fff',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>upload_file</span>
              Select Image
              <input
                type="file"
                accept=".jpg,.jpeg,.png"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    setFile(f);
                    setImageUrl(URL.createObjectURL(f));
                    predictBatch(f, COMPARE_MODELS);
                  }
                }}
              />
            </label>
          </div>
        )}

        {status === 'loading' && (
          <div style={{ padding: 32, textAlign: 'center', color: '#506071' }}>
            Running inference across {COMPARE_MODELS.length} models…
          </div>
        )}

        {error && (
          <div style={{ padding: '12px 16px', background: 'var(--cc-critical-bg)', borderRadius: 8, color: 'var(--cc-critical-fg)', fontSize: 13, fontWeight: 500, marginBottom: 16 }}>
            {error}
          </div>
        )}

        {/* Model cards grid — 3 cols, horizontal scroll below 1024px */}
        {results.length > 0 && (
          <>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, minmax(320px, 1fr))',
                gap: 20,
                overflowX: 'auto',
              }}
            >
              {results.map((res) => {
                const isBest = res.model_name === bestModelName;
                return (
                  <div
                    key={res.model_name}
                    style={{
                      background: 'var(--cc-card-bg)',
                      border: isBest ? '2px solid var(--cc-normal-fg)' : '1px solid var(--cc-border)',
                      borderRadius: 'var(--cc-radius-card)',
                      overflow: 'hidden',
                      boxShadow: 'var(--cc-shadow-card)',
                      position: 'relative',
                    }}
                  >
                    {/* Best badge */}
                    {isBest && (
                      <div
                        style={{
                          position: 'absolute',
                          top: 12,
                          right: 12,
                          zIndex: 10,
                          background: 'var(--cc-normal-fg)',
                          color: '#fff',
                          fontSize: 10,
                          fontWeight: 700,
                          letterSpacing: '0.06em',
                          textTransform: 'uppercase',
                          padding: '3px 8px',
                          borderRadius: 4,
                        }}
                      >
                        BEST FOR THIS CASE
                      </div>
                    )}

                    {/* Card header */}
                    <div style={{ padding: '16px 18px 12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                          <p style={{ fontSize: 15, fontWeight: 600, color: '#191c1e' }}>{res.model_name}</p>
                          <p style={{ fontSize: 12, color: '#767676', marginTop: 2 }}>
                            {res.model_version} · {res.inference_ms}ms
                          </p>
                        </div>
                        <span
                          style={{
                            fontSize: 10,
                            fontWeight: 700,
                            letterSpacing: '0.05em',
                            textTransform: 'uppercase',
                            padding: '2px 8px',
                            background: 'rgba(0,0,0,0.05)',
                            borderRadius: 3,
                            color: '#506071',
                          }}
                        >
                          {res.viz_type}
                        </span>
                      </div>
                    </div>

                    {/* X-ray with GradCAM */}
                    <XrayViewer
                      imageUrl={imageUrl}
                      gradCamBase64={res.grad_cam_base64}
                      opacity={60}
                      heatmapEnabled={true}
                      altText={`X-ray for model ${res.model_name}`}
                      height={220}
                    />

                    {/* Predictions */}
                    <div style={{ padding: '16px 18px' }}>
                      <p
                        style={{
                          fontSize: 11,
                          fontWeight: 700,
                          textTransform: 'uppercase',
                          letterSpacing: '0.05em',
                          color: '#767676',
                          marginBottom: 10,
                        }}
                      >
                        TOP PREDICTIONS
                      </p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {res.predictions.slice(0, 6).map((pred) => {
                          const colors = getRiskColors(getRiskLevel(pred.confidence_pct));
                          const isDim = pred.confidence_pct < 15;
                          return (
                            <div
                              key={pred.label}
                              style={{
                                opacity: isDim ? 0.5 : 1,
                                transition: 'opacity 0.2s',
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                                <span style={{ fontSize: 12, color: isDim ? '#767676' : '#191c1e', fontWeight: isDim ? 400 : 500 }}>
                                  {pred.label}
                                </span>
                                <span style={{ fontSize: 12, fontWeight: 600, color: colors.text }}>
                                  {pred.confidence_pct}%
                                </span>
                              </div>
                              <div style={{ height: 4, borderRadius: 99, background: 'rgba(0,0,0,0.06)' }}>
                                <div
                                  style={{
                                    height: '100%',
                                    width: `${pred.confidence_pct}%`,
                                    borderRadius: 99,
                                    background: colors.text,
                                    transition: 'width 0.5s ease',
                                  }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Cross-Architecture Summary */}
            <div style={{ marginTop: 32 }}>
              <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 20 }}>Cross-Architecture Summary</h2>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
                {[
                  { label: 'Agreement Score',     value: agreementScore, unit: '%', icon: 'handshake' },
                  { label: 'Localization Delta',  value: results.length > 1 ? Math.abs(topConfs[0] - topConfs[topConfs.length - 1]).toFixed(2) : '—', unit: '%', icon: 'my_location' },
                  { label: 'Ensemble Confidence', value: ensembleConf, unit: '%', icon: 'hub' },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    style={{
                      background: 'var(--cc-card-bg)',
                      border: '1px solid var(--cc-border)',
                      borderRadius: 'var(--cc-radius-card)',
                      padding: '20px 24px',
                      boxShadow: 'var(--cc-shadow-card)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                      <span className="material-symbols-outlined" style={{ fontSize: 20, color: 'var(--cc-brand)' }}>
                        {stat.icon}
                      </span>
                      <span style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>
                        {stat.label}
                      </span>
                    </div>
                    <p style={{ fontSize: 32, fontWeight: 700, color: 'var(--cc-brand)' }}>
                      {stat.value}
                      <span style={{ fontSize: 14, fontWeight: 400, color: '#767676', marginLeft: 4 }}>{stat.unit}</span>
                    </p>
                  </div>
                ))}
              </div>

              {/* Narrative blocks */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                {[
                  {
                    title: 'Model Consensus',
                    body: `All ${results.length} architectures were evaluated on the same input. Agreement score reflects how consistently each model ranks the top pathology. Higher scores indicate stronger consensus across architectures.`,
                  },
                  {
                    title: 'Localization Quality',
                    body: `GradCAM heatmaps highlight the regions each model attends to. Localization delta measures the divergence between the highest and lowest confidence models — smaller delta suggests better spatial agreement.`,
                  },
                  {
                    title: 'Ensemble Recommendation',
                    body: `The ensemble confidence (mean of top predictions) provides a risk-adjusted estimate. When ensemble confidence exceeds 70%, escalation to specialist review is recommended per CRITICAL threshold guidelines.`,
                  },
                ].map((block) => (
                  <div
                    key={block.title}
                    style={{
                      background: 'var(--cc-card-bg)',
                      border: '1px solid var(--cc-border)',
                      borderRadius: 'var(--cc-radius-card)',
                      padding: '18px 20px',
                      boxShadow: 'var(--cc-shadow-card)',
                    }}
                  >
                    <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>{block.title}</h3>
                    <p style={{ fontSize: 13, color: '#506071', lineHeight: 1.65 }}>{block.body}</p>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* If came from single result but no file, show link back */}
        {singleResult && !file && !batchResult && (
          <div style={{ marginTop: 24, textAlign: 'center' }}>
            <button
              onClick={() => navigate('/results', { state: { result: singleResult } })}
              style={{
                padding: '10px 24px',
                background: 'transparent',
                color: 'var(--cc-brand)',
                border: '1px solid var(--cc-brand)',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Back to Results
            </button>
          </div>
        )}
      </div>

      <style>{`
        @media (max-width: 1024px) {
          [data-compare-grid] { overflow-x: auto; }
        }
        @media (max-width: 640px) {
          [data-compare-grid] { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </Layout>
  );
}
