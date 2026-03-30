import { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Layout from '../components/layout/Layout';
import UploadZone from '../components/ui/UploadZone';
import ModelCard from '../components/ui/ModelCard';
import LoadingSteps from '../components/ui/LoadingSteps';
import { useModels } from '../hooks/useModels';
import { usePrediction } from '../hooks/usePrediction';
import type { ImageMeta } from '../utils/imageProcessing';

interface Props { darkMode: boolean; onToggleDark: () => void; }

export default function Upload({ darkMode, onToggleDark }: Props) {
  const navigate     = useNavigate();
  const [params]     = useSearchParams();
  const { models }   = useModels();
  const { status, result, predictBatch, predict, error } = usePrediction();

  const [selectedModels, setSelectedModels] = useState<string[]>(['cnn_r1']);
  const [file, setFile]  = useState<File | null>(null);
  const [meta, setMeta]  = useState<ImageMeta | null>(null);
  const isDemo = params.get('demo') === '1';

  const toggleModel = (id: string) => {
    setSelectedModels((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  };

  const selectAll  = () => setSelectedModels(models.map((m) => m.id));
  const clearAll   = () => setSelectedModels([]);

  const handleSubmit = async () => {
    if (!file || selectedModels.length === 0) return;
    if (selectedModels.length === 1) {
      const res = await predict(file, selectedModels[0]);
      if (res) navigate('/results', { state: { result: res, file } });
    } else {
      const res = await predictBatch(file, selectedModels);
      if (res) navigate('/compare', { state: { batchResult: res, file } });
    }
  };

  const allSelected = models.length > 0 && selectedModels.length === models.length;

  return (
    <Layout darkMode={darkMode} onToggleDark={onToggleDark}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 24, fontWeight: 600 }}>Upload &amp; Analyze</h1>
          <p style={{ fontSize: 14, color: '#767676', marginTop: 4 }}>
            Upload a chest X-ray and select one or more AI models to run inference.
          </p>
          {isDemo && (
            <div style={{ marginTop: 10, padding: '10px 16px', background: 'var(--cc-brand-light)', borderRadius: 8, fontSize: 13, color: 'var(--cc-brand)', fontWeight: 500 }}>
              Demo mode — upload any chest X-ray PNG/JPG to test the pipeline.
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28 }}>
          {/* ── Left column ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Upload zone */}
            <div style={{ background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 'var(--cc-radius-card)', padding: 24, boxShadow: 'var(--cc-shadow-card)' }}>
              <h2 style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676', marginBottom: 14 }}>
                Upload Image
              </h2>
              <UploadZone onFile={(f, m) => { setFile(f); setMeta(m); }} />
            </div>

            {/* Model selector */}
            <div style={{ background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 'var(--cc-radius-card)', padding: 24, boxShadow: 'var(--cc-shadow-card)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <h2 style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>
                  Select Model{selectedModels.length > 1 ? 's' : ''} ({selectedModels.length} selected)
                </h2>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    onClick={allSelected ? clearAll : selectAll}
                    style={{ padding: '4px 12px', border: '1px solid var(--cc-border)', borderRadius: 6, background: 'transparent', fontSize: 12, fontWeight: 600, color: 'var(--cc-brand)', cursor: 'pointer' }}
                  >
                    {allSelected ? 'Clear all' : 'Select all'}
                  </button>
                </div>
              </div>

              {selectedModels.length > 1 && (
                <div style={{ padding: '8px 12px', background: 'var(--cc-brand-light)', borderRadius: 7, fontSize: 12, color: 'var(--cc-brand)', fontWeight: 500, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 7 }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 15 }}>compare</span>
                  {selectedModels.length} models selected — results will open in Compare view
                </div>
              )}

              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {models.map((m) => (
                  <ModelCard
                    key={m.id}
                    model={m}
                    selected={selectedModels.includes(m.id)}
                    onSelect={toggleModel}
                  />
                ))}
                {models.length === 0 && (
                  <p style={{ fontSize: 13, color: '#767676' }}>Loading models…</p>
                )}
              </div>
            </div>
          </div>

          {/* ── Right column — preview + actions ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* X-ray preview */}
            <div style={{ background: '#050810', borderRadius: 'var(--cc-radius-card)', overflow: 'hidden', minHeight: 320, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              {meta ? (
                <img
                  src={meta.dataUrl}
                  alt="Uploaded X-ray preview"
                  style={{ width: '100%', maxHeight: 400, objectFit: 'contain', display: 'block' }}
                />
              ) : (
                <span className="material-symbols-outlined" style={{ fontSize: 72, color: 'rgba(255,255,255,0.05)' }}>
                  radiology
                </span>
              )}
              <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)', borderRadius: 6, padding: '3px 10px', display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: meta ? '#4ade80' : '#94a3b8' }} />
                <span style={{ fontSize: 10, fontWeight: 700, color: '#fff', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {meta ? 'READY FOR ANALYSIS' : 'AWAITING UPLOAD'}
                </span>
              </div>
              {meta && (
                <div style={{ position: 'absolute', bottom: 12, left: 12, right: 12, display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)' }}>{meta.name}</span>
                  <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)' }}>{meta.width}×{meta.height}px</span>
                </div>
              )}
            </div>

            {/* Loading steps */}
            {status === 'loading' && (
              <div style={{ background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 'var(--cc-radius-card)', padding: 24 }}>
                <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>Running inference…</h3>
                <LoadingSteps />
              </div>
            )}

            {error && (
              <div style={{ padding: '12px 16px', background: 'var(--cc-critical-bg)', borderRadius: 8, color: 'var(--cc-critical-fg)', fontSize: 13, fontWeight: 500 }}>
                {error}
              </div>
            )}

            {/* Submit button */}
            <button
              onClick={handleSubmit}
              disabled={!file || selectedModels.length === 0 || status === 'loading'}
              style={{
                padding: '14px 24px',
                background: (!file || selectedModels.length === 0 || status === 'loading') ? '#c0c8d0' : 'var(--cc-brand)',
                color: '#fff', border: 'none', borderRadius: 8, fontSize: 15, fontWeight: 600,
                cursor: (!file || selectedModels.length === 0 || status === 'loading') ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
                {selectedModels.length > 1 ? 'compare' : 'analytics'}
              </span>
              {status === 'loading'
                ? 'Analyzing…'
                : selectedModels.length > 1
                  ? `Run ${selectedModels.length} Models`
                  : 'Run Analysis'}
            </button>

            {/* Quick model info */}
            {selectedModels.length === 1 && models.length > 0 && (() => {
              const m = models.find((x) => x.id === selectedModels[0]);
              if (!m) return null;
              return (
                <div style={{ background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 10, padding: '12px 16px', display: 'flex', gap: 20 }}>
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>AUC Score</p>
                    <p style={{ fontSize: 18, fontWeight: 700, color: 'var(--cc-brand)' }}>{m.auc_score.toFixed(2)}</p>
                  </div>
                  <div>
                    <p style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>Inference</p>
                    <p style={{ fontSize: 18, fontWeight: 700, color: '#191c1e' }}>{m.inference_ms}ms</p>
                  </div>
                  <div style={{ flex: 1 }}>
                    <p style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>Architecture</p>
                    <p style={{ fontSize: 13, fontWeight: 500, color: '#191c1e', marginTop: 2 }}>{m.description}</p>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </Layout>
  );
}
