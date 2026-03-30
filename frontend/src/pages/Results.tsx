import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Layout from '../components/layout/Layout';
import XrayViewer from '../components/ui/XrayViewer';
import ConfidenceBadge from '../components/ui/ConfidenceBadge';
import PredictionBar from '../components/ui/PredictionBar';
import { api } from '../api/client';
import { formatDate, getRiskColors, getRiskLevel } from '../utils/format';
import type { PredictResponse } from '../types/api';

interface Props { darkMode: boolean; onToggleDark: () => void; }

// ── Specialist modal ────────────────────────────────────────────────────────
function SpecialistModal({ studyId, onClose }: { studyId: string; onClose: () => void }) {
  const [sent, setSent] = useState(false);
  const [email, setEmail] = useState('');
  const submit = () => { if (email) setSent(true); };

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
        zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--cc-card-bg)', borderRadius: 16, padding: 32, width: 420,
          boxShadow: '0 20px 60px rgba(0,0,0,0.2)', position: 'relative',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{ position: 'absolute', top: 16, right: 16, background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: '#767676' }}
        >
          ×
        </button>
        {sent ? (
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <span className="material-symbols-outlined" style={{ fontSize: 48, color: 'var(--cc-normal-fg)', display: 'block', marginBottom: 16 }}>
              check_circle
            </span>
            <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Review Requested</h3>
            <p style={{ fontSize: 14, color: '#767676' }}>
              Study <strong>{studyId}</strong> has been queued for specialist review. You will be notified when feedback is available.
            </p>
          </div>
        ) : (
          <>
            <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>Request Specialist Review</h3>
            <p style={{ fontSize: 13, color: '#767676', marginBottom: 20 }}>
              Send study <strong>{studyId}</strong> to a radiologist for secondary review.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <label style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676', display: 'block', marginBottom: 6 }}>
                  Radiologist Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="radiologist@hospital.org"
                  style={{
                    width: '100%', padding: '10px 14px', borderRadius: 8,
                    border: '1px solid var(--cc-border)', fontSize: 14,
                    background: 'var(--cc-page-bg)', outline: 'none', boxSizing: 'border-box',
                  }}
                />
              </div>
              <div>
                <label style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676', display: 'block', marginBottom: 6 }}>
                  Priority
                </label>
                <select
                  style={{
                    width: '100%', padding: '10px 14px', borderRadius: 8,
                    border: '1px solid var(--cc-border)', fontSize: 14,
                    background: 'var(--cc-page-bg)', outline: 'none',
                  }}
                >
                  <option>Routine (24–48 hours)</option>
                  <option>Urgent (4–8 hours)</option>
                  <option>STAT (1 hour)</option>
                </select>
              </div>
              <textarea
                placeholder="Optional clinical context for the specialist..."
                rows={3}
                style={{
                  width: '100%', padding: '10px 14px', borderRadius: 8,
                  border: '1px solid var(--cc-border)', fontSize: 13,
                  background: 'var(--cc-page-bg)', resize: 'vertical', outline: 'none', boxSizing: 'border-box',
                }}
              />
              <button
                onClick={submit}
                style={{
                  padding: '11px', background: 'var(--cc-brand)', color: '#fff',
                  border: 'none', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer',
                }}
              >
                Send Request
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Share modal ─────────────────────────────────────────────────────────────
function ShareModal({ studyId, onClose }: { studyId: string; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const link = `${window.location.origin}/results?study=${studyId}`;
  const copy = () => { navigator.clipboard.writeText(link); setCopied(true); setTimeout(() => setCopied(false), 2000); };

  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      onClick={onClose}
    >
      <div
        style={{ background: 'var(--cc-card-bg)', borderRadius: 16, padding: 32, width: 420, boxShadow: '0 20px 60px rgba(0,0,0,0.2)', position: 'relative' }}
        onClick={(e) => e.stopPropagation()}
      >
        <button onClick={onClose} style={{ position: 'absolute', top: 16, right: 16, background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: '#767676' }}>×</button>
        <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>Share Report</h3>
        <p style={{ fontSize: 13, color: '#767676', marginBottom: 20 }}>Share this diagnostic report with colleagues.</p>

        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <input readOnly value={link} style={{ flex: 1, padding: '9px 12px', borderRadius: 8, border: '1px solid var(--cc-border)', fontSize: 13, background: 'var(--cc-page-bg)' }} />
          <button
            onClick={copy}
            style={{ padding: '9px 16px', background: copied ? 'var(--cc-normal-fg)' : 'var(--cc-brand)', color: '#fff', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap' }}
          >
            {copied ? 'Copied!' : 'Copy Link'}
          </button>
        </div>

        {/* Native share if available */}
        {typeof navigator.share === 'function' && (
          <button
            onClick={() => navigator.share({ title: `Clinical Curator — ${studyId}`, url: link })}
            style={{ width: '100%', padding: '10px', background: 'var(--cc-page-bg)', border: '1px solid var(--cc-border)', borderRadius: 8, fontSize: 14, fontWeight: 500, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>share</span>
            Share via System
          </button>
        )}
      </div>
    </div>
  );
}

// ── Main Results page ───────────────────────────────────────────────────────
export default function Results({ darkMode, onToggleDark }: Props) {
  const { state }     = useLocation();
  const navigate      = useNavigate();
  const result        = state?.result as PredictResponse | undefined;
  const uploadedFile  = state?.file as File | undefined;

  const [opacity, setOpacity]           = useState(65);
  const [heatmap, setHeatmap]           = useState(true);
  const [showAll, setShowAll]           = useState(false);
  const [downloading, setDownloading]   = useState(false);
  const [showSpecialist, setSpecialist] = useState(false);
  const [showShare, setShare]           = useState(false);
  const [copiedJson, setCopiedJson]     = useState(false);
  const [selectedModel, setSelectedModel] = useState(result?.model_used ?? '');

  if (!result) {
    return (
      <Layout darkMode={darkMode} onToggleDark={onToggleDark}>
        <div style={{ padding: 60, textAlign: 'center' }}>
          <span className="material-symbols-outlined" style={{ fontSize: 56, color: '#c0c8d0', display: 'block', marginBottom: 16 }}>analytics</span>
          <p style={{ fontSize: 16, color: '#767676', marginBottom: 20 }}>No analysis results found.</p>
          <button
            onClick={() => navigate('/upload')}
            style={{ padding: '11px 28px', background: 'var(--cc-brand)', color: '#fff', border: 'none', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer' }}
          >
            Go to Upload
          </button>
        </div>
      </Layout>
    );
  }

  const predictions   = result.predictions;
  const displayed     = showAll ? predictions : predictions.slice(0, 4);
  const topPrediction = predictions[0];
  const topColors     = getRiskColors(getRiskLevel(topPrediction?.confidence_pct ?? 0));

  // Use uploaded file as data URL if available, otherwise placeholder
  const imageUrl = uploadedFile
    ? URL.createObjectURL(uploadedFile)
    : 'data:image/svg+xml,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="500"><rect width="600" height="500" fill="#050810"/><text x="300" y="260" text-anchor="middle" fill="rgba(255,255,255,0.18)" font-size="16" font-family="sans-serif">Chest X-Ray</text><text x="300" y="285" text-anchor="middle" fill="rgba(255,255,255,0.1)" font-size="12" font-family="sans-serif">Upload an image to see it here</text></svg>'
      );

  const altText = topPrediction
    ? `Chest X-ray — top finding: ${topPrediction.label} at ${topPrediction.confidence_pct}% confidence`
    : 'Chest X-ray image';

  const handleDownloadPdf = async () => {
    setDownloading(true);
    try {
      await api.downloadPdf(result, result.study_id);
    } catch (e) {
      alert(`PDF generation failed: ${(e as Error).message}`);
    } finally {
      setDownloading(false);
    }
  };

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  const MODELS = ['mlp_r1', 'cnn_r1', 'densenet121_r4', 'cnn_lstm_r2', 'ae_r3'];

  return (
    <Layout darkMode={darkMode} onToggleDark={onToggleDark}>
      {showSpecialist && <SpecialistModal studyId={result.study_id} onClose={() => setSpecialist(false)} />}
      {showShare      && <ShareModal     studyId={result.study_id} onClose={() => setShare(false)} />}

      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '28px 24px 80px' }}>
        {/* ── Header ── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <h1 style={{ fontSize: 22, fontWeight: 600 }}>Study ID: {result.study_id}</h1>
              <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', background: 'rgba(74,222,128,0.15)', color: 'var(--cc-normal-fg)', padding: '3px 8px', borderRadius: 4 }}>
                LIVE
              </span>
            </div>
            <p style={{ fontSize: 13, color: '#767676' }}>
              Chest PA · Performed: {formatDate(result.timestamp)} · Model: {result.model_used} {result.model_version} · {result.inference_ms}ms
            </p>
          </div>
          <button
            onClick={() => navigate('/compare', { state: { result, file: uploadedFile } })}
            style={{ padding: '8px 18px', background: 'var(--cc-brand-light)', color: 'var(--cc-brand)', border: '1px solid var(--cc-brand)', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer' }}
          >
            Compare All Models
          </button>
        </div>

        {/* ── Two-column: 60% image | 40% predictions ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 24, alignItems: 'start' }}>

          {/* LEFT — viewer + controls + insight */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 'var(--cc-radius-card)', overflow: 'hidden', boxShadow: 'var(--cc-shadow-card)' }}>
              <XrayViewer
                imageUrl={imageUrl}
                gradCamBase64={result.grad_cam_base64}
                opacity={opacity}
                heatmapEnabled={heatmap}
                pathologicalFocus={topPrediction?.label}
                altText={altText}
                height={400}
              />

              {/* Controls row */}
              <div style={{ padding: '14px 18px', borderTop: '1px solid var(--cc-border)', display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'center' }}>
                {/* Opacity */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5, minWidth: 160 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <label style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>
                      OVERLAY OPACITY
                    </label>
                    <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--cc-brand)' }}>{opacity}%</span>
                  </div>
                  <input
                    type="range" min={0} max={100} value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))}
                    style={{ accentColor: 'var(--cc-brand)', width: '100%', cursor: 'pointer' }}
                  />
                </div>

                {/* Heatmap toggle */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>Heatmap</label>
                  <button
                    aria-label="Toggle GradCAM heatmap overlay"
                    onClick={() => setHeatmap((h) => !h)}
                    style={{
                      padding: '5px 14px', borderRadius: 6,
                      border: `1px solid ${heatmap ? 'var(--cc-brand)' : 'var(--cc-border)'}`,
                      background: heatmap ? 'var(--cc-brand)' : 'transparent',
                      color: heatmap ? '#fff' : '#506071',
                      fontSize: 12, fontWeight: 600, cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 5,
                    }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 15 }}>
                      {heatmap ? 'visibility' : 'visibility_off'}
                    </span>
                    {heatmap ? 'ON' : 'OFF'}
                  </button>
                </div>

                {/* Model selector */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>INFERENCE MODEL</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    style={{ padding: '5px 10px', borderRadius: 6, border: '1px solid var(--cc-border)', background: 'var(--cc-card-bg)', fontSize: 12, fontWeight: 600, color: '#191c1e', cursor: 'pointer' }}
                  >
                    {MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              </div>
            </div>

            {/* Clinical Insight */}
            <div style={{
              background: 'var(--cc-brand-light)',
              border: '1px solid rgba(26,95,168,0.2)',
              borderLeft: '4px solid var(--cc-brand)',
              borderRadius: 'var(--cc-radius-card)',
              padding: '16px 20px',
              display: 'flex', gap: 12,
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: 22, color: 'var(--cc-brand)', flexShrink: 0, marginTop: 1 }}>lightbulb</span>
              <div>
                <p style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--cc-brand)', marginBottom: 6 }}>
                  Clinical Insight
                </p>
                <p style={{ fontSize: 14, color: '#1e3a5f', lineHeight: 1.7 }}>
                  {result.clinical_insight}
                </p>
              </div>
            </div>

            {/* Inference summary row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
              {[
                { icon: 'speed', label: 'Inference Time', value: `${result.inference_ms}ms` },
                { icon: 'hub',   label: 'Model Version',  value: result.model_version },
                { icon: 'check_circle', label: 'Study ID', value: result.study_id },
              ].map((s) => (
                <div key={s.label} style={{ background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 10, padding: '12px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 15, color: 'var(--cc-brand)' }}>{s.icon}</span>
                    <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>{s.label}</span>
                  </div>
                  <p style={{ fontSize: 14, fontWeight: 700, color: '#191c1e' }}>{s.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* RIGHT — predictions panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Top finding highlight */}
            {topPrediction && (
              <div style={{
                background: topColors.bg, border: `1.5px solid ${topColors.text}`,
                borderRadius: 'var(--cc-radius-card)', padding: '14px 18px',
              }}>
                <p style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: topColors.text, marginBottom: 3 }}>
                  TOP FINDING
                </p>
                <p style={{ fontSize: 20, fontWeight: 700, color: topColors.text }}>{topPrediction.label}</p>
                <p style={{ fontSize: 12, color: topColors.text, marginTop: 3, opacity: 0.85 }}>
                  {topPrediction.confidence_pct}% confidence · {topPrediction.risk_level}
                </p>
              </div>
            )}

            {/* Disease list card */}
            <div style={{ background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 'var(--cc-radius-card)', padding: 18, boxShadow: 'var(--cc-shadow-card)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <h2 style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#767676' }}>
                  TOP PREDICTIONS
                </h2>
                <span className="material-symbols-outlined" style={{ fontSize: 18, color: '#767676', cursor: 'pointer' }}>filter_list</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {displayed.map((pred) => {
                  const colors = getRiskColors(getRiskLevel(pred.confidence_pct));
                  return (
                    <div
                      key={pred.label}
                      style={{ padding: '10px 12px', background: colors.bg, borderRadius: 8, borderLeft: `3px solid ${colors.text}` }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: '#191c1e' }}>{pred.label}</span>
                        <ConfidenceBadge confidence={pred.confidence_pct} raw={false} />
                      </div>
                      {/* hideValue=true — badge above already shows the % */}
                      <PredictionBar label="" confidencePct={pred.confidence_pct} hideValue />
                    </div>
                  );
                })}
              </div>

              {/* Show all 14 toggle */}
              {predictions.length > 4 && (
                <button
                  onClick={() => setShowAll((s) => !s)}
                  style={{ marginTop: 12, width: '100%', padding: '8px', border: '1px dashed var(--cc-border)', borderRadius: 8, background: 'transparent', fontSize: 13, fontWeight: 500, color: 'var(--cc-brand)', cursor: 'pointer' }}
                >
                  {showAll ? 'Show fewer ↑' : `Show all ${predictions.length} diseases ↓`}
                </button>
              )}

              {/* Request Specialist */}
              <button
                onClick={() => setSpecialist(true)}
                style={{ marginTop: 10, width: '100%', padding: '10px', border: '1.5px dashed var(--cc-border)', borderRadius: 8, background: 'transparent', fontSize: 13, fontWeight: 500, color: '#506071', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7 }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>person_add</span>
                Request Specialist Review
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Sticky bottom action bar ── */}
      <div style={{
        position: 'fixed', bottom: 0, left: 'var(--cc-sidebar-w)', right: 0,
        height: 60, background: 'var(--cc-card-bg)', borderTop: '1px solid var(--cc-border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
        zIndex: 50, boxShadow: '0 -2px 8px rgba(0,0,0,0.06)',
      }}>
        <button
          onClick={handleDownloadPdf}
          disabled={downloading}
          style={{ padding: '8px 20px', background: 'var(--cc-brand)', color: '#fff', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: downloading ? 'wait' : 'pointer', display: 'flex', alignItems: 'center', gap: 7, opacity: downloading ? 0.7 : 1 }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>picture_as_pdf</span>
          {downloading ? 'Generating…' : 'Download PDF'}
        </button>

        <button
          onClick={handleCopyJson}
          style={{ padding: '8px 20px', background: 'transparent', color: copiedJson ? 'var(--cc-normal-fg)' : 'var(--cc-brand)', border: `1px solid ${copiedJson ? 'var(--cc-normal-fg)' : 'var(--cc-brand)'}`, borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 7 }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>{copiedJson ? 'check' : 'code'}</span>
          {copiedJson ? 'Copied!' : 'Copy JSON'}
        </button>

        <button
          onClick={() => setShare(true)}
          style={{ padding: '8px 20px', background: 'transparent', color: '#506071', border: '1px solid var(--cc-border)', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 7 }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>share</span>
          Share Report
        </button>
      </div>
    </Layout>
  );
}
