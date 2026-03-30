import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '../components/layout/Layout';
import { api } from '../api/client';

/* ─────────────────────────────────────────────────────────────────
   Footer modals: PRIVACY · ETHICS · CONTACT
────────────────────────────────────────────────────────────────── */
function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}
      onClick={onClose}
    >
      <div
        style={{ background: 'var(--cc-card-bg)', borderRadius: 16, padding: '36px 40px', maxWidth: 560, width: '100%', maxHeight: '80vh', overflowY: 'auto', boxShadow: '0 20px 60px rgba(0,0,0,0.22)', position: 'relative' }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{ position: 'absolute', top: 16, right: 20, background: 'none', border: 'none', fontSize: 22, cursor: 'pointer', color: '#767676', lineHeight: 1 }}
        >×</button>
        <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 20, color: '#191c1e' }}>{title}</h2>
        {children}
      </div>
    </div>
  );
}

function PrivacyModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Privacy Policy" onClose={onClose}>
      <div style={{ fontSize: 14, color: '#374151', lineHeight: 1.75, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <p><strong>Data Collection.</strong> Clinical Curator does not collect, store, or transmit any patient data or uploaded images to external servers. All image processing occurs locally on this machine.</p>
        <p><strong>No Persistent Storage.</strong> Uploaded X-ray images are held in memory only for the duration of a single inference request. They are not written to disk and are not retained after the response is returned.</p>
        <p><strong>Inference Logs.</strong> Server-side logs may record HTTP request metadata (timestamp, response code, inference duration) for debugging purposes. No image data or prediction results are included in logs.</p>
        <p><strong>Cookies &amp; Tracking.</strong> This application uses no cookies, analytics, or third-party tracking scripts.</p>
        <p><strong>Research Use Only.</strong> This tool is intended for educational use within the 24AI636 Deep Learning course. It is not subject to HIPAA or GDPR clinical-data regulations, but users are asked not to upload identifiable patient data.</p>
        <p style={{ fontSize: 12, color: '#767676', marginTop: 8 }}>Last updated: March 2026 · Clinical Curator v2.4.0</p>
      </div>
    </Modal>
  );
}

function EthicsModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="AI Ethics Statement" onClose={onClose}>
      <div style={{ fontSize: 14, color: '#374151', lineHeight: 1.75, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ padding: '12px 16px', background: 'var(--cc-caution-bg)', borderLeft: '4px solid var(--cc-caution-fg)', borderRadius: 6 }}>
          <p style={{ fontWeight: 600, color: 'var(--cc-caution-fg)', marginBottom: 4 }}>Not a Clinical Tool</p>
          <p style={{ fontSize: 13, color: '#7c4a00' }}>Clinical Curator is a research and educational prototype. Outputs must never be used for clinical decision-making without qualified radiologist review.</p>
        </div>
        <p><strong>Transparency.</strong> All model architectures, training datasets, and AUC scores are disclosed in the interface. No black-box claims are made. GradCAM visualisations expose model attention to support human oversight.</p>
        <p><strong>Fairness.</strong> The NIH ChestX-ray14 dataset is known to have demographic imbalances. Model performance may vary across patient subgroups. Users should apply clinical judgment accordingly.</p>
        <p><strong>Human Oversight.</strong> Every prediction is clearly labelled as AI-generated. The "Request Specialist Review" feature is provided to facilitate — not replace — expert oversight.</p>
        <p><strong>Accountability.</strong> This system is developed as part of the 24AI636 Deep Learning course (R1–R4). Limitations are documented in the README. Feedback and error reports are encouraged via the Contact page.</p>
        <p><strong>Data Minimisation.</strong> The system requests only the minimum data needed for inference. No demographic metadata is required for a prediction.</p>
        <p style={{ fontSize: 12, color: '#767676', marginTop: 8 }}>24AI636 Deep Learning · AI Research Group · 2024</p>
      </div>
    </Modal>
  );
}

function ContactModal({ onClose }: { onClose: () => void }) {
  const [sent, setSent]     = useState(false);
  const [subject, setSubject] = useState('');
  const [message, setMessage] = useState('');

  return (
    <Modal title="Contact" onClose={onClose}>
      {sent ? (
        <div style={{ textAlign: 'center', padding: '24px 0' }}>
          <span className="material-symbols-outlined" style={{ fontSize: 48, color: 'var(--cc-normal-fg)', display: 'block', marginBottom: 16 }}>mark_email_read</span>
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Message Sent</h3>
          <p style={{ fontSize: 14, color: '#767676' }}>Thank you for your feedback. The research team will respond within 2–3 business days.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <p style={{ fontSize: 14, color: '#506071' }}>
            Questions about the Clinical Curator system, model performance, or the 24AI636 project? Reach out below.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div style={{ padding: '14px 16px', background: 'var(--cc-brand-light)', borderRadius: 10, textAlign: 'center' }}>
              <span className="material-symbols-outlined" style={{ fontSize: 22, color: 'var(--cc-brand)', display: 'block', marginBottom: 4 }}>school</span>
              <p style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--cc-brand)' }}>Course</p>
              <p style={{ fontSize: 13, color: '#191c1e', marginTop: 2 }}>24AI636 Deep Learning</p>
            </div>
            <div style={{ padding: '14px 16px', background: 'var(--cc-brand-light)', borderRadius: 10, textAlign: 'center' }}>
              <span className="material-symbols-outlined" style={{ fontSize: 22, color: 'var(--cc-brand)', display: 'block', marginBottom: 4 }}>hub</span>
              <p style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--cc-brand)' }}>Reviews</p>
              <p style={{ fontSize: 13, color: '#191c1e', marginTop: 2 }}>R1 · R2 · R3 · R4</p>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>Subject</label>
            <select
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid var(--cc-border)', fontSize: 14, background: 'var(--cc-page-bg)' }}
            >
              <option value="">Select a topic…</option>
              <option>Model performance question</option>
              <option>Bug report</option>
              <option>Feature request</option>
              <option>Dataset / training question</option>
              <option>Other</option>
            </select>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#767676' }}>Message</label>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={4}
              placeholder="Describe your question or feedback…"
              style={{ padding: '10px 12px', borderRadius: 8, border: '1px solid var(--cc-border)', fontSize: 14, background: 'var(--cc-page-bg)', resize: 'vertical' }}
            />
          </div>
          <button
            onClick={() => { if (subject && message) setSent(true); }}
            disabled={!subject || !message}
            style={{ padding: '11px', background: (!subject || !message) ? '#c0c8d0' : 'var(--cc-brand)', color: '#fff', border: 'none', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: (!subject || !message) ? 'not-allowed' : 'pointer' }}
          >
            Send Message
          </button>
        </div>
      )}
    </Modal>
  );
}

/* ─────────────────────────────────────────────────────────────────
   Real NIH sample images served from /public/samples/
────────────────────────────────────────────────────────────────── */
const SAMPLES = [
  { id: 'sample1', url: '/samples/sample1.png', label: 'NIH Case 00016051', tag: 'PA View' },
  { id: 'sample2', url: '/samples/sample2.png', label: 'NIH Case 00000003', tag: 'PA View' },
];

async function fetchSampleAsFile(url: string, name: string): Promise<File> {
  const res  = await fetch(url);
  const blob = await res.blob();
  return new File([blob], name, { type: 'image/png' });
}

/* ─────────────────────────────────────────────────────────────────
   Landing page
────────────────────────────────────────────────────────────────── */
const FEATURES = [
  { icon: 'bolt',          title: 'Sub-second Inference',  desc: 'CNN-optimised pipeline delivers results in under 200ms on GPU.' },
  { icon: 'layers',        title: 'Deep Feature Maps',     desc: 'GradCAM, saliency maps, and layer attention for full explainability.' },
  { icon: 'verified_user', title: 'Validated Accuracy',    desc: 'Trained on NIH ChestX-ray14 — 112K images across 14 disease classes.' },
];

const TABS = [
  { id: 'quick',    label: 'Quick (CNN)'           },
  { id: 'research', label: 'Research (All Models)'  },
  { id: 'temporal', label: 'Temporal (History)'     },
  { id: 'latent',   label: 'Latent Space'           },
];

const ALL_MODELS = ['mlp_r1', 'cnn_r1', 'densenet121_r4', 'cnn_lstm_r2', 'ae_r3'];

interface Props { darkMode: boolean; onToggleDark: () => void; }

export default function Landing({ darkMode, onToggleDark }: Props) {
  const navigate = useNavigate();

  const [modal, setModal]               = useState<'privacy' | 'ethics' | 'contact' | null>(null);
  const [sampling, setSampling]         = useState(false);
  const [sampleError, setSampleError]   = useState('');
  const [showSamplePicker, setShowSamplePicker] = useState(false);

  const runSample = async (sample: typeof SAMPLES[0]) => {
    setShowSamplePicker(false);
    setSampling(true);
    setSampleError('');
    try {
      const file    = await fetchSampleAsFile(sample.url, `${sample.id}.png`);
      const results = await api.predictBatch(file, ALL_MODELS);
      navigate('/compare', { state: { batchResult: results, file } });
    } catch (e) {
      setSampleError(`Demo failed: ${(e as Error).message}. Is the backend running?`);
    } finally {
      setSampling(false);
    }
  };

  return (
    <Layout darkMode={darkMode} onToggleDark={onToggleDark} hideSidebar>
      {/* ── Footer modals ── */}
      {modal === 'privacy' && <PrivacyModal  onClose={() => setModal(null)} />}
      {modal === 'ethics'  && <EthicsModal   onClose={() => setModal(null)} />}
      {modal === 'contact' && <ContactModal  onClose={() => setModal(null)} />}

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <section
        style={{
          maxWidth: 1280, margin: '0 auto', padding: '60px 32px 40px',
          display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, alignItems: 'center',
        }}
      >
        {/* Left */}
        <div>
          <span style={{ display: 'inline-block', marginBottom: 20, fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--cc-brand)', background: 'var(--cc-brand-light)', padding: '4px 12px', borderRadius: 20 }}>
            V2.4.0 RESEARCH EDITION
          </span>

          <h1 style={{ fontSize: 44, fontWeight: 700, lineHeight: 1.15, letterSpacing: '-0.02em', color: '#0d1b2a', marginBottom: 20 }}>
            Next-Gen{' '}<span style={{ color: 'var(--cc-brand)' }}>Diagnostic</span>{' '}Precision
          </h1>

          <p style={{ fontSize: 16, color: '#506071', lineHeight: 1.7, maxWidth: 440, marginBottom: 32 }}>
            Clinical Curator integrates deep-learning models trained on NIH ChestX-ray14 directly
            into your workflow — providing explainable AI insights across 14 disease categories.
          </p>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
            <button
              onClick={() => navigate('/upload')}
              style={{ padding: '13px 28px', background: 'var(--cc-brand)', color: '#fff', borderRadius: 8, border: 'none', fontSize: 15, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>upload_file</span>
              Upload X-Ray
            </button>

            {/* View Samples button + picker */}
            <div style={{ position: 'relative' }}>
              <button
                onClick={() => !sampling && setShowSamplePicker((v) => !v)}
                disabled={sampling}
                style={{ padding: '13px 28px', background: 'var(--cc-brand-light)', color: 'var(--cc-brand)', borderRadius: 8, border: '1px solid var(--cc-brand)', fontSize: 15, fontWeight: 600, cursor: sampling ? 'wait' : 'pointer', display: 'flex', alignItems: 'center', gap: 8, opacity: sampling ? 0.7 : 1 }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 18 }}>
                  {sampling ? 'progress_activity' : 'science'}
                </span>
                {sampling ? 'Running all models…' : 'View Samples'}
                {!sampling && <span className="material-symbols-outlined" style={{ fontSize: 16 }}>expand_more</span>}
              </button>

              {showSamplePicker && (
                <div
                  style={{ position: 'absolute', top: '110%', left: 0, zIndex: 50, background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 12, boxShadow: '0 8px 32px rgba(0,0,0,0.14)', overflow: 'hidden', minWidth: 340 }}
                >
                  <p style={{ padding: '10px 16px 6px', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#767676' }}>
                    Select NIH sample — runs all 5 models
                  </p>
                  {SAMPLES.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => runSample(s)}
                      style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 14, padding: '12px 16px', border: 'none', background: 'transparent', cursor: 'pointer', textAlign: 'left', borderTop: '1px solid var(--cc-border)' }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--cc-brand-light)')}
                      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                    >
                      <img src={s.url} alt={s.label} style={{ width: 52, height: 52, objectFit: 'cover', borderRadius: 6, background: '#0a0a0a', flexShrink: 0 }} />
                      <div>
                        <p style={{ fontSize: 13, fontWeight: 600, color: '#191c1e', marginBottom: 2 }}>{s.label}</p>
                        <p style={{ fontSize: 11, color: '#767676' }}>{s.tag} · 1024×1024 · NIH ChestX-ray14</p>
                        <p style={{ fontSize: 11, color: 'var(--cc-brand)', marginTop: 2 }}>Run MLP · CNN · ResNet · LSTM · AE →</p>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {sampleError && (
            <p style={{ marginTop: 12, fontSize: 13, color: 'var(--cc-critical-fg)', background: 'var(--cc-critical-bg)', padding: '8px 12px', borderRadius: 7 }}>
              {sampleError}
            </p>
          )}

          {/* Instructions */}
          <div style={{ marginTop: 28, padding: '16px 18px', background: 'var(--cc-card-bg)', border: '1px solid var(--cc-border)', borderRadius: 10, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <p style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#767676' }}>How it works</p>
            {[
              { icon: 'upload_file',  step: '1', text: 'Upload a chest X-ray (DICOM, PNG, or JPG)' },
              { icon: 'hub',          step: '2', text: 'Select one or more AI models (MLP, CNN, DenseNet, LSTM, AE)' },
              { icon: 'analytics',    step: '3', text: 'View predictions with GradCAM heatmaps and confidence scores' },
              { icon: 'compare',      step: '4', text: 'Compare all architectures side-by-side in the Compare view' },
            ].map((s) => (
              <div key={s.step} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ width: 26, height: 26, borderRadius: '50%', background: 'var(--cc-brand-light)', color: 'var(--cc-brand)', fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  {s.step}
                </div>
                <span className="material-symbols-outlined" style={{ fontSize: 16, color: 'var(--cc-brand)', flexShrink: 0 }}>{s.icon}</span>
                <p style={{ fontSize: 13, color: '#374151' }}>{s.text}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Right — hero card */}
        <div style={{ paddingRight: 24 }}>
          <div style={{ background: '#0a0a0a', borderRadius: 16, overflow: 'hidden', aspectRatio: '4/3', position: 'relative', boxShadow: '0 16px 40px rgba(0,0,0,0.18)' }}>
            <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse at center, #1a2740 0%, #050810 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span className="material-symbols-outlined" style={{ fontSize: 80, color: 'rgba(255,255,255,0.07)' }}>radiology</span>
            </div>
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: 'linear-gradient(90deg, transparent, var(--cc-brand), transparent)', animation: 'scan 2.5s ease-in-out infinite' }} />
            <div style={{ position: 'absolute', top: 20, right: 20, background: 'rgba(255,255,255,0.08)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: '12px 16px' }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.6)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 4 }}>AI Confidence</p>
              <p style={{ fontSize: 26, fontWeight: 700, color: '#fff' }}>98.4%</p>
            </div>
            <div style={{ position: 'absolute', bottom: 20, left: 20, background: 'rgba(255,255,255,0.08)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: '10px 14px', maxWidth: 260 }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Primary Findings</p>
              <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.8)', lineHeight: 1.5 }}>No significant pleural effusion or pneumothorax identified.</p>
            </div>
            {/* Model indicators */}
            <div style={{ position: 'absolute', bottom: 20, right: 20, display: 'flex', flexDirection: 'column', gap: 4 }}>
              {['MLP V1.0', 'CNN V4.2', 'ResNet V9.0'].map((m) => (
                <span key={m} style={{ fontSize: 9, fontWeight: 700, color: '#fff', background: 'rgba(26,95,168,0.7)', padding: '2px 7px', borderRadius: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{m}</span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Model selector tabs ─────────────────────────────────── */}
      <section style={{ maxWidth: 1280, margin: '0 auto', padding: '0 32px 40px' }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => navigate(`/upload?mode=${tab.id}`)}
              style={{ padding: '8px 20px', border: '1px solid var(--cc-border)', borderRadius: 8, background: 'var(--cc-card-bg)', fontSize: 13, fontWeight: 500, color: '#506071', cursor: 'pointer' }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </section>

      {/* ── Feature cards ──────────────────────────────────────── */}
      <section style={{ maxWidth: 1280, margin: '0 auto', padding: '0 32px 40px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20 }}>
          {FEATURES.map((f) => (
            <div key={f.title} style={{ background: 'var(--cc-card-bg)', borderRadius: 'var(--cc-radius-card)', padding: 24, border: '1px solid var(--cc-border)', boxShadow: 'var(--cc-shadow-card)' }}>
              <span className="material-symbols-outlined" style={{ fontSize: 32, color: 'var(--cc-brand)', marginBottom: 16, display: 'block' }}>{f.icon}</span>
              <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>{f.title}</h3>
              <p style={{ fontSize: 13, color: '#767676', lineHeight: 1.6 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Stats bar ──────────────────────────────────────────── */}
      <section style={{ background: 'var(--cc-card-bg)', borderTop: '1px solid var(--cc-border)', borderBottom: '1px solid var(--cc-border)', margin: '0 0 40px' }}>
        <div style={{ maxWidth: 1280, margin: '0 auto', padding: '32px', display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 24 }}>
          {[
            { num: '14',   label: 'Diseases Detected',   sub: 'Pneumonia, Pleural Effusion & more' },
            { num: '8',    label: 'Models Trained',       sub: 'MLP, CNN, DenseNet, LSTM, AE…'      },
            { num: '112K', label: 'Images Trained On',    sub: 'NIH ChestX-ray14 dataset'            },
          ].map((s) => (
            <div key={s.label} style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 36, fontWeight: 700, color: 'var(--cc-brand)', lineHeight: 1 }}>{s.num}</p>
              <p style={{ fontSize: 14, fontWeight: 500, color: '#191c1e', marginTop: 4 }}>{s.label}</p>
              <p style={{ fontSize: 12, color: '#767676', marginTop: 2 }}>{s.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer style={{ maxWidth: 1280, margin: '0 auto', padding: '24px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <span style={{ fontSize: 13, color: '#767676' }}>Clinical Curator © 2024 AI Research Group</span>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--cc-caution-fg)', background: 'var(--cc-caution-bg)', padding: '4px 14px', borderRadius: 20 }}>
          For research/educational use only. Not a clinical tool.
        </span>
        <div style={{ display: 'flex', gap: 20 }}>
          {[
            { key: 'privacy', label: 'PRIVACY' },
            { key: 'ethics',  label: 'ETHICS'  },
            { key: 'contact', label: 'CONTACT' },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setModal(key as 'privacy' | 'ethics' | 'contact')}
              style={{ background: 'none', border: 'none', padding: 0, fontSize: 11, fontWeight: 600, letterSpacing: '0.05em', color: '#767676', cursor: 'pointer', textDecoration: 'underline', textDecorationColor: 'transparent', transition: 'color 0.15s' }}
              onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--cc-brand)')}
              onMouseLeave={(e) => (e.currentTarget.style.color = '#767676')}
            >
              {label}
            </button>
          ))}
        </div>
      </footer>

      <style>{`
        @keyframes scan {
          0%   { top: 0; }
          50%  { top: calc(100% - 2px); }
          100% { top: 0; }
        }
      `}</style>
    </Layout>
  );
}
