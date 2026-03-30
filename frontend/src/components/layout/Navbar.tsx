import { Link, useLocation } from 'react-router-dom';

interface Props {
  darkMode: boolean;
  onToggleDark: () => void;
}

const NAV_LINKS = [
  { to: '/',        label: 'Home'    },
  { to: '/upload',  label: 'Analyze' },
  { to: '/compare', label: 'Compare' },
];

export default function Navbar({ darkMode, onToggleDark }: Props) {
  const { pathname } = useLocation();

  return (
    <header
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: 'var(--cc-nav-height)',
        background: 'var(--cc-card-bg)',
        borderBottom: '1px solid var(--cc-border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        zIndex: 100,
        boxShadow: 'var(--cc-shadow-card)',
      }}
    >
      {/* Left: logo + breadcrumb */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              width: 34,
              height: 34,
              background: 'var(--cc-brand)',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <span className="material-symbols-outlined" style={{ color: '#fff', fontSize: 20 }}>
              radiology
            </span>
          </div>
          <span style={{ fontSize: 17, fontWeight: 700, color: 'var(--cc-brand)', letterSpacing: '-0.02em' }}>
            Clinical Curator
          </span>
        </Link>

        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#767676' }}>
          DIAGNOSTICS
        </span>

        <nav style={{ display: 'flex', gap: 4 }}>
          {NAV_LINKS.map(({ to, label }) => {
            const active = pathname === to;
            return (
              <Link
                key={to}
                to={to}
                style={{
                  padding: '6px 14px',
                  borderRadius: 6,
                  fontSize: 14,
                  fontWeight: active ? 600 : 400,
                  color: active ? 'var(--cc-brand)' : '#506071',
                  background: active ? 'var(--cc-brand-light)' : 'transparent',
                  transition: 'all 0.15s',
                }}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Right: version + About + dark toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
            color: 'var(--cc-brand)',
            background: 'var(--cc-brand-light)',
            padding: '3px 10px',
            borderRadius: 20,
          }}
        >
          V2.4.0 RESEARCH EDITION
        </span>

        <button
          aria-label="Toggle dark mode"
          onClick={onToggleDark}
          style={{
            padding: '6px 8px',
            borderRadius: 6,
            border: '1px solid var(--cc-border)',
            background: 'transparent',
            color: '#506071',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>
            {darkMode ? 'light_mode' : 'dark_mode'}
          </span>
        </button>
      </div>
    </header>
  );
}
