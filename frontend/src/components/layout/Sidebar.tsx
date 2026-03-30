import { Link, useLocation } from 'react-router-dom';

const ITEMS = [
  { to: '/',        icon: 'home',         label: 'Home'         },
  { to: '/upload',  icon: 'upload_file',  label: 'X-Ray Viewer' },
  { to: '/results', icon: 'analytics',    label: 'AI Insights'  },
  { to: '/compare', icon: 'compare',      label: 'Compare'      },
];

export default function Sidebar() {
  const { pathname } = useLocation();

  return (
    <aside
      style={{
        position: 'fixed',
        top: 'var(--cc-nav-height)',
        left: 0,
        bottom: 0,
        width: 'var(--cc-sidebar-w)',
        background: 'var(--cc-sidebar-bg)',
        borderRight: '1px solid var(--cc-border)',
        display: 'flex',
        flexDirection: 'column',
        padding: '20px 12px',
        zIndex: 90,
      }}
    >
      <nav style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {ITEMS.map(({ to, icon, label }) => {
          const active = pathname === to || (to !== '/' && pathname.startsWith(to));
          return (
            <Link
              key={to}
              to={to}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                padding: '10px 14px',
                borderRadius: 8,
                background: active ? 'var(--cc-brand-light)' : 'transparent',
                color: active ? 'var(--cc-brand)' : '#506071',
                fontWeight: active ? 600 : 400,
                fontSize: 13,
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
                transition: 'all 0.15s',
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 20 }}>{icon}</span>
              {label}
            </Link>
          );
        })}
      </nav>

      <div style={{ marginTop: 'auto', borderTop: '1px solid var(--cc-border)', paddingTop: 16, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <Link
          to="/upload"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            padding: '10px 16px',
            background: 'var(--cc-brand)',
            color: '#fff',
            borderRadius: 8,
            fontSize: 13,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>add</span>
          New Scan
        </Link>
      </div>
    </aside>
  );
}
