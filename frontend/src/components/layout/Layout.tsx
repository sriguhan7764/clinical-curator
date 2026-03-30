import type { ReactNode } from 'react';
import Navbar from './Navbar';
import Sidebar from './Sidebar';

interface Props {
  children: ReactNode;
  darkMode: boolean;
  onToggleDark: () => void;
  hideSidebar?: boolean;
}

export default function Layout({ children, darkMode, onToggleDark, hideSidebar = false }: Props) {
  return (
    <>
      <Navbar darkMode={darkMode} onToggleDark={onToggleDark} />
      {!hideSidebar && <Sidebar />}
      <main
        style={{
          paddingTop: 'var(--cc-nav-height)',
          paddingLeft: hideSidebar ? 0 : 'var(--cc-sidebar-w)',
          minHeight: '100vh',
          background: 'var(--cc-page-bg)',
        }}
      >
        {children}
      </main>
    </>
  );
}
