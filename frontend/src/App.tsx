import { useEffect, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import './styles/tokens.css';
import Landing from './pages/Landing';
import Upload  from './pages/Upload';
import Results from './pages/Results';
import Compare from './pages/Compare';

export default function App() {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('cc-theme') === 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('cc-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const props = { darkMode, onToggleDark: () => setDarkMode((d) => !d) };

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"        element={<Landing {...props} />} />
        <Route path="/upload"  element={<Upload  {...props} />} />
        <Route path="/results" element={<Results {...props} />} />
        <Route path="/compare" element={<Compare {...props} />} />
        <Route path="*"        element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
