'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();
  const isProjects = pathname === '/projects' || pathname?.startsWith('/projects/');
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 2);
      const doc = document.documentElement;
      const p = doc.scrollTop / (doc.scrollHeight - doc.clientHeight);
      setProgress(Math.max(0, Math.min(1, p)));
    };
    onScroll();
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => { setOpen(false); }, [pathname]);

  return (
    <header className={`nav-wrap ${scrolled ? 'nav-scrolled' : ''}`}>
      <div className={`nav ${scrolled ? 'nav-compact' : ''}`}>
        {/* LEFT */}
        <div className="left">
          <Link href="/" className="brand" aria-label="Home">Layanne</Link>
          <Link
            href="/projects"
            className={`navlink ${isProjects ? 'active' : ''}`}
            aria-current={isProjects ? 'page' : undefined}
          >
            Projects
          </Link>
        </div>

        {/* CENTER */}
        <nav className="center" aria-label="External links">
          <a href="https://github.com/layanneelassaad" target="_blank" rel="noreferrer">GitHub</a>
          <a href="https://www.linkedin.com/in/layanne-el-assaad-8881ab22a/" target="_blank" rel="noreferrer">LinkedIn</a>
        </nav>

        {/* RIGHT */}
        <div className="right">
          <Link href="/contact" className="cta">contact me</Link>

          <button
            className="menu-btn"
            aria-label="Menu"
            aria-expanded={open}
            aria-controls="mobile-menu"
            onClick={() => setOpen(v => !v)}
          >
            <span />
            <span />
            <span />
          </button>
        </div>
      </div>

      {/* MOBILE PANEL */}
      <nav id="mobile-menu" className={`mobile-panel ${open ? 'open' : ''}`}>
        <Link href="/projects" className="mobile-link">Projects</Link>
        <a href="https://github.com/layanneelassaad" target="_blank" rel="noreferrer" className="mobile-link">GitHub</a>
        <a href="https://www.linkedin.com/in/layanne-el-assaad-8881ab22a/" target="_blank" rel="noreferrer" className="mobile-link">LinkedIn</a>
        <Link href="/contact" className="mobile-link">Contact</Link>
      </nav>

      {/* Scroll progress (hairline) */}
      <div className="nav-progress" style={{ transform: `scaleX(${progress})` }} />
    </header>
  );
}
