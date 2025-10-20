'use client';
import Image from 'next/image';
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import type { Project } from '../data/projects';
export type { Project } from '../data/projects';

function mdInline(s: string) {
  return s
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g,'$1')
    .replace(/\*([^*]+)\*/g,'<em>$1</em>');
}

export default function ProjectModal({ project, onClose }:{
  project: Project | null; onClose: () => void;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);

  // esc to close
  useEffect(() => {
    if (!project) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [project, onClose]);

  // lock background scroll while open
  useEffect(() => {
    if (!project || !mounted) return;
    const prev = document.documentElement.style.overflow;
    document.documentElement.style.overflow = 'hidden';
    return () => { document.documentElement.style.overflow = prev; };
  }, [project, mounted]);

  if (!project || !mounted) return null;

  const websiteLabel = project.slug === 'scout' ? 'Website' : 'Live site';

  const modal = (
    <div className="modal" role="dialog" aria-modal="true" aria-label={`${project.title} details`} onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-topbar">
          <div className="topbar-left">
            <span className="tag">{project.tag}</span>
            <h2>{project.title}</h2>
          </div>
          <div className="topbar-actions">
            {project.github && <a className="btn" href={project.github} target="_blank" rel="noreferrer">GitHub</a>}
            {project.demo && <a className="btn primary" href={project.demo} target="_blank" rel="noreferrer">{websiteLabel}</a>}
          </div>
          <button className="modal-close" aria-label="Close" onClick={onClose}>×</button>
        </div>

        {!project.hideMediaInModal && project.img && (
          <div className="modal-media" style={{ background: project.bg ?? '#000' }}>
            <Image src={project.img} alt={project.title} fill sizes="100vw" style={{ objectFit: project.fit ?? 'cover' }} unoptimized />
          </div>
        )}

        <div className="modal-body">
          {project.longIntro && <p className="lead">{project.longIntro}</p>}
          {project.sections?.map(sec => (
            <div key={sec.title} className="modal-section">
              <h3>{sec.title}</h3>
              <ul>{sec.items.map((it,i)=><li key={i} dangerouslySetInnerHTML={{ __html: mdInline(it) }} />)}</ul>
            </div>
          ))}
          {project.video && (
            <div className="modal-video">
              <video controls playsInline preload="metadata">
                <source src={project.video} type="video/mp4" />
              </video>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const portalTarget = typeof document !== 'undefined' ? document.body : null;
  if (!portalTarget) return null;
  return createPortal(modal, portalTarget);
}
