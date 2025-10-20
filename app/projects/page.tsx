// app/projects/page.tsx
'use client';

import { Suspense } from 'react';
import Image from 'next/image';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import ProjectModal from '../components/ProjectModal';
import type { Project } from '../data/projects';
import { projects } from '../data/projects';

export default function ProjectsPage() {
  // Wrap the hook-using inner component with Suspense
  return (
    <Suspense fallback={null}>
      <ProjectsPageInner />
    </Suspense>
  );
}

function ProjectsPageInner() {
  const router = useRouter();
  const params = useSearchParams();
  const pathname = usePathname();
  const openSlug = params.get('p');
  const open = projects.find(p => p.slug === openSlug) ?? null;

  const openProject = (slug: string) => router.replace(`/projects?p=${slug}`, { scroll: false });
  const closeProject = () => router.replace(pathname, { scroll: false });

  return (
    <section className="container" style={{ padding: '56px 0' }}>
      <h1 style={{ margin: 0, fontSize: 32, lineHeight: 1.15 }}>Projects</h1>

      <div className="grid" style={{ marginTop: 20 }}>
        {projects.map(p => (
          <button
            key={p.slug}
            className="card"
            onClick={() => openProject(p.slug)}
            aria-label={`${p.title} details`}
          >
            {p.img ? (
              <div className="media" style={{ background: p.bg ?? '#f6f6f6' }}>
                <Image
                  src={p.img}
                  alt={p.title}
                  fill
                  sizes="(max-width:900px) 100vw, 50vw"
                  style={{ objectFit: p.fit ?? 'cover' }}
                  unoptimized
                />
              </div>
            ) : (
              <div className="media" />
            )}
            <div className="overlay">
              <span className="tag">{p.tag}</span>
              <h3>{p.title}</h3>
            </div>
          </button>
        ))}
      </div>

      <ProjectModal project={open} onClose={closeProject} />
    </section>
  );
}
