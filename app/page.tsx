// app/page.tsx
'use client';

import { Suspense } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import ProjectModal from './components/ProjectModal';
import type { Project } from './data/projects';
import { projects } from './data/projects';

// Pick & order the 4 cards shown on Home
const featuredOrder = ['scout', 'cloud9', 'transfer-hierarchical', 'pipeline-parallel'];
const featuredHome: Project[] = featuredOrder
  .map(slug => projects.find(p => p.slug === slug))
  .filter(Boolean) as Project[];

// 👉 outer page wraps the inner client UI with Suspense
export default function Home() {
  return (
    <Suspense fallback={null}>
      <HomeInner />
    </Suspense>
  );
}

function HomeInner() {
  const router = useRouter();
  const params = useSearchParams();
  const pathname = usePathname();
  const openSlug = params.get('p');
  const open = featuredHome.find(p => p.slug === openSlug) ?? null;

  const openProject = (slug: string) => router.replace(`/?p=${slug}`, { scroll: false });
  const closeProject = () => router.replace(pathname, { scroll: false });

  return (
    <>
      {/* HERO */}
      <section className="hero">
        <div className="hero-inner container">
          <div className="copy">
            <h1>Hi, I’m Layanne. I'm a master's student in ML at Columbia.</h1>
            <p className="sub">Here are some of my favorite projects.</p>
          </div>
          <div className="portrait">
            <Image src="/assets/square-portrait.jpg" alt="Portrait of Layanne" width={320} height={320} priority />
          </div>
        </div>
      </section>

      {/* FEATURED */}
      <section className="container" style={{ padding: '36px 0 56px' }}>
        <h2 className="section-title">Featured projects</h2>

        <div className="grid">
          {featuredHome.map((p) => (
            <button key={p.slug} className="card" onClick={() => openProject(p.slug)} aria-label={`${p.title} details`}>
              {p.img ? (
                <div className="media" style={{ background: p.bg ?? '#f6f6f6' }}>
                  <Image
                    src={p.img}
                    alt={p.title}
                    fill
                    sizes="(max-width:1100px) 100vw, 50vw"
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

        <div className="more-row">
          <Link href="/projects" className="cta cta-large">more projects</Link>
        </div>
      </section>

      <ProjectModal project={open} onClose={closeProject} />
    </>
  );
}
