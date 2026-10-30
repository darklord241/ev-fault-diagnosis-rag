'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Cpu,
  Gauge,
  Menu,
  Search,
  Settings,
  ShieldCheck,
  Stethoscope,
  X,
  Zap,
} from 'lucide-react';
import { fetchHealth } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';

const links = [
  ['/', 'Overview', Gauge],
  ['/diagnose', 'Diagnose', Stethoscope],
  ['/knowledge', 'Evidence library', BookOpen],
  ['/dtc-lookup', 'DTC index', Search],
  ['/system', 'System', Settings],
] as const;

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => setOpen(false), [path]);

  return (
    <div className="app-shell">
      <button className="mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle navigation">
        {open ? <X size={19} /> : <Menu size={19} />}
      </button>
      <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
        <Link href="/" className="brand">
          <span className="brand-mark"><Zap size={18} fill="currentColor" /></span>
          <span className="brand-name">volt<span>room</span><small>diagnostic studio</small></span>
        </Link>
        <div className="nav-caption">WORKBENCH</div>
        <nav className="main-nav">
          {links.map(([href, title, Icon]) => {
            const active = path === href || (href !== '/' && path.startsWith(href));
            return (
              <Link href={href} key={href} className={active ? 'nav-active' : ''}>
                <Icon size={17} strokeWidth={active ? 2.4 : 1.8} />
                <span>{title}</span>
                {active && <span className="nav-pip" />}
              </Link>
            );
          })}
        </nav>
        <div className="sidebar-note">
          <ShieldCheck size={16} />
          <div><strong>Evidence-first</strong><span>Every answer carries its trail.</span></div>
        </div>
        <div className="sidebar-footer">
          <span className="live-led" /> Local workshop node
          <span className="sidebar-version">v0.4</span>
        </div>
      </aside>
      <main className="main-shell">
        <header className="topbar">
          <div className="crumb"><span>VOLTROOM</span><span>/</span><strong>{links.find(([href]) => href === path)?.[1] || 'Workbench'}</strong></div>
          <div className="topbar-meta"><span className="topbar-live"><i /> services online</span><span className="topbar-date">30 OCT 2026</span><span className="avatar">DV</span></div>
        </header>
        <div className="page-content">{children}</div>
        <footer className="site-footer"><span>VOLTROOM / ELECTRIC TWO-WHEELER DIAGNOSTICS</span><span>Retrieval confidence is not repair certainty.</span></footer>
      </main>
    </div>
  );
}

export function StatusStrip() {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    const load = () => fetchHealth().then((value) => active && (setStatus(value), setError(false))).catch(() => active && setError(true));
    load();
    const timer = setInterval(load, 15000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  return (
    <div className="status-strip">
      <span><Activity size={14} /><i className={`status-led ${status?.status === 'healthy' ? 'green' : 'amber'}`} />{status?.status === 'healthy' ? `${status.default_llm_provider || 'LLM'} ready` : error ? 'Backend unavailable' : 'Checking services'}</span>
      <span><Cpu size={14} />{status?.active_mcp_tools ? `${(status.active_mcp_tools as string[]).length} retrieval tools` : 'Tools pending'}</span>
      <span className="status-end">{status?.backend_retrieval_mode || 'local gateway'} <ArrowUpRight size={13} /></span>
    </div>
  );
}

export function PageTitle({ eyebrow, title, description, action }: { eyebrow: string; title: React.ReactNode; description: string; action?: React.ReactNode }) {
  return <div className="page-title"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="notice error" role="alert"><strong>Something needs attention</strong><span>{message}</span></div>;
}

export function Empty({ title, description }: { title: string; description: string }) {
  return <div className="empty"><BookOpen size={24} /><h3>{title}</h3><p>{description}</p></div>;
}
