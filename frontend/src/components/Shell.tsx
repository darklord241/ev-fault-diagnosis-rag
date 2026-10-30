'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Activity,
  AlertTriangle,
  BookOpen,
  CircleGauge,
  Command,
  Database,
  FileWarning,
  Menu,
  Search,
  Settings,
  ShieldCheck,
  Stethoscope,
  X,
} from 'lucide-react';
import { fetchHealth } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';

const links = [
  ['/', 'Control room', CircleGauge],
  ['/diagnose', 'Case console', Stethoscope],
  ['/knowledge', 'Evidence store', BookOpen],
  ['/dtc-lookup', 'Fault code index', FileWarning],
  ['/system', 'Service status', Settings],
] as const;

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [path]);
  return (
    <div className="app-shell">
      <button className="mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle navigation">{open ? <X size={18} /> : <Menu size={18} />}</button>
      <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
        <Link href="/" className="brand"><span className="brand-mark"><Command size={17} /></span><span><strong>EV / OPS</strong><small>diagnostic control room</small></span></Link>
        <div className="rail-status"><i /> SYSTEM ONLINE <span>LOCAL NODE</span></div>
        <div className="nav-caption">OPERATIONS</div>
        <nav className="main-nav">{links.map(([href, title, Icon]) => { const active = path === href || (href !== '/' && path.startsWith(href)); return <Link href={href} key={href} className={active ? 'nav-active' : ''}><Icon size={16} /><span>{title}</span>{active && <b />}</Link>; })}</nav>
        <div className="rail-divider" />
        <div className="rail-card"><ShieldCheck size={16} /><div><strong>Evidence locked</strong><span>Answers show source lineage and retrieval confidence.</span></div></div>
        <div className="sidebar-footer"><span>BUILD 0.4.0</span><span>© 2026 EV/OPS</span></div>
      </aside>
      <main className="main-shell">
        <header className="topbar"><div className="crumb"><span>EV / OPS</span><b>›</b><strong>{links.find(([href]) => href === path)?.[1] || 'Control room'}</strong></div><div className="topbar-meta"><span><i className="online-dot" /> GATEWAY LINKED</span><span className="topbar-clock">30 OCT 2026 / 18:17 IST</span><span className="operator">DV</span></div></header>
        <div className="page-content">{children}</div>
        <footer className="site-footer"><span>EV / OPS CONTROL ROOM</span><span>FOR TECHNICAL INVESTIGATION ONLY · HV SAFETY PROCEDURES APPLY</span></footer>
      </main>
    </div>
  );
}

export function StatusStrip() {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => { let live = true; const load = () => fetchHealth().then((data) => { if (live) { setStatus(data); setError(false); } }).catch(() => live && setError(true)); load(); const timer = setInterval(load, 15000); return () => { live = false; clearInterval(timer); }; }, []);
  return <div className="status-strip"><span><Activity size={14} /><i className={`signal ${status?.status === 'healthy' ? 'ok' : 'warn'}`} />{status?.status === 'healthy' ? 'ALL SERVICES NOMINAL' : error ? 'GATEWAY OFFLINE' : 'CHECKING SERVICES'}</span><span><Database size={14} /> RETRIEVAL: {status?.backend_retrieval_mode || 'STANDBY'}</span><span><AlertTriangle size={14} /> HV INTERLOCK: MANUAL CHECK</span><span className="status-end">{status?.active_mcp_tools ? `${(status.active_mcp_tools as string[]).length} TOOLS ACTIVE` : 'TOOLS STANDBY'}</span></div>;
}

export function PageTitle({ eyebrow, title, description, action }: { eyebrow: string; title: React.ReactNode; description: string; action?: React.ReactNode }) {
  return <div className="page-title"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}
export function ErrorBox({ message }: { message: string }) { return <div className="notice error" role="alert"><strong>FAULT / REQUEST NOT COMPLETED</strong><span>{message}</span></div>; }
export function Empty({ title, description }: { title: string; description: string }) { return <div className="empty"><BookOpen size={22} /><h3>{title}</h3><p>{description}</p></div>; }
