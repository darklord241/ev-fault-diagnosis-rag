'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  ChevronRight,
  Database,
  LayoutDashboard,
  Search,
  ShieldCheck,
  Stethoscope,
  Settings,
  Zap,
} from 'lucide-react';
import { fetchHealth } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';

const links = [
  ['/', 'Workspace', LayoutDashboard],
  ['/diagnose', 'Diagnose', Stethoscope],
  ['/knowledge', 'Knowledge Base', BookOpen],
  ['/dtc-lookup', 'DTC Lookup', Search],
  ['/system', 'System', Settings],
] as const;

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link href="/" className="brand">
          <span className="brand-icon">
            <Zap size={22} />
          </span>
          <span>
            EV Diag
            <span className="brand-sub">FAULT INTELLIGENCE</span>
          </span>
        </Link>

        <div className="nav-label">DIAGNOSTIC WORKSPACE</div>

        <nav>
          {links.map(([href, title, Icon]) => {
            const isActive =
              path === href ||
              (href !== '/' && path.startsWith(href));
            return (
              <Link
                key={href}
                href={href}
                className={isActive ? 'active' : ''}
              >
                <Icon size={18} />
                {title}
                {isActive && <ChevronRight size={14} className="nav-arrow" />}
              </Link>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <ShieldCheck size={18} />
          <div>
            AI-Grounded Evidence
            <p>Retrieval-augmented diagnosis</p>
          </div>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <span className="breadcrumb">
            EV Fault Diagnosis <ChevronRight size={13} />{' '}
            <strong>
              {links.find(([href]) => href === path)?.[1] || 'Diagnosis'}
            </strong>
          </span>
          <div className="topbar-right">
            <span className="tiny-dot" /> ENGINEERING PREVIEW{' '}
            <span className="avatar">EV</span>
          </div>
        </header>

        <div className="page-content">{children}</div>

        <footer>
          EV Fault Diagnosis · AI-Powered Technical Intelligence
          <span>Grounding scores are retrieval confidence, not diagnostic certainty.</span>
        </footer>
      </main>
    </div>
  );
}

export function StatusStrip() {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let live = true;
    const load = () =>
      fetchHealth()
        .then((v) => {
          if (live) {
            setStatus(v);
            setError(false);
          }
        })
        .catch(() => {
          if (live) setError(true);
        });
    load();
    const timer = setInterval(load, 15000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="status-strip">
      <span>
        <Activity size={15} />
        <i className={status?.status === 'healthy' ? 'dot green' : 'dot amber'} />
        {status?.status === 'healthy'
          ? `${status.default_llm_provider || 'LLM'} · ${status.default_llm_model || 'Ready'}`
          : error
          ? 'Backend not connected'
          : 'Checking status…'}
      </span>
      <span>
        <Database size={15} />
        {status
          ? `MCP: ${status.mcp_transport || 'unknown'} · ${status.backend_retrieval_mode || 'unknown'}`
          : 'Awaiting backend'}
      </span>
      <span className="muted">
        {status?.active_mcp_tools
          ? `${(status.active_mcp_tools as string[]).length} tools connected`
          : 'Tools pending'}
        <ArrowUpRight size={13} />
      </span>
    </div>
  );
}

export function PageTitle({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="page-title">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="notice error" role="alert">
      {message}
    </div>
  );
}

export function Empty({ title, description }: { title: string; description: string }) {
  return (
    <div className="empty">
      <BookOpen size={30} />
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}
