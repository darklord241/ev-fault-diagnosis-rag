'use client';

import { useEffect, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Database,
  RefreshCw,
  Server,
  Wrench,
  Zap,
} from 'lucide-react';
import { fetchHealth, fetchMCPTools, runBatchPipeline } from '@/lib/api';
import type { HealthResponse, MCPTool, BatchWebsiteDelivery } from '@/lib/types';
import { PageTitle, ErrorBox } from '@/components/Shell';

export default function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [healthError, setHealthError] = useState('');
  const [toolsError, setToolsError] = useState('');
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineResult, setPipelineResult] = useState<BatchWebsiteDelivery | null>(null);
  const [pipelineError, setPipelineError] = useState('');

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e.message));

    fetchMCPTools()
      .then((data) => {
        setTools(data.mcp_tools || []);
      })
      .catch((e) => setToolsError(e.message));
  }, []);

  async function handleRunPipeline() {
    setPipelineRunning(true);
    setPipelineError('');
    setPipelineResult(null);
    try {
      const result = await runBatchPipeline();
      setPipelineResult(result);
    } catch (e) {
      setPipelineError((e as Error).message);
    } finally {
      setPipelineRunning(false);
    }
  }

  return (
    <>
      <PageTitle
        eyebrow="SYSTEM OVERVIEW"
        title="Service health & tools."
        description="Monitor the Agent Orchestrator, MCP server, retrieval backend, and connected tools."
      />

      {/* Metrics Grid */}
      <div className="metric-grid">
        <div className="card metric">
          <p>Service Status</p>
          <strong style={{ color: health?.status === 'healthy' ? 'var(--green)' : 'var(--amber)' }}>
            {health?.status === 'healthy' ? '✓' : '—'}
          </strong>
          <span>{health?.status || 'Unknown'}</span>
        </div>
        <div className="card metric">
          <p>LLM Provider</p>
          <strong style={{ fontSize: 20 }}>
            {(health?.default_llm_provider as string) || '—'}
          </strong>
          <span>{(health?.default_llm_model as string) || 'Not configured'}</span>
        </div>
        <div className="card metric">
          <p>MCP Transport</p>
          <strong style={{ fontSize: 20 }}>
            {(health?.mcp_transport as string) || '—'}
          </strong>
          <span>Tool server protocol</span>
        </div>
        <div className="card metric">
          <p>Connected Tools</p>
          <strong>
            {health?.active_mcp_tools
              ? (health.active_mcp_tools as string[]).length
              : '—'}
          </strong>
          <span>MCP retrieval tools</span>
        </div>
      </div>

      <div className="admin-grid">
        {/* Health Details */}
        <section className="card">
          <div className="card-heading">
            <div className="icon-tile">
              <Server size={20} />
            </div>
            <h2>Service Configuration</h2>
          </div>

          {healthError ? (
            <ErrorBox message={healthError} />
          ) : health ? (
            <div>
              {Object.entries(health).map(([key, value]) => (
                <div className="service-row" key={key}>
                  <span style={{ fontWeight: 600, fontSize: 11 }}>
                    {key.replaceAll('_', ' ')}
                  </span>
                  <span style={{ color: 'var(--muted)', fontSize: 11, textAlign: 'right', maxWidth: '60%', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {typeof value === 'object'
                      ? JSON.stringify(value)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="loading">
              <span className="loading-dot" />
              Loading health data…
            </div>
          )}
        </section>

        {/* MCP Tools */}
        <section className="card">
          <div className="card-heading">
            <div className="icon-tile">
              <Wrench size={20} />
            </div>
            <h2>MCP Retrieval Tools</h2>
          </div>

          {toolsError ? (
            <ErrorBox message={toolsError} />
          ) : tools.length > 0 ? (
            tools.map((tool) => (
              <div className="tool-item" key={tool.name}>
                <h3>{tool.name}</h3>
                <p className="muted">{tool.description || 'No description'}</p>
              </div>
            ))
          ) : (
            <div className="loading">
              <span className="loading-dot" />
              Loading tools…
            </div>
          )}
        </section>
      </div>

      {/* Pipeline Runner */}
      <section className="card" style={{ marginTop: 22 }}>
        <div className="card-heading">
          <div className="icon-tile">
            <Database size={20} />
          </div>
          <div>
            <h2>Batch Pipeline</h2>
            <p>Run the full cleaning, summarization & website export pipeline</p>
          </div>
        </div>

        <button
          className="button primary"
          disabled={pipelineRunning}
          onClick={handleRunPipeline}
        >
          {pipelineRunning ? (
            <>
              <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />
              Running pipeline…
            </>
          ) : (
            <>
              <Zap size={16} />
              Run Batch Pipeline
              <ArrowRight size={15} />
            </>
          )}
        </button>

        {pipelineError && <ErrorBox message={pipelineError} />}

        {pipelineResult && (
          <div className="notice success" style={{ marginTop: 16 }}>
            ✅ Pipeline complete! Generated {pipelineResult.total_cases} cases.
            Version: {pipelineResult.version}
          </div>
        )}
      </section>

      {/* spin animation is in globals.css via loading-dot */}
    </>
  );
}
