'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  FileText,
  Shield,
  Wrench,
  Zap,
} from 'lucide-react';
import { diagnose } from '@/lib/api';
import type { GroundedAnswer, EvidenceSource } from '@/lib/types';
import { PageTitle, ErrorBox } from '@/components/Shell';
import { EvidenceDrawer } from '@/components/EvidenceDrawer';

const vehicleOptions = [
  'Ather 450X',
  'Ather 450S',
  'Ola S1 Pro',
  'TVS iQube S',
  'Generic BLDC EV',
];

const subsystemOptions = [
  'Battery/BMS',
  'Motor Controller',
  'Throttle',
  'Charger/Charging',
  'Wiring/Harness',
  'Dashboard/ECU',
];

export default function DiagnosePage() {
  return (
    <Suspense fallback={<div className="loading"><span className="loading-dot" /> Loading…</div>}>
      <DiagnoseContent />
    </Suspense>
  );
}

function DiagnoseContent() {
  const searchParams = useSearchParams();
  const [query, setQuery] = useState(searchParams.get('query') || '');
  const [vehicle, setVehicle] = useState(searchParams.get('vehicle') || '');
  const [subsystem, setSubsystem] = useState(searchParams.get('subsystem') || '');
  const [result, setResult] = useState<GroundedAnswer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedSource, setSelectedSource] = useState<EvidenceSource | null>(null);
  const [showTrace, setShowTrace] = useState(false);

  // Auto-run if query params present
  useEffect(() => {
    const q = searchParams.get('query');
    if (q && !result && !loading) {
      runDiagnosis(q, searchParams.get('vehicle') || '', searchParams.get('subsystem') || '');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runDiagnosis(q: string, v: string, s: string) {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const answer = await diagnose({
        query: q,
        vehicle_model: v || undefined,
        subsystem: s || undefined,
      });
      setResult(answer);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    runDiagnosis(query.trim(), vehicle, subsystem);
  }

  return (
    <>
      <PageTitle
        eyebrow="INTERACTIVE DIAGNOSIS"
        title="AI-powered fault analysis."
        description="Describe the fault symptom, DTC code, or technical question. The agent retrieves evidence and delivers a grounded diagnosis."
      />

      {/* ── Input Form ── */}
      <section className="card" style={{ marginBottom: 24 }}>
        <form onSubmit={handleSubmit}>
          <label htmlFor="diag-query">Fault description or question</label>
          <textarea
            id="diag-query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            minLength={3}
            required
            placeholder="e.g., Scooter shows E-07 error code, motor not responding after full charge..."
            rows={4}
          />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div>
              <label htmlFor="diag-vehicle">Vehicle model</label>
              <select id="diag-vehicle" value={vehicle} onChange={(e) => setVehicle(e.target.value)}>
                <option value="">Any vehicle</option>
                {vehicleOptions.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="diag-subsystem">Target subsystem</label>
              <select id="diag-subsystem" value={subsystem} onChange={(e) => setSubsystem(e.target.value)}>
                <option value="">Any subsystem</option>
                {subsystemOptions.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          </div>

          <button
            className="button primary wide"
            disabled={loading || query.trim().length < 3}
          >
            {loading ? 'Analyzing fault…' : 'Run Diagnosis'}
            <ArrowRight size={17} />
          </button>
        </form>
      </section>

      {error && <ErrorBox message={error} />}

      {loading && (
        <div className="loading">
          <span className="loading-dot" />
          Querying MCP retrieval tools and analyzing evidence…
        </div>
      )}

      {/* ── Results ── */}
      {result && (
        <div style={{ marginTop: 8 }}>
          {/* Meta bar */}
          <div className="status-strip" style={{ marginBottom: 20 }}>
            <span>
              <Zap size={15} />
              Grounding: {(result.grounding_score * 100).toFixed(0)}%
            </span>
            <span>
              <FileText size={15} />
              {result.evidence_sources.length} sources cited
            </span>
            {result.execution_time_seconds && (
              <span>
                <Clock size={15} />
                {result.execution_time_seconds.toFixed(1)}s
              </span>
            )}
            {result.vehicle_model && (
              <span className="muted">{result.vehicle_model}</span>
            )}
          </div>

          {/* Identified Faults */}
          {result.identified_faults.length > 0 && (
            <section className="card" style={{ marginBottom: 16 }}>
              <div className="card-heading">
                <div className="icon-tile">
                  <AlertTriangle size={20} />
                </div>
                <div>
                  <h2>Identified Faults</h2>
                  <p>Likely faults based on retrieved evidence</p>
                </div>
              </div>
              <div className="model-tags">
                {result.identified_faults.map((f) => (
                  <span key={f} className="badge danger">{f}</span>
                ))}
              </div>
            </section>
          )}

          {/* Root Cause Analysis */}
          <section className="card" style={{ marginBottom: 16 }}>
            <div className="card-heading">
              <div className="icon-tile">
                <Wrench size={20} />
              </div>
              <h2>Root Cause Analysis</h2>
            </div>
            <div className="md-render" style={{ fontSize: 13, lineHeight: 1.8 }}>
              <ReactMarkdown>{result.root_cause_analysis}</ReactMarkdown>
            </div>
          </section>

          {/* Diagnostic Procedure */}
          {result.diagnostic_procedure.length > 0 && (
            <section style={{ marginBottom: 16 }}>
              <div className="section-heading">
                <h2>Diagnostic Procedure</h2>
                <span className="badge success">
                  {result.diagnostic_procedure.length} steps
                </span>
              </div>

              {result.diagnostic_procedure.map((step) => (
                <div className="step-card" key={step.step_number}>
                  <div className="step-header">
                    <span className="step-number">{step.step_number}</span>
                    <h3>{step.action}</h3>
                  </div>
                  <dl className="step-body">
                    <dt>COMPONENT CHECKED</dt>
                    <dd>{step.component_checked}</dd>

                    {step.expected_reading && (
                      <>
                        <dt>EXPECTED READING</dt>
                        <dd>{step.expected_reading}</dd>
                      </>
                    )}
                  </dl>

                  {step.caution && (
                    <div className="caution-box">
                      ⚠️ {step.caution}
                    </div>
                  )}

                  {step.citation_source_ids.length > 0 && (
                    <div style={{ marginTop: 10, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {step.citation_source_ids.map((id) => {
                        const source = result.evidence_sources.find(
                          (s) => s.source_id === id
                        );
                        return (
                          <button
                            key={id}
                            className="badge"
                            style={{ cursor: 'pointer' }}
                            onClick={() => source && setSelectedSource(source)}
                          >
                            📎 {id}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </section>
          )}

          {/* Recommended Remedy */}
          <section className="card" style={{ marginBottom: 16 }}>
            <div className="card-heading">
              <div className="icon-tile" style={{ background: '#eaf6f0', color: '#197d5b' }}>
                <CheckCircle2 size={20} />
              </div>
              <h2>Recommended Remedy</h2>
            </div>
            <p style={{ fontSize: 13, color: 'var(--ink)', lineHeight: 1.8 }}>
              {result.recommended_remedy}
            </p>
          </section>

          {/* Safety Warnings */}
          {result.safety_warnings.length > 0 && (
            <section style={{ marginBottom: 16 }}>
              <div className="section-heading">
                <h2>
                  <Shield size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />
                  Safety Warnings
                </h2>
              </div>
              {result.safety_warnings.map((w, i) => (
                <div className="safety-warning" key={i}>
                  ⚠️ {w}
                </div>
              ))}
            </section>
          )}

          {/* Grounding Score */}
          <section className="card" style={{ marginBottom: 16 }}>
            <h2 style={{ marginBottom: 12 }}>Grounding Confidence</h2>
            <div className="score-label">
              <span>Retrieval-based grounding</span>
              <strong>{(result.grounding_score * 100).toFixed(0)}%</strong>
            </div>
            <div className="score-bar-container">
              <div
                className="score-bar-fill"
                style={{ width: `${result.grounding_score * 100}%` }}
              />
            </div>
            <p className="muted" style={{ fontSize: 10, marginTop: 8 }}>
              This score reflects how well the answer is grounded in retrieved evidence.
              It is not a measure of diagnostic certainty.
            </p>
          </section>

          {/* Evidence Sources */}
          {result.evidence_sources.length > 0 && (
            <section style={{ marginBottom: 16 }}>
              <div className="section-heading">
                <h2>Evidence Sources ({result.evidence_sources.length})</h2>
              </div>
              <div className="evidence-list">
                {result.evidence_sources.map((src) => (
                  <div className="evidence-item" key={src.source_id}>
                    <div
                      className="evidence-header"
                      onClick={() => setSelectedSource(src)}
                    >
                      <div>
                        <strong style={{ fontSize: 13 }}>{src.document_title}</strong>
                        <p style={{ fontSize: 10, margin: '4px 0 0' }}>
                          {src.document_type}
                          {src.subsystem && ` · ${src.subsystem}`}
                          {src.section && ` · ${src.section}`}
                        </p>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <span className="badge">
                          {(src.relevance_score * 100).toFixed(0)}% match
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Agent Reasoning Trace */}
          {result.agent_reasoning_trace && result.agent_reasoning_trace.length > 0 && (
            <section className="trace-container">
              <button
                className="trace-toggle"
                onClick={() => setShowTrace(!showTrace)}
              >
                {showTrace ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                Agent Reasoning Trace ({result.agent_reasoning_trace.length} steps)
              </button>
              {showTrace && (
                <ul className="trace-list">
                  {result.agent_reasoning_trace.map((step, i) => (
                    <li key={i}>
                      <span style={{ color: '#6db89c', marginRight: 8 }}>
                        [{i + 1}]
                      </span>
                      {step}
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}
        </div>
      )}

      <EvidenceDrawer source={selectedSource} onClose={() => setSelectedSource(null)} />
    </>
  );
}
