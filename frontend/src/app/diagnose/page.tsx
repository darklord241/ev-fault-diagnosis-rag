'use client';

import { Suspense, useCallback, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';
import {
  ArrowRight,
  Clock,
  Database,
  ExternalLink,
  FileText,
  Trash2,
  Wrench,
} from 'lucide-react';
import {
  clearDiagnosisHistory,
  fetchDiagnosisHistory,
  fetchHealth,
  queryLive,
} from '@/lib/api';
import type {
  DiagnosisHistoryEntry,
  HealthResponse,
  WebsiteCardPayload,
} from '@/lib/types';
import { ErrorBox, PageTitle } from '@/components/Shell';

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
  const initialQuery = searchParams.get('query') || '';
  const initialVehicle = searchParams.get('vehicle') || '';
  const initialSubsystem = searchParams.get('subsystem') || '';
  const [query, setQuery] = useState(initialQuery);
  const [vehicle, setVehicle] = useState(initialVehicle);
  const [subsystem, setSubsystem] = useState(initialSubsystem);
  const [result, setResult] = useState<WebsiteCardPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [history, setHistory] = useState<DiagnosisHistoryEntry[]>([]);
  const [historyError, setHistoryError] = useState('');

  const refreshHistory = useCallback(async () => {
    try {
      const response = await fetchDiagnosisHistory();
      setHistory(response.items);
      setHistoryError('');
    } catch (e) {
      setHistoryError((e as Error).message);
    }
  }, []);

  const runDiagnosis = useCallback(async (q: string, model: string, target: string) => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const answer = await queryLive({
        query: q,
        vehicle_model: model || undefined,
        subsystem: target || undefined,
        top_k: 5,
      });
      setResult(answer);
      void refreshHistory();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [refreshHistory]);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealthError(true));
    void refreshHistory();

    if (initialQuery) {
      void runDiagnosis(
        initialQuery,
        initialVehicle,
        initialSubsystem,
      );
    }
  }, [initialQuery, initialVehicle, initialSubsystem, refreshHistory, runDiagnosis]);

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (query.trim().length < 3) return;
    void runDiagnosis(query.trim(), vehicle, subsystem);
  }

  async function handleClearHistory() {
    try {
      await clearDiagnosisHistory();
      setHistory([]);
      setHistoryError('');
    } catch (e) {
      setHistoryError((e as Error).message);
    }
  }

  function showHistoryEntry(entry: DiagnosisHistoryEntry) {
    setQuery(entry.query);
    setVehicle(entry.vehicle_model || '');
    setSubsystem(entry.subsystem || '');
    setResult(entry.result);
    setError('');
  }

  return (
    <>
      <PageTitle
        eyebrow="RETRIEVAL-FIRST DIAGNOSIS"
        title="Ask about an EV fault."
        description="The agent retrieves source evidence, cleans it, and returns a structured answer with citations."
      />

      {health?.backend_retrieval_mode === 'mock_domain_db' && (
        <div className="notice" role="status">
          Demo retrieval is enabled. Results use sample evidence rather than the live corpus.
        </div>
      )}
      {health?.backend_retrieval_mode === 'live_hybrid_api' && health.retrieval_backend_ready === false && (
        <div className="notice error" role="alert">
          Live retrieval is configured, but the corpus backend is not ready. Requests will fail rather than use sample evidence.
        </div>
      )}
      {health?.backend_retrieval_mode === 'live_hybrid_api' && health.retrieval_backend_ready === true && (
        <div className="notice success" role="status">
          Live retrieval backend is ready.
        </div>
      )}
      {healthError && (
        <div className="notice error" role="alert">
          Could not verify retrieval mode. Check the System page and backend health before relying on a result.
        </div>
      )}

      <section className="card" style={{ marginBottom: 24 }}>
        <form onSubmit={handleSubmit}>
          <label htmlFor="diag-query">Fault description or question</label>
          <textarea
            id="diag-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            minLength={3}
            maxLength={4000}
            required
            placeholder="Describe the symptom, dashboard code, or technical question."
            rows={4}
          />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div>
              <label htmlFor="diag-vehicle">Vehicle model</label>
              <select id="diag-vehicle" value={vehicle} onChange={(event) => setVehicle(event.target.value)}>
                <option value="">Any vehicle</option>
                {vehicleOptions.map((option) => <option key={option} value={option}>{option}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="diag-subsystem">Target subsystem</label>
              <select id="diag-subsystem" value={subsystem} onChange={(event) => setSubsystem(event.target.value)}>
                <option value="">Any subsystem</option>
                {subsystemOptions.map((option) => <option key={option} value={option}>{option}</option>)}
              </select>
            </div>
          </div>

          <button className="button primary wide" disabled={loading || query.trim().length < 3}>
            {loading ? 'Retrieving evidence…' : 'Run Diagnosis'}
            <ArrowRight size={17} />
          </button>
        </form>
      </section>

      {error && <ErrorBox message={error} />}
      {loading && (
        <div className="loading">
          <span className="loading-dot" />
          Searching technical evidence and preparing the answer…
        </div>
      )}

      {result && (
        <div style={{ marginTop: 8 }}>
          {result.raw_chunks_retrieved === 0 && (
            <div className="notice error" role="alert">
              No evidence was retrieved. This is not a diagnosis; do not use it as repair guidance.
            </div>
          )}

          <div className="status-strip" style={{ marginBottom: 20 }}>
            <span><Database size={15} /> {result.unique_chunks_used} evidence chunks used</span>
            <span><FileText size={15} /> {result.sources.length} sources cited</span>
            <span><Clock size={15} /> {result.model_used}</span>
            {result.vehicle_model && <span className="muted">{result.vehicle_model}</span>}
          </div>

          {result.model_used.startsWith('deterministic-') && (
            <div className="notice" role="status">
              The configured LLM was unavailable, so this response is an extractive local fallback. Review the cited sources.
            </div>
          )}

          <section className="card" style={{ marginBottom: 16 }}>
            <div className="card-heading">
              <div className="icon-tile"><Wrench size={20} /></div>
              <h2>Answer</h2>
            </div>
            <p style={{ fontSize: 14, lineHeight: 1.8, color: 'var(--ink)' }}>{result.direct_answer}</p>
            {result.key_takeaways.length > 0 && (
              <>
                <h3 style={{ marginTop: 18 }}>Key takeaways</h3>
                <ul>{result.key_takeaways.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
              </>
            )}
          </section>

          {Object.keys(result.technical_specs).length > 0 && (
            <section className="card" style={{ marginBottom: 16 }}>
              <h2>Technical details</h2>
              <dl className="step-body">
                {Object.entries(result.technical_specs).map(([name, value]) => (
                  <div key={name}>
                    <dt>{name}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          {result.conditions_and_safety.length > 0 && (
            <section style={{ marginBottom: 16 }}>
              <div className="section-heading"><h2>Conditions and safety</h2></div>
              {result.conditions_and_safety.map((item, index) => (
                <div className="safety-warning" key={`${index}-${item}`}>{item}</div>
              ))}
            </section>
          )}

          {result.formatted_markdown && (
            <section className="card" style={{ marginBottom: 16 }}>
              <h2>Full response</h2>
              <div className="md-render" style={{ fontSize: 13, lineHeight: 1.8 }}>
                <ReactMarkdown>{result.formatted_markdown}</ReactMarkdown>
              </div>
            </section>
          )}

          {result.sources.length > 0 && (
            <section style={{ marginBottom: 20 }}>
              <div className="section-heading">
                <h2>Evidence sources</h2>
                <Link
                  className="source-link"
                  href={{
                    pathname: '/knowledge',
                    query: {
                      query: result.query,
                      ...(result.vehicle_model ? { vehicle: result.vehicle_model } : {}),
                      ...(subsystem ? { subsystem } : {}),
                    },
                  }}
                >
                  Explore related chunks <ArrowRight size={14} />
                </Link>
              </div>
              <div className="evidence-list">
                {result.sources.map((source) => (
                  <article className="evidence-item" key={source.chunk_id}>
                    <div className="evidence-header">
                      <div>
                        <strong>{source.title || 'Source title unavailable'}</strong>
                        <p className="muted evidence-meta">
                          {source.page_number != null ? `Page ${source.page_number}` : 'Page not specified'}
                          {' · '}
                          {(source.similarity_score * 100).toFixed(1)}% relevance
                        </p>
                      </div>
                      {source.url && (
                        <a className="source-link" href={source.url} target="_blank" rel="noopener noreferrer">
                          Source <ExternalLink size={14} />
                        </a>
                      )}
                    </div>
                    {source.content && (
                      <blockquote className="evidence-quote">{source.content}</blockquote>
                    )}
                    <details className="source-reference">
                      <summary>Chunk reference</summary>
                      <code>{source.chunk_id}</code>
                    </details>
                  </article>
                ))}
              </div>
              <p className="muted" style={{ fontSize: 10, marginTop: 8 }}>
                Relevance and grounding scores are retrieval signals, not diagnostic certainty.
              </p>
            </section>
          )}
        </div>
      )}

      <section className="card" style={{ marginTop: 24 }}>
        <div className="section-heading">
          <h2>Recent diagnosis history</h2>
          <button className="button secondary" onClick={handleClearHistory} disabled={history.length === 0}>
            <Trash2 size={14} /> Clear
          </button>
        </div>
        {historyError && <ErrorBox message={historyError} />}
        {history.length === 0 && !historyError && <p className="muted">No saved queries yet.</p>}
        <div className="evidence-list">
          {history.map((entry) => (
            <article className="evidence-item" key={`${entry.case_id}-${entry.processed_at}`}>
              <div className="evidence-header">
                <div>
                  <strong>{entry.query}</strong>
                  <p className="muted" style={{ fontSize: 10 }}>
                    {entry.processed_at.replace('T', ' ')}
                    {entry.vehicle_model ? ` · ${entry.vehicle_model}` : ''}
                    {entry.subsystem ? ` · ${entry.subsystem}` : ''}
                  </p>
                </div>
                <button className="button secondary" onClick={() => showHistoryEntry(entry)}>
                  View result
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}
