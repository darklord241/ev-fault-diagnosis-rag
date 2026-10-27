'use client';

import { useEffect, useState } from 'react';
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  FileText,
  Search,
  X,
} from 'lucide-react';
import { fetchEvalAnswers, searchKnowledge } from '@/lib/api';
import type {
  KnowledgeChunk,
  WebsiteCardPayload,
  BatchWebsiteDelivery,
} from '@/lib/types';
import { PageTitle, ErrorBox, Empty } from '@/components/Shell';

export default function KnowledgePage() {
  const [data, setData] = useState<BatchWebsiteDelivery | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [chunkQuery, setChunkQuery] = useState('');
  const [vehicleModel, setVehicleModel] = useState('');
  const [subsystem, setSubsystem] = useState('');
  const [chunks, setChunks] = useState<KnowledgeChunk[]>([]);
  const [chunkSearchLoading, setChunkSearchLoading] = useState(false);
  const [chunkSearchError, setChunkSearchError] = useState('');

  useEffect(() => {
    fetchEvalAnswers()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function handleChunkSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = chunkQuery.trim();
    if (query.length < 3) return;

    setChunkSearchLoading(true);
    setChunkSearchError('');
    setChunks([]);
    try {
      const response = await searchKnowledge({
        query,
        vehicle_model: vehicleModel || undefined,
        subsystem: subsystem || undefined,
      });
      setChunks(response.chunks);
    } catch (e) {
      setChunkSearchError((e as Error).message);
    } finally {
      setChunkSearchLoading(false);
    }
  }

  const filtered =
    data?.cases.filter(
      (c) =>
        c.query.toLowerCase().includes(search.toLowerCase()) ||
        c.case_id.toLowerCase().includes(search.toLowerCase()) ||
        c.direct_answer.toLowerCase().includes(search.toLowerCase())
    ) ?? [];

  return (
    <>
      <PageTitle
        eyebrow="KNOWLEDGE BASE"
        title="Search the source documents."
        description="Retrieve the five most relevant document chunks and inspect their page, source, and full text."
      />

      <section className="card knowledge-search-card">
        <form onSubmit={handleChunkSearch}>
          <label htmlFor="knowledge-query">Search question or topic</label>
          <textarea
            id="knowledge-query"
            value={chunkQuery}
            onChange={(event) => setChunkQuery(event.target.value)}
            minLength={3}
            maxLength={4000}
            required
            rows={3}
            placeholder="For example: What should I check if my battery is not charging?"
          />
          <div className="knowledge-search-filters">
            <div>
              <label htmlFor="knowledge-vehicle">Vehicle model</label>
              <select
                id="knowledge-vehicle"
                value={vehicleModel}
                onChange={(event) => setVehicleModel(event.target.value)}
              >
                <option value="">Any vehicle</option>
                <option>Ather 450S</option>
                <option>Ather 450X</option>
                <option>Ola S1 Pro</option>
                <option>TVS iQube S</option>
                <option>Generic BLDC EV</option>
              </select>
            </div>
            <div>
              <label htmlFor="knowledge-subsystem">Subsystem</label>
              <select
                id="knowledge-subsystem"
                value={subsystem}
                onChange={(event) => setSubsystem(event.target.value)}
              >
                <option value="">Any subsystem</option>
                <option>Battery/BMS</option>
                <option>Motor Controller</option>
                <option>Throttle</option>
                <option>Charger/Charging</option>
                <option>Wiring/Harness</option>
                <option>Dashboard/ECU</option>
              </select>
            </div>
          </div>
          <button
            className="button primary"
            disabled={chunkSearchLoading || chunkQuery.trim().length < 3}
          >
            <Search size={15} />
            {chunkSearchLoading ? 'Searching chunks…' : 'Find top 5 chunks'}
          </button>
        </form>
      </section>

      {chunkSearchError && <ErrorBox message={chunkSearchError} />}
      {chunkSearchLoading && (
        <div className="loading">
          <span className="loading-dot" />
          Searching document chunks…
        </div>
      )}
      {!chunkSearchLoading && chunks.length > 0 && (
        <section className="knowledge-results">
          <div className="section-heading">
            <h2>Top {chunks.length} matching chunks</h2>
            <span className="muted small">Ranked by vector similarity</span>
          </div>
          {chunks.map((chunk, index) => (
            <article className="card knowledge-chunk" key={chunk.chunk_id}>
              <div className="knowledge-chunk-heading">
                <div>
                  <span className="badge">Match {index + 1}</span>
                  <h3>{chunk.title}</h3>
                  <div className="knowledge-chunk-meta">
                    <span>{chunk.source_type} · {chunk.content_type}</span>
                    {chunk.page_number != null && <span>Page {chunk.page_number}</span>}
                    {chunk.section && <span>{chunk.section}</span>}
                    {chunk.vehicle_model && <span>{chunk.vehicle_model}</span>}
                    {chunk.subsystem && <span>{chunk.subsystem}</span>}
                    <span>{(chunk.relevance_score * 100).toFixed(1)}% relevance</span>
                  </div>
                </div>
                {chunk.url && (
                  <a
                    className="source-link"
                    href={chunk.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Open source <ExternalLink size={14} />
                  </a>
                )}
              </div>
              <div className="knowledge-chunk-content">{chunk.content}</div>
              <details className="knowledge-chunk-reference">
                <summary><FileText size={13} /> Chunk reference</summary>
                <code>{chunk.chunk_id}</code>
              </details>
            </article>
          ))}
          <p className="muted small">
            Similarity ranks passages for this search; it is not a measure of diagnostic certainty.
          </p>
        </section>
      )}
      {!chunkSearchLoading && !chunkSearchError && chunkQuery && chunks.length === 0 && (
        <p className="muted knowledge-no-results">No matching document chunks found.</p>
      )}

      <div className="section-heading knowledge-answer-heading">
        <h2>Pre-processed answers</h2>
        <span className="muted small">Saved evaluation summaries</span>
      </div>
      {error && <ErrorBox message={error} />}

      {loading ? (
        <div className="loading">
          <span className="loading-dot" />
          Loading knowledge base…
        </div>
      ) : (
        <>
          {/* Search Bar */}
          <div className="search-field" style={{ marginBottom: 20 }}>
            <Search size={16} style={{ color: 'var(--muted)', marginRight: 8 }} />
            <input
              placeholder="Search queries, answers, or case IDs…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            {search && (
              <button
                className="icon-button"
                style={{ border: 0, padding: 4 }}
                onClick={() => setSearch('')}
              >
                <X size={14} />
              </button>
            )}
          </div>

          {data && (
            <div className="status-strip" style={{ marginBottom: 20 }}>
              <span>
                <BookOpen size={15} />
                {data.total_cases} cases · v{data.version}
              </span>
              <span className="muted">{data.project_title}</span>
            </div>
          )}

          {filtered.length === 0 ? (
            <Empty
              title="No matching answers"
              description="Try a different search term or clear the filter."
            />
          ) : (
            <div className="knowledge-grid">
              {filtered.map((c) => (
                <CaseCard
                  key={c.case_id}
                  caseData={c}
                  isExpanded={expanded === c.case_id}
                  onToggle={() =>
                    setExpanded(expanded === c.case_id ? null : c.case_id)
                  }
                />
              ))}
            </div>
          )}
        </>
      )}
    </>
  );
}

function CaseCard({
  caseData,
  isExpanded,
  onToggle,
}: {
  caseData: WebsiteCardPayload;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="card knowledge-card" onClick={onToggle}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
        <span className="badge">{caseData.case_id}</span>
        <span className="badge success">
          {(caseData.grounding_score * 100).toFixed(0)}% grounded
        </span>
      </div>

      <h3>{caseData.query}</h3>

      {caseData.vehicle_model && (
        <p className="muted" style={{ fontSize: 10, margin: '4px 0 8px' }}>
          {caseData.vehicle_model}
        </p>
      )}

      <p className="answer-preview">{caseData.direct_answer}</p>

      <div style={{ display: 'flex', gap: 8, marginTop: 12, fontSize: 9, color: 'var(--muted)' }}>
        <span>{caseData.sources.length} sources</span>
        <span>·</span>
        <span>{caseData.unique_chunks_used} unique chunks</span>
        <span>·</span>
        <span>{caseData.model_used}</span>
        <span style={{ marginLeft: 'auto' }}>
          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </span>
      </div>

      {/* Expanded Detail */}
      {isExpanded && (
        <div className="knowledge-detail" onClick={(e) => e.stopPropagation()}>
          {/* Key Takeaways */}
          {caseData.key_takeaways.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: 13, marginBottom: 8 }}>📋 Key Takeaways</h3>
              <ul style={{ paddingLeft: 20, margin: 0 }}>
                {caseData.key_takeaways.map((t, i) => (
                  <li key={i} style={{ fontSize: 12, margin: '4px 0', color: '#506760' }}>
                    {t}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Technical Specs */}
          {Object.keys(caseData.technical_specs).length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: 13, marginBottom: 8 }}>⚙️ Technical Specifications</h3>
              <table className="specs-table">
                <thead>
                  <tr>
                    <th>Specification</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(caseData.technical_specs).map(([k, v]) => (
                    <tr key={k}>
                      <td><strong>{k}</strong></td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Conditions & Safety */}
          {caseData.conditions_and_safety.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: 13, marginBottom: 8 }}>🛡️ Conditions & Safety</h3>
              {caseData.conditions_and_safety.map((c, i) => (
                <div className="caution-box" key={i}>
                  ⚠️ {c}
                </div>
              ))}
            </div>
          )}

          {/* Grounding Score Bar */}
          <div style={{ marginTop: 16 }}>
            <div className="score-label">
              <span>Grounding confidence</span>
              <strong>{(caseData.grounding_score * 100).toFixed(0)}%</strong>
            </div>
            <div className="score-bar-container">
              <div
                className="score-bar-fill"
                style={{ width: `${caseData.grounding_score * 100}%` }}
              />
            </div>
          </div>

          {/* Sources */}
          {caseData.sources.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: 13, marginBottom: 8 }}>📚 Sources</h3>
              {caseData.sources.map((s) => (
                <div
                  key={s.chunk_id}
                  style={{
                    borderBottom: '1px solid var(--line)',
                    padding: '10px 0',
                    fontSize: 12,
                  }}
                >
                  <strong>{s.title}</strong>
                  <div style={{ display: 'flex', gap: 8, marginTop: 4, fontSize: 10, color: 'var(--muted)' }}>
                    <span>Similarity: {(s.similarity_score * 100).toFixed(1)}%</span>
                    <span>·</span>
                    <span style={{ fontFamily: 'monospace', fontSize: 9 }}>{s.chunk_id}</span>
                  </div>
                  {s.url && (
                    <a
                      href={s.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        fontSize: 10,
                        color: 'var(--teal)',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 4,
                        marginTop: 4,
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      View source <ExternalLink size={11} />
                    </a>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
