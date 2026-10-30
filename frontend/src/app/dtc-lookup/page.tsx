'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import {
  AlertTriangle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Search,
  ShieldAlert,
  Wrench,
} from 'lucide-react';
import { fetchDTCCatalog, lookupDTC } from '@/lib/api';
import type { DTCRecord } from '@/lib/types';
import { Empty, ErrorBox, PageTitle } from '@/components/Shell';

function normalizeCode(value: string) {
  return value.trim().toUpperCase().replace(/\s+/g, '');
}

export default function DTCLookupPage() {
  const [code, setCode] = useState('');
  const [catalog, setCatalog] = useState<DTCRecord[]>([]);
  const [data, setData] = useState<DTCRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [error, setError] = useState('');
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    fetchDTCCatalog()
      .then(setCatalog)
      .catch(() => setCatalog([]))
      .finally(() => setCatalogLoading(false));
  }, []);

  const suggestions = useMemo(() => {
    const query = normalizeCode(code);
    if (!query || data) return [];
    return catalog
      .filter((record) => record.code.includes(query))
      .slice(0, 6);
  }, [catalog, code, data]);

  async function findCode(value = code) {
    const normalized = normalizeCode(value);
    if (!normalized) return;

    setCode(normalized);
    setLoading(true);
    setError('');
    setData(null);
    setNotFound(false);

    try {
      setData(await lookupDTC(normalized));
    } catch (e) {
      const message = (e as Error).message;
      if (message.includes('404') || message.toLowerCase().includes('not found')) {
        setNotFound(true);
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    void findCode();
  }

  return (
    <>
      <PageTitle
        eyebrow="DTC CODE LOOKUP"
        title="Understand a fault code."
        description="Search EV-specific and generic OBD-II codes for plain-language meaning, symptoms, likely causes, and safe next checks."
      />

      <section className="card dtc-search-card">
        <form onSubmit={handleSubmit} className="lookup-form">
          <div className="dtc-input-wrap">
            <label htmlFor="dtc-code">DTC / Error Code</label>
            <div className="search-field">
              <Search size={17} aria-hidden="true" />
              <input
                id="dtc-code"
                value={code}
                onChange={(event) => {
                  setCode(event.target.value.toUpperCase());
                  setData(null);
                  setNotFound(false);
                }}
                required
                maxLength={50}
                placeholder="Try P0A80, P0100, P0300, E-07, or U0100"
                autoComplete="off"
                aria-describedby="dtc-help"
              />
            </div>
            <span id="dtc-help" className="field-hint">
              {catalogLoading ? 'Loading code catalog…' : `${catalog.length} codes available`}
            </span>
            {suggestions.length > 0 && (
              <div className="dtc-suggestions" role="listbox" aria-label="Matching codes">
                {suggestions.map((record) => (
                  <button
                    type="button"
                    key={record.code}
                    className="dtc-suggestion"
                    onClick={() => void findCode(record.code)}
                  >
                    <span className="dtc-suggestion-code">{record.code}</span>
                    <span>{record.description}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <button className="button primary" disabled={loading || !code.trim()}>
            <Search size={17} />
            {loading ? 'Looking up…' : 'Find code'}
          </button>
        </form>
      </section>

      {error && <ErrorBox message={error} />}

      <section className="card dtc-result-card">
        {data ? (
          <div>
            <div className="dtc-result-header">
              <div>
                <span className="dtc-code-display">{data.code}</span>
                <div className="dtc-badges">
                  <span className="badge success">{data.system}</span>
                  {data.code_type && <span className="badge">{data.code_type}</span>}
                  {data.category && <span className="badge">{data.category}</span>}
                </div>
              </div>
              <Link
                className="button secondary dtc-diagnose-button"
                href={`/diagnose?query=${encodeURIComponent(`${data.code}: ${data.description}`)}&subsystem=${encodeURIComponent(data.system)}`}
              >
                Diagnose this code <ArrowRight size={15} />
              </Link>
            </div>

            <p className="dtc-description">{data.description}</p>

            {data.safety_note && (
              <div className="dtc-safety">
                <ShieldAlert size={18} />
                <div>
                  <strong>Safety first</strong>
                  <p>{data.safety_note}</p>
                </div>
              </div>
            )}

            <div className="dtc-detail-grid">
              {data.symptoms.length > 0 && (
                <div className="dtc-section">
                  <h3><AlertTriangle size={14} /> Symptoms</h3>
                  <ul>{data.symptoms.map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              )}
              {data.possible_causes.length > 0 && (
                <div className="dtc-section">
                  <h3><Wrench size={14} /> Possible causes</h3>
                  <ul>{data.possible_causes.map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              )}
              {data.suggested_checks.length > 0 && (
                <div className="dtc-section">
                  <h3><CheckCircle2 size={14} /> Suggested checks</h3>
                  <ul>{data.suggested_checks.map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              )}
            </div>

            <div className="dtc-source">
              <BookOpen size={14} />
              <span>Reference: {data.source_manual}</span>
              {data.source_url && (
                <a href={data.source_url} target="_blank" rel="noreferrer">Open source</a>
              )}
            </div>
          </div>
        ) : notFound ? (
          <Empty
            title="Code not found"
            description={`DTC code '${code}' is not in the local catalog. Verify the code or use Diagnose to search the technical documents.`}
          />
        ) : (
          <Empty
            title="Search a code"
            description="Choose a suggestion or enter a code to see its meaning, symptoms, causes, checks, and safety guidance."
          />
        )}
      </section>
    </>
  );
}
