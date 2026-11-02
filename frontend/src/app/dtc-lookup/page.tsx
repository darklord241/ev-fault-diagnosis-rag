'use client';

import { useState } from 'react';
import { Search, AlertTriangle, Wrench, CheckCircle2, BookOpen } from 'lucide-react';
import { lookupDTC } from '@/lib/api';
import type { DTCRecord } from '@/lib/types';
import { PageTitle, ErrorBox, Empty } from '@/components/Shell';

export default function DTCLookupPage() {
  const [code, setCode] = useState('');
  const [data, setData] = useState<DTCRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notFound, setNotFound] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!code.trim()) return;

    setLoading(true);
    setError('');
    setData(null);
    setNotFound(false);

    try {
      const result = await lookupDTC(code.trim());
      setData(result);
    } catch (e) {
      const msg = (e as Error).message;
      if (msg.includes('404') || msg.includes('not found')) {
        setNotFound(true);
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle
        eyebrow="DTC CODE LOOKUP"
        title="Understand a fault code."
        description="Enter a Diagnostic Trouble Code to look up its meaning, symptoms, possible causes, and suggested checks."
      />

      <section className="card">
        <form onSubmit={handleSubmit} className="lookup-form">
          <div>
            <label htmlFor="dtc-code">DTC / Error Code</label>
            <input
              id="dtc-code"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              required
              maxLength={50}
              placeholder="e.g., P0A80, E-07, U0100"
            />
          </div>
          <button className="button primary" disabled={loading || !code.trim()}>
            <Search size={17} />
            {loading ? 'Looking up…' : 'Find code'}
          </button>
        </form>
      </section>

      {error && <ErrorBox message={error} />}

      <section className="card" style={{ marginTop: 20 }}>
        {data ? (
          <div>
            {/* Note */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
              <span className="dtc-code-display">{data.code}</span>
              <span className="badge success">{data.system}</span>
            </div>

            <p style={{ fontSize: 14, color: 'var(--ink)', lineHeight: 1.8, marginBottom: 20 }}>
              {data.description}
            </p>

            {/* Note */}
            {data.symptoms.length > 0 && (
              <div className="dtc-section">
                <h3>
                  <AlertTriangle size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />
                  Symptoms
                </h3>
                <ul>
                  {data.symptoms.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Note */}
            {data.possible_causes.length > 0 && (
              <div className="dtc-section">
                <h3>
                  <Wrench size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />
                  Possible Causes
                </h3>
                <ul>
                  {data.possible_causes.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Note */}
            {data.suggested_checks.length > 0 && (
              <div className="dtc-section">
                <h3>
                  <CheckCircle2 size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />
                  Suggested Checks
                </h3>
                <ul>
                  {data.suggested_checks.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Note */}
            <div
              style={{
                borderTop: '1px solid var(--line)',
                paddingTop: 16,
                marginTop: 20,
                fontSize: 11,
                color: 'var(--muted)',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <BookOpen size={14} />
              Reference: {data.source_manual}
            </div>
          </div>
        ) : notFound ? (
          <Empty
            title="Code not found"
            description={`DTC code '${code}' was not found in the database. Verify the code and try again.`}
          />
        ) : (
          <Empty
            title="Enter a DTC code"
            description="Diagnostic Trouble Code details, symptoms, causes, and recommended checks will appear here."
          />
        )}
      </section>
    </>
  );
}
