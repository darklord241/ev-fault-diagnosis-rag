'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowRight,
  BatteryCharging,
  BookOpen,
  ChevronRight,
  Search,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { StatusStrip, PageTitle, ErrorBox } from '@/components/Shell';

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

export default function Home() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [vehicle, setVehicle] = useState('');
  const [subsystem, setSubsystem] = useState('');
  const [error, setError] = useState('');

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    const params = new URLSearchParams({ query: query.trim() });
    if (vehicle) params.set('vehicle', vehicle);
    if (subsystem) params.set('subsystem', subsystem);
    router.push('/diagnose?' + params.toString());
  }

  return (
    <>
      <PageTitle
        eyebrow="AI-POWERED EV DIAGNOSIS"
        title="A clearer path to the problem."
        description="Describe what you observe. AI retrieves grounded technical evidence and walks you through diagnosis, step by step."
      />

      <StatusStrip />

      <div className="home-grid">
        {/* ── Left: Start Diagnosis ── */}
        <section className="card">
          <div className="card-heading">
            <div className="icon-tile">
              <BatteryCharging size={22} />
            </div>
            <div>
              <h2>Start a diagnostic session</h2>
              <p>For electric two-wheelers and charging systems</p>
            </div>
            <span className="badge">RAG-POWERED</span>
          </div>

          <form onSubmit={handleSubmit}>
            <label htmlFor="query">
              Describe the fault or symptom <span>REQUIRED</span>
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              minLength={3}
              maxLength={4000}
              required
              placeholder="For example: Battery not charging, charger LED blinking red. Error code E-07 on dashboard. Started after firmware update."
              rows={5}
            />
            <div className="input-footer">
              <span>Include dashboard codes, LED colors, and when it happens.</span>
              <span>{query.length}/4000</span>
            </div>

            <label htmlFor="vehicle" style={{ marginTop: 20 }}>
              Vehicle model
            </label>
            <select
              id="vehicle"
              value={vehicle}
              onChange={(e) => setVehicle(e.target.value)}
            >
              <option value="">Select a vehicle (optional)</option>
              {vehicleOptions.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>

            <label htmlFor="subsystem" style={{ marginTop: 20 }}>
              Target subsystem
            </label>
            <select
              id="subsystem"
              value={subsystem}
              onChange={(e) => setSubsystem(e.target.value)}
            >
              <option value="">Select a subsystem (optional)</option>
              {subsystemOptions.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>

            {error && <ErrorBox message={error} />}

            <button
              className="button primary wide"
              disabled={query.trim().length < 3}
            >
              Run AI Diagnosis
              <ArrowRight size={17} />
            </button>
          </form>

          <div className="privacy-line">
            <ShieldCheck size={14} /> Evidence-first: every claim cites its source.
          </div>
        </section>

        {/* ── Right: How It Works + Quick Links ── */}
        <aside className="home-aside">
          <section className="card approach">
            <div className="eyebrow">HOW IT WORKS</div>
            <h2>
              Less guessing.
              <br />
              More grounding.
            </h2>
            <ol>
              {[
                [
                  'Describe the symptom',
                  'Start with what you observe — dashboard codes, LED patterns, noises.',
                ],
                [
                  'AI retrieves evidence',
                  'RAG pipeline searches service manuals, DTC tables, and technical docs via MCP tools.',
                ],
                [
                  'Verified diagnosis',
                  'Grounded answer with step-by-step procedure, safety warnings, and cited sources.',
                ],
              ].map(([title, text], i) => (
                <li key={title}>
                  <span>{i + 1}</span>
                  <div>
                    <strong>{title}</strong>
                    <p>{text}</p>
                  </div>
                </li>
              ))}
            </ol>
            <div className="approach-note">
              <Sparkles size={16} />
              <p>
                Every technical claim links back to its evidence. Uncertain causes
                stay labelled as unconfirmed.
              </p>
            </div>
          </section>

          <section className="card" style={{ padding: 22 }}>
            <div className="eyebrow">PRE-PROCESSED ANSWERS</div>
            <div className="model-tags">
              {vehicleOptions.map((v) => (
                <span key={v}>{v}</span>
              ))}
            </div>
            <p style={{ fontSize: 9, margin: '13px 0' }}>
              Cleaned, summarized, and grounded answers ready for display.
            </p>
            <Link
              href="/knowledge"
              style={{
                fontSize: 10,
                color: 'var(--teal)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              Browse the knowledge base <ChevronRight size={14} />
            </Link>
          </section>
        </aside>
      </div>

      {/* ── Quick Links ── */}
      <div className="quick-links" style={{ marginTop: 28 }}>
        <Link href="/dtc-lookup">
          <Search size={19} />
          <div>
            <strong>Look up a DTC code</strong>
            <p>Diagnostic Trouble Codes for EV two-wheelers</p>
          </div>
          <ArrowRight size={17} />
        </Link>
        <Link href="/knowledge">
          <BookOpen size={19} />
          <div>
            <strong>Browse technical knowledge</strong>
            <p>Pre-processed answers, evidence, and summaries</p>
          </div>
          <ArrowRight size={17} />
        </Link>
      </div>
    </>
  );
}
