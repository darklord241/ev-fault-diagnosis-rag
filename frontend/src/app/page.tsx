'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { ArrowRight, BookOpen, ClipboardList, FileWarning, Search, ShieldAlert, Stethoscope } from 'lucide-react';
import { PageTitle, StatusStrip } from '@/components/Shell';
import { ScooterDiagram } from '@/components/ScooterDiagram';

const examples = ['Battery will not charge', 'Motor cuts out under load', 'Dashboard shows E-07'];

export default function Home() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [vehicle, setVehicle] = useState('');
  const [subsystem, setSubsystem] = useState('');
  function submit(event: React.FormEvent) { event.preventDefault(); if (query.trim().length < 3) return; const params = new URLSearchParams({ query: query.trim() }); if (vehicle) params.set('vehicle', vehicle); if (subsystem) params.set('subsystem', subsystem); router.push(`/diagnose?${params.toString()}`); }
  return <>
    <PageTitle eyebrow="SHIFT A / CONTROL ROOM" title={<>Powertrain<br /><span className="title-accent">investigation console.</span></>} description="Observe the system. Isolate the signal. Move from symptom to evidence-backed next action without losing the trail." />
    <StatusStrip />
    <div className="command-grid">
      <section className="command-panel"><div className="panel-title"><div><span className="eyebrow">NEW INVESTIGATION</span><h2>Open a diagnostic case</h2></div><span className="case-number">CASE / AUTO</span></div><p className="panel-lead">Enter the observed symptom exactly as reported. Codes, warning lights, operating conditions, and timing are useful signals.</p><form onSubmit={submit}><label htmlFor="case">OBSERVATION LOG</label><textarea id="case" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Example: battery charge stops at 42%, charger LED flashes red after 20 minutes…" rows={4} /><div className="form-grid"><div><label htmlFor="vehicle">PLATFORM</label><select id="vehicle" value={vehicle} onChange={(e) => setVehicle(e.target.value)}><option value="">ALL PLATFORMS</option><option>Ather 450X</option><option>Ather 450S</option><option>Ola S1 Pro</option><option>TVS iQube S</option><option>Generic BLDC EV</option></select></div><div><label htmlFor="subsystem">TARGET SYSTEM</label><select id="subsystem" value={subsystem} onChange={(e) => setSubsystem(e.target.value)}><option value="">AUTO-ROUTE</option><option>Battery/BMS</option><option>Motor Controller</option><option>Throttle</option><option>Charger/Charging</option><option>Wiring/Harness</option><option>Dashboard/ECU</option></select></div></div><button className="button primary wide" disabled={query.trim().length < 3}><Stethoscope size={15} /> START INVESTIGATION <ArrowRight size={15} /></button></form><div className="example-row"><span>RECENT SIGNALS</span>{examples.map((item) => <button key={item} onClick={() => setQuery(item)}>{item}</button>)}</div></section>
      <aside className="incident-panel"><div className="incident-top"><span className="eyebrow">SHIFT BRIEF / 00:01</span><ShieldAlert size={18} /></div><h2>Before touching<br /><span>high voltage.</span></h2><p>Confirm the vehicle is isolated, the key is removed, and the pack is safe to inspect. This console provides information — not permission to bypass safety procedures.</p><div className="incident-list"><div><b>01</b><span>Record the dashboard code</span></div><div><b>02</b><span>Capture when the fault appears</span></div><div><b>03</b><span>Follow the cited procedure</span></div></div><Link href="/dtc-lookup" className="text-link">OPEN FAULT CODE INDEX <ArrowRight size={14} /></Link></aside>
    </div>
    <ScooterDiagram />
    <div className="control-links"><Link href="/diagnose" className="control-link"><span className="control-icon green"><ClipboardList size={17} /></span><span><small>CONSOLE / 02</small><strong>Case queue</strong><em>Run a grounded investigation and review prior cases.</em></span><ArrowRight size={16} /></Link><Link href="/dtc-lookup" className="control-link"><span className="control-icon amber"><FileWarning size={17} /></span><span><small>REFERENCE / 03</small><strong>Fault code index</strong><em>Search EV-specific and generic OBD-II code records.</em></span><ArrowRight size={16} /></Link><Link href="/knowledge" className="control-link"><span className="control-icon blue"><BookOpen size={17} /></span><span><small>REFERENCE / 04</small><strong>Evidence store</strong><em>Inspect the retrieved passages behind a result.</em></span><ArrowRight size={16} /></Link></div>
  </>;
}
