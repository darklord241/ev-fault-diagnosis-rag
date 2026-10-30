'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { ArrowRight, BookOpen, Search, ShieldCheck, Sparkles, Wrench } from 'lucide-react';
import { PageTitle, StatusStrip } from '@/components/Shell';
import { ScooterDiagram } from '@/components/ScooterDiagram';

const examples = ['Battery will not charge', 'Motor cuts out under load', 'Dashboard shows E-07'];

export default function Home() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [vehicle, setVehicle] = useState('');
  const [subsystem, setSubsystem] = useState('');
  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (query.trim().length < 3) return;
    const params = new URLSearchParams({ query: query.trim() });
    if (vehicle) params.set('vehicle', vehicle);
    if (subsystem) params.set('subsystem', subsystem);
    router.push(`/diagnose?${params.toString()}`);
  }
  return <>
    <PageTitle eyebrow="ELECTRIC TWO-WHEELER / WORKBENCH" title={<>Find the fault.<br /><em>Keep the signal.</em></>} description="A grounded diagnostic workspace for scooters, batteries, controllers, and the people who keep them moving." />
    <StatusStrip />
    <div className="hero-grid">
      <section className="hero-intro">
        <div className="hero-kicker"><span className="pulse-ring" /> READY FOR A CASE</div>
        <h2>Start with what<br />the scooter <span>tells you.</span></h2>
        <p>Describe the symptom in plain language. Voltroom searches service evidence, checks the signal against known faults, and gives you a procedure you can actually follow.</p>
        <form className="case-form" onSubmit={submit}>
          <label htmlFor="case">What are you seeing?</label>
          <textarea id="case" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Battery drops from 60% to 20% under acceleration…" rows={3} />
          <div className="form-row"><select value={vehicle} onChange={(e) => setVehicle(e.target.value)}><option value="">Vehicle / platform</option><option>Ather 450X</option><option>Ather 450S</option><option>Ola S1 Pro</option><option>TVS iQube S</option><option>Generic BLDC EV</option></select><select value={subsystem} onChange={(e) => setSubsystem(e.target.value)}><option value="">System area</option><option>Battery/BMS</option><option>Motor Controller</option><option>Throttle</option><option>Charger/Charging</option><option>Wiring/Harness</option><option>Dashboard/ECU</option></select></div>
          <button className="button primary wide" disabled={query.trim().length < 3}><Wrench size={16} /> Open diagnostic case <ArrowRight size={16} /></button>
        </form>
        <div className="example-row"><span>Try a case</span>{examples.map((item) => <button key={item} onClick={() => setQuery(item)}>{item}</button>)}</div>
      </section>
      <aside className="hero-aside"><div className="aside-index">01 <span>/</span> 04</div><p className="aside-quote">“The fastest path to a repair is a well-framed question.”</p><div className="aside-rule" /><div className="aside-facts"><div><strong>01</strong><span>Describe the symptom</span></div><div><strong>02</strong><span>Retrieve the evidence</span></div><div><strong>03</strong><span>Verify the next step</span></div></div><Link href="/knowledge" className="text-link">Explore the evidence library <ArrowRight size={14} /></Link></aside>
    </div>
    <ScooterDiagram />
    <div className="home-bottom-grid"><Link className="feature-link" href="/dtc-lookup"><span className="feature-icon"><Search size={18} /></span><span><small>REFERENCE / 02</small><strong>DTC index</strong><em>Decode a dashboard code before you touch the loom.</em></span><ArrowRight size={17} /></Link><Link className="feature-link" href="/knowledge"><span className="feature-icon blue"><BookOpen size={18} /></span><span><small>REFERENCE / 03</small><strong>Evidence library</strong><em>Inspect the exact passages behind an answer.</em></span><ArrowRight size={17} /></Link><div className="trust-note"><ShieldCheck size={19} /><span><strong>Built to show its work.</strong><br />No black-box certainty. Sources, confidence, and safety notes stay visible.</span><Sparkles size={16} /></div></div>
  </>;
}
