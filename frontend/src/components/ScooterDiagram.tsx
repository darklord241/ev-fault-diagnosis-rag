'use client';

import Link from 'next/link';
import { ArrowUpRight, Battery, CircleDot, Cpu, Gauge, Zap } from 'lucide-react';

const parts = [
  { id: 'battery', label: 'Battery pack', note: 'BMS / cell health', x: 25, y: 58, icon: Battery, href: '/diagnose?query=Battery%20not%20charging&subsystem=Battery%2FBMS' },
  { id: 'controller', label: 'Controller', note: 'Power delivery', x: 50, y: 68, icon: Cpu, href: '/diagnose?query=Motor%20controller%20fault&subsystem=Motor%20Controller' },
  { id: 'motor', label: 'Hub motor', note: 'Torque / Hall sensors', x: 79, y: 66, icon: Gauge, href: '/diagnose?query=Motor%20cuts%20out%20under%20load&subsystem=Motor%20Controller' },
  { id: 'dash', label: 'Dashboard', note: 'Codes / telemetry', x: 54, y: 25, icon: CircleDot, href: '/dtc-lookup' },
];

export function ScooterDiagram() {
  return (
    <section className="diagram-card">
      <div className="diagram-heading"><div><div className="eyebrow light">SYSTEM MAP / 01</div><h2>Trace the machine.</h2><p>Select a zone to start with the right language.</p></div><Zap size={22} className="diagram-bolt" /></div>
      <div className="scooter-map">
        <svg viewBox="0 0 920 360" role="img" aria-label="Electric scooter system diagram">
          <defs><linearGradient id="frame" x1="0" x2="1"><stop stopColor="#d7ff63" /><stop offset="1" stopColor="#79a2ff" /></linearGradient><filter id="glow"><feGaussianBlur stdDeviation="7" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
          <path className="map-wire faint" d="M170 275 C255 215 302 151 448 122 C565 98 654 126 774 258" />
          <path className="map-wire" d="M170 275 C255 215 302 151 448 122 C565 98 654 126 774 258" />
          <path className="map-frame" d="M240 266 L365 139 L547 139 L680 266 M365 139 L430 266 L547 139 M430 266 L615 266" />
          <path className="map-frame thin" d="M547 139 L585 89 L624 89 L650 142 M585 89 L558 54" />
          <circle className="map-wheel" cx="170" cy="275" r="52" /><circle className="map-wheel" cx="774" cy="275" r="52" />
          <circle className="map-hub" cx="774" cy="275" r="19" /><circle className="map-hub" cx="170" cy="275" r="10" />
          <rect className="map-battery" x="295" y="188" width="100" height="51" rx="8" /><path className="map-seat" d="M340 132 Q385 112 435 132" />
          <path className="map-wire accent" d="M346 188 L346 151 L497 151 L497 139" />
          <path className="map-wire accent-two" d="M395 213 L485 213 L485 159 L680 159" />
          <circle className="map-signal" cx="346" cy="188" r="5" filter="url(#glow)" /><circle className="map-signal blue" cx="497" cy="139" r="5" filter="url(#glow)" />
        </svg>
        {parts.map(({ id, label, note, x, y, icon: Icon, href }) => <Link key={id} href={href} className={`part-pin pin-${id}`} style={{ left: `${x}%`, top: `${y}%` }}><span className="part-icon"><Icon size={15} /></span><span><strong>{label}</strong><small>{note}</small></span><ArrowUpRight size={13} /></Link>)}
      </div>
      <div className="diagram-legend"><span><i className="legend-lime" /> power path</span><span><i className="legend-blue" /> signal path</span><span>4 clickable zones</span></div>
    </section>
  );
}
