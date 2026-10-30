'use client';

import Link from 'next/link';
import { ArrowUpRight, Battery, Cpu, Gauge, Radio, Zap } from 'lucide-react';

const nodes = [
  { title: 'HV BATTERY', sub: 'BMS / CELL GROUPS', value: '58.4 V', state: 'NOMINAL', cls: 'node-green', icon: Battery, href: '/diagnose?query=Battery%20not%20charging&subsystem=Battery%2FBMS' },
  { title: 'VCU / ECU', sub: 'CONTROL LOGIC', value: 'LINKED', state: 'NOMINAL', cls: 'node-green', icon: Cpu, href: '/diagnose?query=Vehicle%20control%20unit%20fault&subsystem=Dashboard%2FECU' },
  { title: 'INVERTER', sub: 'MOTOR CONTROL', value: '42.1 °C', state: 'NOMINAL', cls: 'node-amber', icon: Zap, href: '/diagnose?query=Motor%20controller%20fault&subsystem=Motor%20Controller' },
  { title: 'HUB MOTOR', sub: 'HALL / PHASE', value: '0 RPM', state: 'STANDBY', cls: 'node-blue', icon: Gauge, href: '/diagnose?query=Motor%20cuts%20out%20under%20load&subsystem=Motor%20Controller' },
  { title: 'DASH / CAN', sub: 'FAULT TELEMETRY', value: 'READY', state: 'NOMINAL', cls: 'node-green', icon: Radio, href: '/dtc-lookup' },
];

export function ScooterDiagram() {
  return <section className="topology-panel"><div className="panel-title"><div><span className="eyebrow">TOPOLOGY / POWERTRAIN</span><h2>Vehicle systems map</h2></div><span className="panel-tag">LIVE VIEW / 01</span></div><div className="topology-canvas"><svg viewBox="0 0 900 260" aria-hidden="true"><defs><filter id="line-glow"><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs><path className="topology-line dim" d="M92 130 H795" /><path className="topology-line power" d="M140 130 H350 M490 130 H700" filter="url(#line-glow)" /><path className="topology-line signal" d="M350 130 V68 H700 V130 M350 130 V202 H700 V130" /><circle className="line-node" cx="350" cy="130" r="5" /><circle className="line-node blue-node" cx="700" cy="130" r="5" /><path className="flow-arrow" d="M266 123 l12 7 -12 7 M613 123 l12 7 -12 7" /></svg>{nodes.map(({ title, sub, value, state, cls, icon: Icon, href }) => <Link href={href} key={title} className={`topology-node ${cls}`}><div className="node-top"><Icon size={15} /><span>{state}</span></div><strong>{title}</strong><small>{sub}</small><b>{value}</b><ArrowUpRight size={13} className="node-arrow" /></Link>)}</div><div className="topology-footer"><span><i className="legend-power" /> HIGH-VOLTAGE POWER PATH</span><span><i className="legend-signal" /> LOW-VOLTAGE DATA BUS</span><span className="topology-note">Click a node to open a guided case</span></div></section>;
}
