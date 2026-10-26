'use client';

import { useEffect, useRef, useState } from 'react';
import { ExternalLink, X, FileText, Layers } from 'lucide-react';
import type { EvidenceSource } from '@/lib/types';

export function EvidenceDrawer({
  source,
  onClose,
}: {
  source: EvidenceSource | null;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (source) {
      dialog.current?.showModal();
    } else {
      dialog.current?.close();
    }
  }, [source]);

  if (!source) return null;

  return (
    <dialog ref={dialog} className="evidence-drawer" onCancel={onClose}>
      <div className="drawer-heading">
        <div>
          <div className="eyebrow">EVIDENCE SOURCE</div>
          <h2>Inspect the evidence</h2>
        </div>
        <button className="icon-button" aria-label="Close evidence" onClick={onClose}>
          <X size={20} />
        </button>
      </div>

      <p className="muted">
        Source excerpt from technical documentation. Relevance scores reflect
        retrieval similarity, not diagnostic confidence.
      </p>

      <article style={{ marginTop: 24 }}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <span className="badge">{source.document_type}</span>
          {source.subsystem && <span className="badge">{source.subsystem}</span>}
        </div>

        <h3 style={{ marginBottom: 8 }}>{source.document_title}</h3>

        {source.section && (
          <p className="muted" style={{ fontSize: 11 }}>
            <Layers size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />
            Section: {source.section}
          </p>
        )}

        {source.page_number && (
          <p className="muted" style={{ fontSize: 11 }}>
            <FileText size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />
            Page {source.page_number}
          </p>
        )}

        {source.vehicle_model && (
          <p className="muted" style={{ fontSize: 11 }}>
            Vehicle: {source.vehicle_model}
          </p>
        )}

        <blockquote
          style={{
            background: '#f6f9f7',
            borderLeft: '3px solid #d6e6dc',
            padding: 17,
            margin: '15px 0',
            fontSize: 12,
            color: '#48634f',
            whiteSpace: 'pre-wrap',
            overflowWrap: 'anywhere',
          }}
        >
          {source.content}
        </blockquote>

        <p className="muted" style={{ fontSize: 11 }}>
          Relevance Score: {(source.relevance_score * 100).toFixed(1)}%
        </p>

        {typeof source.metadata?.url === 'string' && (
          <a
            className="source-link"
            href={source.metadata.url as string}
            target="_blank"
            rel="noopener noreferrer"
            style={{ marginTop: 12, display: 'inline-flex', gap: 7 }}
          >
            Original source <ExternalLink size={14} />
          </a>
        )}
      </article>
    </dialog>
  );
}
