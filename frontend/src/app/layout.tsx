import type { Metadata } from 'next';
import './globals.css';
import { Shell } from '@/components/Shell';

export const metadata: Metadata = {
  title: 'EV Fault Diagnosis — AI-Powered Technical Intelligence',
  description:
    'Evidence-grounded electric two-wheeler fault diagnosis using RAG and MCP-based retrieval.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
