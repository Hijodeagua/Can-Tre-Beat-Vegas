import type { AppProps } from 'next/app';
import Link from 'next/link';
import '@/styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-900">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-4">
          <div>
            <h1 className="text-lg font-bold">Can Tre Beat Vegas?</h1>
            <p className="text-xs uppercase tracking-wide text-slate-500">
              NFL model picks vs. the spread
            </p>
          </div>
          <nav className="flex gap-3 text-sm">
            <Link href="/" className="rounded px-2 py-1 text-slate-300 hover:bg-slate-800">
              Today&apos;s Picks
            </Link>
            <Link href="/record" className="rounded px-2 py-1 text-slate-300 hover:bg-slate-800">
              Season Record
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-4xl px-4 py-8">
        <Component {...pageProps} />
      </main>
    </div>
  );
}
