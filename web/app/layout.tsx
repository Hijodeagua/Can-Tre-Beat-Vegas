import type { Metadata } from 'next';
import { Inter, Press_Start_2P } from 'next/font/google';
import Nav from '@/app/components/Nav';
import { THEME } from '@/app/lib/theme';
import './globals.css';

/*
 * Press Start 2P for the wordmark, short headings, table headers and big
 * numbers; Inter for everything else. Self-hosted by next/font at build time,
 * so there is no render-blocking request to Google and no swap flash.
 */
const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-app',
});

const pressStart = Press_Start_2P({
  subsets: ['latin'],
  weight: '400',
  display: 'swap',
  variable: '--font-pixel',
});

export const metadata: Metadata = {
  title: 'Can Tre Beat Vegas — whosyurgoat',
  description:
    'Every forecasting model I run on one board: whichever sport is in season sits on top, with its slate, its track record, and the rest waiting their turn.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${pressStart.variable}`}
      data-theme={THEME === 'techno' ? 'techno' : undefined}
    >
      <body>
        <header
          className="border-b"
          style={{
            // The green rule is the site's own accent, not a sport's — it does
            // not re-tint with the section below it.
            borderTop: '5px solid var(--accent-vegas)',
            borderBottomColor: 'var(--th-border)',
            background: 'var(--th-bar)',
          }}
        >
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-4">
            <div>
              <h1
                className="pixel m-0 text-[12px] leading-[1.5]"
                style={{ color: 'var(--th-ink)' }}
              >
                CAN TRE BEAT{' '}
                <span style={{ color: 'var(--th-accent-text)' }}>VEGAS?</span> 🎰
              </h1>
              {/* The disclaimer kicker stays in the header on every route. */}
              <p
                className="mt-2 text-[12px] uppercase tracking-wide"
                style={{ color: 'var(--th-faint)' }}
              >
                Odds tracker · not betting advice
              </p>
            </div>
            <Nav />
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 pb-16 pt-8">{children}</main>
      </body>
    </html>
  );
}
