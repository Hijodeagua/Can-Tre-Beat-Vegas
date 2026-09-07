import type { Metadata } from 'next';
import { fileUrl, MODEL_DOCS, REPO_URL, SHARED_DOCS, type DocStep } from '@/app/lib/modelDocs';
import { accentVars, sportByKey } from '@/app/lib/sports';

export const metadata: Metadata = {
  title: 'How the models work — Can Tre Beat Vegas',
  description:
    'Each model on the board, step by step — data, rating engine, tuning, score model, grading, ' +
    'simulation, publishing — with a link to the file that does each step.',
};

/**
 * The map of the repo, organised by model rather than by folder: for each
 * model, the steps from data to graded pick, each naming the file that
 * does it. The READMEs explain why; this page says where.
 */

function FileLinks({ files }: { files: string[] }) {
  return (
    <ul className="m-0 mt-2 flex list-none flex-wrap gap-x-4 gap-y-1 p-0">
      {files.map((f) => (
        <li key={f}>
          <a
            href={fileUrl(f)}
            target="_blank"
            rel="noreferrer"
            className="text-[12px] underline-offset-2 hover:underline"
            style={{ color: 'var(--th-muted)', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
          >
            {f}
          </a>
        </li>
      ))}
    </ul>
  );
}

function Steps({ steps }: { steps: DocStep[] }) {
  return (
    <ol className="m-0 mt-4 grid list-none gap-3 p-0">
      {steps.map((s) => (
        <li
          key={s.title}
          className="rounded-lg border p-4"
          style={{ borderColor: 'var(--th-border)', background: 'var(--th-card)' }}
        >
          <div className="pixel text-[9px]" style={{ color: 'var(--th-ink)' }}>
            {s.title.toUpperCase()}
          </div>
          <p className="mt-2 text-[14px] leading-normal" style={{ color: 'var(--th-muted)' }}>
            {s.detail}
          </p>
          <FileLinks files={s.files} />
        </li>
      ))}
    </ol>
  );
}

export default function ModelsPage() {
  return (
    <div>
      <h2 className="pixel m-0 text-[18px] leading-[1.4]" style={{ color: 'var(--th-ink)' }}>
        HOW THE{' '}
        <span
          className="px-[6px] py-[2px]"
          style={{ background: 'var(--th-highlight)', color: 'var(--th-highlight-ink)' }}
        >
          MODELS WORK
        </span>
      </h2>

      <p
        className="mt-4 max-w-[640px] text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        Each model on the board, from its data source to its graded pick, one step at a time.
        Every step links to the file in the{' '}
        <a href={REPO_URL} target="_blank" rel="noreferrer" className="underline-offset-2 hover:underline" style={{ color: 'var(--th-ink)' }}>
          repository
        </a>{' '}
        that does it, so the code can be read in the order it runs. The README beside each
        model explains the reasoning; this page is the map.
      </p>

      <nav className="mt-5 flex flex-wrap gap-2">
        {MODEL_DOCS.map((m) => (
          <a
            key={m.key}
            href={`#${m.key}`}
            className="rounded-full px-3 py-1 text-[13px] no-underline hover:bg-slate-100"
            style={{ color: 'var(--th-muted)', border: '1px solid var(--th-border)' }}
          >
            {m.emoji} {m.name}
          </a>
        ))}
      </nav>

      {MODEL_DOCS.map((m) => {
        const sport = sportByKey(m.key);
        const vars = sport ? accentVars(sport) : accentVars({ accent: '#ffd23f', accentInk: '#06120b' });
        return (
          <section key={m.key} id={m.key} className="mt-10" style={vars}>
            <div
              className="flex flex-wrap items-center justify-between gap-3 rounded-lg px-4 py-3"
              style={{ background: 'var(--sport-accent)' }}
            >
              <h3 className="pixel m-0 text-[12px]" style={{ color: 'var(--sport-accent-ink)' }}>
                {m.emoji} {m.name.toUpperCase()}
              </h3>
              <a
                href={fileUrl(m.readme)}
                target="_blank"
                rel="noreferrer"
                className="pixel text-[8px] tracking-[0.08em] no-underline hover:underline"
                style={{ color: 'var(--sport-accent-ink)' }}
              >
                README ↗
              </a>
            </div>
            <p className="mt-4 text-[14px] leading-normal" style={{ color: 'var(--th-muted)' }}>
              {m.summary}
            </p>
            <dl className="mt-3 grid gap-x-6 gap-y-1 text-[13px] sm:grid-cols-2">
              {m.facts.map((f) => (
                <div key={f.label} className="flex gap-2">
                  <dt className="shrink-0 font-semibold" style={{ color: 'var(--th-ink)' }}>
                    {f.label}
                  </dt>
                  <dd className="m-0" style={{ color: 'var(--th-muted)' }}>{f.value}</dd>
                </div>
              ))}
            </dl>
            <Steps steps={m.steps} />
          </section>
        );
      })}

      <section className="mt-10">
        <h3 className="pixel m-0 text-[11px]" style={{ color: 'var(--th-ink)' }}>
          SHARED PLUMBING
        </h3>
        <Steps steps={SHARED_DOCS} />
      </section>
    </div>
  );
}
