import type { Metadata } from 'next';
import { fmtTimestamp } from '@/app/lib/format';
import { getEmailIndex } from '@/app/lib/emails';
import EmailArchive from './EmailArchive';

export const metadata: Metadata = {
  title: 'Email archive — Can Tre Beat Vegas',
  description:
    'Every report email the models have sent — MLB, soccer, college football, NFL and the ' +
    'weekly models check — sortable by league and date.',
};

/**
 * The archive of every email the pipelines have sent. The list itself is a
 * client component (it filters and sorts); this page only frames it.
 */
export default function EmailsPage() {
  const index = getEmailIndex();

  return (
    <div>
      <h2 className="pixel m-0 text-[18px] leading-[1.4]" style={{ color: 'var(--th-ink)' }}>
        EMAIL{' '}
        <span
          className="px-[6px] py-[2px]"
          style={{ background: 'var(--th-highlight)', color: 'var(--th-highlight-ink)' }}
        >
          ARCHIVE
        </span>
      </h2>

      <p
        className="mt-4 max-w-[640px] text-[14px] leading-normal"
        style={{ color: 'var(--th-muted)', textWrap: 'pretty' } as React.CSSProperties}
      >
        Every report email the models have sent, as it went out. Each pipeline archives its
        email the morning it renders it, so a slate or a set of picks here is the version
        written before the games were played. Pick a league to narrow the list; click a
        header to re-sort.
      </p>

      <EmailArchive />

      {index.generated_at && (
        <p className="mt-8 text-[12px]" style={{ color: 'var(--th-faint)' }}>
          Index generated {fmtTimestamp(index.generated_at)} · {index.emails.length} emails.
        </p>
      )}
    </div>
  );
}
