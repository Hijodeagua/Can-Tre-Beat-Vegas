'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { NAV_COLLAPSE_AT, NAV_SPORTS } from '@/app/lib/sports';

/**
 * Header nav, built from the models array rather than a hand-written list.
 *
 * Flat links are right at three sports. At four or more the sports collapse
 * behind a `Sports ▾` dropdown instead of the row growing — which is a
 * rendering decision here, so adding the fourth sport stays a data change.
 */

interface NavLink {
  href: string;
  label: string;
}

const HOME: NavLink = { href: '/', label: 'Home' };
const SOURCES: NavLink = { href: '/methodology', label: 'Data sources' };

const pillBase = 'rounded-full px-3 py-1 text-[14px] no-underline transition-none';

function pillStyle(active: boolean): React.CSSProperties {
  return active
    ? { background: 'var(--th-ink)', color: '#ffffff' }
    : { color: 'var(--th-muted)' };
}

function Pill({ link, active }: { link: NavLink; active: boolean }) {
  return (
    <Link
      href={link.href}
      aria-current={active ? 'page' : undefined}
      className={`${pillBase} ${active ? 'font-semibold' : 'hover:bg-slate-100'}`}
      style={pillStyle(active)}
    >
      {link.label}
    </Link>
  );
}

export default function Nav() {
  const pathname = usePathname() ?? '/';
  const sportLinks: NavLink[] = NAV_SPORTS.map((s) => ({
    href: s.href as string,
    label: s.navLabel as string,
  }));
  const collapse = sportLinks.length >= NAV_COLLAPSE_AT;
  const isActive = (href: string) =>
    href === '/' ? pathname === '/' : pathname.startsWith(href);

  return (
    <nav className="flex flex-wrap items-center gap-2">
      <Pill link={HOME} active={isActive(HOME.href)} />

      {collapse ? (
        // <details> keeps the dropdown working without client state or any
        // animation, which the Slate layer does not use anyway.
        <details className="relative">
          <summary
            className={`${pillBase} cursor-pointer list-none hover:bg-slate-100`}
            style={pillStyle(sportLinks.some((l) => isActive(l.href)))}
          >
            Sports ▾
          </summary>
          <div
            className="absolute right-0 z-10 mt-1 flex min-w-[160px] flex-col gap-1 rounded-md border p-1"
            style={{ background: 'var(--th-card)', borderColor: 'var(--th-border)' }}
          >
            {sportLinks.map((link) => (
              <Pill key={link.href} link={link} active={isActive(link.href)} />
            ))}
          </div>
        </details>
      ) : (
        sportLinks.map((link) => (
          <Pill key={link.href} link={link} active={isActive(link.href)} />
        ))
      )}

      <Pill link={SOURCES} active={isActive(SOURCES.href)} />
    </nav>
  );
}
