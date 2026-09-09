/**
 * The email archive: every report email each pipeline has sent, copied
 * to `public/emails/<league>/<date>_<type>.html` by
 * `data_jobs/email_archive.py` and listed in `public/emails/index.json`.
 * Imported at build time like the rest of the static data, so the archive
 * page redeploys with each pipeline commit.
 */
import index from '@/public/emails/index.json';

export interface ArchivedEmail {
  /** League key: mlb, soccer, cfb, nfl, or models (the weekly check). */
  league: string;
  /** Email type within the league: futures, slate, grade, update, models. */
  type: string;
  /** The email's content date (the graded day for an MLB grade email). */
  date: string;
  subject: string;
  /** Repo-relative path of the sent HTML. */
  path: string;
  /** Site-relative path under the /vegas basePath. */
  public_path: string;
  /** Absolute permalink. */
  url: string;
  archived_at: string;
}

export interface EmailIndex {
  generated_at: string | null;
  leagues: Record<string, string>;
  emails: ArchivedEmail[];
}

const data = index as unknown as EmailIndex;

export function getEmailIndex(): EmailIndex {
  return data;
}

/** Display name for a league key, falling back to the key itself. */
export function leagueName(key: string): string {
  return data.leagues?.[key] ?? key;
}

/** "update" -> "Update", "models" -> "Models check". */
export function typeLabel(type: string): string {
  if (type === 'models') return 'Models check';
  return type.charAt(0).toUpperCase() + type.slice(1);
}
