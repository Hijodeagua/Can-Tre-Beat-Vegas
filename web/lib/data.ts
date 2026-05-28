// Static-data access for getStaticProps. Reads the JSON the GitHub Actions cron
// commits to the repo-root picks/ directory. We walk up from cwd to find it so
// the build works whether the Vercel project root is the repo root or web/.

import fs from 'node:fs';
import path from 'node:path';

export interface Pick {
  game_id: string;
  home_team: string;
  away_team: string;
  game_time: string;
  vegas_spread: number;
  model_win_prob: number;
  model_ats_prob: number;
  model_lean: 'cover' | 'fade' | 'push';
}

export interface PicksToday {
  generated_at: string;
  num_games: number;
  picks: Pick[];
}

export interface RecordEntry {
  pick_date: string;
  game_id: string;
  home_team: string;
  away_team: string;
  vegas_spread: number;
  model_ats_prob: number | null;
  model_lean: string;
  pick: string;
  result: 'win' | 'loss' | 'push' | 'no_bet' | null;
  correct: boolean | null;
  final_score?: string;
}

export interface RecordFile {
  updated_at?: string;
  total_picks?: number;
  graded_picks?: number;
  picks: RecordEntry[];
}

function findPicksDir(): string | null {
  // Preferred: the copy made by scripts/copy-data.mjs (prebuild), which works
  // when the Vercel project root is web/.
  const bundled = path.join(process.cwd(), 'public', 'picks');
  if (fs.existsSync(bundled)) return bundled;

  // Fallback: walk up to the repo-root picks/ dir (root = repo root).
  let dir = process.cwd();
  for (let i = 0; i < 6; i += 1) {
    const candidate = path.join(dir, 'picks');
    if (fs.existsSync(candidate)) return candidate;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

function read<T>(file: string, fallback: T): T {
  const picksDir = findPicksDir();
  if (!picksDir) return fallback;
  try {
    return JSON.parse(fs.readFileSync(path.join(picksDir, file), 'utf-8')) as T;
  } catch {
    return fallback;
  }
}

export function getPicksToday(): PicksToday {
  return read<PicksToday>('picks_today.json', {
    generated_at: '',
    num_games: 0,
    picks: [],
  });
}

export function getRecord(): RecordFile {
  return read<RecordFile>('record.json', { picks: [] });
}
