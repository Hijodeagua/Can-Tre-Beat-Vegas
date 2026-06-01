// Copies the cron-generated picks JSON from the repo-root picks/ directory into
// web/public/picks/ so the app can be deployed with its Vercel root set to web/.
// Runs automatically before `next build` via the "prebuild" npm script.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(here, '..');
const repoRoot = path.resolve(webRoot, '..');

const srcDir = path.join(repoRoot, 'picks');
const destDir = path.join(webRoot, 'public', 'picks');

const FILES = ['picks_today.json', 'record.json'];

fs.mkdirSync(destDir, { recursive: true });

for (const file of FILES) {
  const src = path.join(srcDir, file);
  const dest = path.join(destDir, file);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dest);
    console.log(`copied ${file} -> public/picks/`);
  } else {
    // Write a valid empty payload so the build and empty-state UI both work.
    const empty =
      file === 'picks_today.json'
        ? { generated_at: '', num_games: 0, picks: [] }
        : { picks: [] };
    fs.writeFileSync(dest, JSON.stringify(empty, null, 2));
    console.log(`no ${file} found — wrote empty placeholder to public/picks/`);
  }
}
