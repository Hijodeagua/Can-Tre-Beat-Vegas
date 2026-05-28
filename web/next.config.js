/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxied at whosyurgoat.app/vegas by the hub's vercel.json.
  basePath: '/vegas',
  // Fully static site: picks are baked in at build time from the JSON the
  // GitHub Actions cron commits. No server runtime needed.
  output: 'export',
  images: { unoptimized: true },
};

module.exports = nextConfig;
