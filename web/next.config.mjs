/** @type {import('next').NextConfig} */
const nextConfig = {
  // This spoke is proxied at whosyurgoat.app/vegas by the hub's vercel.json.
  basePath: '/vegas',
  assetPrefix: '/vegas',
};

export default nextConfig;
