/** @type {import('next').NextConfig} */
const orchestratorApiBaseUrl =
  process.env.ORCHESTRATOR_API_BASE_URL || 'http://127.0.0.1:8080';

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${orchestratorApiBaseUrl}/api/v1/:path*`,
      },
    ];
  },
};
export default nextConfig;
