/** @type {import('next').NextConfig} */
const backendApiBaseUrl =
  process.env.BACKEND_API_BASE_URL || 'http://127.0.0.1:8000';

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${backendApiBaseUrl}/api/v1/:path*`,
      },
    ];
  },
};
export default nextConfig;
