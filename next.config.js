/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },
  // Enable experimental features if needed
  experimental: {
    // Add experimental features here if needed
  },
  // Add Babel configuration here
  webpack: (config, { isServer, dev }) => {
    // Keep the existing config
    const existingConfig = { ...config };

    // Only use babel for testing (not for normal builds)
    if (process.env.NODE_ENV === 'test') {
      // Add babel plugins for testing environment only
      config.module.rules.push({
        test: /\.(js|jsx|ts|tsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['next/babel'],
            plugins: [
              '@babel/plugin-transform-private-methods',
              '@babel/plugin-transform-private-property-in-object',
              '@babel/plugin-transform-class-properties'
            ]
          }
        }
      });
    }

    return existingConfig;
  },
  // Add rewrites for development environment to route API requests to the Python backend
  async rewrites() {
    const isProd = process.env.NODE_ENV === 'production';
    
    // In production, we'll use Vercel's routing in vercel.json
    if (isProd) {
      return [];
    }

    // In development, proxy to the Flask backend
    return [
      {
        source: '/api/task/:path*',
        destination: 'http://127.0.0.1:5000/api/task/:path*', // Flask server port
      }
    ];
  },
}

module.exports = nextConfig 