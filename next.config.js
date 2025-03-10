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
}

module.exports = nextConfig 