const isProd = process.env.NODE_ENV === "production";

// Next.js 14 bootstraps with inline scripts and we don't use nonces, so scripts need
// 'unsafe-inline'; dev mode (React Refresh) also needs 'unsafe-eval'.
const contentSecurityPolicy = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${isProd ? "" : " 'unsafe-eval'"}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self'",
  "media-src 'self' blob:",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: contentSecurityPolicy },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  ...(isProd
    ? [{ key: "Strict-Transport-Security", value: "max-age=31536000" }]
    : []),
];

// WEBAGENT_IMAGE_HOSTS: comma-separated hosts next/image may load over https; empty means none.
const imageRemotePatterns = (process.env.WEBAGENT_IMAGE_HOSTS || "")
  .split(",")
  .map((host) => host.trim())
  .filter(Boolean)
  .map((hostname) => ({ protocol: "https", hostname }));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: imageRemotePatterns,
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
  // Add Babel configuration here
  webpack: (config, { isServer, dev }) => {
    // Keep the existing config
    const existingConfig = { ...config };

    // Only use babel for testing (not for normal builds)
    if (process.env.NODE_ENV === "test") {
      // Add babel plugins for testing environment only
      config.module.rules.push({
        test: /\.(js|jsx|ts|tsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: ["next/babel"],
            plugins: [
              "@babel/plugin-transform-private-methods",
              "@babel/plugin-transform-private-property-in-object",
              "@babel/plugin-transform-class-properties",
            ],
          },
        },
      });
    }

    return existingConfig;
  },
  // /api/task is proxied to the FastAPI backend by src/app/api/task (WEBAGENT_BACKEND_URL,
  // default http://127.0.0.1:8000), in both development and production.
};

module.exports = nextConfig;
