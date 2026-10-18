import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppProvider } from "@/contexts/AppContext";
import { ChatProvider } from "@/contexts/ChatContext";
import BrowserCompatibilityWarning from "@/components/BrowserCompatibilityWarningClient";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "WebAgent",
  description: "A multi-agent research assistant powered by Together AI",
  manifest: "/manifest.json",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={inter.className}>
        <AppProvider>
          <ChatProvider>
            <div className="min-h-screen bg-[#191919] transition-colors">
              <BrowserCompatibilityWarning />
              <div
                id="offline-banner"
                className="fixed top-0 left-0 right-0 bg-yellow-500 text-black p-2 text-center transform transition-transform duration-300 ease-in-out translate-y-[-100%] data-[visible=true]:translate-y-0"
                data-visible="false"
                role="alert"
                aria-live="polite"
              >
                You are currently offline. Some features may be limited.
              </div>
              {children}
            </div>
          </ChatProvider>
        </AppProvider>
      </body>
    </html>
  );
}
