import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppProvider } from "@/contexts/AppContext";
import { ChatProvider } from "@/contexts/ChatContext";
import BrowserCompatibilityWarning from "@/components/BrowserCompatibilityWarningClient";
import OfflineBanner from "@/components/OfflineBanner";

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
              <OfflineBanner />
              {children}
            </div>
          </ChatProvider>
        </AppProvider>
      </body>
    </html>
  );
}
