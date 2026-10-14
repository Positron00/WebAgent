import Chat from '@/components/Chat';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import LogoMark from '@/components/LogoMark';

export default function Home() {
  return (
    <main
      className="min-h-screen flex items-center justify-center font-sans antialiased"
      style={{ backgroundColor: '#191919', color: '#e5e5e5' }}
    >
      <div className="w-[75%] flex flex-col items-center gap-6">
        {/* Logo and Tagline */}
        <div className="flex flex-col items-center gap-4">
          <LogoMark />
          
          <div className="flex items-start gap-1">
            {/* Same face as the chat (Inter), opened up with letter-spacing for a quiet headline */}
            <h1 className="font-sans text-2xl sm:text-3xl font-semibold tracking-wide text-[#c4c4c4] leading-tight py-1">
              Intelligence at your fingertips
            </h1>
            <span className="text-xs text-[#8a8a8a] mt-1">©</span>
          </div>
        </div>

        {/* Chat Interface */}
        <ErrorBoundary>
          <Chat />
        </ErrorBoundary>
      </div>
    </main>
  );
}
