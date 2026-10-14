'use client';

import Image from 'next/image';
import { useChat } from '@/contexts/ChatContext';

/** The page logo; it pulses while an answer is on its way (from submit until the reply or an error). */
export default function LogoMark() {
  const { state } = useChat();

  return (
    <Image
      src="/icons/icon-512-transparent.svg"
      alt="WebChat Logo"
      width={144}
      height={144}
      className={`grayscale opacity-75 ${state.isLoading ? 'logo-thinking' : ''}`}
      aria-busy={state.isLoading}
    />
  );
}
