import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { AppProvider } from '@/contexts/AppContext';
import { ChatProvider } from '@/contexts/ChatContext';

/**
 * Custom render function that wraps components with necessary providers
 * for testing purposes.
 */
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <AppProvider>
      <ChatProvider>
        {children}
      </ChatProvider>
    </AppProvider>
  );
};

/**
 * Custom render method that includes providers
 * @param ui Component to render
 * @param options Additional render options
 * @returns The rendered component with testing utilities
 */
const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options });

// Re-export everything from testing-library
export * from '@testing-library/react';

// Override render method
export { customRender as render }; 