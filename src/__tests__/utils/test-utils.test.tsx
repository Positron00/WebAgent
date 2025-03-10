import React from 'react';
import { render, screen, useApp, useChat } from '@/test-utils';

// Simple test component that uses the AppContext
const TestAppComponent = () => {
  const { theme } = useApp();
  return <div>Current theme: {theme}</div>;
};

// Simple test component that uses the ChatContext
const TestChatComponent = () => {
  const { state } = useChat();
  return <div>Loading: {state.isLoading ? 'yes' : 'no'}</div>;
};

describe('Test Utilities', () => {
  it('provides AppContext to components', () => {
    render(<TestAppComponent />);
    expect(screen.getByText('Current theme: dark')).toBeInTheDocument();
  });

  it('provides ChatContext to components', () => {
    render(<TestChatComponent />);
    expect(screen.getByText('Loading: no')).toBeInTheDocument();
  });
}); 