import React, { ReactElement, createContext } from 'react';
import { render, RenderOptions } from '@testing-library/react';

// Create mock contexts
const MockAppContext = createContext(undefined);
const MockChatContext = createContext(undefined);

// Mock react-markdown
jest.mock('react-markdown', () => {
  return {
    __esModule: true,
    default: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
  };
});

// Mock heroicons
jest.mock('@heroicons/react/24/outline', () => {
  const MockIcon = () => <svg data-testid="mock-icon" />;
  return {
    InformationCircleIcon: MockIcon,
    NewspaperIcon: MockIcon,
    CodeBracketIcon: MockIcon,
    MicrophoneIcon: MockIcon,
    PaperClipIcon: MockIcon,
    ComputerDesktopIcon: MockIcon,
    SparklesIcon: MockIcon,
    PaperAirplaneIcon: MockIcon,
    Cog6ToothIcon: MockIcon,
    XMarkIcon: MockIcon,
    PencilSquareIcon: MockIcon,
    BeakerIcon: MockIcon,
    ArrowPathIcon: MockIcon,
    ChevronDownIcon: MockIcon,
    ChevronUpIcon: MockIcon,
    // Add any other icons used in your components
  };
});

// Define the mock accessibility object that will be used consistently
const mockAccessibility = {
  reducedMotion: false,
  highContrast: false,
  fontSize: 'normal',
  promptStyle: 'balanced',
  knowledgeFocus: 'general',
  citeSources: true,
  agentic: false,
  responseTextColor: '#FFFFFF',
  queryTextColor: '#FFFFFF',
  responseBackgroundColor: '#111827',
  queryBackgroundColor: '#1E3A8A'
};

// Mock the AppContext
jest.mock('@/contexts/AppContext', () => {
  return {
    AppProvider: ({ children }: { children: React.ReactNode }) => {
      return (
        <MockAppContext.Provider
          value={{
            theme: 'dark',
            setTheme: jest.fn(),
            isMenuOpen: false,
            setMenuOpen: jest.fn(),
            showNotification: jest.fn(),
            hideNotification: jest.fn(),
            showModal: jest.fn(),
            hideModal: jest.fn(),
            accessibility: mockAccessibility,
            setAccessibility: jest.fn(),
            isOffline: false
          }}
        >
          {children}
        </MockAppContext.Provider>
      );
    },
    useApp: () => ({
      theme: 'dark',
      setTheme: jest.fn(),
      isMenuOpen: false,
      setMenuOpen: jest.fn(),
      showNotification: jest.fn(),
      hideNotification: jest.fn(),
      showModal: jest.fn(),
      hideModal: jest.fn(),
      accessibility: mockAccessibility,
      setAccessibility: jest.fn(),
      isOffline: false
    })
  };
});

// Mock the ChatContext
jest.mock('@/contexts/ChatContext', () => {
  return {
    ChatProvider: ({ children }: { children: React.ReactNode }) => {
      return (
        <MockChatContext.Provider
          value={{
            state: {
              messages: [],
              isLoading: false
            },
            sendMessage: jest.fn().mockResolvedValue(undefined),
            clearMessages: jest.fn(),
            getMessageHistory: jest.fn().mockReturnValue([])
          }}
        >
          {children}
        </MockChatContext.Provider>
      );
    },
    useChat: () => ({
      state: {
        messages: [],
        isLoading: false
      },
      sendMessage: jest.fn().mockResolvedValue(undefined),
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    })
  };
});

/**
 * Custom render function that wraps components with necessary providers
 * for testing purposes.
 */
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  // Use our mock context providers instead of the actual implementations
  return (
    <MockAppContext.Provider
      value={{
        theme: 'dark',
        setTheme: jest.fn(),
        isMenuOpen: false,
        setMenuOpen: jest.fn(),
        showNotification: jest.fn(),
        hideNotification: jest.fn(),
        showModal: jest.fn(),
        hideModal: jest.fn(),
        accessibility: mockAccessibility,
        setAccessibility: jest.fn(),
        isOffline: false
      }}
    >
      <MockChatContext.Provider
        value={{
          state: {
            messages: [],
            isLoading: false
          },
          sendMessage: jest.fn().mockResolvedValue(undefined),
          clearMessages: jest.fn(),
          getMessageHistory: jest.fn().mockReturnValue([])
        }}
      >
        {children}
      </MockChatContext.Provider>
    </MockAppContext.Provider>
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

// Export mock hooks for tests to use directly
export const useApp = () => ({
  theme: 'dark',
  setTheme: jest.fn(),
  isMenuOpen: false,
  setMenuOpen: jest.fn(),
  showNotification: jest.fn(),
  hideNotification: jest.fn(),
  showModal: jest.fn(),
  hideModal: jest.fn(),
  accessibility: mockAccessibility,
  setAccessibility: jest.fn(),
  isOffline: false
});

export const useChat = () => ({
  state: {
    messages: [],
    isLoading: false
  },
  sendMessage: jest.fn().mockResolvedValue(undefined),
  clearMessages: jest.fn(),
  getMessageHistory: jest.fn().mockReturnValue([])
});

// Override render method
export { customRender as render };
export { mockAccessibility }; 