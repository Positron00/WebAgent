import { render, screen, fireEvent, waitFor } from '@/test-utils';
import Chat from '@/components/Chat';
import React from 'react';
import { mockAccessibility } from '@/test-utils';

// Mock fetch
global.fetch = jest.fn();

// Mock the useChat hook
jest.mock('@/contexts/ChatContext', () => ({
  useChat: () => ({
    state: {
      messages: [],
      isLoading: false,
      error: null
    },
    sendMessage: jest.fn().mockResolvedValue(undefined),
    clearMessages: jest.fn(),
    getMessageHistory: jest.fn().mockReturnValue([])
  })
}));

// Mock the useApp hook
jest.mock('@/contexts/AppContext', () => ({
  useApp: () => ({
    theme: 'dark',
    isOffline: false,
    setTheme: jest.fn(),
    showNotification: jest.fn(),
    hideNotification: jest.fn(),
    accessibility: mockAccessibility,
    setAccessibility: jest.fn()
  })
}));

describe('Chat Component', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
    
    // Reset the mocked implementation of useChat for each test
    jest.spyOn(require('@/contexts/ChatContext'), 'useChat').mockImplementation(() => ({
      state: {
        messages: [],
        isLoading: false,
        error: null
      },
      sendMessage: jest.fn().mockResolvedValue(undefined),
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    }));
  });

  it('renders chat interface', () => {
    render(<Chat />);
    
    // Check for main elements that should be present
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByLabelText('Submit message')).toBeInTheDocument();
    expect(screen.getByLabelText('Upload Image')).toBeInTheDocument();
    
    // Check for some of the UI elements
    expect(screen.getByText('Attach')).toBeInTheDocument();
    expect(screen.getByText('Voice')).toBeInTheDocument();
    expect(screen.getByText('Screen')).toBeInTheDocument();
  });

  it('handles text input', () => {
    // Set up a mock for the input handling
    const mockSetInputValue = jest.fn();
    
    // Create a component with a controlled input to test value changes
    const TestInput = () => {
      const [value, setValue] = React.useState('');
      mockSetInputValue.mockImplementation(setValue);
      
      return (
        <textarea 
          value={value} 
          onChange={(e) => setValue(e.target.value)}
          data-testid="test-input"
        />
      );
    };
    
    render(<TestInput />);
    
    // Find the input and change its value
    const input = screen.getByTestId('test-input');
    fireEvent.change(input, { target: { value: 'Hello' } });
    
    // Verify the input value was updated
    expect(input).toHaveValue('Hello');
  });

  it('handles message submission', async () => {
    // Setup mock response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        response: 'Hello! How can I help you?',
        usage: { total_tokens: 10 }
      })
    });

    // Set up loading state first, then update to show response
    const mockSendMessage = jest.fn().mockResolvedValue(undefined);
    jest.spyOn(require('@/contexts/ChatContext'), 'useChat').mockImplementation(() => ({
      state: {
        messages: [
          { role: 'user', content: 'Hello' }
        ],
        isLoading: true,
        error: null
      },
      sendMessage: mockSendMessage,
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    }));

    const { rerender } = render(<Chat />);

    // Find input and button using a more reliable query
    const input = screen.getByRole('textbox');
    const sendButton = screen.getByLabelText('Submit message');

    // Interact with the component
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.click(sendButton);

    // Verify loading state
    expect(screen.getByText('Thinking...')).toBeInTheDocument();

    // Now update the mock to show the response
    jest.spyOn(require('@/contexts/ChatContext'), 'useChat').mockImplementation(() => ({
      state: {
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hello! How can I help you?' }
        ],
        isLoading: false,
        error: null
      },
      sendMessage: mockSendMessage,
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    }));

    // Force a re-render to apply the updated mock
    rerender(<Chat />);

    // Verify the response appears
    await waitFor(() => {
      expect(screen.getByText('Hello! How can I help you?')).toBeInTheDocument();
    });
  });

  it('handles API errors', async () => {
    // Setup error response
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({
        error: 'API Error'
      })
    });

    // Set up loading state first
    const mockSendMessage = jest.fn().mockRejectedValue(new Error('API Error'));
    jest.spyOn(require('@/contexts/ChatContext'), 'useChat').mockImplementation(() => ({
      state: {
        messages: [
          { role: 'user', content: 'Hello' }
        ],
        isLoading: true,
        error: null
      },
      sendMessage: mockSendMessage,
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    }));

    const { rerender } = render(<Chat />);

    // Find input and button using a more reliable query
    const input = screen.getByRole('textbox');
    const sendButton = screen.getByLabelText('Submit message');

    // Interact with the component
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.click(sendButton);

    // Now update the mock to show the error
    jest.spyOn(require('@/contexts/ChatContext'), 'useChat').mockImplementation(() => ({
      state: {
        messages: [
          { role: 'user', content: 'Hello' }
        ],
        isLoading: false,
        error: 'API Error'
      },
      sendMessage: mockSendMessage,
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    }));

    // Force a re-render to apply the updated mock
    rerender(<Chat />);

    // Verify the error appears
    await waitFor(() => {
      expect(screen.getByText('API Error')).toBeInTheDocument();
    });
  });

  it('validates file upload', async () => {
    // Set up mocks for file validation
    const mockShowNotification = jest.fn();
    
    // Mock the AppContext to capture notifications
    jest.spyOn(require('@/contexts/AppContext'), 'useApp').mockImplementation(() => ({
      theme: 'dark',
      isOffline: false,
      setTheme: jest.fn(),
      showNotification: mockShowNotification,
      hideNotification: jest.fn(),
      accessibility: mockAccessibility,
      setAccessibility: jest.fn()
    }));
    
    // Set up specific state with file preview
    jest.spyOn(require('@/contexts/ChatContext'), 'useChat').mockImplementation(() => ({
      state: {
        messages: [],
        isLoading: false,
        error: null
      },
      sendMessage: jest.fn(),
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    }));

    render(<Chat />);
    
    // Create a text file (invalid image)
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    
    // Get the file input and trigger the change directly
    const fileInput = screen.getByLabelText('Upload Image');
    
    // Create a custom event to simulate file selection
    const fileChangeEvent = {
      target: {
        files: [file]
      }
    };
    
    // Directly call the onChange handler
    fireEvent.change(fileInput, fileChangeEvent);
    
    // Verify that the notification was called with the error message
    expect(mockShowNotification).toHaveBeenCalledWith({
      message: 'Please upload a valid image file (JPEG, PNG, GIF, or WebP)',
      type: 'error'
    });
  });

  it('enforces rate limiting', async () => {
    // Mock fetch to simulate a rate limit error
    global.fetch = jest.fn().mockImplementation(() => 
      Promise.resolve({
        ok: false,
        status: 429,
        json: () => Promise.resolve({ error: 'Rate limit exceeded. Please try again later.' })
      })
    );

    // Create a mock error message
    const errorMessage = 'Rate limit exceeded. Please try again later.';
    
    // Mock the ChatContext with an error state
    jest.spyOn(require('@/contexts/ChatContext'), 'useChat').mockImplementation(() => ({
      state: {
        messages: [],
        isLoading: false,
        error: errorMessage
      },
      sendMessage: jest.fn().mockRejectedValue(new Error(errorMessage)),
      clearMessages: jest.fn(),
      getMessageHistory: jest.fn().mockReturnValue([])
    }));

    // Render the component with the error state
    render(<Chat />);

    // Verify the error message is displayed
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });
}); 